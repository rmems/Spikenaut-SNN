"""Q8.8 export verification core: codec, parsing, section checks, report.

Shared by the ``verify_q88`` CLI and by ``q88_selftest``. Everything that
decides whether the shipped artifacts verify lives here; the CLI is only
argument handling and exit codes, and the self-test only exercises this
module.

What it provides
----------------
1. The Q8.8 codec (``encode_q88`` / ``decode_q88`` / ``encode_q88_hex``) and
   the deliberately broken ``encode_q88_clamp_to_zero`` the self-test uses to
   prove the checker can fail.
2. ``.mem`` and ``snn_model.json`` parsing that surfaces malformed input as
   ``ParseError`` rather than a traceback.
3. ``check_against_floats`` / ``check_signed_section`` -- the two ways a
   section is verified, with and without a JSON float column to compare
   against.
4. ``verify_shipped`` and ``report``: run all four sections and render the
   verdict. An empty or short result set is a hard failure, never a pass.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Paths resolve relative to the repo root (this file's parent's parent), never cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset" / "merged_v2"

MODEL_JSON = DATA_DIR / "snn_model.json"
MEM_THRESHOLDS = DATA_DIR / "parameters.mem"
MEM_DECAY = DATA_DIR / "parameters_decay.mem"
MEM_HIDDEN = DATA_DIR / "parameters_weights.mem"
MEM_OUTPUT = DATA_DIR / "parameters_output_weights.mem"

N_NEURONS = 16
N_INPUTS = 16
N_OUTPUT_WEIGHTS = 48

# verify_shipped() checks exactly these four sections. report() is given this
# number so a section silently dropped upstream fails the run loudly instead
# of quietly shrinking the verdict's scope.
EXPECTED_SECTIONS = 4

# The artifacts a complete run must each cover exactly once. Counting sections
# is not enough: four individually clean results can still be the wrong four.
# Duplicating one verify_shipped() entry while dropping another keeps the count
# at 4 and passes every per-section check, so the verdict would claim the full
# artifact set while never having opened one of the files. Identity, not
# arity, is what "complete" means here.
EXPECTED_ARTIFACTS = (
    "parameters.mem",
    "parameters_decay.mem",
    "parameters_weights.mem",
    "parameters_output_weights.mem",
)

# Pin of the shipped output-weight image. snn_model.json has no output layer,
# so this is the independent reference -- computed from the file as shipped,
# not invented JSON floats. Update both pins together if the artifact is
# intentionally replaced.
#
# What is hashed (NOT the raw file bytes): the canonical token sequence, i.e.
# every parsed 4-digit hex word, whitespace-stripped and upper-cased, joined
# by a single "\n", with no trailing newline, ASCII-encoded. See
# canonical_mem_digest(). Hashing raw bytes made the pin checkout-dependent --
# a Windows checkout with core.autocrlf materializes the same 48 words with
# CRLF endings, and a raw-byte hash then fails on a semantically identical
# artifact. (.gitattributes also pins *.mem to LF as defense in depth.)
#
# Reproduce from the repo root:
#
#   python3 -c "import hashlib; from pathlib import Path; \
#     w=[l.strip().upper() for l in Path('dataset/merged_v2/parameters_output_weights.mem').read_text(encoding='utf-8').splitlines() if l.strip()]; \
#     print(hashlib.sha256(chr(10).join(w).encode('ascii')).hexdigest())"
OUTPUT_WEIGHTS_CANON_SHA256 = (
    "03e83737fb42cce90e2a9a23bf7404caf0d650cab5ea9864aebc634b7a98edcb"
)
OUTPUT_WEIGHTS_HEX: tuple[str, ...] = (
    "FFF9", "001E", "FFFC", "0042", "000E", "0025", "FFE3", "0025",
    "FFD9", "0007", "FFD6", "000C", "001A", "0018", "0009", "FFF0",
    "FFE6", "FFF1", "FFF8", "0005", "FFEC", "0003", "0006", "FFF4",
    "FFEE", "0017", "0000", "FFFD", "FFF2", "FFF8", "0024", "0004",
    "0015", "FFF4", "FFFF", "001C", "001E", "FFE2", "FFFF", "FFEF",
    "FFF5", "FFED", "FFF2", "0012", "FFEF", "0026", "0016", "FFF3",
)

# ---------------------------------------------------------------------------
# Q8.8 codec
# ---------------------------------------------------------------------------
# One signed 16-bit word per value: 8 integer bits, 8 fractional bits.
# Stored as a 4-digit hex word; negatives are two's complement.
#   0100 ->  256 ->  1.0
#   00DA ->  218 ->  0.8515625
#   FFF9 -> -7   -> -0.02734375
Q88_SCALE = 256
Q88_WORD = 1 << 16
Q88_MIN_INT = -(1 << 15)  # -32768 -> -128.0
Q88_MAX_INT = (1 << 15) - 1  # 32767 -> ~127.996
Q88_MIN = Q88_MIN_INT / Q88_SCALE
Q88_MAX = Q88_MAX_INT / Q88_SCALE

# The only padding $readmemh treats as whitespace. Deliberately not
# str.strip()'s default, which also removes NBSP and other Unicode blanks.
ASCII_BLANK = " \t\x0b\x0c"
HEX_LINE_RE = re.compile(r"^[0-9A-Fa-f]{4}$")


class Q88RangeError(ValueError):
    """A float cannot be represented in Q8.8 without saturating."""


def encode_q88(value: float) -> int:
    """Encode a float as a raw unsigned 16-bit Q8.8 word.

    Rounds half away from zero so the mapping is symmetric about zero -- an
    asymmetric rounder would bias negative weights, which is exactly the class
    of bug issue #4 asks about. (In practice the JSON floats sit within 1e-4 LSB
    of an exact Q8.8 grid point, so no value is anywhere near a rounding tie;
    the verifier reports the worst residual so that stays visible.)
    """
    scaled = value * Q88_SCALE
    # A huge-but-finite float (1e308) scales to inf; math.floor(inf) raises
    # OverflowError. Check the scaled magnitude before any rounding or int().
    if not math.isfinite(scaled):
        raise Q88RangeError(
            f"{value!r} overflows Q8.8 scale [{Q88_MIN}, {Q88_MAX}]"
        )
    # floor/ceil already return int; away-from-zero on both sides of zero.
    raw = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if raw < Q88_MIN_INT or raw > Q88_MAX_INT:
        raise Q88RangeError(
            f"{value!r} -> {raw} is outside Q8.8 range [{Q88_MIN}, {Q88_MAX}]"
        )
    return raw & 0xFFFF


def encode_q88_hex(value: float) -> str:
    """Encode a float as the 4-digit uppercase hex word written to a .mem file."""
    return f"{encode_q88(value):04X}"


def decode_q88(word: int) -> float:
    """Decode a raw unsigned 16-bit Q8.8 word to a float (two's complement)."""
    if word < 0 or word >= Q88_WORD:
        raise ValueError(f"not a 16-bit word: {word}")
    signed = word - Q88_WORD if word >= 0x8000 else word
    return signed / Q88_SCALE


def encode_q88_clamp_to_zero(value: float) -> int:
    """DELIBERATELY BROKEN encoder used only by --self-test.

    Models the issue #15 hazard: an exporter that clamps the Q8.8 word to an
    unsigned range, so every inhibitory (negative) weight collapses to 0. The
    resulting file is still 48 well-formed hex lines, which is precisely why a
    format-only check would wave it through.
    """
    raw = encode_q88(value)
    signed = raw - Q88_WORD if raw >= 0x8000 else raw
    return max(signed, 0) & 0xFFFF


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemEntry:
    """One value read out of a .mem file."""

    index: int  # 0-based position in the memory image
    lineno: int  # 1-based line number in the file
    text: str  # the hex token as written
    word: int  # raw unsigned 16-bit value

    @property
    def value(self) -> float:
        """The Q8.8 word decoded to the float it represents."""
        return decode_q88(self.word)


class ParseError(Exception):
    """A shipped artifact could not be read as the format this tool expects.

    Raised for a missing file, a malformed .mem line, invalid UTF-8 in an
    artifact, a numeric field that overflows Q8.8 scale, or model JSON
    whose shape does not match. ``main()`` maps it to exit code 2 --
    "could not check", which is deliberately distinct from exit code 1,
    "checked and the values disagree".
    """


def canonical_mem_digest(entries: list[MemEntry]) -> str:
    """sha256 of the canonical token sequence of a parsed .mem image.

    The digest covers exactly the hex words the tool actually read: each token
    whitespace-stripped and upper-cased, joined by a single ``"\n"``, with no
    trailing newline, ASCII-encoded. Blank lines, trailing whitespace, hex
    letter case and CRLF-vs-LF line endings therefore do not move the digest,
    while any change to a word, its value, or its position does.

    This is deliberately not a hash of the file's raw bytes: those depend on
    how the repository was checked out (Git's ``core.autocrlf`` on Windows
    rewrites LF to CRLF), which would make the pin fail on a semantically
    unchanged artifact. See OUTPUT_WEIGHTS_CANON_SHA256 for exactly what is
    hashed and how to reproduce the pinned value.
    """
    canonical = "\n".join(entry.text.upper() for entry in entries)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def as_finite_float(value: object, where: str) -> float:
    """Convert a JSON scalar to a finite float, or raise ParseError.

    ``load_model()`` used to validate container *shape* only, so a model with
    the right shape but a nonnumeric scalar (``"threshold": null``, an
    object-valued weight) passed every check and then raised an uncaught
    ``TypeError`` at the ``float()`` call -- a traceback instead of the
    documented exit code 2. Every numeric field now routes through here.

    Accepts only real numbers. Rejects ``None``, strings, dicts/lists, NaN and
    the infinities -- and ``bool``, which must be tested first because ``bool``
    subclasses ``int`` in Python, so ``isinstance(True, int)`` is True and an
    unguarded check would silently encode ``True`` as the weight 1.0.

    Also rejects a huge-but-finite number (``1e308``) whose Q8.8 scaling
    overflows: ``isfinite`` is true, but ``value * Q88_SCALE`` is inf and
    ``math.floor`` then raises an uncaught ``OverflowError``. The scaled
    magnitude is checked here, before any rounding or int conversion, so
    the failure is a field-specific ``ParseError`` (exit 2).

    Raises:
        ParseError: ``value`` is not a finite real number, or overflows
            the Q8.8 scale.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(
            f"{where}: expected a finite number, got "
            f"{type(value).__name__} {value!r}"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:  # e.g. an int too large for float
        raise ParseError(
            f"{where}: {value!r} is not representable as a float ({exc})"
        ) from exc
    if not math.isfinite(number):
        raise ParseError(f"{where}: expected a finite number, got {value!r}")
    # Check scaled range before encode_q88 rounds: 1e308 is finite, 1e308*256
    # is not, and math.floor(inf) is OverflowError.
    scaled = number * Q88_SCALE
    if not math.isfinite(scaled):
        raise ParseError(
            f"{where}: {number!r} overflows Q8.8 scale [{Q88_MIN}, {Q88_MAX}]"
        )
    return number


def read_utf8_text(path: Path) -> str:
    """Read an artifact as UTF-8 text, or raise ``ParseError``.

    A corrupted ``.mem`` or JSON file with invalid UTF-8 raises
    ``UnicodeDecodeError`` from the file-read loop, which ``main()`` does
    not catch. That is a parse failure (exit 2), same contract as a
    missing or malformed artifact -- not a traceback.

    ``OSError`` is caught for the same reason. Callers check ``is_file()``
    first, but that only says the path resolves to a file *now* and says
    nothing about whether it can be opened: a read-protected artifact, a
    directory reached directly, a vanished file, or an I/O error all raise
    here after the guard passed. ``OSError`` is the base class for exactly
    that set, and this function does nothing but open one file, so catching
    it is narrow rather than a blanket handler.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path.name}: not valid UTF-8 ({exc})") from exc
    except OSError as exc:
        raise ParseError(f"{path}: cannot be read ({exc})") from exc


def parse_mem(path: Path) -> list[MemEntry]:
    """Parse a $readmemh-style .mem file of 4-digit hex words, one per line."""
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    entries: list[MemEntry] = []
    text = read_utf8_text(path)
    # Split and pad on ASCII only. str.splitlines() also breaks on U+2028,
    # U+2029, U+0085 and friends, and str.strip() removes non-ASCII whitespace
    # such as NBSP -- so an image using those would parse into the right 48
    # words and match the pin, while $readmemh does not treat those UTF-8 byte
    # sequences as Verilog whitespace. That artifact is malformed; the verifier
    # must not accept it. CRLF and bare CR are still normalised, because a
    # Windows checkout is a legitimate encoding of the same image.
    for lineno, raw_line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), start=1
    ):
        token = raw_line.strip(ASCII_BLANK)
        if not token:
            continue
        if not HEX_LINE_RE.match(token):
            raise ParseError(
                f"{path.name}:{lineno}: expected a 4-digit hex word, got {token!r}"
            )
        entries.append(
            MemEntry(
                index=len(entries),
                lineno=lineno,
                text=token,
                word=int(token, 16),
            )
        )
    return entries


def _decode_model_json(path: Path) -> object:
    """Read and decode the model file, keeping every failure in contract."""
    try:
        return json.loads(read_utf8_text(path))
    except json.JSONDecodeError:
        raise
    except ValueError as exc:
        # CPython raises a bare ValueError for an integer literal longer than
        # sys.get_int_max_str_digits() (4300 by default on 3.11+). It is not a
        # JSONDecodeError, so it would escape main() as a traceback.
        raise ParseError(f"{path.name}: unreadable JSON number: {exc}") from exc


def _model_neurons(path: Path, model: object) -> list:
    """Return the neuron list, rejecting any container that is not one.

    Validated before any field is read, so a top-level list/scalar cannot leak
    an AttributeError or TypeError past the ParseError contract.
    """
    if not isinstance(model, dict):
        raise ParseError(
            f"{path.name}: expected a top-level JSON object, got "
            f"{type(model).__name__}"
        )
    neurons = model.get("neurons")
    if not isinstance(neurons, list):
        raise ParseError(f"{path.name}: expected a top-level 'neurons' list")
    if len(neurons) != N_NEURONS:
        raise ParseError(
            f"{path.name}: expected {N_NEURONS} neurons, found {len(neurons)}"
        )
    return neurons


def _validate_neuron(path: Path, index: int, neuron: object) -> None:
    """Assert one neuron has the expected shape and finite numeric scalars."""
    if not isinstance(neuron, dict):
        raise ParseError(
            f"{path.name}: neuron {index} is not a JSON object, got "
            f"{type(neuron).__name__}"
        )
    for key in ("threshold", "decay_rate", "weights"):
        if key not in neuron:
            raise ParseError(f"{path.name}: neuron {index} is missing {key!r}")
    weights = neuron["weights"]
    if not isinstance(weights, list):
        raise ParseError(
            f"{path.name}: neuron {index} 'weights' is not a list, got "
            f"{type(weights).__name__}"
        )
    if len(weights) != N_INPUTS:
        raise ParseError(
            f"{path.name}: neuron {index} has {len(weights)} weights, "
            f"expected {N_INPUTS}"
        )
    # Scalar types, not just container shape: the float() conversions
    # downstream must never see a None, a bool, a string, a container, or
    # a non-finite float.
    as_finite_float(neuron["threshold"], f"{path.name}: neuron {index} 'threshold'")
    as_finite_float(neuron["decay_rate"], f"{path.name}: neuron {index} 'decay_rate'")
    for j, weight in enumerate(weights):
        as_finite_float(weight, f"{path.name}: neuron {index} weight {j}")


def load_model(path: Path) -> dict:
    """Load snn_model.json and assert it has the shape the checker assumes.

    Most rejection routes -- missing file, malformed JSON container, wrong
    neuron count, missing or wrongly-sized fields, a nonnumeric scalar, a
    value whose Q8.8 scaling overflows -- raise ``ParseError``, which
    ``main()`` maps to exit code 2. Two do not, and the difference is worth
    keeping straight:

      * A JSON *syntax* error comes out of ``json.loads`` as
        ``json.JSONDecodeError``. Different type, still exit 2, because the
        file could not be read as a model at all.
      * A value that is a finite float but outside Q8.8 -- ``200.0`` for a
        threshold -- passes every load-time check and only fails later, when
        ``encode_q88`` tries to encode it. The section loops catch that
        ``Q88RangeError`` themselves, record ``UNREPRESENTABLE at index N``,
        and fail the section, so it exits **1**, not 2. That is the right
        classification: the artifacts were read and checked, and a value
        disagrees with the format. ``main()`` also lists ``Q88RangeError``,
        as a backstop for any future path outside those loops.

    So exit 2 means "could not check" and exit 1 means "checked, something
    is wrong" -- and that split, not the exception type, is the contract.

    Two layers, both load-bearing:

      * *shape*: validated before any field is read, so a top-level
        list/scalar or a ``null`` neuron cannot leak an
        ``AttributeError``/``TypeError`` past the contract.
      * *scalar type*: every threshold, decay rate and weight must be a finite
        real number (see ``as_finite_float``). Shape alone let
        ``{"threshold": null}`` or an object-valued weight through, to blow up
        later in ``float()`` with an uncaught ``TypeError``.

    Raises:
        ParseError: the file is missing, or its JSON does not match the
            expected ``{"neurons": [{threshold, decay_rate, weights[16]}, ...]}``
            shape, or a numeric field is not a finite real number.
        json.JSONDecodeError: the file is not valid JSON at all.

    """
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    model = _decode_model_json(path)
    neurons = _model_neurons(path, model)
    for i, neuron in enumerate(neurons):
        _validate_neuron(path, i, neuron)
    return model


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


@dataclass
class Mismatch:
    """One exported value that disagrees with its reference source.

    Carries both sides plus enough coordinates (index, human label, line
    number) to point at the offending line of the .mem file.
    """

    index: int
    label: str  # human-readable coordinate, e.g. "neuron 3 <- input 7"
    lineno: int
    source_float: float
    expected_hex: str
    actual_hex: str

    @property
    def expected_float(self) -> float:
        """The float the reference source says this slot should hold."""
        return decode_q88(int(self.expected_hex, 16))

    @property
    def actual_float(self) -> float:
        """The float the shipped .mem file actually holds in this slot."""
        return decode_q88(int(self.actual_hex, 16))

    def render(self) -> str:
        """Format this mismatch as one aligned report line.

        Names the coordinate, both hex words with their floats, and the
        signed delta -- enough to find the bad value in the file by hand.
        """
        return (
            f"  [{self.index:3d}] {self.label:<26} line {self.lineno:<4} "
            f"json={self.source_float:+.7f}  "
            f"expected {self.expected_hex} ({self.expected_float:+.7f})  "
            f"actual {self.actual_hex} ({self.actual_float:+.7f})  "
            f"delta={self.actual_float - self.expected_float:+.7f}"
        )


@dataclass(frozen=True)
class SectionEvidence:
    """How much independent evidence actually backed a section's verdict.

    Both fields exist so the summary can count what ran rather than assert it:
    ``max_residual_lsb`` is None for a section with no JSON float column to
    compare against, and ``pinned_count`` is 0 when no shipped-file pin was
    supplied. ``report()`` reads them, so no summary line can claim a check
    that did not happen.
    """

    max_residual_lsb: float | None = None
    pinned_count: int = 0


@dataclass
class SectionResult:
    """The verdict for one memory section: what was checked and what broke.

    ``failed`` records section-level invariant breaks (wrong length, pin
    failure, lost sign) that are not tied to any single value, which is why
    it is tracked separately from ``mismatches``.

    ``evidence`` records what independently corroborated the verdict -- see
    ``SectionEvidence``.
    """

    name: str
    source: str
    count: int
    mismatches: list[Mismatch]
    notes: list[str]
    failed: bool = False
    evidence: SectionEvidence = field(default_factory=SectionEvidence)

    @property
    def ok(self) -> bool:
        """Whether this section verified cleanly.

        True only if no value mismatched *and* no section-level invariant
        broke. A section with zero mismatches can still fail (wrong length,
        pin failure, lost sign), so both halves are load-bearing.
        """
        return not self.failed and not self.mismatches

    def render(self) -> str:
        """Format this section's PASS/FAIL block for the report.

        Headline, worst residual, notes, and every mismatch in full -- it
        never truncates, because a partial diff hides regressions.
        """
        status = "PASS" if self.ok else "FAIL"
        head = f"[{status}] {self.name}: {self.count} values  ({self.source})"
        lines = [head]
        residual = self.evidence.max_residual_lsb
        if residual is not None:
            lines.append(
                f"       max |json - mem| = {residual:.6f} LSB "
                f"({residual / Q88_SCALE:.3e} absolute)"
            )
        for note in self.notes:
            lines.append(f"       {note}")
        if self.mismatches:
            lines.append(f"       {len(self.mismatches)} mismatch(es):")
            lines.extend(m.render() for m in self.mismatches)
        return "\n".join(lines)


def check_against_floats(
    name: str,
    source: str,
    floats: list[float],
    labels: list[str],
    entries: list[MemEntry],
) -> SectionResult:
    """Re-derive the Q8.8 encoding of ``floats`` and compare to ``entries``."""
    notes: list[str] = []
    failed = False
    if len(entries) != len(floats):
        notes.append(
            f"LENGTH MISMATCH: file holds {len(entries)} values, "
            f"source has {len(floats)}"
        )
        failed = True

    mismatches: list[Mismatch] = []
    max_residual = 0.0
    for i, source_value in enumerate(floats):
        if i >= len(entries):
            break
        entry = entries[i]
        try:
            expected_hex = encode_q88_hex(source_value)
        except Q88RangeError as exc:
            notes.append(f"UNREPRESENTABLE at index {i}: {exc}")
            failed = True
            continue
        if expected_hex != entry.text.upper():
            mismatches.append(
                Mismatch(
                    index=i,
                    label=labels[i],
                    lineno=entry.lineno,
                    source_float=source_value,
                    expected_hex=expected_hex,
                    actual_hex=entry.text.upper(),
                )
            )
        max_residual = max(max_residual, abs(source_value - entry.value) * Q88_SCALE)

    return SectionResult(
        name=name,
        source=source,
        count=len(entries),
        mismatches=mismatches,
        notes=notes,
        evidence=SectionEvidence(
            max_residual_lsb=max_residual if floats else None
        ),
        failed=failed,
    )


@dataclass(frozen=True)
class ShippedPin:
    """The independent reference for a section with no JSON float column.

    ``snn_model.json`` carries no output layer, so the shipped file is its own
    reference: the sha256 of its canonical token sequence plus the exact hex
    words. Both halves are optional -- a section checked without a pin reports
    ``pinned_count`` 0 rather than claiming a check that was never run.
    """

    hex_words: tuple[str, ...] | None = None
    canon_sha256: str | None = None

    @property
    def present(self) -> bool:
        """Whether either half of the pin was supplied."""
        return self.hex_words is not None or self.canon_sha256 is not None


# Each _check_* below returns (notes, mismatches, failed) so check_signed_section
# can run them in a fixed order and concatenate. That order is load-bearing: it
# is the order the notes appear in the rendered report.
CheckOutcome = tuple[list[str], list[Mismatch], bool]


def _check_canon_sha(entries: list[MemEntry], expected: str | None) -> CheckOutcome:
    """Compare the canonical-token sha256 against the shipped-file pin.

    Hashes parsed tokens rather than raw bytes, so a CRLF checkout of a
    semantically identical artifact still matches.
    """
    if expected is None:
        return [], [], False
    actual = canonical_mem_digest(entries)
    if actual != expected:
        return (
            [
                f"SHIPPED-FILE PIN FAILURE: canonical-token sha256 {actual} "
                f"!= {expected}"
            ],
            [],
            True,
        )
    return [f"canonical-token sha256 pin {actual} matches shipped file"], [], False


def _check_gold_hex(
    entries: list[MemEntry],
    expected_count: int,
    expected_hex: tuple[str, ...] | None,
) -> CheckOutcome:
    """Compare every word against the exact gold hex pin.

    This is what catches well-formed corruption: an FFF9->FFF8 swap decodes and
    re-encodes cleanly, so only a value-by-value pin sees it.
    """
    if expected_hex is None:
        return [], [], False
    notes: list[str] = []
    mismatches: list[Mismatch] = []
    failed = False

    if len(expected_hex) != expected_count:
        notes.append(
            f"GOLD PIN LENGTH MISMATCH: gold has {len(expected_hex)} "
            f"words, expected {expected_count}"
        )
        failed = True

    for i in range(min(len(entries), len(expected_hex))):
        actual = entries[i].text.upper()
        want = expected_hex[i].upper()
        if actual != want:
            mismatches.append(
                Mismatch(
                    index=i,
                    label=f"output weight {i}",
                    lineno=entries[i].lineno,
                    source_float=decode_q88(int(want, 16)),
                    expected_hex=want,
                    actual_hex=actual,
                )
            )

    if len(entries) != len(expected_hex):
        notes.append(
            f"PIN LENGTH MISMATCH: file holds {len(entries)} values, "
            f"gold pin has {len(expected_hex)}"
        )
        failed = True
    return notes, mismatches, failed


def _check_canonical_round_trip(entries: list[MemEntry]) -> CheckOutcome:
    """decode -> re-encode must return the identical word for every entry.

    Proves the float<->hex codec agrees with every word actually in the file.
    """
    notes: list[str] = []
    mismatches: list[Mismatch] = []
    failed = False
    for entry in entries:
        try:
            round_tripped = encode_q88_hex(entry.value)
        except Q88RangeError as exc:
            notes.append(f"UNREPRESENTABLE at index {entry.index}: {exc}")
            failed = True
            continue
        if round_tripped != entry.text.upper():
            mismatches.append(
                Mismatch(
                    index=entry.index,
                    label=f"output weight {entry.index}",
                    lineno=entry.lineno,
                    source_float=entry.value,
                    expected_hex=round_tripped,
                    actual_hex=entry.text.upper(),
                )
            )
    return notes, mismatches, failed


def _sign_census(entries: list[MemEntry]) -> tuple[list[MemEntry], list[MemEntry]]:
    """Split entries into two's-complement negatives and strict positives.

    Zero belongs to neither, which is what makes the counts in the range note
    add up to the section length only when every word is accounted for.
    """
    return (
        [e for e in entries if e.word >= 0x8000],
        [e for e in entries if 0 < e.word < 0x8000],
    )


def _range_note(entries: list[MemEntry], negatives: list, positives: list) -> list[str]:
    """One line describing the section's span and its sign census."""
    if not entries:
        return []
    values = [e.value for e in entries]
    return [
        f"range {min(values):+.7f} .. {max(values):+.7f}  "
        f"({len(negatives)} inhibitory, {len(positives)} excitatory, "
        f"{len(entries) - len(negatives) - len(positives)} zero)"
    ]


def _check_sign_integrity(entries: list[MemEntry]) -> CheckOutcome:
    """Report the value range, then assert the section is genuinely signed.

    This is the only signed section, so it must decode as two's complement and
    must still contain negative (inhibitory) weights. A clamp-to-zero encoder
    produces a file with none -- the issue #15 hazard.
    """
    negatives, positives = _sign_census(entries)
    notes = _range_note(entries, negatives, positives)

    if not negatives:
        notes.append(
            "SIGN INTEGRITY FAILURE: no negative values. This section is signed "
            "Q8.8 and must carry inhibitory weights; an all-non-negative file is "
            "the signature of a clamp-to-zero or unsigned encoder (issue #15)."
        )
        return notes, [], True

    # Every word >= 0x8000 must decode strictly negative -- proves the
    # two's-complement path is what is being exercised, not a sign-magnitude
    # or bare-unsigned read.
    bad_sign = [e for e in negatives if e.value >= 0]
    if bad_sign:
        notes.append(
            f"SIGN DECODE FAILURE on {len(bad_sign)} word(s) >= 0x8000 that "
            "did not decode negative"
        )
        return notes, [], True
    return notes, [], False


def check_signed_section(
    name: str,
    source: str,
    entries: list[MemEntry],
    expected_count: int,
    pin: ShippedPin | None = None,
) -> SectionResult:
    """Check the signed output-weight section.

    snn_model.json has no output layer, so there is no float column to
    cross-validate against. The shipped file is pinned by the sha256 of its
    canonical token sequence and by the exact 48 hex words. Sign-integrity
    and decode->re-encode stay as defense in depth: they catch clamp-to-zero
    and broken codecs, but a well-formed FFF9->FFF8 swap still round-trips.

      * exact value count,
      * shipped-file pin (when ``pin`` carries one); the sha covers the parsed
        tokens, not raw file bytes, so it does not change with the checkout's
        line endings,
      * canonical round-trip: decode -> re-encode is bit-identical, so the
        float<->hex codec agrees with every word actually in the file,
      * sign integrity: this is the only signed section, so it must decode as
        two's complement and must still contain negative (inhibitory) weights.
        A clamp-to-zero encoder produces a file with none.
    """
    pin = pin if pin is not None else ShippedPin()
    notes: list[str] = []
    mismatches: list[Mismatch] = []
    failed = False

    if len(entries) != expected_count:
        notes.append(
            f"LENGTH MISMATCH: file holds {len(entries)} values, "
            f"expected {expected_count}"
        )
        failed = True

    for step_notes, step_mismatches, step_failed in (
        _check_canon_sha(entries, pin.canon_sha256),
        _check_gold_hex(entries, expected_count, pin.hex_words),
        _check_canonical_round_trip(entries),
        _check_sign_integrity(entries),
    ):
        notes.extend(step_notes)
        mismatches.extend(step_mismatches)
        failed = failed or step_failed

    return SectionResult(
        name=name,
        source=source,
        count=len(entries),
        mismatches=mismatches,
        notes=notes,
        failed=failed,
        evidence=SectionEvidence(
            pinned_count=len(entries) if pin.present else 0
        ),
    )


# ---------------------------------------------------------------------------
# The full verification of the shipped artifacts
# ---------------------------------------------------------------------------


def _hidden_weight_column(neurons: list) -> tuple[list[float], list[str]]:
    """Flatten the 16x16 hidden matrix row-major, with a label per value."""
    hidden: list[float] = []
    labels: list[str] = []
    for i, neuron in enumerate(neurons):
        for j, weight in enumerate(neuron["weights"]):
            hidden.append(as_finite_float(weight, f"neurons[{i}].weights[{j}]"))
            labels.append(f"neuron {i:2d} <- input {j:2d}")
    return hidden, labels


def verify_shipped() -> list[SectionResult]:
    """Run all four section checks against the artifacts shipped in the repo.

    Thresholds, decay rates, and hidden weights are re-derived from
    snn_model.json and compared value by value; the output weights have no
    JSON source and are checked against the shipped-file pin plus the signed
    invariants. Returns one SectionResult per section -- it does not print,
    exit, or raise on a mismatch; that is ``report()``'s job.

    Raises:
        ParseError: an artifact is missing or malformed.

    """
    model = load_model(MODEL_JSON)
    neurons = model["neurons"]

    # load_model() has already rejected any nonnumeric field; going through
    # as_finite_float() here too keeps the conversion itself inside the
    # ParseError contract rather than relying on a check made elsewhere.
    thresholds = [
        as_finite_float(n["threshold"], f"neurons[{i}].threshold")
        for i, n in enumerate(neurons)
    ]
    decays = [
        as_finite_float(n["decay_rate"], f"neurons[{i}].decay_rate")
        for i, n in enumerate(neurons)
    ]
    hidden, hidden_labels = _hidden_weight_column(neurons)

    return [
        check_against_floats(
            "thresholds        (parameters.mem)",
            "re-derived from snn_model.json neurons[].threshold",
            thresholds,
            [f"neuron {i:2d} threshold" for i in range(len(thresholds))],
            parse_mem(MEM_THRESHOLDS),
        ),
        check_against_floats(
            "decay rates       (parameters_decay.mem)",
            "re-derived from snn_model.json neurons[].decay_rate",
            decays,
            [f"neuron {i:2d} decay" for i in range(len(decays))],
            parse_mem(MEM_DECAY),
        ),
        check_against_floats(
            "hidden weights    (parameters_weights.mem)",
            "re-derived from snn_model.json neurons[].weights, row-major 16x16",
            hidden,
            hidden_labels,
            parse_mem(MEM_HIDDEN),
        ),
        check_signed_section(
            "output weights    (parameters_output_weights.mem)",
            "shipped-file canonical sha256 + gold hex pin; "
            "snn_model.json has no output layer",
            parse_mem(MEM_OUTPUT),
            N_OUTPUT_WEIGHTS,
            pin=ShippedPin(
                hex_words=OUTPUT_WEIGHTS_HEX,
                canon_sha256=OUTPUT_WEIGHTS_CANON_SHA256,
            ),
        ),
    ]




# ---------------------------------------------------------------------------
# Self-test failure signal
# ---------------------------------------------------------------------------
# Raised by q88_selftest, caught by the verify_q88 CLI. It lives here, in the
# module both of them import, so there is exactly one class object: defined in
# the CLI entry script instead, running that script as __main__ would give the
# self-test a second, unrelated copy and the CLI's handler would not catch it.


class SelfTestFailure(AssertionError):
    """The verifier failed to catch a regression it must catch."""
