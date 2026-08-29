#!/usr/bin/env python3
"""Verify the Q8.8 export pipeline: snn_model.json floats <-> FPGA .mem files.

Regression test for issue #4 ("Validate Q8.8 export pipeline: GPU float to FPGA
roundtrip fidelity"). The 2026-07-12 investigation concluded by hand that the
export is faithful; this turns that prose finding into something executable so
it stays true.

What it does
------------
1. Parses the four shipped ``.mem`` files and ``snn_model.json``.
2. Independently re-derives the expected Q8.8 encoding from the JSON floats and
   compares it, value by value, against what the ``.mem`` files actually
   contain -- thresholds (16), decay rates (16), hidden weights (16x16=256).
3. Checks the 48 signed output weights. ``snn_model.json`` carries no output
   layer and this tool does not invent one. The shipped file is pinned by the
   sha256 of its *canonical token sequence* and the exact 48 hex words
   (computed from the file as shipped). Sign-integrity and decode→re-encode
   stay as defense in depth; they are not enough on their own (a well-formed
   FFF9→FFF8 swap still round-trips).
4. Exits non-zero with a per-value report on any mismatch. An empty or short
   result set is a hard failure, never a pass: reporting a clean verification
   having read nothing is worse than crashing, because it gets believed.

This is the float→Q8.8 encoding half of issue #4. Passing here does not
close #4: weight MSE / bit-identical export is not the close bar
(Hamming-on-holdout is).

``--self-test`` proves the checker can actually fail: clamp-to-zero output
weights (issue #15), 1-LSB hidden-weight drift, truncated files, well-formed
output-weight corruption against the pin, malformed model JSON (a list/null
top level, a null neuron, a non-numeric threshold/decay/weight, or a
huge-but-finite ``1e308`` must surface as ``ParseError``, not a
traceback / ``OverflowError``), an invalid-UTF-8 ``.mem`` (``ParseError``,
not ``UnicodeDecodeError``), an empty or short result set (which must
FAIL while ``worst_residual_lsb`` stays crash-safe), and a CRLF checkout of the
output-weight image (which must still verify). A checker that cannot fail is
worthless, and so is one that passes without reading anything.

Standard library only. Run from anywhere::

    python3 tools/verify_q88.py
    python3 tools/verify_q88.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
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

# Pin of the shipped output-weight image. snn_model.json has no output layer,
# so this is the independent reference — computed from the file as shipped,
# not invented JSON floats. Update both pins together if the artifact is
# intentionally replaced.
#
# What is hashed (NOT the raw file bytes): the canonical token sequence, i.e.
# every parsed 4-digit hex word, whitespace-stripped and upper-cased, joined
# by a single "\n", with no trailing newline, ASCII-encoded. See
# canonical_mem_digest(). Hashing raw bytes made the pin checkout-dependent —
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
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        token = raw_line.strip()
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
    model = json.loads(read_utf8_text(path))
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
    for i, neuron in enumerate(neurons):
        if not isinstance(neuron, dict):
            raise ParseError(
                f"{path.name}: neuron {i} is not a JSON object, got "
                f"{type(neuron).__name__}"
            )
        for key in ("threshold", "decay_rate", "weights"):
            if key not in neuron:
                raise ParseError(f"{path.name}: neuron {i} is missing {key!r}")
        if not isinstance(neuron["weights"], list):
            raise ParseError(
                f"{path.name}: neuron {i} 'weights' is not a list, got "
                f"{type(neuron['weights']).__name__}"
            )
        if len(neuron["weights"]) != N_INPUTS:
            raise ParseError(
                f"{path.name}: neuron {i} has {len(neuron['weights'])} weights, "
                f"expected {N_INPUTS}"
            )
        # Scalar types, not just container shape: the float() conversions
        # downstream must never see a None, a bool, a string, a container, or
        # a non-finite float.
        as_finite_float(neuron["threshold"], f"{path.name}: neuron {i} 'threshold'")
        as_finite_float(neuron["decay_rate"], f"{path.name}: neuron {i} 'decay_rate'")
        for j, weight in enumerate(neuron["weights"]):
            as_finite_float(weight, f"{path.name}: neuron {i} weight {j}")
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


@dataclass
class SectionResult:
    """The verdict for one memory section: what was checked and what broke.

    ``failed`` records section-level invariant breaks (wrong length, pin
    failure, lost sign) that are not tied to any single value, which is why
    it is tracked separately from ``mismatches``.

    ``pinned_count`` is how many words this section actually compared against
    a shipped-file pin (0 when no pin was supplied). ``report()`` counts it
    rather than asserting a pin happened, so the summary can never claim a
    check that was not run.
    """

    name: str
    source: str
    count: int
    mismatches: list[Mismatch]
    notes: list[str]
    max_residual_lsb: float | None = None
    failed: bool = False
    pinned_count: int = 0

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
        if self.max_residual_lsb is not None:
            lines.append(
                f"       max |json - mem| = {self.max_residual_lsb:.6f} LSB "
                f"({self.max_residual_lsb / Q88_SCALE:.3e} absolute)"
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
        max_residual_lsb=max_residual if floats else None,
        failed=failed,
    )


def check_signed_section(
    name: str,
    source: str,
    entries: list[MemEntry],
    expected_count: int,
    expected_hex: tuple[str, ...] | None = None,
    expected_canon_sha256: str | None = None,
) -> SectionResult:
    """Check the signed output-weight section.

    snn_model.json has no output layer, so there is no float column to
    cross-validate against. The shipped file is pinned by the sha256 of its
    canonical token sequence and by the exact 48 hex words. Sign-integrity
    and decode→re-encode stay as defense in depth: they catch clamp-to-zero
    and broken codecs, but a well-formed FFF9→FFF8 swap still round-trips.

      * shipped-file pin (when ``expected_hex`` / ``expected_canon_sha256``
        is given); the sha covers the parsed tokens, not raw file bytes, so
        it does not change with the checkout's line endings,
      * exact value count,
      * canonical round-trip: decode -> re-encode is bit-identical, so the
        float<->hex codec agrees with every word actually in the file,
      * sign integrity: this is the only signed section, so it must decode as
        two's complement and must still contain negative (inhibitory) weights.
        A clamp-to-zero encoder produces a file with none.
    """
    notes: list[str] = []
    failed = False
    mismatches: list[Mismatch] = []

    if len(entries) != expected_count:
        notes.append(
            f"LENGTH MISMATCH: file holds {len(entries)} values, "
            f"expected {expected_count}"
        )
        failed = True

    if expected_canon_sha256 is not None:
        actual_sha = canonical_mem_digest(entries)
        if actual_sha != expected_canon_sha256:
            notes.append(
                f"SHIPPED-FILE PIN FAILURE: canonical-token sha256 {actual_sha} "
                f"!= {expected_canon_sha256}"
            )
            failed = True
        else:
            notes.append(
                f"canonical-token sha256 pin {actual_sha} matches shipped file"
            )

    if expected_hex is not None:
        if len(expected_hex) != expected_count:
            notes.append(
                f"GOLD PIN LENGTH MISMATCH: gold has {len(expected_hex)} "
                f"words, expected {expected_count}"
            )
            failed = True
        n = min(len(entries), len(expected_hex))
        for i in range(n):
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

    # Canonical round-trip: hex -> float -> hex must be the identical word.
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

    negatives = [e for e in entries if e.word >= 0x8000]
    positives = [e for e in entries if 0 < e.word < 0x8000]
    values = [e.value for e in entries]

    if entries:
        notes.append(
            f"range {min(values):+.7f} .. {max(values):+.7f}  "
            f"({len(negatives)} inhibitory, {len(positives)} excitatory, "
            f"{len(entries) - len(negatives) - len(positives)} zero)"
        )

    if not negatives:
        notes.append(
            "SIGN INTEGRITY FAILURE: no negative values. This section is signed "
            "Q8.8 and must carry inhibitory weights; an all-non-negative file is "
            "the signature of a clamp-to-zero or unsigned encoder (issue #15)."
        )
        failed = True
    else:
        # Every word >= 0x8000 must decode strictly negative -- proves the
        # two's-complement path is what is being exercised, not a sign-magnitude
        # or bare-unsigned read.
        bad_sign = [e for e in negatives if e.value >= 0]
        if bad_sign:
            notes.append(
                f"SIGN DECODE FAILURE on {len(bad_sign)} word(s) >= 0x8000 that "
                "did not decode negative"
            )
            failed = True

    return SectionResult(
        name=name,
        source=source,
        count=len(entries),
        mismatches=mismatches,
        notes=notes,
        failed=failed,
        pinned_count=(
            len(entries)
            if expected_hex is not None or expected_canon_sha256 is not None
            else 0
        ),
    )


# ---------------------------------------------------------------------------
# The full verification of the shipped artifacts
# ---------------------------------------------------------------------------


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
    hidden: list[float] = []
    hidden_labels: list[str] = []
    for i, neuron in enumerate(neurons):
        for j, weight in enumerate(neuron["weights"]):
            hidden.append(as_finite_float(weight, f"neurons[{i}].weights[{j}]"))
            hidden_labels.append(f"neuron {i:2d} <- input {j:2d}")

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
            expected_hex=OUTPUT_WEIGHTS_HEX,
            expected_canon_sha256=OUTPUT_WEIGHTS_CANON_SHA256,
        ),
    ]


def worst_residual_lsb(results: list[SectionResult]) -> float:
    """Max residual across sections that have one; 0.0 if none do.

    An empty residual list must not raise — ``report()`` is called with
    signed-only results (``max_residual_lsb`` is None) and in tests with
    no sections at all. ``max()`` on an empty generator is a ValueError.
    """
    residuals = [r.max_residual_lsb for r in results if r.max_residual_lsb is not None]
    return max(residuals) if residuals else 0.0


def report(
    results: list[SectionResult],
    stream=sys.stdout,
    expected_sections: int | None = None,
) -> bool:
    """Print the full per-section report and the verdict; return True if clean.

    An empty ``results`` list is a HARD FAILURE, never a pass. ``all([])`` is
    True, so an unguarded verdict printed "OK: 0 Q8.8 values verified" -- plus
    the claim that 48 output weights were pinned -- having read no artifact at
    all. A verifier that reports a clean pass having checked nothing is worse
    than one that crashes, because it is believed. Pass ``expected_sections``
    to reject a short run too, where a section was dropped upstream.

    The summary reports only what was actually verified: the cross-validated
    count, the worst residual and the pinned-word count are all counted out of
    ``results``, so no line can assert a check that did not run.

    ``worst_residual_lsb`` stays empty-safe (a section may legitimately carry
    no residual); the empty-set verdict is decided here, not there.
    """
    print("Q8.8 export verification", file=stream)
    print(f"repo root: {REPO_ROOT}", file=stream)
    print(f"artifacts: {DATA_DIR.relative_to(REPO_ROOT)}/", file=stream)
    print("", file=stream)
    for result in results:
        print(result.render(), file=stream)
    print("", file=stream)

    total_values = sum(r.count for r in results)
    total_mismatches = sum(len(r.mismatches) for r in results)
    cross_checked = sum(r.count for r in results if r.max_residual_lsb is not None)
    pinned = sum(r.pinned_count for r in results)
    worst = worst_residual_lsb(results)

    # Structural failures: the run itself did not happen, independent of
    # whether the sections it did produce were individually clean.
    structural: list[str] = []
    if not results:
        structural.append(
            "NOTHING WAS CHECKED: 0 sections. all([]) is True, so an unguarded "
            "verdict would report a clean pass having read no artifact."
        )
    else:
        if expected_sections is not None and len(results) != expected_sections:
            structural.append(
                f"INCOMPLETE RUN: {len(results)} section(s), expected "
                f"{expected_sections}. A section was dropped, so this verdict "
                f"would cover less than the artifact set it claims."
            )
        if total_values == 0:
            structural.append(
                "NOTHING WAS CHECKED: sections were present but held 0 values."
            )

    ok = not structural and all(r.ok for r in results)

    if ok:
        lines = [f"OK: {total_values} Q8.8 values verified, 0 mismatches."]
        if cross_checked:
            lines.append(
                f"    {cross_checked} of them cross-validated value-by-value "
                f"against snn_model.json floats"
            )
            lines.append(
                f"    (worst |json - mem| residual {worst:.6f} LSB)."
            )
        if pinned:
            lines.append(
                f"    {pinned} signed output weight(s) pinned to the "
                f"shipped-file canonical-token sha256"
            )
            lines.append(
                "    / gold hex words, plus sign-integrity and canonical "
                "round-trip (defense in depth)."
            )
        lines.append(
            "    This is the float→Q8.8 encoding half of issue #4; it is not "
            "a close of #4."
        )
        lines.append(
            "    Hamming-on-holdout is the close bar, not weight MSE / "
            "bit-identical export."
        )
        print("\n".join(lines), file=stream)
    else:
        lines = ["FAILED:"]
        lines.extend(f"        {reason}" for reason in structural)
        failed = [r.name.split("(")[0].strip() for r in results if not r.ok]
        if failed:
            detail = (
                f"{total_mismatches} value mismatch(es)"
                if total_mismatches
                else "no value mismatches, but a section-level invariant broke "
                "(see notes above)"
            )
            lines.append(
                f"        {len(failed)} of {len(results)} section(s) did not "
                f"verify -- {', '.join(failed)}."
            )
            lines.append(f"        {detail}.")
        print("\n".join(lines), file=stream)
    return ok


# ---------------------------------------------------------------------------
# Self-test: prove the checker can fail
# ---------------------------------------------------------------------------


class SelfTestFailure(AssertionError):
    """The verifier failed to catch a regression it must catch."""


def _require(condition: bool, message: str) -> None:
    """Assert a self-test invariant.

    Raises:
        SelfTestFailure: ``condition`` is false. Used instead of bare
            ``assert`` so the checks survive ``python -O``.

    """
    if not condition:
        raise SelfTestFailure(message)


def _write_mem(path: Path, words: list[int]) -> None:
    """Write raw 16-bit words as a .mem image, one 4-digit hex line each.

    Self-test scratch files only -- callers pass a temp path. Nothing here
    ever writes into ``dataset/``.
    """
    path.write_text("".join(f"{w:04X}\n" for w in words), encoding="utf-8")


# Numbered scenarios self_test() must run. Counted and asserted at the end so
# a section deleted or accidentally skipped fails loudly instead of silently
# shrinking the suite.
SELF_TEST_SECTIONS = 11


def self_test(stream=sys.stdout) -> bool:
    """Prove this checker actually rejects real regressions."""
    print("Q8.8 verifier self-test", file=stream)
    print("", file=stream)
    checks = 0
    sections_run = 0

    # -- 1. codec unit vectors (the examples documented in README.md) ---------
    print("1. Q8.8 codec unit vectors", file=stream)
    vectors = [
        ("0100", 1.0),
        ("00DA", 218 / 256),
        ("00CC", 204 / 256),
        ("0000", 0.0),
        ("FFF9", -7 / 256),
        ("FFFF", -1 / 256),
        ("8000", -128.0),
        ("7FFF", 32767 / 256),
    ]
    for hex_text, expected in vectors:
        got = decode_q88(int(hex_text, 16))
        _require(
            got == expected,
            f"decode_q88({hex_text}) = {got!r}, expected {expected!r}",
        )
        back = encode_q88_hex(got)
        _require(
            back == hex_text,
            f"round-trip {hex_text} -> {got!r} -> {back}, expected {hex_text}",
        )
        checks += 2
    print(f"   ok: {len(vectors)} vectors decode and round-trip", file=stream)
    print("", file=stream)
    sections_run += 1

    # -- 2. the shipped artifacts must pass ----------------------------------
    print("2. shipped artifacts verify", file=stream)
    shipped = verify_shipped()
    _require(
        len(shipped) == EXPECTED_SECTIONS,
        f"verify_shipped() returned {len(shipped)} sections, expected "
        f"{EXPECTED_SECTIONS} -- a section was dropped",
    )
    _require(
        all(r.ok for r in shipped),
        "the shipped .mem files do not verify -- see the default (non-self-test) run",
    )
    _require(
        report(shipped, stream=io.StringIO(), expected_sections=EXPECTED_SECTIONS),
        "the shipped artifacts did not survive the full report() verdict",
    )
    print(
        f"   ok: {len(shipped)} sections, {sum(r.count for r in shipped)} values, "
        f"0 mismatches",
        file=stream,
    )
    checks += 3
    print("", file=stream)
    sections_run += 1

    with tempfile.TemporaryDirectory(prefix="q88-selftest-") as tmpdir:
        tmp = Path(tmpdir)

        # -- 3. THE ONE THAT MATTERS: clamp-to-zero output weights -----------
        #
        # Take the real output weights, re-encode them with the issue #15
        # clamp-to-zero encoder (negatives collapse to 0), and assert this
        # verifier rejects the result. The file is still 48 valid hex lines,
        # so only a sign-aware check catches it.
        print(
            "3. clamp-to-zero output weights are REJECTED (issue #15 hazard)",
            file=stream,
        )
        real = parse_mem(MEM_OUTPUT)
        real_floats = [e.value for e in real]
        n_negative = sum(1 for v in real_floats if v < 0)
        _require(
            n_negative > 0,
            "shipped output weights have no negatives; self-test cannot "
            "demonstrate the clamp regression",
        )

        clamped_path = tmp / "parameters_output_weights_clamped.mem"
        _write_mem(clamped_path, [encode_q88_clamp_to_zero(v) for v in real_floats])
        clamped = parse_mem(clamped_path)

        # 3a. against the true float vector, every negative must be flagged.
        against_truth = check_against_floats(
            "output weights (clamp-to-zero candidate)",
            "true float vector vs clamp-to-zero encoding",
            real_floats,
            [f"output weight {i}" for i in range(len(real_floats))],
            clamped,
        )
        _require(
            not against_truth.ok,
            "verifier ACCEPTED a clamp-to-zero output-weight file compared "
            "against the true floats -- it cannot fail, so it is worthless",
        )
        _require(
            len(against_truth.mismatches) == n_negative,
            f"expected {n_negative} flagged values (one per inhibitory weight), "
            f"got {len(against_truth.mismatches)}",
        )
        checks += 2
        print(
            f"   ok: rejected, {len(against_truth.mismatches)} of "
            f"{len(real_floats)} values flagged "
            f"(exactly the {n_negative} inhibitory weights)",
            file=stream,
        )
        for mismatch in against_truth.mismatches[:3]:
            print(mismatch.render(), file=stream)
        if len(against_truth.mismatches) > 3:
            print(
                f"       ... and {len(against_truth.mismatches) - 3} more",
                file=stream,
            )

        # 3b. and the standalone signed-section check must reject it too,
        #     without needing any float reference at all.
        invariants = check_signed_section(
            "output weights (clamp-to-zero candidate)",
            "signed two's-complement invariants",
            clamped,
            N_OUTPUT_WEIGHTS,
        )
        _require(
            not invariants.ok,
            "sign-integrity check ACCEPTED a clamp-to-zero file -- the shipped "
            "output weights would not be protected, since snn_model.json has "
            "no output layer to compare against",
        )
        _require(
            any("SIGN INTEGRITY FAILURE" in n for n in invariants.notes),
            "clamp-to-zero file was rejected, but not for losing its negative "
            f"weights; notes were {invariants.notes}",
        )
        checks += 2
        print(
            "   ok: sign-integrity check also rejects it with no float "
            "reference needed",
            file=stream,
        )
        print("", file=stream)
        sections_run += 1

        # -- 4. the JSON-backed path must fail too when data drifts ----------
        print("4. a single corrupted hidden weight is REJECTED", file=stream)
        hidden_entries = parse_mem(MEM_HIDDEN)
        model = load_model(MODEL_JSON)
        hidden_floats = [
            float(w) for neuron in model["neurons"] for w in neuron["weights"]
        ]
        corrupt_words = [e.word for e in hidden_entries]
        corrupt_words[137] ^= 0x0001  # one LSB, the smallest possible drift
        corrupt_path = tmp / "parameters_weights_corrupt.mem"
        _write_mem(corrupt_path, corrupt_words)
        corrupted = check_against_floats(
            "hidden weights (corrupt candidate)",
            "snn_model.json vs 1-LSB-corrupted mem",
            hidden_floats,
            [f"hidden {i}" for i in range(len(hidden_floats))],
            parse_mem(corrupt_path),
        )
        _require(
            not corrupted.ok and len(corrupted.mismatches) == 1,
            f"expected exactly 1 mismatch from a 1-LSB corruption, got "
            f"{len(corrupted.mismatches)}",
        )
        _require(
            corrupted.mismatches[0].index == 137,
            f"flagged index {corrupted.mismatches[0].index}, expected 137",
        )
        checks += 2
        print("   ok: rejected, 1 mismatch at the corrupted index", file=stream)
        print(corrupted.mismatches[0].render(), file=stream)
        print("", file=stream)
        sections_run += 1

        # -- 5. truncated file is REJECTED -----------------------------------
        print("5. a truncated .mem is REJECTED", file=stream)
        short_path = tmp / "parameters_output_weights_short.mem"
        _write_mem(short_path, [e.word for e in real][:-1])
        short = check_signed_section(
            "output weights (truncated candidate)",
            "signed two's-complement invariants",
            parse_mem(short_path),
            N_OUTPUT_WEIGHTS,
        )
        _require(not short.ok, "verifier ACCEPTED a 47-value output-weight file")
        checks += 1
        print("   ok: rejected on value count", file=stream)
        print("", file=stream)
        sections_run += 1

        # -- 6. an empty/short result set FAILS; residuals stay crash-safe ----
        #
        # Two distinct concerns that the previous version conflated, and got
        # backwards in the second case:
        #
        #   * Amazon Q: max() over an empty generator of max_residual_lsb
        #     raises ValueError. worst_residual_lsb() must return 0.0 for an
        #     empty list and for signed-only sections. Still true, still tested.
        #   * Codex: report([]) returned SUCCESS -- all([]) is True -- printing
        #     "OK: 0 Q8.8 values verified" plus the claim that 48 output
        #     weights were pinned, having read nothing. The old section 6
        #     asserted that pass and so enshrined the bug. An empty (or short)
        #     result set is now a hard failure.
        print(
            "6. empty/short results FAIL; empty residual list stays crash-safe",
            file=stream,
        )
        _require(
            worst_residual_lsb([]) == 0.0,
            "empty results must report 0.0 residual, not raise "
            "(the max()-on-empty crash must stay fixed)",
        )
        silent = io.StringIO()
        empty_ok = report([], stream=silent)
        _require(
            not empty_ok,
            "report([]) returned SUCCESS -- a verifier that reports a clean "
            "pass having checked nothing is a false-confidence failure",
        )
        empty_text = silent.getvalue()
        _require(
            "OK:" not in empty_text,
            f"report([]) failed but still printed an OK line: {empty_text!r}",
        )
        _require(
            "pinned" not in empty_text,
            "report([]) claimed output weights were pinned, having read nothing",
        )
        checks += 4

        # A short run -- a section dropped upstream -- must fail too, even
        # though every section it does contain verifies cleanly.
        _require(
            shipped[0].ok,
            "sanity: the first shipped section must itself verify, or the "
            "short-run assertion below would prove nothing",
        )
        silent = io.StringIO()
        short_run_ok = report(
            shipped[:1], stream=silent, expected_sections=EXPECTED_SECTIONS
        )
        _require(
            not short_run_ok,
            f"report() accepted 1 of {EXPECTED_SECTIONS} sections as a clean run",
        )
        _require(
            "INCOMPLETE RUN" in silent.getvalue(),
            f"short run was rejected, but not as an incomplete run: "
            f"{silent.getvalue()!r}",
        )
        checks += 3

        # The signed-only path (max_residual_lsb is None) must still pass and
        # must still report 0.0 rather than raising.
        signed_only = [
            check_signed_section(
                "output weights (no residual)",
                "pin + signed invariants",
                parse_mem(MEM_OUTPUT),
                N_OUTPUT_WEIGHTS,
                expected_hex=OUTPUT_WEIGHTS_HEX,
                expected_canon_sha256=OUTPUT_WEIGHTS_CANON_SHA256,
            )
        ]
        _require(
            all(r.max_residual_lsb is None for r in signed_only),
            "signed-only section must carry no residual data (this is the crash path)",
        )
        _require(
            worst_residual_lsb(signed_only) == 0.0,
            "signed-only sections must report 0.0 residual, not raise",
        )
        silent = io.StringIO()
        signed_ok = report(signed_only, stream=silent)
        _require(signed_ok, "signed-only report() should pass and not raise")
        _require(
            "cross-validated" not in silent.getvalue(),
            f"signed-only report() claimed a float cross-validation that never "
            f"happened: {silent.getvalue()!r}",
        )
        checks += 4
        print(
            "   ok: report([]) and a 1-of-4-section run both FAIL; "
            "worst_residual_lsb([]) returns 0.0;\n"
            "       signed-only run passes and claims only the pin",
            file=stream,
        )
        print("", file=stream)
        sections_run += 1

        # -- 7. well-formed output-weight corruption is REJECTED (the pin) -----
        #
        # Sign-integrity + decode→re-encode accept any well-formed 48-word
        # file with at least one high-bit word. The gold pin is what makes
        # FFF9→FFF8 and "47 zeros + FFFF" fail.
        print(
            "7. well-formed output-weight corruption is REJECTED (shipped-file pin)",
            file=stream,
        )
        gold = tuple(e.text.upper() for e in real)
        _require(
            gold == OUTPUT_WEIGHTS_HEX,
            "gold tuple derived from the shipped file does not match the pin",
        )
        _require(
            canonical_mem_digest(real) == OUTPUT_WEIGHTS_CANON_SHA256,
            "shipped output-weight canonical-token sha256 does not match the pin",
        )
        _require(
            len(gold) == N_OUTPUT_WEIGHTS,
            f"shipped file has {len(gold)} words, expected {N_OUTPUT_WEIGHTS}",
        )
        _require(
            gold[0] == "FFF9",
            f"self-test expects shipped word 0 to be FFF9, got {gold[0]}",
        )

        mutated_words = [e.word for e in real]
        mutated_words[0] = 0xFFF8
        mut_path = tmp / "parameters_output_weights_fff8.mem"
        _write_mem(mut_path, mutated_words)
        unpinned_fff8 = check_signed_section(
            "output weights (FFF9→FFF8, no pin)",
            "sign-integrity + round-trip only",
            parse_mem(mut_path),
            N_OUTPUT_WEIGHTS,
        )
        _require(
            unpinned_fff8.ok,
            "sanity: FFF9→FFF8 still passes sign+round-trip; the pin is load-bearing",
        )
        pinned_fff8 = check_signed_section(
            "output weights (FFF9→FFF8)",
            "gold hex pin",
            parse_mem(mut_path),
            N_OUTPUT_WEIGHTS,
            expected_hex=gold,
        )
        _require(
            not pinned_fff8.ok,
            "verifier ACCEPTED FFF9→FFF8 corruption against the gold pin",
        )
        _require(
            len(pinned_fff8.mismatches) == 1 and pinned_fff8.mismatches[0].index == 0,
            f"expected 1 mismatch at index 0, got {pinned_fff8.mismatches}",
        )
        checks += 6
        print("   ok: FFF9→FFF8 rejected by gold pin, accepted without it", file=stream)

        junk_path = tmp / "parameters_output_weights_junk.mem"
        _write_mem(junk_path, [0] * 47 + [0xFFFF])
        unpinned_junk = check_signed_section(
            "output weights (47 zeros + FFFF, no pin)",
            "sign-integrity + round-trip only",
            parse_mem(junk_path),
            N_OUTPUT_WEIGHTS,
        )
        _require(
            unpinned_junk.ok,
            "sanity: 47 zeros + FFFF still passes sign+round-trip; the pin is load-bearing",
        )
        pinned_junk = check_signed_section(
            "output weights (47 zeros + FFFF)",
            "gold hex pin",
            parse_mem(junk_path),
            N_OUTPUT_WEIGHTS,
            expected_hex=gold,
        )
        _require(
            not pinned_junk.ok,
            "verifier ACCEPTED 47 zeros + one FFFF against the gold pin",
        )
        checks += 2
        print("   ok: 47 zeros + FFFF rejected by gold pin, accepted without it", file=stream)
        print("", file=stream)
        sections_run += 1

        # -- 8. malformed model JSON exits through ParseError -----------------
        #
        # CodeRabbit: a list/null top level made model.get("neurons") raise
        # AttributeError, and a null neuron made `key not in neuron` raise
        # TypeError. Both escaped the documented ParseError path, so main()
        # would traceback instead of returning exit code 2. Shape must be
        # validated before any field is read.
        print(
            "8. non-object model records are REJECTED as ParseError",
            file=stream,
        )
        good_model = load_model(MODEL_JSON)

        null_neuron_model = json.loads(json.dumps(good_model))
        null_neuron_model["neurons"][3] = None

        scalar_weights_model = json.loads(json.dumps(good_model))
        scalar_weights_model["neurons"][0]["weights"] = N_INPUTS

        malformed_models = [
            ("list top level", []),
            ("null top level", None),
            ("null neuron entry", null_neuron_model),
            ("non-list neuron weights", scalar_weights_model),
        ]
        for label, payload in malformed_models:
            bad_path = tmp / f"model_{label.replace(' ', '_')}.json"
            bad_path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                load_model(bad_path)
            except ParseError:
                pass
            except Exception as exc:  # AttributeError / TypeError = the bug
                raise SelfTestFailure(
                    f"{label}: load_model raised "
                    f"{type(exc).__name__}({exc}), expected ParseError"
                ) from exc
            else:
                raise SelfTestFailure(
                    f"{label}: load_model ACCEPTED a malformed model"
                )
            checks += 1
        print(
            f"   ok: {len(malformed_models)} malformed models "
            f"({', '.join(label for label, _ in malformed_models)}) "
            f"all raise ParseError",
            file=stream,
        )
        print("", file=stream)
        sections_run += 1

        # -- 9. nonnumeric scalar fields exit through ParseError --------------
        #
        # Codex: load_model() validated container shape but not scalar type,
        # so a model with the right shape and {"threshold": null} -- or an
        # object-valued weight -- passed every check and then raised an
        # uncaught TypeError at float(), tracebacking instead of honoring the
        # documented exit-code-2 parse-error contract.
        #
        # bool is the subtle one: it subclasses int, so isinstance(True, int)
        # is True and a naive numeric guard would silently encode True as 1.0.
        print(
            "9. nonnumeric threshold/decay/weight values are REJECTED as ParseError",
            file=stream,
        )

        # Positive controls first: the guard must not reject valid numbers,
        # including a JSON integer, or it would just be breaking the tool.
        _require(
            as_finite_float(1, "int") == 1.0,
            "as_finite_float rejected a valid JSON integer",
        )
        _require(
            as_finite_float(-0.02734375, "float") == -0.02734375,
            "as_finite_float rejected a valid JSON float",
        )
        try:
            as_finite_float(1e308, "neurons[0].threshold")
        except ParseError as exc:
            _require(
                "neurons[0].threshold" in str(exc) and "Q8.8" in str(exc),
                f"huge-finite ParseError must name the field and Q8.8; got {exc}",
            )
        except OverflowError as exc:
            raise SelfTestFailure(
                "as_finite_float leaked OverflowError on 1e308"
            ) from exc
        else:
            raise SelfTestFailure("as_finite_float ACCEPTED 1e308")
        try:
            encode_q88(1e308)
        except Q88RangeError:
            pass
        except OverflowError as exc:
            raise SelfTestFailure("encode_q88 leaked OverflowError on 1e308") from exc
        else:
            raise SelfTestFailure("encode_q88 ACCEPTED 1e308")
        checks += 4

        bad_scalars = [
            ("null threshold", "threshold", None),
            ("string threshold", "threshold", "0.5"),
            ("object threshold", "threshold", {"value": 0.5}),
            ("bool threshold", "threshold", True),
            ("nan threshold", "threshold", float("nan")),
            ("posinf threshold", "threshold", float("inf")),
            ("null decay", "decay_rate", None),
            ("list decay", "decay_rate", [0.9]),
            ("bool decay", "decay_rate", False),
            ("neginf decay", "decay_rate", float("-inf")),
            ("huge finite threshold", "threshold", 1e308),
        ]
        bad_weights = [
            ("null weight", None),
            ("object weight", {"w": 0.1}),
            ("string weight", "0.1"),
            ("bool weight", True),
            ("nan weight", float("nan")),
        ]

        def _reject(label: str, payload: object, bad_path: Path) -> None:
            """Assert load_model(bad_path) raises ParseError, nothing else."""
            bad_path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                load_model(bad_path)
            except ParseError:
                return
            except Exception as exc:  # TypeError from float() = the bug
                raise SelfTestFailure(
                    f"{label}: load_model raised {type(exc).__name__}({exc}), "
                    f"expected ParseError"
                ) from exc
            raise SelfTestFailure(
                f"{label}: load_model ACCEPTED a nonnumeric field"
            )

        for label, field, payload in bad_scalars:
            broken = json.loads(json.dumps(good_model))
            broken["neurons"][2][field] = payload
            _reject(label, broken, tmp / f"model_{label.replace(' ', '_')}.json")
            checks += 1
        for label, payload in bad_weights:
            broken = json.loads(json.dumps(good_model))
            broken["neurons"][5]["weights"][7] = payload
            _reject(label, broken, tmp / f"model_{label.replace(' ', '_')}.json")
            checks += 1

        print(
            f"   ok: {len(bad_scalars)} bad thresholds/decays and "
            f"{len(bad_weights)} bad weights all raise ParseError\n"
            f"       (null, string, object/list, bool, NaN, +-inf, 1e308); "
            f"valid ints and floats still accepted",
            file=stream,
        )
        print("", file=stream)
        sections_run += 1

        # -- 10. a CRLF checkout of the shipped file still verifies -----------
        #
        # Codex: the pin used to hash the file's raw bytes. On a Windows
        # checkout with Git's core.autocrlf, the same 48 words materialize
        # with CRLF endings, so the hash failed on a semantically unchanged
        # artifact. The pin now hashes the canonical token sequence instead
        # (and .gitattributes pins *.mem to LF as defense in depth).
        print(
            "10. a CRLF-line-ending copy of the shipped file still verifies",
            file=stream,
        )
        # Both fixtures are *constructed*, never assumed from the checkout.
        # Reading the file and calling it the LF form is wrong on a CRLF
        # checkout -- there the two fixtures come out identical and the
        # negative control below fails, reporting a bug in the file that is
        # really a bug in this test. Normalize to LF first, then build CRLF
        # from it, so both forms exist whatever git handed us.
        shipped_bytes = MEM_OUTPUT.read_bytes()
        lf_bytes = shipped_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        _require(
            b"\r\n" in crlf_bytes and b"\r" not in lf_bytes,
            "CRLF fixture is not actually CRLF -- this section would prove nothing",
        )
        _require(
            crlf_bytes != lf_bytes,
            "the two line-ending fixtures are identical -- the file has no newline",
        )
        _require(
            hashlib.sha256(crlf_bytes).hexdigest()
            != hashlib.sha256(lf_bytes).hexdigest(),
            "negative control: a raw-byte hash must differ across line endings, "
            "otherwise this section is not testing the reported bug",
        )
        crlf_path = tmp / "parameters_output_weights_crlf.mem"
        crlf_path.write_bytes(crlf_bytes)
        crlf_entries = parse_mem(crlf_path)
        _require(
            len(crlf_entries) == N_OUTPUT_WEIGHTS,
            f"CRLF copy parsed to {len(crlf_entries)} words, expected "
            f"{N_OUTPUT_WEIGHTS}",
        )
        _require(
            canonical_mem_digest(crlf_entries) == OUTPUT_WEIGHTS_CANON_SHA256,
            "the canonical-token pin is still line-ending dependent",
        )
        crlf_result = check_signed_section(
            "output weights (CRLF checkout)",
            "shipped-file canonical sha256 + gold hex pin",
            crlf_entries,
            N_OUTPUT_WEIGHTS,
            expected_hex=OUTPUT_WEIGHTS_HEX,
            expected_canon_sha256=OUTPUT_WEIGHTS_CANON_SHA256,
        )
        _require(
            crlf_result.ok,
            f"a CRLF copy of the shipped output weights failed verification: "
            f"{crlf_result.notes}",
        )
        checks += 5

        # Same idea one step further: lower-case words with padding around
        # them are the same artifact, and canonicalization must say so.
        noisy_path = tmp / "parameters_output_weights_noisy.mem"
        noisy_path.write_text(
            "".join(f"  {e.text.lower()}  \r\n" for e in real), encoding="utf-8"
        )
        _require(
            canonical_mem_digest(parse_mem(noisy_path)) == OUTPUT_WEIGHTS_CANON_SHA256,
            "canonicalization must normalize hex case and surrounding whitespace",
        )
        checks += 1
        print(
            "   ok: CRLF copy verifies against both pins "
            "(raw bytes differ, canonical tokens do not);\n"
            "       lower-case + padded words canonicalize identically",
            file=stream,
        )
        print("", file=stream)
        sections_run += 1

        # -- 11. invalid UTF-8 .mem exits through ParseError -----------------
        #
        # Codex: parse_mem() opened the file as UTF-8 and iterated lines;
        # a corrupted image with invalid UTF-8 raised UnicodeDecodeError,
        # which main() does not catch. Wrap the decode as ParseError
        # (same pattern as the JSON read path). Temp file only -- never
        # mutate dataset/merged_v2.
        print(
            "11. a .mem with invalid UTF-8 is REJECTED as ParseError",
            file=stream,
        )
        bad_utf8_path = tmp / "parameters_output_weights_bad_utf8.mem"
        bad_utf8_path.write_bytes(b"FFF9\n\xff\xfe\n001E\n")
        try:
            parse_mem(bad_utf8_path)
        except ParseError:
            pass
        except UnicodeDecodeError as exc:
            raise SelfTestFailure(
                "parse_mem leaked UnicodeDecodeError on invalid UTF-8"
            ) from exc
        except Exception as exc:
            raise SelfTestFailure(
                f"parse_mem raised {type(exc).__name__}({exc}), expected ParseError"
            ) from exc
        else:
            raise SelfTestFailure("parse_mem ACCEPTED a .mem with invalid UTF-8")
        try:
            read_utf8_text(bad_utf8_path)
        except ParseError:
            pass
        except UnicodeDecodeError as exc:
            raise SelfTestFailure(
                "read_utf8_text leaked UnicodeDecodeError"
            ) from exc
        else:
            raise SelfTestFailure("read_utf8_text ACCEPTED invalid UTF-8")
        checks += 2

        # An unreadable-but-present artifact takes the same exit-2 path.
        # A read-protected file is the realistic case, but it cannot be
        # exercised as root, so a directory and a vanished path stand in --
        # both raise OSError from read_text after is_file() has passed or
        # been bypassed, which is the failure being guarded.
        for label, target in (
            ("a directory", tmp),
            ("a vanished file", tmp / "does_not_exist.mem"),
        ):
            try:
                read_utf8_text(target)
            except ParseError:
                pass
            except OSError as exc:
                raise SelfTestFailure(
                    f"read_utf8_text leaked {type(exc).__name__} on {label}"
                ) from exc
            else:
                raise SelfTestFailure(f"read_utf8_text ACCEPTED {label}")
            checks += 1
        print("   ok: invalid UTF-8 .mem raises ParseError, not UnicodeDecodeError", file=stream)
        print("   ok: an unreadable artifact raises ParseError, not OSError", file=stream)
        print("", file=stream)
        sections_run += 1

    _require(
        sections_run == SELF_TEST_SECTIONS,
        f"self-test ran {sections_run} numbered sections, expected "
        f"{SELF_TEST_SECTIONS} -- a scenario was dropped",
    )
    checks += 1

    print(
        f"SELF-TEST PASSED: {checks} assertions across {sections_run} sections.\n"
        f"The verifier rejects clamp-to-zero signed encoding, 1-LSB weight "
        f"drift, truncated\n"
        f"memory images, and well-formed output-weight corruption against the "
        f"shipped-file\n"
        f"pin; malformed model JSON, nonnumeric fields, Q8.8-scale overflow "
        f"(1e308), and\n"
        f"invalid-UTF-8 .mem files exit through ParseError rather than a "
        f"traceback; an empty\n"
        f"or short result set FAILS while worst_residual_lsb() stays "
        f"crash-safe; a CRLF\n"
        f"checkout of the output-weight image still verifies; and the "
        f"shipped artifacts are\n"
        f"accepted.",
        file=stream,
    )
    return True


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    It handles ``SelfTestFailure``, ``ParseError``, ``Q88RangeError`` and
    ``json.JSONDecodeError``, and nothing else: there is no catch-all, so a
    bug in the verifier surfaces as a traceback rather than as a clean
    verdict on the artifacts. That is deliberate — a checker that swallows
    its own defects and prints a verdict anyway is the failure this whole
    tool exists to prevent.

    Exit codes are the contract this tool is consumed by:
    0 = everything verified, 1 = a value or invariant did not verify (or a
    self-test assertion failed), 2 = an artifact could not be parsed at all,
    so nothing was actually checked.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Q8.8 export pipeline: snn_model.json floats vs the "
            "FPGA .mem files (issue #4)."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "prove the checker can fail: assert it rejects clamp-to-zero signed "
            "encoding, 1-LSB weight drift, truncated files, well-formed "
            "output-weight corruption, malformed model JSON and nonnumeric "
            "fields, Q8.8-scale overflow, invalid-UTF-8 .mem files, and an "
            "empty or short result set; and that a CRLF checkout still verifies"
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            self_test()
            return 0
        return (
            0
            if report(verify_shipped(), expected_sections=EXPECTED_SECTIONS)
            else 1
        )
    except SelfTestFailure as exc:
        print(f"\nSELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    except (ParseError, Q88RangeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
