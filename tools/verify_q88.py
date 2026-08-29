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
   layer and this tool does not invent one. The shipped file is pinned by
   sha256 and the exact 48 hex words (computed from the file as shipped).
   Sign-integrity and decode→re-encode stay as defense in depth; they are
   not enough on their own (a well-formed FFF9→FFF8 swap still round-trips).
4. Exits non-zero with a per-value report on any mismatch.

This is the float→Q8.8 encoding half of issue #4. Passing here does not
close #4: weight MSE / bit-identical export is not the close bar
(Hamming-on-holdout is).

``--self-test`` proves the checker can actually fail: clamp-to-zero output
weights (issue #15), 1-LSB hidden-weight drift, truncated files, well-formed
output-weight corruption against the pin, and ``report()`` with no residual
data (the empty-``max`` crash). A checker that cannot fail is worthless.

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

# Pin of the shipped output-weight image. snn_model.json has no output layer,
# so this is the independent reference — computed from the file as shipped,
# not invented JSON floats. Update both pins together if the artifact is
# intentionally replaced.
#
#   python3 -c "import hashlib; from pathlib import Path; p=Path('dataset/merged_v2/parameters_output_weights.mem'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
OUTPUT_WEIGHTS_SHA256 = (
    "d6d3aff3c5eba76fd0206fbff377594d2c4a3c22f980e641257f4d2911405469"
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
    if scaled >= 0:
        raw = int(math.floor(scaled + 0.5))
    else:
        raw = int(math.ceil(scaled - 0.5))
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
        return decode_q88(self.word)


class ParseError(Exception):
    pass


def file_sha256(path: Path) -> str:
    """sha256 of a file's exact on-disk bytes. Read-only; never writes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_mem(path: Path) -> list[MemEntry]:
    """Parse a $readmemh-style .mem file of 4-digit hex words, one per line."""
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    entries: list[MemEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
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
    if not path.is_file():
        raise ParseError(f"missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    neurons = model.get("neurons")
    if not isinstance(neurons, list):
        raise ParseError(f"{path.name}: expected a top-level 'neurons' list")
    if len(neurons) != N_NEURONS:
        raise ParseError(
            f"{path.name}: expected {N_NEURONS} neurons, found {len(neurons)}"
        )
    for i, neuron in enumerate(neurons):
        for key in ("threshold", "decay_rate", "weights"):
            if key not in neuron:
                raise ParseError(f"{path.name}: neuron {i} is missing {key!r}")
        if len(neuron["weights"]) != N_INPUTS:
            raise ParseError(
                f"{path.name}: neuron {i} has {len(neuron['weights'])} weights, "
                f"expected {N_INPUTS}"
            )
    return model


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


@dataclass
class Mismatch:
    index: int
    label: str  # human-readable coordinate, e.g. "neuron 3 <- input 7"
    lineno: int
    source_float: float
    expected_hex: str
    actual_hex: str

    @property
    def expected_float(self) -> float:
        return decode_q88(int(self.expected_hex, 16))

    @property
    def actual_float(self) -> float:
        return decode_q88(int(self.actual_hex, 16))

    def render(self) -> str:
        return (
            f"  [{self.index:3d}] {self.label:<26} line {self.lineno:<4} "
            f"json={self.source_float:+.7f}  "
            f"expected {self.expected_hex} ({self.expected_float:+.7f})  "
            f"actual {self.actual_hex} ({self.actual_float:+.7f})  "
            f"delta={self.actual_float - self.expected_float:+.7f}"
        )


@dataclass
class SectionResult:
    name: str
    source: str
    count: int
    mismatches: list[Mismatch]
    notes: list[str]
    max_residual_lsb: float | None = None
    failed: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.mismatches

    def render(self) -> str:
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
    expected_sha256: str | None = None,
    mem_path: Path | None = None,
) -> SectionResult:
    """Check the signed output-weight section.

    snn_model.json has no output layer, so there is no float column to
    cross-validate against. The shipped file is pinned by sha256 and the
    exact 48 hex words. Sign-integrity and decode→re-encode stay as
    defense in depth: they catch clamp-to-zero and broken codecs, but a
    well-formed FFF9→FFF8 swap still round-trips.

      * shipped-file pin (when ``expected_hex`` / ``expected_sha256`` given),
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

    if expected_sha256 is not None:
        if mem_path is None:
            raise ValueError("expected_sha256 requires mem_path")
        actual_sha = file_sha256(mem_path)
        if actual_sha != expected_sha256:
            notes.append(
                f"SHIPPED-FILE PIN FAILURE: sha256 {actual_sha} "
                f"!= {expected_sha256}"
            )
            failed = True
        else:
            notes.append(f"sha256 pin {actual_sha} matches shipped file")

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
    )


# ---------------------------------------------------------------------------
# The full verification of the shipped artifacts
# ---------------------------------------------------------------------------


def verify_shipped() -> list[SectionResult]:
    model = load_model(MODEL_JSON)
    neurons = model["neurons"]

    thresholds = [float(n["threshold"]) for n in neurons]
    decays = [float(n["decay_rate"]) for n in neurons]
    hidden: list[float] = []
    hidden_labels: list[str] = []
    for i, neuron in enumerate(neurons):
        for j, weight in enumerate(neuron["weights"]):
            hidden.append(float(weight))
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
            "shipped-file sha256 + gold hex pin; snn_model.json has no output layer",
            parse_mem(MEM_OUTPUT),
            N_OUTPUT_WEIGHTS,
            expected_hex=OUTPUT_WEIGHTS_HEX,
            expected_sha256=OUTPUT_WEIGHTS_SHA256,
            mem_path=MEM_OUTPUT,
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


def report(results: list[SectionResult], stream=sys.stdout) -> bool:
    print("Q8.8 export verification", file=stream)
    print(f"repo root: {REPO_ROOT}", file=stream)
    print(f"artifacts: {DATA_DIR.relative_to(REPO_ROOT)}/", file=stream)
    print("", file=stream)
    for result in results:
        print(result.render(), file=stream)
    print("", file=stream)

    total_values = sum(r.count for r in results)
    total_mismatches = sum(len(r.mismatches) for r in results)
    ok = all(r.ok for r in results)

    cross_checked = sum(r.count for r in results if r.max_residual_lsb is not None)
    worst = worst_residual_lsb(results)

    if ok:
        print(
            f"OK: {total_values} Q8.8 values verified, 0 mismatches.\n"
            f"    {cross_checked} of them cross-validated value-by-value against "
            f"snn_model.json floats\n"
            f"    (worst |json - mem| residual {worst:.6f} LSB), and "
            f"{N_OUTPUT_WEIGHTS} signed output weights\n"
            f"    pinned to the shipped-file sha256 / gold hex words, plus "
            f"sign-integrity and\n"
            f"    canonical round-trip (defense in depth).\n"
            f"    This is the float→Q8.8 encoding half of issue #4; it is not "
            f"a close of #4.\n"
            f"    Hamming-on-holdout is the close bar, not weight MSE / "
            f"bit-identical export.",
            file=stream,
        )
    else:
        failed = [r.name.split("(")[0].strip() for r in results if not r.ok]
        detail = (
            f"{total_mismatches} value mismatch(es)"
            if total_mismatches
            else "no value mismatches, but a section-level invariant broke "
            "(see notes above)"
        )
        print(
            f"FAILED: {len(failed)} of {len(results)} section(s) did not "
            f"verify -- {', '.join(failed)}.\n"
            f"        {detail}.",
            file=stream,
        )
    return ok


# ---------------------------------------------------------------------------
# Self-test: prove the checker can fail
# ---------------------------------------------------------------------------


class SelfTestFailure(AssertionError):
    """The verifier failed to catch a regression it must catch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def _write_mem(path: Path, words: list[int]) -> None:
    path.write_text("".join(f"{w:04X}\n" for w in words), encoding="utf-8")


def self_test(stream=sys.stdout) -> bool:
    """Prove this checker actually rejects real regressions."""
    print("Q8.8 verifier self-test", file=stream)
    print("", file=stream)
    checks = 0

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

    # -- 2. the shipped artifacts must pass ----------------------------------
    print("2. shipped artifacts verify", file=stream)
    shipped = verify_shipped()
    _require(
        all(r.ok for r in shipped),
        "the shipped .mem files do not verify -- see the default (non-self-test) run",
    )
    print(
        f"   ok: {sum(r.count for r in shipped)} values, 0 mismatches", file=stream
    )
    checks += 1
    print("", file=stream)

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

        # -- 6. report() must not crash when no section has residual data ------
        #
        # Amazon Q: max() over an empty generator of max_residual_lsb raises
        # ValueError. The signed section has no residual; an empty results
        # list has none either. Both must report 0.0, not crash.
        print("6. report() survives an empty residual list", file=stream)
        _require(
            worst_residual_lsb([]) == 0.0,
            "empty results must report 0.0 residual, not raise",
        )
        silent = io.StringIO()
        empty_ok = report([], stream=silent)
        _require(empty_ok, "empty report() should not fail or raise")
        signed_only = [
            check_signed_section(
                "output weights (no residual)",
                "pin + signed invariants",
                parse_mem(MEM_OUTPUT),
                N_OUTPUT_WEIGHTS,
                expected_hex=OUTPUT_WEIGHTS_HEX,
                expected_sha256=OUTPUT_WEIGHTS_SHA256,
                mem_path=MEM_OUTPUT,
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
        checks += 5
        print("   ok: empty and signed-only report() return 0.0 residual", file=stream)
        print("", file=stream)

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
            file_sha256(MEM_OUTPUT) == OUTPUT_WEIGHTS_SHA256,
            "shipped output-weight sha256 does not match the pin",
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

    print(
        f"SELF-TEST PASSED: {checks} assertions. The verifier rejects "
        f"clamp-to-zero\n"
        f"signed encoding, 1-LSB weight drift, truncated memory images, "
        f"and well-formed\n"
        f"output-weight corruption against the shipped-file pin; "
        f"report() survives an\n"
        f"empty residual list; and the shipped artifacts are accepted.",
        file=stream,
    )
    return True


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
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
            "encoding, 1-LSB weight drift, truncated files, and well-formed "
            "output-weight corruption; and that report() survives no residuals"
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            self_test()
            return 0
        return 0 if report(verify_shipped()) else 1
    except SelfTestFailure as exc:
        print(f"\nSELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    except (ParseError, Q88RangeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
