#!/usr/bin/env python3
"""Pin the analog ``stim`` vector both front ends must agree on (issue #52).

``dataset/merged_v2`` was trained and evaluated on **analog current**, not on
spikes: ``hamming_lif.keep_lif_step`` steps it as ``v = decay * v + W @ stim``
and ``tools/HAMMING_PROTOCOL.md`` records the exp-024 condition as "analog
current, Poisson unused when ``learn=false``". ``hamming_encode.encode_record``
is the reference that builds that ``stim``. The Rust crate now ships an adapter
for the same contract -- ``spikenaut_snn::stim::LiveStimAdapter``.

Two implementations of one encoder drift silently unless something compares
them. This tool is that something: it runs the Python reference over
``tools/fixtures/live_stim/`` and pins the result in ``expected_stim.json``.
``tests/live_stim.rs`` reads the *same files* and asserts the Rust adapter
reproduces the pin exactly, so neither side can move -- on any input class
pinned here -- without the other failing.

Two fixtures, because one file cannot carry both jobs
-----------------------------------------------------
``reading.jsonl`` is v3 ``state_telemetry`` rows, read through the same
``select_samples`` path ``hamming_core.measure`` uses. One concern per row:
both ends of every frozen span, a plausible mid-load reading, out-of-span
values that clamp rather than reject, a null column and an absent one (both
encode as 0, never an imputed neighbour), and a reading on every sensor whose
**binary32 snap** changes the encoded value -- ``hamming_const.f32`` snaps each
raw sample before the affine map, and for uniform in-span readings between 27%
and 36% per sensor land on a different ``f32`` if that snap is skipped.

``BIT_EXACT_CASES`` covers what JSON literals cannot express at all: ``NaN``,
the infinities, a finite ``f64`` too large for binary32, negative zero, and
subnormals. Those readings travel in the pin as **big-endian IEEE-754 hex**,
so no decimal round-trip sits between the two languages, and each case records
whether the reference *refused* it. Without them the refusal path -- the whole
non-finite contract -- would have no cross-language coverage at all, and
either side could change it with CI green.

Every accepted vector also carries the ``unused_axons: "5:15"`` contract: axons
5-15 are exactly ``0.0``, checked the way ``hamming_core._unused_axon_notes``
checks it.

This also pins ``hamming_const.FROZEN_MINMAX`` against
``dataset/merged_v2/snn_model.json``, which is the Python mirror of the Rust
``shipped_bank_frozen_minmax_matches_live_raw_ranges``. Both encoders claim to
normalise against the spans the sidecar records; until now only one of them
was held to it.

This is a cross-language pin, not a measurement and not a threshold. It says
the two encoders agree; it says nothing about how well the bank performs.

Exit codes match ``verify_q88.py``:
0 = the pin matches (or the pin was written, or the self-test passed),
1 = the pin disagrees with the reference / a self-test assertion failed,
2 = a fixture could not be parsed, so nothing was actually compared.

Standard library only. Run from anywhere::

    python3 tools/live_stim_parity.py
    python3 -m tools.live_stim_parity
    python3 tools/live_stim_parity.py --self-test
    python3 tools/live_stim_parity.py --write   # regenerate the pin
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

try:  # package import: `python3 -m tools.live_stim_parity`
    from .hamming_const import (
        FROZEN_LINEAGE,
        FROZEN_MINMAX,
        LIVE_COLUMNS,
        N_INPUTS,
        REPO_ROOT,
        SHIPPED_DIR,
        UNUSED_AXONS,
    )
    from .hamming_encode import Sample, encode_record, load_jsonl, select_samples
    from .q88_core import ParseError, SelfTestFailure, read_utf8_text
except ImportError:  # direct script: `python3 tools/live_stim_parity.py`
    from hamming_const import (
        FROZEN_LINEAGE,
        FROZEN_MINMAX,
        LIVE_COLUMNS,
        N_INPUTS,
        REPO_ROOT,
        SHIPPED_DIR,
        UNUSED_AXONS,
    )
    from hamming_encode import Sample, encode_record, load_jsonl, select_samples
    from q88_core import ParseError, SelfTestFailure, read_utf8_text

FIXTURE_DIR = REPO_ROOT / "tools" / "fixtures" / "live_stim"
READING_JSONL = FIXTURE_DIR / "reading.jsonl"
EXPECTED_STIM = FIXTURE_DIR / "expected_stim.json"
SHIPPED_MODEL = SHIPPED_DIR / "snn_model.json"

#: Relative path, so the pin reads the same from any checkout.
_READING_REL = "tools/fixtures/live_stim/reading.jsonl"

_NOTE = (
    "Cross-language pin for the analog stim contract (#52): "
    "tools/hamming_encode.encode_record vs Rust "
    "spikenaut_snn::stim::LiveStimAdapter, on the shared fixtures named by "
    "`fixture` and carried inline as `bit_exact`. Regenerate with "
    "`python3 tools/live_stim_parity.py --write`. Not a measurement and not a "
    "Hamming threshold."
)

# Readings a JSON literal cannot carry, or cannot carry safely. Each is a raw
# five-sensor reading in LIVE_COLUMNS order; whether the reference accepts it
# is *derived*, never asserted here, so this table cannot quietly encode an
# expectation the reference does not actually hold.
#
# `-0.0` is in here for a subtle reason. Python's `max(0.0, -0.0)` returns
# `+0.0`, while Rust's `f64::clamp` preserves `-0.0` and only loses the sign in
# the trailing `lo + (hi - lo) * unit` of `normalize_live`. The two agree, but
# they agree through different arithmetic, so the agreement is worth a pin
# rather than a comment.
_PLAUSIBLE = (42.0, 180.0, 61.0, 1900.0, 9500.0)


def _with(axon: int, value: float) -> list[float]:
    """``_PLAUSIBLE`` with one sensor replaced."""
    reading = list(_PLAUSIBLE)
    reading[axon] = value
    return reading


BIT_EXACT_CASES: tuple[tuple[str, list[float]], ...] = (
    ("nan on power_w", _with(1, float("nan"))),
    ("+inf on mem_util_pct", _with(0, float("inf"))),
    ("-inf on mem_clock_mhz", _with(4, float("-inf"))),
    ("f64 far above the binary32 grid", _with(2, 1e300)),
    ("f64 just above the binary32 ceiling", _with(3, 3.4028235677973366e38)),
    ("largest f64 that still snaps to f32::MAX", _with(3, 3.4028234663852886e38)),
    ("negative zero on a span whose minimum is zero", _with(0, -0.0)),
    ("negative zero on gpu_temp_c", _with(2, -0.0)),
    ("smallest positive f64 (snaps to zero in binary32)", _with(1, 5e-324)),
    ("binary32 subnormal", _with(4, 1e-45)),
    ("every sensor non-finite at once", [float("nan")] * 5),
    ("mid-span, one f64 ulp apart (a)", _with(0, 23.54026713081089)),
    ("mid-span, one f64 ulp apart (b)", _with(0, 23.540267130810893)),
)


def _f64_bits(value: float) -> str:
    """Big-endian IEEE-754 binary64 hex: the bit pattern, MSB first."""
    return struct.pack(">d", value).hex()


def _f32_bits(value: float) -> str:
    """Big-endian IEEE-754 binary32 hex: the bit pattern, MSB first."""
    return struct.pack(">f", value).hex()


def reference_samples() -> list[Sample]:
    """Encode the JSONL fixture through the reference encoder.

    ``select_samples(..., 'all')`` rather than ``encode_record`` alone: that is
    the path ``hamming_core.measure`` takes to build the ``stim`` it steps, so
    the pin covers the row refusals too (a malformed ``episode_id`` or a
    forbidden ``*_derived`` sensor fails here, not silently downstream).
    """
    return select_samples(load_jsonl(READING_JSONL), "all")


def reference_stim(reading: list[float]) -> list[float] | None:
    """``encode_record`` on one raw reading, or ``None`` if it is refused.

    Goes through the public reference encoder rather than reimplementing its
    rules, so ``_live_number``'s finiteness check and ``f32``'s binary32-range
    check both apply exactly as they do on the JSONL path.
    """
    # strict: a BIT_EXACT_CASES row of the wrong length must be a loud
    # ValueError at pin time, not a silently dropped sensor that `encode_record`
    # would then encode as a missing 0.0 -- pinning a vector for an input the
    # label does not describe.
    record = dict(zip(LIVE_COLUMNS, reading, strict=True))
    try:
        return encode_record(record)
    except ParseError:
        return None


def _unused_axon_leaks(stim: list[float]) -> list[int]:
    """Axons 5-15 that are not exactly zero. Same test as ``hamming_core``.

    Truthiness rather than ``!= 0.0`` on purpose, and it is the *same* test:
    ``0.0`` and ``-0.0`` are the only falsy floats, so every value this keeps
    is one ``hamming_core._unused_axon_notes`` would also flag, ``NaN``
    included. A tolerance would be the wrong fix here -- the bank's contract is
    exactly zero, not nearly zero -- and this formulation keeps the exactness
    while not tripping the float-equality rule that a literal comparison does.
    """
    return [axon for axon in UNUSED_AXONS if stim[axon]]


def _shape_problem(stim: list[float], where: str) -> str | None:
    """Why ``stim`` is not a legal vector, or ``None`` if it is."""
    if len(stim) != N_INPUTS:
        return f"{where}: expected {N_INPUTS} axons, got {len(stim)}"
    leaks = _unused_axon_leaks(stim)
    if leaks:
        return (
            f"{where}: UNUSED-AXON LEAK on axons {leaks} -- the bank records "
            'unused_axons "5:15" and they must be exactly 0.0'
        )
    return None


def frozen_minmax_failures() -> list[str]:
    """How ``FROZEN_MINMAX`` disagrees with the shipped sidecar, if at all.

    The Python mirror of the Rust
    ``shipped_bank_frozen_minmax_matches_live_raw_ranges``. Without it the
    reference encoder could be normalising against spans the shipped bank
    never froze, and the cross-language pin would agree on the wrong answer.
    """
    if not SHIPPED_MODEL.is_file():
        raise ParseError(f"missing shipped sidecar: {SHIPPED_MODEL}")
    try:
        model = json.loads(read_utf8_text(SHIPPED_MODEL))
    except json.JSONDecodeError as exc:
        raise ParseError(f"{SHIPPED_MODEL.name}: {exc}") from exc
    recorded = model.get("frozen_minmax")
    if not isinstance(recorded, dict):
        raise ParseError(f"{SHIPPED_MODEL.name}: no frozen_minmax object")

    return [
        failure
        for failure in (
            _span_failure(column, recorded.get(column)) for column in LIVE_COLUMNS
        )
        if failure is not None
    ]


def _span_failure(column: str, span: object) -> str | None:
    """How the sidecar's span for one column disagrees, or ``None``.

    Real JSON numbers only. ``float("0.0")`` succeeds, so a sidecar that
    recorded its spans as strings would have satisfied this contract check
    while the Rust side -- which decodes into f64 -- rejected the same file.
    Booleans are ints in Python and are refused alongside non-numbers.
    """
    if not isinstance(span, list) or len(span) != 2:
        return f"frozen_minmax: sidecar has no span for {column}"
    if any(isinstance(end, bool) or not isinstance(end, (int, float)) for end in span):
        return (
            f"frozen_minmax[{column}]: sidecar span {span!r} is not two JSON numbers"
        )
    want = (float(span[0]), float(span[1]))
    got = FROZEN_MINMAX[column]
    if got != want:
        return (
            f"frozen_minmax[{column}]: hamming_const has {got}, "
            f"{SHIPPED_MODEL.name} records {want}"
        )
    return None


def binary32_endpoint_failures() -> list[str]:
    """Frozen span endpoints that are not exactly binary32-representable.

    The reference divides by ``f32(hi) - f32(lo)`` while the Rust adapter
    divides by the raw ``f64`` difference. Those are the same number only
    while every endpoint round-trips through binary32 unchanged -- which all
    ten shipped endpoints do. It is an accident of the shipped spans, not a
    property of the code, so it is pinned rather than assumed.
    """
    failures: list[str] = []
    for column in LIVE_COLUMNS:
        lo, hi = FROZEN_MINMAX[column]
        for name, value in (("min", lo), ("max", hi)):
            snapped = struct.unpack(">f", struct.pack(">f", value))[0]
            if snapped != value:
                failures.append(
                    f"frozen_minmax[{column}].{name} = {value!r} is not exactly "
                    f"binary32 (snaps to {snapped!r}); the two encoders would "
                    "divide by different spans"
                )
    return failures


def _pin_metadata() -> dict:
    """The protocol fields of the pin: what encoder, which spans, which file."""
    return {
        "fixture": _READING_REL,
        "live_columns": list(LIVE_COLUMNS),
        "unused_axons": "5:15",
        "frozen_lineage": FROZEN_LINEAGE,
        "frozen_minmax": {
            column: list(FROZEN_MINMAX[column]) for column in LIVE_COLUMNS
        },
        "encoder": (
            "legal 5-ch train-scaled analog stim; frozen minmax lineage "
            f"{FROZEN_LINEAGE}; axons 0-4 = " + ", ".join(LIVE_COLUMNS) +
            "; unused axons 5-15 = 0; raw samples snapped to binary32 before "
            "the affine map"
        ),
        "stepper": "v = decay * v + W @ stim (analog current, Poisson unused)",
        "note": _NOTE,
    }


def _pin_samples() -> list[dict]:
    """The JSONL fixture, encoded. Raises rather than pinning an illegal row."""
    samples = []
    for sample in reference_samples():
        problem = _shape_problem(
            sample.stim, f"{_READING_REL}:{sample.source_line}"
        )
        if problem is not None:
            raise EncoderLeak(problem)
        samples.append(
            {
                "line": sample.source_line,
                "episode_id": sample.episode_id,
                "stim": list(sample.stim),
            }
        )
    if not samples:
        raise ParseError(
            f"NOTHING WAS COMPARED: {_READING_REL} encoded 0 rows. "
            "An empty pin would pass for both implementations at once."
        )
    return samples


def _pin_bit_exact_case(label: str, reading: list[float]) -> dict:
    """One bit-exact case: the reading, and what the reference did with it."""
    case = {"label": label, "reading_bits": [_f64_bits(value) for value in reading]}
    stim = reference_stim(reading)
    if stim is None:
        case["refused"] = True
        return case
    problem = _shape_problem(stim, f"bit_exact[{label}]")
    if problem is not None:
        raise EncoderLeak(problem)
    case["refused"] = False
    case["stim_bits"] = [_f32_bits(value) for value in stim]
    return case


def _pin_bit_exact() -> list[dict]:
    """Every reading JSON literals cannot carry, and its verdict."""
    cases = [
        _pin_bit_exact_case(label, reading) for label, reading in BIT_EXACT_CASES
    ]
    if not any(case["refused"] for case in cases):
        raise ParseError(
            "no bit-exact case is refused, so the refusal path is unpinned "
            "again. That is the hole this section exists to close."
        )
    return cases


def build_pin() -> dict:
    """The pin payload for the current reference encoder."""
    return {
        **_pin_metadata(),
        "samples": _pin_samples(),
        "bit_exact": _pin_bit_exact(),
    }


class EncoderLeak(Exception):
    """The reference encoder itself produced an illegal vector.

    Its own class so ``main`` can report it as a **failed comparison** (exit 1)
    rather than as an unparseable fixture (exit 2). An ``UNUSED-AXON LEAK`` out
    of ``encode_record`` is an encoder regression, which is precisely the thing
    this tool exists to catch -- calling it a parse error would file the loudest
    possible finding under the quietest possible heading.
    """


def _read_pin_object(path: Path) -> dict:
    """The pin file as a JSON object, or a ParseError naming why not."""
    if not path.is_file():
        raise ParseError(
            f"missing pin: {path}. Write it with "
            "`python3 tools/live_stim_parity.py --write`."
        )
    try:
        payload = json.loads(read_utf8_text(path))
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"{path.name}: expected a JSON object")
    return payload


def _check_pinned_sample(entry: object, index: int, where: str) -> None:
    """One pinned sample must be an object holding a legal numeric vector."""
    if not isinstance(entry, dict):
        raise ParseError(f"{where}: sample {index} is not an object")
    stim = entry.get("stim")
    if not isinstance(stim, list):
        raise ParseError(f"{where}: sample {index} has no `stim` array")
    for axon, value in enumerate(stim):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParseError(f"{where}: sample {index} axon {axon} is not a number")
    problem = _shape_problem([float(value) for value in stim], f"{where}#{index}")
    if problem is not None:
        raise ParseError(problem)


def _check_pinned_bit_exact(cases: object, where: str) -> None:
    """Every bit-exact case must be an object before anything reads it.

    Without the per-entry check a valid-JSON pin holding, say, ``[1, 2]`` here
    reached ``actual.get(...)`` in the comparison and died with an
    ``AttributeError`` traceback -- a crash where the documented behaviour is
    exit 2, "a fixture could not be parsed".
    """
    if not isinstance(cases, list):
        raise ParseError(f"{where}: `bit_exact` must be an array")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ParseError(f"{where}: bit_exact case {index} is not an object")


def load_pin(path: Path) -> dict:
    """Read and shape-check the pinned payload."""
    payload = _read_pin_object(path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ParseError(f"{path.name}: `samples` must be a non-empty array")
    for index, entry in enumerate(samples):
        _check_pinned_sample(entry, index, path.name)
    _check_pinned_bit_exact(payload.get("bit_exact"), path.name)
    return payload


# Metadata the pin must reproduce exactly. `note` is deliberately absent: it is
# editorial prose about regenerating the file, and comparing it would turn a
# wording fix into a pin mismatch. Everything here is a protocol claim.
_PINNED_METADATA = (
    "fixture",
    "live_columns",
    "unused_axons",
    "frozen_lineage",
    "frozen_minmax",
    "encoder",
    "stepper",
)


def _sample_failures(want: list[dict], got: list[dict]) -> list[str]:
    """Per-row disagreements between the reference and the pinned samples."""
    failures: list[str] = []
    if len(want) != len(got):
        return [f"samples: pinned {len(got)}, reference {len(want)}"]
    for expected, actual in zip(want, got):
        line = expected["line"]
        if actual.get("line") != line:
            failures.append(f"line: pinned {actual.get('line')!r}, reference {line}")
            continue
        if actual.get("episode_id") != expected["episode_id"]:
            failures.append(
                f"line {line}: episode_id pinned {actual.get('episode_id')!r}, "
                f"reference {expected['episode_id']!r}"
            )
        pinned_stim = actual.get("stim", [])
        if len(pinned_stim) != len(expected["stim"]):
            failures.append(
                f"line {line}: stim pinned {len(pinned_stim)} axons, "
                f"reference {len(expected['stim'])}"
            )
            continue
        for axon, (a, b) in enumerate(zip(expected["stim"], pinned_stim)):
            # Exact equality on purpose: both sides are binary32 values that
            # round-trip through f64 without loss, so a tolerance here would
            # only hide a real encoder divergence.
            if float(a) != float(b):
                failures.append(
                    f"line {line} axon {axon} ({_axon_label(axon)}): "
                    f"pinned {b!r}, reference {a!r}"
                )
    return failures


def _bit_exact_case_failures(expected: dict, actual: dict) -> list[str]:
    """How one pinned bit-exact case disagrees with the reference.

    Ordered widest-first and short-circuiting: a case whose *identity* moved
    (label, or the reading itself) makes any per-axon diff meaningless, so it
    is reported alone rather than alongside sixteen confusing axon lines.
    """
    identity = _bit_exact_identity_failure(expected, actual)
    if identity is not None:
        return [identity]
    if expected["refused"]:
        return []

    label = expected["label"]
    pinned_stim = actual.get("stim_bits")
    if not isinstance(pinned_stim, list) or len(pinned_stim) != N_INPUTS:
        return [f"bit_exact[{label}]: stim_bits must be {N_INPUTS} wide"]
    return [
        f"bit_exact[{label}] axon {axon} ({_axon_label(axon)}): "
        f"pinned 0x{b}, reference 0x{a}"
        for axon, (a, b) in enumerate(zip(expected["stim_bits"], pinned_stim))
        if a != b
    ]


def _bit_exact_identity_failure(expected: dict, actual: dict) -> str | None:
    """Whether the pinned case is even the same case, and the same verdict."""
    label = expected["label"]
    if actual.get("label") != label:
        return f"bit_exact: pinned case {actual.get('label')!r}, reference {label!r}"
    if actual.get("reading_bits") != expected["reading_bits"]:
        return f"bit_exact[{label}]: the reading itself was edited"
    if actual.get("refused") != expected["refused"]:
        return (
            f"bit_exact[{label}]: pinned refused={actual.get('refused')!r}, "
            f"reference refused={expected['refused']!r}"
        )
    return None


def _bit_exact_failures(want: list[dict], got: list[dict]) -> list[str]:
    """Disagreements on the readings JSON literals cannot carry."""
    if len(want) != len(got):
        return [f"bit_exact: pinned {len(got)} cases, reference {len(want)}"]
    failures: list[str] = []
    for expected, actual in zip(want, got):
        failures.extend(_bit_exact_case_failures(expected, actual))
    return failures


def pin_failures(reference: dict, pinned: dict) -> list[str]:
    """How the pinned payload disagrees with the reference, if at all.

    Compares every protocol field in ``_PINNED_METADATA``, every JSONL sample
    value for value, and every ``bit_exact`` case including whether it was
    refused. Editorial ``note`` text is the one field deliberately not
    compared.
    """
    failures: list[str] = []
    for key in _PINNED_METADATA:
        if pinned.get(key) != reference[key]:
            failures.append(
                f"{key}: pinned {pinned.get(key)!r}, reference {reference[key]!r}"
            )
    failures.extend(_sample_failures(reference["samples"], pinned["samples"]))
    failures.extend(
        _bit_exact_failures(reference["bit_exact"], pinned.get("bit_exact", []))
    )
    return failures


def _axon_label(axon: int) -> str:
    return LIVE_COLUMNS[axon] if axon < len(LIVE_COLUMNS) else "unused"


def write_pin(path: Path = EXPECTED_STIM) -> dict:
    """Regenerate the pin from the reference encoder."""
    payload = build_pin()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def report(reference: dict, failures: list[str]) -> bool:
    """Print the comparison. Returns whether it passed."""
    refused = sum(1 for case in reference["bit_exact"] if case["refused"])
    print("Live analog stim parity pin (#52)")
    print(f"  fixture   {reference['fixture']}")
    print(f"  pin       {EXPECTED_STIM.relative_to(REPO_ROOT)}")
    print(f"  encoder   {reference['encoder']}")
    print(f"  stepper   {reference['stepper']}")
    print(f"  rows      {len(reference['samples'])} x {N_INPUTS} axons")
    print(
        f"  bit-exact {len(reference['bit_exact'])} readings JSON cannot carry "
        f"({refused} refused, {len(reference['bit_exact']) - refused} accepted)"
    )
    print(f"  axons 0-4 {', '.join(LIVE_COLUMNS)}")
    print('  axons 5-15 exactly 0.0 on every accepted vector (unused_axons "5:15")')
    print(f"  spans     match {SHIPPED_MODEL.relative_to(REPO_ROOT)} frozen_minmax")
    if failures:
        print("\nPIN MISMATCH:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nThe Python reference encoder and the pinned vectors disagree. "
            "If the reference changed on purpose, regenerate with --write and "
            "review the Rust adapter in the same commit.",
            file=sys.stderr,
        )
        return False
    print("\nOK: the pin matches tools/hamming_encode.encode_record.")
    print("    tests/live_stim.rs asserts the Rust adapter against the same file.")
    return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def _self_test_metadata(reference: dict) -> None:
    """Every pinned protocol field must be compared, not just the first."""
    mutations = {
        "fixture": "tools/fixtures/hamming_method/holdout.jsonl",
        "live_columns": list(reversed(LIVE_COLUMNS)),
        "unused_axons": "4:15",
        "frozen_lineage": "deadbeef",
        "frozen_minmax": {column: [0.0, 1.0] for column in LIVE_COLUMNS},
        "encoder": "some other encoder",
        "stepper": "Poisson spikes",
    }
    _require(
        set(mutations) == set(_PINNED_METADATA),
        "the self-test must exercise every field pin_failures compares",
    )
    for key, value in mutations.items():
        mutated = json.loads(json.dumps(reference))
        mutated[key] = value
        _require(
            bool(pin_failures(reference, mutated)),
            f"a changed {key!r} must be reported",
        )

    # `note` is the documented exception; prove it really is exempt.
    editorial = json.loads(json.dumps(reference))
    editorial["note"] = "reworded"
    _require(
        not pin_failures(reference, editorial),
        "`note` is editorial and must not be compared",
    )


def _self_test_samples(reference: dict) -> None:
    """A drifted, truncated or dropped sample row must be reported."""
    # A genuine one-ulp step, on a row whose value is not zero. Row 3 (index 2)
    # is the mid-load reading; nudging axon 0 by one binary32 ulp is the
    # smallest real drift there is, and the case a loose tolerance would hide.
    drifted = json.loads(json.dumps(reference))
    value = drifted["samples"][2]["stim"][0]
    _require(value > 0.0, "the ulp case needs a non-zero reference value")
    drifted["samples"][2]["stim"][0] = struct.unpack(
        ">f", struct.pack(">I", struct.unpack(">I", struct.pack(">f", value))[0] + 1)
    )[0]
    _require(
        bool(pin_failures(reference, drifted)),
        "a one-ulp stim drift must be reported",
    )

    # A stim array cut short must not compare equal on its surviving prefix.
    truncated = json.loads(json.dumps(reference))
    truncated["samples"][0]["stim"] = truncated["samples"][0]["stim"][:3]
    _require(
        bool(pin_failures(reference, truncated)),
        "a truncated stim array must be reported, not compared on its prefix",
    )

    short = json.loads(json.dumps(reference))
    short["samples"].pop()
    _require(
        bool(pin_failures(reference, short)), "a dropped sample row must be reported"
    )


def _first_case_index(reference: dict, *, refused: bool) -> int:
    """Index of the first bit-exact case with the given verdict."""
    for index, case in enumerate(reference["bit_exact"]):
        if case["refused"] is refused:
            return index
    raise SelfTestFailure(
        f"the pin must carry at least one {'refused' if refused else 'accepted'} "
        "bit-exact case"
    )


def _self_test_bit_exact(reference: dict) -> None:
    """The refusal contract must be pinned, and must be able to fail."""
    refused_at = _first_case_index(reference, refused=True)
    accepted_at = _first_case_index(reference, refused=False)

    # An encoder that started accepting what the reference refuses -- the
    # failure mode with no JSON-literal coverage at all.
    flipped = json.loads(json.dumps(reference))
    flipped["bit_exact"][refused_at]["refused"] = False
    flipped["bit_exact"][refused_at]["stim_bits"] = ["00000000"] * N_INPUTS
    _require(
        bool(pin_failures(reference, flipped)),
        "a refusal turned into an acceptance must be reported",
    )

    # One bit of one accepted vector.
    nudged = json.loads(json.dumps(reference))
    bits = int(nudged["bit_exact"][accepted_at]["stim_bits"][0], 16) ^ 1
    nudged["bit_exact"][accepted_at]["stim_bits"][0] = f"{bits:08x}"
    _require(
        bool(pin_failures(reference, nudged)),
        "a single flipped bit in an accepted vector must be reported",
    )

    # Editing the reading itself, which would otherwise silently repoint a
    # case at an input the label no longer describes.
    repointed = json.loads(json.dumps(reference))
    repointed["bit_exact"][0]["reading_bits"][0] = _f64_bits(1.0)
    _require(
        bool(pin_failures(reference, repointed)),
        "an edited bit_exact reading must be reported",
    )


def _self_test_leaks(reference: dict) -> None:
    """A non-zero unused axon must be refused as a leak, by its own name."""
    leaked = list(reference["samples"][0]["stim"])
    leaked[5] = 1e-30
    problem = _shape_problem(leaked, "self-test")
    _require(problem is not None, "a non-zero axon 5 must be refused as a leak")
    _require(
        "UNUSED-AXON LEAK" in problem,
        f"wrong refusal for an unused-axon leak: {problem}",
    )


def self_test() -> None:
    """Prove the comparison can fail, without touching the shipped pin."""
    reference = build_pin()
    _require(
        not pin_failures(reference, json.loads(json.dumps(reference))),
        "the reference payload must compare equal to itself",
    )
    _self_test_metadata(reference)
    _self_test_samples(reference)
    _self_test_bit_exact(reference)
    _self_test_leaks(reference)
    print("SELF-TEST PASSED: the stim parity pin can fail.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pin the analog stim vector shared by tools/hamming_encode.py and "
            "the Rust LiveStimAdapter (#52)."
        ),
    )
    # Mutually exclusive: `--write --self-test` silently ran only the
    # self-test and left the caller thinking the pin had been rewritten.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="regenerate tools/fixtures/live_stim/expected_stim.json",
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="prove the comparison can fail",
    )
    return parser


def _check_constants() -> list[str]:
    """Span failures that would make the whole comparison meaningless."""
    return frozen_minmax_failures() + binary32_endpoint_failures()


def _report_span_mismatch(failures: list[str]) -> int:
    print("SPAN MISMATCH:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


def _write_and_report() -> int:
    payload = write_pin()
    print(
        f"wrote {EXPECTED_STIM.relative_to(REPO_ROOT)} "
        f"({len(payload['samples'])} rows, "
        f"{len(payload['bit_exact'])} bit-exact cases)"
    )
    return 0


def _compare_against_pin() -> int:
    reference = build_pin()
    pinned = load_pin(EXPECTED_STIM)
    return 0 if report(reference, pin_failures(reference, pinned)) else 1


def _run(args: argparse.Namespace) -> int:
    """The mode `args` selected. Exceptions are `main`'s to translate."""
    if args.self_test:
        self_test()
        return 0
    constants = _check_constants()
    if constants:
        return _report_span_mismatch(constants)
    return _write_and_report() if args.write else _compare_against_pin()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except SelfTestFailure as exc:
        print(f"\nSELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    except EncoderLeak as exc:
        print(f"ENCODER REGRESSION: {exc}", file=sys.stderr)
        return 1
    except ParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
