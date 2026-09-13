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
import sys

try:  # package import: `python3 -m tools.live_stim_parity`
    from .hamming_const import LIVE_COLUMNS, N_INPUTS, REPO_ROOT
    from .live_stim_pin import (
        EXPECTED_STIM,
        SHIPPED_MODEL,
        EncoderLeak,
        binary32_endpoint_failures,
        build_pin,
        frozen_minmax_failures,
        load_pin,
        pin_failures,
        write_pin,
    )
    from .q88_core import ParseError, SelfTestFailure
except ImportError:  # direct script: `python3 tools/live_stim_parity.py`
    from hamming_const import LIVE_COLUMNS, N_INPUTS, REPO_ROOT
    from live_stim_pin import (
        EXPECTED_STIM,
        SHIPPED_MODEL,
        EncoderLeak,
        binary32_endpoint_failures,
        build_pin,
        frozen_minmax_failures,
        load_pin,
        pin_failures,
        write_pin,
    )
    from q88_core import ParseError, SelfTestFailure

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
        # Imported here, not at module scope, for the reason `verify_q88` gives:
        # a plain verification run has no use for the mutation suite.
        try:
            from .live_stim_selftest import self_test
        except ImportError:
            from live_stim_selftest import self_test

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
