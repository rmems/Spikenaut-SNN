#!/usr/bin/env python3
"""Publish float-vs-Q8.8 Hamming on a holdout, with its protocol (issue #39).

Successor to the half of #4 that ``verify_q88.py`` does not cover. Encoding
fidelity is already CI-gated; this measures spike-train disagreement between
a float bank and a Q8.8-decoded bank through a keep-LIF (+ optional K-WTA).

What it does
------------
1. Encodes v3 ``state_telemetry`` JSONL with the legal 5-ch train-scaled
   encoder (frozen minmax lineage 74acdd0f; unused axons 5-15 = 0).
2. Steps two 16-unit keep-LIF banks -- JSON floats as written, and ``.mem``
   decoded -- in standard-library Python. This is **not** a claim the Rust
   crate runs spikes.
3. Reports per-tick Hamming (%) and mean Hamming bits for ``k=none`` and
   ``k=4``, naming the split and the parameter sets compared.
4. Prints the full protocol: weights, encoder, episodes, seed.

This is a measurement, not a pass/fail gate. How much disagreement is
acceptable is blocked on the output/decision contract (#20). The harness
must not invent a Hamming threshold.

Default (no paths) runs the in-repo **method fixture**: a tiny deterministic
holdout that yields a pinned Hamming so CI can prove the method. That is
not exp-024. exp-024 used exp-023 PASS Distill knobs scratch (seed 123 /
5 ep) on v3 test ``gpu-000170..198`` (n=117653). Those artifacts are not
in this repository; point ``--jsonl`` / ``--float-json`` / ``--mem-dir``
at them. Do not overwrite ``dataset/merged_v2``.

``--self-test`` proves the checker can actually fail; see ``hamming_selftest``.

Exit codes match ``verify_q88.py``:
0 = measurement published (or self-test passed),
1 = a method pin / self-test assertion failed,
2 = an artifact could not be parsed, so nothing was actually measured.

Standard library only. Run from anywhere::

    python3 tools/measure_hamming.py
    python3 tools/measure_hamming.py --self-test
    python3 tools/measure_hamming.py --condition exp-024 \\
        --jsonl PATH/state_telemetry.jsonl \\
        --float-json PATH/snn_model.json --mem-dir PATH/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # package import: `python3 -m tools.measure_hamming`
    from .hamming_core import (
        CONDITION_EXP024,
        CONDITION_METHOD_FIXTURE,
        CONDITION_SHIPPED,
        I_DRIVE_EXP024,
        ParseError,
        SelfTestFailure,
        SHIPPED_DIR,
        load_expected,
        measure,
        method_fixture_paths,
        pin_matches,
    )
    from .hamming_report import report
    from .q88_core import Q88RangeError
except ImportError:  # direct script: `python3 tools/measure_hamming.py`
    from hamming_core import (
        CONDITION_EXP024,
        CONDITION_METHOD_FIXTURE,
        CONDITION_SHIPPED,
        I_DRIVE_EXP024,
        ParseError,
        SelfTestFailure,
        SHIPPED_DIR,
        load_expected,
        measure,
        method_fixture_paths,
        pin_matches,
    )
    from hamming_report import report
    from q88_core import Q88RangeError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish float-vs-Q8.8 spike-train Hamming on a holdout "
            "(issue #39). Measurement, not a pass/fail gate."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "prove the harness can fail: assert it rejects an empty holdout, "
            "malformed JSONL, a wrong method pin, unlabeled merged_v2 as "
            "exp-024, and that the in-repo fixture still yields its pinned "
            "Hamming"
        ),
    )
    parser.add_argument(
        "--condition",
        choices=(CONDITION_METHOD_FIXTURE, CONDITION_EXP024, CONDITION_SHIPPED),
        default=None,
        help=(
            "which parameter set this run is. Default with no paths is "
            "method-fixture. exp-024 refuses dataset/merged_v2."
        ),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="v3 state_telemetry JSONL (required for exp-024 / shipped-merged-v2)",
    )
    parser.add_argument(
        "--float-json",
        type=Path,
        default=None,
        help="snn_model.json float bank (default: method fixture or shipped)",
    )
    parser.add_argument(
        "--mem-dir",
        type=Path,
        default=None,
        help="directory holding parameters*.mem (default: method fixture or shipped)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default=None,
        help="episode split (default: all for method-fixture, test otherwise)",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="training seed to record on the protocol (exp-024: 123)",
    )
    parser.add_argument(
        "--i-drive",
        type=float,
        default=None,
        help=(
            "Dale I bias added to neurons 12-15 (default 0 on the method "
            f"fixture, {I_DRIVE_EXP024} on exp-024 / shipped-merged-v2)"
        ),
    )
    parser.add_argument(
        "--expect-json",
        type=Path,
        default=None,
        help=(
            "optional method pin (k_none_pct / k_4_pct / ...). Used by the "
            "in-repo fixture. Not a Hamming acceptance threshold."
        ),
    )
    parser.add_argument(
        "--weights-label",
        default=None,
        help="override the protocol's weights line",
    )
    return parser


def _infer_condition(args: argparse.Namespace) -> str:
    """Pick a condition without silently calling the shipped ramp exp-024."""
    if args.condition is not None:
        return args.condition
    if args.jsonl is None and args.float_json is None and args.mem_dir is None:
        return CONDITION_METHOD_FIXTURE
    if args.mem_dir is not None and args.mem_dir.resolve() == SHIPPED_DIR.resolve():
        return CONDITION_SHIPPED
    return CONDITION_EXP024


def _run_measurement(args: argparse.Namespace) -> int:
    condition = _infer_condition(args)

    if condition == CONDITION_METHOD_FIXTURE:
        paths = method_fixture_paths()
        measurement = measure(
            float_json=args.float_json or paths["float_json"],
            mem_dir=args.mem_dir or paths["mem_dir"],
            jsonl=args.jsonl or paths["jsonl"],
            split=args.split or "all",
            condition=CONDITION_METHOD_FIXTURE,
            seed=args.seed
            or "n/a (analog current; Poisson unused when learn=false)",
            i_drive=0.0 if args.i_drive is None else args.i_drive,
            weights_label=args.weights_label
            or (
                "in-repo method fixture tools/fixtures/hamming_method "
                "(NOT exp-023 scratch, NOT shipped merged_v2 ramp)"
            ),
        )
        ok = report(measurement)
        expect_path = args.expect_json
        if expect_path is None and args.jsonl is None and args.float_json is None:
            expect_path = paths["expect"]
        failures = (
            pin_matches(measurement, load_expected(expect_path))
            if expect_path is not None
            else []
        )
        if failures:
            print(
                "METHOD PIN FAILED (harness regression, not a Hamming gate):",
                file=sys.stderr,
            )
            for item in failures:
                print(f"  {item}", file=sys.stderr)
            return 1
        return 0 if ok else 1

    if args.jsonl is None:
        raise ParseError(
            f"condition {condition} needs --jsonl PATH to a v3 "
            "state_telemetry JSONL. The pinned corpus is not in this "
            "repository; this tool will not invent an HF download."
        )

    if condition == CONDITION_SHIPPED:
        float_json = args.float_json or (SHIPPED_DIR / "snn_model.json")
        mem_dir = args.mem_dir or SHIPPED_DIR
        split = args.split or "test"
        seed = args.seed or "n/a (shipped merged_v2 has no training seed here)"
        i_drive = I_DRIVE_EXP024 if args.i_drive is None else args.i_drive
        weights_label = args.weights_label or (
            "shipped merged_v2 ramp -- DIFFERENT condition from exp-024 "
            "(exp-023 PASS Distill knobs scratch is not in this repo)"
        )
    else:
        if args.float_json is None or args.mem_dir is None:
            raise ParseError(
                "condition exp-024 needs --float-json and --mem-dir pointing "
                "at the exp-023 PASS Distill knobs scratch. "
                "dataset/merged_v2 is refused for this condition."
            )
        float_json = args.float_json
        mem_dir = args.mem_dir
        split = args.split or "test"
        seed = args.seed or "123"
        i_drive = I_DRIVE_EXP024 if args.i_drive is None else args.i_drive
        weights_label = args.weights_label or (
            "exp-023 PASS Distill knobs scratch (seed 123 / 5 ep) -- "
            "caller-supplied paths, not shipped merged_v2"
        )

    measurement = measure(
        float_json=float_json,
        mem_dir=mem_dir,
        jsonl=args.jsonl,
        split=split,
        condition=condition,
        seed=seed,
        i_drive=i_drive,
        weights_label=weights_label,
    )
    ok = report(measurement)
    if args.expect_json is not None:
        failures = pin_matches(measurement, load_expected(args.expect_json))
        if failures:
            print(
                "METHOD PIN FAILED (harness regression, not a Hamming gate):",
                file=sys.stderr,
            )
            for item in failures:
                print(f"  {item}", file=sys.stderr)
            return 1
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Catches ``SelfTestFailure``, ``ParseError``, ``Q88RangeError`` and
    ``json.JSONDecodeError`` only. There is no catch-all: a bug in the
    harness surfaces as a traceback rather than as a clean Hamming of 0.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            try:
                from .hamming_selftest import self_test
            except ImportError:
                from hamming_selftest import self_test

            self_test()
            return 0
        return _run_measurement(args)
    except SelfTestFailure as exc:
        print(f"\nSELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    except (ParseError, Q88RangeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
