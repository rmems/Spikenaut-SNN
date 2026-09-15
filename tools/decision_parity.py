#!/usr/bin/env python3
"""Pin the output-row decision contract both languages must agree on (RM-1328).

``tools/decision_core.py`` is the Python reference. ``src/decision.rs`` is
the Rust one. This tool regenerates the golden fixture from the reference
and compares it to ``tools/fixtures/decision/expected.json``.
``tests/decision.rs`` reads the same file, so neither side can move -- on
any case pinned here -- without the other failing.

CI runs both halves. ``--self-test`` proves the comparison can fail.
``--write`` regenerates the pin.

Exit codes match ``verify_q88.py``:
0 = the pin matches (or was written, or the self-test passed),
1 = the pin disagrees / a self-test assertion failed,
2 = a fixture could not be parsed.

Standard library only. Run from anywhere::

    python3 tools/decision_parity.py
    python3 -m tools.decision_parity
    python3 tools/decision_parity.py --self-test
    python3 tools/decision_parity.py --write
"""

from __future__ import annotations

import argparse
import sys

try:
    from .decision_pin import (
        EXPECTED_DECISION,
        build_pin,
        load_pin,
        pin_failures,
        write_pin,
    )
    from .decision_selftest import run_self_test
    from .hamming_const import REPO_ROOT
    from .q88_core import ParseError
except ImportError:
    from decision_pin import (
        EXPECTED_DECISION,
        build_pin,
        load_pin,
        pin_failures,
        write_pin,
    )
    from decision_selftest import run_self_test
    from hamming_const import REPO_ROOT
    from q88_core import ParseError


def _report_ok(reference: dict) -> int:
    cases = reference["cases"]
    errors = sum(1 for case in cases if not case["ok"])
    proposes = sum(
        1 for case in cases if case.get("kind") == "propose"
    )
    abstains = sum(
        1 for case in cases if case.get("kind") == "abstain"
    )
    print(
        f"OK: {EXPECTED_DECISION.relative_to(REPO_ROOT)} "
        f"matches the reference ({len(cases)} cases: "
        f"{proposes} propose, {abstains} abstain, {errors} fail-closed)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pin the Spikenaut output-row decision contract."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate tools/fixtures/decision/expected.json",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the pin comparison can fail",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    try:
        reference = build_pin()
        if args.write:
            write_pin()
            print(
                f"wrote {EXPECTED_DECISION} ({len(reference['cases'])} cases)"
            )
            return 0
        pinned = load_pin()
    except ParseError as exc:
        print(f"could not build or read the pin: {exc}", file=sys.stderr)
        return 2

    failures = pin_failures(reference, pinned)
    if failures:
        print(
            f"FAIL: {len(failures)} disagreement(s) with {EXPECTED_DECISION}",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    return _report_ok(reference)


if __name__ == "__main__":
    raise SystemExit(main())
