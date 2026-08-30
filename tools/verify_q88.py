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
   (computed from the file as shipped). Sign-integrity and decode->re-encode
   stay as defense in depth; they are not enough on their own (a well-formed
   FFF9->FFF8 swap still round-trips).
4. Exits non-zero with a per-value report on any mismatch. An empty or short
   result set is a hard failure, never a pass: reporting a clean verification
   having read nothing is worse than crashing, because it gets believed.

This is the float->Q8.8 encoding half of issue #4. Passing here does not
close #4: weight MSE / bit-identical export is not the close bar
(Hamming-on-holdout is).

``--self-test`` proves the checker can actually fail; see ``q88_selftest`` for
the scenarios it builds and rejects.

Layout: this file is argument handling and exit codes only. ``q88_core`` holds
the codec, parsing and section checks; ``q88_report`` renders the
verdict; ``q88_selftest`` holds the self-test.

Standard library only. Run from anywhere::

    python3 tools/verify_q88.py
    python3 tools/verify_q88.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys

from q88_core import (
    EXPECTED_ARTIFACTS,
    EXPECTED_SECTIONS,
    ParseError,
    Q88RangeError,
    SelfTestFailure,
    verify_shipped,
)
from q88_report import report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    It handles ``SelfTestFailure``, ``ParseError``, ``Q88RangeError`` and
    ``json.JSONDecodeError``, and nothing else: there is no catch-all, so a
    bug in the verifier surfaces as a traceback rather than as a clean
    verdict on the artifacts. That is deliberate -- a checker that swallows
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
            # Imported here, not at module scope: the self-test pulls in
            # tempfile and the whole scenario suite, which a plain
            # verification run has no use for.
            from q88_selftest import self_test

            self_test()
            return 0
        return (
            0
            if report(
                verify_shipped(),
                expected_sections=EXPECTED_SECTIONS,
                expected_artifacts=EXPECTED_ARTIFACTS,
            )
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
