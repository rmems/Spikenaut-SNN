#!/usr/bin/env python3
"""Attest a Spikenaut model-bank manifest before any entry is selectable.

Native checkpoint/manifest validation command for Linear RM-1327. Default
target is the shipped ``dataset/merged_v2`` one-entry bank. ``--self-test``
proves the loader rejects a missing file, a digest mismatch, a duplicate
model ID, an unsupported schema version, and a missing required contract ID.

Standard library only. Run from anywhere::

    python3 tools/verify_model_bank.py
    python3 -m tools.verify_model_bank
    python3 tools/verify_model_bank.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # package import: `python3 -m tools.verify_model_bank`
    from .model_bank import (
        SHIPPED_MANIFEST,
        BankAttestationError,
        BankParseError,
        ModelBank,
        load_model_bank,
    )
    from .model_bank_selftest import self_test
    from .q88_core import SelfTestFailure
except ImportError:  # direct script: `python3 tools/verify_model_bank.py`
    from model_bank import (
        SHIPPED_MANIFEST,
        BankAttestationError,
        BankParseError,
        ModelBank,
        load_model_bank,
    )
    from model_bank_selftest import self_test
    from q88_core import SelfTestFailure


def format_bank_report(bank: ModelBank) -> str:
    """Human-readable attestation report, including computed digests."""
    origin = (
        bank.manifest_path.as_posix()
        if bank.manifest_path is not None
        else f"(legacy wrapper under {bank.root.as_posix()})"
    )
    lines = [
        f"model-bank: attested {len(bank)}/{len(bank)} entries",
        f"  source: {origin}",
    ]
    for entry in bank.entries:
        lines.extend(
            [
                f"  {entry.id}",
                f"    checkpoint: {entry.checkpoint_relative}",
                f"    computed digest: {entry.checkpoint_digest}",
                f"    feature_map_id: {entry.feature_map_id}",
                f"    output_contract_id: {entry.output_contract_id}",
                f"    numeric_format: {entry.numeric_format}",
            ]
        )
        if entry.training_dataset_digest is not None:
            lines.append(
                f"    training_dataset_digest: {entry.training_dataset_digest}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    0 = every entry attested, 1 = an entry or field failed attestation (or a
    self-test assertion failed), 2 = the manifest could not be parsed at all.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Attest a model-bank manifest before selection (Linear RM-1327)."
        )
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=SHIPPED_MANIFEST,
        help=(
            "path to model_bank.json (default: the shipped merged_v2 bank)"
        ),
    )
    parser.add_argument(
        "--select",
        metavar="ID",
        help="after a successful load, select one attested entry by id",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "prove the loader can fail: missing file, digest mismatch, "
            "duplicate model ID, unsupported schema version, missing "
            "required contract ID, and that a copied valid bundle still "
            "attests"
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            self_test()
            return 0
        bank = load_model_bank(args.manifest)
        print(format_bank_report(bank))
        if args.select is not None:
            entry = bank.select(args.select)
            print(f"selected: {entry.id}")
            print(f"computed digest: {entry.checkpoint_digest}")
        return 0
    except SelfTestFailure as exc:
        print(f"\nSELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    except BankAttestationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except BankParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
