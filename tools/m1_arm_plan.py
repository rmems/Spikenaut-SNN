#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Audit causal row eligibility for a proposed RAW/KINETIC/HYBRID comparison."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

if __package__:
    from .m1_cohort import Cohort, add_cohort_args, cohort_from_args
    from .hamming_const import LIVE_COLUMNS
    from .model_bank import BankError, SHIPPED_MANIFEST, load_model_bank
    from .model_bank_json import dumps_manifest
    from .q88_core import ParseError
    from .replay_inputs import SPLIT_NAMES, select_entry, sha256_bytes
else:  # python3 tools/m1_arm_plan.py
    from m1_cohort import Cohort, add_cohort_args, cohort_from_args
    from hamming_const import LIVE_COLUMNS
    from model_bank import BankError, SHIPPED_MANIFEST, load_model_bank
    from model_bank_json import dumps_manifest
    from q88_core import ParseError
    from replay_inputs import SPLIT_NAMES, select_entry, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
CRATE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?")
CRATE_CHECKSUM = re.compile(r"[a-f0-9]{64}")
SCHEMA = "spikenaut.m1-arm-plan.v1"
# Proposed eligibility only: the existing Rust KineticPipeline returns partial SMAs.
SMA_WINDOW = 8
SMA_SOURCES = ("gpu_temp_c", "power_w")
ARMS = ("RAW", "KINETIC", "HYBRID")
UNITS = {
    "mem_util_pct": "%",
    "power_w": "W",
    "gpu_temp_c": "degC",
    "sm_clock_mhz": "MHz",
    "mem_clock_mhz": "MHz",
}


def _dependency_pins(payload: bytes) -> dict:
    try:
        packages = tomllib.loads(payload.decode("utf-8")).get("package")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ParseError(f"invalid Cargo.lock: {exc}") from exc
    if not isinstance(packages, list) or not packages or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        for item in packages
    ):
        raise ParseError("Cargo.lock must contain named [[package]] entries")

    versions: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for name in ("kinetic-signals", "axon-encoder"):
        found = [item for item in packages if item["name"] == name]
        if len(found) != 1 or found[0].get("source") != CRATES_IO:
            raise ParseError(f"Cargo.lock must pin exactly one crates.io {name} version")
        version, checksum = found[0].get("version"), found[0].get("checksum")
        if not isinstance(version, str) or not CRATE_VERSION.fullmatch(version):
            raise ParseError(f"Cargo.lock has an invalid {name} version")
        if not isinstance(checksum, str) or not CRATE_CHECKSUM.fullmatch(checksum):
            raise ParseError(f"Cargo.lock has an invalid {name} checksum")
        versions[name], checksums[name] = version, checksum
    return {
        "versions": versions,
        "checksums": checksums,
        "cargo_lock_sha256": sha256_bytes(payload),
    }


def _dependencies() -> dict:
    return _dependency_pins((ROOT / "Cargo.lock").read_bytes())

def row_eligibility(cohort: Cohort) -> Iterator[tuple[str, str, int, bool, bool]]:
    """Yield RAW and KINETIC readiness after only the current/previous rows.

    Missing measurements break only the affected sensor's run. Every new
    episode starts a fresh run. The cohort has already masked v3 thermal dropouts.
    """
    previous = None
    runs = {name: 0 for name in SMA_SOURCES}
    for sample in cohort.samples:
        if sample.episode_id != previous:
            runs = dict.fromkeys(SMA_SOURCES, 0)
            previous = sample.episode_id
        for name in SMA_SOURCES:
            runs[name] = 0 if name in sample.missing else min(runs[name] + 1, SMA_WINDOW)
        yield (
            cohort.split_by_episode[sample.episode_id],
            sample.episode_id,
            sample.source_line,
            not sample.missing,
            all(run == SMA_WINDOW for run in runs.values()),
        )


def paired_coverage(cohort: Cohort) -> dict:
    counts = {split: Counter(dict.fromkeys(ARMS, 0)) for split in SPLIT_NAMES}
    common = {split: hashlib.sha256() for split in SPLIT_NAMES}
    for split, episode, line, raw, kinetic in row_eligibility(cohort):
        counts[split]["RAW"] += raw
        counts[split]["KINETIC"] += kinetic
        if raw and kinetic:
            counts[split]["HYBRID"] += 1
            common[split].update(f"{episode},{line}\n".encode("utf-8"))
    return {
        split: {
            "available_rows": dict(counts[split]),
            "common_rows": counts[split]["HYBRID"],
            "common_rows_sha256": "sha256:" + common[split].hexdigest(),
        }
        for split in SPLIT_NAMES
    }


def _arm_specs() -> dict:
    raw = [
        {"id": name, "source": name, "unit": UNITS[name], "window_samples": 1}
        for name in LIVE_COLUMNS
    ]
    kinetic = [
        {
            "id": f"{name}.sma",
            "source": name,
            "unit": UNITS[name],
            "window_samples": SMA_WINDOW,
        }
        for name in SMA_SOURCES
    ]
    return {
        arm: {
            "planned_feature_map_id": f"spikenaut.m1.{arm.lower()}-rate.v1",
            "features": features,
            "checkpoint": None,
            "requires_separately_named_rate_checkpoint": True,
        }
        for arm, features in zip(ARMS, (raw, kinetic, raw + kinetic))
    }


def _reference() -> dict:
    entry = select_entry(load_model_bank(SHIPPED_MANIFEST), "merged_v2")
    return {
        "model_id": entry.id,
        "checkpoint_digest": entry.checkpoint_digest,
        "feature_map_id": entry.feature_map_id,
        "output_contract_id": entry.output_contract_id,
        "input_modality": "analog",
        "separate_reference_only": True,
    }


def build_plan(cohort: Cohort) -> dict:
    availability = paired_coverage(cohort)
    return {
        "schema_version": SCHEMA,
        "status": "method-preflight-only",
        "cohort": cohort.report,
        "dependency_pins": _dependencies(),
        "arms": _arm_specs(),
        "analog_reference": _reference(),
        "proposed_encoder": "axon-encoder/rate",
        "sampling": "source row order; physical cadence must be established separately",
        "normalization": "not fitted; later fit on train only",
        "missing_policy": "no imputation; mask unrunnable rows and reset SMA history across missing readings or v3 thermal dropouts",
        "measurement_status": "no features computed, encoding run, or checkpoint evaluated",
        "paired_coverage": availability,
        "splits_without_common_rows": [
            split for split in SPLIT_NAMES if not availability[split]["common_rows"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args(argv)
    try:
        report = build_plan(cohort_from_args(args))
    except (ParseError, BankError, OSError, OverflowError) as exc:
        print(f"M1 arm plan refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(dumps_manifest(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
