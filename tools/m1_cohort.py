#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Preflight an offline M1 cohort without fitting, scoring, or opening a bank."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

if __package__:
    from .hamming_const import LIVE_COLUMNS
    from .hamming_encode import Sample, parse_jsonl_text, select_samples
    from .model_bank_errors import BankParseError
    from .model_bank_json import dumps_manifest
    from .q88_core import ParseError
    from .replay_inputs import (
        SPLIT_NAMES,
        _read_input_bytes,
        _split_manifest_entries,
        _utf8_or_refuse,
        sha256_bytes,
    )
else:  # python3 tools/m1_cohort.py
    from hamming_const import LIVE_COLUMNS
    from hamming_encode import Sample, parse_jsonl_text, select_samples
    from model_bank_errors import BankParseError
    from model_bank_json import dumps_manifest
    from q88_core import ParseError
    from replay_inputs import (
        SPLIT_NAMES,
        _read_input_bytes,
        _split_manifest_entries,
        _utf8_or_refuse,
        sha256_bytes,
    )

ROOT = Path(__file__).resolve().parent
FIXTURE_JSONL = ROOT / "fixtures" / "recurrence" / "telemetry.jsonl"
FIXTURE_SPLIT = ROOT / "fixtures" / "m1" / "splits.json"
SCHEMA_VERSION = "spikenaut.m1-cohort.v1"
PROJECTION = "spikenaut.state-telemetry.v3.live-five-projection.v1"
PROJECTION_COLUMNS = frozenset(("episode_id", *LIVE_COLUMNS))
EPISODE_ID = re.compile(r"gpu-[0-9]{6}")
TEMPERATURE = "gpu_temp_c"
# Recognize exact fixture rows independently of filename, whitespace, or line endings.
FIXTURE_ROWS = 160
FIXTURE_FINGERPRINT = "sha256:51e4173a155e7321afcf07d3a31f5e2e2988051c4dd206d993657a5017febd4f"


@dataclass(frozen=True)
class Cohort:
    samples: list[Sample]
    split_by_episode: dict[str, str]
    report: dict
    # Sensor units before the shipped analog adapter clips or rescales them.
    measurements: list[tuple[float | None, ...]] = field(default_factory=list)


def _require_ascii_episode(episode: object, where: str) -> None:
    if not isinstance(episode, str) or not EPISODE_ID.fullmatch(episode):
        raise ParseError(f"{where}: expected an ASCII gpu-NNNNNN episode ID, got {episode!r}")


def _split_lookup(path: Path, payload: bytes) -> dict[str, str]:
    try:
        splits = _split_manifest_entries(
            path, _utf8_or_refuse(payload, path, "split manifest")
        )
    except BankParseError as exc:
        raise ParseError(f"invalid split manifest: {exc}") from exc
    for split in SPLIT_NAMES:
        if not splits.get(split):
            raise ParseError(f"split {split!r} must declare at least one episode")
    # The shared validator rejects duplicate and cross-split assignments.
    for split, episodes in splits.items():
        for episode in sorted(episodes):
            _require_ascii_episode(episode, f"split {split!r}")
    return {
        episode: split for split, episodes in splits.items() for episode in episodes
    }


def _is_fixture(jsonl: Path, records: list[tuple[int, dict]]) -> bool:
    if jsonl.resolve() == FIXTURE_JSONL:
        return True
    if len(records) != FIXTURE_ROWS:
        return False
    canonical = json.dumps(
        [record for _, record in records],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(canonical) == FIXTURE_FINGERPRINT


def _project_rows(
    records: list[tuple[int, dict]],
) -> tuple[list[Sample], set[int]]:
    """Keep caller order and all ticks; mask only the documented v3 sentinel."""
    projected: list[tuple[int, dict]] = []
    temperature_dropouts: set[int] = set()
    for lineno, record in records:
        extra = record.keys() - PROJECTION_COLUMNS
        if extra:
            raise ParseError(
                f"line {lineno}: unsupported projected columns: {', '.join(sorted(extra))}"
            )
        _require_ascii_episode(record.get("episode_id"), f"line {lineno}")
        if len(record) == 1:
            # Every sensor is absent; preserve the tick for missing-run accounting.
            record = {**record, LIVE_COLUMNS[0]: None}
        if type(record.get(TEMPERATURE)) in (int, float) and record[TEMPERATURE] == 0:
            temperature_dropouts.add(lineno)
        projected.append((lineno, record))

    samples = select_samples(projected, "all")
    for index, sample in enumerate(samples):
        if sample.source_line in temperature_dropouts:
            samples[index] = replace(sample, missing=(*sample.missing, TEMPERATURE))
    return samples, temperature_dropouts

def _verify_coverage(samples: list[Sample], split_by_episode: dict[str, str]) -> None:
    observed = {sample.episode_id for sample in samples}
    undeclared = observed - split_by_episode.keys()
    absent = split_by_episode.keys() - observed
    if undeclared or absent:
        raise ParseError(
            "split manifest must cover the projection exactly: "
            f"undeclared episodes={len(undeclared)} {sorted(undeclared)[:5]}, "
            f"missing episodes={len(absent)} {sorted(absent)[:5]}"
        )


def _summarize(
    samples: list[Sample], split_by_episode: dict[str, str], temperature_dropouts: set[int]
) -> dict[str, dict]:
    rows: Counter[str] = Counter()
    sentinel_count: Counter[str] = Counter()
    missing: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
    episodes: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for sample in samples:
        split = split_by_episode[sample.episode_id]
        rows[split] += 1
        missing[split].update(sample.missing)
        if sample.source_line in temperature_dropouts:
            sentinel_count[split] += 1
        episodes[split].add(sample.episode_id)
    return {
        split: {
            "episodes": sorted(episodes[split]),
            "rows": rows[split],
            "observed": {name: rows[split] - missing[split][name] for name in LIVE_COLUMNS},
            "missing_or_null": {
                name: missing[split][name] - (sentinel_count[split] if name == TEMPERATURE else 0)
                for name in LIVE_COLUMNS
            },
            "dropout_sentinel": {TEMPERATURE: sentinel_count[split]},
        }
        for split in SPLIT_NAMES
    }


def preflight(jsonl: Path, split_manifest: Path, source_revision: str | None) -> Cohort:
    """Validate the projected bytes against an explicit, exhaustive split."""
    split_bytes = _read_input_bytes(split_manifest, "split manifest")
    split_by_episode = _split_lookup(split_manifest, split_bytes)
    payload = _read_input_bytes(jsonl, "telemetry projection")
    records = parse_jsonl_text(
        _utf8_or_refuse(payload, jsonl, "telemetry projection"),
        jsonl.name,
        strict_keys=True,
    )
    samples, temperature_dropouts = _project_rows(records)
    _verify_coverage(samples, split_by_episode)
    is_fixture = _is_fixture(jsonl, records)
    if is_fixture and source_revision is not None:
        raise ParseError("the bundled synthetic fixture cannot declare a source revision")
    report = {
        "schema_version": SCHEMA_VERSION,
        "projection": PROJECTION,
        "live_columns": list(LIVE_COLUMNS),
        "evidence": "synthetic-method-fixture" if is_fixture else "user-provided-unverified",
        "declared_source_revision": source_revision,
        "input_sha256": {
            "jsonl": sha256_bytes(payload),
            "split_manifest": sha256_bytes(split_bytes),
        },
        "splits": _summarize(samples, split_by_episode, temperature_dropouts),
    }
    if len(samples) != len(records):
        raise ParseError("preflight must retain every source row for physical-unit features")
    measurements = []
    for sample, (line, row) in zip(samples, records, strict=True):
        if sample.source_line != line:
            raise ParseError(f"feature measurement and selected sample disagree at line {line}")
        measurements.append(tuple(row.get(name) for name in LIVE_COLUMNS))
    return Cohort(samples, split_by_episode, report, measurements)


def add_cohort_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-revision", help="caller-declared dataset revision; not verified")


def cohort_from_args(args: argparse.Namespace) -> Cohort:
    if (args.jsonl is None) != (args.split_manifest is None):
        raise ParseError("--jsonl and --split-manifest must be supplied together")
    return preflight(
        args.jsonl or FIXTURE_JSONL,
        args.split_manifest or FIXTURE_SPLIT,
        args.source_revision,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args(argv)
    try:
        cohort = cohort_from_args(args)
    except (ParseError, OSError, OverflowError) as exc:
        print(f"M1 cohort refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(dumps_manifest(cohort.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
