"""Deterministic synthetic collector-format data, exclusively for machinery tests."""

from datetime import datetime, timezone
import math
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tools.anticipation.campaign import build_campaign, write_json


def build_fixture(root):
    root = Path(root)
    campaign = build_campaign(root)
    campaign["min_examples_per_session"] = 1
    campaign["synthetic_fixture_only"] = True
    for number, session in enumerate(campaign["sessions"]):
        path = Path(session["path"])
        path.mkdir(parents=True, exist_ok=True)
        start = 1700000000000 + number * 30000
        rows = _fixture_rows(session, start, number)
        pq.write_table(pa.Table.from_pylist(rows), path / "telemetry_batch_1.parquet")

        def stamp(ms):
            return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()

        write_json(
            path / "session_manifest.json",
            {
                "schema_version": 1,
                "session_id": session["session_id"],
                "session_label": session["session_id"],
                "started_at_utc": stamp(start),
                "run_started_at_utc": stamp(start),
                "ended_at_utc": stamp(start + 18100),
                "poll_interval_ms_requested": 100,
                "collector_version": "synthetic-fixture",
                "git_commit": "synthetic-fixture",
                "restart_count": 0,
                "unclean_restart_count": 0,
                "host": {
                    "gpu_name": "synthetic-fixture",
                    "driver": None,
                    "cpu_model": None,
                },
                "workload": {"class": "ai-compute", "label": session["session_id"]},
                "parquet_write_failures": 0,
                "prior_runs": [],
                "timing": {
                    "scope": "latest_process",
                    "poll_interval_ms_requested": 100,
                    "sample_count": len(rows),
                    "observed_interval_ms": {"p50": 100, "p95": 100, "max": 100},
                    "late_sample_count": 0,
                    "skipped_tick_estimate": 0,
                    "elapsed_basis": "monotonic",
                    "row_timestamp_basis": "wall_clock_utc",
                },
            },
        )
    write_json(root / "campaign.json", campaign)
    return root / "campaign.json"


def _fixture_rows(session, start, number):
    rows = []
    for i in range(181):
        phase = i / 8 + number / 3
        rows.append(
            {
                "timestamp_ms": start + i * 100,
                "session_label": session["session_id"],
                "memory_used_mb": 2000 + 100 * math.sin(phase / 2),
                "power_usage_mw": 100000 + 20000 * math.sin(phase),
                "temperature_c": 50 + 3 * math.sin(phase / 4),
                "graphics_clock_mhz": 1500 + 200 * math.sin(phase),
                "memory_clock_mhz": 10000 + 100 * math.cos(phase),
            }
        )
    return rows
