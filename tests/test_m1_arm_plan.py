"""CPU tests for the M1 feature-arm availability and modality boundary."""

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from tools.hamming_encode import Sample
from tools.m1_arm_plan import SMA_WINDOW, build_plan, row_eligibility
from tools.m1_cohort import Cohort, FIXTURE_JSONL

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, module: bool = False) -> subprocess.CompletedProcess[str]:
    entry = ["-m", "tools.m1_arm_plan"] if module else ["tools/m1_arm_plan.py"]
    return subprocess.run(
        [sys.executable, *entry, *args], cwd=ROOT, text=True, capture_output=True, check=False
    )


def rows(*missing: tuple[str, ...]) -> list[Sample]:
    return [Sample("gpu-000001", 1, [], i, columns) for i, columns in enumerate(missing, 1)]


def eligible(samples: list[Sample]) -> list[tuple[bool, bool]]:
    cohort = Cohort(samples, {"gpu-000001": "train", "gpu-000002": "train"}, {})
    return [(raw, kinetic) for _, _, _, raw, kinetic in row_eligibility(cohort)]


class ArmPlanTest(unittest.TestCase):
    def test_cli_both_entry_points_pin_cohort_and_distinct_modality(self) -> None:
        script = run_cli()
        module = run_cli(module=True)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(module.returncode, 0, module.stderr)
        self.assertEqual(script.stdout, module.stdout)
        plan = json.loads(script.stdout)
        self.assertEqual(plan["status"], "method-preflight-only")
        self.assertEqual(plan["cohort"]["evidence"], "synthetic-method-fixture")
        self.assertEqual(
            plan["cohort"]["input_sha256"]["jsonl"],
            "sha256:" + hashlib.sha256(FIXTURE_JSONL.read_bytes()).hexdigest(),
        )
        self.assertEqual(plan["analog_reference"]["model_id"], "merged_v2")
        self.assertTrue(plan["analog_reference"]["separate_reference_only"])
        self.assertEqual(plan["analog_reference"]["input_modality"], "analog")
        self.assertTrue(all(arm["checkpoint"] is None for arm in plan["arms"].values()))
        self.assertEqual(len({arm["planned_feature_map_id"] for arm in plan["arms"].values()}), 3)
        self.assertEqual(plan["paired_coverage"]["test"]["common_rows"], 50)
        self.assertEqual(plan["paired_coverage"]["val"]["common_rows"], 25)
        sequence = "".join(f"gpu-000140,{i}\n" for i in range(72, 97)).encode("ascii")
        self.assertEqual(
            plan["paired_coverage"]["val"]["common_rows_sha256"],
            "sha256:" + hashlib.sha256(sequence).hexdigest(),
        )
        self.assertEqual(plan["splits_without_common_rows"], [])

    def test_history_is_causal(self) -> None:
        prefix = rows(*([()] * SMA_WINDOW))
        longer = prefix + [Sample("gpu-000001", 1, [], SMA_WINDOW + 1, ("power_w",))]
        self.assertEqual(eligible(prefix), eligible(longer)[:SMA_WINDOW])
        self.assertEqual(eligible(prefix), [(True, False)] * (SMA_WINDOW - 1) + [(True, True)])

    def test_null_breaks_a_window_without_imputing(self) -> None:
        samples = rows(*([()] * SMA_WINDOW), ("power_w",), *([()] * SMA_WINDOW))
        mask = eligible(samples)
        self.assertEqual(mask[SMA_WINDOW], (False, False))
        self.assertEqual(mask[SMA_WINDOW + 1:-1], [(True, False)] * (SMA_WINDOW - 1))
        self.assertEqual(mask[-1], (True, True))

    def test_raw_only_sensor_gap_leaves_derived_history_valid(self) -> None:
        samples = rows(*([()] * SMA_WINDOW), ("mem_clock_mhz",))
        self.assertEqual(eligible(samples)[-1], (False, True))

    def test_new_episode_starts_fresh(self) -> None:
        samples = rows(*([()] * SMA_WINDOW)) + [Sample("gpu-000002", 2, [], SMA_WINDOW + 1)]
        self.assertEqual(eligible(samples)[-2:], [(True, True), (True, False)])

    def test_window_tracks_rust_sma_configuration(self) -> None:
        source = (ROOT / "src" / "kinetic.rs").read_text(encoding="utf-8")
        self.assertIn(f"pub const SMA_WINDOW: usize = {SMA_WINDOW};", source)

    def test_short_sessions_explicitly_block_common_coverage(self) -> None:
        cohort = Cohort(
            [
                Sample("gpu-000001", 1, [], 1),
                Sample("gpu-000140", 140, [], 2),
                Sample("gpu-000170", 170, [], 3),
            ],
            {"gpu-000001": "train", "gpu-000140": "val", "gpu-000170": "test"},
            {},
        )
        plan = build_plan(cohort)
        self.assertEqual(plan["splits_without_common_rows"], ["train", "val", "test"])
        self.assertEqual(plan["paired_coverage"]["test"]["available_rows"]["RAW"], 1)
        self.assertEqual(plan["paired_coverage"]["test"]["common_rows"], 0)

    def test_no_switch_can_route_spikes_into_shipped_reference(self) -> None:
        result = run_cli("--model-id", "merged_v2")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
