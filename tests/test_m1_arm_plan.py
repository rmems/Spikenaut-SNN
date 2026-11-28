"""CPU tests for the M1 feature-arm availability and modality boundary."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.test_m1_cohort import CliResult, invoke_main
from tools.hamming_encode import Sample
from tools.m1_arm_plan import SMA_WINDOW, build_plan, main, row_eligibility
from tools.m1_cohort import Cohort, FIXTURE_JSONL
from tools.model_bank import SHIPPED_MANIFEST, load_model_bank

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> CliResult:
    return invoke_main(main, *args)

def rows(*missing: tuple[str, ...]) -> list[Sample]:
    return [Sample("gpu-000001", 1, [], i, columns) for i, columns in enumerate(missing, 1)]


def eligible(samples: list[Sample]) -> list[tuple[bool, bool]]:
    cohort = Cohort(samples, {"gpu-000001": "train", "gpu-000002": "train"}, {})
    return [(raw, kinetic) for _, _, _, raw, kinetic in row_eligibility(cohort)]


class ArmPlanTest(unittest.TestCase):
    def test_cli_pins_cohort_and_distinct_modality(self) -> None:
        first = run_cli()
        second = run_cli()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first, second)
        plan = json.loads(first.stdout)
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

    def test_reference_rejects_a_different_attested_input_contract(self) -> None:
        bank = load_model_bank(SHIPPED_MANIFEST)
        entry = replace(bank.select("merged_v2"), feature_map_id="spikenaut.m1.raw-rate.v1")
        with patch("tools.m1_arm_plan.load_model_bank", return_value=replace(bank, entries=(entry,))):
            result = run_cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("feature_map_id", result.stderr)
        self.assertIn("not the replayed contract", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_invalid_lock_metadata_refuses_cleanly(self) -> None:
        valid = (
            b'[[package]]\nname = "kinetic-signals"\nversion = "1.0.0"\n'
            b'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
            b'checksum = "' + b"a" * 64 + b'"\n'
        )
        cases = (
            (b"version = 4", "named [[package]] entries"),
            (b"package = [1]", "named [[package]] entries"),
            (b"package = [\xff", "invalid Cargo.lock"),
            (b"package = [", "invalid Cargo.lock"),
            (valid.replace(b'version = "1.0.0"', b'version = "bad"'), "invalid kinetic-signals version"),
            (valid.replace(b"a" * 64, b"bad"), "invalid kinetic-signals checksum"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("tools.m1_arm_plan.ROOT", root):
                for payload, expected in cases:
                    with self.subTest(expected=expected):
                        (root / "Cargo.lock").write_bytes(payload)
                        result = run_cli()
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertIn(expected, result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
                        self.assertEqual(result.stdout, "")

    def test_v3_thermal_dropout_breaks_the_candidate_window(self) -> None:
        splits = {"train": ["gpu-000001"], "val": ["gpu-000140"], "test": ["gpu-000170"]}
        temperatures = {
            "train": [35] * (SMA_WINDOW - 1) + [0, 35],
            "val": [35] * SMA_WINDOW,
            "test": [35] * SMA_WINDOW,
        }
        records = [
            {
                "episode_id": episodes[0],
                "mem_util_pct": 0,
                "power_w": 50,
                "gpu_temp_c": temperature,
                "sm_clock_mhz": 180,
                "mem_clock_mhz": 405,
            }
            for split, episodes in splits.items()
            for temperature in temperatures[split]
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, manifest = root / "telemetry.jsonl", root / "splits.json"
            data.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            manifest.write_text(
                json.dumps({"schema_version": "spikenaut.split-manifest.v1", "splits": splits}),
                encoding="utf-8",
            )
            result = run_cli("--jsonl", str(data), "--split-manifest", str(manifest))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        train = report["paired_coverage"]["train"]
        self.assertEqual(train["available_rows"], {"RAW": SMA_WINDOW, "KINETIC": 0, "HYBRID": 0})
        self.assertEqual(report["paired_coverage"]["val"]["common_rows"], 1)
        self.assertEqual(report["splits_without_common_rows"], ["train"])
