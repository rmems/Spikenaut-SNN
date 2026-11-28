"""CPU CLI coverage for the M1 session cohort contract."""

import hashlib
import io
import json
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from tools.m1_cohort import FIXTURE_JSONL, FIXTURE_SPLIT, main

@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def invoke_main(main_function: Callable[[list[str]], int], *args: str) -> CliResult:
    """Exercise argument parsing and the CLI's stdout/stderr contract in process."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main_function(list(args))
        except SystemExit as exc:
            code = int(exc.code)
    return CliResult(code, out.getvalue(), err.getvalue())


def invoke(*args: str) -> CliResult:
    return invoke_main(main, *args)

class CohortTest(unittest.TestCase):
    def setUp(self) -> None:
        with FIXTURE_SPLIT.open(encoding="utf-8") as source:
            self.split = json.load(source)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.data = root / "telemetry.jsonl"
        self.manifest = root / "split.json"
        self.rows = [
            '{"episode_id":"gpu-000001","power_w":0}',
            '{"episode_id":"gpu-000002","power_w":null}',
            '{"episode_id":"gpu-000140","power_w":25}',
            '{"episode_id":"gpu-000170","power_w":50}',
            '{"episode_id":"gpu-000171","gpu_temp_c":20}',
        ]

    def local(self) -> CliResult:
        self.data.write_text("\n".join(self.rows) + "\n", encoding="utf-8")
        self.manifest.write_text(json.dumps(self.split), encoding="utf-8")
        return invoke("--jsonl", str(self.data), "--split-manifest", str(self.manifest))

    def assert_refused(self, problem: str) -> None:
        result = self.local()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(problem, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_default_cli_is_deterministic_and_names_fixture(self) -> None:
        first = invoke()
        second = invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first, second)
        report = json.loads(first.stdout)
        self.assertEqual(report["evidence"], "synthetic-method-fixture")
        self.assertEqual(report["splits"]["test"]["rows"], 64)
        self.assertEqual(invoke("--source-revision", "unverified").returncode, 2)

    def test_real_path_requires_an_explicit_split(self) -> None:
        result = invoke("--jsonl", str(self.data))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be supplied together", result.stderr)

    def test_missing_null_and_observed_zero_are_counted_separately(self) -> None:
        result = self.local()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["evidence"], "user-provided-unverified")
        self.assertEqual(report["splits"]["train"]["observed"]["power_w"], 1)
        self.assertEqual(report["splits"]["train"]["missing_or_null"]["power_w"], 1)
        self.assertEqual(report["splits"]["test"]["missing_or_null"]["power_w"], 1)

    def test_split_overlap(self) -> None:
        self.split["splits"]["test"].append("gpu-000001")
        self.assert_refused("split overlap")

    def test_exact_coverage(self) -> None:
        self.rows[-1] = '{"episode_id":"gpu-000172","power_w":50}'
        self.assert_refused("undeclared episodes=1 ['gpu-000172'], missing episodes=1 ['gpu-000171']")

    def test_requires_every_named_split(self) -> None:
        del self.split["splits"]["val"]
        self.assert_refused("split 'val' must declare at least one episode")

    def test_refuses_interleaved_episodes(self) -> None:
        # Cohort preflight examines all episodes, including train rows.
        # Preflight checks the complete projection, not just the test split.
        self.rows.append(self.rows[0])
        self.assert_refused("interleaved")

    def test_malformed_split_manifest_is_a_clean_refusal(self) -> None:
        self.data.write_text(self.rows[0], encoding="utf-8")
        self.manifest.write_text("{", encoding="utf-8")
        result = invoke("--jsonl", str(self.data), "--split-manifest", str(self.manifest))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("invalid split manifest", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_non_ascii_ids_cannot_alias_an_episode_across_splits(self) -> None:
        episode = "gpu-０００００１"
        self.rows[-1] = json.dumps({"episode_id": episode, "power_w": 20})
        self.split["splits"]["test"][-1] = episode
        self.assert_refused("ASCII gpu-NNNNNN episode ID")

    def test_fully_absent_tick_and_explicit_null_count_as_missing(self) -> None:
        self.rows[0] = '{"episode_id":"gpu-000001"}'
        result = self.local()
        self.assertEqual(result.returncode, 0, result.stderr)
        train = json.loads(result.stdout)["splits"]["train"]
        self.assertEqual(train["rows"], 2)
        self.assertEqual(train["missing_or_null"]["power_w"], 2)
        self.assertEqual(train["missing_or_null"]["gpu_temp_c"], 2)

    def test_temperature_zero_is_a_sentinel_but_other_zeros_are_observed(self) -> None:
        self.rows[0] = '{"episode_id":"gpu-000001","gpu_temp_c":0,"power_w":0}'
        result = self.local()
        self.assertEqual(result.returncode, 0, result.stderr)
        train = json.loads(result.stdout)["splits"]["train"]
        self.assertEqual(train["observed"]["power_w"], 1)
        self.assertEqual(train["observed"]["gpu_temp_c"], 0)
        self.assertEqual(train["missing_or_null"]["gpu_temp_c"], 1)
        self.assertEqual(train["dropout_sentinel"]["gpu_temp_c"], 1)

    def test_reformatted_fixture_remains_synthetic_and_hashes_original_bytes(self) -> None:
        records = [json.loads(row) for row in FIXTURE_JSONL.read_text().splitlines()]
        payload = ("\r\n".join(json.dumps(row, sort_keys=True) for row in records) + "\r\n").encode()
        self.data.write_bytes(payload)
        args = ("--jsonl", str(self.data), "--split-manifest", str(FIXTURE_SPLIT))
        result = invoke(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["evidence"], "synthetic-method-fixture")
        self.assertEqual(report["input_sha256"]["jsonl"], "sha256:" + hashlib.sha256(payload).hexdigest())
        self.assertEqual(invoke(*args, "--source-revision", "unverified").returncode, 2)

    def test_external_preflight_does_not_read_bundled_fixture(self) -> None:
        with patch("tools.m1_cohort.FIXTURE_JSONL", self.data.with_name("no-fixture.jsonl")):
            result = self.local()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_refuses_duplicate_row_keys(self) -> None:
        self.rows[0] = '{"episode_id":"gpu-000001","power_w":0,"power_w":1}'
        self.assert_refused("duplicate JSON key 'power_w'")

    def test_refuses_extra_input_columns(self) -> None:
        self.rows[0] = '{"episode_id":"gpu-000001","power_w":0,"workload_id":"game"}'
        self.assert_refused("unsupported projected columns: workload_id")

    def test_refuses_nonfinite_sensor(self) -> None:
        self.rows[0] = '{"episode_id":"gpu-000001","power_w":NaN}'
        self.assert_refused("expected a finite number")
