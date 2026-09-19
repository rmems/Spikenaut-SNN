"""Frozen model-bank replay tests (Linear RM-1692 / GH #60).

Run::

    pytest -q -s
    python3 -m unittest tests.test_replay_frozen -v
    python3 tools/replay_frozen.py

Standard library only; pytest is an optional runner.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.model_bank import (
    BankAttestationError,
    load_model_bank,
)
from tools.q88_core import ParseError
from tools.replay_core import (
    MISSING_POLICY_REJECT,
    REPLAY_FIXTURE_DIR,
    REPO_ROOT,
    ReplayConfig,
    load_replay_inputs,
    load_split_manifest,
    manifest_json,
    build_manifest,
    replay,
    trace_jsonl,
)
from tools.replay_frozen import main as cli_main

FIXTURE_JSONL = REPLAY_FIXTURE_DIR / "telemetry.jsonl"
SHIPPED_MANIFEST = REPO_ROOT / "dataset" / "merged_v2" / "model_bank.json"
TAMPERED_MANIFEST = (
    REPO_ROOT / "tools" / "fixtures" / "model_bank" / "tampered-checkpoint"
    / "model_bank.json"
)
SHIPPED_CHECKPOINT = REPO_ROOT / "dataset" / "merged_v2" / "snn_model.json"

CONFIG = ReplayConfig(split="all", k=4, i_drive=0.0, missing_policy="encode-zero")


def _fixture_inputs(**kwargs):
    return load_replay_inputs(
        kwargs.get("manifest", SHIPPED_MANIFEST),
        kwargs.get("model_id"),
        kwargs.get("jsonl", FIXTURE_JSONL),
        kwargs.get("split", "all"),
        kwargs.get("split_manifest"),
    )


def _fixture_replay(**kwargs):
    bank, entry, samples = _fixture_inputs(**kwargs)
    return bank, entry, samples, replay(entry, samples, CONFIG)


class TestFixtureReplay(unittest.TestCase):
    """The committed method fixture drives the full composition."""

    def test_fixture_runs_and_records_sessions(self):
        _, entry, _, result = _fixture_replay()
        self.assertEqual(entry.id, "merged_v2")
        self.assertEqual(result.sessions, ["gpu-000001", "gpu-000002"])
        self.assertEqual(
            result.steps_per_session, {"gpu-000001": 5, "gpu-000002": 5}
        )
        self.assertEqual(len(result.trace_rows), 10)
        self.assertGreater(result.spikes_fired, 0)

    def test_trace_row_schema(self):
        _, _, _, result = _fixture_replay()
        row = result.trace_rows[1]
        for key in (
            "step",
            "session",
            "source_line",
            "missing",
            "stim",
            "spikes",
            "scores",
            "decision",
        ):
            self.assertIn(key, row)
        self.assertEqual(len(row["stim"]), 16)
        self.assertEqual(len(row["scores"]), 3)
        self.assertEqual(
            row["decision"]["winning_action"] in {"comfort", "temp", "power"},
            True,
        )
        # Unused axons stay exactly zero.
        self.assertTrue(all(v == 0.0 for v in row["stim"][5:]))

    def test_deterministic_across_runs(self):
        _, _, _, first = _fixture_replay()
        _, _, _, second = _fixture_replay()
        self.assertEqual(
            trace_jsonl(first.trace_rows), trace_jsonl(second.trace_rows)
        )
        self.assertEqual(first.trace_rows, second.trace_rows)

    def test_missing_sensors_named_per_step(self):
        _, _, _, result = _fixture_replay()
        # Fixture line 4 lacks sm_clock_mhz; line 5 has mem_clock_mhz null.
        step3 = result.trace_rows[3]
        self.assertEqual(step3["missing"], ["sm_clock_mhz"])
        step4 = result.trace_rows[4]
        self.assertEqual(step4["missing"], ["mem_clock_mhz"])
        # Missing stays distinguishable from an observed zero.
        self.assertEqual(step3["stim"][3], 0.0)
        self.assertEqual(result.missing_counts["sm_clock_mhz"], 1)
        self.assertEqual(result.missing_counts["mem_clock_mhz"], 1)

    def test_reject_missing_policy(self):
        _, entry, samples = _fixture_inputs()
        with self.assertRaises(ParseError) as ctx:
            replay(
                entry,
                samples,
                ReplayConfig(
                    split="all", k=4, i_drive=0.0,
                    missing_policy=MISSING_POLICY_REJECT,
                ),
            )
        self.assertIn("sm_clock_mhz", str(ctx.exception))

    def test_checkpoint_bytes_unchanged(self):
        before = hashlib.sha256(SHIPPED_CHECKPOINT.read_bytes()).hexdigest()
        _fixture_replay()
        after = hashlib.sha256(SHIPPED_CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(before, after)


class TestAttestation(unittest.TestCase):
    """The replay consumes only digest-verified checkpoint bytes."""

    def test_tampered_checkpoint_refused(self):
        with self.assertRaises(BankAttestationError):
            load_model_bank(TAMPERED_MANIFEST)


class TestSessionReset(unittest.TestCase):
    """Membrane state resets at every episode boundary."""

    def test_identical_sessions_replay_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two_sessions.jsonl"
            row = {
                "episode_id": "gpu-000001",
                "mem_util_pct": 40.0,
                "power_w": 180.0,
                "gpu_temp_c": 58.0,
                "sm_clock_mhz": 1900.0,
                "mem_clock_mhz": 8000.0,
            }
            lines = []
            for session in ("gpu-000003", "gpu-000004"):
                for _ in range(4):
                    r = dict(row)
                    r["episode_id"] = session
                    lines.append(json.dumps(r))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _, _, _, result = _fixture_replay(jsonl=path)
        self.assertEqual(result.sessions, ["gpu-000003", "gpu-000004"])
        # With a reset at the boundary, session 2 reproduces session 1
        # step-for-step; without it the membrane would carry over and the
        # spike trains would differ.
        half = len(result.trace_rows) // 2
        for a, b in zip(
            result.trace_rows[:half], result.trace_rows[half:]
        ):
            self.assertEqual(a["spikes"], b["spikes"])
            self.assertEqual(a["scores"], b["scores"])

    def test_interleaved_sessions_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interleaved.jsonl"
            rows = []
            for session in ("gpu-000005", "gpu-000006", "gpu-000005"):
                rows.append(
                    json.dumps(
                        {
                            "episode_id": session,
                            "mem_util_pct": 10.0,
                            "power_w": 100.0,
                            "gpu_temp_c": 40.0,
                            "sm_clock_mhz": 1400.0,
                            "mem_clock_mhz": 5000.0,
                        }
                    )
                )
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(ParseError):
                _fixture_inputs(jsonl=path)


class TestMalformedInputs(unittest.TestCase):
    def test_malformed_jsonl_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("{not json\n", encoding="utf-8")
            with self.assertRaises(ParseError):
                _fixture_inputs(jsonl=path)

    def test_non_finite_telemetry_refused(self):
        # JSON cannot carry NaN through the strict loader, but 1e400 -> inf
        # does parse as a float literal and must still be refused.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inf.jsonl"
            path.write_text(
                '{"episode_id": "gpu-000007", "mem_util_pct": 1e400,'
                ' "power_w": 100.0, "gpu_temp_c": 40.0,'
                ' "sm_clock_mhz": 1400.0, "mem_clock_mhz": 5000.0}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ParseError) as ctx:
                _fixture_inputs(jsonl=path)
            self.assertIn("mem_util_pct", str(ctx.exception))

    def test_non_v3_row_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.jsonl"
            path.write_text(
                '{"episode_id": "gpu-000008", "unrelated": 1}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ParseError):
                _fixture_inputs(jsonl=path)


class TestSplitManifest(unittest.TestCase):
    def _manifest(self, tmp: Path, splits: dict) -> Path:
        path = tmp / "splits.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "spikenaut.split-manifest.v1",
                    "splits": splits,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(
                Path(tmp),
                {
                    "train": ["gpu-000001", "gpu-000002"],
                    "test": ["gpu-000002"],
                },
            )
            with self.assertRaises(ParseError) as ctx:
                load_split_manifest(path)
            self.assertIn("overlap", str(ctx.exception))

    def test_manifest_selects_named_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(
                Path(tmp),
                {
                    "train": ["gpu-000001"],
                    "test": ["gpu-000002"],
                },
            )
            _, _, samples = _fixture_inputs(
                split="test", split_manifest=path
            )
            self.assertEqual(
                [s.episode_id for s in samples], ["gpu-000002"] * 5
            )

    def test_unknown_split_name_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(
                Path(tmp), {"holdout": ["gpu-000001"]}
            )
            with self.assertRaises(ParseError):
                load_split_manifest(path)

    def test_undeclared_split_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(
                Path(tmp), {"train": ["gpu-000001"]}
            )
            with self.assertRaises(ParseError):
                _fixture_inputs(split="test", split_manifest=path)


class TestCli(unittest.TestCase):
    """End-to-end through the argument surface."""

    def test_cli_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli_main(["--out-dir", tmp])
            self.assertEqual(code, 0)
            trace = (Path(tmp) / "trace.jsonl").read_text()
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        self.assertEqual(len(trace.strip().split("\n")), 10)
        self.assertEqual(
            manifest["schema_version"], "spikenaut.replay-manifest.v1"
        )
        self.assertEqual(
            manifest["model"]["output_contract_id"],
            "spikenaut.output-contract.supervisor-v3.rm-1150",
        )
        self.assertEqual(
            manifest["output"]["vocabulary"], ["comfort", "temp", "power"]
        )
        self.assertIn("trace_sha256", manifest)

    def test_cli_deterministic_artifacts(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(cli_main(["--out-dir", a]), 0)
            self.assertEqual(cli_main(["--out-dir", b]), 0)
            for name in ("trace.jsonl", "manifest.json"):
                self.assertEqual(
                    (Path(a) / name).read_bytes(),
                    (Path(b) / name).read_bytes(),
                )

    def test_cli_tampered_bank_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli_main(
                [
                    "--bank-manifest",
                    str(TAMPERED_MANIFEST),
                    "--out-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse((Path(tmp) / "trace.jsonl").exists())

    def test_cli_bad_out_dir_does_not_hide_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli_main(
                [
                    "--bank-manifest", str(TAMPERED_MANIFEST),
                    "--out-dir", str(Path(tmp) / "nested" / "deep"),
                ]
            )
            self.assertEqual(code, 2)


class TestManifestContents(unittest.TestCase):
    def test_manifest_fields(self):
        bank, entry, samples, result = _fixture_replay()
        manifest = build_manifest(
            entry=entry,
            bank=bank,
            jsonl=FIXTURE_JSONL,
            split_manifest_path=None,
            config=CONFIG,
            result=result,
            trace_bytes=trace_jsonl(result.trace_rows),
        )
        model = manifest["model"]
        self.assertEqual(model["id"], "merged_v2")
        self.assertTrue(model["checkpoint_digest"].startswith("sha256:"))
        self.assertEqual(
            model["feature_map_id"], "spikenaut.feature-map.live-exp-025.v1"
        )
        self.assertEqual(
            manifest["encoder"]["frozen_minmax_lineage"], "74acdd0f"
        )
        self.assertEqual(
            manifest["input"]["sessions"], ["gpu-000001", "gpu-000002"]
        )
        self.assertTrue(
            manifest["trace_sha256"].startswith("sha256:")
        )
        # Round-trips through the canonical serializer unchanged.
        self.assertEqual(
            manifest_json(manifest),
            manifest_json(json.loads(manifest_json(manifest))),
        )


if __name__ == "__main__":
    unittest.main()
