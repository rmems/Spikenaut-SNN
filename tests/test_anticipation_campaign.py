"""Failures caught: stimulus overlap, changed split assignment, pooled-session metrics."""

import numpy as np
import pytest
from tools.anticipation.campaign import build_campaign, make_schedule, allocation_bytes
from tools.anticipation.report import metrics, select_baseline


def test_schedule_covers_only_burst_phase_and_is_seeded(tmp_path):
    one = make_schedule(1001)
    assert one == make_schedule(1001)
    assert one != make_schedule(1002)
    assert one[0]["start_s"] == 20
    assert one[-1]["end_s"] == 130
    assert all(a["end_s"] == b["start_s"] for a, b in zip(one, one[1:]))
    assert all(0 < x["end_s"] - x["start_s"] <= 8 for x in one)
    assert allocation_bytes() < 2 * 1024**3
    campaign = build_campaign(tmp_path)
    assert [s["split"] for s in campaign["sessions"]] == ["train"] * 6 + [
        "validation"
    ] * 3 + ["test"] * 3
    assert len({s["seed"] for s in campaign["sessions"]}) == 12


def test_primary_weights_sessions_equally_and_physical_errors():
    actual = np.zeros((4, 4))
    pred = np.array([[0, 0, 2, 4]] * 3 + [[0, 0, 6, 12]])
    out = metrics(actual, pred, ["a", "a", "a", "b"], np.array([1, 1, 2, 4]))
    assert out["primary"] == 2.0
    assert out["mae"] == [0, 0, 4, 8]
    assert out["rmse"] == [0, 0, 4, 8]  # mean of session RMSEs
    assert out["per_session"]["a"]["primary"] == 1
    assert out["per_session"]["b"]["primary"] == 3


def test_baseline_selection_uses_validation_only():
    options = {
        "persistence": {"validation": {"primary": 2}, "test": {"primary": 0}},
        "history_ridge": {"validation": {"primary": 1}, "test": {"primary": 100}},
    }
    assert select_baseline(options) == "history_ridge"


def test_metrics_reject_nonfinite_or_empty():
    with pytest.raises(ValueError):
        metrics(np.zeros((0, 4)), np.zeros((0, 4)), [], np.ones(4))
    with pytest.raises(ValueError):
        metrics(np.zeros((1, 4)), np.full((1, 4), np.nan), ["a"], np.ones(4))


def test_workload_initialization_failure_leaves_incomplete_report(
    tmp_path, monkeypatch
):
    from tools.anticipation import campaign
    import json

    binary = tmp_path / "collector"
    binary.write_text("unused")

    def unavailable():
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(campaign, "Stimulus", unavailable)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        campaign.capture(tmp_path / "run", binary)
    status = json.loads((tmp_path / "run/capture-status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["sessions"] == []
    assert (
        len(json.loads((tmp_path / "run/campaign.json").read_text())["sessions"]) == 12
    )


def test_ridge_prediction_restores_intercept_and_changes_with_sensor():
    from tools.anticipation.report import ridge_fit, ridge_predict

    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.column_stack([3 * x[:, 0] + 7, 2 * x[:, 0] - 4, x[:, 0], -x[:, 0]])
    weights = ridge_fit(x, y, 0.001)
    assert np.allclose(ridge_predict(np.array([[0.0]]), weights), [[7, -4, 0, 0]])
    assert np.allclose(
        ridge_predict(np.array([[1.0]]), weights), [[10, -2, 1, -1]], atol=0.001
    )


def test_clean_early_collector_exit_is_an_incomplete_capture(tmp_path, monkeypatch):
    from tools.anticipation import campaign
    import json
    import sys
    import time

    binary = tmp_path / "collector"
    binary.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """import os,json,time
from pathlib import Path
p=Path(os.environ['SESSION_DIR'])/'session_manifest.json'
p.write_text(json.dumps({'ended_at_utc':'done','parquet_write_failures':0}))
time.sleep(0.1)
"""
    )
    binary.chmod(0o755)

    class NoStimulus:
        class Torch:
            def cuda_max(self):
                return 0

        def seed(self, seed):
            pass

        def run(self, event, origin):
            return event

    stimulus = NoStimulus()
    # Allocator metadata is reached only by the flawed acceptance path.
    from types import SimpleNamespace

    stimulus.torch = SimpleNamespace(
        cuda=SimpleNamespace(
            max_memory_reserved=lambda: 0, max_memory_allocated=lambda: 0
        )
    )
    monkeypatch.setattr(campaign, "Stimulus", lambda: stimulus)
    waits = []

    def shortened_wait(deadline):
        waits.append(deadline)
        if len(waits) == 2:
            time.sleep(0.3)

    monkeypatch.setattr(campaign, "wait_until", shortened_wait)
    with pytest.raises(RuntimeError, match="collector exited"):
        campaign.capture(tmp_path / "run", binary)
    assert (
        json.loads((tmp_path / "run/capture-status.json").read_text())["status"]
        == "incomplete"
    )


def test_prediction_identity_and_duplicates_are_rejected():
    from tools.anticipation.report import checked_predictions

    artifact = {
        "arm": "uniform",
        "seed": 123,
        "predictions": [
            {
                "session_id": "a",
                "frame_index": 1,
                "split": "test",
                "prediction": [0] * 4,
            }
        ],
    }
    with pytest.raises(ValueError, match="identity"):
        checked_predictions(artifact, {"arm": "mixed", "seed": 123})
    artifact["predictions"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        checked_predictions(artifact, {"arm": "uniform", "seed": 123})
