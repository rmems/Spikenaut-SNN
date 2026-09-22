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


def test_zero_evaluation_budget_marks_all_runs_unfinished(tmp_path):
    from tools.anticipation.evaluate import evaluate
    from tools.anticipation.campaign import write_json
    import json

    # Deadline is already exhausted: no Julia executable may be launched.
    prepared = tmp_path / "prepared.json"
    write_json(prepared, {"schema_version": "anticipation-prepared-v1"})
    result = evaluate(
        prepared, tmp_path / "results", julia="must-not-run", budget_seconds=0
    )
    assert result["status"] == "incomplete"
    assert result["reason"] == "budget_exhausted_before_baselines"
    summary = json.loads((tmp_path / "results/snn/summary.json").read_text())
    assert len(summary["runs"]) == 6
    assert all(r["status"] == "unfinished" for r in summary["runs"])


def test_evaluator_always_records_missing_input_failure(tmp_path):
    from tools.anticipation.evaluate import evaluate
    import json

    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path / "missing.json", tmp_path / "results", julia="must-not-run")
    report = json.loads((tmp_path / "results/budget-report.json").read_text())
    assert report["status"] == "incomplete"
    assert "FileNotFoundError" in report["reason"]


def test_evaluator_refuses_partially_written_attempt_directory(tmp_path):
    from tools.anticipation.evaluate import evaluate
    from tools.anticipation.campaign import write_json

    prepared = tmp_path / "prepared.json"
    write_json(prepared, {"schema_version": "anticipation-prepared-v1"})
    output = tmp_path / "results"
    output.mkdir()
    (output / "baselines.json").write_text("partial attempt")

    with pytest.raises(FileExistsError, match="already attempted"):
        evaluate(prepared, output, julia="must-not-run")

    assert (output / "baselines.json").read_text() == "partial attempt"


def test_evaluator_reserves_child_startup_and_finalization_budget(
    tmp_path, monkeypatch
):
    from tools.anticipation import evaluate as evaluator
    from tools.anticipation.campaign import write_json

    prepared = tmp_path / "prepared.json"
    write_json(prepared, {"schema_version": "anticipation-prepared-v1"})
    clock = iter((100.0, 100.0, 100.0, 110.0, 110.0, 111.0, 111.0, 112.0))
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(evaluator, "_python_stage", lambda *args: True)

    status = evaluator.evaluate(
        prepared, tmp_path / "results", julia="/bin/true", budget_seconds=200
    )

    assert status["trainer_budget_seconds"] == 161.5
    assert status["trainer_timeout_seconds"] == 190.0
    assert status["trainer_command"][1].startswith("--project=")
    assert float(status["trainer_command"][-1]) == 161.5


@pytest.mark.parametrize(
    "declared, expected",
    [
        (
            ["temperature", "power", "temperature_future", "power_future"],
            ["temperature", "power", "temperature_future", "power_future"],
        ),
        (
            None,
            [
                "temperature_change_1s_c",
                "power_change_1s_w",
                "temperature_change_5s_c",
                "power_change_5s_w",
            ],
        ),
    ],
)
def test_comparison_uses_prepared_target_names_with_fallback(
    tmp_path, declared, expected
):
    from tools.anticipation.campaign import write_json
    from tools.anticipation.report import comparison

    prepared_data = {
        "schema_version": "anticipation-prepared-v1",
        "normalization": {"y_std": [1, 1, 1, 1]},
    }
    if declared is not None:
        prepared_data["target_names"] = declared
    prepared = tmp_path / "prepared.json"
    output = tmp_path / "results"
    write_json(prepared, prepared_data)
    baseline = {
        "persistence": {
            "validation": {"primary": 1.0},
            "test": {
                "primary": 1.0,
                "mae": [1, 1, 1, 1],
                "rmse": [1, 1, 1, 1],
                "per_session": {},
            },
        }
    }

    result = comparison(prepared, output, baseline_results=baseline)

    assert result["target_names"] == expected


def test_shutdown_timeout_preserves_executed_stimulus_audit(tmp_path, monkeypatch):
    from tools.anticipation import campaign
    from types import SimpleNamespace
    import sys
    import json

    binary = tmp_path / "collector"
    binary.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """import os,signal,json,time
from pathlib import Path
signal.signal(signal.SIGINT,signal.SIG_IGN)
(Path(os.environ['SESSION_DIR'])/'session_manifest.json').write_text(json.dumps({'ended_at_utc':None,'parquet_write_failures':0}))
time.sleep(30)
"""
    )
    binary.chmod(0o755)
    stimulus = SimpleNamespace(
        seed=lambda seed: None,
        run=lambda event, origin: dict(event),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                max_memory_reserved=lambda: 0, max_memory_allocated=lambda: 0
            )
        ),
    )
    monkeypatch.setattr(campaign, "Stimulus", lambda: stimulus)
    monkeypatch.setattr(campaign, "wait_until", lambda deadline: None)
    monkeypatch.setattr(campaign, "SHUTDOWN_TIMEOUT_SECONDS", 0.1, raising=False)
    with pytest.raises(RuntimeError, match="graceful shutdown"):
        campaign.capture(tmp_path / "run", binary)
    audit = json.loads(
        (tmp_path / "run/raw/session-01/stimulus-audit.json").read_text()
    )
    assert audit["status"] == "shutdown_timeout"
    assert len(audit["actual_schedule"]) > 0


def test_audit_write_failure_still_stops_collector(tmp_path, monkeypatch):
    from tools.anticipation import campaign
    from types import SimpleNamespace
    import sys

    binary = tmp_path / "collector"
    binary.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """import os,json,time
from pathlib import Path
(Path(os.environ['SESSION_DIR'])/'session_manifest.json').write_text('{}')
time.sleep(30)
"""
    )
    binary.chmod(0o755)
    stimulus = SimpleNamespace(
        seed=lambda seed: None,
        run=lambda event, origin: dict(event),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                max_memory_reserved=lambda: 0, max_memory_allocated=lambda: 0
            )
        ),
    )
    monkeypatch.setattr(campaign, "Stimulus", lambda: stimulus)
    monkeypatch.setattr(campaign, "wait_until", lambda deadline: None)
    original_write = campaign.write_json

    def broken_audit(path, value):
        if path.name == "stimulus-audit.json":
            raise OSError("audit disk error")
        original_write(path, value)

    monkeypatch.setattr(campaign, "write_json", broken_audit)
    original_popen = campaign.subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(campaign.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="audit disk error"):
        campaign.capture(tmp_path / "run", binary)
    stopped = children[0].poll() is not None
    if not stopped:  # Clean up a regression's real process before asserting.
        children[0].kill()
        children[0].wait()
    assert stopped, "audit failure must not orphan the collector"


def test_session_audit_does_not_swallow_keyboard_interrupt(tmp_path, monkeypatch):
    from tools.anticipation import campaign

    def interrupt_diagnostics(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign, "_session_diagnostics", interrupt_diagnostics)

    with pytest.raises(KeyboardInterrupt):
        campaign._publish_session_audit(object(), tmp_path, {}, "actual_schedule", None)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_json_writer_preserves_staging_link_target(tmp_path, link_kind):
    from tools.anticipation.campaign import write_json

    destination = tmp_path / "report.json"
    sentinel = tmp_path / "source.json"
    sentinel.write_text("source sentinel")
    staging = tmp_path / "report.json.tmp"
    if link_kind == "symlink":
        staging.symlink_to(sentinel)
    else:
        staging.hardlink_to(sentinel)
    write_json(destination, {"complete": True})
    assert sentinel.read_text() == "source sentinel"
    assert destination.read_text() == '{\n  "complete": true\n}\n'
