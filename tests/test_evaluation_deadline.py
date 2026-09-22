"""Python evaluation stages share the trainer's wall-clock deadline."""

import subprocess  # nosec B404
import sys
import time

import pytest

from tools.anticipation import evaluate as evaluator


def test_python_stage_terminates_stalled_worker(tmp_path, monkeypatch):
    original = subprocess.run

    def stalled(command, **kwargs):
        return original([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)

    monkeypatch.setattr(evaluator.subprocess, "run", stalled)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        evaluator._python_stage("baselines", tmp_path / "input", tmp_path, 0.08)
    assert time.monotonic() - started < 0.8


def test_exhausted_budget_skips_comparison(tmp_path, monkeypatch):
    from tools.anticipation.campaign import write_json

    prepared = tmp_path / "prepared.json"
    write_json(prepared, {})
    output = tmp_path / "results"
    output.mkdir()
    status = {"budget_seconds": 0, "status": "complete", "reason": None}
    monkeypatch.setattr(
        evaluator.subprocess, "run", lambda *a, **k: pytest.fail("unbounded comparison")
    )
    evaluator._finalize(
        prepared, output, output / "snn", status, True, time.monotonic()
    )
    assert status["status"] == "incomplete"
    assert status["reason"] == "shared_budget_exhausted"
