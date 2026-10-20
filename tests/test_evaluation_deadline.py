"""Python evaluation stages share the trainer's wall-clock deadline."""

import json
import math
import os
import subprocess  # nosec B404
import sys
import time

import pytest

from tools.anticipation import evaluate as evaluator
from tools.anticipation import evaluation_worker


def test_worker_module_is_importable_from_output_working_directory(tmp_path):
    env = evaluator._worker_environment({"PYTHONPATH": "previous-entry"})
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(evaluator._REPOSITORY_ROOT),
        "previous-entry",
    ]

    # Same argv and cwd as the real stage launch. Without the repository root
    # on PYTHONPATH this failed with "No module named 'tools'".
    completed = subprocess.run(  # nosec B603
        [sys.executable, "-m", evaluator._EVALUATION_WORKER, "--help"],
        cwd=tmp_path,
        env=evaluator._worker_environment(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    not hasattr(os, "O_DIRECTORY"), reason="stage logs need POSIX O_DIRECTORY"
)
def test_baselines_stage_runs_real_worker_from_output_directory(tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(_tiny_prepared_dataset()))
    output = tmp_path / "results"
    output.mkdir()

    assert evaluator._python_stage("baselines", prepared, output, 60.0) is True

    results = json.loads((output / "baselines.json").read_text())
    assert set(results) == {"persistence", "current_ridge", "history_ridge"}
    assert json.loads((output / "baselines-status.json").read_text()) == {
        "complete": True
    }


def test_worker_rejects_unknown_stage_without_writing_status(tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}\n")

    with pytest.raises(ValueError, match="unsupported evaluation stage"):
        evaluation_worker.run("bogus", prepared, tmp_path)

    assert list(tmp_path.iterdir()) == [prepared]


def _tiny_prepared_dataset():
    sessions = []
    for number, split in enumerate(("train", "train", "validation", "test")):
        frames = [
            {
                "x": [
                    math.sin((i + number) / 3),
                    math.cos((i + number) / 5),
                    math.sin((i + number) / 7),
                    math.cos((i + number) / 2),
                    0.5,
                ]
            }
            for i in range(40)
        ]
        examples = [
            {
                "frame_index": i,
                "history_indices": [i - 1, i - 2, i - 3, i - 4],
                "y": [frames[i]["x"][j] * 0.5 for j in range(4)],
            }
            for i in range(4, 40)
        ]
        sessions.append(
            {
                "session_id": f"session-{number}",
                "split": split,
                "frames": frames,
                "examples": examples,
            }
        )
    return {
        "schema_version": "anticipation-prepared-v1",
        "feature_map_id": "synthetic-test",
        "feature_map": ["a", "b", "c", "d", "e"],
        "normalization": {
            "x_mean": [0.0] * 5,
            "x_std": [1.0] * 5,
            "y_mean": [0.0] * 4,
            "y_std": [1.0] * 4,
        },
        "sessions": sessions,
    }


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
