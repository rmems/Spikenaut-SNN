"""Path-confinement regression tests for the PR #64 path-traversal fixes.

Each test constructs a sink target as a symlink whose resolved location is
outside the intended trusted base. ``Path(base / name).resolve()`` follows the
symlink and lands outside ``base.resolve()``, so the ``resolve()`` +
``is_relative_to`` confinement guard must reject it with ``ValueError``.

External tools (bubblewrap / julia) are never invoked: the worker's baseline
computation and ``evaluate._python_stage``'s ``subprocess.run`` are monkeypatched
so the test reaches the confinement decision point without heavy setup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.anticipation import evaluate as evaluator
from tools.anticipation import evaluation_worker
from tools.anticipation import task_verification


def _make_escaping_symlink(base: Path, name: str, outside: Path) -> Path:
    """Create ``base/name`` as a symlink whose resolved target is under ``outside``."""
    base.mkdir(parents=True, exist_ok=True)
    outside.mkdir(parents=True, exist_ok=True)
    target = outside / f"escaped-{name}"
    link = base / name
    link.symlink_to(target)
    assert (base / name).parent == base
    assert not (base / name).resolve().is_relative_to(base.resolve())
    return target.resolve()


def test_python_fixture_rejects_verifier_path_escaping_hermes_home(tmp_path):
    hermes_home = tmp_path / "session-01" / "home"
    scratch = tmp_path / "session-01" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    escaped_target = _make_escaping_symlink(hermes_home, "harness-runner.py", outside)

    session = {
        "seed": 2026092001,
        "hermes_home": str(hermes_home),
        "scratch_path": str(scratch),
        "task": {"family": "python-bugfix", "fixture_records": 4},
    }

    with pytest.raises(ValueError):
        task_verification._python_fixture(session, scratch, 4)

    assert not escaped_target.exists(), (
        f"verifier bytes escaped hermes_home: written to {escaped_target}"
    )


def test_evaluation_worker_run_rejects_status_file_escaping_output(
    tmp_path, monkeypatch
):
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({}) + "\n")
    output = tmp_path / "results"
    outside = tmp_path / "outside"
    escaped_target = _make_escaping_symlink(output, "baselines-status.json", outside)

    monkeypatch.setattr(evaluation_worker, "baselines", lambda data, out: None)

    with pytest.raises(ValueError):
        evaluation_worker.run("baselines", prepared, output)

    assert not escaped_target.exists(), (
        f"status file escaped output base: written to {escaped_target}"
    )


def test_python_stage_rejects_log_path_escaping_output(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "results"
    outside = tmp_path / "outside"
    escaped_target = _make_escaping_symlink(output, "baselines.log", outside)

    monkeypatch.setattr(
        evaluator.subprocess, "run", lambda *a, **k: pytest.fail("guard did not run")
    )

    with pytest.raises(ValueError):
        evaluator._python_stage("baselines", prepared, output, 10.0)

    assert not escaped_target.exists(), (
        f"stage log escaped output base: opened at {escaped_target}"
    )


def test_python_stage_refuses_existing_log_symlink_without_truncating_target(
    tmp_path, monkeypatch
):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    sentinel = output / "sentinel.log"
    sentinel.write_text("do not truncate\n")
    (output / "baselines.log").symlink_to(sentinel)

    monkeypatch.setattr(
        evaluator.subprocess, "run", lambda *a, **k: pytest.fail("log was opened")
    )

    with pytest.raises(FileExistsError):
        evaluator._python_stage("baselines", prepared, output, 10.0)

    assert sentinel.read_text() == "do not truncate\n"


def test_exclusive_log_creation_is_relative_to_retained_output_descriptor(
    tmp_path, monkeypatch
):
    output = tmp_path / "results"
    output.mkdir()
    real_open = evaluator.os.open
    observed_dir_fd = None

    def audited_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal observed_dir_fd
        if path == "baselines.log":
            observed_dir_fd = dir_fd
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evaluator.os, "open", audited_open)

    with evaluator._open_exclusive_log(output, "baselines.log") as log:
        log.write("captured\n")

    assert observed_dir_fd is not None
    assert (output / "baselines.log").read_text() == "captured\n"
