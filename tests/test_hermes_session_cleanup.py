"""Session cleanup must fit the acquisition clock without abandoning model ownership."""

import time

import pytest

from tests.hermes_fixture import FakeRuntime
from tests.ollama_fixture import ollama_server
from tools.anticipation.hermes_campaign import HermesStimulus, OllamaRuntime


def test_session_forwards_absolute_cleanup_deadline(tmp_path):
    received = {}

    class DeadlineRuntime(FakeRuntime):
        def close(self, *, deadline):
            received["deadline"] = deadline
            return {"unloaded": True}

    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=DeadlineRuntime()
    )
    origin = time.monotonic() - 125
    error = stimulus._finish_task(None, {}, origin, {"cleanup_deadline_s": 130}, None)
    assert error is None
    assert received["deadline"] == origin + 130


def test_dripping_cleanup_obeys_session_deadline_and_reconciles(monkeypatch):
    with ollama_server() as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint, durable_cleanup_timeout_seconds=0.8)
        runtime.prepare()
        runtime.select("gemma4:12b", 262144)
        state["ps_drip_interval_after_load"] = 0.02
        original_models = runtime._models
        requests = 0

        def one_stalled_response(deadline=None):
            nonlocal requests
            requests += 1
            if requests > 1:
                state["ps_drip_interval_after_load"] = 0
            return original_models(deadline)

        monkeypatch.setattr(runtime, "_models", one_stalled_response)
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                runtime.close(deadline=started + 0.08)
            assert time.monotonic() - started < 0.3
        finally:
            state["ps_drip_interval_after_load"] = 0
            if runtime._cleanup_thread is not None:
                runtime._cleanup_thread.join(2)
            else:
                runtime.close()
        assert runtime._owned_model is None
        assert state["loaded"] is False


def test_capture_finalization_does_not_restart_deferred_cleanup(tmp_path, monkeypatch):
    import json

    from tests.test_hermes_lifecycle import _graceful_fake_collector
    from tools.anticipation import campaign

    with ollama_server() as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint, durable_cleanup_timeout_seconds=1.5)
        root = tmp_path / "run"

        class ExpiringStimulus(HermesStimulus):
            def prepare(self):
                return runtime.prepare()

            def seed(self, seed):
                return runtime.select("gemma4:12b", 262144)

            def prepare_session(self, session):
                pass

            def run(self, event, origin):
                state["ps_drip_interval_after_load"] = 0.02
                runtime.close(deadline=time.monotonic() + 0.08)

        stimulus = ExpiringStimulus(
            tmp_path, hermes_executable=tmp_path / "hermes", runtime=runtime
        )
        plan = _cleanup_capture_plan(root)
        monkeypatch.setattr(campaign, "wait_until", lambda deadline: None)
        collector = _graceful_fake_collector(tmp_path)
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                campaign.capture(
                    root,
                    collector,
                    campaign=plan,
                    stimulus_factory=lambda: stimulus,
                )
            assert time.monotonic() - started < 1.0
            status = json.loads((root / "capture-status.json").read_text())
            assert status["status"] == "incomplete"
            assert "deferred" in status["cleanup_error"]
            manifest = json.loads(
                (root / "raw/session-01/session_manifest.json").read_text()
            )
            assert manifest["ended_at_utc"] == "done"
        finally:
            state["ps_drip_interval_after_load"] = 0
            if runtime._cleanup_thread is not None:
                runtime._cleanup_thread.join(3)
            runtime.close()


def _cleanup_capture_plan(root):
    return {
        "schema_version": "anticipation-campaign-v1",
        "actual_audit_key": "actual_bot_tasks",
        "duration_s": 150,
        "poll_interval_ms": 100,
        "sessions": [
            {
                "session_id": "session-01",
                "split": "train",
                "seed": 1,
                "path": str(root / "raw/session-01"),
                "task": {},
            }
        ],
    }
