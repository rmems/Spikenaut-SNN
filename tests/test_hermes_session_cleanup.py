"""Session cleanup must fit the acquisition clock without abandoning model ownership."""

import time

import pytest

from tests.hermes_fixture import FakeRuntime
from tests.ollama_fixture import ollama_server
from tools.anticipation.hermes_campaign import HermesStimulus, OllamaRuntime


class _TrackingRuntime(FakeRuntime):
    closed = False

    def close(self, *, deadline):
        self.closed = True
        return super().close(deadline=deadline)


class _WaitFailure:
    pid = 12345

    @staticmethod
    def poll():
        return None

    @staticmethod
    def wait(_timeout=None):
        raise OSError("wait failed")


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


def test_session_cleanup_does_not_swallow_keyboard_interrupt(tmp_path):
    class InterruptingRuntime(FakeRuntime):
        def close(self, *, deadline):
            raise KeyboardInterrupt

    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=InterruptingRuntime(),
    )
    origin = time.monotonic()
    record = {}
    task_config = {"cleanup_deadline_s": 130}

    with pytest.raises(KeyboardInterrupt):
        stimulus._finish_task(None, record, origin, task_config, None)

    assert stimulus.session_records() == [record]
    assert record["status"] == "invalid"
    assert record["model_cleanup_error"].startswith("KeyboardInterrupt:")


def test_process_wait_error_still_closes_model_and_records_task(tmp_path, monkeypatch):
    from tools.anticipation import hermes_campaign

    runtime = _TrackingRuntime()
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=runtime,
    )
    monkeypatch.setattr(hermes_campaign, "_signal_process_group", lambda *_: None)
    origin = time.monotonic()
    process = _WaitFailure()
    record = {}
    task_config = {"cleanup_deadline_s": 130}

    with pytest.raises(OSError, match="wait failed"):
        stimulus._finish_task(process, record, origin, task_config, None)

    assert runtime.closed is True
    assert stimulus.session_records() == [record]
    assert record["process_cleanup_error"] == "OSError: wait failed"
    assert record["model_cleanup"]["unloaded"] is True


def test_pending_task_error_wins_over_process_wait_error(tmp_path, monkeypatch):
    from tools.anticipation import hermes_campaign

    runtime = _TrackingRuntime()
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=runtime,
    )
    monkeypatch.setattr(hermes_campaign, "_signal_process_group", lambda *_: None)
    origin = time.monotonic()
    process = _WaitFailure()
    record = {}
    task_config = {"cleanup_deadline_s": 130}
    task_error = ValueError("task failed")

    result = stimulus._finish_task(
        process,
        record,
        origin,
        task_config,
        task_error,
    )

    assert result is task_error
    assert runtime.closed is True
    assert stimulus.session_records() == [record]
    assert record["process_cleanup_error"] == "OSError: wait failed"


def test_process_wait_uses_remaining_cleanup_deadline(tmp_path, monkeypatch):
    from tools.anticipation import hermes_campaign

    observed = {}

    class TimeoutProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout):
            observed["timeout"] = timeout
            raise hermes_campaign.subprocess.TimeoutExpired("hermes", timeout)

    runtime = _TrackingRuntime()
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=runtime,
    )
    monkeypatch.setattr(hermes_campaign, "_signal_process_group", lambda *_: None)
    origin = time.monotonic() - 129
    process = TimeoutProcess()
    record = {}
    task_config = {"cleanup_deadline_s": 130}

    with pytest.raises(hermes_campaign.subprocess.TimeoutExpired):
        stimulus._finish_task(process, record, origin, task_config, None)

    assert 0 <= observed["timeout"] <= 1
    assert runtime.closed is True
    assert stimulus.session_records() == [record]
    assert record["process_cleanup_error"].startswith("TimeoutExpired:")


def test_cleanup_interrupt_wins_over_pending_task_error(tmp_path):
    class InterruptingRuntime(FakeRuntime):
        def close(self, *, deadline):
            raise KeyboardInterrupt

    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=InterruptingRuntime(),
    )
    origin = time.monotonic()
    record = {}
    task_config = {"cleanup_deadline_s": 130}
    task_error = ValueError("task failed")

    with pytest.raises(KeyboardInterrupt):
        stimulus._finish_task(None, record, origin, task_config, task_error)

    assert stimulus.session_records() == [record]
    assert record["model_cleanup_error"].startswith("KeyboardInterrupt:")


def test_process_cleanup_interrupt_retries_and_preserves_exception(
    tmp_path, monkeypatch
):
    from tools.anticipation import hermes_campaign

    cleanup_interrupt = KeyboardInterrupt()
    signal_calls = 0

    class InterruptingProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout):
            assert 0 <= timeout <= 130
            return -9

    def interrupt_once(*_args):
        nonlocal signal_calls
        signal_calls += 1
        if signal_calls == 1:
            raise cleanup_interrupt

    runtime = _TrackingRuntime()
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=runtime,
    )
    monkeypatch.setattr(hermes_campaign, "_signal_process_group", interrupt_once)
    origin = time.monotonic()
    record = {}
    process = InterruptingProcess()
    task_config = {"cleanup_deadline_s": 130}

    with pytest.raises(KeyboardInterrupt) as caught:
        stimulus._finish_task(process, record, origin, task_config, None)

    assert caught.value is cleanup_interrupt
    assert signal_calls == 2
    assert runtime.closed is True
    assert record["process_cleanup_error"].startswith("KeyboardInterrupt:")
    assert stimulus.session_records() == [record]


def test_expired_process_cleanup_retry_gets_dedicated_reap_interval(
    tmp_path, monkeypatch
):
    from tools.anticipation import hermes_campaign
    from tools.anticipation import hermes_cleanup

    waits = []
    errors = []

    class ExpiredProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout):
            waits.append(timeout)
            error = hermes_campaign.subprocess.TimeoutExpired("hermes", timeout)
            errors.append(error)
            raise error

    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=_TrackingRuntime(),
    )
    monkeypatch.setattr(hermes_campaign, "_signal_process_group", lambda *_: None)
    expired_origin = time.monotonic() - 131
    process = ExpiredProcess()
    record = {}
    task_config = {"cleanup_deadline_s": 130}

    with pytest.raises(hermes_campaign.subprocess.TimeoutExpired) as caught:
        stimulus._finish_task(process, record, expired_origin, task_config, None)

    assert caught.value is errors[0]
    assert waits == [0, hermes_cleanup.TASK_REAP_TIMEOUT_SECONDS]


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
