"""Collector identity and shutdown lifecycle regression tests."""

import json
import os

import pytest

from tools.anticipation import campaign


class _GracefulProcess:
    """Collector stand-in that exits 0 on SIGINT (no fork, no memfd)."""

    def __init__(self):
        self.signals = []

    def poll(self):
        return 0 if self.signals else None

    def send_signal(self, number):
        self.signals.append(number)

    @staticmethod
    def wait(timeout=None):
        return 0

    @staticmethod
    def kill():
        pytest.fail("a graceful collector must not be killed")


def test_collector_signal_interrupt_kills_and_reaps_child(tmp_path):
    class InterruptOnSignal:
        def __init__(self):
            self.killed = False
            self.waits = []

        def poll(self):
            return -9 if self.killed else None

        @staticmethod
        def send_signal(_number):
            raise KeyboardInterrupt

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return -9

    process = InterruptOnSignal()

    with pytest.raises(KeyboardInterrupt):
        campaign._stop_collector(process, tmp_path, {}, None)

    # Previously the interrupt escaped before the guarded wait, orphaning it.
    assert process.killed is True
    assert process.waits == [campaign.SHUTDOWN_TIMEOUT_SECONDS]


def test_capture_cleanup_error_is_raised_even_inside_caller_except_block(tmp_path):
    collector = tmp_path / "collector"
    collector.write_bytes(b"never launched: the plan has no sessions")

    class FailingClose:
        @staticmethod
        def close():
            raise RuntimeError("model unload failed")

    try:
        raise KeyError("unrelated exception the caller is handling")
    except KeyError:
        with pytest.raises(RuntimeError, match="model unload failed"):
            campaign.capture(
                tmp_path / "run",
                collector,
                campaign={"sessions": []},
                stimulus_factory=FailingClose,
            )

    status = json.loads((tmp_path / "run" / campaign.CAPTURE_STATUS_NAME).read_text())
    assert status["status"] == "incomplete"
    assert status["cleanup_error"] == "RuntimeError: model unload failed"


def test_session_audit_error_is_raised_even_inside_caller_except_block(
    tmp_path, monkeypatch
):
    directory = tmp_path / "session-01"
    process = _GracefulProcess()

    def record_stimuli(directory, _process, _stimulus, _session, _state):
        (directory / "session_manifest.json").write_text(
            json.dumps({"ended_at_utc": "done", "parquet_write_failures": 0})
        )

    class BrokenAuditStimulus:
        @staticmethod
        def seed(_seed):
            return None

        @staticmethod
        def session_records():
            raise RuntimeError("audit accessor failed")

    monkeypatch.setattr(
        campaign, "_launch_verified_collector", lambda *_args: process
    )
    monkeypatch.setattr(campaign, "_record_stimuli", record_stimuli)
    session = {"session_id": "session-01", "seed": 1, "path": str(directory)}
    plan = {"collector_sha256": "unused", "actual_audit_key": "actual_bot_tasks"}

    try:
        raise KeyError("unrelated exception the caller is handling")
    except KeyError:
        # Previously the caller's handled exception made the session ignore
        # its own audit failure and be accepted as complete.
        with pytest.raises(RuntimeError, match="audit accessor failed"):
            campaign._capture_session(
                session, BrokenAuditStimulus(), tmp_path / "collector", plan
            )

    assert process.signals, "collector must still receive its graceful SIGINT"
    audit = json.loads((directory / campaign.STIMULUS_AUDIT_NAME).read_text())
    assert audit["audit_error"] == "RuntimeError: audit accessor failed"
    assert audit["status"] == "shutdown_requested"


def test_malformed_manifest_still_records_collector_exit_code(tmp_path):
    (tmp_path / "session_manifest.json").write_text("[]")
    record = {"session_id": "session-01", "status": "shutdown_requested"}

    with pytest.raises(RuntimeError, match="manifest incomplete"):
        campaign._confirm_collector_shutdown(tmp_path, record, 0)

    audit = json.loads((tmp_path / campaign.STIMULUS_AUDIT_NAME).read_text())
    assert audit["collector_exit_code"] == 0
    assert audit["status"] == "collector_stopped"


def test_missing_manifest_still_records_collector_exit_code(tmp_path):
    record = {"session_id": "session-01", "status": "shutdown_requested"}

    with pytest.raises(FileNotFoundError):
        campaign._confirm_collector_shutdown(tmp_path, record, 3)

    audit = json.loads((tmp_path / campaign.STIMULUS_AUDIT_NAME).read_text())
    assert audit["collector_exit_code"] == 3


def test_collector_wait_interrupt_kills_and_reaps_child(tmp_path):
    class InterruptingProcess:
        def __init__(self):
            self.killed = False
            self.wait_calls = 0

        @staticmethod
        def poll():
            return None

        @staticmethod
        def send_signal(_signal):
            return None

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            if timeout is not None and not self.killed:
                raise KeyboardInterrupt
            return -9

    process = InterruptingProcess()

    with pytest.raises(KeyboardInterrupt):
        campaign._stop_collector(process, tmp_path, {}, None)

    assert process.killed is True
    assert process.wait_calls == 2


def test_collector_kill_failure_does_not_wait_indefinitely():
    class ExitedProcess:
        @staticmethod
        def kill():
            raise ProcessLookupError

        @staticmethod
        def poll():
            return 7

        @staticmethod
        def wait(*_args, **_kwargs):
            pytest.fail("kill failure must not be followed by an unbounded wait")

    assert campaign._kill_and_reap_collector(ExitedProcess()) == 7


def test_collector_retries_kill_after_cleanup_interrupt():
    class InterruptingKillProcess:
        def __init__(self):
            self.kill_calls = 0
            self.wait_calls = 0

        def kill(self):
            self.kill_calls += 1
            if self.kill_calls == 1:
                raise KeyboardInterrupt

        @staticmethod
        def poll():
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            assert timeout == campaign.SHUTDOWN_TIMEOUT_SECONDS
            return -9

    process = InterruptingKillProcess()

    assert campaign._kill_and_reap_collector(process) == -9
    assert process.kill_calls == 2
    assert process.wait_calls == 1


def test_verified_collector_descriptor_is_immutable_after_source_replacement(tmp_path):
    collector = tmp_path / "collector"
    original = b"verified collector bytes"
    collector.write_bytes(original)

    descriptor = campaign._open_verified_collector(
        collector, campaign.sha256(collector)
    )
    try:
        collector.write_bytes(b"replacement collector bytes")
        os.lseek(descriptor, 0, os.SEEK_SET)
        assert os.read(descriptor, len(original) + 1) == original
    finally:
        os.close(descriptor)


def test_capture_rejects_collector_changed_after_preflight(tmp_path, monkeypatch):
    collector = tmp_path / "collector"
    collector.write_text("original collector")
    plan = {
        "collector_sha256": campaign.sha256(collector),
        "actual_audit_key": "actual_schedule",
    }
    collector.write_text("changed collector")
    session = {
        "session_id": "session-01",
        "seed": 2026092001,
        "path": str(tmp_path / "session-01"),
        "schedule": [],
    }

    class NoopStimulus:
        @staticmethod
        def seed(_seed):
            return None

    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("changed collector was executed"),
    )
    stimulus = NoopStimulus()

    with pytest.raises(RuntimeError, match="changed after campaign preflight"):
        campaign._capture_session(session, stimulus, collector, plan)
