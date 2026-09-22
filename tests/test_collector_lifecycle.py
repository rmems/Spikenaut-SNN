"""Collector identity and shutdown lifecycle regression tests."""

import os

import pytest

from tools.anticipation import campaign


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
