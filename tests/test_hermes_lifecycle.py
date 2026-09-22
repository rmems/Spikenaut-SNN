"""Hermes lifecycle regression tests."""

from __future__ import annotations


import json


import os


from pathlib import Path


import signal


import subprocess  # nosec B404


import time


import pytest


from tests.hermes_fixture import (
    FakeRuntime,
    _write_fake_hermes,
    _wait_for_fixture_ready_before_timeout,
)


def test_graceful_sigterm_is_valid_timeboxed_workload_with_partial_usage(
    tmp_path, monkeypatch
):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    ready = tmp_path / "hermes-ready"
    fake.parent.mkdir()
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,signal,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        " print(json.dumps({'type':'result','session_id':'s','exit_code':130,'tokens':{'input':0,'output':0},'error':'Interrupted'}),flush=True)\n"
        " time.sleep(2.2)\n"
        " raise SystemExit(130)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "print(json.dumps({'type':'system','subtype':'init','session_id':'s','model':'gemma4:12b'}),flush=True)\n"
        "print(json.dumps({'type':'tool_use','name':'read_file','tool_call_id':'c','input':{'path':'input.csv'}}),flush=True)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(.01)\n"
    )
    fake.chmod(0o755)
    _wait_for_fixture_ready_before_timeout(monkeypatch, ready)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=fake,
        hard_timeout_seconds=0.05,
        runtime=FakeRuntime(),
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    record = stimulus.run(session["task"], time.monotonic() - 20)
    assert record["execution_status"] == "timeboxed"
    assert record["workload_status"] == "valid"
    assert record["task_outcome"] == "incomplete"
    assert record["parent_stop_reason"] == "0.05s_agent_timebox"
    assert record["configured_timebox_seconds"] == 0.05
    assert record["effective_timebox_seconds"] == 0.05
    assert record["framework_interruption"] == "Interrupted"
    assert record["usage_quality"] == "partial_after_parent_sigterm"
    assert Path(record["stdout_jsonl_path"]).exists()
    assert Path(record["stderr_path"]).exists()


def test_session_deadline_records_shorter_effective_timebox(tmp_path, monkeypatch):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    ready = tmp_path / "hermes-ready"
    fake.parent.mkdir()
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,signal,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        " print(json.dumps({'type':'result','session_id':'s','exit_code':130,'tokens':{'input':0,'output':0},'error':'Interrupted'}),flush=True)\n"
        " raise SystemExit(130)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "print(json.dumps({'type':'system','subtype':'init','session_id':'s','model':'gemma4:12b'}),flush=True)\n"
        "print(json.dumps({'type':'tool_use','name':'read_file','tool_call_id':'c','input':{'path':'input.csv'}}),flush=True)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(.01)\n"
    )
    fake.chmod(0o755)
    _wait_for_fixture_ready_before_timeout(monkeypatch, ready)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=fake,
        hard_timeout_seconds=1.0,
        runtime=FakeRuntime(),
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    origin = time.monotonic() - 20
    session["task"]["hard_deadline_s"] = 20.1

    record = stimulus.run(session["task"], origin)

    assert 0 < record["effective_timebox_seconds"] < 1.0
    assert record["configured_timebox_seconds"] == 1.0
    assert record["parent_stop_reason"] == (
        f"{record['effective_timebox_seconds']:g}s_agent_timebox"
    )


def test_timebox_requires_interrupted_terminal_result(tmp_path, monkeypatch):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    ready = tmp_path / "hermes-ready"
    fake.parent.mkdir()
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,signal,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        " print(json.dumps({'type':'result','session_id':'s','exit_code':0,'tokens':{'input':2,'output':1}}),flush=True)\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "print(json.dumps({'type':'system','subtype':'init','session_id':'s','model':'gemma4:12b'}),flush=True)\n"
        "print(json.dumps({'type':'tool_use','name':'read_file','tool_call_id':'c','input':{'path':'input.csv'}}),flush=True)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(.01)\n"
    )
    fake.chmod(0o755)
    _wait_for_fixture_ready_before_timeout(monkeypatch, ready)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=fake,
        hard_timeout_seconds=0.05,
        runtime=FakeRuntime(),
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="without an Interrupted terminal result"):
        stimulus.run(session["task"], origin)


def test_sigterm_without_terminal_result_is_not_valid_timebox(tmp_path, monkeypatch):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin/hermes"
    ready = tmp_path / "hermes-ready"
    fake.parent.mkdir()
    _write_fake_hermes(
        fake,
        [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "model": "gemma4:12b",
            },
            {
                "type": "tool_use",
                "name": "read_file",
                "tool_call_id": "c",
                "input": {"path": "input.csv"},
            },
        ],
        sleep_seconds=30,
        ready_path=ready,
    )
    _wait_for_fixture_ready_before_timeout(monkeypatch, ready)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=fake,
        hard_timeout_seconds=0.05,
        runtime=FakeRuntime(),
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="terminal result"):
        stimulus.run(session["task"], origin)


def test_timebox_requiring_sigkill_is_invalid_even_with_prior_result(
    tmp_path, monkeypatch
):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin/hermes"
    ready = tmp_path / "hermes-ready"
    fake.parent.mkdir()
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "for event in ["
        "{'type':'system','subtype':'init','session_id':'s','model':'gemma4:12b'},"
        "{'type':'tool_use','name':'read_file','tool_call_id':'c','input':{'path':'input.csv'}},"
        "{'type':'result','session_id':'s','exit_code':130,'tokens':{'input':0,'output':0}}]:"
        " print(json.dumps(event),flush=True)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(.01)\n"
    )
    fake.chmod(0o755)
    _wait_for_fixture_ready_before_timeout(monkeypatch, ready)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    session["task"]["termination_grace_s"] = 0.05
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(
        tmp_path,
        hermes_executable=fake,
        hard_timeout_seconds=0.05,
        runtime=FakeRuntime(),
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="SIGKILL"):
        stimulus.run(session["task"], origin)


def test_process_group_signal_race_preserves_timebox_cleanup(tmp_path, monkeypatch):
    from tools.anticipation import hermes_campaign

    class TrackingRuntime(FakeRuntime):
        closed = False

        def close(self, *, deadline=None):
            self.closed = True
            return super().close(deadline=deadline)

    class RacedProcess:
        pid = 12345
        returncode = None

        def __init__(self):
            self.communications = 0

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications <= 2:
                raise subprocess.TimeoutExpired("fake-hermes", timeout)
            self.returncode = -9
            return (
                '{"type":"system","subtype":"init","model":"gemma4:12b"}\n'
                '{"type":"tool_use","name":"read_file","input":{"path":"input.csv"}}\n'
                '{"type":"result","exit_code":130,"error":"Interrupted","tokens":{}}\n',
                "",
            )

        def poll(self):
            return self.returncode

        def wait(self, _timeout=None):
            return self.returncode

    runtime = TrackingRuntime()
    process = RacedProcess()
    monkeypatch.setattr(hermes_campaign.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        hermes_campaign.os,
        "killpg",
        lambda *args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    session = hermes_campaign.build_hermes_campaign(tmp_path)["sessions"][0]
    session["task"]["termination_grace_s"] = 0.01
    Path(session["path"]).mkdir(parents=True)
    stimulus = hermes_campaign.HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        hard_timeout_seconds=0.01,
        runtime=runtime,
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="SIGKILL"):
        stimulus.run(session["task"], origin)

    assert runtime.closed is True


def test_normal_hermes_exit_still_cleans_process_group(tmp_path, monkeypatch):
    from tools.anticipation import hermes_campaign

    class ExitedProcess:
        pid = 12345
        returncode = 0

        def communicate(self, timeout=None):
            return (
                '{"type":"system","subtype":"init","session_id":"s",'
                '"model":"gemma4:12b"}\n'
                '{"type":"tool_use","name":"read_file","tool_call_id":"c",'
                '"input":{"path":"input.csv"}}\n'
                '{"type":"result","session_id":"s","exit_code":0,'
                '"tokens":{"input":1,"output":1}}\n',
                "",
            )

        def poll(self):
            return self.returncode

        def wait(self, _timeout=None):
            return self.returncode

    signals = []
    monkeypatch.setattr(
        hermes_campaign.subprocess, "Popen", lambda *args, **kwargs: ExitedProcess()
    )
    monkeypatch.setattr(
        hermes_campaign.os,
        "killpg",
        lambda process_group, signal_number: signals.append(
            (process_group, signal_number)
        ),
    )
    session = hermes_campaign.build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = hermes_campaign.HermesStimulus(
        tmp_path,
        hermes_executable=tmp_path / "hermes",
        runtime=FakeRuntime(),
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    record = stimulus.run(session["task"], time.monotonic() - 20)

    assert record["workload_status"] == "valid"
    assert signals == [(12345, signal.SIGKILL)]


def test_final_signal_race_preserves_original_error_and_model_cleanup(
    tmp_path, monkeypatch
):
    from tools.anticipation import hermes_campaign

    class TrackingRuntime(FakeRuntime):
        closed = False

        def close(self, *, deadline=None):
            self.closed = True
            return super().close(deadline=deadline)

    class ExitedProcess:
        pid = 12345
        returncode = 0

        def communicate(self, timeout=None):
            return "{malformed\n", ""

        def poll(self):
            return None

        def wait(self, _timeout=None):
            return 0

    runtime = TrackingRuntime()
    monkeypatch.setattr(
        hermes_campaign.subprocess, "Popen", lambda *a, **k: ExitedProcess()
    )
    monkeypatch.setattr(
        hermes_campaign.os,
        "killpg",
        lambda *args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    session = hermes_campaign.build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = hermes_campaign.HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=runtime
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="malformed object-like"):
        stimulus.run(session["task"], origin)

    assert runtime.closed is True


def test_model_cleanup_after_session_deadline_invalidates_workload(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin/hermes"
    fake.parent.mkdir()
    _write_fake_hermes(
        fake,
        [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "model": "gemma4:12b",
            },
            {
                "type": "tool_use",
                "name": "read_file",
                "tool_call_id": "c",
                "input": {"path": "input.csv"},
            },
            {"type": "result", "exit_code": 0, "tokens": {"input": 3, "output": 1}},
        ],
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    session["task"]["cleanup_deadline_s"] = 19
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="cleanup exceeded"):
        stimulus.run(session["task"], origin)
    assert stimulus.session_records()[0]["workload_status"] == "invalid"


def test_shared_capture_stops_collector_and_closes_stimulus_on_task_failure(
    tmp_path, monkeypatch
):
    from tools.anticipation import campaign

    collector = _graceful_fake_collector(tmp_path)
    root = tmp_path / "run"
    plan = {
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
                "task": {"family": "fake"},
            }
        ],
    }
    state = {"closed": False}

    class FailingStimulus:
        def prepare(self):
            return {"ready": True}

        def seed(self, seed):
            return {"model": "fake"}

        def prepare_session(self, session):
            pass

        def run(self, event, origin):
            raise RuntimeError("Hermes failed")

        def session_records(self):
            return [{"status": "incomplete", "family": "fake"}]

        def close(self):
            state["closed"] = True
            return {"unloaded": True}

    monkeypatch.setattr(campaign, "wait_until", lambda deadline: None)
    with pytest.raises(RuntimeError, match="Hermes failed"):
        campaign.capture(
            root, collector, campaign=plan, stimulus_factory=FailingStimulus
        )
    assert state["closed"] is True
    status = json.loads((root / "capture-status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["stimulus_cleanup"] == {"unloaded": True}
    audit = json.loads((root / "raw/session-01/stimulus-audit.json").read_text())
    assert audit["actual_bot_tasks"] == [{"status": "incomplete", "family": "fake"}]


def test_session_record_failure_still_finalizes_real_collector_and_global_cleanup(
    tmp_path, monkeypatch
):
    from tools.anticipation import campaign
    import sys

    collector = tmp_path / "collector"
    collector.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,signal,time\nfrom pathlib import Path\n"
        "p=Path(os.environ['SESSION_DIR'])/'session_manifest.json'\n"
        "p.write_text(json.dumps({'ended_at_utc':None,'parquet_write_failures':0}))\n"
        "def stop(*_):\n"
        " p.write_text(json.dumps({'ended_at_utc':'done','parquet_write_failures':0})); raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT,stop)\nwhile True: time.sleep(.01)\n"
    )
    collector.chmod(0o755)
    root = tmp_path / "run"
    plan = {
        "schema_version": "anticipation-campaign-v1",
        "actual_audit_key": "actual_bot_tasks",
        "sessions": [
            {
                "session_id": "session-01",
                "seed": 1,
                "path": str(root / "raw/session-01"),
                "task": {"family": "fake"},
            }
        ],
    }
    state = {"closed": False}

    class BrokenAuditStimulus:
        def prepare(self):
            return {"ready": True}

        def seed(self, seed):
            return {"model": "fake"}

        def run(self, event, origin):
            return {"status": "ran"}

        def session_records(self):
            raise RuntimeError("audit accessor failed")

        def close(self):
            state["closed"] = True
            return {"unloaded": True}

    monkeypatch.setattr(campaign, "wait_until", lambda deadline: None)
    with pytest.raises(RuntimeError, match="audit accessor failed"):
        campaign.capture(
            root, collector, campaign=plan, stimulus_factory=BrokenAuditStimulus
        )
    assert state["closed"] is True
    manifest = json.loads((root / "raw/session-01/session_manifest.json").read_text())
    assert manifest["ended_at_utc"] == "done"
    status = json.loads((root / "capture-status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["stimulus_cleanup"] == {"unloaded": True}


def _graceful_fake_collector(tmp_path):
    import sys

    collector = tmp_path / "collector"
    collector.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,signal,time\nfrom pathlib import Path\n"
        "p=Path(os.environ['SESSION_DIR'])/'session_manifest.json'\n"
        "p.write_text(json.dumps({'ended_at_utc':None,'parquet_write_failures':0}))\n"
        "def stop(*_):\n"
        " p.write_text(json.dumps({'ended_at_utc':'done','parquet_write_failures':0})); raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT,stop)\nwhile True: time.sleep(.1)\n"
    )
    collector.chmod(0o755)
    return collector
