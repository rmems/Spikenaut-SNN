"""Failures caught: mixed protocols, ambient Hermes state, and leaked children or models."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import pytest


class FakeRuntime:
    def prepare(self):
        return {"ollama_version": "test"}

    def select(self, model, context):
        return {
            "model": model,
            "architecture": "test",
            "advertised_context_length": context,
            "digest": "digest",
            "quantization": "Q-test",
            "residency": {"size": 10, "size_vram": 6, "context_length": context},
        }

    def close(self):
        return {"model": "test", "unloaded": True}


@contextmanager
def ollama_server(*, initially_loaded=False, unload_sticks=False):
    state = {
        "loaded": initially_loaded,
        "requests": [],
        "unload_sticks": unload_sticks,
        "model": "gemma4:12b",
        "context": 262144,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _json(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state["requests"].append(("GET", self.path, None))
            if self.path == "/api/version":
                return self._json({"version": "0.33.3"})
            if self.path == "/api/ps":
                models = []
                if state["loaded"]:
                    model = state.get("model", "gemma4:12b")
                    models = [
                        {
                            "name": model,
                            "model": model,
                            "digest": "f87405c6d8adfull",
                            "size": 9_800_000_000,
                            "size_vram": 8_700_000_000,
                            "context_length": state["context"],
                        }
                    ]
                return self._json({"models": models})
            self.send_error(404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(("POST", self.path, payload))
            if self.path == "/api/show":
                contexts = {
                    "gemma4:12b": ("gemma4", 262144),
                    "granite4.2:8b": ("granite", 131072),
                    "Ornith-1.5-9B:latest": ("qwen3", 262144),
                }
                architecture, context = contexts[payload["model"]]
                return self._json(
                    {
                        "details": {"quantization_level": "Q6_K"},
                        "model_info": {
                            "general.architecture": architecture,
                            f"{architecture}.context_length": context,
                        },
                    }
                )
            if self.path != "/api/generate":
                return self.send_error(404)
            if payload.get("keep_alive") == 0:
                if not state["unload_sticks"]:
                    state["loaded"] = False
            else:
                state["loaded"] = True
                state["model"] = payload["model"]
                state["context"] = payload["options"]["num_ctx"]
            self._json({"done": True, "response": "", "load_duration": 10})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_protocol_is_separate_deterministic_and_every_split_has_each_task(tmp_path):
    from tools.anticipation.hermes_campaign import build_hermes_campaign

    one = build_hermes_campaign(tmp_path / "run")
    again = build_hermes_campaign(tmp_path / "run")
    assert one == again
    assert one["protocol_id"] == "hermes-ollama-inference-v1"
    assert one["workload_class"] == "ai-compute"
    assert one["model_resource_envelope"] == {
        "resident_models": 1,
        "concurrent_agent_runs": 1,
        "context_policy": "advertised-architecture-maximum",
    }
    assert "allocation_limit_bytes" not in one
    assert [s["split"] for s in one["sessions"]] == ["train"] * 6 + [
        "validation"
    ] * 3 + ["test"] * 3
    assert len({s["seed"] for s in one["sessions"]}) == 12
    assert [s["model"] for s in one["sessions"]] == [
        "gemma4:12b",
        "granite4.2:8b",
        "Ornith-1.5-9B:latest",
    ] * 4
    assert [s["advertised_context_length"] for s in one["sessions"]] == [
        262144,
        131072,
        262144,
    ] * 4
    for model in {s["model"] for s in one["sessions"]}:
        assert [s["split"] for s in one["sessions"] if s["model"] == model] == [
            "train",
            "train",
            "validation",
            "test",
        ]
    for split in ("train", "validation", "test"):
        assert {
            s["task"]["family"] for s in one["sessions"] if s["split"] == split
        } == {"csv-aggregation", "json-transformation", "python-bugfix"}
    assert all(len(s["task"]["prompt_sha256"]) == 64 for s in one["sessions"])
    assert len({s["task"]["prompt_sha256"] for s in one["sessions"]}) == 12
    assert [s["task"]["input_target_tokens"] for s in one["sessions"][:6]] == [
        1024,
        8192,
        16384,
        1024,
        8192,
        16384,
    ]


def test_session_files_config_and_argv_are_hermetic_and_bounded(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    campaign = build_hermes_campaign(tmp_path)
    session = campaign["sessions"][0]
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=Path("/opt/hermes"), runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    config = (Path(session["hermes_home"]) / "config.yaml").read_text()
    assert "default: gemma4:12b" in config
    assert "provider: custom" in config
    assert "base_url: http://127.0.0.1:11434/v1" in config
    assert config.count("262144") == 2
    assert "fallback_providers: []" in config
    assert "toolsets: [terminal, file]" in config
    assert "mcp_servers: {}" in config
    assert "memory_enabled: false" in config
    assert "user_profile_enabled: false" in config
    assert "enabled: []" in config
    assert "model_upgrade_enabled: false" in config
    argv = stimulus.command(session)
    assert argv == [
        "/opt/hermes",
        "chat",
        "--query-file",
        session["prompt_path"],
        "--oneshot",
        "--format",
        "stream-json",
        "--source",
        "tool",
        "--max-turns",
        "4",
        "--run-budget",
        "80",
        "--in",
        session["scratch_path"],
        "--ignore-rules",
        "--toolsets",
        "terminal,file",
    ]
    assert Path(session["prompt_path"]).read_text()
    assert list(Path(session["scratch_path"]).iterdir())

    env = stimulus.environment(
        session,
        {
            "PATH": "/bin",
            "HOME": "/home/test",
            "OPENAI_API_KEY": "secret",
            "AWS_SECRET_ACCESS_KEY": "secret",
        },
    )
    assert env["PATH"] == "/bin" and env["HOME"] == "/home/test"
    assert env["HERMES_HOME"] == session["hermes_home"]
    assert env["OPENAI_API_KEY"] == "no-key-required"
    assert "AWS_SECRET_ACCESS_KEY" not in env


def _write_fake_hermes(path, events, exit_code=0, sleep_seconds=0):
    path.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,time\n"
        f"events={events!r}\n"
        "for event in events:\n print(json.dumps(event), flush=True)\n"
        f"time.sleep({sleep_seconds!r})\n"
        f"raise SystemExit({exit_code})\n"
    )
    path.chmod(0o755)


def test_fake_cli_stream_records_verified_tool_use_and_usage(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    _write_fake_hermes(
        fake,
        [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s1",
                "model": "gemma4:12b",
            },
            {
                "type": "tool_use",
                "tool_name": "read_file",
                "tool_call_id": "c1",
                "input": {"path": "input.csv"},
            },
            {
                "type": "tool_result",
                "tool_name": "read_file",
                "tool_call_id": "c1",
                "output": "ok",
            },
            {
                "type": "result",
                "exit_code": 0,
                "session_id": "s1",
                "tokens": {"input": 9, "output": 4},
            },
        ],
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    record = stimulus.run(session["task"], __import__("time").monotonic() - 20)
    assert record["status"] == "complete"
    assert record["tool_call_count"] == 1
    assert record["tool_calls"][0]["name"] == "read_file"
    assert record["usage"] == {"input": 9, "output": 4}
    assert Path(record["stdout_jsonl_path"]).exists()


def test_no_tool_call_or_cli_error_marks_task_incomplete(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    _write_fake_hermes(fake, [{"type": "result", "exit_code": 0, "tokens": {}}])
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    with pytest.raises(RuntimeError, match="tool call"):
        stimulus.run(session["task"], __import__("time").monotonic() - 20)
    assert stimulus.session_records()[0]["status"] == "incomplete"


def test_tool_call_without_terminal_result_is_incomplete(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    _write_fake_hermes(
        fake,
        [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s1",
                "model": "gemma4:12b",
            },
            {
                "type": "tool_use",
                "tool_name": "read_file",
                "tool_call_id": "c1",
                "input": {},
            },
        ],
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    with pytest.raises(RuntimeError, match="terminal result"):
        stimulus.run(session["task"], __import__("time").monotonic() - 20)
    assert stimulus.session_records()[0]["status"] == "incomplete"


def test_hard_timeout_kills_process_group_and_is_audited(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    _write_fake_hermes(fake, [], sleep_seconds=30)
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
    with pytest.raises(RuntimeError, match="hard timeout"):
        stimulus.run(session["task"], __import__("time").monotonic() - 20)
    record = stimulus.session_records()[0]
    assert record["status"] == "hard_timeout"
    assert Path(record["stdout_jsonl_path"]).exists()
    assert Path(record["stderr_path"]).exists()


def test_runtime_refuses_existing_model_and_verifies_owned_unload():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(initially_loaded=True) as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint)
        with pytest.raises(RuntimeError, match="already loaded"):
            runtime.prepare()
        assert runtime.close()["unloaded"] is False
        assert state["loaded"] is True
    with ollama_server() as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint)
        runtime.prepare()
        metadata = runtime.select("gemma4:12b", 262144)
        assert metadata["digest"] == "f87405c6d8adfull"
        assert metadata["residency"] == {
            "size": 9_800_000_000,
            "size_vram": 8_700_000_000,
            "size_cpu": 1_100_000_000,
            "context_length": 262144,
        }
        assert runtime.close()["unloaded"] is True
        assert state["loaded"] is False


def test_runtime_rejects_non_loopback_and_propagates_unload_failure():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with pytest.raises(ValueError, match="127.0.0.1"):
        OllamaRuntime(endpoint="http://localhost:11434")
    with ollama_server(unload_sticks=True) as (endpoint, _):
        runtime = OllamaRuntime(endpoint=endpoint)
        runtime.prepare()
        runtime.select("gemma4:12b", 262144)
        with pytest.raises(RuntimeError, match="remained resident"):
            runtime.close()


def test_runtime_excludes_disallowed_models_and_requires_architecture_context():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server() as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint)
        runtime.prepare()
        for model in ("muse-glimmer:30b", "nemotron-3.5-lightning:30b"):
            with pytest.raises(ValueError, match="excluded"):
                runtime.select(model, 1)
        assert not state["loaded"]


def test_shared_capture_stops_collector_and_closes_stimulus_on_task_failure(
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
        "signal.signal(signal.SIGINT,stop)\nwhile True: time.sleep(.1)\n"
    )
    collector.chmod(0o755)
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
