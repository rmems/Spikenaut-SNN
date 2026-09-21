"""Failures caught: mixed protocols, ambient Hermes state, and leaked children or models."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

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
def ollama_server(
    *,
    initially_loaded=False,
    unload_sticks=False,
    preload_response_delay=0,
    post_load_context=None,
    post_load_model=None,
    extra_resident=False,
    ps_failures_after_load=0,
):
    state = {
        "loaded": initially_loaded,
        "requests": [],
        "unload_sticks": unload_sticks,
        "model": "gemma4:12b",
        "context": 262144,
        "ps_failures_after_load": ps_failures_after_load,
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
                if state["loaded"] and state["ps_failures_after_load"]:
                    state["ps_failures_after_load"] -= 1
                    return self.send_error(503)
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
                    if extra_resident:
                        models.append(
                            {
                                "name": "unowned:latest",
                                "model": "unowned:latest",
                                "context_length": 1024,
                            }
                        )
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
                state["model"] = post_load_model or payload["model"]
                state["context"] = post_load_context or payload["options"]["num_ctx"]
                if preload_response_delay:
                    time.sleep(preload_response_delay)
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
    assert one["protocol_id"] == "hermes-ollama-inference-v2"
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
    assert f"cwd: {session['scratch_path']}" in config
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
    prompt = Path(session["prompt_path"]).read_text()
    assert str(Path(session["scratch_path"]) / "input.csv") in prompt
    assert str(Path(session["scratch_path"]) / "output.json") in prompt
    assert str(Path(session["scratch_path"]) / "verify.py") in prompt
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
    assert env["HERMES_SAFE_MODE"] == "1"
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


def test_fake_cli_stream_records_diagnostics_real_schema_cwd_and_verified_task(
    tmp_path,
):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "s1",
            "model": "gemma4:12b",
        },
        {
            "type": "tool_use",
            "name": "read_file",
            "tool_call_id": "c1",
            "input": {"path": str(tmp_path / "hermes/session-01/scratch/input.csv")},
        },
        {
            "type": "tool_result",
            "name": "read_file",
            "tool_call_id": "c1",
            "output": "ok",
        },
        {
            "type": "result",
            "exit_code": 0,
            "session_id": "s1",
            "tokens": {"input": 9, "output": 4},
        },
    ]
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,os\n"
        f"events={events!r}\n"
        "print(json.dumps(events[0]), flush=True)\n"
        "print('  ⚠ tirith security scanner enabled but not available', flush=True)\n"
        "from pathlib import Path\n"
        "rows=Path('input.csv').read_text().splitlines()[1:]\n"
        "totals={}\n"
        "for row in rows:\n"
        " category,amount=row.split(','); totals[category]=totals.get(category,0)+int(amount)\n"
        "Path('output.json').write_text(json.dumps(dict(sorted(totals.items()))))\n"
        "for event in events[1:]: print(json.dumps(event), flush=True)\n"
        "Path('observed-cwd.txt').write_text(os.getcwd())\n"
    )
    fake.chmod(0o755)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    record = stimulus.run(session["task"], __import__("time").monotonic() - 20)
    assert record["workload_status"] == "valid"
    assert record["task_outcome"] == "verified_complete"
    assert record["verifier"]["status"] == "passed"
    assert record["tool_call_count"] == 1
    assert record["tool_calls"][0]["name"] == "read_file"
    assert record["usage"] == {"input": 9, "output": 4}
    assert record["stdout_diagnostics"] == [
        "  ⚠ tirith security scanner enabled but not available"
    ]
    assert (
        Path(session["scratch_path"], "observed-cwd.txt").read_text()
        == session["scratch_path"]
    )
    assert Path(record["stdout_jsonl_path"]).exists()
    assert record["model_cleanup"]["unloaded"] is True


def test_nonzero_bot_result_is_valid_workload_but_incomplete_task(tmp_path):
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
                "name": "terminal",
                "tool_call_id": "c",
                "input": {"command": "false"},
            },
            {
                "type": "tool_result",
                "name": "terminal",
                "tool_call_id": "c",
                "output": "",
                "is_error": True,
            },
            {
                "type": "result",
                "exit_code": 1,
                "session_id": "s",
                "tokens": {"input": 10, "output": 2},
                "text": "I finished successfully",
            },
        ],
        exit_code=1,
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    record = stimulus.run(session["task"], time.monotonic() - 20)
    assert record["workload_status"] == "valid"
    assert record["task_outcome"] == "incomplete"
    assert record["result_status"] == {"exit_code": 1, "error": None}


def test_explicit_hermes_error_is_infrastructure_failure(tmp_path):
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
            {
                "type": "result",
                "exit_code": 1,
                "error": "provider transport failed",
                "tokens": {"input": 10, "output": 2},
            },
        ],
        exit_code=1,
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    with pytest.raises(RuntimeError, match="explicit error"):
        stimulus.run(session["task"], time.monotonic() - 20)


def test_out_of_scope_path_is_audited_and_prevents_task_success(tmp_path):
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
                "input": {"path": "/tmp/outside.txt"},
            },
            {"type": "result", "exit_code": 0, "tokens": {"input": 3, "output": 1}},
        ],
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    record = stimulus.run(session["task"], time.monotonic() - 20)
    assert record["workload_status"] == "valid"
    assert record["task_outcome"] == "incomplete"
    assert record["out_of_scope_tool_calls"][0]["resolved"] == "/tmp/outside.txt"


@pytest.mark.parametrize("name", ["", "web_search"])
def test_empty_or_unsupported_tool_name_is_rejected(tmp_path, name):
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
            {"type": "tool_use", "name": name, "tool_call_id": "c", "input": {}},
            {"type": "result", "exit_code": 0, "tokens": {"input": 3, "output": 1}},
        ],
    )
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(tmp_path, hermes_executable=fake, runtime=FakeRuntime())
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    with pytest.raises(RuntimeError, match="unsupported or empty"):
        stimulus.run(session["task"], time.monotonic() - 20)


def test_malformed_object_like_stream_line_fails_but_diagnostic_does_not():
    from tools.anticipation.hermes_campaign import HermesStimulus

    events, diagnostics = HermesStimulus._events("plain warning\n")
    assert events == [] and diagnostics == ["plain warning"]
    with pytest.raises(RuntimeError, match="malformed"):
        HermesStimulus._events("plain warning\n{broken\n")


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
    assert stimulus.session_records()[0]["status"] == "invalid"


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
                "name": "read_file",
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
    assert stimulus.session_records()[0]["status"] == "invalid"


def test_graceful_sigterm_is_valid_timeboxed_workload_with_partial_usage(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,signal,time\n"
        "print(json.dumps({'type':'system','subtype':'init','session_id':'s','model':'gemma4:12b'}),flush=True)\n"
        "print(json.dumps({'type':'tool_use','name':'read_file','tool_call_id':'c','input':{'path':'input.csv'}}),flush=True)\n"
        "def stop(*_):\n"
        " print(json.dumps({'type':'result','session_id':'s','exit_code':130,'tokens':{'input':0,'output':0},'error':'Interrupted'}),flush=True); raise SystemExit(130)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "while True: time.sleep(.01)\n"
    )
    fake.chmod(0o755)
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
    assert record["parent_stop_reason"] == "100s_agent_timebox"
    assert record["framework_interruption"] == "Interrupted"
    assert record["usage_quality"] == "partial_after_parent_sigterm"
    assert Path(record["stdout_jsonl_path"]).exists()
    assert Path(record["stderr_path"]).exists()


def test_sigterm_without_terminal_result_is_not_valid_timebox(tmp_path):
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
        ],
        sleep_seconds=30,
    )
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
    with pytest.raises(RuntimeError, match="terminal result"):
        stimulus.run(session["task"], time.monotonic() - 20)


def test_timebox_requiring_sigkill_is_invalid_even_with_prior_result(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin/hermes"
    fake.parent.mkdir()
    fake.write_text(
        "#!" + os.sys.executable + "\n"
        "import json,signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "for event in ["
        "{'type':'system','subtype':'init','session_id':'s','model':'gemma4:12b'},"
        "{'type':'tool_use','name':'read_file','tool_call_id':'c','input':{'path':'input.csv'}},"
        "{'type':'result','session_id':'s','exit_code':130,'tokens':{'input':0,'output':0}}]:"
        " print(json.dumps(event),flush=True)\n"
        "while True: time.sleep(.01)\n"
    )
    fake.chmod(0o755)
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
    with pytest.raises(RuntimeError, match="SIGKILL"):
        stimulus.run(session["task"], time.monotonic() - 20)


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
    with pytest.raises(RuntimeError, match="cleanup exceeded"):
        stimulus.run(session["task"], time.monotonic() - 20)
    assert stimulus.session_records()[0]["workload_status"] == "invalid"


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


def test_runtime_cleans_model_after_preload_response_timeout():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_response_delay=0.15) as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint, preload_timeout_seconds=0.02)
        runtime.prepare()
        with pytest.raises(Exception):
            runtime.select("gemma4:12b", 262144)
        assert state["loaded"] is True
        cleanup = runtime.close()
        assert cleanup == {"model": "gemma4:12b", "unloaded": True}
        assert state["loaded"] is False


def test_runtime_model_switch_validation_failure_cleans_new_model():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(post_load_context=17) as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint)
        runtime.prepare()
        with pytest.raises(RuntimeError, match="resident context"):
            runtime.select("granite4.2:8b", 131072)
        assert state["model"] == "granite4.2:8b"
        assert runtime.close()["model"] == "granite4.2:8b"
        assert state["loaded"] is False


def test_runtime_never_unloads_unowned_wrong_resident_model():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(post_load_model="other:latest") as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint)
        runtime.prepare()
        with pytest.raises(RuntimeError, match="unexpected resident"):
            runtime.select("granite4.2:8b", 131072)
        assert runtime.close()["unloaded"] is False
        assert state["loaded"] is True
        assert not any(
            request[2] and request[2].get("keep_alive") == 0
            for request in state["requests"]
        )


@pytest.mark.parametrize(
    "server_options, error_match",
    [
        ({"extra_resident": True}, "exactly one"),
        ({"ps_failures_after_load": 1}, "HTTP Error 503"),
    ],
)
def test_runtime_retains_model_identity_for_other_post_load_failures(
    server_options, error_match
):
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(**server_options) as (endpoint, state):
        runtime = OllamaRuntime(endpoint=endpoint)
        runtime.prepare()
        with pytest.raises(Exception, match=error_match):
            runtime.select("granite4.2:8b", 131072)
        cleanup = runtime.close()
        assert cleanup == {"model": "granite4.2:8b", "unloaded": True}
        assert state["loaded"] is False


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
