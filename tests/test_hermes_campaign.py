"""Hermes campaign regression tests."""

from __future__ import annotations


import json


import os


from pathlib import Path


import time


import pytest


from tests.hermes_fixture import FakeRuntime, _write_fake_hermes


def test_protocol_is_separate_deterministic_and_every_split_has_each_task(tmp_path):
    from tools.anticipation.hermes_campaign import build_hermes_campaign

    one = build_hermes_campaign(tmp_path / "run")
    again = build_hermes_campaign(tmp_path / "run")
    assert one == again
    assert one["protocol_id"] == "hermes-ollama-inference-v4"
    assert one["workload_class"] == "ai-compute"
    assert one["model_resource_envelope"] == {
        "resident_models": 1,
        "concurrent_agent_runs": 1,
        "context_policy": "advertised-architecture-maximum",
        "preload_keep_alive_seconds": 180,
        "preload_completion_timeout_seconds": 180,
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


def test_session_files_config_and_argv_are_hermetic_and_bounded(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    campaign = build_hermes_campaign(tmp_path)
    session = campaign["sessions"][0]
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=Path("/opt/hermes"), runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    _assert_hermetic_config(session)
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
        "file",
    ]
    _assert_session_prompt(session)

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


def test_csv_verifier_rejects_correct_values_in_wrong_key_order(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        item
        for item in build_hermes_campaign(tmp_path)["sessions"]
        if item["task"]["family"] == "csv-aggregation"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    expected = stimulus._verifier_plan["expected"]
    reversed_output = dict(reversed(list(expected.items())))
    Path(stimulus._verifier_plan["path"]).write_text(json.dumps(reversed_output))

    verification = stimulus._verify_fixture(session)

    assert verification == {"status": "failed", "reason": "output key order mismatch"}


def test_hermes_executable_is_explicit_for_api_and_cli(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, main

    with pytest.raises(TypeError, match="hermes_executable"):
        HermesStimulus(tmp_path, runtime=FakeRuntime())

    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--collector", str(tmp_path / "collector")])
    assert error.value.code == 2


def test_relative_hermes_executable_survives_scratch_working_directory(
    tmp_path, monkeypatch
):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
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
            {"type": "tool_use", "name": "read_file", "tool_call_id": "c", "input": {}},
            {"type": "result", "exit_code": 0, "tokens": {"input": 1, "output": 1}},
        ],
    )
    monkeypatch.chdir(tmp_path)
    session = build_hermes_campaign(tmp_path)["sessions"][0]
    Path(session["path"]).mkdir(parents=True)
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=Path("bin/hermes"), runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)

    record = stimulus.run(session["task"], time.monotonic() - 20)

    assert record["workload_status"] == "valid"


def test_fake_cli_stream_records_diagnostics_real_schema_cwd_and_verified_task(
    tmp_path,
):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    fake = tmp_path / "bin" / "hermes"
    fake.parent.mkdir()
    events = _successful_csv_events(tmp_path)
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
    assert record["workload_status"] == "valid"
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
                "name": "read_file",
                "tool_call_id": "c",
                "input": {"path": "missing.csv"},
            },
            {
                "type": "tool_result",
                "name": "read_file",
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
    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="explicit error"):
        stimulus.run(session["task"], origin)


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


@pytest.mark.parametrize("name", ["", "web_search", "terminal"])
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
    origin = time.monotonic() - 20
    with pytest.raises(RuntimeError, match="unsupported or empty"):
        stimulus.run(session["task"], origin)


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
    origin = __import__("time").monotonic() - 20
    with pytest.raises(RuntimeError, match="tool call"):
        stimulus.run(session["task"], origin)
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
    origin = __import__("time").monotonic() - 20
    with pytest.raises(RuntimeError, match="terminal result"):
        stimulus.run(session["task"], origin)
    assert stimulus.session_records()[0]["status"] == "invalid"


def test_protocol_covers_each_model_and_task_in_each_split(tmp_path):
    from tools.anticipation.hermes_campaign import build_hermes_campaign

    one = build_hermes_campaign(tmp_path / "run")
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


def test_protocol_prompts_and_budgets_are_predeclared(tmp_path):
    from tools.anticipation.hermes_campaign import build_hermes_campaign

    one = build_hermes_campaign(tmp_path / "run")
    assert all(len(s["task"]["prompt_sha256"]) == 64 for s in one["sessions"])
    assert len({s["task"]["prompt_sha256"] for s in one["sessions"]}) == 12
    assert all(s["task"]["termination_grace_s"] == 5.0 for s in one["sessions"])
    assert [s["task"]["input_target_tokens"] for s in one["sessions"][:6]] == [
        1024,
        8192,
        16384,
        1024,
        8192,
        16384,
    ]


def _assert_hermetic_config(session):
    config = (Path(session["hermes_home"]) / "config.yaml").read_text()
    assert "default: gemma4:12b" in config
    assert "provider: custom" in config
    assert "base_url: http://127.0.0.1:11434/v1" in config
    assert config.count("262144") == 2
    assert "fallback_providers: []" in config
    assert "toolsets: [file]" in config
    assert "terminal:" not in config
    assert "mcp_servers: {}" in config
    assert "memory_enabled: false" in config
    assert "user_profile_enabled: false" in config
    assert "enabled: []" in config
    assert "model_upgrade_enabled: false" in config


def _successful_csv_events(tmp_path):
    return [
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


def _assert_session_prompt(session):
    prompt = Path(session["prompt_path"]).read_text()
    assert str(Path(session["scratch_path"]) / "input.csv") in prompt
    assert str(Path(session["scratch_path"]) / "output.json") in prompt
    assert "the harness will verify the output" in prompt
    assert not (Path(session["scratch_path"]) / "verify.py").exists()
    assert list(Path(session["scratch_path"]).iterdir())
