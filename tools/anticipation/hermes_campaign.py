"""Capture sensors during bounded Hermes agent tasks on local Ollama models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from tools.anticipation.campaign import capture

from .ollama_runtime import OllamaRuntime as OllamaRuntime, _local_endpoint

from .hermes_protocol import build_hermes_campaign as build_hermes_campaign, _prompt
from .task_verification import (
    write_fixture,
    verify_fixture,
)

TASK_CLEANUP_ERRORS = (HTTPException, OSError, RuntimeError, TypeError, ValueError)


PROTOCOL_ID = "hermes-ollama-inference-v4"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
HARD_TIMEOUT_SECONDS = 100.0
SIGTERM_GRACE_SECONDS = 5.0
PRELOAD_KEEP_ALIVE_SECONDS = 180
PRELOAD_COMPLETION_TIMEOUT_SECONDS = 180
DURABLE_CLEANUP_TIMEOUT_SECONDS = 30
DURABLE_CLEANUP_INITIAL_BACKOFF_SECONDS = 0.05
DURABLE_CLEANUP_MAX_BACKOFF_SECONDS = 1.0
CONTROL_PLANE_TIMEOUT_SECONDS = 30
VERIFIER_AS_LIMIT_BYTES = 512 * 1024 * 1024
VERIFIER_NPROC_LIMIT = 8192
VERIFIER_FSIZE_LIMIT_BYTES = 1024 * 1024
VERIFIER_CPU_LIMIT_SECONDS = 4
MODEL_PLAN = (
    ("gemma4:12b", 262144),
    ("granite4.2:8b", 131072),
    ("Ornith-1.5-9B:latest", 262144),
)
EXCLUDED_MODELS = {"muse-glimmer:30b", "nemotron-3.5-lightning:30b"}
ALLOWED_TOOLS = {
    "read_file",
    "write_file",
    "patch",
    "search_files",
}
_DEFAULT_REQUEST_TIMEOUT = object()


def _signal_process_group(process, signal_number):
    """Signal a child process group unless it has already disappeared."""
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


class HermesStimulus:
    """Run one isolated, bounded Hermes task per sensor session."""

    def __init__(
        self,
        root,
        *,
        hermes_executable,
        model=DEFAULT_MODEL,
        endpoint=DEFAULT_ENDPOINT,
        hard_timeout_seconds=HARD_TIMEOUT_SECONDS,
        runtime=None,
    ):
        self.root = Path(root).resolve()
        self.hermes_executable = Path(hermes_executable).resolve()
        self.model = model
        self.endpoint = _local_endpoint(endpoint)
        self.hard_timeout_seconds = float(hard_timeout_seconds)
        self.runtime = runtime or OllamaRuntime(endpoint=self.endpoint, model=model)
        if hasattr(self.runtime, "set_cleanup_report_path"):
            self.runtime.set_cleanup_report_path(self.root / "ollama-cleanup.json")
        self._seed = None
        self._session = None
        self._records = []
        self._runtime_metadata = None
        self._verifier_plan = None

    def prepare(self):
        if not self.hermes_executable.is_file():
            raise FileNotFoundError(self.hermes_executable)
        return self.runtime.prepare()

    def seed(self, seed):
        self._seed = int(seed)
        self._records = []
        index = self._seed - 2026092001
        if index < 0 or index >= 12:
            raise ValueError(f"seed is not assigned to this campaign: {seed}")
        model, expected_context = MODEL_PLAN[index % len(MODEL_PLAN)]
        self._runtime_metadata = self.runtime.select(model, expected_context)
        self.model = model
        return dict(self._runtime_metadata)

    def _write_config(self, home, context_length):
        api_base = self.endpoint + "/v1"
        home.mkdir(parents=True, exist_ok=False)
        (home / "config.yaml").write_text(
            "model:\n"
            f"  default: {self.model}\n"
            "  provider: custom\n"
            f"  base_url: {api_base}\n"
            "  api_key: no-key-required\n"
            f"  context_length: {context_length}\n"
            f"  ollama_num_ctx: {context_length}\n"
            "fallback_providers: []\n"
            "toolsets: [file]\n"
            "mcp_servers: {}\n"
            "memory:\n"
            "  memory_enabled: false\n"
            "  user_profile_enabled: false\n"
            "skills:\n"
            "  external_dirs: []\n"
            "  project_discovery: false\n"
            "  auto_load: []\n"
            "plugins:\n"
            "  enabled: []\n"
            "auxiliary:\n"
            "  title_generation:\n"
            "    enabled: false\n"
            "    model_upgrade_enabled: false\n"
            "  background_review:\n"
            "    enabled: false\n"
        )

    def _write_fixture(self, session, scratch):
        return write_fixture(session, scratch)

    def prepare_session(self, session):
        if self._seed != session["seed"]:
            raise RuntimeError("session seed does not match prepared Hermes stimulus")
        home = Path(session["hermes_home"])
        scratch = Path(session["scratch_path"])
        if self._runtime_metadata is None or self.model != session["model"]:
            raise RuntimeError("session model does not match selected Ollama runtime")
        self._write_config(home, self._runtime_metadata["advertised_context_length"])
        scratch.mkdir(parents=True, exist_ok=False)
        self._verifier_plan = self._write_fixture(session, scratch)
        prompt = _prompt(
            session["task"]["family"],
            session["task"]["fixture_records"],
            session["task"]["input_target_tokens"],
            session["seed"],
            scratch,
        )
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if digest != session["task"]["prompt_sha256"]:
            raise RuntimeError("planned Hermes prompt digest mismatch")
        prompt_path = Path(session["prompt_path"])
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt + "\n")
        self._session = session

    def command(self, session):
        return [
            str(self.hermes_executable),
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

    def environment(self, session, inherited=None):
        inherited = os.environ if inherited is None else inherited
        env = {
            key: inherited[key]
            for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ")
            if key in inherited
        }
        env.update(
            {
                "HERMES_HOME": session["hermes_home"],
                # Skip all bundled and user plugin discovery without passing
                # --safe-mode, which would also discard this isolated config.
                "HERMES_SAFE_MODE": "1",
                "OPENAI_API_KEY": "no-key-required",
                "OPENAI_BASE_URL": self.endpoint + "/v1",
            }
        )
        return env

    @staticmethod
    def _events(stdout):
        events = []
        diagnostics = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as error:
                if stripped.startswith(("{", "[")):
                    raise RuntimeError(
                        "Hermes stream contained a malformed object-like line"
                    ) from error
                diagnostics.append(line)
                continue
            if not isinstance(event, dict):
                raise RuntimeError("Hermes stream-json line was not an object")
            events.append(event)
        return events, diagnostics

    @staticmethod
    def _positive_usage(usage):
        return isinstance(usage, dict) and any(
            isinstance(value, (int, float)) and value > 0 for value in usage.values()
        )

    @staticmethod
    def _scope_tool_calls(tool_calls, scratch):
        scratch = Path(scratch).resolve()
        outside = []
        for call in tool_calls:
            call["explicit_paths"] = []
            values = call.get("input")
            if not isinstance(values, dict):
                continue
            for key in ("path", "cwd", "workdir"):
                value = values.get(key)
                if not isinstance(value, str) or not value:
                    continue
                resolved, in_scope = _resolve_tool_path(value, scratch)
                path_audit = {
                    "field": key,
                    "provided": value,
                    "resolved": str(resolved),
                    "in_scratch": in_scope,
                }
                call["explicit_paths"].append(path_audit)
                if not in_scope:
                    outside.append(
                        {
                            "tool_call_id": call.get("tool_call_id"),
                            "name": call.get("name"),
                            **path_audit,
                        }
                    )
        return outside

    def _verify_fixture(self, session):
        return verify_fixture(session, self._verifier_plan)

    def run(self, event, origin):
        session = self._session
        if session is None or event is not session["task"]:
            raise RuntimeError("Hermes session was not prepared")
        started_mono = time.monotonic()
        record = _task_record(event, started_mono, origin, self._runtime_metadata)
        process = None
        pending_error = None
        timeboxed = False
        forced_kill = False
        try:
            process = subprocess.Popen(
                self.command(session),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.environment(session),
                cwd=session["scratch_path"],
                start_new_session=True,
            )
            stdout, stderr, timeboxed, forced_kill = self._communicate_task(
                process, event, origin, record
            )
            ended_mono = time.monotonic()
            self._write_process_logs(session, stdout, stderr, record)
            tool_calls, tool_results, result, init = self._record_task_events(
                stdout, record, process, ended_mono, origin
            )
            result_error = self._validate_task_result(
                tool_calls, tool_results, result, init, forced_kill, timeboxed
            )
            self._record_task_outcome(
                record, session, tool_calls, process, result, timeboxed, result_error
            )
        except BaseException as error:
            pending_error = error
            record["status"] = "invalid"
            record["workload_status"] = "invalid"
            record.setdefault("task_outcome", "unknown")
            record.setdefault("error", f"{type(error).__name__}: {error}")
            raise
        finally:
            pending_error = self._finish_task(
                process, record, origin, event, pending_error
            )
        if pending_error is not None:
            raise pending_error
        return record

    @staticmethod
    def _write_process_logs(session, stdout, stderr, record):
        output_dir = Path(session["path"])
        stdout_path = output_dir / "hermes-stream.jsonl"
        stderr_path = output_dir / "hermes-stderr.log"
        stdout_path.write_text(stdout or "")
        stderr_path.write_text(stderr or "")
        record["stdout_jsonl_path"] = str(stdout_path)
        record["stderr_path"] = str(stderr_path)

    def session_records(self):
        return list(self._records)

    def close(self):
        return self.runtime.close()

    def _communicate_task(self, process, event, origin, record):
        timeboxed = forced_kill = False
        remaining = origin + event["hard_deadline_s"] - time.monotonic()
        timeout = min(self.hard_timeout_seconds, max(0.001, remaining))
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeboxed = True
            record["configured_timebox_seconds"] = self.hard_timeout_seconds
            record["effective_timebox_seconds"] = timeout
            record["parent_stop_reason"] = f"{timeout:g}s_agent_timebox"
            record["termination_grace_s"] = event["termination_grace_s"]
            _signal_process_group(process, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(
                    timeout=event["termination_grace_s"]
                )
            except subprocess.TimeoutExpired:
                forced_kill = True
                _signal_process_group(process, signal.SIGKILL)
                stdout, stderr = process.communicate()

        return stdout, stderr, timeboxed, forced_kill

    def _record_task_events(self, stdout, record, process, ended_mono, origin):
        events, diagnostics = self._events(stdout)
        tool_calls, tool_results = _tool_events(events)
        result, init = _terminal_events(events)
        record.update(
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
            actual_end_s=ended_mono - origin,
            exit_code=process.returncode,
            session_id=init.get("session_id")
            or (result.get("session_id") if result else None),
            model=init.get("model"),
            tool_call_count=len(tool_calls),
            tool_calls=tool_calls,
            tool_results=tool_results,
            stdout_diagnostics=diagnostics,
            result_status={
                "exit_code": result.get("exit_code") if result else None,
                "error": result.get("error") if result else None,
            },
            usage=result.get("tokens") if result else None,
        )
        return tool_calls, tool_results, result, init

    def _validate_task_result(
        self, tool_calls, tool_results, result, init, forced_kill, timeboxed
    ):
        _validate_tools(tool_calls, tool_results)
        if init.get("model") != self.model:
            raise RuntimeError("Hermes did not report the configured local model")
        if result is None:
            raise RuntimeError("Hermes stream ended without a terminal result")
        if forced_kill:
            raise RuntimeError("Hermes timebox required SIGKILL")
        return self._validate_terminal_status(result, timeboxed)

    def _validate_terminal_status(self, result, timeboxed):
        result_error = result.get("error")
        parent_interrupt = timeboxed and result_error == "Interrupted"
        if timeboxed and not parent_interrupt:
            raise RuntimeError(
                "Hermes timebox ended without an Interrupted terminal result"
            )
        if result_error and not parent_interrupt:
            raise RuntimeError("Hermes terminal result reported an explicit error")
        if not timeboxed and not self._positive_usage(result.get("tokens")):
            raise RuntimeError("Hermes terminal result omitted positive token usage")

        return result_error

    def _record_task_outcome(
        self, record, session, tool_calls, process, result, timeboxed, result_error
    ):
        outside = self._scope_tool_calls(tool_calls, session["scratch_path"])
        record["out_of_scope_tool_calls"] = outside
        record["verifier"] = self._verify_fixture(session)
        record["workload_status"] = "valid"
        if timeboxed:
            record["status"] = "timeboxed"
            record["execution_status"] = "timeboxed"
            record["usage_quality"] = "partial_after_parent_sigterm"
            record["framework_interruption"] = result_error
            record["task_outcome"] = "incomplete"
        else:
            record["status"] = "workload_valid"
            record["execution_status"] = "completed"
            record["usage_quality"] = "reported_positive"
            successful_exit = process.returncode == 0 and result.get("exit_code") == 0
            record["task_outcome"] = (
                "verified_complete"
                if successful_exit
                and record["verifier"]["status"] == "passed"
                and not outside
                else "incomplete"
            )

    def _finish_task(self, process, record, origin, event, pending_error):
        if process is not None:
            leader_running = process.poll() is None
            # Descendants can survive after the Hermes group leader exits.
            # Always signal the isolated process group before model cleanup.
            _signal_process_group(process, signal.SIGKILL)
            if leader_running:
                process.wait()
                record["forced_kill"] = True
        try:
            record["model_cleanup"] = self.runtime.close(
                deadline=origin + event["cleanup_deadline_s"]
            )
            record["model_cleanup_end_s"] = time.monotonic() - origin
            if record["model_cleanup_end_s"] > event["cleanup_deadline_s"]:
                raise RuntimeError(
                    "owned model cleanup exceeded the 130s session deadline"
                )
        except TASK_CLEANUP_ERRORS as error:
            record["model_cleanup_error"] = f"{type(error).__name__}: {error}"
            record["status"] = "invalid"
            record["workload_status"] = "invalid"
            if pending_error is None:
                pending_error = error
        record.setdefault("ended_at_utc", datetime.now(timezone.utc).isoformat())
        record.setdefault("actual_end_s", time.monotonic() - origin)
        if not self._records or self._records[-1] is not record:
            self._records.append(record)
        return pending_error


def _resolve_tool_path(value, scratch):
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = scratch / resolved
    resolved = resolved.resolve()
    in_scope = resolved == scratch or scratch in resolved.parents
    return resolved, in_scope


def _validate_tools(tool_calls, tool_results):
    invalid_tools = [
        tool_event
        for tool_event in (*tool_calls, *tool_results)
        if not isinstance(tool_event["name"], str)
        or not tool_event["name"].strip()
        or tool_event["name"] not in ALLOWED_TOOLS
    ]
    if not tool_calls:
        raise RuntimeError("Hermes task completed without an observed tool call")
    if invalid_tools:
        raise RuntimeError("Hermes reported an unsupported or empty tool name")


def _tool_events(events):
    tool_calls = [
        {
            "name": e.get("name"),
            "tool_call_id": e.get("tool_call_id"),
            "input": e.get("input"),
        }
        for e in events
        if e.get("type") == "tool_use"
    ]
    tool_results = [
        {
            "name": e.get("name"),
            "tool_call_id": e.get("tool_call_id"),
            "is_error": e.get("is_error"),
            "duration_ms": e.get("duration_ms"),
        }
        for e in events
        if e.get("type") == "tool_result"
    ]
    return tool_calls, tool_results


def _terminal_events(events):
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"),
        {},
    )
    return result, init


def _task_record(event, started_mono, origin, runtime_metadata):
    return {
        "family": event["family"],
        "prompt_sha256": event["prompt_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_start_s": started_mono - origin,
        "status": "running",
        "runtime": dict(runtime_metadata or {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collector", required=True, type=Path)
    parser.add_argument("--hermes", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = build_hermes_campaign(args.output)
    capture(
        args.output,
        args.collector,
        campaign=campaign,
        stimulus_factory=lambda: HermesStimulus(
            args.output, hermes_executable=args.hermes
        ),
    )


if __name__ == "__main__":
    main()
