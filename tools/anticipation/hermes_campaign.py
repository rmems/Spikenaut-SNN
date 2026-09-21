"""Capture sensors during bounded Hermes agent tasks on local Ollama models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import tempfile

from tools.anticipation.campaign import capture

from .ollama_runtime import OllamaRuntime as OllamaRuntime, _local_endpoint


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


def _prompt(family, records, target_tokens, seed, scratch):
    scratch = Path(scratch).resolve()
    common = (
        f"Work only in the scratch directory {scratch}. Do not use the network, install "
        "anything, access other directories, or run repository operations. Use the enabled "
        "file tools to perform the task; the harness will verify the output. Then answer concisely. "
        f"The synthetic fixture seed is {seed}. "
    )
    if family == "csv-aggregation":
        return common + (
            f"Read all {records} data rows in {scratch / 'input.csv'} "
            f"(approximately {target_tokens} input "
            "tokens), aggregate amount by category, and "
            f"write {scratch / 'output.json'} as an object with alphabetically sorted category "
            "keys and numeric totals."
        )
    if family == "json-transformation":
        return common + (
            f"Read all {records} records in {scratch / 'input.json'} "
            f"(approximately {target_tokens} input "
            "tokens), retain enabled records, sort by id, "
            f"and write {scratch / 'output.json'} containing objects with id and score fields. "
            "The harness will run the fixed verifier."
        )
    return common + (
        f"Read the {records}-line {scratch / 'specification.txt'} "
        f"(approximately {target_tokens} input tokens) plus {scratch / 'transform.py'}. "
        f"Fix the small bug in {scratch / 'transform.py'} so it follows the specification. "
        "The harness will verify the result outside the scratch directory."
    )


def build_hermes_campaign(root):
    root = Path(root).resolve()
    sessions = _hermes_sessions(root)
    return {
        "schema_version": "anticipation-campaign-v1",
        "protocol_id": PROTOCOL_ID,
        "feature_map_id": "anticipation-observed-gpu-v1",
        "workload_class": "ai-compute",
        "actual_audit_key": "actual_bot_tasks",
        "duration_s": 150,
        "active_window_s": [20, 130],
        "poll_interval_ms": 100,
        "min_examples_per_session": 500,
        "models": [
            {"model": model, "advertised_context_length": context}
            for model, context in MODEL_PLAN
        ],
        "excluded_models": sorted(EXCLUDED_MODELS),
        "model_resource_envelope": {
            "resident_models": 1,
            "concurrent_agent_runs": 1,
            "context_policy": "advertised-architecture-maximum",
            "preload_keep_alive_seconds": PRELOAD_KEEP_ALIVE_SECONDS,
            "preload_completion_timeout_seconds": PRELOAD_COMPLETION_TIMEOUT_SECONDS,
        },
        "training_budget_seconds": 1200,
        "promising_criterion": {
            "primary_improvement": 0.05,
            "max_target_degradation": 0.05,
        },
        "sessions": sessions,
    }


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

    def _write_config(self, home, context_length, scratch):
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
        family = session["task"]["family"]
        count = session["task"]["fixture_records"]
        rng = random.Random(session["seed"])
        if family == "csv-aggregation":
            rows = ["category,amount"]
            totals = {}
            for _ in range(count):
                category = rng.choice(("alpha", "beta", "delta", "gamma"))
                amount = rng.randint(1, 99)
                rows.append(f"{category},{amount}")
                totals[category] = totals.get(category, 0) + amount
            (scratch / "input.csv").write_text("\n".join(rows) + "\n")
            expected = dict(sorted(totals.items()))
        elif family == "json-transformation":
            records = [
                {
                    "id": f"item-{i:05}",
                    "score": rng.randint(0, 1000),
                    "enabled": rng.choice((True, False)),
                }
                for i in range(count)
            ]
            rng.shuffle(records)
            (scratch / "input.json").write_text(json.dumps(records, indent=2) + "\n")
            expected = [
                {"id": r["id"], "score": r["score"]}
                for r in sorted(records, key=lambda x: x["id"])
                if r["enabled"]
            ]
        else:
            return _python_fixture(session, scratch, count)
        return {
            "kind": "json-output",
            "path": str(scratch / "output.json"),
            "expected": expected,
        }

    def prepare_session(self, session):
        if self._seed != session["seed"]:
            raise RuntimeError("session seed does not match prepared Hermes stimulus")
        home = Path(session["hermes_home"])
        scratch = Path(session["scratch_path"])
        if self._runtime_metadata is None or self.model != session["model"]:
            raise RuntimeError("session model does not match selected Ollama runtime")
        self._write_config(
            home, self._runtime_metadata["advertised_context_length"], scratch
        )
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
        plan = self._verifier_plan
        if not isinstance(plan, dict):
            return {"status": "failed", "reason": "missing verifier plan"}
        try:
            if plan["kind"] == "json-output":
                return _verify_json_output(plan)
            verifier = Path(plan["path"])
            if hashlib.sha256(verifier.read_bytes()).hexdigest() != plan["sha256"]:
                return {"status": "failed", "reason": "fixed verifier was modified"}
            command = self._verification_command(session, verifier)
            completed = _run_verifier(command)
            expected_stdout = json.dumps(plan["expected"], separators=(",", ":")) + "\n"
            verified = completed.returncode == 0 and completed.stderr == ""
            verified = verified and completed.stdout == expected_stdout
            return {
                "status": "passed" if verified else "failed",
                "kind": "fixed-python-test",
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except BaseException as error:
            return {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }

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

    @staticmethod
    def _verification_command(session, verifier):
        sandbox, limiter = _verification_tools()
        runtime_roots = _verification_runtime_roots()
        command = [
            limiter,
            f"--as={VERIFIER_AS_LIMIT_BYTES}",
            f"--nproc={VERIFIER_NPROC_LIMIT}",
            f"--fsize={VERIFIER_FSIZE_LIMIT_BYTES}",
            f"--cpu={VERIFIER_CPU_LIMIT_SECONDS}",
            "--",
            sandbox,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--clearenv",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for root in runtime_roots:
            command.extend(("--ro-bind", str(root), str(root)))
        loader_cache = Path("/etc/ld.so.cache")
        if loader_cache.exists():
            command.extend(("--ro-bind", str(loader_cache), str(loader_cache)))
        command.extend(
            (
                "--ro-bind",
                str(verifier),
                "/harness/runner.py",
                "--ro-bind",
                str(Path(session["scratch_path"]) / "transform.py"),
                "/work/transform.py",
                "--chdir",
                "/work",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "PATH",
                "/usr/bin:/bin",
                str(Path(sys.executable).resolve()),
                "-I",
                "-B",
                "/harness/runner.py",
                "/work/transform.py",
            )
        )
        return command

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
            record["model_cleanup"] = self.runtime.close()
            record["model_cleanup_end_s"] = time.monotonic() - origin
            if record["model_cleanup_end_s"] > event["cleanup_deadline_s"]:
                raise RuntimeError(
                    "owned model cleanup exceeded the 130s session deadline"
                )
        except BaseException as error:
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


def _hermes_sessions(root):
    sessions = []
    families = (
        "csv-aggregation",
        "json-transformation",
        "python-bugfix",
        "json-transformation",
        "python-bugfix",
        "csv-aggregation",
        "python-bugfix",
        "csv-aggregation",
        "json-transformation",
        "csv-aggregation",
        "json-transformation",
        "python-bugfix",
    )
    token_targets = (1024, 8192, 16384) * 4
    for i in range(1, 13):
        sessions.append(_hermes_session(root, i, families[i - 1], token_targets[i - 1]))
    return sessions


def _hermes_session(root, i, family, target_tokens):
    seed = 2026092000 + i
    records = max(
        32,
        target_tokens
        // {"csv-aggregation": 4, "json-transformation": 18, "python-bugfix": 14}[
            family
        ],
    )
    model, advertised_context = MODEL_PLAN[(i - 1) % len(MODEL_PLAN)]
    scratch = root / "hermes" / f"session-{i:02}" / "scratch"
    prompt = _prompt(family, records, target_tokens, seed, scratch)
    return {
        "session_id": f"session-{i:02}",
        "split": "train" if i <= 6 else "validation" if i <= 9 else "test",
        "seed": seed,
        "model": model,
        "advertised_context_length": advertised_context,
        "path": str(root / "raw" / f"session-{i:02}"),
        "hermes_home": str(root / "hermes" / f"session-{i:02}" / "home"),
        "scratch_path": str(root / "hermes" / f"session-{i:02}" / "scratch"),
        "prompt_path": str(root / "hermes" / f"session-{i:02}" / "prompt.txt"),
        "task": {
            "family": family,
            "fixture_records": records,
            "input_target_tokens": target_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "start_s": 20.0,
            "hard_deadline_s": 120.0,
            "termination_grace_s": SIGTERM_GRACE_SECONDS,
            "cleanup_deadline_s": 130.0,
        },
    }


def _verification_runtime_roots():
    runtime_roots = []
    for root in (
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path(sys.base_prefix).resolve(),
    ):
        if not root.exists() or any(
            root.is_relative_to(bound) for bound in runtime_roots
        ):
            continue
        runtime_roots = [
            bound for bound in runtime_roots if not bound.is_relative_to(root)
        ]
        runtime_roots.append(root)
    return runtime_roots


def _verify_json_output(plan):
    actual = json.loads(Path(plan["path"]).read_text())
    if actual != plan["expected"]:
        return {"status": "failed", "reason": "output mismatch"}
    if isinstance(plan["expected"], dict) and list(actual) != list(plan["expected"]):
        return {"status": "failed", "reason": "output key order mismatch"}
    return {"status": "passed", "kind": "direct-json-comparison"}


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


def _run_verifier(command):
    # Regular files are subject to the sandbox RLIMIT_FSIZE. Never accumulate
    # candidate-controlled outer stdout/stderr in an unbounded host pipe.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, timeout=5, check=False
        )
        captured = []
        for stream in (stdout, stderr):
            stream.seek(0)
            value = stream.read(65537)
            if len(value) > 65536:
                raise RuntimeError("verifier output exceeded 65536 bytes")
            captured.append(value.decode("utf-8", errors="replace"))
    return subprocess.CompletedProcess(command, completed.returncode, *captured)


def _python_fixture(session, scratch, count):
    lines = [
        "Specification: normalize each input word by stripping surrounding whitespace, "
        "convert it to lowercase, discard empty values, and preserve input order."
    ]
    lines.extend(
        f"Example note {i:04}: normalization is deterministic and must not sort values."
        for i in range(1, count)
    )
    (scratch / "specification.txt").write_text("\n".join(lines) + "\n")
    (scratch / "transform.py").write_text(
        "def normalize(words):\n"
        "    return sorted(w.strip().upper() for w in words if w.strip())\n"
    )
    verifier = Path(session["hermes_home"]) / "harness-runner.py"
    verifier.write_bytes(Path(__file__).with_name("verifier_runner.py").read_bytes())
    return {
        "kind": "fixed-python-test",
        "path": str(verifier),
        "sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
        "expected": ["beta", "alpha", "gamma"],
    }


def _task_record(event, started_mono, origin, runtime_metadata):
    return {
        "family": event["family"],
        "prompt_sha256": event["prompt_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_start_s": started_mono - origin,
        "status": "running",
        "runtime": dict(runtime_metadata or {}),
    }


def _verification_tools():
    sandbox = shutil.which("bwrap")
    if sandbox is None:
        raise RuntimeError("bubblewrap is required for Python verification")
    limiter = shutil.which("prlimit")
    if limiter is None:
        raise RuntimeError("prlimit is required for Python verification")
    return sandbox, limiter


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
