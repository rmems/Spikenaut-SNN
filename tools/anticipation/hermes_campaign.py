"""Capture sensors during bounded Hermes agent tasks on local Ollama models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from tools.anticipation.campaign import capture


PROTOCOL_ID = "hermes-ollama-inference-v3"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
HARD_TIMEOUT_SECONDS = 100.0
SIGTERM_GRACE_SECONDS = 5.0
PRELOAD_KEEP_ALIVE_SECONDS = 180
PRELOAD_COMPLETION_TIMEOUT_SECONDS = 180
MODEL_PLAN = (
    ("gemma4:12b", 262144),
    ("granite4.2:8b", 131072),
    ("Ornith-1.5-9B:latest", 262144),
)
EXCLUDED_MODELS = {"muse-glimmer:30b", "nemotron-3.5-lightning:30b"}
ALLOWED_TOOLS = {
    "terminal",
    "process_manage",
    "read_file",
    "write_file",
    "patch",
    "search_files",
}


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
        "file or terminal tools to perform the task, verify the output, then answer concisely. "
        f"The synthetic fixture seed is {seed}. "
    )
    if family == "csv-aggregation":
        return common + (
            f"Read all {records} data rows in {scratch / 'input.csv'} "
            f"(approximately {target_tokens} input "
            "tokens), aggregate amount by category, and "
            f"write {scratch / 'output.json'} as an object with alphabetically sorted category "
            f"keys and numeric totals. Run python {scratch / 'verify.py'} to verify the result."
        )
    if family == "json-transformation":
        return common + (
            f"Read all {records} records in {scratch / 'input.json'} "
            f"(approximately {target_tokens} input "
            "tokens), retain enabled records, sort by id, "
            f"and write {scratch / 'output.json'} containing objects with id and score fields. "
            f"Run python {scratch / 'verify.py'} to verify the result."
        )
    return common + (
        f"Read the {records}-line {scratch / 'specification.txt'} "
        f"(approximately {target_tokens} input tokens) plus {scratch / 'transform.py'} and "
        f"{scratch / 'test_transform.py'}. Fix the small bug in {scratch / 'transform.py'} so "
        f"it follows the specification. Do not modify the verifier. Run python "
        f"{scratch / 'test_transform.py'} to verify the fix."
    )


def build_hermes_campaign(root):
    root = Path(root).resolve()
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
        seed = 2026092000 + i
        family = families[i - 1]
        target_tokens = token_targets[i - 1]
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
        sessions.append(
            {
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
        )
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


def _local_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama endpoint must be http://127.0.0.1:<port>")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Ollama endpoint must be http://127.0.0.1:<port>") from error
    return endpoint.rstrip("/")


class OllamaRuntime:
    """Own one preloaded Ollama model and verify its eventual removal."""

    def __init__(
        self,
        endpoint=DEFAULT_ENDPOINT,
        model=DEFAULT_MODEL,
        *,
        preload_timeout_seconds=120,
        request_timeout_seconds=15,
        preload_completion_timeout_seconds=PRELOAD_COMPLETION_TIMEOUT_SECONDS,
        preload_keep_alive_seconds=PRELOAD_KEEP_ALIVE_SECONDS,
    ):
        self.endpoint = _local_endpoint(endpoint)
        self.model = model
        self.preload_timeout_seconds = preload_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.preload_completion_timeout_seconds = preload_completion_timeout_seconds
        self.preload_keep_alive_seconds = preload_keep_alive_seconds
        self._owned_model = None
        self._load_outcome_uncertain = False
        self._preload_thread = None
        self._preload_done = None
        self._preload_outcome = None
        self._preload_deadline = None

    def _request(self, method, path, payload=None, *, timeout=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.endpoint + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(
            request,
            timeout=self.request_timeout_seconds if timeout is None else timeout,
        ) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict):
            raise RuntimeError(f"Ollama {path} returned a non-object response")
        return result

    def _models(self):
        models = self._request("GET", "/api/ps").get("models")
        if not isinstance(models, list):
            raise RuntimeError("Ollama /api/ps response omitted models")
        return models

    @staticmethod
    def _contains_model(models, model):
        return any(
            (entry.get("model") or entry.get("name")) == model for entry in models
        )

    def _unload_exact(self, model):
        self._request("POST", "/api/generate", {"model": model, "keep_alive": 0})
        if self._contains_model(self._models(), model):
            raise RuntimeError(f"Ollama model {model} remained resident after unload")

    def _start_preload(self, model, context_length):
        self._preload_done = threading.Event()
        self._preload_outcome = {}
        self._preload_deadline = (
            time.monotonic() + self.preload_completion_timeout_seconds
        )

        def request_model():
            try:
                response = self._request(
                    "POST",
                    "/api/generate",
                    {
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": f"{self.preload_keep_alive_seconds}s",
                        "options": {"num_ctx": context_length},
                    },
                    timeout=self.preload_completion_timeout_seconds,
                )
                if response.get("done") is not True:
                    raise RuntimeError(
                        "Ollama preload response did not confirm completion"
                    )
                self._preload_outcome["response"] = response
            except BaseException as error:
                self._preload_outcome["error"] = error
            finally:
                self._preload_done.set()

        self._preload_thread = threading.Thread(
            target=request_model,
            name=f"ollama-preload-{model}",
            daemon=False,
        )
        self._preload_thread.start()

    def _wait_for_preload_completion(self):
        if self._preload_thread is None:
            return False
        remaining = max(0.0, self._preload_deadline - time.monotonic())
        self._preload_thread.join(remaining + 0.1)
        return not self._preload_thread.is_alive()

    def prepare(self):
        existing = self._models()
        if existing:
            names = [m.get("name") or m.get("model") or "unknown" for m in existing]
            raise RuntimeError(f"Ollama model already loaded: {', '.join(names)}")
        return {"ollama_version": self._request("GET", "/api/version").get("version")}

    def select(self, model, expected_context_length):
        if model in EXCLUDED_MODELS:
            raise ValueError(f"model is excluded from this campaign: {model}")
        if self._owned_model is not None:
            self.close()
        elif self._models():
            raise RuntimeError(
                "another Ollama model became resident before session setup"
            )
        show = self._request("POST", "/api/show", {"model": model})
        model_info = show.get("model_info")
        if not isinstance(model_info, dict):
            raise RuntimeError(f"Ollama /api/show omitted model_info for {model}")
        architecture = model_info.get("general.architecture")
        context_key = f"{architecture}.context_length" if architecture else None
        context_length = model_info.get(context_key) if context_key else None
        if not isinstance(context_length, int) or context_length <= 0:
            raise RuntimeError(
                f"Ollama /api/show omitted architecture context length for {model}"
            )
        if context_length != expected_context_length:
            raise RuntimeError(
                f"advertised context for {model} is {context_length}, planned {expected_context_length}"
            )
        # The server can complete a load even if its response is lost. Claim only
        # this exact requested identity before the request so later cleanup can
        # reconcile that ambiguous outcome without touching another model.
        self._owned_model = model
        self._load_outcome_uncertain = True
        self._start_preload(model, context_length)
        if not self._preload_done.wait(self.preload_timeout_seconds):
            raise TimeoutError(
                f"Ollama preload timed out after {self.preload_timeout_seconds}s; "
                "owned request continues until cleanup"
            )
        if "error" in self._preload_outcome:
            raise self._preload_outcome["error"]
        self._load_outcome_uncertain = False
        loaded = self._models()
        if len(loaded) != 1:
            raise RuntimeError(
                "Ollama preload did not produce exactly one resident model"
            )
        resident = loaded[0]
        name = resident.get("model") or resident.get("name")
        if name != model:
            raise RuntimeError(f"unexpected resident Ollama model: {name}")
        if resident.get("context_length") != context_length:
            raise RuntimeError(
                f"resident context length is {resident.get('context_length')}, expected {context_length}"
            )
        self.model = model
        size = resident.get("size")
        size_vram = resident.get("size_vram")
        size_cpu = (
            max(0, size - size_vram)
            if isinstance(size, int) and isinstance(size_vram, int)
            else None
        )
        return {
            "model": model,
            "architecture": architecture,
            "advertised_context_length": context_length,
            "digest": resident.get("digest"),
            "quantization": (show.get("details") or {}).get("quantization_level"),
            "preload": {
                "logical_timeout_seconds": self.preload_timeout_seconds,
                "completion_timeout_seconds": self.preload_completion_timeout_seconds,
                "keep_alive_seconds": self.preload_keep_alive_seconds,
            },
            "residency": {
                "size": size,
                "size_vram": size_vram,
                "size_cpu": size_cpu,
                "context_length": resident.get("context_length"),
            },
        }

    def close(self):
        owned_model = self._owned_model
        if owned_model is None:
            return {"model": self.model, "unloaded": False}

        if self._load_outcome_uncertain:
            completed = self._wait_for_preload_completion()
            if not completed or "error" in self._preload_outcome:
                try:
                    self._unload_exact(owned_model)
                except BaseException:
                    pass
                error = self._preload_outcome.get("error")
                message = (
                    f"Ollama model {owned_model} cleanup remains uncertain; "
                    "preload request completion was not confirmed"
                )
                if error is not None:
                    raise RuntimeError(message) from error
                raise RuntimeError(message)

            self._unload_exact(owned_model)
            self._owned_model = None
            self._load_outcome_uncertain = False
            return {"model": owned_model, "unloaded": True}

        try:
            resident = self._models()
        except BaseException:
            resident = None
        if resident is not None and not any(
            (entry.get("model") or entry.get("name")) == owned_model
            for entry in resident
        ):
            self._owned_model = None
            return {"model": owned_model, "unloaded": False}
        self._unload_exact(owned_model)
        self._owned_model = None
        self._load_outcome_uncertain = False
        return {"model": owned_model, "unloaded": True}


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
        self.hermes_executable = Path(hermes_executable)
        self.model = model
        self.endpoint = _local_endpoint(endpoint)
        self.hard_timeout_seconds = float(hard_timeout_seconds)
        self.runtime = runtime or OllamaRuntime(endpoint=self.endpoint, model=model)
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
            "toolsets: [terminal, file]\n"
            "terminal:\n"
            f"  cwd: {scratch}\n"
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
            (scratch / "test_transform.py").write_text(
                "from transform import normalize\n"
                "assert normalize([' Beta ', '', 'ALPHA', ' gamma ']) == ['beta', 'alpha', 'gamma']\n"
                "print('ok')\n"
            )
            verifier = scratch / "test_transform.py"
            return {
                "kind": "fixed-python-test",
                "path": str(verifier),
                "sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
            }
        (scratch / "verify.py").write_text(
            "import json\nfrom pathlib import Path\n"
            f"expected={expected!r}\n"
            "actual=json.loads(Path('output.json').read_text())\n"
            "assert actual == expected, (actual, expected)\nprint('ok')\n"
        )
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
            "terminal,file",
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
                resolved = Path(value)
                if not resolved.is_absolute():
                    resolved = scratch / resolved
                resolved = resolved.resolve()
                in_scope = resolved == scratch or scratch in resolved.parents
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
                actual = json.loads(Path(plan["path"]).read_text())
                if actual != plan["expected"]:
                    return {"status": "failed", "reason": "output mismatch"}
                return {"status": "passed", "kind": "direct-json-comparison"}
            verifier = Path(plan["path"])
            if hashlib.sha256(verifier.read_bytes()).hexdigest() != plan["sha256"]:
                return {"status": "failed", "reason": "fixed verifier was modified"}
            completed = subprocess.run(
                [sys.executable, str(verifier)],
                cwd=session["scratch_path"],
                env=self.environment(session),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
            return {
                "status": "passed" if completed.returncode == 0 else "failed",
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
        record = {
            "family": event["family"],
            "prompt_sha256": event["prompt_sha256"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "actual_start_s": started_mono - origin,
            "status": "running",
            "runtime": dict(self._runtime_metadata or {}),
        }
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
            remaining = origin + event["hard_deadline_s"] - time.monotonic()
            timeout = min(self.hard_timeout_seconds, max(0.001, remaining))
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timeboxed = True
                record["parent_stop_reason"] = "100s_agent_timebox"
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

            ended_mono = time.monotonic()
            self._write_process_logs(session, stdout, stderr, record)
            events, diagnostics = self._events(stdout)
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
            result = next(
                (e for e in reversed(events) if e.get("type") == "result"), None
            )
            init = next(
                (
                    e
                    for e in events
                    if e.get("type") == "system" and e.get("subtype") == "init"
                ),
                {},
            )
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
            invalid_tools = [
                tool_event
                for tool_event in (*tool_calls, *tool_results)
                if not isinstance(tool_event["name"], str)
                or not tool_event["name"].strip()
                or tool_event["name"] not in ALLOWED_TOOLS
            ]
            if not tool_calls:
                raise RuntimeError(
                    "Hermes task completed without an observed tool call"
                )
            if invalid_tools:
                raise RuntimeError("Hermes reported an unsupported or empty tool name")
            if init.get("model") != self.model:
                raise RuntimeError("Hermes did not report the configured local model")
            if result is None:
                raise RuntimeError("Hermes stream ended without a terminal result")
            if forced_kill:
                raise RuntimeError("Hermes timebox required SIGKILL")
            result_error = result.get("error")
            parent_interrupt = timeboxed and result_error == "Interrupted"
            if result_error and not parent_interrupt:
                raise RuntimeError("Hermes terminal result reported an explicit error")
            if not timeboxed and not self._positive_usage(result.get("tokens")):
                raise RuntimeError(
                    "Hermes terminal result omitted positive token usage"
                )

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
                successful_exit = (
                    process.returncode == 0 and result.get("exit_code") == 0
                )
                record["task_outcome"] = (
                    "verified_complete"
                    if successful_exit
                    and record["verifier"]["status"] == "passed"
                    and not outside
                    else "incomplete"
                )
        except BaseException as error:
            pending_error = error
            record["status"] = "invalid"
            record["workload_status"] = "invalid"
            record.setdefault("task_outcome", "unknown")
            record.setdefault("error", f"{type(error).__name__}: {error}")
        finally:
            if process is not None and process.poll() is None:
                _signal_process_group(process, signal.SIGKILL)
                process.wait()
                forced_kill = True
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
