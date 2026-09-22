"""Predeclared Hermes model and task matrix."""

from pathlib import Path
import hashlib

PROTOCOL_ID = "hermes-ollama-inference-v4"
SIGTERM_GRACE_SECONDS = 5.0
PRELOAD_KEEP_ALIVE_SECONDS = 180
PRELOAD_COMPLETION_TIMEOUT_SECONDS = 180
EXCLUDED_MODELS = {"muse-glimmer:30b", "nemotron-3.5-lightning:30b"}
MODEL_PLAN = (
    ("gemma4:12b", 262144),
    ("granite4.2:8b", 131072),
    ("Ornith-1.5-9B:latest", 262144),
)


def _session_split(index):
    if index <= 6:
        return "train"
    if index <= 9:
        return "validation"
    return "test"


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
        "Keep normalize as a pure function using string/list operations, loops or "
        "comprehensions; do not add imports, introspection, I/O, or process control. "
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
        "split": _session_split(i),
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
