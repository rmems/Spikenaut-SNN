"""Run baselines, Julia training, and evaluation under one shared time budget."""

import argparse
import json
from pathlib import Path
import subprocess
import time
from .campaign import sha256, write_json
from .report import baselines, comparison


def unfinished_summary(prepared, snn, reason):
    runs = []
    for seed in (123, 456, 789):
        for arm in ("uniform", "mixed"):
            checkpoint = snn / f"{arm}-{seed}" / "checkpoint.json"
            run = {
                "arm": arm,
                "seed": seed,
                "status": "unfinished",
                "reason": reason,
                "checkpoint": None,
                "predictions": None,
            }
            if checkpoint.exists():
                run["checkpoint"] = str(checkpoint.relative_to(snn))
                run["status"] = "unfinished_evaluation"
            runs.append(run)
    return {
        "schema_version": "anticipation-snn-runs-v1",
        "prepared_sha256": sha256(prepared),
        "runs": runs,
    }


def evaluate(
    prepared, output, *, julia="julia", julia_version=None, budget_seconds=1200
):
    if not 0 <= budget_seconds <= 1200:
        raise ValueError("pilot evaluation budget must be between 0 and 1200 seconds")
    started = time.monotonic()
    prepared, output = Path(prepared), Path(output)
    snn = output / "snn"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "evaluation already attempted; never extend or replace silently"
        )
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": "anticipation-budget-v1",
        "budget_seconds": budget_seconds,
        "status": "incomplete",
        "reason": None,
    }
    baseline_results = None
    try:
        if time.monotonic() - started >= budget_seconds:
            status["reason"] = "budget_exhausted_before_baselines"
            return status
        data = json.loads(prepared.read_text())
        baseline_results = baselines(data, output)
        remaining = max(0, budget_seconds - (time.monotonic() - started))
        status["baseline_elapsed_seconds"] = time.monotonic() - started
        if remaining <= 0:
            status["reason"] = "budget_exhausted_after_baselines"
            return status
        grace = min(30.0, remaining * 0.15)
        trainer_budget = remaining - grace
        status["trainer_budget_seconds"] = trainer_budget
        status["trainer_timeout_seconds"] = remaining
        command = [julia]
        if julia_version:
            command.append("+" + julia_version)
        command += [
            "--startup-file=no",
            str(Path(__file__).with_name("train.jl")),
            str(prepared),
            str(snn),
            str(trainer_budget),
        ]
        status["trainer_command"] = command
        with (output / "training.log").open("w") as log:
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=log, timeout=remaining, check=False
                )
                status["trainer_exit_code"] = process.returncode
                if process.returncode:
                    status["reason"] = "trainer_failed"
            except subprocess.TimeoutExpired:
                status["reason"] = "shared_budget_exhausted"
        if status["reason"] is None:
            status["status"] = "complete"
    except Exception as error:
        status["reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        status["compute_elapsed_seconds"] = time.monotonic() - started
        try:
            if not (snn / "summary.json").exists():
                write_json(
                    snn / "summary.json",
                    unfinished_summary(prepared, snn, status["reason"]),
                )
            if baseline_results is not None:
                result = comparison(prepared, output, baseline_results=baseline_results)
                if not result["complete"]:
                    status["status"] = "incomplete"
                    status["reason"] = status["reason"] or "unfinished_comparisons"
        except Exception as error:
            status["status"] = "incomplete"
            status["finalization_error"] = f"{type(error).__name__}: {error}"
            status["reason"] = status["reason"] or status["finalization_error"]
            raise
        finally:
            status["total_elapsed_seconds"] = time.monotonic() - started
            write_json(output / "budget-report.json", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--julia-version")
    parser.add_argument("--budget-seconds", type=float, default=1200)
    args = parser.parse_args()
    result = evaluate(
        args.prepared,
        args.output,
        julia=args.julia,
        julia_version=args.julia_version,
        budget_seconds=args.budget_seconds,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)
