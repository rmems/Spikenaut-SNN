"""Predeclared real-sensor acquisition. Stimuli never supply model features."""

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
import time

SHUTDOWN_TIMEOUT_SECONDS = 30
MATRIX_SIZE = 2048
TRANSFER_ELEMENTS = 16 * 1024 * 1024


def allocation_bytes():
    """Three float32 matrices plus one host and one device transfer buffer."""
    return 4 * (3 * MATRIX_SIZE**2 + 2 * TRANSFER_ELEMENTS)


def make_schedule(seed):
    rng = random.Random(seed)
    events, start = [], 20.0
    while start < 130:
        end = min(130.0, start + rng.choice([0.5, 1, 2, 4, 8]))
        events.append(
            {
                "start_s": start,
                "end_s": end,
                "kind": rng.choice(["compute", "transfer", "mixed", "rest"]),
            }
        )
        start = end
    return events


def build_campaign(root):
    root = Path(root).resolve()
    return {
        "schema_version": "anticipation-campaign-v1",
        "feature_map_id": "anticipation-observed-gpu-v1",
        "duration_s": 150,
        "poll_interval_ms": 100,
        "min_examples_per_session": 500,
        "allocation_limit_bytes": 2 * 1024**3,
        "explicit_tensor_bytes": allocation_bytes(),
        "training_budget_seconds": 1200,
        "promising_criterion": {
            "primary_improvement": 0.05,
            "max_target_degradation": 0.05,
        },
        "sessions": [
            {
                "session_id": f"session-{i:02}",
                "split": "train" if i <= 6 else "validation" if i <= 9 else "test",
                "seed": 2026092000 + i,
                "path": str(root / "raw" / f"session-{i:02}"),
                "schedule": make_schedule(2026092000 + i),
            }
            for i in range(1, 13)
        ],
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_until(deadline):
    while time.monotonic() < deadline:
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))


class Stimulus:
    """Fixed reusable buffers, bounded GPU work, no power/thermal control writes."""

    def __init__(self):
        import torch

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; real capture requires the declared GPU stimuli"
            )
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(0.10)
        self.a = torch.empty((MATRIX_SIZE, MATRIX_SIZE), device="cuda")
        self.b = torch.empty_like(self.a)
        self.c = torch.empty_like(self.a)
        self.host = torch.empty(TRANSFER_ELEMENTS, dtype=torch.float32, pin_memory=True)
        self.device = torch.empty_like(self.host, device="cuda")
        if allocation_bytes() + torch.cuda.memory_reserved() > 2 * 1024**3:
            raise RuntimeError("workload allocation budget exceeded")

    def seed(self, seed):
        self.torch.manual_seed(seed)
        self.a.uniform_(-0.1, 0.1)
        self.b.uniform_(-0.1, 0.1)
        self.host.fill_(seed % 17)
        self.device.copy_(self.host)
        self.torch.cuda.synchronize()

    def run(self, event, origin):
        started = time.monotonic() - origin
        iterations = 0
        while time.monotonic() < origin + event["end_s"]:
            if event["kind"] == "rest":
                wait_until(origin + event["end_s"])
                break
            if event["kind"] in ("compute", "mixed"):
                self.torch.mm(self.a, self.b, out=self.c)
            if event["kind"] in ("transfer", "mixed"):
                self.device.copy_(self.host, non_blocking=True)
                self.host.copy_(self.device, non_blocking=True)
            self.torch.cuda.synchronize()
            iterations += 1
        return dict(
            event,
            actual_start_s=started,
            actual_end_s=time.monotonic() - origin,
            iterations=iterations,
        )


def capture(root, collector):
    root, collector = Path(root).resolve(), Path(collector).resolve()
    campaign = build_campaign(root)
    campaign["collector_sha256"] = sha256(collector)
    # Fail before allocating or recording if any capture has already been attempted.
    if (root / "campaign.json").exists():
        raise FileExistsError(
            "campaign already exists; recordings cannot be replaced or resumed silently"
        )
    write_json(root / "campaign.json", campaign)
    status = {
        "schema_version": "capture-status-v1",
        "status": "running",
        "sessions": [],
    }
    write_json(root / "capture-status.json", status)
    active_session = None
    try:
        stimulus = Stimulus()
        for session in campaign["sessions"]:
            active_session = session["session_id"]
            stimulus.seed(session["seed"])
            directory = Path(session["path"])
            directory.mkdir(parents=True, exist_ok=False)
            env = dict(
                os.environ,
                SESSION_DIR=str(directory),
                SESSION_LABEL=session["session_id"],
                WORKLOAD_CLASS="ai-compute",
                POLL_INTERVAL_MS="100",
            )
            actual = []
            started = None
            with (directory / "collector.log").open("w") as log:
                process = subprocess.Popen(
                    [str(collector)], env=env, stdout=log, stderr=log
                )
                try:
                    ready_deadline = time.monotonic() + 15
                    while not (directory / "session_manifest.json").exists():
                        if (
                            process.poll() is not None
                            or time.monotonic() > ready_deadline
                        ):
                            raise RuntimeError("collector failed to initialize")
                        time.sleep(0.02)
                    origin = time.monotonic()
                    started = datetime.now(timezone.utc).isoformat()
                    print(
                        json.dumps(
                            {
                                "session": session["session_id"],
                                "state": "capturing",
                                "started": started,
                            }
                        ),
                        flush=True,
                    )
                    wait_until(origin + 20)
                    for event in session["schedule"]:
                        if process.poll() is not None:
                            raise RuntimeError("collector exited during stimuli")
                        actual.append(stimulus.run(event, origin))
                    wait_until(origin + 150)
                    if process.poll() is not None:
                        raise RuntimeError("collector exited before scheduled shutdown")
                finally:
                    record = {
                        "session_id": session["session_id"],
                        "started_at_utc": started,
                        "actual_schedule": actual,
                        "status": "shutdown_requested",
                    }
                    try:
                        record["peak_cuda_reserved_bytes"] = (
                            stimulus.torch.cuda.max_memory_reserved()
                        )
                        record["peak_cuda_allocated_bytes"] = (
                            stimulus.torch.cuda.max_memory_allocated()
                        )
                        write_json(directory / "stimulus-audit.json", record)
                    finally:
                        # Collector cleanup is mandatory even if diagnostics or disk writes fail.
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                        try:
                            exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            record["collector_exit_code"] = process.wait()
                            record["status"] = "shutdown_timeout"
                            write_json(directory / "stimulus-audit.json", record)
                            raise RuntimeError(
                                "collector failed graceful shutdown; capture incomplete"
                            ) from None
            manifest = json.loads((directory / "session_manifest.json").read_text())
            record["collector_exit_code"] = exit_code
            record["status"] = "collector_stopped"
            write_json(directory / "stimulus-audit.json", record)
            if (
                exit_code
                or not manifest.get("ended_at_utc")
                or manifest.get("parquet_write_failures") != 0
            ):
                raise RuntimeError("collector manifest incomplete or write failures")
            record["status"] = "complete"
            write_json(directory / "stimulus-audit.json", record)
            status["sessions"].append(record)
            write_json(root / "capture-status.json", status)
            print(
                json.dumps({"session": session["session_id"], "state": "complete"}),
                flush=True,
            )
        status["status"] = "complete"
    except BaseException as error:
        status["status"] = "incomplete"
        status["reason"] = f"{type(error).__name__}: {error}"
        status["failed_session_id"] = active_session
        attempted = {s["session_id"] for s in status["sessions"]} | {active_session}
        status["not_attempted_session_ids"] = [
            s["session_id"]
            for s in campaign["sessions"]
            if s["session_id"] not in attempted
        ]
        raise
    finally:
        write_json(root / "capture-status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collector", required=True, type=Path)
    args = parser.parse_args()
    capture(args.output, args.collector)
