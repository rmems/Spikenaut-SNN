"""Shared subprocess and model fixtures for Hermes campaign tests."""

from __future__ import annotations


import os


import subprocess


import time


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

    def close(self, *, deadline=None):
        return {"model": "test", "unloaded": True}


def _write_fake_hermes(path, events, exit_code=0, sleep_seconds=0, ready_path=None):
    source = (
        "#!" + os.sys.executable + "\n"
        "import json,time\n"
        "from pathlib import Path\n"
        f"events={events!r}\n"
        "for event in events:\n print(json.dumps(event), flush=True)\n"
    )
    if ready_path:
        source += f"Path({str(ready_path)!r}).touch()\n"
    source += f"time.sleep({sleep_seconds!r})\nraise SystemExit({exit_code})\n"
    path.write_text(source)
    path.chmod(0o755)


def _wait_for_fixture_ready_before_timeout(monkeypatch, ready_path):
    real_popen = subprocess.Popen

    def synchronized_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while not ready_path.exists():
            if process.poll() is not None:
                raise RuntimeError("fixture exited before signaling readiness")
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise RuntimeError("fixture did not signal readiness")
            time.sleep(0.005)
        return process

    monkeypatch.setattr(subprocess, "Popen", synchronized_popen)
