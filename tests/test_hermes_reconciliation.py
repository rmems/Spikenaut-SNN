"""Owned-model reconciliation regressions."""

from __future__ import annotations


import json


import time


import pytest


from tests.ollama_fixture import ollama_server


def test_durable_cleanup_outlives_reconciliation_deadline(tmp_path):
    from tools.anticipation.campaign import capture
    from tools.anticipation.hermes_campaign import (
        HermesStimulus,
        OllamaRuntime,
        build_hermes_campaign,
    )

    collector = tmp_path / "collector"
    collector.write_bytes(b"not started because preload fails")
    hermes = tmp_path / "hermes"
    hermes.write_text("not started because preload fails")
    root = tmp_path / "capture"
    plan = build_hermes_campaign(root)
    plan["sessions"] = plan["sessions"][:1]

    with ollama_server(preload_visibility_delay=0.2) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            preload_completion_timeout_seconds=0.02,
            preload_keep_alive_seconds=0.1,
            cleanup_reconciliation_timeout_seconds=0.05,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        status = json.loads((root / "capture-status.json").read_text())
        assert "cleanup remains uncertain" in status["cleanup_error"]
        runtime._cleanup_thread.join(1)
        assert runtime._cleanup_thread.is_alive() is False
        assert runtime._cleanup_error is None
        assert state["loaded"] is False


def test_durable_cleanup_retries_transient_status_failure(tmp_path):
    from tools.anticipation.campaign import capture
    from tools.anticipation.hermes_campaign import (
        HermesStimulus,
        OllamaRuntime,
        build_hermes_campaign,
    )

    collector = tmp_path / "collector"
    collector.write_bytes(b"not started because preload fails")
    hermes = tmp_path / "hermes"
    hermes.write_text("not started because preload fails")
    root = tmp_path / "capture"
    plan = build_hermes_campaign(root)
    plan["sessions"] = plan["sessions"][:1]

    with ollama_server(preload_visibility_delay=0.2, ps_failures_after_load=1) as (
        endpoint,
        state,
    ):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            preload_completion_timeout_seconds=0.02,
            preload_keep_alive_seconds=0.1,
            cleanup_reconciliation_timeout_seconds=0.05,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        status = json.loads((root / "capture-status.json").read_text())
        assert "cleanup remains uncertain" in status["cleanup_error"]
        runtime._cleanup_thread.join(1)
        assert runtime._cleanup_thread.is_alive() is False
        assert runtime._cleanup_error is None
        assert state["loaded"] is False
        report = json.loads((root / "ollama-cleanup.json").read_text())
        assert report["state"] == "confirmed_absent"
        assert report["final_error"] is None
        unloads = [
            request
            for request in state["requests"]
            if request[0:2] == ("POST", "/api/generate")
            and request[2].get("keep_alive") == 0
        ]
        assert len(unloads) == 1


def test_durable_cleanup_rejects_unowned_resident_model(tmp_path):
    from tools.anticipation.campaign import capture
    from tools.anticipation.hermes_campaign import (
        HermesStimulus,
        OllamaRuntime,
        build_hermes_campaign,
    )

    collector = tmp_path / "collector"
    collector.write_bytes(b"not started because preload fails")
    hermes = tmp_path / "hermes"
    hermes.write_text("not started because preload fails")
    root = tmp_path / "capture"
    plan = build_hermes_campaign(root)
    plan["sessions"] = plan["sessions"][:1]

    with ollama_server(
        preload_visibility_delay=0.2, post_load_model="unowned:latest"
    ) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            preload_completion_timeout_seconds=0.02,
            cleanup_reconciliation_timeout_seconds=0.05,
            durable_cleanup_timeout_seconds=0.1,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        runtime._cleanup_thread.join(1)
        assert runtime._cleanup_thread.is_alive() is False
        assert "unexpected Ollama models" in str(runtime._cleanup_error)
        assert state["loaded"] is True
        report = json.loads((root / "ollama-cleanup.json").read_text())
        assert report["state"] == "terminal_failure"
        assert "unowned:latest" in report["final_error"]
        assert not any(
            request[2] and request[2].get("keep_alive") == 0
            for request in state["requests"]
        )


def test_durable_cleanup_records_terminal_failure_after_retry_deadline(tmp_path):
    from tools.anticipation.campaign import capture
    from tools.anticipation.hermes_campaign import (
        HermesStimulus,
        OllamaRuntime,
        build_hermes_campaign,
    )

    collector = tmp_path / "collector"
    collector.write_bytes(b"not started because preload fails")
    hermes = tmp_path / "hermes"
    hermes.write_text("not started because preload fails")
    root = tmp_path / "capture"
    plan = build_hermes_campaign(root)
    plan["sessions"] = plan["sessions"][:1]

    with ollama_server(preload_visibility_delay=0.2, unload_sticks=True) as (
        endpoint,
        state,
    ):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            preload_completion_timeout_seconds=0.02,
            cleanup_reconciliation_timeout_seconds=0.05,
            durable_cleanup_timeout_seconds=0.1,
        )
        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        runtime._cleanup_thread.join(1)
        assert runtime._cleanup_thread.is_alive() is False
        assert "durable cleanup could not confirm absence" in str(
            runtime._cleanup_error
        )
        assert state["loaded"] is True
        report = json.loads((root / "ollama-cleanup.json").read_text())
        assert report["model"] == "gemma4:12b"
        assert report["state"] == "terminal_failure"
        assert report["retry_timeout_seconds"] == 0.1
        assert report["retry_deadline_utc"]
        assert "remained resident after unload" in report["final_error"]


def test_durable_cleanup_bounds_each_dripping_status_request(tmp_path):
    from tools.anticipation.campaign import capture
    from tools.anticipation.hermes_campaign import (
        HermesStimulus,
        OllamaRuntime,
        build_hermes_campaign,
    )

    collector = tmp_path / "collector"
    collector.write_bytes(b"not started because preload fails")
    hermes = tmp_path / "hermes"
    hermes.write_text("not started because preload fails")
    root = tmp_path / "capture"
    plan = build_hermes_campaign(root)
    plan["sessions"] = plan["sessions"][:1]

    with ollama_server(
        preload_visibility_delay=0.05, ps_drip_interval_after_load=0.01
    ) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            request_timeout_seconds=0.02,
            preload_completion_timeout_seconds=0.02,
            cleanup_reconciliation_timeout_seconds=0.03,
            durable_cleanup_timeout_seconds=0.1,
        )
        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        runtime._cleanup_thread.join(0.5)
        assert runtime._cleanup_thread.is_alive() is False
        assert "durable cleanup could not confirm absence" in str(
            runtime._cleanup_error
        )
        report = json.loads((root / "ollama-cleanup.json").read_text())
        assert report["state"] == "terminal_failure"
        assert "end-to-end deadline" in report["final_error"]
        assert state["loaded"] is True


def test_normal_cleanup_bounds_a_dripping_status_request():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server() as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            request_timeout_seconds=0.02,
            durable_cleanup_timeout_seconds=0.1,
        )
        runtime.prepare()
        runtime.select("granite4.2:8b", 131072)
        state["ps_drip_interval_after_load"] = 0.01
        started = time.monotonic()

        with pytest.raises(TimeoutError, match="end-to-end deadline"):
            runtime.close()

        assert time.monotonic() - started < 0.5
        assert state["loaded"] is True
        state["ps_drip_interval_after_load"] = 0
        assert runtime.close() == {"model": "granite4.2:8b", "unloaded": True}


def test_cleanup_deadline_starts_after_waiting_for_late_preload_completion():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_response_delay=0.08) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            preload_completion_timeout_seconds=0.2,
            request_timeout_seconds=0.02,
            durable_cleanup_timeout_seconds=0.03,
        )
        runtime.prepare()
        with pytest.raises(TimeoutError, match="timed out"):
            runtime.select("granite4.2:8b", 131072)

        assert runtime.close() == {"model": "granite4.2:8b", "unloaded": True}
        assert state["loaded"] is False


def test_preload_transport_timeout_bounds_durable_cleanup():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_response_delay=0.3) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            request_timeout_seconds=0.03,
            preload_completion_timeout_seconds=0.02,
            cleanup_reconciliation_timeout_seconds=0.1,
            durable_cleanup_timeout_seconds=0.1,
        )
        runtime.prepare()
        with pytest.raises(TimeoutError, match="timed out"):
            runtime.select("gemma4:12b", 262144)

        started = time.monotonic()
        cleanup = runtime.close()

        assert time.monotonic() - started < 0.2
        assert cleanup == {"model": "gemma4:12b", "unloaded": True}
        assert state["loaded"] is False
        time.sleep(0.1)
        assert runtime._preload_thread.is_alive() is False


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_cleanup_report_preserves_staging_link_target(tmp_path, link_kind):
    from tools.anticipation.hermes_campaign import OllamaRuntime

    sentinel = tmp_path / "source.json"
    sentinel.write_text("source sentinel")
    staging = tmp_path / "ollama-cleanup.json.tmp"
    if link_kind == "symlink":
        staging.symlink_to(sentinel)
    else:
        staging.hardlink_to(sentinel)
    runtime = OllamaRuntime(cleanup_report_path=tmp_path / "ollama-cleanup.json")
    runtime._write_cleanup_report("model", "confirmed_absent")
    assert sentinel.read_text() == "source sentinel"
    assert (
        json.loads((tmp_path / "ollama-cleanup.json").read_text())["state"]
        == "confirmed_absent"
    )
