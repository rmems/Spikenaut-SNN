"""Hermes runtime regression tests."""

from __future__ import annotations


import json


import time


import pytest


from tests.ollama_fixture import ollama_server


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
        assert metadata["preload"] == {
            "logical_timeout_seconds": 120,
            "completion_timeout_seconds": 180,
            "keep_alive_seconds": 180,
        }
        preload = next(
            request[2]
            for request in state["requests"]
            if request[0:2] == ("POST", "/api/generate")
            and request[2].get("keep_alive") != 0
        )
        assert preload["keep_alive"] == "180s"
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
        with pytest.raises(TimeoutError, match="timed out"):
            runtime.select("gemma4:12b", 262144)
        assert state["loaded"] is True
        cleanup = runtime.close()
        assert cleanup == {"model": "gemma4:12b", "unloaded": True}
        assert state["loaded"] is False


def test_runtime_joins_owned_preload_after_logical_timeout():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_visibility_delay=0.12) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.02,
            preload_completion_timeout_seconds=0.5,
        )
        runtime.prepare()
        with pytest.raises(TimeoutError, match="timed out"):
            runtime.select("gemma4:12b", 262144)

        assert state["loaded"] is False
        cleanup = runtime.close()

        assert cleanup == {"model": "gemma4:12b", "unloaded": True}
        assert state["loaded"] is False
        unloads = [
            request
            for request in state["requests"]
            if request[0:2] == ("POST", "/api/generate")
            and request[2].get("keep_alive") == 0
        ]
        assert [request[2]["model"] for request in unloads] == ["gemma4:12b"]


def test_capture_waits_for_owned_late_preload_and_returns_with_model_absent(tmp_path):
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
            preload_completion_timeout_seconds=0.5,
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        assert time.monotonic() - started >= 0.15
        assert state["loaded"] is False
        status = json.loads((root / "capture-status.json").read_text())
        assert status["status"] == "incomplete"
        assert status["stimulus_cleanup"] == {
            "model": "gemma4:12b",
            "unloaded": True,
        }
        preload = next(
            request[2]
            for request in state["requests"]
            if request[0:2] == ("POST", "/api/generate")
            and request[2].get("keep_alive") != 0
        )
        assert preload["keep_alive"] == "180s"


def test_capture_reconciles_load_after_preload_request_timeout(tmp_path):
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
            preload_completion_timeout_seconds=0.05,
            preload_keep_alive_seconds=0.1,
            cleanup_reconciliation_timeout_seconds=0.5,
        )
        started = time.monotonic()

        with pytest.raises(TimeoutError, match="timed out"):
            capture(
                root,
                collector,
                campaign=plan,
                stimulus_factory=lambda: HermesStimulus(
                    root, hermes_executable=hermes, runtime=runtime
                ),
            )

        assert time.monotonic() - started >= 0.15
        assert state["loaded"] is False
        status = json.loads((root / "capture-status.json").read_text())
        assert status["stimulus_cleanup"] == {
            "model": "gemma4:12b",
            "unloaded": True,
        }
        unloads = [
            request
            for request in state["requests"]
            if request[0:2] == ("POST", "/api/generate")
            and request[2].get("keep_alive") == 0
        ]
        assert unloads


def test_preload_transport_has_end_to_end_deadline_despite_drip_response():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_drip_interval=0.01) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.2,
            request_timeout_seconds=0.02,
            preload_completion_timeout_seconds=0.03,
            cleanup_reconciliation_timeout_seconds=0.2,
        )
        runtime.prepare()
        started = time.monotonic()

        with pytest.raises(TimeoutError, match="end-to-end deadline"):
            runtime.select("granite4.2:8b", 131072)

        assert time.monotonic() - started < 0.5
        cleanup = runtime.close()
        assert cleanup == {"model": "granite4.2:8b", "unloaded": True}
        assert state["loaded"] is False


def test_preload_transport_deadline_cancels_dripping_response_headers():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_header_drip_interval=0.01) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.2,
            request_timeout_seconds=0.02,
            preload_completion_timeout_seconds=0.03,
            cleanup_reconciliation_timeout_seconds=0.2,
        )
        runtime.prepare()
        started = time.monotonic()

        with pytest.raises(TimeoutError, match="end-to-end deadline"):
            runtime.select("granite4.2:8b", 131072)

        assert time.monotonic() - started < 0.5
        cleanup = runtime.close()
        assert cleanup == {"model": "granite4.2:8b", "unloaded": True}
        assert state["loaded"] is False


def test_prepare_version_request_has_an_end_to_end_deadline():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(version_drip_interval=0.01) as (endpoint, _):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            request_timeout_seconds=0.02,
            control_plane_timeout_seconds=0.05,
        )

        with pytest.raises(TimeoutError, match="end-to-end deadline"):
            runtime.prepare()


@pytest.mark.parametrize("stage", ["preflight", "show", "postload"])
def test_select_control_plane_requests_have_end_to_end_deadlines(stage: str):
    from tools.anticipation.hermes_campaign import OllamaRuntime

    options = {}
    if stage == "show":
        options["show_drip_interval"] = 0.01
    elif stage == "postload":
        options["ps_drip_interval_after_load"] = 0.01

    with ollama_server(**options) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            request_timeout_seconds=0.02,
            control_plane_timeout_seconds=0.05,
        )
        runtime.prepare()
        if stage == "preflight":
            state["ps_drip_interval_before_load"] = 0.01

        with pytest.raises(TimeoutError, match="end-to-end deadline"):
            runtime.select("granite4.2:8b", 131072)

        if stage == "postload":
            state["ps_drip_interval_after_load"] = 0
            assert runtime.close() == {"model": "granite4.2:8b", "unloaded": True}


def test_runtime_slow_failed_preload_returns_bounded_and_finishes_cleanup():
    from tools.anticipation.hermes_campaign import OllamaRuntime

    with ollama_server(preload_never_completes_delay=0.3) as (endpoint, state):
        runtime = OllamaRuntime(
            endpoint=endpoint,
            preload_timeout_seconds=0.01,
            preload_completion_timeout_seconds=0.05,
            cleanup_reconciliation_timeout_seconds=0.05,
        )
        runtime.prepare()
        with pytest.raises(TimeoutError, match="timed out"):
            runtime.select("gemma4:12b", 262144)

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
            runtime.close()
        assert time.monotonic() - started < 0.2
        runtime._cleanup_thread.join(1)
        assert runtime._preload_thread.is_alive() is False
        assert runtime._cleanup_thread.is_alive() is False
        assert runtime._cleanup_error is None
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
        with pytest.raises(RuntimeError, match="unexpected Ollama models"):
            runtime.close()
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
        if server_options.get("extra_resident"):
            with pytest.raises(RuntimeError, match="unexpected Ollama models"):
                runtime.close()
        else:
            cleanup = runtime.close()
            assert cleanup == {"model": "granite4.2:8b", "unloaded": True}
        assert state["loaded"] is False
