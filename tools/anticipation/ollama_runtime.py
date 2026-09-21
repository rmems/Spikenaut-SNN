"""Deadline-bounded ownership and cleanup of local Ollama models."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from .campaign import write_json

from .ollama_http import DeadlineHTTPClient, _DEFAULT_REQUEST_TIMEOUT

DEFAULT_MODEL = "gemma4:12b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
PRELOAD_KEEP_ALIVE_SECONDS = 180
PRELOAD_COMPLETION_TIMEOUT_SECONDS = 180
DURABLE_CLEANUP_TIMEOUT_SECONDS = 30
DURABLE_CLEANUP_INITIAL_BACKOFF_SECONDS = 0.05
DURABLE_CLEANUP_MAX_BACKOFF_SECONDS = 1.0
CONTROL_PLANE_TIMEOUT_SECONDS = 30
EXCLUDED_MODELS = {"muse-glimmer:30b", "nemotron-3.5-lightning:30b"}


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
    _validate_port(parsed)
    return endpoint.rstrip("/")


def _validate_port(parsed):
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Ollama endpoint must be http://127.0.0.1:<port>") from error


@dataclass
class RuntimeTimeouts:
    preload_timeout_seconds: float | None = 120
    request_timeout_seconds: float | None = 15
    preload_completion_timeout_seconds: float | None = (
        PRELOAD_COMPLETION_TIMEOUT_SECONDS
    )
    preload_keep_alive_seconds: float | None = PRELOAD_KEEP_ALIVE_SECONDS
    control_plane_timeout_seconds: float | None = CONTROL_PLANE_TIMEOUT_SECONDS
    cleanup_reconciliation_timeout_seconds: float | None = None
    durable_cleanup_timeout_seconds: float | None = DURABLE_CLEANUP_TIMEOUT_SECONDS


class OllamaRuntime:
    """Own one preloaded Ollama model and verify its eventual removal."""

    def __init__(
        self,
        endpoint=DEFAULT_ENDPOINT,
        model=DEFAULT_MODEL,
        *,
        cleanup_report_path=None,
        **timeout_options,
    ):
        options = RuntimeTimeouts(**timeout_options)
        self.endpoint = _local_endpoint(endpoint)
        self.model = model
        self.preload_timeout_seconds = options.preload_timeout_seconds
        self.request_timeout_seconds = options.request_timeout_seconds
        self.preload_completion_timeout_seconds = (
            options.preload_completion_timeout_seconds
        )
        self.preload_keep_alive_seconds = options.preload_keep_alive_seconds
        self.control_plane_timeout_seconds = options.control_plane_timeout_seconds
        self.cleanup_reconciliation_timeout_seconds = (
            max(
                options.preload_completion_timeout_seconds,
                options.preload_keep_alive_seconds,
            )
            + options.request_timeout_seconds
            if options.cleanup_reconciliation_timeout_seconds is None
            else options.cleanup_reconciliation_timeout_seconds
        )
        self.durable_cleanup_timeout_seconds = options.durable_cleanup_timeout_seconds
        self._cleanup_report_path = (
            Path(cleanup_report_path).resolve()
            if cleanup_report_path is not None
            else None
        )
        self._owned_model = None
        self._load_outcome_uncertain = False
        self._preload_thread = None
        self._preload_done = None
        self._preload_outcome = None
        self._preload_deadline = None
        self._preload_transport_deadline = None
        self._cleanup_thread = None
        self._cleanup_error = None

    def set_cleanup_report_path(self, path):
        self._cleanup_report_path = Path(path).resolve()

    def _write_cleanup_report(
        self, model, state, *, retry_deadline_utc=None, final_error=None
    ):
        if self._cleanup_report_path is None:
            return
        report = {
            "schema_version": "ollama-cleanup-v1",
            "model": model,
            "state": state,
            "retry_timeout_seconds": self.durable_cleanup_timeout_seconds,
            "retry_deadline_utc": retry_deadline_utc,
            "final_error": final_error,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            write_json(self._cleanup_report_path, report)
        except OSError as error:
            self._cleanup_error = RuntimeError(
                f"could not persist Ollama cleanup report: {error}"
            )

    def _request(
        self,
        method,
        path,
        payload=None,
        *,
        timeout=_DEFAULT_REQUEST_TIMEOUT,
        deadline_seconds=None,
    ):
        client = DeadlineHTTPClient(self.endpoint, self.request_timeout_seconds)
        return client.request(
            method, path, payload, timeout=timeout, deadline_seconds=deadline_seconds
        )

    @staticmethod
    def _remaining(deadline):
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Ollama request exceeded its end-to-end deadline")
        return remaining

    def _models(self, deadline=None):
        models = self._request(
            "GET", "/api/ps", deadline_seconds=self._remaining(deadline)
        ).get("models")
        if not isinstance(models, list):
            raise RuntimeError("Ollama /api/ps response omitted models")
        return models

    @staticmethod
    def _contains_model(models, model):
        return any(
            (entry.get("model") or entry.get("name")) == model for entry in models
        )

    def _unload_exact(self, model, deadline=None):
        self._request(
            "POST",
            "/api/generate",
            {"model": model, "keep_alive": 0},
            deadline_seconds=self._remaining(deadline),
        )
        remaining = self._models(deadline)
        if self._contains_model(remaining, model):
            raise RuntimeError(f"Ollama model {model} remained resident after unload")
        if remaining:
            names = [entry.get("model") or entry.get("name") for entry in remaining]
            raise RuntimeError(
                "unexpected Ollama models remained resident after owned-model cleanup: "
                + ", ".join(str(name) for name in names)
            )

    def _start_preload(self, model, context_length):
        self._preload_done = threading.Event()
        self._preload_outcome = {}
        self._preload_deadline = (
            time.monotonic() + self.preload_completion_timeout_seconds
        )
        transport_timeout = (
            self.preload_completion_timeout_seconds + self.request_timeout_seconds
        )
        self._preload_transport_deadline = time.monotonic() + transport_timeout

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
                    # Outlive the logical deadline long enough to observe a
                    # normal server completion, but keep the non-daemon worker
                    # bounded if Ollama never finishes the response.
                    timeout=transport_timeout,
                    deadline_seconds=transport_timeout,
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
            daemon=True,
        )
        self._preload_thread.start()

    def _wait_for_preload_completion(self):
        if self._preload_thread is None:
            return False
        remaining = max(0.0, self._preload_deadline - time.monotonic())
        self._preload_thread.join(remaining + 0.1)
        return not self._preload_thread.is_alive()

    def _reconcile_uncertain_load(self, model):
        """Wait for a timed-out server load, then remove the exact owned model."""
        deadline = time.monotonic() + self.cleanup_reconciliation_timeout_seconds
        last_error = None
        while True:
            try:
                resident = self._models(deadline)
                if self._contains_model(resident, model):
                    self._unload_exact(model, deadline)
                    return
            except BaseException as error:
                last_error = error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))

        message = (
            f"Ollama model {model} cleanup remains uncertain; "
            "timed-out preload never became observable during reconciliation"
        )
        self._start_durable_cleanup(model)
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def _start_durable_cleanup(self, model):
        """Keep the process alive until the accepted preload can be reconciled."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        self._cleanup_error = None
        self._write_cleanup_report(model, "waiting_for_preload")

        self._cleanup_thread = threading.Thread(
            target=self._finish_cleanup,
            args=(model,),
            name=f"ollama-cleanup-{model}",
            daemon=False,
        )
        self._cleanup_thread.start()

    def prepare(self):
        deadline = time.monotonic() + self.control_plane_timeout_seconds
        existing = self._models(deadline)
        if existing:
            names = [m.get("name") or m.get("model") or "unknown" for m in existing]
            raise RuntimeError(f"Ollama model already loaded: {', '.join(names)}")
        version = self._request(
            "GET", "/api/version", deadline_seconds=self._remaining(deadline)
        )
        return {"ollama_version": version.get("version")}

    def select(self, model, expected_context_length):
        if model in EXCLUDED_MODELS:
            raise ValueError(f"model is excluded from this campaign: {model}")
        if self._owned_model is not None:
            self.close()
        deadline = time.monotonic() + self.control_plane_timeout_seconds
        if self._models(deadline):
            raise RuntimeError(
                "another Ollama model became resident before session setup"
            )
        show = self._request(
            "POST",
            "/api/show",
            {"model": model},
            deadline_seconds=self._remaining(deadline),
        )
        architecture, context_length = self._advertised_context(
            show, model, expected_context_length
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
        resident = self._resident_model(model, context_length)
        self.model = model
        return self._selection_metadata(
            model, architecture, context_length, resident, show
        )

    def close(self, *, deadline=None):
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            raise TimeoutError(
                "owned model cleanup remains deferred to supervised worker"
            )
        owned_model = self._owned_model
        if owned_model is None:
            return {"model": self.model, "unloaded": False}
        if self._load_outcome_uncertain:
            return self._close_uncertain(owned_model, deadline=deadline)
        cleanup_deadline = time.monotonic() + self.durable_cleanup_timeout_seconds
        if deadline is None:
            return self._close_loaded(owned_model, cleanup_deadline)
        return self._close_session_model(owned_model, min(deadline, cleanup_deadline))

    def _close_session_model(self, owned_model, deadline):
        try:
            return self._close_loaded(owned_model, deadline)
        except TimeoutError:
            # Return control so the collector can stop on its acquisition clock;
            # retain ownership and reconcile the model in the supervised worker.
            self._start_durable_cleanup(owned_model)
            raise

    def _close_loaded(self, owned_model, cleanup_deadline):
        try:
            resident = self._models(cleanup_deadline)
        except BaseException:
            resident = None
        if resident is not None and not self._contains_model(resident, owned_model):
            self._reject_unowned_residents(resident)
            self._owned_model = None
            return {"model": owned_model, "unloaded": False}
        self._unload_exact(owned_model, cleanup_deadline)
        self._owned_model = None
        self._load_outcome_uncertain = False
        return {"model": owned_model, "unloaded": True}

    @staticmethod
    def _advertised_context(show, model, expected_context_length):
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
        return architecture, context_length

    def _resident_model(self, model, context_length):
        postload_deadline = time.monotonic() + self.control_plane_timeout_seconds
        loaded = self._models(postload_deadline)
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
        return resident

    def _selection_metadata(self, model, architecture, context_length, resident, show):
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

    def _finish_cleanup(self, model):
        transport_remaining = max(
            0.0, self._preload_transport_deadline - time.monotonic()
        )
        self._preload_thread.join(transport_remaining + 0.1)
        deadline = time.monotonic() + self.durable_cleanup_timeout_seconds
        retry_deadline_utc = (
            datetime.now(timezone.utc)
            + timedelta(seconds=self.durable_cleanup_timeout_seconds)
        ).isoformat()
        self._write_cleanup_report(
            model, "retrying", retry_deadline_utc=retry_deadline_utc
        )
        backoff = DURABLE_CLEANUP_INITIAL_BACKOFF_SECONDS
        last_error = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminal_cleanup_failure(model, last_error, retry_deadline_utc)
                return
            try:
                self._try_durable_unload(model, deadline)
                self._owned_model = None
                self._load_outcome_uncertain = False
                self._write_cleanup_report(
                    model,
                    "confirmed_absent",
                    retry_deadline_utc=retry_deadline_utc,
                )
                return
            except BaseException as error:
                last_error = error

            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(backoff, remaining))
            backoff = min(backoff * 2, DURABLE_CLEANUP_MAX_BACKOFF_SECONDS)

    def _terminal_cleanup_failure(self, model, last_error, retry_deadline_utc):
        if last_error is None:
            last_error = TimeoutError("Ollama cleanup exceeded its end-to-end deadline")
        final_error = f"{type(last_error).__name__}: {last_error}"
        self._cleanup_error = RuntimeError(
            f"Ollama model {model} durable cleanup could not confirm "
            f"absence after {self.durable_cleanup_timeout_seconds}s; "
            f"last error: {final_error}"
        )
        self._write_cleanup_report(
            model,
            "terminal_failure",
            retry_deadline_utc=retry_deadline_utc,
            final_error=final_error,
        )

    def _try_durable_unload(self, model, deadline):
        if self._preload_thread.is_alive():
            raise RuntimeError(
                "Ollama preload transport exceeded its end-to-end deadline"
            )
        resident = self._models(deadline)
        if self._contains_model(resident, model):
            self._unload_exact(model, deadline)
        elif resident:
            names = [entry.get("model") or entry.get("name") for entry in resident]
            raise RuntimeError(
                "unexpected Ollama models remained resident during durable "
                "cleanup: " + ", ".join(str(name) for name in names)
            )

    def _close_uncertain(self, owned_model, *, deadline=None):
        if deadline is not None:
            self._start_durable_cleanup(owned_model)
            raise TimeoutError(
                "owned model cleanup exceeded session budget: preload remains uncertain"
            )
        completed = self._wait_for_preload_completion()
        if not completed or "error" in self._preload_outcome:
            self._reconcile_uncertain_load(owned_model)
            self._owned_model = None
            self._load_outcome_uncertain = False
            return {"model": owned_model, "unloaded": True}

        cleanup_deadline = time.monotonic() + self.durable_cleanup_timeout_seconds
        self._unload_exact(owned_model, cleanup_deadline)
        self._owned_model = None
        self._load_outcome_uncertain = False
        return {"model": owned_model, "unloaded": True}

    @staticmethod
    def _reject_unowned_residents(resident):
        if resident:
            names = [entry.get("model") or entry.get("name") for entry in resident]
            raise RuntimeError(
                "unexpected Ollama models remained resident while owned model was absent: "
                + ", ".join(str(name) for name in names)
            )
