"""Local HTTP fixture for deterministic model-lifecycle failures."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


@dataclass
class ServerOptions:
    initially_loaded: object = False
    unload_sticks: object = False
    preload_response_delay: object = 0
    preload_visibility_delay: object = 0
    preload_never_completes_delay: object = 0
    post_load_context: object = None
    post_load_model: object = None
    extra_resident: object = False
    ps_failures_after_load: object = 0
    preload_drip_interval: object = 0
    preload_header_drip_interval: object = 0
    version_drip_interval: object = 0
    show_drip_interval: object = 0
    ps_drip_interval_before_load: object = 0
    ps_drip_interval_after_load: object = 0


@contextmanager
def ollama_server(**options):
    config = ServerOptions(**options)
    state = {
        "loaded": config.initially_loaded,
        "ever_loaded": config.initially_loaded,
        "requests": [],
        "unload_sticks": config.unload_sticks,
        "model": "gemma4:12b",
        "context": 262144,
        "ps_failures_after_load": config.ps_failures_after_load,
        "version_drip_interval": config.version_drip_interval,
        "show_drip_interval": config.show_drip_interval,
        "ps_drip_interval_before_load": config.ps_drip_interval_before_load,
        "ps_drip_interval_after_load": config.ps_drip_interval_after_load,
    }

    Handler = partial(OllamaHandler, state=state, config=config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class OllamaHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, state, config, **kwargs):
        self.state = state
        self.config = config
        super().__init__(*args, **kwargs)

    def log_request(self, code="-", size="-"):
        pass

    def _json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _drip_json(self, value, interval, *, include_headers=False):
        body = json.dumps(value).encode()
        if include_headers:
            payload = (
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            payload = body
        try:
            for byte in payload:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.state["requests"].append(("GET", self.path, None))
        if self.path == "/api/version":
            if self.state["version_drip_interval"]:
                return self._drip_json(
                    {"version": "0.33.3"}, self.state["version_drip_interval"]
                )
            return self._json({"version": "0.33.3"})
        if self.path == "/api/ps":
            return self._models_response()
        self.send_error(404)

    def _models_response(self):
        if self.state["loaded"] and self.state["ps_failures_after_load"]:
            self.state["ps_failures_after_load"] -= 1
            return self.send_error(503)
        models = _resident_models(self.state, self.config)
        if self.state["loaded"] and self.state["ps_drip_interval_after_load"]:
            return self._drip_json(
                {"models": models}, self.state["ps_drip_interval_after_load"]
            )
        if not self.state["loaded"] and self.state["ps_drip_interval_before_load"]:
            return self._drip_json(
                {"models": models}, self.state["ps_drip_interval_before_load"]
            )
        return self._json({"models": models})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["requests"].append(("POST", self.path, payload))
        if self.path == "/api/show":
            return self._show_response(payload)
        if self.path != "/api/generate":
            return self.send_error(404)
        return self._generate_response(payload)

    def _generate_response(self, payload):
        if payload.get("keep_alive") == 0:
            if not self.state["unload_sticks"]:
                self.state["loaded"] = False
        else:
            if not self._load_model(payload):
                return self._json({"done": False})
        response = {"done": True, "response": "", "load_duration": 10}
        if self.config.preload_header_drip_interval and payload.get("keep_alive") != 0:
            return self._drip_json(
                response, self.config.preload_header_drip_interval, include_headers=True
            )
        if self.config.preload_drip_interval and payload.get("keep_alive") != 0:
            return self._drip_json(response, self.config.preload_drip_interval)
        self._json(response)

    def _show_response(self, payload):
        contexts = {
            "gemma4:12b": ("gemma4", 262144),
            "granite4.2:8b": ("granite", 131072),
            "Ornith-1.5-9B:latest": ("qwen3", 262144),
        }
        architecture, context = contexts[payload["model"]]
        response = {
            "details": {"quantization_level": "Q6_K"},
            "model_info": {
                "general.architecture": architecture,
                f"{architecture}.context_length": context,
            },
        }
        if self.state["show_drip_interval"]:
            return self._drip_json(response, self.state["show_drip_interval"])
        return self._json(response)

    def _load_model(self, payload):
        if self.config.preload_never_completes_delay:
            time.sleep(self.config.preload_never_completes_delay)
            return False
        if self.config.preload_visibility_delay:
            time.sleep(self.config.preload_visibility_delay)
        self.state["loaded"] = True
        self.state["ever_loaded"] = True
        self.state["model"] = self.config.post_load_model or payload["model"]
        self.state["context"] = (
            self.config.post_load_context or payload["options"]["num_ctx"]
        )
        if self.config.preload_response_delay:
            time.sleep(self.config.preload_response_delay)
        return True


def _resident_models(state, config):
    models = []
    if state["loaded"]:
        model = state.get("model", "gemma4:12b")
        models = [
            {
                "name": model,
                "model": model,
                "digest": "f87405c6d8adfull",
                "size": 9_800_000_000,
                "size_vram": 8_700_000_000,
                "context_length": state["context"],
            }
        ]
    if config.extra_resident and state["ever_loaded"]:
        models.append(
            {
                "name": "unowned:latest",
                "model": "unowned:latest",
                "context_length": 1024,
            }
        )
    return models
