"""End-to-end deadline enforcement for local Ollama HTTP requests."""

from http.client import HTTPConnection
import json
import socket
import threading
from urllib.parse import urlsplit

_DEFAULT_REQUEST_TIMEOUT = object()


class DeadlineHTTPClient:
    def __init__(self, endpoint, request_timeout_seconds):
        self.endpoint = endpoint
        self.request_timeout_seconds = request_timeout_seconds

    def request(
        self,
        method,
        path,
        payload=None,
        *,
        timeout=_DEFAULT_REQUEST_TIMEOUT,
        deadline_seconds=None,
    ):
        data = None if payload is None else json.dumps(payload).encode()
        socket_timeout = self._request_timeout(path, timeout, deadline_seconds)
        expired = threading.Event()
        endpoint = urlsplit(self.endpoint)
        connection = HTTPConnection(
            endpoint.hostname, endpoint.port, timeout=socket_timeout
        )
        responses = []

        def expire_request():
            expired.set()
            self._close_request(connection, responses)

        timer = None
        if deadline_seconds is not None:
            timer = threading.Timer(deadline_seconds, expire_request)
            timer.daemon = True
            timer.start()
        try:
            try:
                body = self._response_body(connection, method, path, data, responses)
            except BaseException as error:
                if expired.is_set():
                    raise TimeoutError(
                        f"Ollama {path} exceeded {deadline_seconds}s end-to-end deadline"
                    ) from error
                raise
            result = self._decode_response(body, expired, path, deadline_seconds)
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()
        return result

    @staticmethod
    def _response_body(connection, method, path, data, responses):
        connection.request(
            method,
            path,
            body=data,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        responses.append(response)
        try:
            body = response.read()
            if not 200 <= response.status < 300:
                raise RuntimeError(
                    f"HTTP Error {response.status}: Ollama {path} request failed"
                )
        finally:
            response.close()
        return body

    @staticmethod
    def _close_request(connection, responses):
        sockets = [connection.sock]
        for active_response in responses:
            raw = getattr(getattr(active_response, "fp", None), "raw", None)
            sockets.append(getattr(raw, "_sock", None))
        for active_socket in {item for item in sockets if item is not None}:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            active_socket.close()
        connection.close()

    def _request_timeout(self, path, timeout, deadline_seconds):
        socket_timeout = (
            self.request_timeout_seconds
            if timeout is _DEFAULT_REQUEST_TIMEOUT
            else timeout
        )
        if deadline_seconds is not None and deadline_seconds <= 0:
            raise TimeoutError(f"Ollama {path} exceeded its end-to-end deadline")
        if deadline_seconds is not None:
            socket_timeout = min(socket_timeout, deadline_seconds)
        return socket_timeout

    @staticmethod
    def _decode_response(body, expired, path, deadline_seconds):
        if expired.is_set():
            raise TimeoutError(
                f"Ollama {path} exceeded {deadline_seconds}s end-to-end deadline"
            )
        result = json.loads(body)
        if not isinstance(result, dict):
            raise RuntimeError(f"Ollama {path} returned a non-object response")
        return result
