"""HTTP admission contract for the independent, read-only updater foundation."""

from contextlib import ExitStack, closing, contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import time

import pytest


pytestmark = pytest.mark.linux_only

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/browser_updater.py"
ORIGIN = "https://updater.localhost"
IDENTITY = {"provider": "basic", "org_id": "", "user_id": "operator"}
TOKEN = "disposable-access-token"


@pytest.fixture
def updater():
    spec = importlib.util.spec_from_file_location("browser_updater", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def backend():
    state = {"requests": [], "status": 200, "body": None, "type": "application/json"}

    class Auth(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append((self.path, dict(self.headers)))
            body = state["body"]
            if body is None:
                body = json.dumps({**IDENTITY, "expires_at": time.time() + 60}).encode()
            if state.get("hold") is not None:
                state["entered"].set()
                state["hold"].wait(5)
            self.send_response(state["status"])
            self.send_header("Content-Type", state["type"])
            self.send_header(
                "Content-Length", str(len(body) + state.get("extra_length", 0))
            )
            self.send_header("Location", "http://external.invalid/never-follow")
            self.send_header("Set-Cookie", "never-relay=secret")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass  # The deadline test deliberately disconnects first.

        def log_message(self, *args):
            pass

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Auth)) as server:
        state["port"] = server.server_port
        yield state


@pytest.fixture
def config_path(tmp_path, backend):
    path = tmp_path / "updater.json"
    path.write_text(
        json.dumps({
            "public_origin": ORIGIN,
            "listen_port": 0,
            "backend_port": backend["port"],
            "administrators": [IDENTITY],
        })
    )
    path.chmod(0o600)
    return path


@pytest.fixture
def service(updater, config_path):
    with serving(updater.UpdaterServer(updater.load_config(config_path))) as server:
        yield server


def request(
    service,
    *,
    method="POST",
    path="/browser-updater/api/status",
    body=b"{}",
    headers=None,
):
    fields = {
        "Host": "updater.localhost",
        "Origin": ORIGIN,
        "Content-Type": "application/json",
        "Cookie": f"__Host-hermes_session_at={TOKEN}; __Host-hermes_session_rt=do-not-forward",
    }
    for name, value in (headers or {}).items():
        if value is None:
            fields.pop(name, None)
        else:
            fields[name] = value
    with closing(
        http.client.HTTPConnection("127.0.0.1", service.server_port, timeout=5)
    ) as client:
        client.request(method, path, body=body, headers=fields)
        response = client.getresponse()
        return response.status, dict(response.getheaders()), response.read()


def test_only_allowlisted_stock_identity_can_read_capabilities(
    service, backend, capsys
):
    status, headers, body = request(service)
    assert status == 200
    assert json.loads(body) == {"capabilities": []}
    assert headers["Cache-Control"] == "no-store"
    assert "Set-Cookie" not in headers and "Access-Control-Allow-Origin" not in headers
    path, sent = backend["requests"][-1]
    assert path == "/api/auth/me"
    assert sent["Authorization"] == f"Bearer {TOKEN}"
    assert "Cookie" not in sent and "Origin" not in sent
    for change in ({"provider": "another"}, {"org_id": None}, {"user_id": "viewer"}):
        backend["body"] = json.dumps({
            **IDENTITY,
            **change,
            "expires_at": time.time() + 60,
        }).encode()
        status, _, body = request(service)
        assert status == 403 and json.loads(body) == {"error": "Forbidden"}
    output = capsys.readouterr()
    assert TOKEN not in output.out + output.err
    assert "do-not-forward" not in output.out + output.err


def test_accepts_quoted_cookie_set_by_stock_hermes_for_padded_tokens(service, backend):
    status, _, _ = request(
        service, headers={"Cookie": '__Host-hermes_session_at="cGFkZGVk="'}
    )
    assert status == 200
    assert backend["requests"][-1][1]["Authorization"] == "Bearer cGFkZGVk="


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.invalid"},
        {"Origin": "https://attacker.invalid"},
        {"Origin": None},
        {"Origin": "null"},
        {"Origin": ORIGIN + "/"},
        {"Cookie": None},
        {"Cookie": "__Host-hermes_session_rt=refresh-only"},
        {"Cookie": f"hermes_session_at={TOKEN}"},
        {"Cookie": f'__Host-hermes_session_at="{TOKEN}'},
        pytest.param(
            {"Cookie": "__Host-hermes_session_at=x" + "=" * 8192}, id="oversized-token"
        ),
        {"Cookie": f"__Host-hermes_session_at={TOKEN}; __Host-hermes_session_at=other"},
        {"Cookie": f"__Host-hermes_session_at={TOKEN}; hermes_session_at=other"},
        {"Authorization": f"Bearer {TOKEN}", "Cookie": None},
    ],
)
def test_rejects_untrusted_request_before_contacting_backend(service, backend, headers):
    status, _, body = request(service, headers=headers)
    assert status == 403 and json.loads(body) == {"error": "Forbidden"}
    assert not backend["requests"]


@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"method": "GET"}, 405),
        ({"method": "OPTIONS"}, 405),
        ({"path": "/api/auth/me"}, 404),
        ({"path": "/browser-updater/api/status?target=other"}, 404),
        ({"path": "/browser-updater/api/status/"}, 404),
        ({"path": "//browser-updater/api/status"}, 404),
        ({"body": b'{"command":"update"}'}, 400),
        ({"body": b"[]"}, 400),
        ({"body": b"not-json"}, 400),
        ({"body": b" " * 1025}, 413),
        ({"headers": {"Content-Type": "text/plain"}}, 415),
        ({"headers": {"Transfer-Encoding": "chunked"}}, 400),
    ],
)
def test_endpoint_is_narrow_and_has_no_update_operation(
    service, backend, kwargs, status
):
    response, _, _ = request(service, **kwargs)
    assert response == status
    assert not backend["requests"]


@pytest.mark.parametrize(
    "status,body,content_type,expected",
    [
        (401, b"{}", "application/json", 403),
        (403, b"{}", "application/json", 403),
        (302, b"", "text/html", 503),
        (500, b"{}", "application/json", 503),
        (200, b"<html>login</html>", "text/html", 503),
        (200, b"not-json", "application/json", 503),
        (200, b"{}", "application/json", 503),
        (200, b"[]", "application/json", 503),
        pytest.param(
            200, b"x" * 16385, "application/json", 503, id="oversized-response"
        ),
        pytest.param(
            200,
            json.dumps({**IDENTITY, "expires_at": 10**400}).encode(),
            "application/json",
            503,
            id="overflowing-expiry",
        ),
        (
            200,
            json.dumps({**IDENTITY, "expires_at": 1}).encode(),
            "application/json",
            403,
        ),
    ],
)
def test_authentication_failures_are_closed_and_do_not_relay_cookies(
    service,
    backend,
    status,
    body,
    content_type,
    expected,
):
    backend.update(status=status, body=body, type=content_type)
    result, headers, payload = request(service)
    assert result == expected
    assert json.loads(payload) == {
        "error": "Forbidden" if expected == 403 else "Authentication unavailable"
    }
    assert "Set-Cookie" not in headers
    assert len(backend["requests"]) == 1  # Redirects cannot send credentials elsewhere.


@pytest.mark.parametrize(
    "duplicate",
    [
        b"Host: updater.localhost",
        b"Origin: https://updater.localhost",
        b"Content-Length: 2",
        b"Cookie: __Host-hermes_session_at=other",
    ],
)
def test_duplicate_security_headers_are_rejected(service, backend, duplicate):
    with socket.create_connection(
        ("127.0.0.1", service.server_port), timeout=5
    ) as client:
        client.sendall(
            b"POST /browser-updater/api/status HTTP/1.1\r\n"
            b"Host: updater.localhost\r\nOrigin: https://updater.localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 2\r\n"
            b"Cookie: __Host-hermes_session_at=disposable-access-token\r\n"
            + duplicate
            + b"\r\n\r\n{}"
        )
        with http.client.HTTPResponse(client) as response:
            response.begin()
            assert response.status in (400, 403)
            response.read()
    assert not backend["requests"]


def test_truncated_body_is_rejected_and_full_capacity_recovers(service, backend):
    headers = (
        b"POST /browser-updater/api/status HTTP/1.1\r\n"
        b"Host: updater.localhost\r\nOrigin: https://updater.localhost\r\n"
        b"Content-Type: application/json\r\nContent-Length: 100\r\n"
        b"Cookie: __Host-hermes_session_at=disposable-access-token\r\n\r\n{}"
    )
    with ExitStack() as stack:
        clients = [
            stack.enter_context(
                socket.create_connection(
                    ("127.0.0.1", service.server_port),
                    timeout=5,
                )
            )
            for _ in range(16)
        ]
        for client in clients:
            client.sendall(headers)
        with pytest.raises((
            http.client.RemoteDisconnected,
            ConnectionResetError,
            BrokenPipeError,
        )):
            request(service)
        for client in clients:
            client.shutdown(socket.SHUT_WR)
            with http.client.HTTPResponse(client) as response:
                response.begin()
                assert response.status == 400
                response.read()
    assert not backend["requests"]
    assert request(service)[0] == 200


def test_deadlines_close_partial_requests_and_stalled_authentication(
    updater,
    service,
    backend,
    monkeypatch,
):
    monkeypatch.setattr(updater, "REQUEST_TIMEOUT", 0.2)
    with socket.create_connection(
        ("127.0.0.1", service.server_port), timeout=5
    ) as client:
        client.sendall(b"POST /browser-updater/api/status HTTP/1.1\r\nHost: ")
        stop = threading.Event()

        def trickle():
            while not stop.wait(0.02):
                try:
                    client.sendall(b"x")
                except OSError:
                    return

        writer = threading.Thread(target=trickle)
        writer.start()
        try:
            assert client.recv(1024) == b""
        finally:
            stop.set()
            writer.join(timeout=5)
            assert not writer.is_alive()
    assert not backend["requests"]
    monkeypatch.setattr(updater, "REQUEST_TIMEOUT", 10)
    monkeypatch.setattr(updater, "AUTH_TIMEOUT", 0.2)
    backend.update(hold=threading.Event(), entered=threading.Event())
    try:
        assert request(service)[0] == 503
        assert backend["entered"].is_set()
    finally:
        backend["hold"].set()


def test_truncated_backend_response_is_unavailable(service, backend):
    backend["extra_length"] = 10
    assert request(service)[0] == 503


def test_backend_down_does_not_reuse_previous_authorization(service, backend):
    assert request(service)[0] == 200
    # Reserve an unlistened port: no unrelated process can receive credentials.
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        from dataclasses import replace

        service.config = replace(service.config, backend_port=closed.getsockname()[1])
        assert request(service)[0] == 503


@pytest.mark.parametrize(
    "change",
    [
        {"public_origin": "http://updater.localhost"},
        {"public_origin": ORIGIN + "/subpath"},
        {"public_origin": "https://user:secret@updater.localhost"},
        {"backend_port": True},
        {"backend_port": 0},
        {"listen_port": -1},
        {"backend_url": "https://external.invalid"},
        {"administrators": [{"provider": "basic", "user_id": "operator"}]},
    ],
)
def test_invalid_configuration_is_refused(updater, config_path, change):
    config = json.loads(config_path.read_text())
    config.update(change)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        updater.load_config(config_path)


def test_empty_allowlist_and_unsafe_config_file_fail_closed(updater, config_path):
    config = json.loads(config_path.read_text())
    config["administrators"] = []
    config_path.write_text(json.dumps(config))
    with serving(updater.UpdaterServer(updater.load_config(config_path))) as server:
        assert request(server)[0] == 403
    config_path.write_text(json.dumps(config) + " " * 16384)
    with pytest.raises(ValueError):
        updater.load_config(config_path)
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o666)
    with pytest.raises(ValueError):
        updater.load_config(config_path)
