#!/usr/bin/env python3
"""Independent, explicitly configured browser-only updater service.

Root-origin HTTPS only. Route /browser-updater/api/ to this loopback listener,
without stripping the prefix; leave Hermes routes and WebSockets untouched.
The proxy must preserve Host and Origin and must not cache these responses.
Without a maintenance configuration it is read-only. Provisioning the updater
and its dashboard unit is a separate operator action, never a release update.
"""

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
import threading
import time
from urllib.parse import urlsplit


REQUEST_TIMEOUT = 10
AUTH_TIMEOUT = 3
MAX_AUTH_RESPONSE = 16384

# -I -S deliberately omits the script directory; only import our trusted siblings.
sys.path.insert(0, str(Path(__file__).absolute().parent))


@dataclass(frozen=True)
class Config:
    public_origin: str
    listen_port: int
    backend_port: int
    administrators: frozenset
    maintenance: dict | None = None
    config_path: Path | None = None
    configuration_sha256: str | None = None


def identity(value):
    if not isinstance(value, dict):
        raise ValueError("Invalid administrator identity")
    provider, org, user = (value.get(key) for key in ("provider", "org_id", "user_id"))
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(user, str)
        or not user
        or (org is not None and not isinstance(org, str))
        or "org_id" not in value
    ):
        raise ValueError("Invalid administrator identity")
    # Empty and null organization IDs are distinct; never broaden an allowlist.
    return provider, org, user


def load_config(path):
    with Path(path).open("rb") as stream:
        meta = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(meta.st_mode)
            or meta.st_uid != os.getuid()
            or stat.S_IMODE(meta.st_mode) != 0o600
        ):
            raise ValueError("Configuration must be an owned mode-0600 regular file")
        raw = stream.read(16385)
        if len(raw) > 16384:
            raise ValueError("Updater configuration exceeds size limit")
        value = json.loads(raw)
    if not isinstance(value, dict) or set(value) - {"maintenance"} != {
        "public_origin",
        "listen_port",
        "backend_port",
        "administrators",
    }:
        raise ValueError("Invalid updater configuration fields")
    origin = value["public_origin"]
    if not isinstance(origin, str) or not re.fullmatch(
        r"https://[a-z0-9.-]+(?::[0-9]+)?", origin
    ):
        raise ValueError("A root HTTPS public origin is required")
    parsed = urlsplit(origin)
    if not parsed.hostname or parsed.port == 0:
        raise ValueError("Invalid public origin")
    for name, minimum in (("listen_port", 0), ("backend_port", 1)):
        if type(value[name]) is not int or not minimum <= value[name] <= 65535:
            raise ValueError("Invalid loopback port")
    admins = value["administrators"]
    if not isinstance(admins, list) or any(
        not isinstance(admin, dict) or set(admin) != {"provider", "org_id", "user_id"}
        for admin in admins
    ):
        raise ValueError("Invalid administrator allowlist")
    maintenance = value.get("maintenance")
    if "maintenance" in value:
        from browser_download import source_url
        from browser_setup import E

        E.keys(
            maintenance,
            "installation state_dir source unit unit_sha256 working_directory",
        )
        source_url(maintenance["source"])
        E.hex_value(maintenance["unit_sha256"])
        for name in ("installation", "state_dir", "working_directory"):
            E.no_links(E.absolute_path(maintenance[name]))
        E.require(
            isinstance(maintenance["unit"], str)
            and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*\.service", maintenance["unit"]
            ),
            "Invalid dashboard unit",
        )
    return Config(
        origin,
        value["listen_port"],
        value["backend_port"],
        frozenset(map(identity, admins)),
        maintenance,
        Path(path).absolute(),
        hashlib.sha256(raw).hexdigest(),
    )


def shutdown_socket(connection):
    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # Completion can race the absolute deadline.


def authorize(config, token):
    """Use stock bearer verification; never rotate or persist Hermes cookies."""
    try:
        with closing(
            http.client.HTTPConnection(
                "127.0.0.1", config.backend_port, timeout=AUTH_TIMEOUT
            )
        ) as conn:
            conn.connect()
            timer = threading.Timer(AUTH_TIMEOUT, shutdown_socket, args=(conn.sock,))
            timer.daemon = True
            timer.start()
            try:
                conn.request(
                    "GET", "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
                )
                with conn.getresponse() as response:
                    if response.status in (401, 403):
                        return 403
                    if (
                        response.status != 200
                        or response.getheader("Content-Type", "").split(";")[0]
                        != "application/json"
                    ):
                        return 503
                    raw = response.read(MAX_AUTH_RESPONSE + 1)
                    if response.length not in (None, 0):
                        return 503
                if len(raw) > MAX_AUTH_RESPONSE:
                    return 503
                value = json.loads(raw)
                who = identity(value)
                expires = value.get("expires_at")
                if type(expires) not in (int, float) or not math.isfinite(expires):
                    return 503
                return (
                    who
                    if expires > time.time() and who in config.administrators
                    else 403
                )
            finally:
                timer.cancel()
    except (OSError, http.client.HTTPException, ValueError, OverflowError):
        return 503


class UpdaterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config):
        self.config = config
        self.slots = threading.BoundedSemaphore(16)
        self.updates = None
        super().__init__(("127.0.0.1", config.listen_port), UpdaterHandler)
        try:
            if config.maintenance is not None:
                from browser_update_jobs import Updates

                self.updates = Updates(config)
        except BaseException:
            self.server_close()
            raise

    def server_close(self):
        super().server_close()
        if self.updates is not None:
            self.updates.close()

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Request lines/headers may contain secrets; do not log exception details.
        print("Updater request failed", file=sys.stderr)


class UpdaterHandler(BaseHTTPRequestHandler):
    # HTTP/1.0 closes after one request, including rejected request bodies.
    server_version = "BrowserUpdater"
    sys_version = ""

    def handle(self):
        # A socket timeout alone resets on every read and permits slow trickles.
        timer = threading.Timer(
            REQUEST_TIMEOUT, shutdown_socket, args=(self.connection,)
        )
        timer.daemon = True
        timer.start()
        try:
            super().handle()
        finally:
            timer.cancel()

    def log_message(self, *args):
        pass

    def reply(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def send_error(self, code, message=None, explain=None):
        self.reply(code, {"error": "Request rejected"})

    def method_not_allowed(self):
        self.send_error(405)

    do_GET = do_HEAD = do_OPTIONS = do_PUT = do_PATCH = do_DELETE = method_not_allowed

    def do_POST(self):
        config = self.server.config
        if self.headers.get_all("Host", []) != [
            urlsplit(config.public_origin).netloc
        ] or self.headers.get_all("Origin", []) != [config.public_origin]:
            self.reply(403, {"error": "Forbidden"})
            return
        # BaseHTTPRequestHandler normalizes leading // before assigning path.
        paths = {
            f"/browser-updater/api/{name}": name
            for name in ("status", "offer", "apply")
        }
        operation = paths.get(self.requestline.split()[1])
        if operation is None or (operation != "status" and self.server.updates is None):
            self.send_error(404)
            return
        if self.headers.get_all("Content-Type", []) != ["application/json"]:
            self.send_error(415)
            return
        lengths = self.headers.get_all("Content-Length", [])
        if (
            self.headers.get_all("Transfer-Encoding")
            or len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
        ):
            self.send_error(400)
            return
        length = int(lengths[0]) if len(lengths[0]) < 8 else 1025
        if length > 1024:
            self.send_error(413)
            return
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw)
            if len(raw) != length or not isinstance(body, dict):
                raise ValueError("Expected complete object")
            if operation == "apply":
                if (
                    set(body) != {"offer_id", "confirm_restart_and_rollback"}
                    or body["confirm_restart_and_rollback"] is not True
                    or not isinstance(body["offer_id"], str)
                    or not re.fullmatch(r"[a-f0-9]{32}", body["offer_id"])
                ):
                    raise ValueError("Explicit consent and exact offer required")
            elif body:
                raise ValueError("Expected empty object")
        except ValueError:
            self.send_error(400)
            return
        cookies = self.headers.get_all("Cookie", [])
        tokens = []
        for cookie in cookies:
            for part in cookie.split(";"):
                name, separator, value = part.strip().partition("=")
                if name in (
                    "__Host-hermes_session_at",
                    "__Secure-hermes_session_at",
                    "hermes_session_at",
                ):
                    # Stock Hermes quotes values containing "=" (padded tokens),
                    # and browsers send the quotes back.
                    if len(value) > 1 and value[0] == value[-1] == '"':
                        value = value[1:-1]
                    tokens.append((name, value if separator else ""))
        if (
            len(cookies) != 1
            or len(tokens) != 1
            or tokens[0][0] != "__Host-hermes_session_at"
            or len(tokens[0][1]) > 8192
            or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", tokens[0][1])
        ):
            self.reply(403, {"error": "Forbidden"})
            return
        result = authorize(config, tokens[0][1])
        if isinstance(result, int):
            self.reply(
                result,
                {
                    "error": "Forbidden"
                    if result == 403
                    else "Authentication unavailable"
                },
            )
            return
        updates = self.server.updates
        if updates is None:
            self.reply(200, {"capabilities": []})
            return
        try:
            actions = {
                "status": updates.status,
                "offer": lambda: updates.offer(result),
                "apply": lambda: updates.apply(
                    result, body["offer_id"], body["confirm_restart_and_rollback"]
                ),
            }
            value = actions[operation]()
        except ValueError as error:
            self.reply(409, {"error": str(error)})
            return
        except OSError:
            self.reply(503, {"error": "Updater state unavailable"})
            return
        self.reply(200 if operation == "status" else 202, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Trusted service entry point; never use the downloaded launcher",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.dashboard:
            from browser_update_jobs import admit_start
            from browser_setup import E

            E.require(
                config.maintenance is not None,
                "Dashboard maintenance is not configured",
            )
            E.require(
                os.getcwd() == config.maintenance["working_directory"],
                "Unexpected dashboard working directory",
            )
            code = E.lifecycle(
                argparse.Namespace(
                    command="start",
                    install_root=config.maintenance["installation"],
                    port=config.backend_port,
                    timeout=30,
                ),
                admit_start=lambda receipt: admit_start(config, receipt),
            )
            parser.exit(code)
        with UpdaterServer(config) as server:
            print(
                f"Browser updater listening on 127.0.0.1:{server.server_port}",
                flush=True,
            )
            server.serve_forever()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Updater unavailable: {error}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
