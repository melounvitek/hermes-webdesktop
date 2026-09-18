"""Optional Linux host shells. Profile scoping is not an OS security boundary.

Wire: binary UTF-8 input/output; text JSON resize. Reattach replays the bounded
byte tail (clear the terminal first). 4410 = exit, 4409 = replaced, 4401 = auth.
Settings live in plugins.entries.browser-terminal.settings in the server profile.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import secrets
import signal
import sys
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError


NAME = "browser-terminal"
AUTH_LEASE_SECONDS = 60


def spawn_shell(*args, **kwargs):
    from hermes_cli.pty_bridge import PtyBridge

    class HostShellBridge(PtyBridge):
        def close(self):
            if self._closed:
                return

            def job_groups():
                groups = set()
                for stat in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields = stat.read_text().rsplit(")", 1)[1].split()
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        continue
                    # PTY children keep their session ID even after the shell exits.
                    # Ignore zombies: their parent, not kill(), must reap them.
                    if (int(fields[3]) == self.pid and fields[0] != "Z"
                            and int(stat.parent.name) != self.pid):
                        groups.add(int(fields[2]))
                return groups

            # Interactive job control puts foreground/background jobs in separate
            # groups. Kill those first so a live shell can reap its children.
            try:
                for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGKILL):
                    groups = job_groups()
                    if not groups:
                        break
                    for group in groups:
                        with suppress(ProcessLookupError):
                            os.killpg(group, sig)
                    deadline = time.monotonic() + 0.5
                    while job_groups() and time.monotonic() < deadline:
                        time.sleep(0.02)
            finally:
                super().close()

    return HostShellBridge.spawn(*args, **kwargs)


class Size(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cols: int = Field(default=80, ge=1, le=2000)
    rows: int = Field(default=24, ge=1, le=1000)


class Create(Size):
    cwd: str | None = Field(default=None, max_length=4096)


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_sessions: int = Field(default=8, ge=1, le=64)
    max_tickets: int = Field(default=64, ge=1, le=1024)
    ticket_ttl: int = Field(default=30, ge=1, le=60)
    detached_ttl: int = Field(default=300, ge=1, le=3600)
    buffer_bytes: int = Field(default=262144, ge=1024, le=1048576)


class TicketLogFilter(logging.Filter):
    # Uvicorn INFO handshake logs include query strings; its DEBUG protocol log
    # can see the request before a handler runs. Never log this plugin's query.
    pattern = re.compile(r"(/api/plugins/browser-terminal/ws)\?[^\s\"']*")

    def filter(self, record):
        message = record.getMessage()
        if "/api/plugins/browser-terminal/ws?" in message:
            record.msg = self.pattern.sub(r"\1?[redacted]", message)
            record.args = ()
        return True


def installation_config():
    try:
        from hermes_constants import get_process_hermes_home
        from hermes_cli.config import load_config
        from hermes_cli.web_server_profiles import _config_profile_scope, _hermes_home_scope

        with _hermes_home_scope(get_process_hermes_home()), _config_profile_scope("current"):
            cfg = load_config()
        plugins = cfg.get("plugins") or {}
        if (NAME not in plugins.get("enabled", []) or NAME in plugins.get("disabled", [])
                or not Path(__file__).is_file()
                or not Path(__file__).with_name("manifest.json").is_file()):
            raise HTTPException(404, "Plugin not found")
        return (plugins.get("entries", {}).get(NAME) or {}).get("settings") or {}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Browser terminal plugin configuration is unavailable") from exc


def same_origin(connection):
    origin = connection.headers.get("origin", "")
    try:
        parsed = urlsplit(origin)
        expected_scheme = "https" if connection.url.scheme in {"https", "wss"} else "http"
        expected = urlsplit(f"{expected_scheme}://{connection.headers.get('host', '')}")
        if (parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment
                or (parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
                != (expected.scheme, expected.hostname, expected.port or (443 if expected.scheme == "https" else 80))):
            raise ValueError
    except ValueError:
        raise HTTPException(403, "Same-origin browser request required")


def loopback_connection(connection):
    try:
        return (getattr(connection.app.state, "auth_required", None) is False
                and ipaddress.ip_address(connection.client.host).is_loopback
                and getattr(connection.app.state, "bound_host", None) in {"127.0.0.1", "::1", "localhost"}
                and connection.url.hostname in {"127.0.0.1", "::1", "localhost"})
    except (ValueError, AttributeError):
        return False


def principal(request):
    from hermes_cli.dashboard_auth.base import Session
    from hermes_cli.web_server import _has_valid_session_token, _SESSION_TOKEN

    session = getattr(request.state, "session", None)
    if getattr(request.app.state, "auth_required", None) is True:
        if isinstance(session, Session) and session.provider and session.user_id and session.expires_at > time.time():
            return ("session", session.provider, session.org_id, session.user_id), session.expires_at
        raise HTTPException(401, "Verified dashboard session required")
    # The stock token path is safe only on a genuinely loopback listener/peer.
    if loopback_connection(request) and _has_valid_session_token(request):
        return ("loopback", hashlib.sha256(_SESSION_TOKEN.encode()).hexdigest()), float("inf")
    raise HTTPException(401, "Verified dashboard session or loopback token required")


def profile_home(profile):
    from hermes_constants import get_process_hermes_home
    from hermes_cli.web_server_profiles import _resolve_profile_dir, _is_current_profile

    return (get_process_hermes_home() if _is_current_profile(profile)
            else _resolve_profile_dir(profile.strip())).resolve()


@dataclass
class Shell:
    session: object
    owner: tuple
    home: Path
    ws: object = None
    lease_until: float = 0
    auth_until: float = 0
    expiry_task: asyncio.Task | None = None

    def authorized(self):
        return self.lease_until > time.monotonic() and self.auth_until > time.time()

    async def stop_expiry(self):
        task = self.expiry_task
        if task is not None and task is not asyncio.current_task():
            # Once revoked, let the bounded socket close finish rather than
            # abandon it when a heartbeat or removal races with expiry.
            if self.lease_until:
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
            self.expiry_task = None


@dataclass
class Ticket:
    sid: str
    owner: tuple
    home: Path
    origin: str
    deadline: float
    auth_until: float


class Terminals:
    def __init__(self, limits):
        self.limits = limits
        self.shells = {}
        self.tickets = {}
        self.lock = asyncio.Lock()

    async def remove(self, sid, code=4410):
        shell = self.shells.get(sid)
        if shell is not None:
            await self.revoke(sid, shell, code=code)
            await shell.session.close()
            # Keep ownership until cleanup finishes: cancelling the reaper during
            # shutdown must leave this shell visible to lifespan cleanup.
            self.shells.pop(sid, None)

    async def revoke(self, sid, shell, code=4401):
        self.tickets = {k: t for k, t in self.tickets.items() if t.sid != sid}
        ws = shell.ws
        if ws is not None:
            # Stop both PTY output and input before waiting for the close handshake.
            shell.ws = None
            shell.session.detach(ws)
        await shell.stop_expiry()
        shell.lease_until = 0
        if ws is not None:
            with suppress(WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
                await asyncio.wait_for(ws.close(code=code), 2)

    async def renew(self, sid, shell, auth_until):
        await shell.stop_expiry()
        if not shell.authorized():
            await self.revoke(sid, shell)
        shell.lease_until = time.monotonic() + AUTH_LEASE_SECONDS
        shell.auth_until = auth_until
        shell.expiry_task = asyncio.create_task(self.expire(sid, shell))

    async def expire(self, sid, shell):
        try:
            while shell.authorized():
                await asyncio.sleep(min(shell.lease_until - time.monotonic(),
                                        shell.auth_until - time.time()))
            # Never take manager.lock: another shell's cleanup may hold it.
            await self.revoke(sid, shell)
        finally:
            shell.expiry_task = None

    async def reap_once(self):
        async with self.lock:
            try:
                installation_config()
                enabled = True
            except HTTPException:
                enabled = False
            now = time.monotonic()
            self.tickets = {k: t for k, t in self.tickets.items()
                            if t.deadline > now and t.auth_until > time.time()}
            for sid, shell in list(self.shells.items()):
                if not shell.authorized():
                    await self.revoke(sid, shell)
                session = shell.session
                if (not enabled or not shell.home.is_dir() or not session.alive
                        or (not session.attached and session.last_detached_at is not None
                            and now - session.last_detached_at > self.limits.detached_ttl)):
                    await self.remove(sid, code=4410 if enabled else 4401)

    async def reap(self):
        while True:
            await self.reap_once()
            await asyncio.sleep(0.5)

    def owned(self, sid, owner, home):
        shell = self.shells.get(sid)
        if shell is None or shell.owner != owner or shell.home != home or not shell.session.alive:
            raise HTTPException(404, "Terminal session not found")
        return shell


@asynccontextmanager
async def lifespan(app):
    log_filter = TicketLogFilter()
    loggers = [logging.getLogger(name) for name in ("uvicorn.error", "uvicorn.access", "websockets.server")]
    for logger in loggers:
        logger.addFilter(log_filter)
    manager = None
    reaper = None
    try:
        try:
            limits = Limits.model_validate(installation_config())
            manager = Terminals(limits)
            app.state.browser_terminal = manager
            reaper = asyncio.create_task(manager.reap())
        except (HTTPException, ValidationError):
            app.state.browser_terminal = None
        yield
    finally:
        app.state.browser_terminal = None
        if reaper is not None:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
        if manager is not None:
            async with manager.lock:
                for sid in list(manager.shells):
                    await manager.remove(sid)
                manager.tickets.clear()
        for logger in loggers:
            logger.removeFilter(log_filter)


router = APIRouter(lifespan=lifespan)


def manager_for(connection):
    installation_config()
    manager = getattr(connection.app.state, "browser_terminal", None)
    if manager is None:
        raise HTTPException(503, "Browser terminal unavailable: router lifespan or settings unsupported; restart the server")
    return manager


@router.post("/sessions")
async def create_session(body: Create, request: Request, response: Response, profile: str = "current"):
    same_origin(request)
    owner, _ = principal(request)
    manager = manager_for(request)
    home = profile_home(profile)
    if sys.platform != "linux":
        raise HTTPException(503, "Browser terminal currently requires Linux")
    try:
        import pwd
        from hermes_cli.pty_bridge import PtyBridge
        from hermes_cli.pty_session import PtySession
        from hermes_cli.web_server_profiles import _config_profile_scope
        from hermes_cli.config import load_config
        from tools.environments.local import served_profile_child_env
    except ImportError as exc:
        raise HTTPException(503, "Browser terminal requires Hermes PTY support") from exc
    if not PtyBridge.is_available():
        raise HTTPException(503, "Browser terminal requires the ptyprocess package")
    shell = pwd.getpwuid(os.getuid()).pw_shell or "/bin/sh"
    if not Path(shell).is_file() or not os.access(shell, os.X_OK):
        raise HTTPException(503, "Host account shell is unavailable")
    with _config_profile_scope(profile):
        configured = (load_config().get("terminal") or {}).get("cwd")
        raw_cwd = body.cwd if body.cwd is not None else (configured if configured and configured != "." else str(Path.home()))
        try:
            cwd = Path(raw_cwd).expanduser()
            if not cwd.is_absolute():
                raise ValueError
            cwd = cwd.resolve(strict=True)
            if not cwd.is_dir() or not os.access(cwd, os.X_OK):
                raise ValueError
        except (TypeError, ValueError, OSError, RuntimeError):
            raise HTTPException(400, "cwd must resolve to an existing absolute host directory")
        # Host shell, not the agent's terminal backend. No client env and no
        # launch-profile residue, provider keys, dashboard token or startup hooks.
        env = served_profile_child_env(target_home=home, base={
            "HOME": str(Path.home()), "PATH": os.environ.get("PATH", os.defpath), "SHELL": shell,
            "LANG": "C.UTF-8", "TERM": "xterm-256color",
        })
    async with manager.lock:
        if len(manager.shells) >= manager.limits.max_sessions:
            raise HTTPException(429, "Terminal session limit reached")
        sid = secrets.token_urlsafe(32)
        spawn = asyncio.create_task(asyncio.to_thread(
            spawn_shell, [shell, "-i"], cwd=str(cwd), env=env, cols=body.cols, rows=body.rows))
        try:
            bridge = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            # Cancellation cannot interrupt fork/exec running in a worker thread.
            bridge = await spawn
            await asyncio.to_thread(bridge.close)
            raise
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, "Unable to start host shell") from exc
        session = PtySession(sid, bridge, buffer_cap=manager.limits.buffer_bytes, read_timeout=0.1)
        session.last_detached_at = time.monotonic()
        manager.shells[sid] = Shell(session, owner, home)
        await session.start()
    response.headers["Cache-Control"] = "no-store"
    return {"id": sid, "shell": shell, "cwd": str(cwd)}


@router.post("/sessions/{sid}/ticket")
async def mint_ticket(sid: str, request: Request, response: Response, profile: str = "current"):
    same_origin(request)
    owner, auth_until = principal(request)
    manager = manager_for(request)
    home = profile_home(profile)
    async with manager.lock:
        shell = manager.owned(sid, owner, home)
        now = time.monotonic()
        manager.tickets = {k: t for k, t in manager.tickets.items() if t.deadline > now}
        if len(manager.tickets) >= manager.limits.max_tickets:
            raise HTTPException(429, "WebSocket ticket limit reached")
        await manager.renew(sid, shell, auth_until)
        value = secrets.token_urlsafe(32)
        manager.tickets[value] = Ticket(sid, owner, home, request.headers["origin"],
                                       now + manager.limits.ticket_ttl, auth_until)
    response.headers["Cache-Control"] = "no-store"
    return {"ticket": value}


@router.post("/sessions/{sid}/heartbeat")
async def heartbeat(sid: str, request: Request, response: Response, profile: str = "current"):
    same_origin(request)
    owner, auth_until = principal(request)
    manager = manager_for(request)
    home = profile_home(profile)
    async with manager.lock:
        shell = manager.owned(sid, owner, home)
        await manager.renew(sid, shell, auth_until)
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str, request: Request, profile: str = "current"):
    same_origin(request)
    owner, _ = principal(request)
    manager = manager_for(request)
    home = profile_home(profile)
    async with manager.lock:
        manager.owned(sid, owner, home)
        await manager.remove(sid)
    return {"ok": True}


@router.websocket("/ws")
async def websocket(ws: WebSocket):
    shell = None
    try:
        same_origin(ws)
        manager = manager_for(ws)
        async with manager.lock:
            values = ws.query_params.getlist("ticket")
            ticket = manager.tickets.pop(values[0], None) if len(values) == 1 else None
            if (ticket is None or ticket.deadline <= time.monotonic() or ticket.auth_until <= time.time()
                    or ticket.origin != ws.headers["origin"] or not ticket.home.is_dir()):
                raise HTTPException(401, "Invalid ticket")
            if ticket.owner[0] == "loopback":
                from hermes_cli.web_server import _SESSION_TOKEN
                identity = ("loopback", hashlib.sha256(_SESSION_TOKEN.encode()).hexdigest())
                if not loopback_connection(ws) or ticket.owner != identity:
                    raise HTTPException(401, "Loopback authentication changed")
            elif getattr(ws.app.state, "auth_required", None) is not True:
                raise HTTPException(401, "Dashboard authentication changed")
            shell = manager.owned(ticket.sid, ticket.owner, ticket.home)
            if not shell.authorized():
                await manager.revoke(ticket.sid, shell)
                raise HTTPException(401, "Authorization lease expired")
            if shell.ws is not None:
                previous = shell.ws
                shell.ws = None
                shell.session.detach(previous)
                with suppress(WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
                    await asyncio.wait_for(previous.close(code=4409), 2)
            await ws.accept()
            # Expiry can run while accept/replacement close is suspended. Attach
            # only after rechecking, with no old socket for PtySession to await.
            if not shell.authorized():
                raise HTTPException(401, "Authorization lease expired")
            shell.ws = ws
            async with asyncio.timeout(5):
                if not await shell.session.attach(ws):
                    return
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if shell.ws is not ws:
                break
            if not shell.authorized():
                async with manager.lock:
                    if not shell.authorized():
                        await manager.revoke(ticket.sid, shell)
                        break
            data = message.get("bytes")
            if data is not None:
                if len(data) > 65536:
                    await ws.close(code=1009)
                    break
                if not await shell.session.write(ws, data):
                    break
            else:
                text = message.get("text", "")
                if len(text) > 1024:
                    await ws.close(code=1009)
                    break
                try:
                    payload = json.loads(text)
                    if not isinstance(payload, dict) or payload.pop("type", None) != "resize":
                        raise ValueError
                    size = Size.model_validate(payload)
                except (ValueError, ValidationError):
                    await ws.close(code=1008)
                    break
                shell.session.bridge.resize(size.cols, size.rows)
    except HTTPException:
        await ws.close(code=4401)
    except asyncio.TimeoutError:
        await ws.close(code=4401)
    except (WebSocketDisconnect, OSError, RuntimeError):
        # A transport disconnect leaves the same shell available until its TTL.
        pass
    finally:
        if shell is not None:
            shell.session.detach(ws)
            if shell.ws is ws:
                shell.ws = None
