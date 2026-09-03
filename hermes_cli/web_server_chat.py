"""Chat/terminal WebSocket plumbing: PTY bridge selection and registry, WS
client/origin/auth gates, chat argv resolution, gateway/sidecar URL building.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import asyncio
import atexit
import concurrent.futures
import hmac
import os
import re
import sys
import tempfile
import threading
import urllib.request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
from typing import Optional
from hermes_cli.pty_session import PtySessionRegistry

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# /api/pty — PTY-over-WebSocket bridge for the dashboard "Chat" tab: spawns the
# same ``hermes --tui`` binary the CLI uses behind a pseudo-terminal and forwards
# bytes + resize escapes; the browser renders the ANSI through xterm.js.
# Auth: ``?token=<session_token>`` query param (browsers can't set Authorization
# on the WS upgrade), same ephemeral ``_SESSION_TOKEN`` as REST.

# PTY bridge: POSIX uses pty_bridge (fcntl/termios/ptyprocess); native Windows
# uses win_pty_bridge (pywinpty/ConPTY).  Both expose the same surface —
# spawn/read/write/resize/close/is_available — so the handler needs no guards.
if sys.platform.startswith("win"):
    try:
        from hermes_cli.win_pty_bridge import WinPtyBridge as PtyBridge, PtyUnavailableError
        _PTY_BRIDGE_AVAILABLE = True
    except ImportError:  # pragma: no cover - pywinpty missing
        PtyBridge = None  # type: ignore[assignment]
        _PTY_BRIDGE_AVAILABLE = False

        class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
            """Stub when win_pty_bridge cannot be imported."""
else:
    try:
        from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError
        _PTY_BRIDGE_AVAILABLE = True
    except ImportError:  # pragma: no cover - dev env without ptyprocess
        PtyBridge = None  # type: ignore[assignment]
        _PTY_BRIDGE_AVAILABLE = False

        class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
            """Stub on platforms where pty_bridge can't be imported."""
_RESIZE_RE = re.compile(rb"\x1b\[RESIZE:(\d+);(\d+)\]")
_PTY_READ_CHUNK_TIMEOUT = 0.2

# Back-off between idle PTY reads so a quiet terminal does not spin the event
# loop (keeps dashboard idle CPU low).
_PTY_IDLE_BACKOFF = 0.05
PTY_REGISTRY = PtySessionRegistry(
    ttl=30 * 60, max_sessions=16, buffer_cap=1 * 1024 * 1024, read_timeout=_PTY_READ_CHUNK_TIMEOUT)


async def _legacy_pump(ws: "WebSocket", bridge) -> None:
    """Original 1:1 socket<->PTY pump: stream until disconnect, then close the
    bridge. Used when no ``?attach=`` token is supplied (keep-alive opt-in)."""
    loop = asyncio.get_running_loop()

    async def pump_pty_to_ws() -> None:
        try:
            while True:
                chunk = await loop.run_in_executor(None, bridge.read, _PTY_READ_CHUNK_TIMEOUT)
                if chunk is None:  # EOF
                    return
                if not chunk:  # no data this tick; yield control and retry
                    await asyncio.sleep(_PTY_IDLE_BACKOFF)
                    continue
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    return
        finally:
            # Child exited (EOF) or the send side broke.  Close the WebSocket so
            # the writer loop's ``ws.receive()`` returns instead of blocking
            # forever on a half-open browser socket (no FIN, common on
            # macOS/launchd) — otherwise the PTY's fds leak and auto-reconnect
            # stacks a fresh PTY on each orphan.  Reap the bridge here too
            # (close() is idempotent): if the handler task is cancelled the
            # instant we close the WS, the writer's ``finally`` can be skipped.
            try:
                await asyncio.to_thread(bridge.close)
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass

    reader_task = asyncio.create_task(pump_pty_to_ws())

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # ws.receive() after the socket is already disconnected
                # (e.g. closed by the reader task above).
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue
            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue
            bridge.write(raw)
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        await asyncio.to_thread(bridge.close)


# Starlette's TestClient reports the peer as "testclient"; treat it as
# loopback so tests don't need to rewrite request scope.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _ws_client_reason(ws: "WebSocket") -> Optional[str]:
    """Return a rejection reason token for the peer IP, or None when allowed.

    Loopback bind: only loopback clients — the legacy ``?token=`` is the only
    auth, so LAN hosts must not get to guess it.  Explicit non-loopback bind
    (``--host 0.0.0.0``/``::``/LAN IP, always with ``--insecure``): any peer;
    DNS-rebinding is still blocked by :func:`_ws_host_origin_reason`.  Gated
    mode: any peer — ``proxy_headers=True`` rewrites ``ws.client.host`` to the
    X-Forwarded-For value and the OAuth gate + ``?ticket=`` is the auth.
    An empty peer on a loopback bind fails closed (misconfigured proxy / unix
    socket must not reach a loopback-only surface).
    """
    from hermes_cli.web_server import app
    if getattr(app.state, "auth_required", False):
        return None
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return None
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        return f"missing_or_empty_peer bound={bound_host or '?'}"
    if client_host in _LOOPBACK_HOSTS:
        return None
    return f"peer_not_loopback peer={client_host} bound={bound_host or '?'}"


def _ws_client_is_allowed(ws: "WebSocket") -> bool:
    """True when the peer IP passes :func:`_ws_client_reason`."""
    return _ws_client_reason(ws) is None


def _ws_host_origin_reason(ws: "WebSocket") -> Optional[str]:
    """Return a Host/Origin rejection reason (``host_mismatch …`` /
    ``origin_mismatch …``), or None when allowed.

    HTTP middleware does not run for WebSocket routes, so the DNS-rebinding
    Host check is repeated here before accepting the upgrade; a browser Origin
    header, when present, must target the same bound host.  Non-web origins
    (packaged Electron: file://, null, app://) are trusted — the upstream
    credential check is the real auth boundary there.
    """
    from hermes_cli.web_server import _is_accepted_host, app
    bound_host = getattr(app.state, "bound_host", None)
    if not bound_host:
        return None

    trusted_public_hosts = getattr(app.state, "trusted_public_hosts", frozenset())

    host_header = ws.headers.get("host", "")
    if not _is_accepted_host(host_header, bound_host, trusted_public_hosts):
        return f"host_mismatch host={host_header or '?'} bound={bound_host}"

    origin = ws.headers.get("origin", "")
    if not origin:
        return None

    parsed = urllib.parse.urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        return None

    if not parsed.netloc or not _is_accepted_host(parsed.netloc, bound_host, trusted_public_hosts):
        return f"origin_mismatch origin={origin} bound={bound_host}"
    return None


def _ws_host_origin_is_allowed(ws: "WebSocket") -> bool:
    """True when the upgrade passes the dashboard Host/Origin guard."""
    from hermes_cli.web_server import _ws_host_origin_reason
    return _ws_host_origin_reason(ws) is None


def _ws_request_is_allowed(ws: "WebSocket") -> bool:
    """Return True when the WebSocket upgrade matches dashboard boundaries."""
    return _ws_host_origin_is_allowed(ws) and _ws_client_is_allowed(ws)


_GATEWAY_WS_PROTOCOL = "hermes-gateway-v1"
_GATEWAY_WS_TICKET_PROTOCOL_PREFIX = "hermes-gateway-ticket."


def _gateway_ws_ticket_from_subprotocol(ws: "WebSocket") -> tuple[str, str]:
    """Return ``(ticket, reason)`` from an unambiguous gateway protocol set."""
    raw = str(ws.headers.get("sec-websocket-protocol", "") or "")
    protocols = [value.strip() for value in raw.split(",") if value.strip()]
    ticket_protocols = [
        value for value in protocols if value.startswith(_GATEWAY_WS_TICKET_PROTOCOL_PREFIX)]
    if not ticket_protocols:
        return "", "none"
    if _GATEWAY_WS_PROTOCOL not in protocols or len(ticket_protocols) != 1:
        return "", "invalid"
    ticket = ticket_protocols[0][len(_GATEWAY_WS_TICKET_PROTOCOL_PREFIX):]
    return (ticket, "ok") if ticket else ("", "invalid")


def _ws_auth_reason(ws: "WebSocket") -> tuple[Optional[str], str]:
    """Validate WS-upgrade auth; return ``(reason, credential)``.

    ``reason`` is None when accepted, else ``no_credential`` / ``token_mismatch``
    / ``ticket_invalid`` / ``internal_invalid``; ``credential`` names what was
    presented (``ticket``, ``ticket-subprotocol``, ``internal``, ``token``,
    ``none``) so the accept path can log *how* a peer authed.

    Loopback / ``--insecure``: legacy ``?token=<_SESSION_TOKEN>``, constant-time
    compared.  Gated: ``?ticket=`` (browser-minted, single-use, 30s TTL) or
    ``?internal=`` (process-lifetime credential used only by WS clients the
    server spawns itself — multi-use so the PTY child can reconnect; never
    injected into the SPA, see ``dashboard_auth.ws_tickets``).  The legacy
    token is unconditionally rejected in gated mode: a leaked ``_SESSION_TOKEN``
    must not grant WS access once the gate is engaged.  Rejections are
    audit-logged so "WS keeps closing" can be debugged from the log.
    """
    from hermes_cli.web_server import _SESSION_TOKEN, app
    auth_required = bool(getattr(app.state, "auth_required", False))
    if auth_required:
        # Lazy import — keeps this function importable in test harnesses
        # that don't bring in the dashboard_auth layer.
        from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
        from hermes_cli.dashboard_auth.ws_tickets import (
            TicketInvalid, consume_internal_credential, consume_ticket)

        def _reject(reason: str) -> None:
            audit_log(
                AuditEvent.WS_TICKET_REJECTED,
                reason=reason,
                ip=(ws.client.host if ws.client else ""),
                path=ws.url.path)

        def _stamp_identity(info) -> None:
            # Server-minted {user_id, provider} stamped onto the WS object is the
            # sole identity authority downstream (gateway transport / controller
            # registration); a client can never supply it through RPC params.
            # Only the two identity fields are carried — bookkeeping such as
            # ``minted_at`` is not part of the identity contract.
            ws._hermes_auth_identity = {
                "user_id": info.get("user_id"), "provider": info.get("provider")}

        internal = ws.query_params.get("internal", "")
        if internal:
            try:
                _stamp_identity(consume_internal_credential(internal))
                return None, "internal"
            except TicketInvalid as exc:
                _reject(f"internal: {exc}")
                return "internal_invalid", "internal"

        protocol_ticket, protocol_reason = _gateway_ws_ticket_from_subprotocol(ws)
        if protocol_reason == "invalid":
            return "ticket_invalid", "ticket-subprotocol"
        ticket = protocol_ticket or ws.query_params.get("ticket", "")
        if not ticket:
            return "no_credential", "none"

        try:
            _stamp_identity(consume_ticket(ticket))
            if protocol_ticket:
                # Select only the stable public protocol during accept. The
                # ticket-bearing protocol is a credential and must never be
                # reflected back to the browser or retained after admission.
                ws._hermes_ws_subprotocol = _GATEWAY_WS_PROTOCOL
                return None, "ticket-subprotocol"
            return None, "ticket"
        except TicketInvalid as exc:
            _reject(str(exc))
            return "ticket_invalid", "ticket"

    token = ws.query_params.get("token", "")
    if not token:
        return "no_credential", "none"
    if hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode()):
        return None, "token"
    return "token_mismatch", "token"


def _ws_auth_ok(ws: "WebSocket") -> bool:
    """True when the WS-upgrade credential is accepted. See _ws_auth_reason."""
    from hermes_cli.web_server import _ws_auth_reason
    return _ws_auth_reason(ws)[0] is None


def _resolve_chat_argv(
    resume: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    profile: Optional[str] = None,
    active_session_file: Optional[str] = None) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve the argv + cwd + env for the chat PTY (what ``hermes --tui`` runs).

    Tests monkeypatch this to inject a tiny fake command so nothing has to
    build the TUI bundle.  Env contract with the child:

    * ``HERMES_TUI_RESUME`` — session resume (``ui-tui`` does not parse argv, so
      ``--resume`` cannot be appended); resolved to the newest descendant first.
    * ``HERMES_TUI_GATEWAY_URL`` — attach to this process's in-memory
      ``tui_gateway`` instead of spawning a Python gateway subprocess.  SKIPPED
      for profile-scoped chats: the dashboard's gateway runs under the
      dashboard's own profile, so a scoped chat must spawn its own.
    * ``HERMES_TUI_SIDECAR_URL`` — mirror dispatcher emits to ``/api/pub``.
    * ``HERMES_TUI_ACTIVE_SESSION_FILE`` — the TUI writes its current session id
      there, a cross-process breadcrumb for reconnecting after a WS drop.
    * ``profile`` scopes the ENTIRE chat by pointing ``HERMES_HOME`` at the
      profile dir; every spawned process resolves ``get_hermes_home()`` from
      that at import, the same propagation ``hermes -p <name>`` performs.
    """
    from hermes_cli.web_server import (
        _config_profile_scope,
        _open_session_db_for_profile,
        _resolve_profile_dir,
        _session_latest_descendant)
    from hermes_cli.main import PROJECT_ROOT, _apply_tui_python_env, _make_tui_argv

    profile_dir: Optional[Path] = None
    requested = (profile or "").strip()
    if requested and requested.lower() != "current":
        profile_dir = _resolve_profile_dir(requested)

    argv, cwd = _make_tui_argv(PROJECT_ROOT / "ui-tui", tui_dev=False)
    # Build via the single spawn-env factory (profile-home contract applied;
    # secrets kept — the spawned agent needs provider creds).  An explicit
    # profile scope overrides HERMES_HOME before config is bridged into the env.
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    if profile_dir is not None:
        env["HERMES_HOME"] = str(profile_dir)
    try:
        from hermes_cli.config import (
            apply_terminal_config_to_env, read_raw_config, terminal_config_owned_env_vars)

        if profile_dir is not None:
            # The dashboard already bridged its own terminal config into
            # os.environ at startup. Remove only keys explicitly owned by that
            # launch profile before applying the selected profile; operator
            # exports for keys the launch profile omits remain valid fallbacks.
            raw_launch_terminal = read_raw_config().get("terminal")
            for env_var in terminal_config_owned_env_vars(raw_launch_terminal):
                env.pop(env_var, None)
            with _config_profile_scope(requested):
                apply_terminal_config_to_env(env=env)
        else:
            apply_terminal_config_to_env(env=env)
    except Exception:
        _log.warning("Failed to apply terminal config bridge for dashboard chat", exc_info=True)
    _apply_tui_python_env(env)
    env.setdefault("NODE_ENV", "production")
    # Mouse tracking would swallow wheel events the browser needs for
    # transcript scrolling; disable it for the dashboard PTY only.
    env.setdefault("HERMES_TUI_DISABLE_MOUSE", "1")
    env.setdefault("HERMES_TUI_INLINE", "1")
    # xterm.js always renders 24-bit RGB, but chalk in the child picks its
    # depth from the SERVER env — hosted deploys under a process manager have
    # no COLORTERM, so hex colors snap to the 256 palette (bronze -> salmon).
    # Backfill; setdefault so an explicit operator value still wins.
    env.setdefault("COLORTERM", "truecolor")
    env["HERMES_TUI_DASHBOARD"] = "1"

    if resume:
        _resume_db = _open_session_db_for_profile(
            requested if profile_dir is not None else None, read_only=True)
        try:
            latest_resume, _latest_path = _session_latest_descendant(resume, _resume_db)
        finally:
            _resume_db.close()
        if latest_resume:
            resume = latest_resume
        env["HERMES_TUI_RESUME"] = resume

    if sidecar_url:
        env["HERMES_TUI_SIDECAR_URL"] = sidecar_url

    if active_session_file:
        env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file

    # Without the attach URL, gatewayClient spawns its own `tui_gateway.entry`,
    # which inherits the profile HERMES_HOME set above.
    if profile_dir is None and (gateway_ws_url := _build_gateway_ws_url()):
        env["HERMES_TUI_GATEWAY_URL"] = gateway_ws_url

    return list(argv), str(cwd) if cwd else None, env


# Hosts that mean "listen on every interface" — bind to them, but an
# in-container client must NOT dial them: 0.0.0.0 routes through the wildcard
# stack and behind a forward proxy (HTTPS_PROXY without 0.0.0.0 in NO_PROXY)
# gets MITM'd into a failed handshake.  Clients dial loopback instead.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})


def _resolve_client_ws_host() -> Optional[str]:
    """Return the host the in-container WS client should dial.

    ``HERMES_DASHBOARD_WS_HOST`` wins always (operators behind a forward proxy
    pin a routable host); a wildcard bind becomes ``127.0.0.1`` (dashboard and
    TUI child share the container); any other bind host is preserved verbatim.
    """
    from hermes_cli.web_server import app
    explicit = os.environ.get("HERMES_DASHBOARD_WS_HOST", "").strip()
    if explicit:
        return explicit

    host = getattr(app.state, "bound_host", None)
    if not host:
        return None

    if host in _WILDCARD_HOSTS:
        return "127.0.0.1"

    return host


def _server_internal_ws_url(path: str, **extra_qs) -> Optional[str]:
    """``ws://<client host>:<port><path>?<auth>&<extra>`` for server-spawned WS
    clients, or None when unbound.

    Loopback / ``--insecure``: ``?token=<_SESSION_TOKEN>``.  Gated: the legacy
    token is rejected by ``_ws_auth_ok``, so the PTY child authenticates with
    the process-lifetime internal credential (``?internal=``) — NOT a single-use
    browser ticket: the child reads the URL once and reuses it on every
    reconnect, and a 30s-TTL ticket can expire before a slow cold boot dials.
    """
    from hermes_cli.web_server import _SESSION_TOKEN, app
    host = _resolve_client_ws_host()
    port = getattr(app.state, "bound_port", None)

    if not host or not port:
        return None

    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"

    if getattr(app.state, "auth_required", False):
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        auth = {"internal": internal_ws_credential()}
    else:
        auth = {"token": _SESSION_TOKEN}

    return f"ws://{netloc}{path}?{urllib.parse.urlencode({**auth, **extra_qs})}"


def _build_gateway_ws_url() -> Optional[str]:
    """ws:// URL the PTY child attaches to for JSON-RPC gateway traffic."""
    return _server_internal_ws_url("/api/ws")


def _build_sidecar_url(channel: str) -> Optional[str]:
    """ws:// URL the PTY child publishes events to, or None when unbound."""
    return _server_internal_ws_url("/api/pub", channel=channel)


async def _resolve_chat_argv_async(
    resume: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    profile: Optional[str] = None,
    active_session_file: Optional[str] = None) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve chat argv without blocking the dashboard event loop.

    ``_resolve_chat_argv`` may run ``npm install`` / ``npm run build``; keep
    that off the WebSocket loop so keepalives keep flowing.  The async lock
    preserves one-build-at-a-time when several tabs connect at once without
    occupying worker threads while queued connections wait.
    """
    from hermes_cli.web_server import _get_chat_argv_lock, _resolve_chat_argv, app
    kwargs = {"resume": resume, "sidecar_url": sidecar_url, "profile": profile}
    if active_session_file is not None:
        kwargs["active_session_file"] = active_session_file

    async with _get_chat_argv_lock(app):
        return await asyncio.to_thread(_resolve_chat_argv, **kwargs)


def _active_session_file_for_channel(app: "FastAPI", channel: str) -> Path:
    """Return the per-channel file where a dashboard TUI writes its active sid."""
    from hermes_cli.web_server import _get_pty_active_session_files
    files = _get_pty_active_session_files(app)
    existing = files.get(channel)
    if existing is not None:
        return existing

    fd, raw_path = tempfile.mkstemp(prefix="hermes-pty-active-", suffix=".json")
    os.close(fd)
    path = Path(raw_path)
    files[channel] = path
    return path


# Console commands run in a worker thread; on timeout asyncio cancels the
# awaitable but the thread keeps running, so a stuck worker would exhaust the
# shared default pool.  A small dedicated pool caps the leak and bounds
# concurrent console execution regardless of reconnects.
_CONSOLE_EXECUTOR_MAX_WORKERS = 4
_console_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_console_executor_lock = threading.Lock()


def _get_console_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the bounded console worker pool (once per process)."""
    global _console_executor
    if _console_executor is None:
        with _console_executor_lock:
            if _console_executor is None:
                _console_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CONSOLE_EXECUTOR_MAX_WORKERS, thread_name_prefix="hermes-console")
                # Tear down on interpreter exit without waiting on in-flight
                # workers: a stuck 60s console command must not block shutdown.
                atexit.register(
                    lambda: _console_executor
                    and _console_executor.shutdown(wait=False, cancel_futures=True))
    return _console_executor
