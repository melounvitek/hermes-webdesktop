"""Session-backed MCP OAuth flows for the gateway (mcp.servers.oauth.*).

Mirrors the *provider* OAuth model used by the dashboard rather than the
FastAPI-request-coupled MCP dashboard flow: ``start`` kicks off a background
worker and returns ``{session_id, auth_url, flow}``; ``poll`` reports
``{status: pending|approved|error}`` until tokens land on disk for that server
in that profile.

No OAuth logic is reimplemented here — the token machinery is the same one
``hermes mcp login`` uses (``_probe_single_server`` under
``force_interactive_oauth``).  ``DashboardOAuthFlow`` is reused verbatim as the
thread-safe bridge (``publish_authorization_url`` / ``deliver_callback``); the
only new piece is a tiny loopback HTTP listener on ``127.0.0.1:<port>/callback``
that feeds ``deliver_callback`` instead of a FastAPI route.

Client contract: ``start(profile, name)`` → open ``auth_url`` in the browser →
poll until ``status`` is ``approved`` (tokens persisted) or ``error``.

Remote-backend variant: when the desktop app runs on a DIFFERENT machine than
the gateway, a gateway-side ``127.0.0.1`` listener is unreachable from the
user's browser.  The client binds its OWN loopback listener, passes its
``client_redirect_uri`` to ``start``, and relays the redirect back via
``deliver_callback_flow``.  State verification stays server-side in
``DashboardOAuthFlow.deliver_callback`` — a relayed code with the wrong
``state`` is rejected exactly like a forged loopback hit.
"""

from __future__ import annotations

import http.server
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# session_id -> record wrapping the shared DashboardOAuthFlow bridge plus bookkeeping.
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()

# How long a completed/abandoned session lingers before GC (seconds).
_SESSION_TTL_SECONDS = 900
# Cap concurrent in-flight flows so a runaway client can't exhaust ports/threads.
_MAX_PENDING = 12


def _gc_sessions() -> None:
    """Drop expired sessions. Called opportunistically on start."""
    cutoff = time.time() - _SESSION_TTL_SECONDS
    with _sessions_lock:
        stale = [sid for sid, rec in _sessions.items() if rec["created_at"] < cutoff]
        for sid in stale:
            rec = _sessions.pop(sid, None)
            if rec is not None:
                _shutdown_listener(rec)


def _shutdown_listener(rec: Dict[str, Any]) -> None:
    server = rec.get("httpd")
    if server is None:
        return
    for stop in (server.shutdown, server.server_close):
        try:
            stop()
        except Exception:
            pass
    rec["httpd"] = None


def _validate_client_redirect_uri(uri: str) -> str:
    """Validate a client-supplied loopback redirect URI.

    Only plain-http loopback URLs (``http://127.0.0.1:<port>/...`` or
    ``localhost``) are accepted, per RFC 8252 native-app rules — anything else
    is rejected so the gateway can't pin an attacker-controlled redirect into a
    DCR registration.
    """
    parsed = urlparse(str(uri or "").strip())
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "http"
        or host not in ("127.0.0.1", "localhost", "::1")
        or not parsed.port
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "client_redirect_uri must be a loopback http URL like "
            "http://127.0.0.1:<port>/callback"
        )
    return f"http://{'[' + host + ']' if ':' in host else host}:{parsed.port}{parsed.path or '/callback'}"


def _start_loopback_listener(flow) -> "http.server.HTTPServer":
    """Bind a loopback callback listener that feeds the flow's deliver_callback.

    Returns the HTTPServer already serving on a daemon thread; the caller reads
    the bound port off ``server_address`` to pin ``flow.redirect_uri`` BEFORE the
    worker starts the flow (the redirect URI is fixed at authorization).
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — stdlib naming
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") not in ("/callback", ""):
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            code, state, error = ((qs.get(k) or [None])[0] for k in ("code", "state", "error"))
            body = b"<h1>Authorization received</h1><p>You can close this tab and return to Hermes.</p>"
            status = 200
            try:
                flow.deliver_callback(code=code, state=state, error=error)
            except Exception:
                body = b"<h1>OAuth callback rejected</h1><p>The callback was invalid or already used.</p>"
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass

        def log_message(self, *_a):  # silence stdlib request logging
            return

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.5},
        daemon=True,
        name=f"mcp-oauth-cb-{flow.server_name}",
    ).start()
    return httpd


def _worker(session_id: str, hermes_home: str, server_name: str, cfg: dict, reconnect_live: bool) -> None:
    """Drive the interactive MCP OAuth probe under the shared dashboard bridge.

    Same HERMES_HOME override + secret-scope + force_interactive_oauth +
    dashboard_oauth_flow wrapping around ``_probe_single_server`` as
    ``web_server._run_dashboard_mcp_oauth``, keyed to our session record.  On
    success the token file exists on disk and the server config is (re)saved
    into the profile's config.yaml; on failure the prior token/manager state is
    restored.
    """
    from hermes_cli.mcp_config import (
        _oauth_tokens_present,
        _probe_single_server,
        _save_mcp_server,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    rec = _sessions.get(session_id)
    flow = rec["flow"] if rec else None
    try:
        from agent.secret_scope import (
            build_profile_secret_scope,
            reset_secret_scope,
            set_secret_scope,
        )
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import force_interactive_oauth
        from tools.mcp_oauth_manager import get_manager

        home_token = set_hermes_home_override(hermes_home)
        secret_token = set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
        try:
            with force_interactive_oauth(), dashboard_oauth_flow(flow):
                from tools.mcp_oauth import HermesTokenStorage

                manager = get_manager()
                storage = HermesTokenStorage(server_name)
                backup = storage.snapshot()
                previous_entry = None
                try:
                    previous_entry = manager.remove(server_name, hermes_home=hermes_home)
                    tools = _probe_single_server(
                        server_name,
                        cfg,
                        connect_timeout=max(float(cfg.get("connect_timeout", 0) or 0), 315),
                    )
                    if not _oauth_tokens_present(server_name):
                        raise RuntimeError(
                            "The server responded, but no OAuth token was obtained — "
                            "this provider may require a manually-registered OAuth client."
                        )
                    _save_mcp_server(server_name, cfg)
                    if flow is not None:
                        flow.tools = [{"name": t, "description": d} for t, d in tools]
                        flow.mark_approved()
                    if reconnect_live:
                        from tools.mcp_tool import reconnect_mcp_server

                        reconnect_mcp_server(server_name)
                except Exception:
                    storage.restore(backup, only_if_absent=True)
                    manager.restore_entry(server_name, previous_entry, hermes_home=hermes_home)
                    raise
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        msg = str(exc)
        try:
            from tools.mcp_oauth import humanize_oauth_registration_error

            msg = humanize_oauth_registration_error(
                server_name, exc, server_url=cfg.get("url") if isinstance(cfg, dict) else None
            ) or msg
        except Exception:
            pass
        if flow is not None:
            flow.mark_error(msg)
    finally:
        if flow is not None:
            flow.mark_worker_done()
        if rec is not None:
            _shutdown_listener(rec)


def start_flow(
    hermes_home: str,
    server_name: str,
    cfg: dict,
    *,
    reconnect_live: bool = False,
    url_timeout: float = 30.0,
    client_redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Begin an MCP OAuth flow and return ``{session_id, auth_url, flow}``.

    ``cfg`` is the server's resolved config (must have ``url`` and be
    OAuth-capable); ``hermes_home`` the resolved profile home.  Blocks up to
    ``url_timeout`` for the worker to publish the authorization URL.

    ``client_redirect_uri`` (remote-backend variant): a loopback URL the CLIENT
    hosts.  When set and valid, no gateway-side listener is bound — the OAuth
    ``redirect_uri`` is pinned to the client's listener and the client relays
    ``code``/``state`` via ``deliver_callback_flow``.  Invalid values raise
    ``ValueError``.
    """
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    if client_redirect_uri is not None:
        client_redirect_uri = _validate_client_redirect_uri(client_redirect_uri)

    _gc_sessions()

    with _sessions_lock:
        active = [r for r in _sessions.values() if not r["flow"].worker_done]
        if len(active) >= _MAX_PENDING:
            raise RuntimeError("Too many MCP OAuth flows are already in progress")
        if any(r["server_name"] == server_name and r["hermes_home"] == hermes_home for r in active):
            raise RuntimeError(f"MCP OAuth for '{server_name}' is already in progress")

    session_id = secrets.token_urlsafe(24)
    flow = DashboardOAuthFlow(
        flow_id=session_id,
        server_name=server_name,
        profile=None,
        hermes_home=hermes_home,
        redirect_uri="",  # set below once the loopback port is known
        reconnect_live=reconnect_live,
    )
    if client_redirect_uri:
        # Client hosts the callback listener; a 127.0.0.1 port here would be
        # unreachable from the user's browser anyway.
        httpd = None
        flow.redirect_uri = client_redirect_uri
    else:
        httpd = _start_loopback_listener(flow)
        flow.redirect_uri = f"http://127.0.0.1:{httpd.server_address[1]}/callback"

    rec = {
        "session_id": session_id,
        "server_name": server_name,
        "hermes_home": hermes_home,
        "flow": flow,
        "httpd": httpd,
        "created_at": time.time(),
    }
    with _sessions_lock:
        _sessions[session_id] = rec

    threading.Thread(
        target=_worker,
        args=(session_id, hermes_home, server_name, dict(cfg), reconnect_live),
        daemon=True,
        name=f"mcp-oauth-{server_name}",
    ).start()

    try:
        auth_url = None
        # wait_for_authorization_url is async; run its wait synchronously.
        deadline = time.time() + url_timeout
        while time.time() < deadline:
            snap = flow.snapshot()
            if snap.get("authorization_url"):
                auth_url = snap["authorization_url"]
                break
            if snap.get("status") == "error":
                raise RuntimeError(snap.get("error") or "MCP OAuth flow failed before authorization")
            time.sleep(0.1)
        if not auth_url:
            raise TimeoutError("Timed out waiting for MCP authorization URL")
    except Exception:
        flow.mark_error("Timed out waiting for MCP authorization URL")
        _shutdown_listener(rec)
        raise

    return {
        "session_id": session_id,
        "auth_url": auth_url,
        # Mirrors the provider-OAuth ``flow`` discriminator: open a URL then poll
        # (no user_code to type, unlike device_code).
        "flow": "pkce",
    }


def _lookup(session_id: str, server_name: str) -> "tuple[Dict[str, Any] | None, str | None]":
    """Find a session record; returns ``(rec, None)`` or ``(None, error_message)``."""
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is None:
        return None, "OAuth session not found or expired"
    if rec["server_name"] != server_name:
        return None, "server name mismatch for session"
    return rec, None


def poll_flow(session_id: str, server_name: str) -> Dict[str, Any]:
    """Poll a session's status → ``{status, error_message?, auth_url?, tools?}``.

    ``status`` is ``pending`` | ``approved`` | ``error`` — the provider poll
    vocabulary (``authorization_required`` from the bridge maps to ``pending``
    since the client only needs to know whether to keep waiting).
    """
    rec, err = _lookup(session_id, server_name)
    if rec is None:
        return {"status": "error", "error_message": err}

    flow = rec["flow"]
    snap = flow.snapshot()
    raw = snap.get("status")
    status = raw if raw in ("approved", "error") else "pending"
    out: Dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "error_message": snap.get("error"),
        "auth_url": snap.get("authorization_url"),
    }
    if status == "approved":
        out["tools"] = list(getattr(flow, "tools", []) or [])
    return out


def deliver_callback_flow(
    session_id: str,
    server_name: str,
    *,
    code: Optional[str],
    state: Optional[str],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Relay a client-captured OAuth redirect into a session's flow.

    Remote-backend companion to ``start_flow(client_redirect_uri=...)``.
    Security is unchanged from the gateway-listener path — the underlying
    ``DashboardOAuthFlow.deliver_callback`` verifies ``state`` (constant-time)
    and rejects replays.  Returns ``{ok: true}`` or ``{ok: false, error_message}``.
    """
    rec, err = _lookup(session_id, server_name)
    if rec is None:
        return {"ok": False, "error_message": err}
    try:
        rec["flow"].deliver_callback(code=code, state=state, error=error)
    except ValueError as exc:
        return {"ok": False, "error_message": str(exc)}
    return {"ok": True, "session_id": session_id}
