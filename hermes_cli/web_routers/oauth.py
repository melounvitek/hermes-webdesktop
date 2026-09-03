"""OAuth provider dashboard routes: catalog/status, disconnect, and in-browser device-code login flows.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import logging
import asyncio
import sys
import time
from fastapi import APIRouter
from hermes_cli.web_deps import LateState, late
from fastapi import HTTPException, Request
from hermes_cli.web_models import OAuthSubmitBody
from typing import Any, Callable, Dict, Optional
import threading
import os
import secrets

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# web_server helpers/state, late-bound so monkeypatch.setattr(web_server, ...) stays authoritative.
_external_process_cli_command = late("_external_process_cli_command")
_oauth_profile_name = late("_oauth_profile_name")
_profile_scope = late("_profile_scope")
_require_token = late("_require_token")
_resolve_profile_dir = late("_resolve_profile_dir")
_minimax_poller = late("_minimax_poller")
_nous_poller = late("_nous_poller")
_truncate_token = late("_truncate_token")
_xai_device_poller = late("_xai_device_poller")
_oauth_sessions = LateState("_oauth_sessions")
_oauth_sessions_lock = LateState("_oauth_sessions_lock")
_OAUTH_PROVIDER_CATALOG = LateState("_OAUTH_PROVIDER_CATALOG")


def _http_response_error_detail(resp: Any) -> str:
    """Best-effort extraction of a short provider error detail."""
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts = [
                str(error.get(key, "")).strip()
                for key in ("message", "error_description", "code", "type")
                if str(error.get(key, "")).strip()
            ]
            if parts:
                return ": ".join(parts)
        if isinstance(error, str) and error.strip():
            return error.strip()
        for key in ("detail", "message", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = str(getattr(resp, "text", "") or "").strip()
    return text[:500]


def _codex_device_code_start_error(resp: Any) -> str:
    """Dashboard-facing OpenAI Codex device-code start failure."""
    status = getattr(resp, "status_code", "unknown")
    detail = _http_response_error_detail(resp)
    lower = detail.lower()
    if "device" in lower and ("authori" in lower or "enable" in lower):
        message = (
            "OpenAI rejected the device-code login request. Your OpenAI "
            "account may need device-code authorization enabled before Hermes "
            "can start this dashboard login. Enable device-code authorization "
            "in OpenAI, then return here and click Login again."
        )
    else:
        message = (
            "OpenAI rejected the device-code login request. Please try Login "
            "again from the dashboard after checking your OpenAI account settings."
        )
    if detail:
        return f"{message} (HTTP {status}: {detail})"
    return f"{message} (HTTP {status})"


def _new_oauth_session(
    provider_id: str, flow: str, profile: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    sess = {
        "session_id": sid,
        "provider": provider_id,
        "flow": flow,
        "profile": _oauth_profile_name(profile),
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _start_poller(target, sid: str) -> None:
    threading.Thread(target=target, args=(sid,), daemon=True, name=f"oauth-poll-{sid[:6]}").start()


def _codex_full_login_worker(session_id: str) -> None:
    """Run the complete OpenAI Codex device-code flow.

    Codex has its own ``/api/accounts/deviceauth/usercode`` (returns
    ``device_auth_id``) and ``/api/accounts/deviceauth/token`` (polled until
    200) endpoints; success yields ``authorization_code`` + ``code_verifier``
    exchanged at CODEX_OAUTH_TOKEN_URL. Replicated inline rather than calling
    ``_codex_device_code_login`` because that helper prints/blocks/polls in one
    function — the dashboard needs the user_code before polling completes.
    """
    try:
        import httpx
        from hermes_cli.auth import CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL
        issuer = "https://auth.openai.com"

        # Step 1: request device code
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                f"{issuer}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code != 200:
            raise RuntimeError(_codex_device_code_start_error(resp))
        device_data = resp.json()
        user_code = device_data.get("user_code", "")
        device_auth_id = device_data.get("device_auth_id", "")
        poll_interval = max(3, int(device_data.get("interval", "5")))
        if not user_code or not device_auth_id:
            raise RuntimeError("device-code response missing user_code or device_auth_id")
        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            sess["user_code"] = user_code
            sess["verification_url"] = f"{issuer}/codex/device"
            sess["device_auth_id"] = device_auth_id
            sess["interval"] = poll_interval
            sess["expires_in"] = 15 * 60  # OpenAI's effective limit
            sess["expires_at"] = time.time() + sess["expires_in"]
            # Captured now (not re-derived after cancel pops the session) so a
            # cancelled session can never fall back to the caller's current
            # profile scope at save time.
            session_profile = sess.get("profile")

        # Step 2: poll until authorized
        deadline = time.monotonic() + sess["expires_in"]
        code_resp = None
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while time.monotonic() < deadline:
                if sess.get("cancelled"):
                    _log.info("oauth/device: openai-codex login cancelled (session=%s)", session_id)
                    return
                time.sleep(poll_interval)
                if sess.get("cancelled"):
                    _log.info("oauth/device: openai-codex login cancelled (session=%s)", session_id)
                    return
                poll = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )
                if poll.status_code == 200:
                    code_resp = poll.json()
                    break
                if poll.status_code in {403, 404}:
                    continue  # user hasn't authorized yet
                raise RuntimeError(f"deviceauth/token poll returned {poll.status_code}")

        if code_resp is None:
            with _oauth_sessions_lock:
                sess["status"] = "expired"
                sess["error_message"] = "Device code expired before approval"
            return

        if sess.get("cancelled"):
            _log.info("oauth/device: openai-codex login cancelled before token exchange (session=%s)", session_id)
            return

        # Step 3: exchange authorization_code for tokens
        authorization_code = code_resp.get("authorization_code", "")
        code_verifier = code_resp.get("code_verifier", "")
        if not authorization_code or not code_verifier:
            raise RuntimeError("device-auth response missing authorization_code/code_verifier")
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": f"{issuer}/deviceauth/callback",
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if token_resp.status_code != 200:
            raise RuntimeError(f"token exchange returned {token_resp.status_code}")
        tokens = token_resp.json()
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        if not access_token:
            raise RuntimeError("token exchange did not return access_token")

        from hermes_cli.auth import _save_codex_tokens

        # The cancellation check and the save are one atomic critical section
        # under the lock cancel_oauth_session() uses; otherwise DELETE could
        # flip "cancelled" between the check and the save and tokens would be
        # persisted after the user believed the login was aborted.
        with _oauth_sessions_lock:
            if sess.get("cancelled"):
                _log.info("oauth/device: openai-codex login cancelled before token save (session=%s)", session_id)
                return
            with _profile_scope(session_profile):
                _save_codex_tokens({"access_token": access_token, "refresh_token": refresh_token})
            sess["status"] = "approved"
        _log.info("oauth/device: openai-codex login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("codex device-code worker failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id)
            if s:
                s["status"] = "error"
                s["error_message"] = str(e)


# Hand-written status card shapes per provider id: (hauth getter name, shaper).
# Providers absent here fall through to the slug-driven ``get_auth_status``.
def _nous_status(raw):
    # Refresh-free snapshot so listing providers never performs an OAuth refresh.
    return {
        "logged_in": bool(raw.get("logged_in")),
        "source": "nous_portal",
        "source_label": raw.get("portal_base_url") or "Nous Portal",
        "token_preview": _truncate_token(raw.get("access_token")),
        "expires_at": raw.get("access_expires_at"),
        "has_refresh_token": bool(raw.get("has_refresh_token")),
    }


def _codex_status(raw):
    return {
        "logged_in": bool(raw.get("logged_in")),
        "source": raw.get("source") or "openai_codex",
        "source_label": raw.get("auth_mode") or "OpenAI Codex",
        "token_preview": _truncate_token(raw.get("api_key")),
        "expires_at": None,
        "has_refresh_token": False,
        "last_refresh": raw.get("last_refresh"),
    }


def _qwen_status(raw):
    return {
        "logged_in": bool(raw.get("logged_in")),
        "source": "qwen_cli",
        "source_label": raw.get("auth_store_path") or "Qwen CLI",
        "token_preview": _truncate_token(raw.get("access_token")),
        "expires_at": raw.get("expires_at"),
        "has_refresh_token": bool(raw.get("has_refresh_token")),
    }


def _minimax_status(raw):
    return {
        "logged_in": bool(raw.get("logged_in")),
        "source": "minimax_oauth",
        "source_label": f"MiniMax ({raw.get('region', 'global')})",
        "token_preview": None,
        "expires_at": raw.get("expires_at"),
        "has_refresh_token": True,
    }


def _xai_status(raw):
    # source_label is a human-readable origin (auth-store path / credential
    # source), not the internal auth_mode string ("oauth_pkce").
    return {
        "logged_in": bool(raw.get("logged_in")),
        "source": raw.get("source") or "xai_oauth",
        "source_label": raw.get("auth_store") or raw.get("source") or "xAI Grok OAuth",
        "token_preview": _truncate_token(raw.get("api_key")),
        "expires_at": None,
        "has_refresh_token": True,
        "last_refresh": raw.get("last_refresh"),
    }


_PROVIDER_STATUS: Dict[str, tuple[str, Callable[[dict], dict]]] = {
    "nous": ("get_nous_auth_status_local", _nous_status),
    "openai-codex": ("get_codex_auth_status", _codex_status),
    "qwen-oauth": ("get_qwen_auth_status", _qwen_status),
    "minimax-oauth": ("get_minimax_oauth_auth_status", _minimax_status),
    "xai-oauth": ("get_xai_oauth_auth_status", _xai_status),
}


def _resolve_provider_status(provider_id: str, status_fn) -> Dict[str, Any]:
    """Dispatch to the right status helper for an OAuth provider entry."""
    if status_fn is not None:
        try:
            return status_fn()
        except Exception as e:
            return {"logged_in": False, "error": str(e)}
    try:
        from hermes_cli import auth as hauth
        entry = _PROVIDER_STATUS.get(provider_id)
        if entry is not None:
            getter, shape = entry
            return shape(getattr(hauth, getter)())
        # Catalog-derived providers (status_fn=None, no hand-written card) still
        # reflect real login state via the canonical slug-driven dispatcher, so
        # a new OAuth/account provider plugin never renders permanently logged-out.
        raw = hauth.get_auth_status(provider_id)
        if isinstance(raw, dict) and "logged_in" in raw:
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or raw.get("provider") or provider_id,
                "source_label": (
                    raw.get("source_label")
                    or raw.get("auth_store")
                    or raw.get("auth_store_path")
                    or raw.get("base_url")
                    or raw.get("name")
                    or ""
                ),
                "token_preview": _truncate_token(raw.get("access_token") or raw.get("api_key")),
                "expires_at": raw.get("expires_at") or raw.get("access_expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


async def _start_nous_device_code(profile: Optional[str]) -> Dict[str, Any]:
    from hermes_cli.auth import _request_device_code, PROVIDER_REGISTRY
    import httpx
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url
    ).rstrip("/")
    client_id = pconfig.client_id
    scope = pconfig.scope

    def _do_nous_device_request():
        with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
            return (
                _request_device_code(
                    client=client, portal_base_url=portal_base_url, client_id=client_id, scope=scope
                ),
                scope,
            )

    device_data, effective_scope = await asyncio.get_running_loop().run_in_executor(None, _do_nous_device_request)
    sid, sess = _new_oauth_session("nous", "device_code", profile=profile)
    sess.update(
        device_code=str(device_data["device_code"]), interval=int(device_data["interval"]),
        expires_at=time.time() + int(device_data["expires_in"]), portal_base_url=portal_base_url,
        client_id=client_id, scope=effective_scope,
    )
    _start_poller(_nous_poller, sid)
    return {
        "session_id": sid,
        "flow": "device_code",
        "user_code": str(device_data["user_code"]),
        "verification_url": str(device_data["verification_uri_complete"]),
        "expires_in": int(device_data["expires_in"]),
        "poll_interval": int(device_data["interval"]),
    }


async def _start_codex_device_code(profile: Optional[str]) -> Dict[str, Any]:
    # The full Codex helper polls inline, so it runs in a worker thread and
    # proxies user_code + verification_url back via the session dict.
    sid, _ = _new_oauth_session("openai-codex", "device_code", profile=profile)
    threading.Thread(target=_codex_full_login_worker, args=(sid,), daemon=True, name=f"oauth-codex-{sid[:6]}").start()
    # Block briefly until the worker has populated the user_code, OR error.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(sid)
        if s and (s.get("user_code") or s["status"] != "pending"):
            break
        await asyncio.sleep(0.1)
    with _oauth_sessions_lock:
        s = _oauth_sessions.get(sid, {})
    if s.get("status") == "error":
        raise HTTPException(status_code=500, detail=s.get("error_message") or "device-auth failed")
    if not s.get("user_code"):
        raise HTTPException(status_code=504, detail="device-auth timed out before returning a user code")
    return {
        "session_id": sid,
        "flow": "device_code",
        "user_code": s["user_code"],
        "verification_url": s["verification_url"],
        "expires_in": int(s.get("expires_in") or 900),
        "poll_interval": int(s.get("interval") or 5),
    }


async def _start_minimax_device_code(profile: Optional[str]) -> Dict[str, Any]:
    # Device-code flow with a PKCE extension: verifier + challenge from
    # _minimax_pkce_pair bind the token exchange to the original session.
    from hermes_cli.auth import (
        _minimax_pkce_pair, _minimax_request_user_code, MINIMAX_OAUTH_CLIENT_ID, MINIMAX_OAUTH_GLOBAL_BASE,
    )
    import httpx
    verifier, challenge, state = _minimax_pkce_pair()
    portal_base_url = (os.getenv("MINIMAX_PORTAL_BASE_URL") or MINIMAX_OAUTH_GLOBAL_BASE).rstrip("/")

    def _do_minimax_request():
        with httpx.Client(
            timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}, follow_redirects=True
        ) as client:
            return _minimax_request_user_code(
                client=client, portal_base_url=portal_base_url, client_id=MINIMAX_OAUTH_CLIENT_ID,
                code_challenge=challenge, state=state,
            )

    device_data = await asyncio.get_event_loop().run_in_executor(None, _do_minimax_request)
    sid, sess = _new_oauth_session("minimax-oauth", "device_code", profile=profile)
    # MiniMax's `interval` is in milliseconds (defensive default 2000ms in _minimax_poll_token).
    interval_raw = device_data.get("interval")
    sess.update(
        interval_ms=int(interval_raw) if interval_raw is not None else None,
        user_code=str(device_data["user_code"]), code_verifier=verifier, state=state,
        portal_base_url=portal_base_url, client_id=MINIMAX_OAUTH_CLIENT_ID, region="global",
    )
    # `expired_in` is overloaded — a unix-ms timestamp OR seconds-from-now.
    # Mirror the heuristic in _minimax_poll_token; keep the raw value for the
    # poller and derive expires_at + UI-friendly expires_in seconds.
    expired_in_raw = int(device_data["expired_in"])
    sess["expired_in_raw"] = expired_in_raw
    if expired_in_raw > 1_000_000_000_000:  # likely unix-ms
        expires_at_ts = expired_in_raw / 1000.0
        expires_in_seconds = max(0, int(expires_at_ts - time.time()))
    else:
        expires_at_ts = time.time() + expired_in_raw
        expires_in_seconds = expired_in_raw
    sess["expires_at"] = expires_at_ts
    _start_poller(_minimax_poller, sid)
    return {
        "session_id": sid,
        "flow": "device_code",
        "user_code": str(device_data["user_code"]),
        "verification_url": str(device_data["verification_uri"]),
        "expires_in": expires_in_seconds,
        "poll_interval": max(2, (sess["interval_ms"] or 2000) // 1000),
    }


async def _start_xai_device_code(profile: Optional[str]) -> Dict[str, Any]:
    from hermes_cli.auth import _xai_oauth_request_device_code
    import httpx

    def _do_xai_device_request():
        with httpx.Client(timeout=httpx.Timeout(20.0), headers={"Accept": "application/json"}) as client:
            return _xai_oauth_request_device_code(client)

    device_data = await asyncio.get_running_loop().run_in_executor(None, _do_xai_device_request)
    sid, sess = _new_oauth_session("xai-oauth", "device_code", profile=profile)
    sess.update(
        device_code=str(device_data["device_code"]), interval=int(device_data["interval"]),
        expires_at=time.time() + int(device_data["expires_in"]),
    )
    _start_poller(_xai_device_poller, sid)
    return {
        "session_id": sid,
        "flow": "device_code",
        "user_code": str(device_data["user_code"]),
        "verification_url": str(device_data.get("verification_uri_complete") or device_data["verification_uri"]),
        "expires_in": int(device_data["expires_in"]),
        "poll_interval": int(device_data["interval"]),
    }


_DEVICE_CODE_STARTERS = {
    "nous": _start_nous_device_code,
    "openai-codex": _start_codex_device_code,
    "minimax-oauth": _start_minimax_device_code,
    "xai-oauth": _start_xai_device_code,
}


async def _start_device_code_flow(
    provider_id: str, profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Initiate a device-code flow (Nous, OpenAI Codex, MiniMax, or xAI).

    Calls the provider's device-auth endpoint via the CLI helpers, then spawns
    a background poller. Returns the user-facing display fields (verification
    link + user code).
    """
    starter = _DEVICE_CODE_STARTERS.get(provider_id)
    if starter is None:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support device-code flow")
    return await starter(profile)


def _oauth_provider_disconnect_command(provider: Dict[str, Any]) -> Optional[str]:
    """Shell command that clears an external provider's credentials.

    External providers store credentials outside Hermes, so the disconnect API
    refuses them (never silently delete files another CLI owns); instead the
    GUI gets a command to run in the embedded terminal, so the user sees
    exactly what executes. Claude Code has no scriptable logout, so we remove
    the credential the way logout does: the macOS Keychain entry
    (``Claude Code-credentials``) and/or ``~/.claude/.credentials.json`` — the
    two sources ``read_claude_code_credentials()`` consults. None for providers
    we can't safely clear (the GUI shows a manual hint).
    """
    if provider.get("flow") != "external":
        return None
    if provider.get("id") == "claude-code":
        rm_file = "rm -f ~/.claude/.credentials.json"
        if sys.platform == "darwin":
            return f'security delete-generic-password -s "Claude Code-credentials" 2>/dev/null; {rm_file}'
        return rm_file
    return None


def _oauth_provider_disconnect_hint(provider: Dict[str, Any], status: Dict[str, Any]) -> Optional[str]:
    """Return the manual disconnect path when the API cannot clear this provider."""
    # "anthropic" is flow == "external" (no in-dashboard login) but Hermes still
    # OWNS its credential (the PKCE file ~/.hermes/.anthropic_oauth.json and its
    # credential-pool entry, written by `hermes auth add anthropic`), so it is
    # excluded from the "external providers can't be auto-disconnected" rule.
    if provider.get("flow") == "external" and provider.get("id") != "anthropic":
        if _oauth_provider_disconnect_command(provider):
            # Fallback wording for surfaces without the one-click "run in terminal" path.
            return "Managed outside Hermes — run the disconnect command to remove it."
        return "Managed by that provider's CLI; remove it there."
    if status.get("source") == "env_var":
        return "Remove the API key from Settings → Keys instead."
    return None


def _build_oauth_catalog() -> list[Dict[str, Any]]:
    """Build the Accounts-tab provider list.

    Membership is the union of ``_OAUTH_PROVIDER_CATALOG`` (hand-tuned cards
    with bespoke flow / status_fn / cli_command, incl. the Anthropic PKCE card
    and the synthetic claude-code row) and every accounts-tab provider in the
    unified ``provider_catalog()``, so a plugin-added OAuth/external provider
    appears automatically. Explicit cards win on metadata and come first in
    curated order; catalog-only providers follow in ``hermes model`` order.
    """
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()

    for entry in _OAUTH_PROVIDER_CATALOG:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        rows.append(dict(entry))

    try:
        from hermes_cli.provider_catalog import provider_catalog
        for d in provider_catalog():
            if d.tab != "accounts" or d.slug in seen:
                continue
            seen.add(d.slug)
            rows.append({
                "id": d.slug,
                "name": d.label,
                "flow": "external",
                "cli_command": f"hermes auth add {d.slug}",
                "docs_url": d.signup_url or "",
                "status_fn": None,
            })
    except Exception:
        pass

    return rows


@router.get("/api/providers/oauth")
async def list_oauth_providers(profile: Optional[str] = None):
    """Enumerate every OAuth-capable LLM provider with current status.

    Per provider: id (used in the DELETE path), name, flow
    ("device_code" | "external"), cli_command (manual fallback),
    disconnect_command (shell command for external providers, else null),
    docs_url, and status {logged_in, source slug, source_label, token_preview
    (last N chars, never the full token), expires_at, has_refresh_token}.
    """
    def _run():
        with _profile_scope(profile):
            providers = []
            for p in _build_oauth_catalog():
                status = _resolve_provider_status(p["id"], p.get("status_fn"))
                disconnect_hint = _oauth_provider_disconnect_hint(p, status)
                providers.append({
                    "id": p["id"],
                    "name": p["name"],
                    "flow": p["flow"],
                    "cli_command": _external_process_cli_command(p["id"], p["cli_command"]),
                    "docs_url": p["docs_url"],
                    "disconnect_hint": disconnect_hint,
                    "disconnect_command": _oauth_provider_disconnect_command(p),
                    "disconnectable": disconnect_hint is None,
                    "status": status,
                })
            return {"providers": providers}

    return await asyncio.to_thread(_run)


def _reject_if_not_disconnectable(provider: Dict[str, Any], status: Dict[str, Any]) -> None:
    disconnect_hint = _oauth_provider_disconnect_hint(provider, status)
    if disconnect_hint:
        raise HTTPException(
            status_code=400,
            detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
        )


@router.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(
    provider_id: str, request: Request, profile: Optional[str] = None,
):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    def _run():
        with _profile_scope(profile):
            catalog_by_id = {p["id"]: p for p in _build_oauth_catalog()}
            provider = catalog_by_id.get(provider_id)
            if provider is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown provider: {provider_id}. "
                           f"Available: {', '.join(sorted(catalog_by_id))}",
                )

            _reject_if_not_disconnectable(provider, {})
            _reject_if_not_disconnectable(
                provider, _resolve_provider_status(provider_id, provider.get("status_fn"))
            )

            # Anthropic clears only the Hermes-managed PKCE file and auth-store
            # entry; the external claude-code row was rejected above so we never
            # pretend to remove ~/.claude/* credentials owned by the CLI.
            if provider_id == "anthropic":
                cleared = False
                try:
                    from agent.anthropic_adapter import _get_hermes_oauth_file
                    oauth_file = _get_hermes_oauth_file()
                    if oauth_file.exists():
                        oauth_file.unlink()
                        cleared = True
                except Exception:
                    pass
                try:
                    from hermes_cli.auth import clear_provider_auth
                    cleared = clear_provider_auth("anthropic") or cleared
                except Exception:
                    pass
                _log.info("oauth/disconnect: %s", provider_id)
                return {"ok": bool(cleared), "provider": provider_id}

            try:
                from hermes_cli.auth import clear_provider_auth, invalidate_nous_auth_status_cache
                cleared = clear_provider_auth(provider_id)
                if provider_id == "nous":
                    invalidate_nous_auth_status_cache()
                _log.info("oauth/disconnect: %s (cleared=%s)", provider_id, cleared)
                return {"ok": bool(cleared), "provider": provider_id}
            except Exception as e:
                _log.exception("disconnect %s failed", provider_id)
                raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_run)


# In-browser device-code flows (Nous, OpenAI Codex, MiniMax, xAI):
#   1. POST /api/providers/oauth/{provider}/start hits the provider's
#      device-auth endpoint, spawns a poller thread that polls the token
#      endpoint every `interval` seconds, and returns
#      {session_id, flow, user_code, verification_url, expires_in, poll_interval}.
#   2. UI opens verification_url and shows user_code, then polls
#      GET /api/providers/oauth/{provider}/poll/{session_id} until status != "pending".
#   3. On "approved" the poller has already saved creds; UI refreshes the list.
# Anthropic has NO dashboard PKCE flow: an unattended endpoint minting Claude
# subscription tokens outside Anthropic's own client violates its OAuth usage
# policy; that card is flow == "external" (`hermes auth add anthropic`).
# Sessions are in-memory (single-process) and expire after 15 minutes; /start
# GCs expired ones so the dict is bounded.

_OAUTH_SESSION_TTL_SECONDS = 15 * 60


def _gc_oauth_sessions() -> None:
    """Drop expired sessions. Called opportunistically on /start."""
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        stale = [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]
        for sid in stale:
            _oauth_sessions.pop(sid, None)


def _validate_oauth_profile(profile: Optional[str]) -> None:
    profile_name = _oauth_profile_name(profile)
    if profile_name:
        _resolve_profile_dir(profile_name)


@router.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(
    provider_id: str, request: Request, profile: Optional[str] = None,
):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _gc_oauth_sessions()
    _validate_oauth_profile(profile)
    catalog_entry = next((p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id), None)
    if catalog_entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    if catalog_entry["flow"] == "external":
        raise HTTPException(
            status_code=400,
            detail=f"{provider_id} uses an external CLI; run `{catalog_entry['cli_command']}` manually",
        )
    try:
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id, profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("oauth/start %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported flow")


@router.post("/api/providers/oauth/{provider_id}/submit")
async def submit_oauth_code(
    provider_id: str, body: OAuthSubmitBody, request: Request, profile: Optional[str] = None,
):
    """Submit the auth code for PKCE flows. Token-protected."""
    _require_token(request)
    raise HTTPException(status_code=400, detail=f"submit not supported for {provider_id}")


@router.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(
    provider_id: str, session_id: str, profile: Optional[str] = None,
):
    """Poll a session's status (no auth — read-only state). One endpoint serves
    every device-code flow: all report progress via the worker-updated ``status``."""
    _validate_oauth_profile(profile)
    requested_profile = _oauth_profile_name(profile)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["provider"] != provider_id:
        raise HTTPException(status_code=400, detail="Provider mismatch for session")
    if sess.get("profile") != requested_profile:
        raise HTTPException(status_code=400, detail="OAuth session profile mismatch")
    return {
        "session_id": session_id,
        "status": sess["status"],
        "error_message": sess.get("error_message"),
        "expires_at": sess.get("expires_at"),
    }


@router.delete("/api/providers/oauth/sessions/{session_id}")
async def cancel_oauth_session(
    session_id: str, request: Request, profile: Optional[str] = None,
):
    """Cancel a pending OAuth session. Token-protected.

    Marks the session dict ``cancelled`` before popping it so a background
    worker still holding that dict (e.g. the Codex poller) stops
    polling/exchanging/saving instead of completing the login after the user
    believed it was aborted.
    """
    _require_token(request)
    _validate_oauth_profile(profile)
    requested_profile = _oauth_profile_name(profile)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        if sess is not None:
            if sess.get("profile") != requested_profile:
                raise HTTPException(status_code=400, detail="OAuth session profile mismatch")
            sess["cancelled"] = True
            _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    return {"ok": True, "session_id": session_id}
