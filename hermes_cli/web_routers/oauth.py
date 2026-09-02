"""OAuth provider dashboard routes: catalog/status, disconnect, and in-browser device-code login flows.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import logging
from fastapi import APIRouter
from fastapi import HTTPException, Request
from hermes_cli.web_models import OAuthSubmitBody
from typing import Any, Dict, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()


def _oauth_provider_disconnect_command(provider: Dict[str, Any]) -> Optional[str]:
    """Shell command that clears an external provider's credentials.

    External providers store their credentials outside Hermes, so the disconnect
    API deliberately refuses them (we never delete files another CLI owns on the
    user's behalf via a silent API call). For the ones we know how to clear we
    instead hand the GUI a command it can *run in the embedded terminal* — the
    user sees exactly what executes, and Hermes then stops resolving the token.

    Claude Code has no scriptable logout (only the interactive ``/logout``), so
    we remove the credential the same way logout does: the macOS Keychain entry
    (``Claude Code-credentials``) and/or the ``~/.claude/.credentials.json``
    file — the two sources ``read_claude_code_credentials()`` consults. Returns
    None for providers we can't safely clear (the GUI shows a manual hint).
    """
    from hermes_cli.web_server import sys
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
    # "anthropic" is flow == "external" (no in-dashboard OAuth login, see the
    # catalog entry) but, unlike other external providers, Hermes still OWNS
    # the credential it can show here: the Hermes-managed PKCE file
    # (~/.hermes/.anthropic_oauth.json) and its credential-pool entry, both
    # written by `hermes auth add anthropic` in the terminal. Those are ours
    # to clear via the API, so this provider is excluded from the generic
    # "external providers can't be auto-disconnected" rule below.
    if provider.get("flow") == "external" and provider.get("id") != "anthropic":
        if _oauth_provider_disconnect_command(provider):
            # The GUI offers a one-click "run in terminal" path; this hint is the
            # fallback wording for surfaces that only show text.
            return "Managed outside Hermes — run the disconnect command to remove it."
        return "Managed by that provider's CLI; remove it there."
    if status.get("source") == "env_var":
        return "Remove the API key from Settings → Keys instead."
    return None


def _build_oauth_catalog() -> list[Dict[str, Any]]:
    """Build the Accounts-tab provider list.

    MEMBERSHIP is the union of:
      1. ``_OAUTH_PROVIDER_CATALOG`` — the explicit, hand-tuned cards that carry
         bespoke flow / status_fn / cli_command (including the api-key Anthropic
         PKCE card and the synthetic claude-code subscription row, which are not
         catalog providers), and
      2. every accounts-tab provider in the unified ``provider_catalog()`` (the
         ``hermes model`` universe) — so any OAuth/external provider added as a
         plugin appears automatically, with sensible defaults, even if no
         explicit card was written for it.

    The explicit catalog wins on metadata; the unified catalog guarantees we
    never silently drop a provider the CLI picker offers. Order: explicit cards
    first (their curated order), then any catalog-only providers appended in
    ``hermes model`` order.
    """
    from hermes_cli.web_server import _OAUTH_PROVIDER_CATALOG
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()

    # 1. Explicit hand-tuned cards (authoritative metadata + curated order).
    for entry in _OAUTH_PROVIDER_CATALOG:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        rows.append(dict(entry))

    # 2. Catalog accounts-providers not already covered — keeps the Accounts tab
    #    in lockstep with the `hermes model` universe (zero-edit for new plugins).
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

    Response shape (per provider):
        id              stable identifier (used in DELETE path)
        name            human label
        flow            "device_code" | "external"
        cli_command     fallback CLI command for users to run manually
        disconnect_command  shell command that clears an external provider's
                            creds (run in the embedded terminal), else null
        docs_url        external docs/portal link for the "Learn more" link
        status:
          logged_in        bool — currently has usable creds
          source           short slug ("hermes_pkce", "claude_code", ...)
          source_label     human-readable origin (file path, env var name)
          token_preview    last N chars of the token, never the full token
          expires_at       ISO timestamp string or null
          has_refresh_token bool

    Membership is derived from the unified provider_catalog() so this stays in
    sync with the `hermes model` picker; _OAUTH_OVERRIDES supplies per-provider
    flow/status/cli metadata.
    """
    from hermes_cli.web_server import (
        _external_process_cli_command,
        _profile_scope,
        _resolve_provider_status,
        asyncio,
    )
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


@router.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(
    provider_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    from hermes_cli.web_server import (
        _profile_scope,
        _require_token,
        _resolve_provider_status,
        asyncio,
    )
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

            disconnect_hint = _oauth_provider_disconnect_hint(provider, {})
            if disconnect_hint:
                raise HTTPException(
                    status_code=400,
                    detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
                )

            status = _resolve_provider_status(provider_id, provider.get("status_fn"))
            disconnect_hint = _oauth_provider_disconnect_hint(provider, status)
            if disconnect_hint:
                raise HTTPException(
                    status_code=400,
                    detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
                )

            # Anthropic clears only the Hermes-managed PKCE file and auth-store entry.
            # The separate claude-code catalog row is external/read-only and rejected
            # above so we never pretend to remove ~/.claude/* credentials owned by the CLI.
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
                # Also clear the credential pool entry if present.
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


# ---------------------------------------------------------------------------
# OAuth Phase 2 — in-browser device-code flows
# ---------------------------------------------------------------------------
#
# Anthropic previously had a dashboard-triggered PKCE flow here too (server
# generates a claude.ai/oauth/authorize URL, exchanges the code for tokens at
# the Anthropic token endpoint, persists them). It was removed: an unattended
# HTTP endpoint minting Claude Pro/Max subscription tokens outside Anthropic's
# own client sits on the wrong side of Anthropic's usage policies for OAuth
# credentials. The "anthropic" catalog entry is now flow == "external" and
# points at `hermes auth add anthropic` (terminal PKCE, unaffected) instead.
#
#   Device code (Nous, OpenAI Codex):
#     1. POST /api/providers/oauth/{nous|openai-codex}/start
#          → server hits provider's device-auth endpoint
#          → gets { user_code, verification_url, device_code, interval, expires_in }
#          → spawns background poller thread that polls the token endpoint
#            every `interval` seconds until approved/expired
#          → stores poll status in _oauth_sessions[session_id]
#          → returns { session_id, flow: "device_code", user_code,
#                      verification_url, expires_in, poll_interval }
#     2. UI opens verification_url in a new tab and shows user_code.
#     3. UI polls GET /api/providers/oauth/{provider}/poll/{session_id}
#          every 2s until status != "pending".
#     4. On "approved" the background thread has already saved creds; UI
#        refreshes the providers list.
#
# Sessions are kept in-memory only (single-process FastAPI) and time out
# after 15 minutes. A periodic cleanup runs on each /start call to GC
# expired sessions so the dict doesn't grow without bound.

_OAUTH_SESSION_TTL_SECONDS = 15 * 60


def _gc_oauth_sessions() -> None:
    """Drop expired sessions. Called opportunistically on /start."""
    from hermes_cli.web_server import _oauth_sessions, _oauth_sessions_lock, time
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        stale = [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]
        for sid in stale:
            _oauth_sessions.pop(sid, None)


def _validate_oauth_profile(profile: Optional[str]) -> None:
    from hermes_cli.web_server import _oauth_profile_name, _resolve_profile_dir
    profile_name = _oauth_profile_name(profile)
    if profile_name:
        _resolve_profile_dir(profile_name)


@router.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(
    provider_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Initiate an OAuth login flow. Token-protected."""
    from hermes_cli.web_server import (
        _OAUTH_PROVIDER_CATALOG,
        _require_token,
        _start_device_code_flow,
    )
    _require_token(request)
    _gc_oauth_sessions()
    _validate_oauth_profile(profile)
    valid = {p["id"] for p in _OAUTH_PROVIDER_CATALOG}
    if provider_id not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    catalog_entry = next(p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id)
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
    provider_id: str,
    body: OAuthSubmitBody,
    request: Request,
    profile: Optional[str] = None,
):
    """Submit the auth code for PKCE flows. Token-protected."""
    from hermes_cli.web_server import _require_token
    _require_token(request)
    raise HTTPException(status_code=400, detail=f"submit not supported for {provider_id}")


@router.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(
    provider_id: str,
    session_id: str,
    profile: Optional[str] = None,
):
    """Poll a session's status (no auth — read-only state).

    Shared by the device-code flows (Nous, OpenAI Codex, MiniMax, xAI).
    Each surfaces progress through the same background-worker-updated
    ``status`` field, so a single poll endpoint serves them all.
    """
    from hermes_cli.web_server import _oauth_profile_name, _oauth_sessions, _oauth_sessions_lock
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
    session_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Cancel a pending OAuth session. Token-protected.

    Marks the session dict ``cancelled`` before popping it so any
    background worker still holding a reference to that same dict (e.g.
    the Codex device-code poller) observes the cancellation and stops
    polling/exchanging/saving instead of completing the login after the
    user believed it was aborted.
    """
    from hermes_cli.web_server import (
        _oauth_profile_name,
        _oauth_sessions,
        _oauth_sessions_lock,
        _require_token,
    )
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
