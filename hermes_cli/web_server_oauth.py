"""Dashboard OAuth/login-status helpers: provider catalog, per-provider device pollers, Anthropic/Copilot/Claude-Code status probes.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import functools
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# ---------------------------------------------------------------------------
# OAuth provider endpoints — status + disconnect (Phase 1)
# ---------------------------------------------------------------------------
#
# Phase 1 surfaces *which OAuth providers exist* and whether each is
# connected, plus a disconnect button. Anthropic subscription OAuth is
# deliberately delegated away from the dashboard: its card is external and
# points to the supported terminal path. Phase 2 adds in-browser device-code
# flows for providers that support them. For unconnected providers we return
# the canonical ``hermes auth add <provider>`` command so the dashboard can
# surface a one-click copy.


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """Return ``...XXXXXX`` (last N chars) for safe display in the UI.

    We never expose more than the trailing ``visible`` characters of an
    OAuth access token. JWT prefixes (the part before the first dot) are
    stripped first when present so the visible suffix is always part of
    the signing region rather than a meaningless header chunk.

    Returns the Entra-ID placeholder when handed a callable (Azure Foundry
    bearer provider) — the callable is NEVER invoked here.
    """
    if not value:
        return ""
    if callable(value) and not isinstance(value, str):
        # Entra ID bearer provider — never reveal a minted token in the UI.
        return "<entra-id-bearer>"
    s = str(value)
    if "." in s and s.count(".") >= 2:
        # Looks like a JWT — show the trailing piece of the signature only.
        s = s.rsplit(".", 1)[-1]
    if len(s) <= visible:
        return s
    return f"…{s[-visible:]}"


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Status for the "Anthropic API Key" catalog entry.

    Two sources, in priority order:
    1. ``~/.hermes/.anthropic_oauth.json`` — Hermes-managed terminal PKCE
       credentials (the dashboard no longer has a Connect button for this)
    2. ``ANTHROPIC_API_KEY`` → ``ANTHROPIC_TOKEN`` → ``CLAUDE_CODE_OAUTH_TOKEN``
       env vars (registry order) — from ``.env``, the shell, or an external
       secret source like Bitwarden (whose keys are injected into the process
       env during ``load_hermes_dotenv()``, so the same check covers them)

    Claude Code's ``~/.claude/.credentials.json`` is deliberately NOT read
    here — it has its own dedicated catalog entry (``claude-code`` →
    ``_claude_code_only_status``). Reporting it under the API-key entry
    double-counts the token and shadows a real ANTHROPIC_API_KEY.
    """
    try:
        from agent.anthropic_adapter import (
            read_hermes_oauth_credentials,
            _get_hermes_oauth_file,
        )
    except ImportError:
        read_hermes_oauth_credentials = None  # type: ignore
        _get_hermes_oauth_file = None  # type: ignore

    hermes_creds = None
    if read_hermes_oauth_credentials:
        try:
            hermes_creds = read_hermes_oauth_credentials()
        except Exception:
            hermes_creds = None
    if hermes_creds and hermes_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "hermes_pkce",
            "source_label": f"Hermes PKCE ({_get_hermes_oauth_file() if _get_hermes_oauth_file else None})",
            "token_preview": _truncate_token(hermes_creds.get("accessToken")),
            "expires_at": hermes_creds.get("expiresAt"),
            "has_refresh_token": bool(hermes_creds.get("refreshToken")),
        }

    # Env-var / secret-source path. ``get_env_value`` checks the process
    # environment first (where Bitwarden-sourced secrets land) then .env.
    env_var_order: tuple = ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        env_var_order = PROVIDER_REGISTRY["anthropic"].api_key_env_vars
    except (ImportError, KeyError):
        pass
    try:
        from hermes_cli.config import get_env_value
    except ImportError:
        get_env_value = None  # type: ignore
    try:
        from hermes_cli.env_loader import format_secret_source_suffix
    except ImportError:
        format_secret_source_suffix = None  # type: ignore

    for var in env_var_order:
        value = (get_env_value(var) if get_env_value else None) or os.getenv(var)
        if not value:
            continue
        suffix = format_secret_source_suffix(var) if format_secret_source_suffix else ""
        return {
            "logged_in": True,
            "source": "env_var",
            "source_label": f"{var}{suffix}",
            "token_preview": _truncate_token(value),
            "expires_at": None,
            "has_refresh_token": False,
        }
    return {"logged_in": False, "source": None}


def _claude_code_only_status() -> Dict[str, Any]:
    """Surface Claude Code CLI credentials as their own provider entry.

    Independent of the Anthropic entry above so users can see whether their
    Claude Code subscription tokens are actively flowing into Hermes even
    when they also have a separate Hermes-managed PKCE login.
    """
    try:
        from agent.anthropic_adapter import read_claude_code_credentials
        creds = read_claude_code_credentials()
    except Exception:
        creds = None
    if creds and creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code_cli",
            "source_label": "~/.claude/.credentials.json",
            "token_preview": _truncate_token(creds.get("accessToken")),
            "expires_at": creds.get("expiresAt"),
            "has_refresh_token": bool(creds.get("refreshToken")),
        }
    return {"logged_in": False, "source": None}


def _copilot_acp_status() -> Dict[str, Any]:
    """Status for copilot-acp — credentials are owned by the Copilot CLI.

    ``logged_in`` is claimed only on positive evidence (a supported env token
    or a known on-disk GitHub Copilot credential store, via
    ``auth.get_external_process_provider_status``). The Copilot CLI may also
    hold its session in an OS keychain Hermes can't read, so the unverified
    state is presented as "managed by the Copilot CLI" — never as signed out.
    """
    try:
        from hermes_cli.auth import get_external_process_provider_status
        status = get_external_process_provider_status("copilot-acp") or {}
    except Exception:
        status = {}
    verified = bool(status.get("auth_verified"))
    configured = bool(status.get("configured"))
    if verified:
        source_label = status.get("auth_source") or "Copilot credentials detected"
    elif configured:
        found = status.get("resolved_command") or status.get("command") or "copilot"
        source_label = f"Managed by the GitHub Copilot CLI ({found})"
    else:
        source_label = "GitHub Copilot CLI not found on PATH"
    return {
        "logged_in": verified,
        "source": "copilot_cli",
        "source_label": source_label,
        "token_preview": None,
        "expires_at": None,
        "has_refresh_token": False,
        "configured": configured,
    }


def _external_process_cli_command(provider_id: str, default: str) -> str:
    """Render an external-process provider's sign-in command with the CLI the
    user actually has configured.

    The static catalog assumes the default executable name; users who point
    Hermes at a custom binary (``HERMES_COPILOT_ACP_COMMAND`` /
    ``COPILOT_CLI_PATH``) would otherwise be told to run a command that isn't
    the one Hermes spawns. Non-external-process providers get ``default`` back
    untouched.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, get_external_process_provider_status
        pconfig = PROVIDER_REGISTRY.get(provider_id)
        if not pconfig or pconfig.auth_type != "external_process":
            return default
        status = get_external_process_provider_status(provider_id) or {}
        command = str(status.get("command") or "").strip()
        if command:
            parts = default.split(" ", 1)
            tail = f" {parts[1]}" if len(parts) > 1 else ""
            return f"{command}{tail}"
    except Exception:
        pass
    return default


# Explicit, hand-tuned OAuth/account provider cards. These carry the bits that
# can't be derived from the unified provider catalog: the OAuth ``flow`` shape,
# the per-provider ``status_fn``, the ``cli_command`` fallback, and curated
# display order. They are the OVERRIDE BASE for ``_build_oauth_catalog()``,
# which unions them with every accounts-tab provider in ``provider_catalog()``
# so newly-added OAuth/external providers appear automatically (no hand edit).
# This tuple also still includes two entries that are NOT catalog providers but
# must show on the Accounts tab: the Anthropic credential-status card and the
# synthetic ``claude-code`` subscription row.
# ``flow`` describes the account-management shape so the UI can pick the right
# behavior: ``device_code`` = show code + verification URL + poll, and
# ``external`` = read-only/delegated to a terminal or third-party CLI.
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    {
        "id": "nous",
        "name": "Nous Portal",
        "flow": "device_code",
        "cli_command": "hermes auth add nous",
        "docs_url": "https://portal.nousresearch.com",
        "status_fn": None,  # dispatched via auth.get_nous_auth_status
    },
    {
        "id": "openai-codex",
        "name": "ChatGPT or Codex Subscription",
        "flow": "device_code",
        "cli_command": "hermes auth add openai-codex",
        "docs_url": "https://platform.openai.com/docs",
        "status_fn": None,  # dispatched via auth.get_codex_auth_status
    },
    {
        "id": "qwen-oauth",
        "name": "Qwen (via Qwen CLI)",
        "flow": "external",
        "cli_command": "hermes auth add qwen-oauth",
        "docs_url": "https://github.com/QwenLM/qwen-code",
        "status_fn": None,  # dispatched via auth.get_qwen_auth_status
    },
    {
        "id": "minimax-oauth",
        "name": "MiniMax (OAuth)",
        # MiniMax's flow is structurally device-code (verification URI +
        # user code, backend polls the token endpoint) with a PKCE
        # extension for code-binding. The dashboard renders the same UX
        # as Nous's device-code flow; the PKCE bit is a security
        # extension that doesn't change the operator experience.
        "flow": "device_code",
        "cli_command": "hermes auth add minimax-oauth",
        "docs_url": "https://www.minimax.io",
        "status_fn": None,  # dispatched via auth.get_minimax_oauth_auth_status
    },
    {
        "id": "xai-oauth",
        "name": "xAI Grok OAuth (SuperGrok / Premium+)",
        # Device code is the default because it works in remote shells,
        # containers, and desktop installs without requiring a reachable
        # 127.0.0.1 callback.
        "flow": "device_code",
        "cli_command": "hermes auth add xai-oauth",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth",
        "status_fn": None,  # dispatched via auth.get_xai_oauth_auth_status
    },
    {
        "id": "copilot-acp",
        "name": "GitHub Copilot (ACP)",
        "flow": "external",
        # `copilot login` is the CLI's non-interactive device-code login
        # subcommand; the previous `copilot /login` form is not a valid
        # invocation (slash-commands only exist inside an interactive
        # session, reachable as `copilot -i /login`).
        "cli_command": "copilot login",
        "docs_url": "https://docs.github.com/en/copilot",
        "status_fn": _copilot_acp_status,
    },
    # ── Anthropic / Claude entries sit at the bottom.
    #
    # This card is deliberately flow == "external" (no in-dashboard "Connect"
    # button walking the user through claude.ai/oauth/authorize from the web
    # server). Hermes previously reimplemented that subscription-OAuth PKCE
    # dance itself for the dashboard (issues #87887/#87888); that surface was
    # removed because it lets an unattended, scriptable HTTP endpoint mint
    # Claude Pro/Max subscription tokens outside Anthropic's own client,
    # which sits on the wrong side of Anthropic's usage policies for OAuth
    # credentials. Login still works via the terminal (`hermes auth add
    # anthropic`, unaffected by this change) or a plain API key below.
    {
        "id": "anthropic",
        "name": "Anthropic API Key",
        "flow": "external",
        "cli_command": "hermes auth add anthropic",
        "docs_url": "https://docs.claude.com/en/api/getting-started",
        "status_fn": _anthropic_oauth_status,
    },
    {
        "id": "claude-code",
        "name": "Anthropic OAuth: Required Extra Usage Credits to Use Subscription",
        "flow": "external",
        "cli_command": "claude setup-token",
        "docs_url": "https://docs.claude.com/en/docs/claude-code",
        "status_fn": _claude_code_only_status,
    },
)
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()


def _oauth_profile_name(profile: Optional[str]) -> Optional[str]:
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return None
    return requested


def _oauth_session_profile(
    session_id: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Return the profile that owns an OAuth session, if one was provided."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        profile = sess.get("profile") if sess else None
    return profile or _oauth_profile_name(fallback)


def _oauth_poller(label: str):
    """Wrap a background device-code poller body ``fn(session_id, sess)``.

    Looks up the session (a vanished session is a no-op), marks it
    ``approved`` when the body returns, and on any exception records
    ``error`` + ``error_message`` on the session instead of raising — the
    thread has no caller to report to; the dashboard reads the status.
    """
    def deco(fn):
        @functools.wraps(fn)
        def poller(session_id: str) -> None:
            with _oauth_sessions_lock:
                sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            try:
                fn(session_id, sess)
                with _oauth_sessions_lock:
                    sess["status"] = "approved"
                _log.info("oauth/device: %s login completed (session=%s)", label, session_id)
            except Exception as e:
                _log.warning("%s device-code poll failed (session=%s): %s", label, session_id, e)
                with _oauth_sessions_lock:
                    sess["status"] = "error"
                    sess["error_message"] = str(e)
        return poller
    return deco


@_oauth_poller("nous")
def _nous_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.web_server import _profile_scope
    from hermes_cli.auth import (
        _poll_for_token,
        refresh_nous_oauth_from_state,
    )
    from datetime import datetime, timezone
    import httpx
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    device_code = sess["device_code"]
    interval = sess["interval"]
    scope = sess.get("scope")
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
        token_data = _poll_for_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            device_code=device_code,
            expires_in=expires_in,
            poll_interval=interval,
        )
    # Same post-processing as _nous_device_code_login (validate/refresh JWT)
    now = datetime.now(timezone.utc)
    token_ttl = int(token_data.get("expires_in") or 0)
    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": token_data.get("inference_base_url"),
        "client_id": client_id,
        "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": (
            datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
            if token_ttl else None
        ),
        "expires_in": token_ttl,
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        full_state = refresh_nous_oauth_from_state(
            auth_state,
            timeout_seconds=15.0,
            force_refresh=False,
        )
        from hermes_cli.auth import persist_nous_credentials
        persist_nous_credentials(full_state)


@_oauth_poller("minimax")
def _minimax_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller that drives a MiniMax OAuth flow to completion.

    Mirrors `_nous_poller` but calls the MiniMax-specific token endpoint,
    which uses a PKCE-style ``code_verifier`` + ``user_code`` rather than
    the ``device_code`` field used by Nous. On success, builds the same
    auth_state dict that ``_minimax_oauth_login`` (the CLI flow) builds
    and persists via ``_minimax_save_auth_state`` — so the dashboard
    path leaves the system in the same state as
    ``hermes auth add minimax-oauth``.
    """
    from hermes_cli.web_server import _profile_scope
    from hermes_cli.auth import (
        _minimax_poll_token,
        _minimax_resolve_token_expiry_unix,
        _minimax_save_auth_state,
        MINIMAX_OAUTH_GLOBAL_INFERENCE,
        MINIMAX_OAUTH_SCOPE,
    )
    from datetime import datetime, timezone
    import httpx
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    user_code = sess["user_code"]
    code_verifier = sess["code_verifier"]
    interval_ms = sess.get("interval_ms")
    expired_in_raw = sess["expired_in_raw"]
    with httpx.Client(
        timeout=httpx.Timeout(15.0),
        headers={"Accept": "application/json"},
        follow_redirects=True,
    ) as client:
        token_data = _minimax_poll_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            user_code=user_code,
            code_verifier=code_verifier,
            expired_in=expired_in_raw,
            interval_ms=interval_ms,
        )
    # Build the auth_state dict in the same shape as the CLI flow's
    # `_minimax_oauth_login` so `_minimax_save_auth_state` writes
    # the canonical record. Region is fixed to "global" for the
    # dashboard path; cn-region operators can still use the CLI
    # flow which supports `--region cn`.
    now = datetime.now(timezone.utc)
    expires_at_ts = _minimax_resolve_token_expiry_unix(
        int(token_data["expired_in"]), now=now,
    )
    expires_in_s = max(0, int(expires_at_ts - now.timestamp()))
    auth_state = {
        "provider": "minimax-oauth",
        "region": sess.get("region", "global"),
        "portal_base_url": portal_base_url,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "client_id": client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(
            expires_at_ts, tz=timezone.utc
        ).isoformat(),
        "expires_in": expires_in_s,
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        _minimax_save_auth_state(auth_state)


@_oauth_poller("xai")
def _xai_device_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller for xAI's OAuth device-code flow."""
    from hermes_cli.web_server import _profile_scope
    import httpx
    from hermes_cli.auth import (
        _save_xai_oauth_tokens,
        _xai_oauth_discovery,
        _xai_oauth_poll_device_token,
        mark_provider_active_if_unset,
        unsuppress_credential_source,
    )

    device_code = sess["device_code"]
    interval = int(sess["interval"])
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    discovery = _xai_oauth_discovery(20.0)
    with httpx.Client(
        timeout=httpx.Timeout(20.0),
        headers={"Accept": "application/json"},
    ) as client:
        token_data = _xai_oauth_poll_device_token(
            client,
            token_endpoint=discovery["token_endpoint"],
            device_code=device_code,
            expires_in=expires_in,
            poll_interval=interval,
        )
    tokens = {
        "access_token": str(token_data.get("access_token", "") or "").strip(),
        "refresh_token": str(token_data.get("refresh_token", "") or "").strip(),
        "id_token": str(token_data.get("id_token", "") or "").strip(),
        "expires_in": token_data.get("expires_in"),
        "token_type": str(token_data.get("token_type") or "Bearer").strip() or "Bearer",
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        _save_xai_oauth_tokens(
            tokens,
            discovery=discovery,
            last_refresh=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            auth_mode="oauth_device_code",
            # Persist credentials without hijacking an existing active
            # chat provider.
            set_active=False,
        )
        # Mirror `hermes auth add xai-oauth`: first credential may become
        # active when none is set yet; never overwrite an existing choice.
        mark_provider_active_if_unset("xai-oauth")
        # The singleton write above is the single source of truth: the
        # credential-pool load seeds it as the canonical ``device_code``
        # entry. Do NOT also insert a parallel ``manual:dashboard_*`` pool
        # entry — that duplicates the single-use refresh token across two
        # entries and triggers rotation churn / ``refresh_token_reused``.
        # An interactive dashboard login is also an explicit re-enable
        # signal, so clear any ``device_code`` suppression left by a
        # prior ``hermes auth remove xai-oauth`` (mirrors auth_add_command
        # and the ``hermes model`` re-login path in _login_xai_oauth).
        unsuppress_credential_source("xai-oauth", "device_code")
