"""xAI Grok OAuth: token store, discovery, refresh, device-code login.

Split out of ``hermes_cli/auth.py``; every moved name is re-imported there, so
``hermes_cli.auth.<name>`` keeps resolving (and monkeypatching) as before. Origin-internal
helpers are imported lazily inside each function (no import cycle; patches on
``hermes_cli.auth.<helper>`` still intercept).
"""

from __future__ import annotations

import logging
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from hermes_cli.auth_codex import _load_auth_store_maybe_locked, _refresh_payload_access_token
from hermes_cli.auth_constants import (
    AUTH_LOCK_TIMEOUT_SECONDS,
    AuthError,
    DEFAULT_XAI_OAUTH_BASE_URL,
    DEVICE_CODE_GRANT_TYPE,
    XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_DEVICE_CODE_URL,
    XAI_OAUTH_DISCOVERY_URL,
    XAI_OAUTH_SCOPE,
    _FORM_JSON_HEADERS,
    _xai_err,
    httpx,
)
from utils import env_float

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")


def _xai_oauth_state_from_store(auth_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return usable xAI OAuth state from provider state or credential pool."""
    from hermes_cli.auth import _load_provider_state
    state = _load_provider_state(auth_store, "xai-oauth")
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if isinstance(tokens, dict):
        access_token = str(tokens.get("access_token", "") or "").strip()
        refresh_token = str(tokens.get("refresh_token", "") or "").strip()
        if access_token and refresh_token:
            return state

    credential_pool = auth_store.get("credential_pool")
    entries = (
        credential_pool.get("xai-oauth")
        if isinstance(credential_pool, dict)
        else None
    )
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            access_token = str(entry.get("access_token", "") or "").strip()
            refresh_token = str(entry.get("refresh_token", "") or "").strip()
            if not access_token or not refresh_token:
                continue
            merged = dict(state or {})
            merged["tokens"] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": str(entry.get("token_type") or "Bearer"),
            }
            if entry.get("last_refresh"):
                merged["last_refresh"] = entry.get("last_refresh")
            merged.setdefault("auth_mode", "oauth_pkce")
            return merged

    return state if isinstance(state, dict) else None


def _xai_oauth_state_has_usable_tokens(state: Optional[Dict[str, Any]]) -> bool:
    tokens = state.get("tokens") if isinstance(state, dict) else None
    return (
        isinstance(tokens, dict)
        and bool(str(tokens.get("access_token", "") or "").strip())
        and bool(str(tokens.get("refresh_token", "") or "").strip())
    )


def _read_xai_oauth_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    from hermes_cli.auth import _load_global_auth_store
    auth_store = _load_auth_store_maybe_locked(_lock)
    state = _xai_oauth_state_from_store(auth_store)
    if not _xai_oauth_state_has_usable_tokens(state):
        global_state = _xai_oauth_state_from_store(_load_global_auth_store())
        if _xai_oauth_state_has_usable_tokens(global_state):
            state = global_state
    if not state:
        raise _xai_err(
            "No xAI OAuth credentials stored. Select xAI Grok OAuth (SuperGrok / Premium+) in `hermes model`.",
            "xai_auth_missing", relogin=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _xai_err(
            "xAI OAuth state is missing tokens. Re-authenticate with `hermes model`.",
            "xai_auth_invalid_shape", relogin=True,
        )
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not access_token:
        raise _xai_err(
            "xAI OAuth state is missing access_token. Re-authenticate with `hermes model`.",
            "xai_auth_missing_access_token", relogin=True,
        )
    if not refresh_token:
        raise _xai_err(
            "xAI OAuth state is missing refresh_token. Re-authenticate with `hermes model`.",
            "xai_auth_missing_refresh_token", relogin=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
        "discovery": state.get("discovery") or {},
        "redirect_uri": state.get("redirect_uri"),
    }


def _write_through_xai_oauth_to_global_root(state: Dict[str, Any]) -> None:
    """Persist a rotated xAI OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard (#43589): xAI rotates the
    refresh_token on every refresh, so when a profile session refreshes a grant it resolved from the
    root fallback, the rotated chain must land back in root.

    Only updates ``providers.xai-oauth`` in the root store; never touches the profile store (the
    caller already saved that). Swallows all errors — a failed write-through degrades to the pre-
    existing behavior (root stale), it must never break the profile's own successful save.
    """
    from hermes_cli.auth import _global_auth_file_path, _persist_provider_state_to_store
    global_path = _global_auth_file_path()
    if global_path is None:
        # Classic mode (profile == root); the profile save already hit root.
        return
    # Seat belt: under pytest, refuse to write the real user's
    # ~/.hermes/auth.json even when HERMES_HOME points at a profile path
    # (mirrors the read-side guard in _load_global_auth_store). Uses the
    # unmodified HOME env, not Path.home() which fixtures may monkeypatch.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return
            except Exception:
                return
    try:
        _persist_provider_state_to_store(
            "xai-oauth",
            state,
            global_path,
            set_active=False,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("xAI OAuth: write-through to global root failed: %s", exc)


def _save_xai_oauth_tokens(
    tokens: Dict[str, Any],
    *,
    discovery: Optional[Dict[str, Any]] = None,
    redirect_uri: str = "",
    last_refresh: Optional[str] = None,
    auth_mode: str = "oauth_device_code",
    set_active: bool = True,
) -> None:
    """Persist xAI OAuth tokens into the auth store.

    When *set_active* is True (default), also promote ``xai-oauth`` to ``active_provider`` —
    appropriate for intentional model/auth login. Pass ``set_active=False`` for side-tool credential
    bootstrap (TTS/setup, tools config, dashboard token save, token refresh) so inference routing is
    unchanged.
    """
    from hermes_cli.auth import _auth_store_lock, _global_auth_file_path, _load_auth_store, _load_provider_state_with_source, _same_path, _save_auth_store, _store_provider_state, _utc_now_z, _write_through_xai_oauth_to_global_root
    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        # A profile that lacks its own xai-oauth block is reading the root
        # grant through _load_provider_state's fallback. When such a profile
        # refreshes the (rotating) grant, we must write the rotated chain back
        # to root too, or root is left holding a revoked refresh token (#43589).
        # #74339: the old key-presence check (_profile_has_own_xai_oauth_state)
        # decided write-through based on whether the profile had a
        # providers.xai-oauth key BEFORE the save — but _store_provider_state
        # unconditionally creates that key below. Use
        # _load_provider_state_with_source to learn where the grant was
        # resolved from and write back only to that source.
        state, source_path = _load_provider_state_with_source(
            auth_store, "xai-oauth"
        )
        if state is None:
            state = {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = auth_mode
        if discovery:
            state["discovery"] = discovery
        if redirect_uri:
            state["redirect_uri"] = redirect_uri
        global_root = _global_auth_file_path()
        is_from_root = bool(
            source_path is not None
            and global_root is not None
            and _same_path(source_path, global_root)
        )
        if is_from_root:
            # Grant was resolved from root — write back to root only.
            # Do NOT call _store_provider_state on the profile auth_store
            # (it would create a shadowing providers.xai-oauth key that
            # disables write-through on the next refresh — #74339).
            _write_through_xai_oauth_to_global_root(state)
        else:
            # Profile genuinely owns this — write to profile store.
            _store_provider_state(
                auth_store, "xai-oauth", state, set_active=set_active
            )
            _save_auth_store(auth_store)


def _xai_access_token_is_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    if not isinstance(access_token, str) or "." not in access_token:
        return False
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return float(exp) <= (time.time() + max(0, int(skew_seconds)))
    except Exception:
        return False


def _xai_proactive_refresh_skew_seconds(access_token: str) -> int:
    """How far before JWT ``exp`` to proactively refresh xAI OAuth tokens.

    SuperGrok sessions ship multi-hour tokens where the gateway-oriented hour-long skew makes sense,
    but device-code logins often return ~15-minute JWTs; the full skew would force a refresh on
    every credential resolution, burning single-use refresh tokens and racing concurrent callers
    into ``invalid_grant`` quarantine.
    """
    max_skew = XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    if not isinstance(access_token, str) or "." not in access_token:
        return max_skew
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return max_skew
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return max_skew
        remaining = float(exp) - time.time()
        if remaining <= 0:
            return max_skew
        if remaining <= 45 * 60:
            return min(120, max_skew)
        return max_skew
    except Exception:
        return max_skew


def _is_xai_origin_host(host: str) -> bool:
    """``x.ai`` is the bare apex, so an exact match or any ``.x.ai`` suffix is accepted."""
    return host == "x.ai" or host.endswith(".x.ai")


def _xai_validate_oauth_endpoint(url: str, *, field: str) -> str:
    """Refuse any OIDC discovery endpoint that isn't HTTPS on the xAI origin.

    The discovery result is cached in auth.json, so a single MITM at login could plant a malicious
    ``token_endpoint`` that receives the refresh_token forever. Pinning scheme + host (RFC 8414 §2:
    HTTPS issuer, same-origin token_endpoint) removes that persistence; ``x.ai`` is the bare apex,
    so an exact match or any ``.x.ai`` suffix is accepted.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise _xai_err(
            f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}.",
            "xai_discovery_invalid",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise _xai_err(
            f"xAI OIDC discovery {field} is missing a hostname: {url!r}.",
            "xai_discovery_invalid",
        )
    if not _is_xai_origin_host(host):
        raise _xai_err(
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin "
            f"(expected x.ai or a *.x.ai subdomain). Refusing to use a cached "
            f"endpoint that may have been substituted by a MITM during initial "
            f"discovery; re-authenticate with `hermes model` to re-fetch.",
            "xai_discovery_invalid",
        )
    return url


def _xai_validate_inference_base_url(value: str, *, fallback: str) -> str:
    """Refuse a non-xAI base_url for the OAuth-authenticated inference path.

    Pin the inference origin to ``api.x.ai`` (or any ``*.x.ai`` subdomain xAI may add). On
    rejection, fall back to the default and log a warning rather than raise — a bad env var should
    not deadlock authentication, but it should also never leak the bearer.

    ``value`` is the already-stripped, trailing-slash-trimmed candidate from env. Empty input
    returns ``fallback`` unchanged.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return fallback
    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.warning(
            "Ignoring malformed xAI base_url override %r; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if parsed.scheme != "https":
        logger.warning(
            "Refusing non-HTTPS xAI base_url override %r (xai-oauth bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate, fallback,
        )
        return fallback
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning(
            "Ignoring xAI base_url override %r with no hostname; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if not _is_xai_origin_host(host):
        logger.warning(
            "Refusing xAI base_url override %r — host %r is not on the xAI origin "
            "(expected x.ai or a *.x.ai subdomain). The xai-oauth bearer is only "
            "valid against xAI's inference API; sending it elsewhere would leak "
            "the credential. Falling back to %s.",
            candidate, host, fallback,
        )
        return fallback
    return candidate


def _xai_oauth_discovery(timeout_seconds: float = 15.0) -> Dict[str, str]:
    try:
        response = httpx.get(
            XAI_OAUTH_DISCOVERY_URL,
            headers={"Accept": "application/json"},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise _xai_err(f"xAI OIDC discovery failed: {exc}", "xai_discovery_failed") from exc
    if response.status_code != 200:
        raise _xai_err(
            f"xAI OIDC discovery returned status {response.status_code}.",
            "xai_discovery_failed",
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise _xai_err(
            f"xAI OIDC discovery returned invalid JSON: {exc}",
            "xai_discovery_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise _xai_err(
            "xAI OIDC discovery response was not a JSON object.",
            "xai_discovery_incomplete",
        )
    authorization_endpoint = str(payload.get("authorization_endpoint", "") or "").strip()
    token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
    if not authorization_endpoint or not token_endpoint:
        raise _xai_err(
            "xAI OIDC discovery response was missing required endpoints.",
            "xai_discovery_incomplete",
        )
    _xai_validate_oauth_endpoint(authorization_endpoint, field="authorization_endpoint")
    _xai_validate_oauth_endpoint(token_endpoint, field="token_endpoint")
    return {
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
    }


def _xai_tokens_from_payload(payload: Dict[str, Any], access_token: str, fallback_refresh: str) -> Dict[str, Any]:
    """Token block persisted for xAI OAuth; falls back to *fallback_refresh* when none is rotated in."""
    return {
        "access_token": access_token,
        "refresh_token": str(payload.get("refresh_token") or fallback_refresh).strip(),
        "id_token": str(payload.get("id_token") or "").strip(),
        "expires_in": payload.get("expires_in"),
        "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
    }


def refresh_xai_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    token_endpoint: str = "",
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    from hermes_cli.auth import _nonempty_str, _utc_now_z, _xai_oauth_discovery
    del access_token
    if not _nonempty_str(refresh_token):
        raise _xai_err(
            "xAI OAuth is missing refresh_token. Re-authenticate with `hermes model`.",
            "xai_auth_missing_refresh_token", relogin=True,
        )
    endpoint = token_endpoint.strip() or _xai_oauth_discovery(timeout_seconds)["token_endpoint"]
    # Re-validate cached endpoints on the refresh hot path: an auth.json
    # written by an older Hermes (or hand-edited) may carry a non-xAI
    # token_endpoint that would receive every future refresh_token in
    # plaintext if we trusted it blindly. Cheap suffix check; fast-fail
    # with a clear error so the user can re-run `hermes model` to refetch.
    _xai_validate_oauth_endpoint(endpoint, field="token_endpoint")
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_OAUTH_CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    if response.status_code != 200:
        detail = response.text.strip()
        # ``403`` from xAI's token endpoint is almost always a tier /
        # entitlement gate (the OAuth grant exists but the account isn't
        # on the allowlist for API access).  Re-running ``hermes model``
        # won't fix that — surface a separate error code so
        # ``format_auth_error`` doesn't append a misleading
        # re-authenticate hint, and point users at the ``XAI_API_KEY``
        # fallback.  See #26847.
        if response.status_code == 403:
            raise _xai_err(
                "xAI token refresh failed with HTTP 403."
                + (f" Response: {detail}" if detail else "")
                + " This OAuth account is not authorized for xAI API"
                  " access — xAI may be restricting API/OAuth use to"
                  " specific SuperGrok tiers despite the in-app"
                  " subscription being active. Re-logging in won't"
                  " change that; set ``XAI_API_KEY`` and switch to"
                  " ``provider: xai`` (API-key path) if available, or"
                  " upgrade your subscription at https://x.ai/grok.",
                "xai_oauth_tier_denied", relogin=False,
            )
        raise _xai_err(
            "xAI token refresh failed."
            + (f" Response: {detail}" if detail else ""),
            "xai_refresh_failed", relogin=response.status_code in {400, 401},
        )
    payload, refreshed_access = _refresh_payload_access_token(
        response,
        provider="xai-oauth",
        invalid_json=("xAI token refresh returned invalid JSON: {exc}", "xai_refresh_invalid_json"),
        invalid_json_relogin=False,
        strict_str=False,
        invalid_response=(
            "xAI token refresh response was not a JSON object.",
            "xai_refresh_invalid_response",
        ),
        missing_access=(
            "xAI token refresh response was missing access_token.",
            "xai_refresh_missing_access_token",
        ),
    )
    return {
        **_xai_tokens_from_payload(payload, refreshed_access, refresh_token),
        "last_refresh": _utc_now_z(),
    }


def _refresh_xai_oauth_tokens(
    tokens: Dict[str, Any],
    *,
    token_endpoint: str,
    redirect_uri: str = "",
    timeout_seconds: float,
) -> Dict[str, Any]:
    # Re-persist whatever auth_mode is already stored (legacy pre-device-code
    # logins may still carry ``oauth_pkce``): the refresh hot path must not
    # relabel how the grant was originally obtained.
    from hermes_cli.auth import _load_auth_store, _load_provider_state, refresh_xai_oauth_pure
    try:
        state = _load_provider_state(_load_auth_store(), "xai-oauth") or {}
        auth_mode = str(state.get("auth_mode") or "oauth_device_code")
    except Exception:
        auth_mode = "oauth_device_code"
    refreshed = refresh_xai_oauth_pure(
        str(tokens.get("access_token", "") or ""),
        str(tokens.get("refresh_token", "") or ""),
        token_endpoint=token_endpoint,
        timeout_seconds=timeout_seconds,
    )
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]
    if refreshed.get("id_token"):
        updated_tokens["id_token"] = refreshed["id_token"]
    if refreshed.get("expires_in") is not None:
        updated_tokens["expires_in"] = refreshed["expires_in"]
    if refreshed.get("token_type"):
        updated_tokens["token_type"] = refreshed["token_type"]
    _save_xai_oauth_tokens(
        updated_tokens,
        discovery={"token_endpoint": token_endpoint},
        redirect_uri=redirect_uri,
        last_refresh=refreshed["last_refresh"],
        auth_mode=auth_mode,
        # Refresh must not flip active_provider — TTS/side tools can refresh
        # xAI tokens while chat still routes through another provider.
        set_active=False,
    )
    return updated_tokens


def _quarantine_xai_oauth_tokens(exc: AuthError) -> None:
    """Clear dead xAI tokens from auth.json after a terminal refresh failure.

    Terminal = HTTP 400/401/403 (invalid_grant, token revoked). Subsequent sessions then fail fast
    without a network retry. Mirrors credential_pool.py quarantine. Best-effort: persistence
    failures are logged and swallowed (caller re-raises the original error regardless).
    """
    from hermes_cli.auth import _last_auth_error_marker, _load_auth_store, _load_provider_state, _save_auth_store, _store_provider_state
    try:
        _q_store = _load_auth_store()
        _q_state = _load_provider_state(_q_store, "xai-oauth") or {}
        _q_tokens = dict(_q_state.get("tokens") or {})
        _q_tokens.pop("access_token", None)
        _q_tokens.pop("refresh_token", None)
        _q_state["tokens"] = _q_tokens
        _q_state["last_auth_error"] = _last_auth_error_marker(
            "xai-oauth", exc,
            reason="runtime_refresh_failure", default_code="xai_refresh_failed",
        )
        _store_provider_state(_q_store, "xai-oauth", _q_state, set_active=False)
        _save_auth_store(_q_store)
    except Exception as _save_exc:
        logger.debug(
            "xAI OAuth: failed to persist quarantined state: %s", _save_exc,
        )


def _xai_oauth_inference_base_url() -> str:
    return _xai_validate_inference_base_url(
        os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
        or os.getenv("XAI_BASE_URL", "").strip().rstrip("/"),
        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )


def resolve_xai_oauth_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    from hermes_cli.auth import _auth_store_lock, _is_terminal_xai_oauth_refresh_error, _refresh_xai_oauth_tokens, _xai_oauth_discovery
    def _view(data: Dict[str, Any]) -> tuple[Dict[str, Any], str, str, str, bool]:
        tokens = dict(data["tokens"])
        access_token = str(tokens.get("access_token", "") or "").strip()
        discovery = dict(data.get("discovery") or {})
        token_endpoint = str(discovery.get("token_endpoint", "") or "").strip()
        redirect_uri = str(data.get("redirect_uri", "") or "").strip()
        effective_skew = (
            int(refresh_skew_seconds)
            if refresh_skew_seconds is not None
            else _xai_proactive_refresh_skew_seconds(access_token)
        )
        should_refresh = bool(force_refresh)
        if (not should_refresh) and refresh_if_expiring:
            should_refresh = _xai_access_token_is_expiring(access_token, effective_skew)
        return tokens, access_token, token_endpoint, redirect_uri, should_refresh

    data = _read_xai_oauth_tokens()
    refresh_timeout_seconds = env_float("HERMES_XAI_REFRESH_TIMEOUT_SECONDS", 20)
    tokens, access_token, token_endpoint, redirect_uri, should_refresh = _view(data)
    if should_refresh:
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            data = _read_xai_oauth_tokens(_lock=False)
            tokens, access_token, token_endpoint, redirect_uri, should_refresh = _view(data)
            if should_refresh:
                if not token_endpoint:
                    token_endpoint = _xai_oauth_discovery(refresh_timeout_seconds)["token_endpoint"]
                try:
                    tokens = _refresh_xai_oauth_tokens(
                        tokens,
                        token_endpoint=token_endpoint,
                        redirect_uri=redirect_uri,
                        timeout_seconds=refresh_timeout_seconds,
                    )
                    access_token = str(tokens.get("access_token", "") or "").strip()
                except AuthError as exc:
                    if _is_terminal_xai_oauth_refresh_error(exc):
                        _quarantine_xai_oauth_tokens(exc)
                    raise

    base_url = _xai_oauth_inference_base_url()
    return {
        "provider": "xai-oauth",
        "base_url": base_url,
        "api_key": access_token,
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        # Display/telemetry only. Device-code is the only supported xAI OAuth
        # flow, so report it unconditionally — auth.json may still carry a
        # legacy ``oauth_pkce`` label, which the refresh path preserves as-is.
        "auth_mode": "oauth_device_code",
    }


def _login_xai_oauth(
    args,
    pconfig: ProviderConfig,
    *,
    force_new_login: bool = False,
) -> None:
    from hermes_cli.auth import _is_remote_session, _offer_existing_oauth_credentials, _print_login_success, _update_config_for_provider, _xai_oauth_device_code_login, resolve_xai_oauth_runtime_credentials, unsuppress_credential_source
    del pconfig

    if not force_new_login and _offer_existing_oauth_credentials(
        "xai-oauth",
        resolve=resolve_xai_oauth_runtime_credentials,
        is_expiring=_xai_access_token_is_expiring,
        display_name="xAI OAuth",
        default_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ):
        return

    print()
    print("Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    print("(Hermes creates its own local OAuth session)")
    print()

    timeout_seconds = float(getattr(args, "timeout", None) or 20.0)
    open_browser = not getattr(args, "no_browser", False)
    if _is_remote_session():
        open_browser = False

    creds = _xai_oauth_device_code_login(
        timeout_seconds=timeout_seconds,
        open_browser=open_browser,
    )
    _save_xai_oauth_tokens(
        creds["tokens"],
        discovery=creds.get("discovery"),
        redirect_uri=creds.get("redirect_uri", ""),
        last_refresh=creds.get("last_refresh"),
        auth_mode="oauth_device_code",
    )
    # An explicit interactive re-login is a strong signal the user wants the
    # xAI credential re-enabled. ``hermes auth remove xai-oauth`` leaves a
    # ``device_code`` suppression marker that otherwise stops the singleton
    # seed from re-creating the pool entry, so ``hermes auth list`` would show
    # nothing even though the agent still works via the singleton fallback.
    # Clear it here (same helper ``auth_add_command`` uses). This is kept OUT
    # of ``_save_xai_oauth_tokens`` on purpose — that helper is shared with the
    # refresh hot path, which must never mutate suppression state.
    unsuppress_credential_source("xai-oauth", "device_code")
    config_path = _update_config_for_provider("xai-oauth", creds.get("base_url", DEFAULT_XAI_OAUTH_BASE_URL))
    _print_login_success("xai-oauth", config_path, show_auth_state=True)


def _xai_oauth_request_device_code(
    client: httpx.Client,
    *,
    scope: str = XAI_OAUTH_SCOPE,
) -> Dict[str, Any]:
    response = client.post(
        XAI_OAUTH_DEVICE_CODE_URL,
        headers=_FORM_JSON_HEADERS,
        data={
            "client_id": XAI_OAUTH_CLIENT_ID,
            "scope": scope,
        },
    )
    if response.status_code != 200:
        raise _xai_err(
            f"xAI device-code request failed (HTTP {response.status_code})."
            + (f" Response: {response.text.strip()}" if response.text else ""),
            "device_code_request_failed",
        )
    payload = response.json()
    required = (
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise _xai_err(
            f"xAI device-code response missing fields: {', '.join(missing)}",
            "device_code_invalid",
        )
    return payload


def _xai_oauth_poll_device_token(
    client: httpx.Client,
    *,
    token_endpoint: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    from hermes_cli.auth import _poll_device_token_generic
    def _validate(payload: Dict[str, Any]) -> None:
        for field_name, article in (("access_token", "an"), ("refresh_token", "a")):
            if not payload.get(field_name):
                raise _xai_err(
                    f"xAI device-code token response did not include {article} {field_name}.",
                    "xai_device_token_invalid",
                )

    def _error(response, error_payload) -> Exception:
        description = (
            error_payload.get("error_description")
            or error_payload.get("error")
            or response.text
        )
        return _xai_err(
            f"xAI device-code token polling failed: {description}",
            "xai_device_token_failed",
        )

    return _poll_device_token_generic(
        lambda: client.post(
            token_endpoint,
            headers=_FORM_JSON_HEADERS,
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": XAI_OAUTH_CLIENT_ID,
                "device_code": device_code,
            },
        ),
        expires_in=int(expires_in),
        poll_interval=max(1, int(poll_interval)),
        validate_success=_validate,
        on_non_json_error=lambda _r: _xai_err(
            "xAI device-code token polling returned a non-JSON error response.",
            "xai_device_token_failed",
        ),
        on_error=_error,
        on_timeout=lambda: _xai_err(
            "Timed out waiting for xAI device authorization.",
            "device_code_timeout",
        ),
    )


def _xai_oauth_device_code_login(
    *,
    timeout_seconds: float = 20.0,
    open_browser: bool = True,
) -> Dict[str, Any]:
    from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session, _print_device_code_instructions, _utc_now_z, _xai_oauth_discovery, _xai_oauth_poll_device_token
    discovery = _xai_oauth_discovery(timeout_seconds)
    token_endpoint = discovery["token_endpoint"]
    timeout = httpx.Timeout(max(20.0, timeout_seconds))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        device_data = _xai_oauth_request_device_code(client)
        verification_url = str(
            device_data.get("verification_uri_complete")
            or device_data["verification_uri"]
        )
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        _print_device_code_instructions(
            verification_url,
            user_code,
            open_browser=open_browser and not _is_remote_session() and _can_open_graphical_browser(),
            swallow_open_errors=True,
        )
        print(f"Waiting for approval (polling every {max(1, interval)}s)...")

        payload = _xai_oauth_poll_device_token(
            client,
            token_endpoint=token_endpoint,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    access_token = str(payload.get("access_token", "") or "").strip()
    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    if not access_token or not refresh_token:
        raise _xai_err(
            "xAI device-code token response was missing required tokens.",
            "xai_device_token_invalid",
        )
    base_url = _xai_oauth_inference_base_url()
    return {
        "tokens": _xai_tokens_from_payload(payload, access_token, refresh_token),
        "discovery": discovery,
        "redirect_uri": "",
        "base_url": base_url,
        "last_refresh": _utc_now_z(),
        "source": "oauth-device-code",
    }
