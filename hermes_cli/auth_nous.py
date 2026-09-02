"""Nous Portal OAuth: device-code login, refresh, shared-store mirroring, JWT selection, status.

Split out of ``hermes_cli/auth.py``; every moved name is re-imported there, so
``hermes_cli.auth.<name>`` keeps resolving (and monkeypatching) as before. Origin-internal
helpers are imported lazily inside each function (no import cycle; patches on
``hermes_cli.auth.<helper>`` still intercept).
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional
from urllib.parse import urlparse
from hermes_cli.auth_codex import _pool_entries
from hermes_cli.auth_constants import (
    _decode_jwt_claims,
    AUTH_LOCK_TIMEOUT_SECONDS,
    AuthError,
    DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_INFERENCE_URL,
    DEFAULT_NOUS_PORTAL_URL,
    DEFAULT_NOUS_SCOPE,
    DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    NOUS_AUTH_PATH_INVOKE_JWT,
    NOUS_BILLING_MANAGE_SCOPE,
    NOUS_DEVICE_CODE_SOURCE,
    NOUS_INFERENCE_INVOKE_SCOPE,
    NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    _nous_err,
    httpx,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")


def _token_fingerprint(token: Any) -> Optional[str]:
    """Return a short hash fingerprint for telemetry without leaking token bytes."""
    if not isinstance(token, str):
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]


def _oauth_trace_enabled() -> bool:
    raw = os.getenv("HERMES_OAUTH_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _oauth_trace(event: str, *, sequence_id: Optional[str] = None, **fields: Any) -> None:
    if not _oauth_trace_enabled():
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info("oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _iso_after(now: datetime, ttl_seconds: int) -> str:
    """ISO timestamp *ttl_seconds* after *now* (UTC)."""
    return datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc).isoformat()


# Nous agent-key slots; a fresh login persists them as None, quarantine strips them.
_NOUS_EMPTY_AGENT_KEY_FIELDS: Dict[str, Any] = {
    "agent_key": None,
    "agent_key_id": None,
    "agent_key_expires_at": None,
    "agent_key_expires_in": None,
    "agent_key_reused": None,
    "agent_key_obtained_at": None,
}


_NOUS_STALE_PORTAL_HOSTS: FrozenSet[str] = frozenset({
    "api.nousresearch.com",
})


def _is_terminal_nous_refresh_error(exc: Exception) -> bool:
    return _is_terminal_refresh_error(exc, "nous")


def _is_terminal_xai_oauth_refresh_error(exc: Exception) -> bool:
    return _is_terminal_refresh_error(exc, "xai-oauth")


def _is_terminal_codex_oauth_refresh_error(exc: Exception) -> bool:
    return _is_terminal_refresh_error(exc, "openai-codex")


def _format_nous_entitlement_auth_error(error: AuthError) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability="Nous model access",
        )
        if message:
            return message
    except Exception:
        pass
    return f"{error} Check credits or billing in Nous Portal, then retry."


def _migrate_stale_nous_portal_url(providers: Dict[str, Any]) -> None:
    nous = providers.get("nous")
    if not isinstance(nous, dict):
        return
    stored = (nous.get("portal_base_url") or "").strip()
    if stored:
        parsed = urlparse(stored)
        if parsed.hostname in _NOUS_STALE_PORTAL_HOSTS:
            logger.warning(
                "auth: migrating stale nous portal_base_url %s -> %s",
                stored, DEFAULT_NOUS_PORTAL_URL,
            )
            nous["portal_base_url"] = DEFAULT_NOUS_PORTAL_URL


# Allowlist of hosts the Nous Portal proxy is willing to forward inference
# JWTs to. Sending a bearer anywhere else would leak it.
#
# This is consulted only for URLs coming from the NETWORK side (Portal
# refresh responses). User-controlled env-var overrides
# (NOUS_INFERENCE_BASE_URL) bypass validation — that's the documented
# dev/staging escape hatch and the env source is already trusted (the
# user set it themselves).
_ALLOWED_NOUS_INFERENCE_HOSTS: FrozenSet[str] = frozenset({
    "inference-api.nousresearch.com",
})


def _validate_nous_inference_url_from_network(url: Optional[str]) -> Optional[str]:
    """Validate a Portal-returned inference URL against the host allowlist.

    Defense-in-depth: a compromised refresh response from the Portal API (MITM, malicious response
    injection) could otherwise redirect every subsequent proxy request — bearing the user's
    inference JWT — to an attacker-controlled endpoint.
    """
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except Exception:
        return None
    if parsed.scheme != "https":
        logger.warning(
            "nous: refusing non-https inference URL scheme %r from Portal response",
            parsed.scheme,
        )
        return None
    if parsed.hostname not in _ALLOWED_NOUS_INFERENCE_HOSTS:
        logger.warning(
            "nous: refusing inference URL host %r from Portal response "
            "(not in allowlist); falling back to default",
            parsed.hostname,
        )
        return None
    return cleaned.rstrip("/")


def _nous_inference_env_override() -> Optional[str]:
    """Return the user-set ``NOUS_INFERENCE_BASE_URL`` override, if any.

    Documented dev/staging escape hatch. The env source is trusted (the OS user set it), so unlike
    Portal-returned URLs it is intentionally NOT gated by the network host allowlist.
    Returns a trailing-slash-stripped string, or ``None`` when unset/blank.
    """
    from hermes_cli.auth import _optional_base_url
    return _optional_base_url(os.getenv("NOUS_INFERENCE_BASE_URL"))


def _nous_portal_env_override() -> Optional[str]:
    """Return the user/deployment-set Portal base URL override, if any.

    ``HERMES_PORTAL_BASE_URL`` / ``NOUS_PORTAL_BASE_URL`` are the documented dev/staging escape
    hatch (e.g. hosted agents on the staging Portal). Like the inference override, the env source
    is trusted and must NOT be gated by ``_NOUS_PORTAL_ALLOWED_HOSTS``: that allowlist rejects an
    untrusted NETWORK-provided value persisted to auth.json, not one the operator configured.
    """
    from hermes_cli.auth import _optional_base_url
    return _optional_base_url(
        os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
    )


def _scope_values(raw_scope: Any) -> set[str]:
    # OAuth token responses normally return a space-separated string. Keep
    # collection support for JWT ``scp`` claims and older stored test fixtures.
    scopes: set[str] = set()
    if isinstance(raw_scope, str):
        for part in raw_scope.replace(",", " ").split():
            cleaned = part.strip()
            if cleaned:
                scopes.add(cleaned)
    elif isinstance(raw_scope, (list, tuple, set, frozenset)):
        for item in raw_scope:
            if isinstance(item, str):
                scopes.update(_scope_values(item))
    return scopes


def _nous_invoke_jwt_status(
    token: Any,
    *,
    scope: Any = None,
    expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
) -> Optional[str]:
    """Return None when the token can be used for inference, else a reason."""
    from hermes_cli.auth import _is_expiring
    claims = _decode_jwt_claims(token)
    if not claims:
        return "access_token_not_jwt"
    scopes = (
        _scope_values(scope)
        | _scope_values(claims.get("scope"))
        | _scope_values(claims.get("scp"))
    )
    if NOUS_INFERENCE_INVOKE_SCOPE not in scopes:
        return "missing_inference_invoke_scope"
    exp = claims.get("exp")
    skew = max(0, int(min_ttl_seconds))
    if isinstance(exp, (int, float)):
        if float(exp) <= (time.time() + skew):
            return "invoke_jwt_expiring"
        return None
    if _is_expiring(expires_at, skew):
        return "invoke_jwt_expiry_unknown_or_expiring"
    return None


def _nous_invoke_jwt_is_usable(
    token: Any,
    *,
    scope: Any = None,
    expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
) -> bool:
    from hermes_cli.auth import _nous_invoke_jwt_status
    return (
        _nous_invoke_jwt_status(
            token,
            scope=scope,
            expires_at=expires_at,
            min_ttl_seconds=min_ttl_seconds,
        )
        is None
    )


def _assert_nous_inference_jwt_usable(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
) -> None:
    from hermes_cli.auth import _nous_invoke_jwt_status
    token = state.get("access_token") if access_token is None else access_token
    reason = _nous_invoke_jwt_status(
        token,
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    )
    if reason is None:
        return
    raise _nous_err(
        "Nous Portal access token is not a usable inference JWT "
        f"({reason}). Re-authenticate with: hermes auth add nous",
        reason, relogin=True,
    )


def _log_nous_invoke_jwt_selected(
    *,
    access_token: Any,
    sequence_id: Optional[str] = None,
) -> None:
    logger.debug("Nous inference auth: using NAS invoke JWT")
    _oauth_trace(
        "nous_invoke_jwt_selected",
        sequence_id=sequence_id,
        access_token_fp=_token_fingerprint(access_token),
    )


def _nous_jwt_expires_at(token: Any, fallback_expires_at: Any = None) -> Optional[str]:
    claims = _decode_jwt_claims(token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        try:
            return datetime.fromtimestamp(float(exp), tz=timezone.utc).isoformat()
        except Exception:
            pass
    return fallback_expires_at if isinstance(fallback_expires_at, str) else None


def _set_nous_agent_key_from_invoke_jwt(
    state: Dict[str, Any],
    *,
    obtained_at: Optional[str] = None,
) -> None:
    from hermes_cli.auth import _coerce_ttl_seconds, _nonempty_str, _parse_iso_timestamp
    access_token = state.get("access_token")
    if not _nonempty_str(access_token):
        return
    now = datetime.now(timezone.utc)
    existing_obtained_at = state.get("agent_key_obtained_at")
    if obtained_at:
        effective_obtained_at = obtained_at
    elif (
        state.get("agent_key") == access_token
        and isinstance(existing_obtained_at, str)
        and existing_obtained_at.strip()
    ):
        effective_obtained_at = existing_obtained_at
    else:
        effective_obtained_at = now.isoformat()
    expires_at = _nous_jwt_expires_at(access_token, state.get("expires_at"))
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("expires_in"))
    )
    if expires_at:
        state["expires_at"] = expires_at
        state["expires_in"] = expires_in
    state["agent_key"] = access_token
    state["agent_key_id"] = None
    state["agent_key_expires_at"] = expires_at
    state["agent_key_expires_in"] = expires_in
    state["agent_key_reused"] = False
    state["agent_key_obtained_at"] = effective_obtained_at


def _select_nous_invoke_jwt(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
    sequence_id: Optional[str] = None,
) -> None:
    from hermes_cli.auth import _nonempty_str
    if _nonempty_str(access_token):
        state["access_token"] = access_token
    _set_nous_agent_key_from_invoke_jwt(state)
    _log_nous_invoke_jwt_selected(
        access_token=state.get("access_token"),
        sequence_id=sequence_id,
    )


_NOUS_EFFECTIVE_STATE_IGNORED_KEYS = frozenset({
    # These are derived from expires_at/JWT exp and naturally tick down between
    # reads. Persisting only these changes makes auth.json noisy and defeats
    # the mtime-keyed auth-status cache.
    "expires_in",
    "agent_key_expires_in",
})


def _nous_effective_provider_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key not in _NOUS_EFFECTIVE_STATE_IGNORED_KEYS
    }


NOUS_SHARED_STORE_FILENAME = "nous_auth.json"


_nous_shared_lock_holder = threading.local()


def _nous_shared_auth_dir() -> Path:
    """Resolve the directory that holds the shared Nous token store.

    Honors ``HERMES_SHARED_AUTH_DIR`` so tests can redirect it. Defaults to
    ``<hermes-root>/shared/`` (``~/.hermes/shared/`` on POSIX, ``%LOCALAPPDATA%\\hermes\\shared\\``
    on Windows), outside any named profile so all profiles under one root share the store.
    """
    override = os.getenv("HERMES_SHARED_AUTH_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "shared"


def _nous_shared_store_path() -> Path:
    path = _nous_shared_auth_dir() / NOUS_SHARED_STORE_FILENAME
    # Seat belt: if pytest is running and this resolves to a path under the
    # real user's Hermes root, refuse rather than silently corrupt cross-profile
    # state. Tests must set HERMES_SHARED_AUTH_DIR to a tmp_path (conftest
    # does not do this automatically — mirror the _auth_file_path() guard
    # so forgetting to set it fails loudly instead of writing to the real
    # shared store).
    if os.environ.get("PYTEST_CURRENT_TEST"):
        from hermes_constants import get_default_hermes_root
        real_home_shared = (
            get_default_hermes_root() / "shared" / NOUS_SHARED_STORE_FILENAME
        ).resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_shared:
            raise RuntimeError(
                f"Refusing to touch real user shared Nous auth store during test run: "
                f"{path}. Set HERMES_SHARED_AUTH_DIR to a tmp_path in your test fixture."
            )
    return path


@contextmanager
def _nous_shared_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-profile lock for the shared Nous OAuth store.

    Lock ordering invariant: if both this and ``_auth_store_lock`` need to be held, acquire
    ``_auth_store_lock`` FIRST. All runtime refresh paths follow this order.
    """
    from hermes_cli.auth import _file_lock
    try:
        lock_path = _nous_shared_store_path().with_suffix(".lock")
    except RuntimeError:
        # No HERMES_HOME yet (pre-setup): fall through without locking.
        yield
        return

    with _file_lock(
        lock_path,
        _nous_shared_lock_holder,
        timeout_seconds,
        "Timed out waiting for shared Nous auth lock",
    ):
        yield


# OAuth fields mirrored between a profile's Nous state and the shared cross-profile store.
_NOUS_SHARED_STATE_KEYS = (
    "access_token",
    "refresh_token",
    "token_type",
    "scope",
    "client_id",
    "portal_base_url",
    "inference_base_url",
    "obtained_at",
    "expires_at",
)


def _merge_shared_nous_oauth_state(state: Dict[str, Any]) -> bool:
    """Copy fresher shared OAuth tokens into a profile-local Nous state."""
    from hermes_cli.auth import _nonempty_str, _parse_iso_timestamp, _read_shared_nous_state
    shared = _read_shared_nous_state()
    if not shared:
        return False

    shared_refresh = shared.get("refresh_token")
    if not _nonempty_str(shared_refresh):
        return False

    local_refresh = state.get("refresh_token")
    shared_access_exp = _parse_iso_timestamp(shared.get("expires_at")) or 0.0
    local_access_exp = _parse_iso_timestamp(state.get("expires_at")) or 0.0
    refresh_changed = shared_refresh.strip() != str(local_refresh or "").strip()
    fresher_access = shared_access_exp > local_access_exp
    if not refresh_changed and not fresher_access:
        return False

    for key in _NOUS_SHARED_STATE_KEYS:
        value = shared.get(key)
        if value not in {None, ""}:
            state[key] = value
    return True


def _nous_shared_shape(src: Dict[str, Any]) -> Dict[str, Any]:
    """The defaulted OAuth core (tokens + routing + expiry) shared across profiles."""
    return {
        "access_token": src.get("access_token"),
        "refresh_token": src.get("refresh_token"),
        "token_type": src.get("token_type") or "Bearer",
        "scope": src.get("scope") or DEFAULT_NOUS_SCOPE,
        "client_id": src.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": src.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
        "inference_base_url": src.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL,
        "obtained_at": src.get("obtained_at"),
        "expires_at": src.get("expires_at"),
    }


def _write_shared_nous_state(state: Dict[str, Any]) -> None:
    """Persist a minimal copy of the Nous OAuth state to the shared store.

    Best-effort: any failure is swallowed after logging. The shared store is a convenience layer;
    the per-profile auth.json remains the source of truth.
    """
    from hermes_cli.auth import _nonempty_str, _write_private_file_atomic
    refresh_token = state.get("refresh_token")
    access_token = state.get("access_token")
    # No refresh_token = nothing worth sharing across profiles
    if not (_nonempty_str(refresh_token) and _nonempty_str(access_token)):
        return

    shared = {
        "_schema": 1,
        **_nous_shared_shape(state),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            _write_private_file_atomic(
                path, json.dumps(shared, indent=2, sort_keys=True), replace=os.replace,
            )
        _oauth_trace(
            "nous_shared_store_written",
            path=str(path),
            refresh_token_fp=_token_fingerprint(refresh_token),
        )
    except Exception as exc:
        logger.debug("Failed to write shared Nous auth store: %s", exc)


def _read_shared_nous_state() -> Optional[Dict[str, Any]]:
    """Return the shared Nous OAuth state if present and well-formed.

    Returns ``None`` when the file is missing, unreadable, malformed, or lacks required fields;
    callers treat that as "no shared credentials, fall through to device-code".
    """
    from hermes_cli.auth import _nonempty_str
    try:
        path = _nous_shared_store_path()
    except RuntimeError:
        # Test seat belt tripped — treat as missing
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.debug("Shared Nous auth store at %s is unreadable: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if not (_nonempty_str(payload.get("refresh_token")) and _nonempty_str(payload.get("access_token"))):
        return None
    return payload


def _clear_shared_nous_state(reason: str) -> None:
    """Remove the shared Nous OAuth store after a terminal token failure."""
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _oauth_trace("nous_shared_store_cleared", reason=reason)
    except Exception as exc:
        logger.debug("Failed to clear shared Nous auth store: %s", exc)


# Error codes per provider for which retrying the SAME refresh token cannot succeed.
# ``*_refresh_failed`` covers HTTP 400/401/403 from the token endpoint (invalid_grant, token
# revoked, refresh_token_reused); ``*_auth_missing_refresh_token`` means the pool entry has no
# refresh token at all. All must also carry ``relogin_required=True``; transient failures
# (429, 5xx) do not.
_OAUTH_GRANT_DEAD_CODES = frozenset({"invalid_grant", "invalid_token", "refresh_token_reused"})


_TERMINAL_REFRESH_ERROR_CODES: Dict[str, FrozenSet[str]] = {
    "nous": _OAUTH_GRANT_DEAD_CODES,
    "xai-oauth": frozenset({"xai_refresh_failed", "xai_auth_missing_refresh_token"}),
    "openai-codex": _OAUTH_GRANT_DEAD_CODES | {"codex_refresh_failed", "codex_auth_missing_refresh_token"},
}


def _is_terminal_refresh_error(exc: Exception, provider: str) -> bool:
    """True when retrying the same *provider* refresh token cannot succeed."""
    return (
        isinstance(exc, AuthError)
        and exc.provider == provider
        and exc.code in _TERMINAL_REFRESH_ERROR_CODES[provider]
        and bool(exc.relogin_required)
    )


def _quarantine_nous_oauth_state(
    state: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> None:
    """Keep routing metadata but remove dead OAuth material so it is not replayed."""
    from hermes_cli.auth import _FLAT_OAUTH_TOKEN_KEYS, _auth_file_path, _last_auth_error_marker, invalidate_nous_auth_status_cache
    # Forensic logging BEFORE we clear the token material. A hosted agent
    # can take a terminal invalid_grant and get quarantined here silently: the
    # only downstream signal is a "No access token found" WARNING once the pool
    # is already empty, which is too late to root-cause. A managed log drain may
    # be WARNING-only, so this MUST be logger.warning (INFO never reaches it).
    #
    # Redaction safety: emit ONLY the 12-char SHA-256 hex prefix of the refresh
    # token (correlates to NAS's refreshTokenHash without leaking the secret) plus
    # sizes/booleans. NEVER pass a raw token/agent_key into the log call — Hermes
    # has a known bug class where credential-shaped literals get corrupted in logs.
    forensic: Dict[str, Any] = {
        "reason": reason,
        "error_code": error.code,
        # No session_id field exists on Nous state; provenance is client_id +
        # agent_key_id (both non-secret routing identifiers).
        "client_id": state.get("client_id"),
        "agent_key_id": state.get("agent_key_id"),
        "refresh_token_fp": _token_fingerprint(state.get("refresh_token")),
    }

    # On-disk integrity of the auth store at the moment of quarantine.
    try:
        auth_path = _auth_file_path()
        forensic["auth_json_path"] = str(auth_path)
        try:
            st = os.stat(auth_path)
            forensic["auth_json_size"] = st.st_size
            forensic["auth_json_mtime"] = st.st_mtime
            forensic["auth_json_exists"] = True
        except FileNotFoundError:
            forensic["auth_json_exists"] = False
    except Exception as exc:  # pragma: no cover - never let logging break quarantine
        forensic["auth_json_stat_error"] = repr(exc)

    # Was the token already past its own expiry when it was rejected?
    already_expired: Optional[bool] = None
    expires_at_raw = state.get("expires_at")
    if isinstance(expires_at_raw, str) and expires_at_raw:
        try:
            parsed = datetime.fromisoformat(expires_at_raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            already_expired = parsed < datetime.now(timezone.utc)
        except ValueError:
            already_expired = None
    forensic["token_already_expired"] = already_expired

    logger.warning(
        "Nous OAuth state quarantined (terminal auth death): %s",
        json.dumps(forensic, sort_keys=True, ensure_ascii=False),
    )

    for key in (*_FLAT_OAUTH_TOKEN_KEYS, *_NOUS_EMPTY_AGENT_KEY_FIELDS):
        state.pop(key, None)
    state["last_auth_error"] = _last_auth_error_marker("nous", error, reason=reason)
    _clear_shared_nous_state(reason)
    invalidate_nous_auth_status_cache()


def _quarantine_nous_pool_entries(
    auth_store: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> bool:
    """Remove singleton-seeded Nous pool entries that contain dead OAuth state."""
    entries = _pool_entries(auth_store, "nous")
    if entries is None:
        return False
    pool = auth_store["credential_pool"]

    retained = []
    removed = False
    singleton_sources = {NOUS_DEVICE_CODE_SOURCE, f"manual:{NOUS_DEVICE_CODE_SOURCE}"}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source") in singleton_sources:
            removed = True
            continue
        retained.append(entry)

    if removed:
        pool["nous"] = retained
        _oauth_trace(
            "nous_pool_device_code_quarantined",
            reason=reason,
            error_code=error.code,
        )
    return removed


def _try_import_shared_nous_state(
    *,
    timeout_seconds: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Attempt to rehydrate Nous OAuth state from the shared store.

    Runs a forced refresh with the stored refresh_token to mint a fresh inference JWT scoped to
    this profile and returns the auth_state dict ready for ``persist_nous_credentials()``.
    Returns ``None`` on any failure (expired token, portal unreachable) so the caller falls
    through to the normal device-code flow.
    """
    from hermes_cli.auth import _read_shared_nous_state, _write_shared_nous_state, refresh_nous_oauth_from_state
    try:
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
            shared = _read_shared_nous_state()
            if not shared:
                return None

            # Build a full state dict so refresh_nous_oauth_from_state has every
            # field it needs. force_refresh=True gets us a fresh access_token
            # for this profile.
            state: Dict[str, Any] = {
                **_nous_shared_shape(shared),
                "agent_key": None,
                "agent_key_expires_at": None,
                "tls": {"insecure": False, "ca_bundle": None},
            }

            def _persist_shared_refresh(updated_state: Dict[str, Any], _reason: str) -> None:
                _write_shared_nous_state(updated_state)

            refreshed = refresh_nous_oauth_from_state(
                state,
                timeout_seconds=timeout_seconds,
                force_refresh=True,
                on_state_update=_persist_shared_refresh,
            )
            _write_shared_nous_state(refreshed)
    except AuthError as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
            error_code=getattr(exc, "code", None),
        )
        if _is_terminal_nous_refresh_error(exc):
            _clear_shared_nous_state("shared_import_terminal_refresh_failure")
        logger.debug("Shared Nous import failed: %s", exc)
        return None
    except Exception as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
        )
        logger.debug("Shared Nous import failed: %s", exc)
        return None

    return refreshed


def _refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        headers={"x-nous-refresh-token": refresh_token},
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
        },
    )

    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise _nous_err("Refresh response missing access_token", "invalid_token", relogin=True)
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise _nous_err("Refresh token exchange failed", relogin=True) from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token", "refresh_token_reused"}

    # Detect the OAuth 2.1 "refresh token reuse" signal from the Nous portal
    # server and surface an actionable message.  This fires when an external
    # process (health-check script, monitoring tool, custom self-heal hook)
    # called POST /api/oauth/token with Hermes's refresh_token without
    # persisting the rotated token back to auth.json — the server then
    # retires the original RT, Hermes's next refresh uses it, and the whole
    # session chain gets revoked as a token-theft signal (#15099).
    lowered = description.lower()
    if code == "refresh_token_reused" or "reuse" in lowered or "reuse detected" in lowered:
        description = (
            "Nous Portal detected refresh-token reuse and revoked this session.\n"
            "This usually means an external process (monitoring script, "
            "custom self-heal hook, or another Hermes install sharing "
            "~/.hermes/auth.json) called POST /api/oauth/token with Hermes's "
            "refresh token without persisting the rotated token back.\n"
            "Nous refresh tokens are single-use — only Hermes may call the "
            "refresh endpoint. For health checks, use `hermes auth status` "
            "instead.\n"
            "Re-authenticate with: hermes auth add nous"
        )
        relogin = True

    raise _nous_err(description, code, relogin=relogin)


def _refresh_nous_or_quarantine(
    *,
    client: httpx.Client,
    auth_store: Dict[str, Any],
    state: Dict[str, Any],
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
    reason: str,
    persist: Callable[[], None],
) -> Dict[str, Any]:
    """Redeem the Nous refresh token; on a terminal failure quarantine state + pool, persist, re-raise."""
    from hermes_cli.auth import _refresh_access_token
    try:
        return _refresh_access_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            refresh_token=refresh_token,
        )
    except AuthError as exc:
        if _is_terminal_nous_refresh_error(exc):
            _quarantine_nous_oauth_state(state, exc, reason=reason)
            _quarantine_nous_pool_entries(auth_store, exc, reason=reason)
            persist()
        raise


def _apply_nous_refreshed_tokens(
    state: Dict[str, Any],
    refreshed: Dict[str, Any],
    refresh_token: str,
    *,
    inference_base_url: Optional[str] = None,
) -> None:
    """Write a successful Nous token-refresh payload into *state* (tokens + expiry fields).

    *inference_base_url*, when given, is the healed network-provenance URL to persist alongside
    the rotated tokens (key order in auth.json is preserved from the original login shape).
    """
    from hermes_cli.auth import _coerce_ttl_seconds
    now = datetime.now(timezone.utc)
    access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
    state["access_token"] = refreshed["access_token"]
    state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
    state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
    state["scope"] = refreshed.get("scope") or state.get("scope")
    if inference_base_url is not None:
        state["inference_base_url"] = inference_base_url
    state["obtained_at"] = now.isoformat()
    state["expires_in"] = access_ttl
    state["expires_at"] = _iso_after(now, access_ttl)


def _healed_nous_inference_url(refreshed: Dict[str, Any]) -> str:
    """Validated network-provenance inference URL from a refresh payload, healed to the default.

    When the Portal-returned URL is rejected by the allowlist (returns None), reset to the
    production default instead of leaving a previously-persisted bad host (e.g. a stale staging
    URL) in place — otherwise a poisoned auth.json keeps re-validating to None on every refresh
    and silently re-uses the dead endpoint.
    """
    return (
        _validate_nous_inference_url_from_network(refreshed.get("inference_base_url"))
        or DEFAULT_NOUS_INFERENCE_URL
    )


def fetch_nous_models(
    *,
    inference_base_url: str,
    api_key: str,
    timeout_seconds: float = 15.0,
    verify: bool | str = True,
) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    from hermes_cli.auth import _nonempty_str
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise _nous_err(description, "models_fetch_failed")

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if _nonempty_str(model_id):
            mid = model_id.strip()
            # Skip Hermes models — they're not reliable for agentic tool-calling
            if "hermes" in mid.lower():
                continue
            model_ids.append(mid)

    # Sort: prefer opus > pro > haiku/flash > sonnet (sonnet is cheap/fast,
    # users who want the best model should see opus first).
    def _model_priority(mid: str) -> tuple:
        low = mid.lower()
        if "opus" in low:
            return (0, mid)
        if "pro" in low and "sonnet" not in low:
            return (1, mid)
        if "sonnet" in low:
            return (3, mid)
        return (2, mid)

    model_ids.sort(key=_model_priority)
    return list(dict.fromkeys(model_ids))


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    from hermes_cli.auth import _nonempty_str
    key = state.get("agent_key")
    if not _nonempty_str(key):
        return False
    return _nous_invoke_jwt_is_usable(
        key,
        scope=state.get("scope"),
        expires_at=state.get("agent_key_expires_at"),
        min_ttl_seconds=max(0, int(min_ttl_seconds)),
    )


def refresh_nous_oauth_pure(
    access_token: str,
    refresh_token: str,
    client_id: str,
    portal_base_url: str,
    inference_base_url: str,
    *,
    token_type: str = "Bearer",
    scope: str = DEFAULT_NOUS_SCOPE,
    obtained_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    agent_key: Optional[str] = None,
    agent_key_expires_at: Optional[str] = None,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json directly.

    ``on_state_update`` is called after a successful access-token refresh. Callers that own
    persistent state can use it to save the newly rotated refresh token before later validation can
    fail.
    """
    from hermes_cli.auth import _assert_nous_inference_jwt_usable, _nous_invoke_jwt_status, _refresh_access_token, _resolve_verify, _select_nous_invoke_jwt
    state: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/"),
        "inference_base_url": (inference_base_url or DEFAULT_NOUS_INFERENCE_URL).rstrip("/"),
        "token_type": token_type or "Bearer",
        "scope": scope or DEFAULT_NOUS_SCOPE,
        "obtained_at": obtained_at,
        "expires_at": expires_at,
        "agent_key": agent_key,
        "agent_key_expires_at": agent_key_expires_at,
        "tls": {
            "insecure": bool(insecure),
            "ca_bundle": ca_bundle,
        },
    }
    verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        current_invoke_jwt_status = _nous_invoke_jwt_status(
            state.get("access_token"),
            scope=state.get("scope"),
            expires_at=state.get("expires_at"),
        )
        if force_refresh or current_invoke_jwt_status is not None:
            refresh_token_value = state.get("refresh_token")
            if not isinstance(refresh_token_value, str) or not refresh_token_value:
                if current_invoke_jwt_status is not None:
                    raise _nous_err(
                        "Nous Portal access token is not a usable inference JWT "
                        f"({current_invoke_jwt_status}) and no refresh token is available. "
                        "Re-authenticate with: hermes auth add nous",
                        current_invoke_jwt_status, relogin=True,
                    )
                raise _nous_err(
                    "No refresh token is available for Nous Portal.",
                    relogin=True,
                )
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=refresh_token_value,
            )
            _apply_nous_refreshed_tokens(
                state, refreshed, refresh_token_value,
                inference_base_url=_healed_nous_inference_url(refreshed),
            )
            if on_state_update is not None:
                on_state_update(dict(state), "post_refresh_access_token")

        _assert_nous_inference_jwt_usable(state)
        _select_nous_invoke_jwt(state)

    return state


def refresh_nous_oauth_from_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth from a state dict. Thin wrapper around refresh_nous_oauth_pure."""
    tls = state.get("tls") or {}
    return refresh_nous_oauth_pure(
        state.get("access_token", ""),
        state.get("refresh_token", ""),
        state.get("client_id", "hermes-cli"),
        state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL),
        state.get("inference_base_url", DEFAULT_NOUS_INFERENCE_URL),
        token_type=state.get("token_type", "Bearer"),
        scope=state.get("scope", DEFAULT_NOUS_SCOPE),
        obtained_at=state.get("obtained_at"),
        expires_at=state.get("expires_at"),
        agent_key=state.get("agent_key"),
        agent_key_expires_at=state.get("agent_key_expires_at"),
        timeout_seconds=timeout_seconds,
        insecure=tls.get("insecure"),
        ca_bundle=tls.get("ca_bundle"),
        force_refresh=force_refresh,
        on_state_update=on_state_update,
    )


def persist_nous_credentials(
    creds: Dict[str, Any],
    *,
    label: Optional[str] = None,
):
    """Persist Nous OAuth credentials as the singleton provider state

    Nous credentials are read from two places: ``providers.nous`` (401 recovery, pool seeding) and
    ``credential_pool.nous`` (runtime ``pool.select()``). Writing only a pool entry left the
    singleton empty and made expiry recovery fail silently, so this writes the singleton and then
    ``load_pool("nous")`` upserts the canonical ``device_code`` entry in place (never duplicates).
    ``label`` is embedded in the singleton so re-seeding keeps the user's display name.
    """
    from hermes_cli.auth import _save_active_provider_state, _write_shared_nous_state
    from agent.credential_pool import load_pool

    state = dict(creds)
    if label and str(label).strip():
        state["label"] = str(label).strip()

    _save_active_provider_state("nous", state)

    # Mirror to the shared store so a new profile can one-tap import
    # these credentials via `hermes auth add nous --type oauth`. Best-
    # effort: any I/O failure is logged and swallowed (the per-profile
    # auth.json is still the source of truth).
    _write_shared_nous_state(state)

    pool = load_pool("nous")
    return next(
        (e for e in pool.entries() if e.source == NOUS_DEVICE_CODE_SOURCE),
        None,
    )


def _sync_nous_pool_from_auth_store() -> None:
    """Best-effort pool reseed after providers.nous changes; never fail login."""
    try:
        from agent.credential_pool import load_pool

        load_pool("nous")
    except Exception as exc:
        logger.debug("Failed to sync Nous credential pool from auth store: %s", exc)


class _NousStatePersister:
    """Writes Nous provider state to its source store, skipping no-op writes.

    Writes where only derived TTL countdowns changed are skipped; this keeps the mtime-keyed Nous
    auth-status cache warm during read paths. Every real write is mirrored to the shared store so
    sibling profiles don't hold stale refresh_tokens after rotation (best-effort — failures are
    logged and swallowed inside ``_write_shared_nous_state``).
    """

    def __init__(
        self,
        auth_store: Dict[str, Any],
        state: Dict[str, Any],
        state_source_path: Optional[Path],
        sequence_id: str,
    ) -> None:
        self._auth_store = auth_store
        self._state = state
        self._source_path = state_source_path
        self._sequence_id = sequence_id
        self._persisted_state = dict(state)
        self.persisted_any = False

    def persist(self, reason: str) -> None:
        from hermes_cli.auth import _save_provider_state_to_source, _write_shared_nous_state
        state = self._state
        if (
            _nous_effective_provider_state(state)
            == _nous_effective_provider_state(self._persisted_state)
        ):
            _oauth_trace(
                "nous_state_persist_skipped",
                sequence_id=self._sequence_id,
                reason=reason,
            )
            return
        try:
            _save_provider_state_to_source(self._auth_store, "nous", state, self._source_path)
        except Exception as exc:
            _oauth_trace(
                "nous_state_persist_failed",
                sequence_id=self._sequence_id,
                reason=reason,
                error_type=type(exc).__name__,
            )
            raise
        _oauth_trace(
            "nous_state_persisted",
            sequence_id=self._sequence_id,
            reason=reason,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
            access_token_fp=_token_fingerprint(state.get("access_token")),
        )
        self._persisted_state = dict(state)
        self.persisted_any = True
        _write_shared_nous_state(state)


def _nous_effective_routing(state: Dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve every routing value that shared OAuth state can replace.

    Returns ``(portal_url, stored_inference_url, effective_inference_url, client_id)``. The
    stored inference URL is re-validated network-provenance (persisted); the effective one layers
    the runtime-only ``NOUS_INFERENCE_BASE_URL`` override on top and must never be persisted.
    """
    from hermes_cli.auth import _NOUS_PORTAL_ALLOWED_HOSTS, _optional_base_url
    portal_url = (
        _optional_base_url(state.get("portal_base_url"))
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or DEFAULT_NOUS_PORTAL_URL
    ).rstrip("/")

    # A persisted/stale portal_base_url is where the refresh token gets
    # POSTed on refresh — reject any host outside the allowlist so a
    # poisoned value can't exfiltrate the bearer, healing to the default.
    # Trusted operator env overrides bypass this network-value gate.
    env_portal_override = _nous_portal_env_override()
    if env_portal_override:
        portal_url = env_portal_override.rstrip("/")
    else:
        parsed_portal_url = urlparse(portal_url)
        portal_host = parsed_portal_url.hostname
        loopback_http = (
            parsed_portal_url.scheme == "http"
            and portal_host in {"localhost", "127.0.0.1"}
        )
        trusted_scheme = parsed_portal_url.scheme == "https" or loopback_http
        if (
            not portal_host
            or portal_host not in _NOUS_PORTAL_ALLOWED_HOSTS
            or not trusted_scheme
        ):
            logger.warning(
                "auth: ignoring invalid portal_base_url %r "
                "(host %r or scheme not allowed), using default",
                portal_url,
                portal_host,
            )
            portal_url = DEFAULT_NOUS_PORTAL_URL

    stored_inference_url = (
        _validate_nous_inference_url_from_network(
            _optional_base_url(state.get("inference_base_url"))
        )
        or DEFAULT_NOUS_INFERENCE_URL
    )
    return (
        portal_url,
        stored_inference_url,
        _nous_inference_env_override() or stored_inference_url,
        str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID),
    )


def resolve_nous_runtime_credentials(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    stale_access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve Nous inference credentials for runtime use.

    Ensures access_token is a valid inference-scoped JWT, refreshing it when
    needed. Concurrent processes coordinate through the auth store file lock.

    ``stale_access_token`` is the bearer that just failed upstream (401). When
    set together with ``force_refresh``, the refresh POST is skipped if the
    store — re-read under the lock — already holds a *different*, usable
    token: another process won the rotation, so this caller adopts it instead
    of rotating the shared grant again (otherwise N concurrent processes at the
    same expiry issue N refreshes, each invalidating a sibling's fresh token).
    """
    from hermes_cli.auth import _assert_nous_inference_jwt_usable, _auth_file_path, _coerce_ttl_seconds, _nous_invoke_jwt_status, _parse_iso_timestamp, _provider_state_transaction, _resolve_verify, _select_nous_invoke_jwt, _sync_nous_pool_from_auth_store, _tls_state_from_verify
    sequence_id = uuid.uuid4().hex[:12]

    with _provider_state_transaction("nous") as (
        auth_store,
        state,
        state_source_path,
    ):

        if not state:
            raise _nous_err("Hermes is not logged into Nous Portal.", relogin=True)

        def _already_rotated_by_peer(token: Any) -> bool:
            return bool(
                force_refresh
                and stale_access_token
                and isinstance(token, str)
                and token
                and token != stale_access_token
                and _nous_invoke_jwt_status(
                    token,
                    scope=state.get("scope"),
                    expires_at=state.get("expires_at"),
                ) is None
            )

        persister = _NousStatePersister(auth_store, state, state_source_path, sequence_id)
        _persist_state = persister.persist

        (
            portal_base_url,
            stored_inference_base_url,
            inference_base_url,
            client_id,
        ) = _nous_effective_routing(state)

        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
        _oauth_trace(
            "nous_runtime_credentials_start",
            sequence_id=sequence_id,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
        )

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                with _nous_shared_store_lock(
                    timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)
                ):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        (
                            portal_base_url,
                            stored_inference_base_url,
                            inference_base_url,
                            client_id,
                        ) = _nous_effective_routing(state)
                        _persist_state("runtime_shared_merge_missing_access_token")

            if not isinstance(access_token, str) or not access_token:
                raise _nous_err(
                    "No access token found for Nous Portal login.",
                    relogin=True,
                )

            invoke_jwt_status = _nous_invoke_jwt_status(
                access_token,
                scope=state.get("scope"),
                expires_at=state.get("expires_at"),
            )
            # Under the store lock: if the bearer that failed upstream is no
            # longer the one on disk and the on-disk one is usable, a peer
            # already rotated — adopt, never re-POST the shared grant.
            if _already_rotated_by_peer(access_token):
                _oauth_trace(
                    "refresh_skipped_peer_rotated",
                    sequence_id=sequence_id,
                    access_token_fp=_token_fingerprint(access_token),
                )
                force_refresh = False
            if force_refresh or invoke_jwt_status is not None:
                with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        (
                            portal_base_url,
                            stored_inference_base_url,
                            inference_base_url,
                            client_id,
                        ) = _nous_effective_routing(state)
                        invoke_jwt_status = _nous_invoke_jwt_status(
                            access_token,
                            scope=state.get("scope"),
                            expires_at=state.get("expires_at"),
                        )
                        _persist_state("post_shared_merge_access_unusable")
                        if _already_rotated_by_peer(access_token):
                            _oauth_trace(
                                "refresh_skipped_peer_rotated",
                                sequence_id=sequence_id,
                                access_token_fp=_token_fingerprint(access_token),
                            )
                            force_refresh = False

                    if force_refresh or invoke_jwt_status is not None:
                        if not isinstance(refresh_token, str) or not refresh_token:
                            reason = invoke_jwt_status or "force_refresh"
                            raise _nous_err(
                                "Nous Portal access token is not a usable inference JWT "
                                f"({reason}) and no refresh token is available. "
                                "Re-authenticate with: hermes auth add nous",
                                reason, relogin=True,
                            )

                        refresh_reason = "force_refresh" if force_refresh else (invoke_jwt_status or "access_unusable")
                        _oauth_trace(
                            "refresh_start",
                            sequence_id=sequence_id,
                            reason=refresh_reason,
                            refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        refreshed = _refresh_nous_or_quarantine(
                            client=client,
                            auth_store=auth_store,
                            state=state,
                            portal_base_url=portal_base_url,
                            client_id=client_id,
                            refresh_token=refresh_token,
                            reason="runtime_access_refresh_failure",
                            persist=lambda: _persist_state("terminal_runtime_access_refresh_failure"),
                        )
                        previous_refresh_token = refresh_token
                        # The validated, network-provenance URL is what gets persisted to
                        # auth.json (with the rotated tokens, so a later JWT validation
                        # failure cannot leave the stores on stale metadata). The
                        # NOUS_INFERENCE_BASE_URL env override is layered on for the
                        # client/return value only — it is never persisted.
                        stored_inference_base_url = _healed_nous_inference_url(refreshed)
                        inference_base_url = (
                            _nous_inference_env_override() or stored_inference_base_url
                        )
                        _apply_nous_refreshed_tokens(
                            state, refreshed, refresh_token,
                            inference_base_url=stored_inference_base_url,
                        )
                        access_token = state["access_token"]
                        refresh_token = state["refresh_token"]
                        _oauth_trace(
                            "refresh_success",
                            sequence_id=sequence_id,
                            reason=refresh_reason,
                            previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
                            new_refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        # Persist immediately so validation failures cannot drop rotated refresh tokens.
                        _persist_state("post_refresh_access_token")

            _assert_nous_inference_jwt_usable(
                state,
                access_token=access_token,
            )
            _select_nous_invoke_jwt(
                state,
                access_token=access_token,
                sequence_id=sequence_id,
            )

            # Persist routing and TLS metadata for non-interactive refresh.
            # Persist the validated, network-provenance URL — NEVER the env
            # override (which is a runtime-only overlay; persisting it would
            # leak a dev/staging host into auth.json and survive unsetting it).
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = stored_inference_base_url
            state["client_id"] = client_id
            state["tls"] = _tls_state_from_verify(verify)

        _persist_state("resolve_nous_runtime_credentials_final")

    if persister.persisted_any:
        _sync_nous_pool_from_auth_store()

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise _nous_err("Failed to resolve a Nous inference API key", "server_error")

    expires_at = state.get("agent_key_expires_at")
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("agent_key_expires_in"))
    )

    return {
        "provider": "nous",
        "base_url": inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": NOUS_AUTH_PATH_INVOKE_JWT,
        # Preserve the public semantic source label while exposing the concrete
        # store separately for diagnostics. Refresh persistence uses
        # state_source_path internally and must not overload this field.
        "auth_path": NOUS_AUTH_PATH_INVOKE_JWT,
        "state_path": str(state_source_path or _auth_file_path()),
    }


def _empty_nous_auth_status() -> Dict[str, Any]:
    return {
        "logged_in": False,
        "portal_base_url": None,
        "inference_base_url": None,
        "access_expires_at": None,
        "agent_key_expires_at": None,
        "has_refresh_token": False,
        "inference_credential_present": False,
        "credential_source": None,
    }


def _snapshot_nous_pool_status() -> Dict[str, Any]:
    """Best-effort status from the credential pool.

    This is a fallback only. The auth-store provider state is the runtime source of truth because it
    is what ``resolve_nous_runtime_credentials()`` refreshes.
    """
    from hermes_cli.auth import _parse_iso_timestamp
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("nous")
        if not pool or not pool.has_credentials():
            return _empty_nous_auth_status()

        entries = list(pool.entries())
        if not entries:
            return _empty_nous_auth_status()

        def _entry_sort_key(entry: Any) -> tuple[float, float, int]:
            agent_exp = _parse_iso_timestamp(getattr(entry, "agent_key_expires_at", None)) or 0.0
            access_exp = _parse_iso_timestamp(getattr(entry, "expires_at", None)) or 0.0
            priority = int(getattr(entry, "priority", 0) or 0)
            return (agent_exp, access_exp, -priority)

        entry = max(entries, key=_entry_sort_key)
        runtime_key = getattr(entry, "runtime_api_key", None)
        if not runtime_key:
            return _empty_nous_auth_status()
        access_token = getattr(entry, "access_token", None)
        auth_type = str(getattr(entry, "auth_type", "") or "").strip().lower()
        refresh_token = getattr(entry, "refresh_token", None)
        is_portal_oauth = bool(access_token) and (
            auth_type.startswith("oauth") or bool(refresh_token)
        )
        label = getattr(entry, "label", "unknown")
        portal_status_url = None
        if is_portal_oauth:
            portal_status_url = (
                getattr(entry, "portal_base_url", None)
                or DEFAULT_NOUS_PORTAL_URL
            )

        return {
            "logged_in": is_portal_oauth,
            "portal_base_url": portal_status_url,
            "inference_base_url": getattr(entry, "inference_base_url", None)
            or getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None),
            "access_token": access_token if is_portal_oauth else None,
            "access_expires_at": getattr(entry, "expires_at", None),
            "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
            "has_refresh_token": bool(refresh_token),
            "inference_credential_present": True,
            "credential_source": f"pool:{label}",
            "source": f"pool:{label}",
        }
    except Exception:
        return _empty_nous_auth_status()


def _nous_status_from_state(state: Dict[str, Any], *, logged_in: bool, source: str) -> Dict[str, Any]:
    """Auth-store-backed Nous status snapshot (shared by the live and refresh-free variants)."""
    access_token = state.get("access_token")
    return {
        "logged_in": logged_in,
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")),
        "access_token": access_token,
        "inference_credential_present": bool(access_token or state.get("agent_key")),
        "credential_source": "auth_store",
        "source": source,
    }


def _compute_nous_auth_status() -> Dict[str, Any]:
    """Uncached implementation of get_nous_auth_status(). See that function."""
    from hermes_cli.auth import get_provider_auth_state, resolve_nous_runtime_credentials
    state = get_provider_auth_state("nous")
    if state:
        base_status = _nous_status_from_state(
            state, logged_in=bool(state.get("access_token")), source="auth_store",
        )
        try:
            creds = resolve_nous_runtime_credentials()
            refreshed_state = get_provider_auth_state("nous") or state
            base_status.update(
                {
                    "logged_in": True,
                    "portal_base_url": refreshed_state.get("portal_base_url") or base_status.get("portal_base_url"),
                    "inference_base_url": creds.get("base_url")
                    or refreshed_state.get("inference_base_url")
                    or base_status.get("inference_base_url"),
                    "access_expires_at": refreshed_state.get("expires_at") or base_status.get("access_expires_at"),
                    "agent_key_expires_at": creds.get("expires_at")
                    or refreshed_state.get("agent_key_expires_at")
                    or base_status.get("agent_key_expires_at"),
                    "has_refresh_token": bool(refreshed_state.get("refresh_token")),
                    "inference_credential_present": True,
                    "credential_source": "auth_store",
                    "source": f"runtime:{creds.get('source', 'portal')}",
                    "key_id": creds.get("key_id"),
                }
            )
            return base_status
        except AuthError as exc:
            base_status.update({
                "logged_in": False,
                "error": str(exc),
                "relogin_required": bool(getattr(exc, "relogin_required", False)),
                "error_code": getattr(exc, "code", None),
            })
            return base_status

    return _snapshot_nous_pool_status()


def get_nous_auth_status_local() -> Dict[str, Any]:
    """Refresh-free Nous auth snapshot for read-only display surfaces.

    Unlike :func:`get_nous_auth_status`, this NEVER calls ``resolve_nous_runtime_credentials()`` and
    therefore never performs an OAuth refresh POST or consumes a single-use refresh token. It
    reports the persisted auth-store state, classifying the access token with a local invoke-JWT
    decode only.

    ``logged_in`` here means "a persisted login exists that the runtime can use or refresh": a
    currently-usable invoke JWT, or a refresh token that has not been terminally quarantined. It
    does not prove the refresh token is still accepted server-side — only a live resolve can do
    that.
    """
    from hermes_cli.auth import _nous_invoke_jwt_status, get_provider_auth_state
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        state = None

    if not state:
        return _snapshot_nous_pool_status()

    access_token = state.get("access_token")
    jwt_reason = _nous_invoke_jwt_status(
        access_token,
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    )
    last_err = state.get("last_auth_error")
    terminal = bool(
        isinstance(last_err, dict)
        and last_err.get("relogin_required")
        and not (access_token or state.get("refresh_token"))
    )
    logged_in = (jwt_reason is None) or (
        bool(state.get("refresh_token")) and not terminal
    )

    status = _nous_status_from_state(state, logged_in=logged_in, source="auth_store_local")
    if terminal and isinstance(last_err, dict):
        status["relogin_required"] = True
        status["error_code"] = last_err.get("code")
        status["error"] = last_err.get("message") or "re-login required"
    return status


# Enum values reported on the dashboard /api/status as ``nous_session_valid``.
# NAS's health sweep re-mints the bootstrap session ONLY on "terminal"; "valid"
# and "unknown" are no-ops. Keep this set small and stable — NAS parses it with
# a permissive schema, so new members are non-breaking but should stay rare.
NOUS_SESSION_VALID = "valid"


NOUS_SESSION_TERMINAL = "terminal"


NOUS_SESSION_UNKNOWN = "unknown"


def get_nous_session_validity() -> str:
    """Classify the Nous bootstrap session for the dashboard /api/status probe.

    Determinable with NO working token — it reads local auth-store state only, which is exactly the
    condition a dead hosted box is in. This function is called by the frequently-polled public
    ``/api/status`` endpoint, so it must never resolve credentials or perform an OAuth refresh.

    ANTI-FLAP CONTRACT: only a *terminal* failure maps to "terminal". A normal mid-rotation blip, a
    transient network error, or a merely-expiring token must NOT report "terminal" (that would
    trigger a spurious NAS re-mint on a healthy box).
    """
    from hermes_cli.auth import _nous_invoke_jwt_status, get_provider_auth_state
    # A persisted quarantine marker is the strongest, most stable terminal
    # signal: the refresh path writes `last_auth_error.relogin_required=True`
    # into the Nous provider state when it clears dead tokens (the exact path
    # that produced the incident's "No access token found"). Read it directly
    # so we report "terminal" even after the in-memory AuthError is long gone.
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        return NOUS_SESSION_UNKNOWN

    if not state:
        return NOUS_SESSION_UNKNOWN

    last_err = state.get("last_auth_error")
    # Only terminal while there is no usable credential left. If a later
    # successful login repopulated tokens, the stale marker must not
    # keep reporting terminal.
    if (
        isinstance(last_err, dict)
        and last_err.get("relogin_required")
        and not (state.get("access_token") or state.get("refresh_token"))
    ):
        return NOUS_SESSION_TERMINAL

    if _nous_invoke_jwt_status(
        state.get("access_token"),
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    ) is None:
        return NOUS_SESSION_VALID

    # Missing, malformed, expired, or merely expiring credentials are not proof
    # of a terminal session. Runtime inference/keepalive paths own refreshes;
    # the health endpoint remains side-effect free and reports indeterminate.
    return NOUS_SESSION_UNKNOWN


def _pool_first_oauth_status(
    provider_id: str,
    *,
    is_expiring: Callable[[str, int], bool],
    auth_mode: str,
    resolve: Callable[[], Dict[str, Any]],
    on_pool_miss: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Status snapshot for a store-backed OAuth provider (Codex, xAI).

    Checks the credential pool first (where `hermes auth` / `hermes model` store device_code
    tokens), optionally consults *on_pool_miss* for a pool-derived degraded status, then falls
    back to the legacy provider state via *resolve*.
    """
    from hermes_cli.auth import _auth_file_path
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if api_key and not is_expiring(api_key, 0):
                    return {
                        "logged_in": True,
                        "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        "auth_mode": auth_mode,
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "api_key": api_key,
                    }
            if on_pool_miss is not None:
                degraded = on_pool_miss()
                if degraded:
                    return degraded
    except Exception:
        pass

    try:
        creds = resolve()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_store": str(_auth_file_path()),
            "error": str(exc),
        }


def _nous_device_code_login(
    *,
    portal_base_url: Optional[str] = None,
    inference_base_url: Optional[str] = None,
    client_id: Optional[str] = None,
    scope: Optional[str] = None,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    insecure: bool = False,
    ca_bundle: Optional[str] = None,
    on_verification: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Run the Nous device-code flow and return full OAuth state without persisting."""
    from hermes_cli.auth import PROVIDER_REGISTRY, _coerce_ttl_seconds, _is_remote_session, _optional_base_url, _poll_for_token, _print_device_code_instructions, _request_device_code, _tls_state_from_verify, format_auth_error, refresh_nous_oauth_from_state
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        portal_base_url
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url
    ).rstrip("/")
    requested_inference_url = (
        inference_base_url
        or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url
    ).rstrip("/")
    client_id = client_id or pconfig.client_id
    scope = scope or pconfig.scope
    timeout = httpx.Timeout(timeout_seconds)
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        device_data = _request_device_code(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            scope=scope,
        )

        verification_url = str(device_data["verification_uri_complete"])
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        _print_device_code_instructions(
            verification_url, user_code, open_browser=open_browser, failure_dash="—",
        )

        # Surface the verification URL/code to an out-of-band consumer (e.g. the
        # TUI gateway, whose stdout is a JSON-RPC pipe — a plain print() there is
        # dropped). Fired AFTER the print/browser block and BEFORE polling blocks,
        # so the consumer can render the link while we wait. Best-effort.
        if on_verification is not None:
            try:
                on_verification(verification_url, user_code)
            except Exception:
                pass

        effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
        print(f"Waiting for approval (polling every {effective_interval}s)...")

        token_data = _poll_for_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    now = datetime.now(timezone.utc)
    token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    expires_at = now.timestamp() + token_expires_in
    resolved_inference_url = (
        _optional_base_url(token_data.get("inference_base_url"))
        or requested_inference_url
    )
    if resolved_inference_url != requested_inference_url:
        print(f"Using portal-provided inference URL: {resolved_inference_url}")

    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": resolved_inference_url,
        "client_id": client_id,
        "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "expires_in": token_expires_in,
        "tls": _tls_state_from_verify(verify),
        **_NOUS_EMPTY_AGENT_KEY_FIELDS,
    }
    try:
        return refresh_nous_oauth_from_state(
            auth_state,
            timeout_seconds=timeout_seconds,
            force_refresh=False,
        )
    except AuthError as exc:
        if exc.code == "subscription_required":
            portal_url = auth_state.get(
                "portal_base_url", DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")
            message = format_auth_error(exc)
            print()
            print(message)
            print(f"  Subscribe here: {portal_url}/billing")
            print()
            print("After subscribing, run `hermes model` again to finish setup.")
            raise SystemExit(1)
        raise


def _mirror_nous_state_best_effort(auth_state: Dict[str, Any]) -> None:
    """Mirror to the shared store + reseed the pool, swallowing all errors (same as _login_nous)."""
    from hermes_cli.auth import _sync_nous_pool_from_auth_store, _write_shared_nous_state
    try:
        _write_shared_nous_state(auth_state)
    except Exception:
        pass
    try:
        _sync_nous_pool_from_auth_store()
    except Exception:
        pass


def step_up_nous_billing_scope(
    *,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    on_verification: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """Re-run the device flow requesting ``billing:manage`` and persist the result.

    Lazy step-up triggered by ``403 insufficient_scope``. The user must be ADMIN/OWNER and select
    "Allow Remote Spending" in the portal, otherwise the server silently downscopes and this returns
    False. Reuses the held credential's portal/inference URLs + client_id so the step-up targets the
    same deployment, and persists like ``_login_nous`` but WITHOUT the model picker.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY, _nous_device_code_login, _save_active_provider_state, get_provider_auth_state
    prior = get_provider_auth_state("nous") or {}
    pconfig = PROVIDER_REGISTRY["nous"]

    # Build the step-up scope: existing scopes (if any) + billing:manage, deduped,
    # order-stable. Fall back to the standard inference+tool+billing set.
    _raw_scope = prior.get("scope")
    prior_scope = _raw_scope if isinstance(_raw_scope, str) else ""
    requested: list[str] = []
    for tok in (prior_scope.split() or [NOUS_INFERENCE_INVOKE_SCOPE, "tool:invoke"]):
        if tok and tok not in requested:
            requested.append(tok)
    if NOUS_BILLING_MANAGE_SCOPE not in requested:
        requested.append(NOUS_BILLING_MANAGE_SCOPE)
    scope = " ".join(requested)

    auth_state = _nous_device_code_login(
        portal_base_url=prior.get("portal_base_url") or None,
        inference_base_url=prior.get("inference_base_url") or None,
        client_id=prior.get("client_id") or pconfig.client_id,
        scope=scope,
        open_browser=open_browser,
        timeout_seconds=timeout_seconds,
        on_verification=on_verification,
    )

    _save_active_provider_state("nous", auth_state)
    _mirror_nous_state_best_effort(auth_state)

    granted = auth_state.get("scope")
    return isinstance(granted, str) and NOUS_BILLING_MANAGE_SCOPE in granted.split()


def _pick_nous_model_after_login(auth_state: Dict[str, Any], inference_base_url: str) -> Optional[str]:
    """Fetch the curated Nous model list (tier/policy-filtered) and run the interactive picker.

    Returns the selected model id, or None when the user skipped / nothing was selectable.
    Raises on any fetch failure so the caller can print the "Login succeeded, but..." notice.
    """
    from hermes_cli.auth import _prompt_model_selection
    runtime_key = auth_state.get("agent_key") or auth_state.get("access_token")
    if not isinstance(runtime_key, str) or not runtime_key:
        raise _nous_err("No runtime API key available to fetch models", "invalid_token")

    from hermes_cli.models import (
        get_curated_nous_model_ids, get_pricing_for_provider,
        check_nous_free_tier, partition_nous_models_by_tier,
        nous_policy_allowed_ids, restrict_to_nous_policy,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )
    model_ids = get_curated_nous_model_ids()

    print()
    unavailable_models: list = []
    unavailable_message = ""
    if model_ids:
        pricing = get_pricing_for_provider("nous")
        # Force fresh account data for model selection so recent credit
        # purchases are reflected immediately.
        free_tier = check_nous_free_tier(force_fresh=True)
        _portal_for_recs = auth_state.get("portal_base_url", "")
        # Narrow before the tier split, so a rescued id still has to
        # pass the free/paid predicate.
        _policy_allowed = nous_policy_allowed_ids()
        _policy_narrowed = False
        if free_tier:
            try:
                from hermes_cli.nous_account import (
                    format_nous_portal_entitlement_message,
                    get_nous_portal_account_info,
                )

                _account_info = get_nous_portal_account_info(force_fresh=True)
                unavailable_message = (
                    format_nous_portal_entitlement_message(
                        _account_info,
                        capability="paid Nous models",
                    )
                    or ""
                )
            except Exception:
                unavailable_message = ""
        # The Portal's free/paidRecommendedModels endpoint is the source of
        # truth for what's available *right now*. Augment the curated list with
        # anything new the Portal flags so users on older Hermes builds still
        # see newly-launched models without a CLI release.
        union = (
            union_with_portal_free_recommendations
            if free_tier
            else union_with_portal_paid_recommendations
        )
        model_ids, pricing = union(model_ids, pricing, _portal_for_recs)
        _before_policy = model_ids
        model_ids = restrict_to_nous_policy(
            model_ids, _policy_allowed, rescue_empty=True,
        )
        _policy_narrowed = model_ids != _before_policy
        if free_tier:
            model_ids, unavailable_models = partition_nous_models_by_tier(
                model_ids, pricing, free_tier=True,
            )
    _portal = auth_state.get("portal_base_url", "")
    if model_ids:
        from hermes_cli.nous_account import nous_policy_notice

        _policy_notice = nous_policy_notice(removed=_policy_narrowed)
        if _policy_notice:
            print(_policy_notice)
        print(f"Showing {len(model_ids)} curated models — use \"Enter custom model name\" for others.")
        return _prompt_model_selection(
            model_ids, pricing=pricing,
            unavailable_models=unavailable_models,
            portal_url=_portal,
            unavailable_message=unavailable_message,
            confirm_provider="nous",
            confirm_base_url=inference_base_url,
            confirm_api_key=runtime_key,
        )
    elif unavailable_models:
        _url = (_portal or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        print("No free models currently available.")
        print(unavailable_message or f"Upgrade at {_url} to access paid models.")
    else:
        print("No curated models available for Nous Portal.")
    return None


def _offer_shared_nous_import(timeout_seconds: float) -> Optional[Dict[str, Any]]:
    """Codex-style auto-import: offer to rehydrate a Nous credential from another profile.

    Checks the shared store before launching a fresh device-code flow. Returns the refreshed
    auth state when the user accepted and the import succeeded, else None.
    """
    from hermes_cli.auth import _prompt_yes_no, _read_shared_nous_state
    shared = _read_shared_nous_state()
    if not shared:
        return None
    try:
        shared_path = _nous_shared_store_path()
    except RuntimeError:
        shared_path = None
    print()
    if shared_path:
        print(f"Found existing Nous OAuth credentials at {shared_path}")
    else:
        print("Found existing shared Nous OAuth credentials")
    if not _prompt_yes_no("Import these credentials? [Y/n]: ", default="y"):
        return None
    print("Rehydrating Nous session from shared credentials...")
    auth_state = _try_import_shared_nous_state(timeout_seconds=timeout_seconds)
    if auth_state is None:
        print("Could not refresh shared credentials — falling back to device-code login.")
    return auth_state


def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _nous_device_code_login, _save_active_provider_state, _save_auth_store, _save_model_choice, _sync_nous_pool_from_auth_store, _update_config_for_provider, _write_shared_nous_state, format_auth_error
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    insecure = bool(getattr(args, "insecure", False))
    ca_bundle = (
        getattr(args, "ca_bundle", None)
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )

    try:
        auth_state = _offer_shared_nous_import(timeout_seconds)
        if auth_state is None:
            auth_state = _nous_device_code_login(
                portal_base_url=getattr(args, "portal_url", None),
                inference_base_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None) or pconfig.client_id,
                scope=getattr(args, "scope", None),
                open_browser=not getattr(args, "no_browser", False),
                timeout_seconds=timeout_seconds,
                insecure=insecure,
                ca_bundle=ca_bundle,
            )

        inference_base_url = auth_state["inference_base_url"]

        # Snapshot the prior active_provider BEFORE _save_provider_state
        # overwrites it to "nous".  If the user picks "Skip (keep current)"
        # during model selection below, we restore this so the user's previous
        # provider (e.g. openrouter) is preserved.
        with _auth_store_lock():
            _prior_store = _load_auth_store()
            prior_active_provider = _prior_store.get("active_provider")

        saved_to = _save_active_provider_state("nous", auth_state)

        # Mirror to the shared store so other profiles can one-tap import
        # these credentials. Best-effort: any I/O failure is logged and
        # swallowed inside the helper.
        _write_shared_nous_state(auth_state)
        _sync_nous_pool_from_auth_store()

        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")

        # Resolve model BEFORE writing provider to config.yaml so we never
        # leave the config in a half-updated state (provider=nous but model
        # still set to the previous provider's model, e.g. opus from
        # OpenRouter).  The auth.json active_provider was already set above.
        selected_model = None
        try:
            selected_model = _pick_nous_model_after_login(auth_state, inference_base_url)
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")

        # Write provider + model atomically so config is never mismatched.
        # If no model was selected (user picked "Skip (keep current)",
        # model list fetch failed, or no curated models were available),
        # preserve the user's previous provider — don't silently switch
        # them to Nous with a mismatched model.  The Nous OAuth tokens
        # stay saved for future use.
        if not selected_model:
            # Restore the prior active_provider that _save_provider_state
            # overwrote to "nous".  config.yaml model.provider is left
            # untouched, so the user's previous provider is fully preserved.
            with _auth_store_lock():
                auth_store = _load_auth_store()
                if prior_active_provider:
                    auth_store["active_provider"] = prior_active_provider
                else:
                    auth_store.pop("active_provider", None)
                _save_auth_store(auth_store)
            print()
            print("No provider change. Nous credentials saved for future use.")
            print("  Run `hermes model` again to switch to Nous Portal.")
            return

        config_path = _update_config_for_provider(
            "nous", inference_base_url, default_model=selected_model,
        )
        if selected_model:
            _save_model_choice(selected_model)
            print(f"Default model set to: {selected_model}")
        print(f"  Config updated: {config_path} (model.provider=nous)")

    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)
