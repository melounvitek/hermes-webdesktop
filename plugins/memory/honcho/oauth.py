"""OAuth credential storage and refresh for the Honcho memory provider.

An access token authenticates like a scoped API key, so it is stored as the
host's ``apiKey``; the refresh token is exchanged before expiry to keep it live.

Refresh tokens rotate with single-use reuse detection: replaying a stale token
revokes the grant, so every refresh persists the rotated token atomically and is
serialized (in-process lock + cross-process file lock). A failed exchange never
raises into the agent: transient failures retry once immediately (the replay
grace window is short); a permanent error such as invalid_grant marks the grant
dead so nothing keeps hitting the endpoint and callers surface a re-login
prompt. A server-side 401 on a locally-valid token is recovered via
``force_refresh_token``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACCESS_TOKEN_PREFIX = "hch-at-"
REFRESH_TOKEN_PREFIX = "hch-rt-"

_REFRESH_SKEW_SECONDS = 120  # refresh this early so an in-flight request never races expiry
_REFRESH_TIMEOUT_SECONDS = 15.0  # short: sits on the path to a memory call
_REFRESH_RETRY_DELAY_SECONDS = 2.0  # replayed refresh tokens are honored only briefly after rotation
# Cap for one exchange cycle (attempt + pause + retry): it runs under the global
# refresh locks, so it must not hold them for two full HTTP timeouts.
_REFRESH_TOTAL_BUDGET_SECONDS = 20.0
# After a transient failure, fail open without re-exchanging for this long so N
# waiting threads don't serialize N exchange cycles against a failing endpoint.
_REFRESH_FAILURE_COOLDOWN_SECONDS = 30.0
# OAuth error codes a retry can never fix — the grant itself is dead.
_PERMANENT_OAUTH_ERRORS = frozenset({"invalid_grant", "invalid_client", "unauthorized_client"})

# Derived from the canonical prefixes so a prefix change can't silently break redaction.
_TOKEN_VALUE_RE = re.compile(
    rf"({re.escape(ACCESS_TOKEN_PREFIX)}|{re.escape(REFRESH_TOKEN_PREFIX)})[A-Za-z0-9._~+/=-]+"
)


def redact_tokens(text: str) -> str:
    """Replace any embedded token values with their prefix plus a placeholder."""
    return _TOKEN_VALUE_RE.sub(lambda m: f"{m.group(1)}[redacted]", text)


_redact_tokens = redact_tokens  # backward-compat alias for older importers


class OAuthRefreshError(Exception):
    """Token endpoint rejected the refresh. ``permanent`` means re-login is required."""

    def __init__(self, message: str, *, error: str = "", permanent: bool = False):
        super().__init__(message)
        self.error = error
        self.permanent = permanent


# Serializes refresh across threads; state is re-checked under it so racing
# callers don't replay a rotated refresh token.
_refresh_lock = threading.Lock()


def _os_lock(fh, lock: bool) -> None:
    if os.name == "nt":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if lock else fcntl.LOCK_UN)


@contextmanager
def _config_refresh_lock(path: Path):
    """Machine-wide advisory lock (``<config>.lock``) around read-refresh-persist:
    a sibling process sharing this honcho.json must not replay the single-use
    refresh token. Best-effort — without flock it degrades to in-process only."""
    fh = None
    try:
        lock_path = Path(f"{path}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+b")
        _os_lock(fh, True)
    except Exception:
        logger.debug("Honcho OAuth cross-process lock unavailable; in-process only", exc_info=True)
        if fh is not None:
            fh.close()
            fh = None
    try:
        yield
    finally:
        if fh is not None:
            with suppress(Exception):
                _os_lock(fh, False)
            fh.close()


# Per-grant state keyed by (config path, host); single-key dict ops are atomic
# under the GIL, so no separate lock.
# (expires_at, access): lets the hot path skip the honcho.json read while the
# token is well clear of expiry. A stale entry can't break auth (the token stays
# valid until its own expiry); it only defers noticing out-of-band rotation.
_expiry_cache: dict[tuple[str, str], tuple[float, str]] = {}
# sha256 of the permanently rejected refresh token; a re-login rotates the
# token, so the digest comparison self-clears.
_dead_grants: dict[tuple[str, str], str] = {}
# monotonic time of the last transient exchange failure (drives the fail-open cooldown).
_refresh_failure_at: dict[tuple[str, str], float] = {}
# (config mtime_ns, verdict): reauth_required only changes when the file is
# rewritten, so an unchanged mtime skips the parse.
_reauth_check_cache: dict[tuple[str, str], tuple[int, bool]] = {}


def _in_failure_cooldown(key: tuple[str, str]) -> bool:
    failed_at = _refresh_failure_at.get(key)
    return failed_at is not None and (time.monotonic() - failed_at) < _REFRESH_FAILURE_COOLDOWN_SECONDS


def _refresh_token_digest(cred: OAuthCredential) -> str:
    return hashlib.sha256(cred.refresh_token.encode("utf-8")).hexdigest()


def _grant_is_dead(key: tuple[str, str], cred: OAuthCredential) -> bool:
    return _dead_grants.get(key) == _refresh_token_digest(cred)


def _mark_grant_dead(key: tuple[str, str], cred: OAuthCredential) -> None:
    _dead_grants[key] = _refresh_token_digest(cred)
    _reauth_check_cache.pop(key, None)  # verdict changed without a config rewrite


def _load_cred(path: Path, host: str, raw: dict[str, Any] | None = None) -> OAuthCredential | None:
    """Credential from ``host``'s block in ``raw`` (or the file at ``path``)."""
    source = raw if raw is not None else _read_config(path)
    return OAuthCredential.from_host_block((source.get("hosts") or {}).get(host) or {})


def reauth_required(path: Path, host: str) -> bool:
    """True when ``host``'s OAuth grant is dead and only a new login fixes it."""
    key = (str(path), host)
    if key not in _dead_grants:
        return False
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    cached = _reauth_check_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    cred = _load_cred(path, host)
    result = cred is not None and _grant_is_dead(key, cred)
    _reauth_check_cache[key] = (mtime, result)
    return result


def any_dead_grants() -> bool:
    """Cheap predicate letting hot-path callers skip config resolution when healthy."""
    return bool(_dead_grants)


def _num(value, cast):
    """``cast(value)``, or ``cast(0)`` when the stored value is malformed."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(0)


def is_oauth_access_token(value: str | None) -> bool:
    """True when ``value`` is an OAuth access token (vs a static API key)."""
    return bool(value) and value.startswith(ACCESS_TOKEN_PREFIX)


@dataclass
class OAuthCredential:
    """An OAuth grant as stored in a honcho.json host block: ``access_token`` is
    the host's ``apiKey``, the rest lives in its ``oauth`` sub-block.
    ``expires_at`` is absolute epoch seconds."""

    access_token: str
    refresh_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    scope: str = "write"
    token_type: str = "Bearer"
    # Transient consent peer name — set only on a fresh grant, never persisted.
    consent_peer_name: str | None = None

    @classmethod
    def from_host_block(cls, block: dict[str, Any]) -> "OAuthCredential | None":
        """Build a credential from a honcho.json host block, or None if incomplete."""
        oauth = block.get("oauth")
        access = block.get("apiKey")
        if not isinstance(oauth, dict) or not is_oauth_access_token(access):
            return None
        refresh, endpoint, client_id = oauth.get("refreshToken"), oauth.get("tokenEndpoint"), oauth.get("clientId")
        if not (refresh and endpoint and client_id):
            return None
        return cls(
            access_token=access,
            refresh_token=str(refresh),
            expires_at=_num(oauth.get("expiresAt", 0), float),
            client_id=str(client_id),
            token_endpoint=str(endpoint),
            scope=str(oauth.get("scope", "write")),
            token_type=str(oauth.get("tokenType", "Bearer")),
        )

    @classmethod
    def from_token_response(
        cls,
        body: dict[str, Any],
        *,
        now: float,
        client_id: str,
        token_endpoint: str,
        scope: str = "write",
        token_type: str = "Bearer",
        what: str = "grant",
    ) -> "OAuthCredential":
        """Build a credential from an OAuth token response; ``expires_in`` is relative to ``now``."""
        access, refresh = body.get("access_token"), body.get("refresh_token")
        if not is_oauth_access_token(access) or not refresh:
            raise ValueError(f"{what} missing access_token/refresh_token")
        return cls(
            access_token=access,
            refresh_token=str(refresh),
            expires_at=now + _num(body.get("expires_in", 0), int),
            client_id=client_id,
            token_endpoint=token_endpoint,
            scope=str(body.get("scope", scope)),
            token_type=str(body.get("token_type", token_type)),
        )

    def oauth_block(self) -> dict[str, Any]:
        """The ``oauth`` sub-block to persist (the access token lives in apiKey)."""
        return {
            "refreshToken": self.refresh_token,
            "expiresAt": int(self.expires_at),
            "clientId": self.client_id,
            "tokenEndpoint": self.token_endpoint,
            "scope": self.scope,
            "tokenType": self.token_type,
        }

    def is_expired(self, *, now: float, skew: float = _REFRESH_SKEW_SECONDS) -> bool:
        """True when the access token is within ``skew`` seconds of expiry."""
        return now >= (self.expires_at - skew)


# HTTP indirection: tests monkeypatch these module attributes, and the refresh
# path looks them up at call time.
def _http_json(method: str, url: str, *, data=None, timeout: float, strict: bool = True) -> tuple[int, Any]:
    """Return ``(status, parsed JSON body)``. ``strict`` raises on non-2xx /
    non-JSON; otherwise a 4xx passes through (RFC 8628 polling reads the OAuth
    error off a 400) and a non-JSON / non-object body parses to ``{}``."""
    import httpx

    resp = httpx.request(method, url, data=data, timeout=timeout)
    if strict:
        resp.raise_for_status()
        return resp.status_code, resp.json()
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code, body if isinstance(body, dict) else {}


def _http_post_form_status(url: str, data: dict[str, str], timeout: float) -> tuple[int, dict[str, Any]]:
    """POST form-encoded ``data``; return ``(status, body)`` without raising on 4xx."""
    return _http_json("POST", url, data=data, timeout=timeout, strict=False)


def _exchange_refresh_token(
    cred: OAuthCredential, *, now: float, timeout: float = _REFRESH_TIMEOUT_SECONDS
) -> OAuthCredential:
    """Run the refresh_token grant and return the rotated credential. Raises
    ``OAuthRefreshError`` (with the endpoint's error body) on an error response,
    transport errors as-is; callers fail open."""
    status, body = _http_post_form_status(
        cred.token_endpoint,
        {"grant_type": "refresh_token", "client_id": cred.client_id, "refresh_token": cred.refresh_token},
        timeout,
    )
    if status >= 400:
        error = str(body.get("error") or "")
        description = str(body.get("error_description") or "")
        detail = " — ".join(p for p in (error, description) if p) or "no error body"
        raise OAuthRefreshError(
            _redact_tokens(f"token endpoint returned HTTP {status}: {detail}"),
            error=error,
            permanent=error in _PERMANENT_OAUTH_ERRORS,
        )
    return OAuthCredential.from_token_response(
        body, now=now, client_id=cred.client_id, token_endpoint=cred.token_endpoint,
        scope=cred.scope, token_type=cred.token_type, what="refresh response",
    )


def _exchange_with_retry(cred: OAuthCredential, *, now: float) -> OAuthCredential:
    """Exchange the refresh token, retrying once on transient failure. The retry
    cannot wait (replay grace window is short) and the cycle is capped by
    ``_REFRESH_TOTAL_BUDGET_SECONDS`` because it runs under the refresh locks."""
    deadline = time.monotonic() + _REFRESH_TOTAL_BUDGET_SECONDS
    try:
        return _exchange_refresh_token(cred, now=now)
    except OAuthRefreshError as exc:
        if exc.permanent:
            raise
        first: Exception = exc
    except Exception as exc:
        first = exc
    remaining = deadline - time.monotonic() - _REFRESH_RETRY_DELAY_SECONDS
    if remaining <= 0:
        raise first
    logger.warning("Honcho OAuth token exchange failed, retrying once: %s", _redact_tokens(str(first)))
    time.sleep(_REFRESH_RETRY_DELAY_SECONDS)
    return _exchange_refresh_token(cred, now=now, timeout=min(remaining, _REFRESH_TIMEOUT_SECONDS))


def _rotate_and_persist(
    path: Path,
    host: str,
    key: tuple[str, str],
    cred: OAuthCredential,
    *,
    now: float,
    op_label: str = "refresh",
) -> OAuthCredential | None:
    """Exchange ``cred`` and persist the rotation; ``None`` on failure (logged).
    A permanent OAuth error marks the grant dead so later calls skip the endpoint
    until a new login rotates the refresh token."""
    try:
        rotated = _exchange_with_retry(cred, now=now)
    except OAuthRefreshError as exc:
        if exc.permanent:
            _mark_grant_dead(key, cred)
            logger.error(
                "Honcho OAuth grant for host %s is no longer valid (%s); "
                "run 'hermes honcho setup' to re-authenticate", host, exc,
            )
            return None
        _refresh_failure_at[key] = time.monotonic()
        logger.warning("Honcho OAuth %s failed for host %s: %s", op_label, host, exc)
        return None
    except Exception as exc:
        _refresh_failure_at[key] = time.monotonic()
        logger.warning("Honcho OAuth %s failed for host %s: %s", op_label, host, _redact_tokens(str(exc)))
        return None
    _persist_credential(path, host, rotated)
    return rotated


def _read_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_config(path: Path, raw: dict[str, Any]) -> None:
    """Write ``raw`` to ``path`` atomically with 0600 on the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(raw, indent=2) + "\n")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins on scalars/lists)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _persist_credential(path: Path, host: str, cred: OAuthCredential, raw: dict[str, Any] | None = None) -> None:
    """Write ``cred`` into ``host``'s block (apiKey + oauth) of ``raw`` (default:
    the file's current content), leaving the rest intact; marks the grant live."""
    raw = _read_config(path) if raw is None else raw
    block = raw.setdefault("hosts", {}).setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    _atomic_write_config(path, raw)
    key = (str(path), host)
    _expiry_cache[key] = (cred.expires_at, cred.access_token)
    _dead_grants.pop(key, None)
    _refresh_failure_at.pop(key, None)


def ensure_fresh_token(
    path: Path,
    host: str,
    raw: dict[str, Any] | None = None,
    *,
    now: float | None = None,
) -> tuple[str | None, bool]:
    """Return ``(access_token, refreshed)`` for ``host``, refreshing if near expiry.
    ``(None, False)`` when the host has no OAuth credential (plain API key).
    Refresh failures are swallowed: the current (possibly stale) token comes back
    with ``refreshed=False``; a permanently rejected grant is marked dead and the
    session's 401 recovery escalates it to the user."""
    now = time.time() if now is None else now
    key = (str(path), host)

    # Hot path: trust the cached expiry while well clear of the skew window (no
    # disk read). Bypassed when an explicit ``raw`` is supplied.
    if raw is None:
        cached = _expiry_cache.get(key)
        if cached is not None and now < cached[0] - _REFRESH_SKEW_SECONDS:
            return cached[1], False

    cred = _load_cred(path, host, raw)
    if cred is None:
        _expiry_cache.pop(key, None)
        return None, False

    _expiry_cache[key] = (cred.expires_at, cred.access_token)
    if not cred.is_expired(now=now) or _in_failure_cooldown(key):
        return cred.access_token, False

    with _refresh_lock, _config_refresh_lock(path):
        # Re-read under both locks: another thread or process may have just
        # rotated the token — adopt theirs instead of replaying the old one.
        current = _load_cred(path, host) or cred
        if not current.is_expired(now=now):
            return current.access_token, current.access_token != cred.access_token
        # The lock holder we waited on may have just failed; fail open too.
        if _grant_is_dead(key, current) or _in_failure_cooldown(key):
            return current.access_token, False
        rotated = _rotate_and_persist(path, host, key, current, now=now)
        if rotated is None:
            return current.access_token, False
        logger.info("Honcho OAuth token refreshed for host %s", host)
        return rotated.access_token, True


def force_refresh_token(path: Path, host: str) -> str | None:
    """Rotate ``host``'s token now, ignoring local expiry (recovers a 401 on a
    token the local clock still thinks is valid)."""
    now = time.time()
    key = (str(path), host)
    with _refresh_lock, _config_refresh_lock(path):
        cred = _load_cred(path, host)
        if cred is None:
            _expiry_cache.pop(key, None)
            return None
        # Dead grant, or an exchange just failed transiently: callers fail open.
        if _grant_is_dead(key, cred) or _in_failure_cooldown(key):
            return None
        cached = _expiry_cache.get(key)
        # Another thread or process already rotated: adopt the newer on-disk token.
        if cached is not None and cred.access_token != cached[1] and not cred.is_expired(now=now):
            _expiry_cache[key] = (cred.expires_at, cred.access_token)
            return cred.access_token
        rotated = _rotate_and_persist(path, host, key, cred, now=now, op_label="forced refresh")
        if rotated is None:
            return None
        logger.info("Honcho OAuth token force-refreshed for host %s after an auth failure", host)
        return rotated.access_token


def install_grant(
    path: Path,
    host: str,
    grant: dict[str, Any],
    *,
    client_id: str,
    token_endpoint: str,
    apply_config: bool = True,
    now: float | None = None,
) -> OAuthCredential:
    """Apply a fresh OAuth grant (an OAuthTokenResponse dict) to ``path`` for ``host``.
    Deep-merges the grant's ``config`` into the file root (preserving other hosts
    and root keys), then writes the host's ``apiKey`` and ``oauth`` block.
    ``apply_config=False`` stores tokens only."""
    now = time.time() if now is None else now
    cred = OAuthCredential.from_token_response(grant, now=now, client_id=client_id, token_endpoint=token_endpoint)
    raw = _read_config(path)
    granted_config = grant.get("config")
    if isinstance(granted_config, dict):
        cred.consent_peer_name = granted_config.get("peerName")
        if apply_config:
            _deep_merge(raw, granted_config)
    _persist_credential(path, host, cred, raw)
    return cred


def apply_token_to_client(client: Any, token: str) -> bool:
    """Rotate the live Honcho client's Bearer in place. The SDK builds its auth
    header per request from ``_http.api_key``, so mutating it rotates every
    holder of the singleton. Returns False on an SDK shape change (caller resets)."""
    http = getattr(client, "_http", None)
    if http is None or not hasattr(http, "api_key"):
        return False
    http.api_key = token
    return True
