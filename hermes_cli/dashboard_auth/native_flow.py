"""Gateway-brokered RFC 8252 (OAuth 2.0 for Native Apps) authorization store.

The desktop cannot be a direct OAuth client of the upstream IDP (the Portal
``client_id`` is per gateway instance and only accepts the gateway's own
``/auth/callback`` redirect), so the gateway brokers: it is the authorization
server *to the desktop* and an OAuth client *to the Portal* — still textbook
RFC 8252: system browser, loopback redirect, PKCE, tokens returned to the app
and never as cookies.

  1. Desktop generates its own PKCE pair (cv_d, cc_d) + ``state``, opens a
     loopback listener, and opens the system browser at
     ``/auth/native/authorize`` with cc_d, state and its loopback redirect_uri.
  2. The gateway stashes a *pending authorization* (:func:`register_pending`)
     keyed by an opaque ``broker_state`` and runs the EXISTING upstream flow;
     broker_state rides inside the gateway's own PKCE cookie, so no desktop
     secret reaches the Portal.
  3. On the upstream callback (or a successful password login) the gateway
     mints a one-time gateway code bound to cc_d (:func:`complete_pending`) and
     302s the browser to ``redirect_uri?code=<gw_code>&state=<state>``.
  4. The desktop POSTs ``/auth/native/token`` with gw_code + cv_d; the gateway
     checks ``S256(cv_d) == cc_d`` (:func:`redeem_code`), consumes the code and
     returns the upstream tokens in the JSON body.
  5. The desktop keeps them in the OS keychain and uses ``Authorization: Bearer``.

Security properties: PKCE binding (an intercepted gw_code is useless without
cv_d), single use (redemption pops the entry), short TTLs, 256-bit opaque
handles compared in constant time, no secret logging. In-memory and
process-local (single dashboard process); functional API keeps ``time.time``
patchable in tests.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from hermes_cli.dashboard_auth.base import Session

# Pending authorization: the whole interactive login (mirrors the PKCE cookie).
_PENDING_TTL_SECONDS = 600
# Minted code: only the loopback redirect + immediate token POST.
_CODE_TTL_SECONDS = 120
# Global cap so a misbehaving client cannot grow the store unbounded.
_MAX_ENTRIES = 256
# Per-IP cap on PENDING entries: /auth/native/authorize is a public pre-auth
# route, so one spammer must not fill the global store (600 s each) and lock
# out legitimate native logins.
_MAX_PENDING_PER_IP = 8

_lock = threading.Lock()


@dataclass
class _Pending:
    """In-flight native authorization awaiting the upstream callback."""

    code_challenge: str  # the DESKTOP's S256 challenge (cc_d), base64url no-pad
    redirect_uri: str  # the desktop's loopback redirect (127.0.0.1:<port>/...)
    client_state: str  # the desktop's own ``state`` (echoed back on redirect)
    client_ip: str  # requester IP at authorize time (per-IP pending cap)
    expires_at: int


@dataclass
class _IssuedCode:
    """A minted one-time gateway authorization code bound to a Session."""

    code_challenge: str  # cc_d — verified against cv_d at redemption
    session: Session
    expires_at: int


_pending: Dict[str, _Pending] = {}  # broker_state -> _Pending
_issued: Dict[str, _IssuedCode] = {}  # gw_code -> _IssuedCode


class NativeFlowError(Exception):
    """Base for native-flow failures (bad/expired/replayed handle, PKCE fail)."""


class PendingNotFound(NativeFlowError):
    """The broker_state is unknown or expired (login window lapsed)."""


class CodeInvalid(NativeFlowError):
    """The gateway code is unknown, expired, already redeemed, or PKCE-mismatched."""


def _b64url_no_pad(raw: bytes) -> str:
    """Base64url without ``=`` padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _s256(verifier: str) -> str:
    """RFC 7636 S256 transform: base64url(sha256(ascii(verifier)))."""
    return _b64url_no_pad(hashlib.sha256(verifier.encode("ascii")).digest())


def _gc_locked(now: int) -> None:
    """Drop expired pending + issued entries. Caller holds ``_lock``."""
    for store in (_pending, _issued):
        for k in [k for k, v in store.items() if v.expires_at < now]:
            store.pop(k, None)


def _capacity_ok_locked() -> bool:
    return (len(_pending) + len(_issued)) < _MAX_ENTRIES


def _now(now: Optional[int]) -> int:
    return int(time.time()) if now is None else now


def register_pending(
    *,
    code_challenge: str,
    redirect_uri: str,
    client_state: str,
    client_ip: str = "",
    now: Optional[int] = None,
) -> str:
    """Stash a pending native authorization; return an opaque ``broker_state``.

    ``code_challenge`` is the DESKTOP's cc_d (the verifier is never seen until
    redemption). Raises ``NativeFlowError`` (fail closed) when the store is at
    capacity or ``client_ip`` already holds ``_MAX_PENDING_PER_IP`` entries.
    """
    now = _now(now)
    broker_state = secrets.token_urlsafe(32)
    with _lock:
        _gc_locked(now)
        if not _capacity_ok_locked():
            raise NativeFlowError("native-flow authorization store at capacity")
        if client_ip and (
            sum(1 for v in _pending.values() if v.client_ip == client_ip)
            >= _MAX_PENDING_PER_IP
        ):
            raise NativeFlowError(
                "too many pending native authorizations from this address"
            )
        _pending[broker_state] = _Pending(
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            client_state=client_state,
            client_ip=client_ip,
            expires_at=now + _PENDING_TTL_SECONDS,
        )
    return broker_state


def get_pending(broker_state: str, *, now: Optional[int] = None) -> _Pending:
    """Peek (without consuming) the pending authorization; raises
    :class:`PendingNotFound` if unknown or expired."""
    now = _now(now)
    with _lock:
        _gc_locked(now)
        entry = _pending.get(broker_state)
        if entry is None:
            raise PendingNotFound("unknown or expired native authorization")
        return entry


def complete_pending(
    broker_state: str,
    *,
    session: Session,
    now: Optional[int] = None,
) -> str:
    """Consume a pending authorization (single use) and mint a one-time gateway
    code bound to the desktop's challenge + the verified ``session``.

    Raises :class:`PendingNotFound` if the broker_state is unknown/expired.
    """
    now = _now(now)
    with _lock:
        _gc_locked(now)
        pending = _pending.pop(broker_state, None)
        if pending is None:
            raise PendingNotFound("unknown or expired native authorization")
        if not _capacity_ok_locked():
            raise NativeFlowError("native-flow code store at capacity")
        gw_code = secrets.token_urlsafe(32)
        _issued[gw_code] = _IssuedCode(
            code_challenge=pending.code_challenge,
            session=session,
            expires_at=now + _CODE_TTL_SECONDS,
        )
    return gw_code


def redeem_code(
    *,
    code: str,
    code_verifier: str,
    now: Optional[int] = None,
) -> Session:
    """Verify PKCE + consume a gateway code; return the bound :class:`Session`.

    The entry is popped BEFORE the PKCE check so a wrong verifier cannot be
    retried against the same code: on any failure the code is already consumed
    (no oracle, no replay). Raises :class:`CodeInvalid`.
    """
    now = _now(now)
    with _lock:
        _gc_locked(now)
        issued = _issued.pop(code, None)
    if issued is None:
        raise CodeInvalid("unknown, expired, or already-redeemed code")
    if issued.expires_at < now:
        raise CodeInvalid("code expired")
    if not hmac.compare_digest(issued.code_challenge, _s256(code_verifier)):
        raise CodeInvalid("PKCE verification failed")
    return issued.session


def _reset_for_tests() -> None:
    """Test-only: drop all pending + issued state."""
    with _lock:
        _pending.clear()
        _issued.clear()
