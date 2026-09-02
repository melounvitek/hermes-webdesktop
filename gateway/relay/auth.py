"""Gateway-side relay authentication primitives. EXPERIMENTAL.

Gateway half of two HMAC schemes whose wire bytes must match the connector's
TypeScript exactly:

1. **WS upgrade auth** (gateway → connector): ``Authorization: Bearer <token>``
   on the ``/relay`` upgrade, ``token = make_upgrade_token(gateway_id, secret)``.
   Mirrors ``relayAuthToken.ts`` ``makeToken``: ``base64url(f"{payload}:{exp}:{sig}")``
   with ``sig = HMAC_SHA256(f"{payload}:{exp}", secret).hexdigest()``.
2. **Inbound delivery signature** (connector → gateway): ``x-relay-timestamp`` +
   ``x-relay-signature`` headers, ``sig = HMAC_SHA256(f"{ts}.{body_json}", key)``
   over the EXACT body bytes, with a replay-window skew check (``deliverySigning.ts``).

Both verify against a multi-secret list (primary first, then a secondary during
a rotation window) so a rotation doesn't invalidate outstanding tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional, Sequence

# Header names used by the connector's deliverySigning.ts.
DELIVERY_TS_HEADER = "x-relay-timestamp"
DELIVERY_SIG_HEADER = "x-relay-signature"

_DEFAULT_MAX_SKEW_SECONDS = 300  # connector default replay window
_DEFAULT_UPGRADE_TTL_SECONDS = 300  # connector makeUpgradeToken default


def sign(payload: str, secret: str) -> str:
    """HMAC-SHA256 hex digest (UTF-8) — the connector's ``sign``."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_signature(payload: str, sig_hex: str, secrets: Sequence[str]) -> bool:
    """Constant-time check of ``sig_hex`` against ANY of ``secrets`` (rotation window).

    Length-mismatched candidates are skipped without a timing leak.
    """
    try:
        sig_buf = bytes.fromhex(sig_hex)
    except (ValueError, TypeError):
        return False
    if len(sig_buf) == 0:
        return False
    for secret in secrets:
        if not secret:
            continue
        expected = bytes.fromhex(sign(payload, secret))
        if len(expected) != len(sig_buf):
            continue
        if hmac.compare_digest(sig_buf, expected):
            return True
    return False


def make_token(payload: str, secret: str, ttl_seconds: int = 0) -> str:
    """``base64url(f"{payload}:{exp}:{sig}")``; ``exp`` unix seconds (0 = never).

    base64url is unpadded to match Node's ``Buffer.toString("base64url")``.
    """
    exp = int(time.time()) + ttl_seconds if ttl_seconds > 0 else 0
    signed = f"{payload}:{exp}"
    raw = f"{signed}:{sign(signed, secret)}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_upgrade_token(
    gateway_id: str, secret: str, ttl_seconds: int = _DEFAULT_UPGRADE_TTL_SECONDS
) -> str:
    """WS-upgrade bearer: ``payload = gateway_id`` (the connector peeks it to index its verify list)."""
    return make_token(gateway_id, secret, ttl_seconds)


def verify_token(token: str, secrets: Sequence[str]) -> Optional[str]:
    """Verify a ``make_token`` token; return the payload or None.

    Splits from the right so a payload may itself contain colons.
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, TypeError):
        return None
    parts = decoded.split(":")
    if len(parts) < 3:
        return None
    sig = parts[-1]
    try:
        exp = int(parts[-2])
    except ValueError:
        return None
    payload = ":".join(parts[:-2])
    if exp != 0 and int(time.time()) > exp:
        return None
    return payload if verify_signature(f"{payload}:{exp}", sig, secrets) else None


def verify_delivery_signature(
    body_json: str,
    timestamp: Optional[str],
    signature: Optional[str],
    verify_keys: Sequence[str],
    max_skew_seconds: int = _DEFAULT_MAX_SKEW_SECONDS,
    *,
    now: Optional[int] = None,
) -> bool:
    """Verify a connector→gateway inbound delivery signature.

    ``body_json`` MUST be the exact request body bytes decoded as UTF-8 (no
    re-serialization — the connector signed the literal body).
    """
    if not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False
    current = now if now is not None else int(time.time())
    if abs(current - ts) > max_skew_seconds:
        return False
    return verify_signature(f"{ts}.{body_json}", signature, verify_keys)
