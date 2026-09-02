"""Nous Portal Remote Spending HTTP client (Phase 2b).

Thin, fail-loud client for the four ``/api/billing/*`` endpoints the terminal billing screens drive.
Companion to ``hermes_cli/nous_account.py`` (which owns read-only entitlement/balance) — this module
owns the *write* side: buy credits, poll a charge, configure auto-reload.

- **Money is decimal, never float.** The server emits decimal STRINGS (``"142.5"`` — not fixed 2dp).
We parse with :class:`decimal.Decimal` and never round-trip through float.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

DEFAULT_PORTAL_BASE_URL = "https://portal.nousresearch.com"

# Default HTTP timeout (seconds). Charge/poll calls are quick; keep this tight so
# a hung portal doesn't freeze the TUI.
DEFAULT_TIMEOUT = 15.0


# =============================================================================
# Typed errors
# =============================================================================


class BillingError(Exception):
    """A billing HTTP call failed.

    Carries what a surface needs to render the right message and affordance: server ``error``
    code, HTTP ``status``, optional ``message``, the ``portalUrl`` deep-link (present on every
    gate denial), ``retry_after`` seconds (429/503), and the parsed ``payload`` when available.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        error: Optional[str] = None,
        portal_url: Optional[str] = None,
        retry_after: Optional[int] = None,
        payload: Optional[dict[str, Any]] = None,
        actor: Optional[str] = None,
        code: Optional[str] = None,
        recovery: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.portal_url = portal_url
        self.retry_after = retry_after
        self.payload = payload or {}
        # Remote-Spending contract extras (NAS PR #481): `actor` (self|admin) on a
        # revoke, `code` (the new machine code dual-emitted alongside `error`), and
        # `recovery` (reconnect|login|enable_account_toggle). Additive — absent on
        # older NAS / unrelated errors.
        self.actor = actor
        self.code = code
        self.recovery = recovery


class BillingScopeRequired(BillingError):
    """``403 insufficient_scope`` — the held token lacks ``billing:manage``.

    The lazy step-up trigger: catching this kicks off a fresh device-connect that requests
    ``billing:manage`` (and tells the user an ADMIN must select "Allow Remote Spending"). Also fires
    mid-session if the scope is stripped on refresh after the user loses ADMIN.
    """


class BillingAuthError(BillingError):
    """``401`` — missing/invalid bearer token (not logged in / expired)."""


class BillingRemoteSpendingRevoked(BillingError):
    """``403 remote_spending_revoked`` — THIS terminal's spending was revoked.

    Distinct from ``insufficient_scope`` (never had the grant) and from ``session_revoked`` (full
    logout). The terminal stays logged in; only the money path is cut. ``actor`` is ``"admin"`` or
    ``"self"`` (absent → treat as ``"self"``); recovery is **reconnect** (re-consent device-auth).
    """


class BillingSessionRevoked(BillingAuthError):
    """``401 session_revoked`` — the whole session was logged out.

    Stronger than a spend-revoke: recovery is **re-login** (full device-auth), not just reconnect.
    Subclass of :class:`BillingAuthError` so existing 401 handling still treats it as not-logged-in,
    but the typed code lets the surface route to re-login with the right copy.
    """


class BillingTransient(BillingError):
    """A deterministic non-charge outcome: the request definitely did NOT reach/complete at Stripe, so
    it's always safe to retry after backoff — never the "maybe charged" ambiguity of a real
    5xx/timeout. Covers 429 rate limiting, 503 gate-unavailable, Stripe being down, and the daily
    upgrade cap — distinct failure modes that share this one contract property. Catch this (not the
    old ad-hoc subclass hierarchy) wherever the intent is "any transient, definitely-not-charged
    billing failure, back off and retry/poll".
    """


class BillingRateLimited(BillingTransient):
    """``429 rate_limited`` or ``503 temporarily_unavailable``.

    NOT a payment failure. Carries ``retry_after`` (seconds) — back off and tell the user "try again
    in N min"; never auto-retry-spam (the limiter is 5/org/hr + 5/token/hr and easy to dig deeper
    into). A 503 is the gate backend failing closed — back off, do NOT treat as revoked.
    """


class BillingStripeUnavailable(BillingTransient):
    """``503 stripe_unavailable`` — Stripe itself is down.

    TRANSIENT: back off and retry using Retry-After; this is NOT the same as being throttled by our
    own rate limiter, so surfaces must not render "rate limited" copy for it — they should read
    ``.error`` to tell the two apart.
    """


class BillingUpgradeCapExceeded(BillingTransient):
    """``429 upgrade_cap_exceeded`` — the org hit its 5-upgrades/day cap.

    Distinct from the hourly ``rate_limited`` charge cap (same HTTP status, different meaning + no
    useful short-Retry-After backoff). A BillingTransient sibling of BillingRateLimited (not a
    subclass) — surfaces must read ``.error`` to distinguish the failure mode.
    """


# =============================================================================
# Base-URL + auth resolution
# =============================================================================


def resolve_portal_base_url(state: Optional[dict[str, Any]] = None) -> str:
    """Resolve the portal base URL with login-time precedence."""
    env = os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    if state:
        stored = state.get("portal_base_url")
        if isinstance(stored, str) and stored.strip():
            return stored.strip().rstrip("/")
    return DEFAULT_PORTAL_BASE_URL


def _absolutize_portal_url(portal_url: Optional[str]) -> Optional[str]:
    """Resolve a (possibly relative) server portalUrl to an absolute URL.

    The server emits ``portalUrl`` relative by design — it doesn't know which deployment the
    client points at — so it is resolved against the client's portal base (preview/staging/prod)
    to be clickable. Idempotent: absolute URLs pass through unchanged.
    """
    if not (isinstance(portal_url, str) and portal_url.strip()):
        return portal_url
    base = resolve_portal_base_url()
    # urljoin needs a trailing slash on the base to treat it as a directory and
    # join an absolute path like "/billing?..." against the host. An already-
    # absolute portal_url (with its own scheme/host) is returned as-is.
    return urllib.parse.urljoin(base.rstrip("/") + "/", portal_url)


# Short-lived cache for the resolved (token, base). `resolve_nous_access_token`
# acquires two cross-process file locks + reads two files on every call (even on
# its fast path), which is wasteful when the 2s/5-min charge poll loop calls a
# billing endpoint ~150x per purchase. Cache the result briefly: the resolver
# only ever returns a token with >=120s of life (its refresh skew), so a 30s
# cache can never hand back an about-to-expire token. A 401 still surfaces
# normally (the cache holds a valid token, not the HTTP outcome).
_TOKEN_CACHE_TTL_SECONDS = 30.0
_token_cache: tuple[float, str, str] | None = None  # (cached_at, token, base)


def invalidate_cached_token() -> None:
    """Bust the 30s token cache so post-step-up replays use the freshly-scoped token.

    ``_request`` only self-busts the cache on a 401 (an expired/invalid token), not on a 403 scope
    denial — so after a step-up grant, the cache would otherwise still hold the pre-grant unscoped
    token and the immediate replay would 403 again. Callers outside this module (e.g.
    """
    global _token_cache
    _token_cache = None


def _billing_not_logged_in(exc: Optional[BaseException] = None) -> "BillingAuthError":
    """Build the canonical 'not logged in' BillingAuthError (single source)."""
    err = BillingAuthError(
        "Not logged into Nous Portal — run `hermes portal` to log in.",
        status=401,
        error="invalid_token",
    )
    if exc is not None:
        err.__cause__ = exc
    return err


def _resolve_token_and_base(*, use_cache: bool = True) -> tuple[str, str]:
    """Return ``(access_token, portal_base_url)`` for billing calls.

    The result is cached for ``_TOKEN_CACHE_TTL_SECONDS`` to keep the charge poll loop from re-
    locking + re-reading the auth store on every 2s tick. Pass ``use_cache=False`` to force a fresh
    resolution (e.g. after a 401).
    """
    global _token_cache

    if use_cache and _token_cache is not None:
        cached_at, token, base = _token_cache
        if (time.time() - cached_at) < _TOKEN_CACHE_TTL_SECONDS:
            return token, base

    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception:
        state = {}

    base = resolve_portal_base_url(state)

    try:
        from hermes_cli.auth import AuthError, resolve_nous_access_token
    except ImportError:
        # auth module unavailable — fall back to the raw stored token.
        token = state.get("access_token")
        if not (isinstance(token, str) and token.strip()):
            raise _billing_not_logged_in()
    else:
        try:
            token = resolve_nous_access_token()
        except AuthError as exc:
            raise _billing_not_logged_in(exc) from exc
    resolved = (token.strip(), base)
    _token_cache = (time.time(), *resolved)
    return resolved


# =============================================================================
# HTTP plumbing
# =============================================================================


def _retry_after_seconds(headers: Any) -> Optional[int]:
    """Parse a ``Retry-After`` header (integer seconds) — None if absent/bad."""
    from agent.retry_utils import parse_retry_after_seconds

    seconds = parse_retry_after_seconds(headers)
    return None if seconds is None else int(seconds)


# Error routing for _raise_for_error: server ``error`` code alone, then
# (status, error), then status alone, then the generic fallback.  Values are
# (exception class, fallback message when the server sent no ``message``).
# session_revoked is a full logout (→ re-login), stronger than a 401 expired
# token; both stay BillingAuthError-compatible.  remote_spending_revoked is NOT
# the same as never having the scope: disable spend UI, recovery is reconnect.
# Business 403s (cli_billing_disabled / role_required / no_payment_method /
# monthly_cap_exceeded / …) fall through to a generic BillingError with
# code/recovery, using the raw error code as the message.
_ERRORS_BY_CODE: dict[str, tuple[type[BillingError], str]] = {
    "stripe_unavailable": (BillingStripeUnavailable, "Stripe is temporarily unavailable — try again shortly."),
    "upgrade_cap_exceeded": (BillingUpgradeCapExceeded, "Daily plan-change limit reached — try again tomorrow."),
}
_ERRORS_BY_STATUS_CODE: dict[tuple[int, str], tuple[type[BillingError], str]] = {
    (401, "session_revoked"): (BillingSessionRevoked, "Your session was logged out — log in again."),
    (403, "remote_spending_revoked"): (BillingRemoteSpendingRevoked, "Remote spending was stopped for this terminal."),
    (403, "insufficient_scope"): (BillingScopeRequired, "This action needs the billing:manage scope."),
}
_ERRORS_BY_STATUS: dict[int, tuple[type[BillingError], str]] = {
    401: (BillingAuthError, "Authentication required."),
    403: (BillingError, "Billing request denied."),
    429: (BillingRateLimited, "Rate limited — try again shortly."),
    503: (BillingRateLimited, "Rate limited — try again shortly."),
}


def _raise_for_error(status: int, payload: dict[str, Any], headers: Any = None) -> None:
    """Map an HTTP error response to the right typed :class:`BillingError`.

    Recognizes the Remote-Spending gate contract: 403 ``remote_spending_revoked`` (reconnect),
    401 ``session_revoked`` (re-login), 503 ``temporarily_unavailable`` (fail-closed → back off,
    NOT revoked). Business-denial codes flow through as a generic BillingError carrying
    ``error``/``code``/``recovery`` for the surface to map.
    """
    p = payload if isinstance(payload, dict) else {}
    error = p.get("error")
    message = p.get("message")
    common = {
        "status": status,
        "error": error,
        "portal_url": _absolutize_portal_url(p.get("portalUrl")),
        "retry_after": _retry_after_seconds(headers),
        "payload": p,
        "actor": p.get("actor"),
        "code": p.get("code"),
        "recovery": p.get("recovery"),
    }
    key = error if isinstance(error, str) else None
    cls, fallback = (
        _ERRORS_BY_CODE.get(key)
        or _ERRORS_BY_STATUS_CODE.get((status, key))
        or _ERRORS_BY_STATUS.get(status)
        or (BillingError, f"Billing request failed ({status}).")
    )
    raise cls(message or (error if cls is BillingError else None) or fallback, **common)


def _request(
    method: str,
    path: str,
    *,
    body: Optional[dict[str, Any]] = None,
    extra_headers: Optional[dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    _retried_auth: bool = False,
) -> dict[str, Any]:
    """Make an authenticated billing request; return the parsed JSON dict.

    Raises a typed :class:`BillingError` on any non-2xx response (or transport failure). 2xx with an
    empty body returns ``{}``. A 401 triggers exactly one retry with a freshly-resolved token
    (bypassing the short token cache) so a cached-but-just-expired token self-heals instead of
    failing the call.
    """
    token, base = _resolve_token_and_base(use_cache=not _retried_auth)
    url = f"{base}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                # A 2xx with a non-JSON body means the endpoint isn't actually
                # serving the billing API here — e.g. a reverse-proxy / SPA
                # fallback HTML page when the route isn't deployed on this
                # deployment. Surface it as a typed, non-auth error so callers
                # degrade gracefully ("unavailable") instead of crashing with a
                # raw JSONDecodeError that reads as "not logged in".
                raise BillingError(
                    "Billing endpoint returned a non-JSON response "
                    "(it may not be available on this deployment).",
                    error="endpoint_unavailable",
                    status=getattr(resp, "status", None),
                ) from exc
    except urllib.error.HTTPError as exc:
        # A 401 on a cached token → drop the cache and retry once with a fresh
        # (refresh-aware) resolve before surfacing the auth error.
        if exc.code == 401 and not _retried_auth:
            invalidate_cached_token()
            return _request(
                method,
                path,
                body=body,
                extra_headers=extra_headers,
                timeout=timeout,
                _retried_auth=True,
            )
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            raw = ""
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {}
        _raise_for_error(exc.code, payload, getattr(exc, "headers", None))
        raise  # unreachable; _raise_for_error always raises
    except urllib.error.URLError as exc:
        raise BillingError(
            f"Could not reach Nous Portal: {exc.reason}", error="network_error"
        ) from exc
    except TimeoutError as exc:
        # urlopen() wraps CONNECT-phase timeouts in URLError, but a timeout
        # during resp.read() surfaces as a bare TimeoutError — normalize it so
        # transport failures always honor the typed-BillingError contract.
        raise BillingError(
            "Could not reach Nous Portal: timed out", error="network_error"
        ) from exc


# =============================================================================
# The four endpoints
# =============================================================================


def _require_str(value: Any, message: str, error: str) -> str:
    """Return ``value.strip()`` or raise a typed BillingError when it is not a non-blank str."""
    if not (isinstance(value, str) and value.strip()):
        raise BillingError(message, error=error)
    return value.strip()


def _post_idempotent(path: str, body: dict[str, Any], idempotency_key: str, what: str, timeout: float) -> dict[str, Any]:
    """POST with a mandatory ``Idempotency-Key`` header (missing header is a server 400)."""
    key = _require_str(
        idempotency_key, f"Idempotency-Key is required for {what}.", "idempotency_key_required"
    )
    return _request("POST", path, body=body, extra_headers={"Idempotency-Key": key}, timeout=timeout)


def get_billing_state(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/state`` — role-tiered overview (no scope required)."""
    return _request("GET", "/api/billing/state", timeout=timeout)


def patch_auto_top_up(
    *,
    enabled: bool,
    threshold: float | str,
    top_up_amount: float | str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``PATCH /api/billing/auto-top-up`` — configure auto-reload (scope required).

    Body is strict server-side: extra keys (``maxMonthlySpend``, a payment method) are rejected with
    400. Numbers are sent as JSON numbers per the contract.
    """
    body = {"enabled": bool(enabled), "threshold": float(threshold), "topUpAmount": float(top_up_amount)}
    return _request("PATCH", "/api/billing/auto-top-up", body=body, timeout=timeout)


def post_charge(
    *,
    amount_usd: float | str,
    idempotency_key: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``POST /api/billing/charge`` — buy credits (scope required).

    ``Idempotency-Key`` is MANDATORY (missing header is a server 400): generate a UUID per user-
    confirmed purchase and reuse it on retry. Returns ``202 {chargeId}`` — money is NOT
    confirmed yet; poll with :func:`get_charge_status`.
    """
    return _post_idempotent(
        "/api/billing/charge", {"amountUsd": float(amount_usd)}, idempotency_key, "a charge", timeout
    )


def get_charge_status(charge_id: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/charge/{id}`` — poll a charge (scope required).

    Returns ``{status: "pending"|"settled"|"failed", ...}``. An unknown or foreign id returns
    ``{status:"pending"}`` (never 404, never another org's data) — so a ``pending`` that never
    resolves past the 5-min cap is a *timeout*, not an error.
    """
    charge_id = _require_str(charge_id, "A charge id is required.", "invalid_charge_id")
    # urllib does not need manual quoting for the opaque ids the server mints, but
    # guard against a stray slash that would change the path shape.
    safe_id = urllib.parse.quote(charge_id, safe="")
    return _request("GET", f"/api/billing/charge/{safe_id}", timeout=timeout)


def get_subscription_state(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/subscription`` — current plan, tiers, usage (no scope).

    Returns the raw JSON dict from NAS (WS1 Phase A). Read-only — no ``billing:manage`` scope
    required. Raises :class:`BillingAuthError` on 401 and :class:`BillingError` on other non-2xx.
    """
    return _request("GET", "/api/billing/subscription", timeout=timeout)


# =============================================================================
# Subscription change (V3) — preview + the pending-change resource + upgrade
# =============================================================================
#
# Mutating the plan splits into a chargeless lane and the single money route:
#   - preview  → a quote (no mutation, no charge) of what a change would do.
#   - PUT/DELETE pending-change → schedule / clear a downgrade or cancellation
#     (chargeless; takes effect at period end).
#   - POST upgrade → the ONE route that charges (prorate + charge the card on the
#     subscription + flip the plan, in one Stripe op).
# All require the ``billing:manage`` scope (a 403 insufficient_scope raises
# :class:`BillingScopeRequired`, driving the device step-up) — including preview,
# which issues live Stripe calls and reveals charge amounts.


def post_subscription_preview(*, subscription_type_id: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``POST /api/billing/subscription/preview`` — a chargeless effect quote.

    Quotes a change to ``subscription_type_id`` without mutating anything: ``effect`` is
    ``charge_now`` (an upgrade → ``amountDueNowCents`` is the prorated upfront charge),
    ``scheduled`` (a downgrade → ``effectiveAt`` is period end), ``no_op`` (already on the tier), or
    ``blocked`` (``reason`` says why the commit would be refused).
    """
    return _request(
        "POST", "/api/billing/subscription/preview",
        body={"subscriptionTypeId": subscription_type_id}, timeout=timeout,
    )


def put_subscription_pending_change(
    *,
    subscription_type_id: str | None = None,
    cancel: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``PUT /api/billing/subscription/pending-change`` — set the end-of-period intent.

    A subscription has at most one pending disposition: ``cancel=True`` schedules a
    cancellation, a ``subscription_type_id`` schedules a downgrade / same-price change. UPGRADES
    are rejected here — they charge immediately, use :func:`post_subscription_upgrade`.
    Chargeless; needs ``billing:manage``.
    """
    if cancel:
        body: dict[str, Any] = {"type": "cancellation"}
    else:
        body = {
            "type": "tier_change",
            "subscriptionTypeId": _require_str(
                subscription_type_id,
                "A subscription tier is required to schedule a plan change.",
                "invalid_subscription_type",
            ),
        }
    return _request("PUT", "/api/billing/subscription/pending-change", body=body, timeout=timeout)


def delete_subscription_pending_change(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``DELETE /api/billing/subscription/pending-change`` — clear it (resume / undo).

    Removes a scheduled downgrade OR cancellation in one call, restoring the active tier and
    renewal. Chargeless, but it re-enables recurring spend, so it requires ``billing:manage``
    and honors the org kill-switch.
    """
    return _request("DELETE", "/api/billing/subscription/pending-change", timeout=timeout)


def post_subscription_upgrade(
    *,
    subscription_type_id: str,
    idempotency_key: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``POST /api/billing/subscription/upgrade`` — immediate paid upgrade.

    The SINGLE money route: one Stripe op prorates, charges the card already on the subscription,
    and flips the plan. ``Idempotency-Key`` is MANDATORY (a missing header is a server 400, not a
    default) — reuse the same key on retry so a replay cannot double-charge.
    """
    return _post_idempotent(
        "/api/billing/subscription/upgrade",
        {"subscriptionTypeId": subscription_type_id},
        idempotency_key,
        "an upgrade",
        timeout,
    )
