"""Auth-gate middleware for the dashboard.

Engaged when ``app.state.auth_required is True``; a no-op otherwise (loopback
mode is handled by the legacy ``_SESSION_TOKEN`` ``auth_middleware``). Allows
the auth-bootstrap routes and static assets through unauthenticated; for
everything else demands a bearer token or a valid session cookie and attaches
the verified :class:`Session` to ``request.state.session``. HTML routes are
redirected to ``/login``; ``/api/*`` routes get a 401 JSON envelope.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from hermes_cli.dashboard_auth import list_session_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    ProviderError,
    RefreshExpiredError,
)
from hermes_cli.dashboard_auth.cookies import (
    clear_session_cookies,
    clear_sso_attempt_cookie,
    detect_https,
    read_session_cookies,
    read_session_provider,
    read_sso_attempt_cookie,
    set_session_cookies,
    set_session_provider_cookie,
    set_sso_attempt_cookie,
)
from hermes_cli.dashboard_auth.prefix import prefix_from_request
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
from hermes_cli.dashboard_auth.request_utils import (
    access_token_max_age as _expires_in_seconds,
    client_ip as _client_ip,
    extract_bearer as _extract_bearer,
    is_safe_next_path,
    unreachable_response,
)

_log = logging.getLogger(__name__)

# Prefix-matched (``path == p or path.startswith(p)``) bypass list: auth
# bootstrap routes and static asset mounts. ``/assets/`` with the trailing
# slash matches ``/assets/foo.css`` but not ``/assetsleak``.
_GATE_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/auth/login",
    "/auth/callback",
    "/auth/native/authorize",
    "/auth/native/token",
    "/auth/native/refresh",
    "/auth/password-login",
    "/auth/logout",
    "/login",
    "/api/auth/providers",
    "/api/mcp/oauth/callback/",
    "/assets/",
    "/favicon.ico",
    "/ds-assets/",
    "/fonts/",
    "/fonts-terminal/",
)


def _path_is_public(path: str) -> bool:
    """True if ``path`` bypasses the gate.

    :data:`PUBLIC_API_PATHS` (shared with the legacy middleware) is matched
    exactly so ``/api/status`` never exposes ``/api/status/extension``;
    :data:`_GATE_PUBLIC_PREFIXES` is prefix-matched.
    """
    if path in PUBLIC_API_PATHS:
        return True
    return any(path == p or path.startswith(p) for p in _GATE_PUBLIC_PREFIXES)


def _ordered_session_providers(
    provider_hint: str | None,
) -> list[DashboardAuthProvider]:
    """Session providers with the hinted one first (stable sort).

    The hint is a routing preference, not authoritative: a stale/unknown hint
    (provider renamed or removed) leaves the normal registration order intact.
    """
    providers = list_session_providers()
    if provider_hint:
        providers.sort(key=lambda provider: provider.name != provider_hint)
    return providers


def _safe_next_target(request: Request) -> str:
    """URL-encoded ``next`` value for the login redirect, or ``""``.

    Only same-origin relative paths outside the auth flow and ``/api`` are
    kept (see :func:`is_safe_next_path`); the query string is preserved. SPA
    deep links that are dropped fall back to the SPA's own
    ``sessionStorage["hermes.lastLocation"]``.
    """
    path = request.url.path
    if not path or not is_safe_next_path(path):
        return ""
    query = request.url.query
    return quote(f"{path}?{query}" if query else path, safe="")


def _unauth_response(request: Request, *, reason: str) -> Response:
    """API routes -> 401 JSON with ``login_url``; HTML routes -> 302 -> /login.

    fetch() follows a 302 opaquely into the cross-origin OAuth dance, so API
    routes never get redirects; the SPA's global 401 handler navigates to
    ``login_url`` when ``error`` is ``unauthenticated`` or ``session_expired``.
    Both shapes carry ``next=`` and the active proxy prefix.
    """
    next_param = _safe_next_target(request)
    prefix = prefix_from_request(request)
    login_url = f"{prefix}/login?next={next_param}" if next_param else f"{prefix}/login"

    if request.url.path.startswith("/api/"):
        error_code = (
            "session_expired" if reason == "invalid_or_expired_session"
            else "unauthenticated"
        )
        return JSONResponse(
            {
                "error": error_code,
                "detail": "Unauthorized",
                "reason": reason,
                "login_url": login_url,
            },
            status_code=401,
        )
    return RedirectResponse(url=login_url, status_code=302)


def _auto_sso_response(request: Request) -> Response | None:
    """302 straight to ``/auth/login`` on an unauthenticated HTML load, or ``None``.

    Only when: the request is a document load (not ``/api/*``); exactly one
    interactive provider is registered and it is OAuth-style (a password
    provider must render the form); and the one-shot loop-guard cookie is
    absent. A present marker means the portal had no session for us last time
    — clear it and fall back to ``/login`` rather than ping-pong. Removes the
    interstitial click, not a security check: ``/auth/login`` runs the
    unchanged PKCE flow.
    """
    if request.url.path.startswith("/api/"):
        return None

    if read_sso_attempt_cookie(request):
        resp = _unauth_response(request, reason="no_cookie")
        clear_sso_attempt_cookie(resp, prefix=prefix_from_request(request))
        return resp

    providers = list_session_providers()
    if len(providers) != 1:
        return None
    provider = providers[0]
    if getattr(provider, "supports_password", False):
        return None

    prefix = prefix_from_request(request)
    next_param = _safe_next_target(request)
    auth_login = f"{prefix}/auth/login?provider={quote(provider.name, safe='')}"
    if next_param:
        auth_login = f"{auth_login}&next={next_param}"

    resp = RedirectResponse(url=auth_login, status_code=302)
    set_sso_attempt_cookie(resp, use_https=detect_https(request), prefix=prefix)
    audit_log(
        AuditEvent.LOGIN_START,
        provider=provider.name, reason="auto_sso", ip=_client_ip(request),
    )
    return resp


def _verify_access_token(
    request: Request, *, access_token: str, provider_hint: str | None = None,
    audit: bool = True,
):
    """Run ``verify_session`` across the provider stack; Session or ``None``.

    A provider that does not recognise the token returns ``None`` and the
    next is tried. A ``ProviderError`` (IDP/JWKS unreachable) must NOT abort
    the chain — the token may belong to a different, reachable provider. If
    no provider verifies AND at least one was unreachable, raises
    ``ProviderError(name)`` so the caller returns 503 instead of forcing a
    re-login through a possibly-unreachable refresh.
    """
    unreachable_provider: str | None = None
    for provider in _ordered_session_providers(provider_hint):
        try:
            session = provider.verify_session(access_token=access_token)
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: provider %r unreachable during %s: %s",
                provider.name, "verify" if audit else "bearer verify", e,
            )
            if audit:
                audit_log(
                    AuditEvent.SESSION_VERIFY_FAILURE,
                    provider=provider.name,
                    reason="provider_unreachable",
                    ip=_client_ip(request),
                )
            if unreachable_provider is None:
                unreachable_provider = provider.name
            continue
        if session is not None:
            return session
    if unreachable_provider is not None:
        raise ProviderError(unreachable_provider)
    return None


def _verify_bearer(request: Request, *, access_token: str):
    """Verify a native-app bearer token (no cookie, no server-side refresh —
    the desktop rotates via ``/auth/native/refresh``). Same 503-on-outage
    semantics as the cookie path.
    """
    return _verify_access_token(request, access_token=access_token, audit=False)


async def gated_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Engaged only when ``app.state.auth_required is True``."""
    if not getattr(request.app.state, "auth_required", False):
        return await call_next(request)

    # Already authenticated by the token-auth seam (service caller on a
    # registered token route): not a cookie session, must not bounce to /login.
    if getattr(request.state, "token_authenticated", False):
        return await call_next(request)

    if _path_is_public(request.url.path):
        return await call_next(request)

    # RFC 8252 native-app bearer path: the same provider-minted access token
    # the cookie flow stores, verified with the same provider stack, no cookie
    # read or set. A presented-but-invalid bearer gets the structured 401 so
    # the desktop refreshes/re-logs instead of following a cookie redirect.
    bearer = _extract_bearer(request)
    if bearer:
        try:
            bearer_session = _verify_bearer(request, access_token=bearer)
        except ProviderError as e:
            return unreachable_response(str(e))
        if bearer_session is not None:
            request.state.session = bearer_session
            return await call_next(request)
        return _unauth_response(request, reason="invalid_or_expired_session")

    at, _rt = read_session_cookies(request)
    provider_hint = read_session_provider(request)
    if not at and not _rt:
        # No session at all: try the silent portal bounce before /login.
        auto = _auto_sso_response(request)
        if auto is not None:
            return auto
        return _unauth_response(request, reason="no_cookie")

    # An absent AT with a present RT is the COMMON expiry case (the AT cookie's
    # Max-Age tracks the token TTL, so the browser evicts it first) — skip
    # straight to refresh rather than bouncing to /login.
    session = None
    if at:
        try:
            session = _verify_access_token(
                request, access_token=at, provider_hint=provider_hint,
            )
        except ProviderError as e:
            return unreachable_response(str(e))

    if session is None:
        # Rotate via the refresh token before forcing re-login. On success the
        # rotated cookies are re-set and the request served transparently.
        try:
            refreshed = _attempt_refresh(
                request, refresh_token=_rt, provider_hint=provider_hint,
            )
        except ProviderError as e:
            # Uncertain (provider unreachable), not rejected: keep the cookies.
            return unreachable_response(str(e))
        if refreshed is not None:
            new_session, refreshing_provider = refreshed
            request.state.session = new_session
            response = await call_next(request)
            # Writing the ROTATED RT back is mandatory: Portal runs reuse
            # detection, so replaying the stale RT would revoke the session.
            set_session_cookies(
                response,
                access_token=new_session.access_token,
                refresh_token=new_session.refresh_token,
                access_token_expires_in=_expires_in_seconds(new_session),
                use_https=detect_https(request),
                prefix=prefix_from_request(request),
                provider=refreshing_provider,
            )
            audit_log(
                AuditEvent.REFRESH_SUCCESS,
                provider=refreshing_provider,
                user_id=new_session.user_id,
                ip=_client_ip(request),
            )
            return response

        audit_log(
            AuditEvent.SESSION_VERIFY_FAILURE,
            reason="no_provider_recognises",
            ip=_client_ip(request),
        )
        response = _unauth_response(request, reason="invalid_or_expired_session")
        # Refresh failed (or no RT): clear the dead cookies under the active
        # prefix so the deletion Path matches the set Path.
        clear_session_cookies(response, prefix=prefix_from_request(request))
        return response

    request.state.session = session
    response = await call_next(request)
    if not provider_hint and session.provider:
        set_session_provider_cookie(
            response,
            provider=session.provider,
            use_https=detect_https(request),
            prefix=prefix_from_request(request),
        )
    return response


def _attempt_refresh(request: Request, *, refresh_token, provider_hint: str | None = None):
    """Rotate an expired session via the refresh token; ``(Session, provider_name)`` or ``None``.

    ``RefreshExpiredError`` rejects the token for that candidate only (Basic
    raises it for foreign opaque tokens too), so remaining providers are
    tried. If none succeeds and any raised ``ProviderError``, re-raises with
    that provider's name so the caller returns 503 without clearing cookies.
    """
    if not refresh_token:
        return None
    unavailable_provider: str | None = None
    for provider in _ordered_session_providers(provider_hint):
        try:
            new_session = provider.refresh_session(refresh_token=refresh_token)
        except RefreshExpiredError:
            audit_log(
                AuditEvent.REFRESH_FAILURE,
                provider=provider.name,
                reason="refresh_expired",
                ip=_client_ip(request),
            )
            continue
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: provider %r unreachable during refresh: %s",
                provider.name, e,
            )
            audit_log(
                AuditEvent.REFRESH_FAILURE,
                provider=provider.name,
                reason="provider_unreachable",
                ip=_client_ip(request),
            )
            if unavailable_provider is None:
                unavailable_provider = provider.name
            continue
        if new_session is not None:
            return new_session, provider.name
    if unavailable_provider is not None:
        raise ProviderError(unavailable_provider)
    return None
