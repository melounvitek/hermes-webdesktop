"""Request-level helpers shared by the auth routes and both middlewares."""
from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import JSONResponse

# Paths a post-login redirect must never land on: the auth flow itself (would
# loop) and any ``/api/*`` target (renders raw JSON in the address bar and is
# indistinguishable from an attacker weaponising the redirect).
_NEXT_DENY_PREFIXES = ("/login", "/auth/", "/api/auth/")


def client_ip(request: Request) -> str:
    """First ``X-Forwarded-For`` hop, else the peer address."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def extract_bearer(request: Request) -> str:
    """``Authorization: Bearer <token>`` value (scheme case-insensitive), or ``""``."""
    parts = request.headers.get("authorization", "").split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def is_safe_next_path(path: str) -> bool:
    """True if ``path`` is a same-origin post-login target.

    Rejects non-relative and protocol-relative (``//evil``) values, the auth
    routes themselves, and every ``/api`` path.
    """
    if not path.startswith("/") or path.startswith("//"):
        return False
    if any(path == p or path.startswith(p) for p in _NEXT_DENY_PREFIXES):
        return False
    return not (path == "/api" or path.startswith("/api/"))


def access_token_max_age(session) -> int:
    """Cookie Max-Age for the access token: seconds to ``exp``, floored at 60."""
    return max(60, int(session.expires_at) - int(time.time()))


def unreachable_response(provider_name: str) -> JSONResponse:
    """503 for a transient IDP/backing-store outage (never a forced re-login)."""
    return JSONResponse(
        {"detail": f"Auth provider {provider_name!r} unreachable"},
        status_code=503,
    )
