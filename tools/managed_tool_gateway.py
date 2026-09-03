"""Generic managed-tool gateway helpers for Nous-hosted vendor passthroughs."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, Optional

from hermes_constants import get_hermes_home
from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_GATEWAY_DOMAIN = "nousresearch.com"
_DEFAULT_TOOL_GATEWAY_SCHEME = "https"
_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120


@dataclass(frozen=True)
class ManagedToolGatewayConfig:
    vendor: str
    gateway_origin: str
    nous_user_token: str
    managed_mode: bool


def _clean(value: object) -> Optional[str]:
    """*value* stripped when it is a non-blank string, else None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def auth_json_path():
    """Return the Hermes auth store path, respecting HERMES_HOME overrides."""
    return get_hermes_home() / "auth.json"


def _read_nous_provider_state() -> Optional[dict]:
    try:
        path = auth_json_path()
        if not path.is_file():
            return None
        providers = json.loads(path.read_text(encoding="utf-8-sig")).get("providers", {})
        nous_provider = providers.get("nous", {}) if isinstance(providers, dict) else None
        if isinstance(nous_provider, dict):
            return nous_provider
    except Exception:
        pass
    return None


def _parse_timestamp(value: object) -> Optional[datetime]:
    normalized = _clean(value)
    if normalized is None:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _access_token_is_expiring(expires_at: object, skew_seconds: int) -> bool:
    expires = _parse_timestamp(expires_at)
    if expires is None:
        return True
    return (expires - datetime.now(timezone.utc)).total_seconds() <= max(0, int(skew_seconds))


def _read_user_token_override() -> Optional[str]:
    """Read the TOOL_GATEWAY_USER_TOKEN override through the secret scope. Scope verdict is
    authoritative when installed (a scoped miss must NOT borrow the process env under
    multiplex); ``os.environ`` only when unscoped."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            explicit = get_secret("TOOL_GATEWAY_USER_TOKEN")
        except UnscopedSecretError:
            explicit = os.getenv("TOOL_GATEWAY_USER_TOKEN")
    except Exception:
        explicit = os.getenv("TOOL_GATEWAY_USER_TOKEN")
    return _clean(explicit)


def peek_nous_access_token() -> Optional[str]:
    """Cheap token probe: env override or cached auth-store token, no expiry check and no
    network — availability scans must stay off the synchronous OAuth refresh path (that lives
    in :func:`read_nous_access_token`)."""
    return _read_user_token_override() or _clean((_read_nous_provider_state() or {}).get("access_token"))


def read_nous_access_token() -> Optional[str]:
    """Read a Nous Subscriber OAuth access token from auth store or env override."""
    explicit = _read_user_token_override()
    if explicit:
        return explicit
    nous_provider = _read_nous_provider_state() or {}
    cached_token = peek_nous_access_token()
    if cached_token and not _access_token_is_expiring(nous_provider.get("expires_at"), _NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
        return cached_token
    try:
        from hermes_cli.auth import resolve_nous_access_token

        refreshed_token = _clean(resolve_nous_access_token(refresh_skew_seconds=_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS))
        if refreshed_token:
            return refreshed_token
    except Exception as exc:
        logger.debug("Nous access token refresh failed: %s", exc)
    return cached_token


def get_tool_gateway_scheme() -> str:
    """Return configured shared gateway URL scheme."""
    scheme = os.getenv("TOOL_GATEWAY_SCHEME", "").strip().lower()
    if not scheme:
        return _DEFAULT_TOOL_GATEWAY_SCHEME
    if scheme in {"http", "https"}:
        return scheme
    raise ValueError("TOOL_GATEWAY_SCHEME must be 'http' or 'https'")


def build_vendor_gateway_url(vendor: str) -> str:
    """Return the gateway origin for a specific vendor."""
    explicit_vendor_url = os.getenv(f"{vendor.upper().replace('-', '_')}_GATEWAY_URL", "").strip().rstrip("/")
    if explicit_vendor_url:
        return explicit_vendor_url
    shared_domain = os.getenv("TOOL_GATEWAY_DOMAIN", "").strip().strip("/") or _DEFAULT_TOOL_GATEWAY_DOMAIN
    return f"{get_tool_gateway_scheme()}://{vendor}-gateway.{shared_domain}"


def resolve_managed_tool_gateway(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> Optional[ManagedToolGatewayConfig]:
    """Resolve shared managed-tool gateway config for a vendor."""
    if not managed_nous_tools_enabled():
        return None
    gateway_origin = (gateway_builder or build_vendor_gateway_url)(vendor)
    nous_user_token = (token_reader or read_nous_access_token)()
    if not gateway_origin or not nous_user_token:
        return None
    return ManagedToolGatewayConfig(vendor=vendor, gateway_origin=gateway_origin, nous_user_token=nous_user_token, managed_mode=True)


def is_managed_tool_gateway_ready(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    """True when a gateway URL and a likely-usable Nous token are present. Defaults to
    :func:`peek_nous_access_token` (no OAuth refresh); callers about to make a real request
    use :func:`resolve_managed_tool_gateway` instead."""
    return resolve_managed_tool_gateway(vendor, gateway_builder=gateway_builder, token_reader=token_reader or peek_nous_access_token) is not None
