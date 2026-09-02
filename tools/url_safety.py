"""URL safety checks — blocks requests to private/internal network addresses (SSRF).

``security.allow_private_urls: true`` (config.yaml) disables private-IP blocking
for environments whose DNS resolves public names to private/benchmark ranges.
Even then, cloud metadata hostnames/IPs are **always** blocked.

Limitations:
  - DNS rebinding (TOCTOU): Hermes-owned direct httpx paths should use
    ``create_ssrf_safe_client()`` / ``create_ssrf_safe_async_client()`` so the
    policy is re-applied at TCP connect and the socket dials the validated IP
    while preserving Host/SNI semantics.
  - Redirect bypass is mitigated by httpx response hooks re-validating each
    redirect target (see ``redirect_target_from_response``).
"""

import ipaddress
import logging
import os
import socket
import asyncio
import re
from typing import Any, Optional
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlparse, urlsplit, urlunsplit

from hermes_constants import get_hermes_home_override
from utils import is_truthy_value

logger = logging.getLogger(__name__)


# Proxy env vars: when set, the runtime should delegate DNS to the proxy.
_PROXY_ENV_VARS = (
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
)

_HTTP_SCHEMES = frozenset({"http", "https"})


def _proxy_is_configured() -> bool:
    return any(os.environ.get(v) for v in _PROXY_ENV_VARS)


def normalize_url_for_request(url: str) -> str:
    """Return an ASCII-safe HTTP URL for Hermes-owned URL tools (IRI -> URI).

    Browsers expect URIs but users/models often supply IRIs (``https://wttr.in/Köln``).
    Preserves URL syntax and existing percent escapes while IDNA-encoding the host
    and percent-encoding non-ASCII path/query/fragment text. URL tool inputs only —
    arbitrary shell commands must not be rewritten.
    """
    if not isinstance(url, str):
        return url

    raw = url.strip()
    if not raw:
        return raw

    # Repair model-emitted whitespace between scheme separator and authority
    # (``https:// docs.example``); that position is never meaningful in HTTP URLs.
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw

    if parsed.scheme.lower() not in _HTTP_SCHEMES:
        return raw

    netloc = parsed.netloc
    hostname = parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)

    path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    query = quote(parsed.query, safe="/%:@!$&'()*+,;=?")
    fragment = quote(parsed.fragment, safe="/%:@!$&'()*+,;=?")

    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


# Unambiguously credential-bearing query param names. Deliberately narrow: bare
# English words that double as page facets (``code``, ``key``, ``auth``,
# ``session``, ``sig``) are EXCLUDED so ordinary browsing is not blocked.
_SENSITIVE_QUERY_PARAM_NAMES = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "awsaccesskeyid",
    "client_secret",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwd",
    "secret",
    "session_id",
    "signature",
    "token",
    "x_amz_security_token",
    "x_amz_signature",
    "x-amz-security-token",
    "x-amz-signature",
})


def sensitive_query_param_name(url: str) -> Optional[str]:
    """Return the first credential-named query parameter in ``url`` (with a value), if any.

    Checked before handing URLs to third-party fetch/browser backends: prefix-based
    token redaction catches known vendor key shapes; this catches opaque magic links,
    OAuth codes, signed-URL signatures and custom ``?token=...`` values.
    """
    if not isinstance(url, str) or "?" not in url:
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in _HTTP_SCHEMES or not parsed.query:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES:
            return key
    return None


# Cloud metadata hostnames — always blocked regardless of DNS or config toggle.
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata.goog",
})

# Cloud metadata / credential endpoints (the #1 SSRF target) and the link-local
# range they live in — always blocked. IPv4-mapped IPv6 forms are listed because
# resolvers may return ``::ffff:x.x.x.x`` and ipaddress treats those as distinct.
_ALWAYS_BLOCKED_IPS = frozenset({
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/DO/Oracle metadata
    ipaddress.ip_address("169.254.170.2"),     # AWS ECS task metadata (task IAM creds)
    ipaddress.ip_address("169.254.169.253"),   # Azure IMDS wire server
    ipaddress.ip_address("fd00:ec2::254"),     # AWS metadata (IPv6)
    ipaddress.ip_address("100.100.100.200"),   # Alibaba Cloud metadata
    ipaddress.ip_address("::ffff:169.254.169.254"),
    ipaddress.ip_address("::ffff:169.254.170.2"),
    ipaddress.ip_address("::ffff:169.254.169.253"),
    ipaddress.ip_address("::ffff:100.100.100.200"),
})
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),    # Entire link-local range (no legit agent target)
    ipaddress.ip_network("::ffff:169.254.0.0/112"), # IPv4-mapped link-local range
)

# Exact HTTPS hostnames allowed to resolve to private/benchmark-space IPs
# (QQ media legitimately resolves to 198.18.0.0/15 behind local proxy infra).
_TRUSTED_PRIVATE_IP_HOSTS = frozenset({
    "multimedia.nt.qq.com.cn",
})

_MAX_SSRF_CONNECT_IPS = 8

# 100.64.0.0/10 (CGNAT, RFC 6598) is neither is_private nor is_global in
# ipaddress — must be blocked explicitly (Tailscale/WireGuard, cloud internal nets).
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Global toggle cache (process lifetime; see _global_allow_private_urls).
_allow_private_resolved = False
_cached_allow_private: bool = False


def _global_allow_private_urls() -> bool:
    """Return True when the user has opted out of private-IP blocking.

    Priority: ``HERMES_ALLOW_PRIVATE_URLS`` env, ``security.allow_private_urls``,
    legacy ``browser.allow_private_urls``. A multiplex gateway serves several
    independently configured profiles in one process, so profile-scoped turns
    (``get_hermes_home_override()`` set) bypass the process-global cache —
    otherwise the first profile's opt-out would disable blocking for every later one.
    ``read_raw_config()`` already provides path/mtime caching for that path.
    """
    global _allow_private_resolved, _cached_allow_private

    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()

    if _allow_private_resolved:
        return _cached_allow_private

    _allow_private_resolved = True
    _cached_allow_private = _resolve_allow_private_urls()
    return _cached_allow_private


def _resolve_allow_private_urls() -> bool:
    """Resolve the effective private-URL toggle from the active config scope."""
    env_val = os.getenv("HERMES_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env_val in {"true", "1", "yes"}:
        return True
    if env_val in {"false", "0", "no"}:
        return False  # explicit false does not fall through to config

    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        for section in ("security", "browser"):  # preferred, then legacy
            block = cfg.get(section, {})
            if isinstance(block, dict) and is_truthy_value(
                block.get("allow_private_urls"), default=False
            ):
                return True
    except Exception:
        pass  # config unavailable (tests, early import) — keep default

    return False


def _reset_allow_private_cache() -> None:
    """Reset the cached toggle — only for tests."""
    global _allow_private_resolved, _cached_allow_private
    _allow_private_resolved = False
    _cached_allow_private = False


def _normalize_hostname(host: Optional[str]) -> str:
    return (host or "").strip().lower().rstrip(".")


def _is_ip_literal(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def _sockaddr_ip(sockaddr: Any) -> str:
    """Return the IP string from a getaddrinfo sockaddr, minus any IPv6 scope ID."""
    ip_str = sockaddr[0]
    return ip_str.split("%")[0] if "%" in ip_str else ip_str


def _is_always_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip in _ALWAYS_BLOCKED_IPS or any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS)


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP should be blocked for SSRF protection."""
    # IPv4-mapped IPv6 (``::ffff:x.x.x.x``) is classified by its embedded IPv4.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified or ip in _CGNAT_NETWORK)


def is_always_blocked_url(url: str) -> bool:
    """Return True when the URL targets the always-blocked floor (cloud metadata).

    Narrower than ``is_safe_url``: only the sentinel hostnames/IPs, regardless of
    backend, routing, or ``allow_private_urls``. For callers that deliberately bypass
    the full check (e.g. hybrid cloud browser routing private URLs to a local sidecar)
    but must still enforce the non-negotiable floor. Returns False for ordinary
    private/loopback URLs, DNS failures on non-sentinel hosts, and parse errors
    (the caller's ordinary fail-closed path handles those).
    """
    try:
        hostname = _normalize_hostname(urlparse(url).hostname)
        if not hostname:
            return False

        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning(
                "Blocked request to internal hostname (always-blocked floor): %s",
                hostname,
            )
            return True

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip = None

        if ip is not None:
            if _is_always_blocked_ip(ip):
                logger.warning(
                    "Blocked request to cloud metadata address "
                    "(always-blocked floor): %s",
                    hostname,
                )
                return True
            return False

        try:
            addr_info = socket.getaddrinfo(
                hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except socket.gaierror:
            return False

        for _family, _, _, _, sockaddr in addr_info:
            ip_str = _sockaddr_ip(sockaddr)
            try:
                resolved = ipaddress.ip_address(ip_str)
            except ValueError:
                logger.warning("Unparseable IP address %r for hostname %s — skipping address", sockaddr[0], hostname)
                continue
            if _is_always_blocked_ip(resolved):
                logger.warning(
                    "Blocked request to cloud metadata address "
                    "(always-blocked floor): %s -> %s",
                    hostname,
                    ip_str,
                )
                return True

        return False

    except Exception as exc:
        logger.debug("is_always_blocked_url error for %s: %s", url, exc)
        return False


def _allows_private_ip_resolution(hostname: str, scheme: str) -> bool:
    """Return True when a trusted HTTPS hostname may bypass IP-class blocking."""
    return scheme == "https" and hostname in _TRUSTED_PRIVATE_IP_HOSTS


def is_safe_url(url: str) -> bool:
    """Return True if the URL target is not a private/internal address.

    Resolves the hostname and checks every answer. Fails closed on DNS errors and
    unexpected exceptions. ``allow_private_urls`` skips private-IP blocking, but
    cloud metadata endpoints remain blocked regardless.
    """
    try:
        parsed = urlparse(url)
        hostname = _normalize_hostname(parsed.hostname)
        scheme = (parsed.scheme or "").strip().lower()
        if scheme not in _HTTP_SCHEMES:
            logger.warning("Blocked request — unsupported URL scheme: %s", scheme or "<empty>")
            return False
        if not hostname:
            return False

        # Metadata hostnames are blocked BEFORE consulting the toggle.
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

        allow_all_private = _global_allow_private_urls()
        allow_private_ip = _allows_private_ip_resolution(hostname, scheme)

        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            # Sandbox/proxy environments may block direct DNS; when a proxy is
            # configured, delegate resolution to it (metadata hostnames were already
            # rejected above). Literal IPs need no DNS, so a failure on one is not a
            # proxy symptom — keep them fail-closed.
            if not _is_ip_literal(hostname) and _proxy_is_configured():
                logger.debug(
                    "DNS resolution failed for %s — proxy configured, "
                    "allowing through for proxy-side resolution",
                    hostname,
                )
                return True
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False

        for family, _, _, _, sockaddr in addr_info:
            ip_str = _sockaddr_ip(sockaddr)
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                logger.warning("Blocked request — unparseable IP address %r for hostname %s", sockaddr[0], hostname)
                return False

            if _is_always_blocked_ip(ip):
                logger.warning(
                    "Blocked request to cloud metadata address: %s -> %s",
                    hostname, ip_str,
                )
                return False

            if not allow_all_private and not allow_private_ip and _is_blocked_ip(ip):
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname, ip_str,
                )
                return False

        if allow_all_private:
            logger.debug(
                "Allowing private/internal resolution (security.allow_private_urls=true): %s",
                hostname,
            )
        elif allow_private_ip:
            logger.debug(
                "Allowing trusted hostname despite private/internal resolution: %s",
                hostname,
            )

        return True

    except Exception as exc:
        # Fail closed: parsing edge cases must not become SSRF bypass vectors.
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False


async def async_is_safe_url(url: str) -> bool:
    """Same rules as :func:`is_safe_url`, with the blocking DNS work off the event loop."""
    return await asyncio.to_thread(is_safe_url, url)


class SSRFConnectionBlocked(ValueError):
    """Raised when connect-time DNS resolution violates the URL safety policy."""


def _safe_connect_scheme(host: str, port: int, schemes_by_origin: dict[tuple[str, int], str]) -> str:
    return schemes_by_origin.get((host, port)) or ("https" if port == 443 else "http")


def _resolved_http_connect_ips(host: str, port: int, scheme: str) -> list[str]:
    """Resolve and validate *host* at TCP-connect time; return dialable IP strings.

    Closes the DNS-rebinding gap between pre-flight validation and connection
    setup for direct httpx clients.
    """
    hostname = _normalize_hostname(host)
    if not hostname:
        raise SSRFConnectionBlocked("Blocked request with empty hostname")

    if hostname in _BLOCKED_HOSTNAMES:
        raise SSRFConnectionBlocked(f"Blocked request to internal hostname: {hostname}")

    allow_all_private = _global_allow_private_urls()
    allow_private_ip = _allows_private_ip_resolution(hostname, scheme)

    try:
        addr_info = socket.getaddrinfo(
            hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise SSRFConnectionBlocked(
            f"Blocked request - DNS resolution failed for: {hostname}"
        ) from exc

    safe_ips: list[str] = []
    seen: set[str] = set()
    for _family, _, _, _, sockaddr in addr_info:
        ip_str = _sockaddr_ip(sockaddr)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise SSRFConnectionBlocked(
                f"Blocked request - unparseable IP address {sockaddr[0]!r} for hostname {hostname}"
            ) from exc

        if _is_always_blocked_ip(ip):
            raise SSRFConnectionBlocked(
                f"Blocked request to cloud metadata address during connect: {hostname} -> {ip_str}"
            )

        if not allow_all_private and not allow_private_ip and _is_blocked_ip(ip):
            raise SSRFConnectionBlocked(
                f"Blocked request to private/internal address during connect: {hostname} -> {ip_str}"
            )

        if ip_str not in seen and len(safe_ips) < _MAX_SSRF_CONNECT_IPS:
            safe_ips.append(ip_str)
            seen.add(ip_str)

    if not safe_ips:
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no results for: {hostname}")
    return safe_ips


class _SSRFGuardedAsyncNetworkBackend:
    """httpcore backend that re-resolves + validates at connect time and dials a vetted IP.

    Host/SNI stay on the original hostname; Unix sockets are refused outright.
    """

    def __init__(self, schemes_by_origin_var: Any):
        from httpcore._backends.auto import AutoBackend

        self._backend = AutoBackend()
        self._schemes_by_origin_var = schemes_by_origin_var

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        import httpcore

        scheme = _safe_connect_scheme(host, port, self._schemes_by_origin_var.get({}))
        ips = await asyncio.to_thread(_resolved_http_connect_ips, host, port, scheme)

        last_exc: Exception | None = None
        for ip in ips:
            try:
                return await self._backend.connect_tcp(
                    ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no usable IPs for: {host}")

    async def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options: Any = None) -> Any:
        raise SSRFConnectionBlocked("Blocked Unix socket connection in SSRF-safe transport")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _SSRFGuardedNetworkBackend:
    """httpcore backend that re-resolves + validates at connect time and dials a vetted IP.

    Host/SNI stay on the original hostname; Unix sockets are refused outright.
    """

    def __init__(self, schemes_by_origin_var: Any):
        from httpcore._backends.sync import SyncBackend

        self._backend = SyncBackend()
        self._schemes_by_origin_var = schemes_by_origin_var

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        import httpcore

        scheme = _safe_connect_scheme(host, port, self._schemes_by_origin_var.get({}))
        ips = _resolved_http_connect_ips(host, port, scheme)

        last_exc: Exception | None = None
        for ip in ips:
            try:
                return self._backend.connect_tcp(
                    ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no usable IPs for: {host}")

    def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options: Any = None) -> Any:
        raise SSRFConnectionBlocked("Blocked Unix socket connection in SSRF-safe transport")

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


def _origin_scheme_context(request: Any) -> dict[tuple[str, int], str]:
    host = request.url.host
    port = request.url.port
    scheme = request.url.scheme
    if not host or port is None or scheme not in _HTTP_SCHEMES:
        return {}
    return {(host, port): scheme}


def _install_ssrf_guard_on_transport(transport: Any, schemes_by_origin_var: Any, *, is_async: bool = False) -> None:
    """Swap the transport's pool network backend for the SSRF-guarded one (idempotent).

    Only the client's direct transport is guarded; proxy mounts are left alone so
    final-target resolution is delegated to the (trusted) proxy egress.
    """
    state = getattr(transport, "__dict__", {}) if transport is not None else {}
    if transport is None or state.get("_hermes_ssrf_guarded", False):
        return

    label = "async httpx transport" if is_async else "httpx transport"
    pool = state.get("_pool")
    if pool is None or not hasattr(pool, "_network_backend"):
        raise SSRFConnectionBlocked(f"Unsupported {label} cannot be made SSRF-safe")
    backend_cls = _SSRFGuardedAsyncNetworkBackend if is_async else _SSRFGuardedNetworkBackend
    pool._network_backend = backend_cls(schemes_by_origin_var)

    method_name = "handle_async_request" if is_async else "handle_request"
    handle = getattr(transport, method_name, None)
    if handle is None:
        raise SSRFConnectionBlocked(f"Unsupported {label} cannot be made SSRF-safe")

    async def guarded_async(request: Any) -> Any:
        token = schemes_by_origin_var.set(_origin_scheme_context(request))
        try:
            return await handle(request)
        finally:
            schemes_by_origin_var.reset(token)

    def guarded_sync(request: Any) -> Any:
        token = schemes_by_origin_var.set(_origin_scheme_context(request))
        try:
            return handle(request)
        finally:
            schemes_by_origin_var.reset(token)

    setattr(transport, method_name, guarded_async if is_async else guarded_sync)
    transport._hermes_ssrf_guarded = True


def _install_ssrf_guard_on_client(client: Any, *, is_async: bool = False) -> None:
    import contextvars

    var_name = "hermes_ssrf_async_origin_schemes" if is_async else "hermes_ssrf_origin_schemes"
    _install_ssrf_guard_on_transport(
        getattr(client, "__dict__", {}).get("_transport"),
        contextvars.ContextVar(var_name),
        is_async=is_async,
    )


def create_ssrf_safe_async_client(**kwargs: Any) -> Any:
    """Create an ``httpx.AsyncClient`` with connect-time SSRF validation.

    Direct HTTP(S) connections are resolved, validated, and dialed by IP at
    TCP-connect time while the request hostname is preserved for Host, SNI, and
    certificate verification. Proxied requests delegate resolution to the proxy.
    """
    import httpx

    client = httpx.AsyncClient(**kwargs)
    _install_ssrf_guard_on_client(client, is_async=True)
    return client


def create_ssrf_safe_client(**kwargs: Any) -> Any:
    """Create an ``httpx.Client`` with connect-time SSRF validation."""
    import httpx

    client = httpx.Client(**kwargs)
    _install_ssrf_guard_on_client(client)
    return client


def redirect_target_from_response(response: Any) -> Optional[str]:
    """Return the redirect target visible from inside an httpx response hook.

    ``response.next_request`` is frequently ``None`` inside hooks (populated later
    by the redirect follower), which would make an SSRF redirect guard silently
    never fire. Resolve from the ``Location`` header first (relative via
    ``urljoin``), falling back to ``next_request``.
    """
    if not getattr(response, "is_redirect", False):
        return None

    headers = getattr(response, "headers", {}) or {}
    location = headers.get("location")
    if location:
        return urljoin(str(getattr(response, "url", "")), str(location))

    next_request = getattr(response, "next_request", None)
    if next_request:
        return str(next_request.url)

    return None
