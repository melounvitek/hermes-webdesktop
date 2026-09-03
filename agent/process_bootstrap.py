"""Process-level bootstrap helpers for ``run_agent``.

Lazy OpenAI SDK import (``_OpenAIProxy`` keeps ``isinstance`` and
``patch("run_agent.OpenAI")`` working), crash-resistant stdio (``_SafeWriter``),
env-only HTTP proxy resolution, and Codex dual-stack (Happy Eyeballs)
connection racing. ``run_agent`` re-exports every name.
"""

from __future__ import annotations

import errno
import os
import selectors
import socket
import sys
import time
import urllib.request
from typing import Any, Optional

from utils import base_url_hostname, normalize_proxy_url


_OPENAI_CLS_CACHE = None
_HAPPY_EYEBALLS_DELAY_SECONDS = 0.25


def _interleave_addrinfos(addrinfos: list[tuple]) -> list[tuple]:
    """Round-robin the resolved address families (deduped), preserving resolver order within each."""
    queues: dict[int, list[tuple]] = {}
    seen: set[tuple] = set()
    for addrinfo in addrinfos:
        family, socktype, proto, _canonname, sockaddr = addrinfo
        marker = (family, socktype, proto, sockaddr)
        if marker not in seen:
            seen.add(marker)
            queues.setdefault(family, []).append(addrinfo)
    interleaved: list[tuple] = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                interleaved.append(queue.pop(0))
    return interleaved


def _happy_eyeballs_create_connection(
    address: tuple[str, int],
    timeout: Optional[float],
    source_address: Optional[tuple[str, int]] = None,
    socket_options=(),
):
    """RFC 8305-style connect: staggered non-blocking attempts across families.

    ``socket.create_connection`` tries addresses serially, so broken-but-
    advertised IPv6 can burn the whole timeout per AAAA record before IPv4.
    """
    host, port = address
    addrinfos = _interleave_addrinfos(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
    if not addrinfos:
        raise OSError(f"getaddrinfo returned no addresses for {host}")

    selector = selectors.DefaultSelector()
    active: set[socket.socket] = set()
    winner = None
    last_error: Optional[OSError] = None
    deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
    next_launch = time.monotonic()
    pending = list(addrinfos)
    in_progress = {
        0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR,
        getattr(errno, "WSAEWOULDBLOCK", 10035),
    }

    def start_attempt(addrinfo):
        family, socktype, proto, _canonname, sockaddr = addrinfo
        candidate = socket.socket(family, socktype, proto)
        try:
            if source_address is not None:
                local_infos = socket.getaddrinfo(source_address[0], source_address[1], family=family, type=socktype)
                if not local_infos:
                    raise OSError(f"getaddrinfo returned no local {family} address for {source_address[0]}")
                candidate.bind(local_infos[0][4])
            candidate.setblocking(False)
            result = candidate.connect_ex(sockaddr)
            if result in (0, errno.EISCONN):
                return candidate
            if result not in in_progress:
                raise OSError(result, os.strerror(result))
            selector.register(candidate, selectors.EVENT_WRITE)
            active.add(candidate)
            return None
        except Exception:
            candidate.close()
            raise

    try:
        while pending or active:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise socket.timeout("timed out")

            if pending and now >= next_launch:
                try:
                    winner = start_attempt(pending.pop(0))
                except OSError as exc:
                    last_error = exc
                    if not active:
                        next_launch = now
                    continue
                if winner is not None:
                    break
                next_launch = now + _HAPPY_EYEBALLS_DELAY_SECONDS

            wait_timeout = None if deadline is None else max(0.0, deadline - now)
            if pending:
                until_launch = max(0.0, next_launch - now)
                wait_timeout = until_launch if wait_timeout is None else min(wait_timeout, until_launch)

            for key, _mask in selector.select(wait_timeout):
                candidate = key.fileobj
                error_code = candidate.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                selector.unregister(candidate)
                active.discard(candidate)
                if error_code == 0:
                    winner = candidate
                    break
                candidate.close()
                last_error = OSError(error_code, os.strerror(error_code))
            if winner is not None:
                break
            if not active and pending:
                next_launch = time.monotonic()

        if winner is None:
            raise last_error if last_error is not None else OSError(f"Could not connect to {host}:{port}")

        try:
            selector.unregister(winner)
        except Exception:
            pass
        active.discard(winner)
        winner.settimeout(timeout)
        for option in socket_options or ():
            winner.setsockopt(*option)
        winner.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return winner
    finally:
        for candidate in active:
            try:
                selector.unregister(candidate)
            except Exception:
                pass
            candidate.close()
        selector.close()


class _HappyEyeballsSyncBackend:
    """httpcore sync backend with concurrent IPv6/IPv4 connection fallback."""

    def __init__(self):
        self._fallback = None

    def _default_backend(self):
        if self._fallback is None:
            from httpcore import SyncBackend

            self._fallback = SyncBackend()
        return self._fallback

    def connect_tcp(
        self, host: str, port: int, timeout: Optional[float] = None,
        local_address: Optional[str] = None, socket_options=None,
    ):
        from httpcore import ConnectError, ConnectTimeout
        from httpcore._backends.sync import SyncStream

        source_address = None if local_address is None else (local_address, 0)
        try:
            sock = _happy_eyeballs_create_connection(
                (host, port), timeout, source_address=source_address, socket_options=socket_options or ()
            )
        except socket.timeout as exc:
            raise ConnectTimeout(str(exc)) from exc
        except OSError as exc:
            raise ConnectError(str(exc)) from exc
        return SyncStream(sock)

    def connect_unix_socket(self, *args, **kwargs):
        return self._default_backend().connect_unix_socket(*args, **kwargs)

    def sleep(self, seconds: float) -> None:
        self._default_backend().sleep(seconds)


def _uses_codex_cloud_transport(base_url: str) -> bool:
    return base_url_hostname(base_url).lower() == "chatgpt.com" and "/backend-api/codex" in str(base_url).lower()


def _enable_happy_eyeballs(transport, skip_pool_types: tuple = ()) -> None:
    """Install the racing backend on one httpx transport.

    Reaches into private ``transport._pool._network_backend`` (httpcore pinned
    1.0.x); hasattr-guarded so an incompatible httpcore degrades to the default
    serial backend. Pools of ``skip_pool_types`` (proxies) are left alone.
    """
    pool = getattr(transport, "_pool", None)
    if pool is None or not hasattr(pool, "_network_backend"):
        return
    if skip_pool_types and isinstance(pool, skip_pool_types):
        return
    pool._network_backend = _HappyEyeballsSyncBackend()


def enable_happy_eyeballs_on_client(client) -> None:
    """Install the racing backend on every direct transport of a ready-built httpx.Client.

    For callers that build clients inline (Codex OAuth/device-login). Proxy-backed
    pools are skipped (TCP connect goes to the proxy host); async clients need
    nothing (anyio already races per RFC 8305). Best-effort.
    """
    try:
        import httpcore

        proxy_pool_types = tuple(
            t for t in (getattr(httpcore, "HTTPProxy", None), getattr(httpcore, "SOCKSProxy", None)) if t is not None
        )
    except Exception:
        return
    transports = [getattr(client, "_transport", None), *(getattr(client, "_mounts", None) or {}).values()]
    for transport in transports:
        _enable_happy_eyeballs(transport, proxy_pool_types)


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Module-level proxy that looks like ``openai.OpenAI`` but imports lazily."""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


class _SafeWriter:
    """Transparent stdio wrapper swallowing OSError/ValueError from broken pipes.

    Headless runs (systemd, Docker) lose the stdout pipe → ``OSError: [Errno 5]``;
    subagent threads can see the shared handle close → ``ValueError``. Either
    would otherwise crash the agent (often via double-fault in an except handler).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _get_proxy_from_env() -> Optional[str]:
    """First configured proxy URL from HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (any case), or None."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Env-configured proxy unless NO_PROXY excludes this base URL."""
    proxy = _get_proxy_from_env()
    host = base_url_hostname(base_url) if proxy and base_url else ""
    if not host:
        return proxy
    try:
        if urllib.request.proxy_bypass_environment(host):
            return None
    except Exception:
        pass
    return proxy


def build_keepalive_http_client(base_url: str = "", *, async_mode: bool = False, verify: Any = True) -> Optional[Any]:
    """httpx client for OpenAI SDK calls with env-only proxy policy (None on failure).

    Explicit no-proxy mounts disable httpx's ``trust_env`` path so macOS system
    proxies (which omit the ExceptionsList) are never applied. ``keepalive_expiry``
    reaps idle connections before reverse proxies' 30-60 s timeouts (a custom
    socket_options transport broke streaming and stripped TCP_NODELAY). ``verify``
    goes on the client AND the mounts, since a mounted transport owns its SSL context.
    """
    try:
        import httpx

        proxy = _get_proxy_for_base_url(base_url)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=20.0)
        timeout = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=10.0)  # read=None for SSE streaming
        transport_cls = httpx.AsyncHTTPTransport if async_mode else httpx.HTTPTransport
        client_cls = httpx.AsyncClient if async_mode else httpx.Client
        mounts = None
        if proxy is None:
            mounts = {"http://": transport_cls(verify=verify), "https://": transport_cls(verify=verify)}
            # Async transports race natively (anyio happy_eyeballs_delay=0.25).
            if not async_mode and _uses_codex_cloud_transport(base_url):
                for transport in mounts.values():
                    _enable_happy_eyeballs(transport)
        return client_cls(limits=limits, timeout=timeout, proxy=proxy, mounts=mounts, verify=verify)
    except Exception:
        return None


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


# Drop-in for ``openai.OpenAI`` (also re-exported via ``run_agent``).
OpenAI = _OpenAIProxy()


__all__ = [
    "OpenAI",
    "_OpenAIProxy",
    "_load_openai_cls",
    "_SafeWriter",
    "_install_safe_stdio",
    "_get_proxy_from_env",
    "_get_proxy_for_base_url",
    "build_keepalive_http_client",
    "enable_happy_eyeballs_on_client",
]
