"""OpenAI/Anthropic wire-client lifecycle + credential refresh for ``AIAgent``.

``ClientLifecycleMixin`` owns the shared primary client, the single-slot per-request client caches (owner-
thread close vs stranger-thread abort, #29507), credential refresh/rotation, and route-derived default
headers. Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import threading
from typing import Any, Optional

from agent.lazy_forward import forward as _forward, forward_static as _forward_static, lazy_attr as _lazy_attr
from hermes_cli.timeouts import get_provider_request_timeout
from utils import base_url_host_matches, env_float

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


# Qwen Portal headers — mimics QwenCode CLI for portal.qwen.ai; shared by __init__ and
# _apply_client_headers_for_base_url.
_QWEN_CODE_VERSION = "0.14.1"


def _routermint_headers() -> dict:
    """Return the User-Agent RouterMint needs to avoid Cloudflare 1010 blocks."""
    from hermes_cli import __version__ as _HERMES_VERSION

    return {
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    }


def _qwen_portal_headers() -> dict:
    """Return default HTTP headers required by Qwen Portal API."""
    import platform as _plat

    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


# Route-specific default headers; first host match wins (order preserved from the original chain).
# Builders resolve their module lazily so run_agent keeps its import-time cost and avoids cycles.
_ROUTE_DEFAULT_HEADERS = (
    ("openrouter.ai", lambda self, url: _lazy_attr("agent.auxiliary_client", "build_or_headers")()),
    ("ai-gateway.vercel.sh", lambda self, url: dict(_lazy_attr("agent.auxiliary_client", "_AI_GATEWAY_HEADERS"))),
    ("integrate.api.nvidia.com", lambda self, url: _lazy_attr("agent.auxiliary_client", "build_nvidia_nim_headers")(url)),
    ("api.routermint.com", lambda self, url: _routermint_headers()),
    ("githubcopilot.com", lambda self, url: _lazy_attr("hermes_cli.models", "copilot_default_headers")()),
    ("api.kimi.com", lambda self, url: dict(_lazy_attr("agent.auxiliary_client", "_AI_GATEWAY_HEADERS"))),
    ("portal.qwen.ai", lambda self, url: _qwen_portal_headers()),
    ("chatgpt.com", lambda self, url: _lazy_attr("agent.codex_headers", "codex_cloudflare_headers")(
        self._client_kwargs.get("api_key", ""), base_url=url,
    )),
    # Covers provider=xai and provider=xai-oauth (api.x.ai).
    ("x.ai", lambda self, url: _lazy_attr("tools.xai_http", "hermes_xai_default_headers")()),
)


class ClientLifecycleMixin:
    """Wire-client construction, caching, teardown and credential refresh (see module docstring)."""

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """Check if an OpenAI client is closed.

        ``is_closed`` is a bool property on httpx.Client but a method on openai.OpenAI; a bare getattr
        returned
        the always-truthy bound method and recreated the client on every call.
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # Handle method (openai SDK) vs property (httpx)
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    @staticmethod
    def _build_keepalive_http_client(base_url: str = "", *, verify: Any = True) -> Any:
        """Build the shared OpenAI httpx client used by main and aux paths."""
        from agent.process_bootstrap import build_keepalive_http_client

        return build_keepalive_http_client(base_url, verify=verify)

    _create_openai_client = _forward("agent.agent_runtime_helpers", "create_openai_client")

    _force_close_tcp_sockets = _forward_static("agent.agent_runtime_helpers", "force_close_tcp_sockets")

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # Force-close TCP sockets first to prevent CLOSE-WAIT accumulation,
        # then do the graceful SDK-level close.
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _retire_shared_openai_client(self, client: Any, *, reason: str) -> None:
        """Ownership-safe retirement of a replaced shared OpenAI client.

        ``close()`` releases raw FDs from the calling thread; the shared client has no owning thread and other
        threads may still hold its fd in an SSL BIO. A recycled fd then gets a TLS record written into an
        unrelated
        file (the SQLite-header corruption family). So: ``shutdown()`` the sockets (FD-safe from any thread)
        and
        let GC release the FDs once every borrower has unwound.
        """
        if client is None:
            return
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            logger.info(
                "Shared OpenAI client retired (%s, tcp_shutdown=%d, "
                "fd_release=deferred_to_gc) %s",
                reason,
                shutdown_count,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "Shared OpenAI client retire failed (%s) %s error=%s",
                reason,
                self._client_log_context(),
                exc,
            )

    def _drain_transports_after_abandonment(self, *, reason: str) -> int:
        """FD-safe transport drain for an abandoned (timed-out) worker; returns sockets shut down.

        The worker may be blocked in an OpenSSL read; hard-closing from the timeout thread releases FDs under
        a live
        BIO (native corruption / SIGSEGV). Only ``shutdown()`` so the read settles with EOF and the worker
        closes itself.
        """
        drained = 0
        # Shared primary client (codex-direct / MoA stream on it directly).
        try:
            client = getattr(self, "client", None)
            if client is not None:
                drained += self._force_close_tcp_sockets(client)
        except Exception:
            logger.debug("Abandoned-worker drain: shared client sweep failed",
                         exc_info=True)
        # Cached per-request wire clients: abort (shutdown + poison the reuse
        # slot) so the unwinding worker discards them instead of re-caching.
        try:
            with self._openai_client_lock():
                cache = getattr(self, "_request_client_cache", None)
                cached = cache["client"] if cache else None
            if cached is not None:
                self._abort_request_openai_client(cached, reason=reason)
        except Exception:
            logger.debug("Abandoned-worker drain: request client abort failed",
                         exc_info=True)
        try:
            with self._openai_client_lock():
                cache = getattr(self, "_request_anthropic_client_cache", None)
                cached = cache["client"] if cache else None
            if cached is not None:
                self._abort_request_anthropic_client(cached, reason=reason)
        except Exception:
            logger.debug("Abandoned-worker drain: anthropic client abort failed",
                         exc_info=True)
        # Codex app-server session watches a private interrupt event.
        try:
            codex_session = getattr(self, "_codex_session", None)
            request_interrupt = getattr(codex_session, "request_interrupt", None)
            if callable(request_interrupt):
                request_interrupt()
        except Exception:
            logger.debug("Abandoned-worker drain: codex interrupt failed",
                         exc_info=True)
        # Inline (cron-style) request abort hook, when registered.
        try:
            abort_active = getattr(self, "_active_request_abort", None)
            if callable(abort_active):
                abort_active(reason)
        except Exception:
            logger.debug("Abandoned-worker drain: active request abort failed",
                         exc_info=True)
        logger.info(
            "Abandoned-worker transports drained (%s, tcp_shutdown=%d, "
            "fd_release=deferred_to_worker) %s",
            reason,
            drained,
            self._client_log_context(),
        )
        return drained

    def _build_primary_client_for_active_provider(self, *, reason: str) -> Any:
        """Build the shared client shape required by the active provider.

        MoA is a virtual provider whose ``client`` is an in-process facade, not an SDK client; generic rebuild
        paths (rotation, timeout, dead-connection cleanup) must preserve that.
        """
        if (getattr(self, "provider", "") or "").strip().lower() == "moa":
            from agent.moa_loop import build_moa_facade

            return build_moa_facade(self, self.model)
        return self._create_openai_client(
            self._client_kwargs,
            reason=reason,
            shared=True,
        )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._build_primary_client_for_active_provider(
                    reason=reason,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared primary client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        # Never hard-close the replaced shared client here (#70773): the caller may not own the thread
        # still unwinding on the old pool. Retire: sockets shut down, FD release deferred to GC.
        self._retire_shared_openai_client(old_client, reason=f"replace:{reason}")
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client
            old_client = client
            try:
                new_client = self._create_openai_client(
                    self._client_kwargs, reason=reason, shared=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed to recreate closed OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                raise RuntimeError("Failed to recreate closed OpenAI client") from exc
            self.client = new_client

        logger.warning(
            "Detected closed shared OpenAI client; recreated before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return new_client

    _cleanup_dead_connections = _forward("agent.agent_runtime_helpers", "cleanup_dead_connections")

    @staticmethod
    def _api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
        """Return True when the outbound request still contains native image parts."""
        if not isinstance(api_kwargs, dict):
            return False
        candidates = []
        messages = api_kwargs.get("messages")
        if isinstance(messages, list):
            candidates.extend(messages)
        # Responses API payloads use `input`; after conversion, image parts can
        # still be present there instead of in `messages`.
        response_input = api_kwargs.get("input")
        if isinstance(response_input, list):
            candidates.extend(response_input)

        def _contains_image(value: Any) -> bool:
            if isinstance(value, dict):
                ptype = value.get("type")
                if ptype in {"image_url", "input_image"}:
                    return True
                return any(_contains_image(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_image(v) for v in value)
            return False

        return any(_contains_image(item) for item in candidates)

    def _copilot_headers_for_request(self, *, is_vision: bool) -> dict:
        from hermes_cli.copilot_auth import copilot_request_headers

        return copilot_request_headers(is_agent_turn=True, is_vision=is_vision)

    # Close reasons reported by a request worker's own finally after a response — the only closes
    # that come from the FD-owning thread AND attest a healthy pool. Poisoning (#29507) still wins.
    _REQUEST_CLIENT_REUSE_REASONS = frozenset({
        "request_complete",
        "stream_request_complete",
    })

    def _request_client_cache_ref(self) -> dict:
        # Lazy init — tests build agents via AIAgent.__new__ without __init__.
        cache = getattr(self, "_request_client_cache", None)
        if cache is None:
            cache = {"client": None, "kwargs": None, "poisoned": False, "in_use": False}
            self._request_client_cache = cache
        return cache

    def _create_request_openai_client(self, *, reason: str, api_kwargs: Optional[dict] = None) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if self.provider == "moa":
            return primary_client
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        # Per-request clients must not run the SDK retry loop: the outer loop owns retries/rotation/
        # fallback, and SDK retries stretch a hung request ~3x past our stale detector.
        request_kwargs["max_retries"] = 0
        if (
            base_url_host_matches(str(request_kwargs.get("base_url", "")), "githubcopilot.com")
            and self._api_kwargs_have_image_parts(api_kwargs or {})
        ):
            request_kwargs["default_headers"] = self._copilot_headers_for_request(is_vision=True)
        # Reuse the cached wire client while kwargs are unchanged (openai.OpenAI + httpx pool costs
        # ~19-35ms/call). Single checked-out slot: `in_use` keeps two concurrent calls off one pool.
        stale = None
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            cached = cache["client"]
            if cached is not None and not cache["in_use"]:
                if (
                    not cache["poisoned"]
                    and cache["kwargs"] == request_kwargs
                    and not self._is_openai_client_closed(cached)
                ):
                    cache["in_use"] = True
                    return cached
                # kwargs changed, poisoned by a cross-thread abort (#29507), or externally closed — rebuild.
                stale = cached
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
        if stale is not None:
            # Safe to close here: in_use was False, so no worker thread owns the pool's FDs.
            self._close_openai_client(stale, reason=f"reuse_evict:{reason}", shared=False)
        client = self._create_openai_client(request_kwargs, reason=reason, shared=False)
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is None:
                cache["client"] = client
                # Snapshot nested dicts (default_headers) so an aliased inner object can't compare equal after
                # mutation.
                cache["kwargs"] = {
                    k: dict(v) if isinstance(v, dict) else v
                    for k, v in request_kwargs.items()
                }
                cache["poisoned"] = False
                cache["in_use"] = True
            # else: a concurrent call holds the slot — hand this client out untracked (fully closed later).
        return client

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is client:
                if reason in self._REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                    # Clean finish on the owning thread — keep the wire client
                    # (and its warm httpx pool) for the next sequential call.
                    cache["in_use"] = False
                    return
                # Failure / kill / abort: drop the slot and do a real close on the owning worker thread
                # (#29507).
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        self._close_openai_client(client, reason=reason, shared=False)

    def _close_cached_request_openai_client(self, *, reason: str) -> None:
        """Teardown hook: really close the cached per-request wire client."""
        with self._openai_client_lock():
            cache = getattr(self, "_request_client_cache", None)
            client = cache["client"] if cache else None
            in_use = bool(cache["in_use"]) if cache else False
            if cache is not None:
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        if client is None:
            return
        if in_use:
            # A worker has this client checked out; close() here would release FDs from a stranger thread
            # (#29507). Abort the sockets; the worker's own finally does the real close.
            self._abort_request_openai_client(client, reason=f"{reason}_in_flight")
            return
        self._close_openai_client(client, reason=reason, shared=False)

    def _abort_request_openai_client(self, client: Any, *, reason: str) -> None:
        """Cross-thread abort: shut sockets down without releasing FDs.

        For stranger-thread callers (interrupt loop, stale detector). ``close()`` from a non-owning thread
        raced the
        live SSL BIO and corrupted unrelated FDs; ``shutdown(SHUT_RDWR)`` unblocks the owner's recv/send so it
        closes from its own context.
        """
        if client is None:
            return
        # A pool whose sockets were shut from a stranger thread must never be reused: poison the slot.
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is client:
                cache["poisoned"] = True
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            # tcp_force_closed=0 means the abort found no sockets — the worker stays blocked (#72975). WARN.
            _log = logger.warning if shutdown_count == 0 else logger.info
            _log(
                "OpenAI client aborted (%s, shared=False, tcp_force_closed=%d, "
                "deferred_close=stranger_thread) %s%s",
                reason,
                shutdown_count,
                self._client_log_context(),
                (
                    " — no sockets found; in-flight request may keep running "
                    "until the provider finishes"
                    if shutdown_count == 0
                    else ""
                ),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client abort failed (%s, shared=False) %s error=%s",
                reason,
                self._client_log_context(),
                exc,
            )

    def _request_anthropic_client_cache_ref(self) -> dict:
        # Lazy init — tests build agents via AIAgent.__new__ without __init__.
        cache = getattr(self, "_request_anthropic_client_cache", None)
        if cache is None:
            cache = {"client": None, "key": None, "poisoned": False, "in_use": False}
            self._request_anthropic_client_cache = cache
        return cache

    def _request_anthropic_client_key(self) -> tuple:
        """Cache key covering everything that forces a fresh client: credential rotation, base URL / region,

        timeout (model switch), and the 1M-context beta flag.
        """
        if getattr(self, "provider", None) == "bedrock":
            region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
            return ("bedrock", region)
        return (
            "direct",
            self._anthropic_api_key,
            getattr(self, "_anthropic_base_url", None),
            get_provider_request_timeout(self.provider, self.model),
            bool(getattr(self, "_oauth_1m_beta_disabled", False)),
        )

    def _create_request_anthropic_client(self, *, reason: str) -> Any:
        """Build (or reuse) a request-local Anthropic client for one in-flight call.

        The stale/interrupt watchdog must never ``close()`` the client a worker is still reading (fd recycled
        under
        a live SSL BIO → TLS record in a SQLite header). A per-request client lets the stranger ``shutdown()``
        while
        the owner closes. Single-slot cache keyed as ``_request_anthropic_client_key``; ``in_use`` gives a
        second
        concurrent call a fresh untracked client. Mirrors ``_rebuild_anthropic_client`` construction.
        """
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        key = self._request_anthropic_client_key()

        stale = None
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            cached = cache["client"]
            if cached is not None and not cache["in_use"]:
                if (
                    not cache["poisoned"]
                    and cache["key"] == key
                    and not self._is_openai_client_closed(cached)
                ):
                    cache["in_use"] = True
                    return cached
                # Key changed, poisoned by a cross-thread abort, or externally closed — rebuild.
                stale = cached
                cache["client"] = None
                cache["key"] = None
                cache["poisoned"] = False
        if stale is not None:
            # Safe to close from this thread: in_use was False, so no worker
            # thread owns the pool's FDs (same #29507 reasoning as OpenAI).
            self._close_request_anthropic_client(stale, reason=f"reuse_evict:{reason}")

        if key[0] == "bedrock":
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            client = build_anthropic_bedrock_client(key[1])
        else:
            from agent.anthropic_adapter import build_anthropic_client
            client = build_anthropic_client(
                self._anthropic_api_key,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
                drop_context_1m_beta=key[4],
            )
        logger.debug(
            "Anthropic request client created (%s, shared=False) provider=%s model=%s",
            reason,
            getattr(self, "provider", None),
            getattr(self, "model", None),
        )
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            if cache["client"] is None:
                cache["client"] = client
                cache["key"] = key
                cache["poisoned"] = False
                cache["in_use"] = True
            # else: a concurrent call holds the slot — hand this client out untracked (fully closed later).
        return client

    def _close_request_anthropic_client(self, client: Any, *, reason: str) -> None:
        """Owner-thread close of a request-local Anthropic client.

        Clean finish (``_REQUEST_CLIENT_REUSE_REASONS``) keeps the pool warm in the slot; any other outcome
        force-closes the TCP sockets (CLOSE-WAIT hygiene) then does the graceful SDK close.
        """
        if client is None:
            return
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            if cache["client"] is client:
                if reason in self._REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                    cache["in_use"] = False
                    return
                cache["client"] = None
                cache["key"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        try:
            self._force_close_tcp_sockets(client)
            client.close()
            logger.info(
                "Anthropic client closed (%s, shared=False) provider=%s model=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
            )
        except Exception as exc:
            logger.debug(
                "Anthropic client close failed (%s, shared=False) provider=%s model=%s error=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                exc,
            )

    def _close_cached_request_anthropic_client(self, *, reason: str) -> None:
        """Teardown hook: really close the cached per-request Anthropic client."""
        with self._openai_client_lock():
            cache = getattr(self, "_request_anthropic_client_cache", None)
            client = cache["client"] if cache else None
            in_use = bool(cache["in_use"]) if cache else False
            if cache is not None:
                cache["client"] = None
                cache["key"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        if client is None:
            return
        if in_use:
            # A worker thread has this client checked out for an in-flight
            # request — same #29507 reasoning as the OpenAI teardown hook.
            self._abort_request_anthropic_client(client, reason=f"{reason}_in_flight")
            return
        try:
            self._force_close_tcp_sockets(client)
            client.close()
        except Exception:
            pass

    def _abort_request_anthropic_client(self, client: Any, *, reason: str) -> None:
        """Cross-thread abort for request-local Anthropic clients.

        Stranger threads only ``shutdown(SHUT_RDWR)`` the sockets — SDK ``close()`` would race the owner's
        live SSL BIO.
        """
        if client is None:
            return
        # A pool whose sockets were shut from a stranger thread must never be reused: poison the slot.
        with self._openai_client_lock():
            cache = self._request_anthropic_client_cache_ref()
            if cache["client"] is client:
                cache["poisoned"] = True
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            # Same contract as the OpenAI abort (#72975): zero sockets shut down is a WARNING, not success.
            _log = logger.warning if shutdown_count == 0 else logger.info
            _log(
                "Anthropic client aborted (%s, shared=False, tcp_force_closed=%d, "
                "deferred_close=stranger_thread) provider=%s model=%s%s",
                reason,
                shutdown_count,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                (
                    " — no sockets found; in-flight request may keep running "
                    "until the provider finishes"
                    if shutdown_count == 0
                    else ""
                ),
            )
        except Exception as exc:
            logger.debug(
                "Anthropic client abort failed (%s, shared=False) provider=%s model=%s error=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                exc,
            )

    _run_codex_stream = _forward("agent.codex_runtime", "run_codex_stream")

    _run_codex_create_stream_fallback = _forward("agent.codex_runtime", "run_codex_create_stream_fallback")

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider not in {"openai-codex", "xai-oauth"}:
            return False

        # Guard against silent account swap: a non-singleton credential (manual pool entry, explicit
        # api_key=) must not be replaced by the device_code singleton's tokens. The pool's reactive
        # recovery owns that case; this singleton fallback MUST only fire on singleton tokens.
        try:
            if self.provider == "openai-codex":
                from hermes_cli.auth import resolve_codex_runtime_credentials

                singleton_now = resolve_codex_runtime_credentials(
                    refresh_if_expiring=False,
                )
            else:
                from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

                singleton_now = resolve_xai_oauth_runtime_credentials(
                    refresh_if_expiring=False,
                )
        except Exception as exc:
            logger.debug("%s singleton read failed: %s", self.provider, exc)
            return False

        singleton_key = str(singleton_now.get("api_key") or "").strip()
        active_key = str(self.api_key or "").strip()
        if singleton_key and active_key and singleton_key != active_key:
            logger.debug(
                "%s singleton tokens differ from the active api_key; "
                "skipping singleton force-refresh to avoid silent account swap. "
                "Reactive credential rotation should go through the pool.",
                self.provider,
            )
            return False

        try:
            if self.provider == "openai-codex":
                from hermes_cli.auth import resolve_codex_runtime_credentials

                old_key = str(self.api_key or "").strip()
                creds = resolve_codex_runtime_credentials(force_refresh=force)
            else:
                from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

                old_key = str(self.api_key or "").strip()
                creds = resolve_xai_oauth_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("%s credential refresh failed: %s", self.provider, exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        # Return False when no NEW token was minted: resolve_codex_runtime_credentials returns the same
        # stale token when refresh fails.
        new_key = api_key.strip()
        if old_key and new_key == old_key:
            logger.debug(
                "%s credential refresh returned the same token; "
                "refresh likely failed silently",
                self.provider,
            )
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason=f"{self.provider}_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(
        self,
        *,
        force: bool = True,
    ) -> bool:
        if self.provider != "nous":
            return False
        # Portal serves anthropic/* on the native Messages route, so either client kind may hold the
        # expiring invoke JWT.
        if self.api_mode not in ("chat_completions", "anthropic_messages"):
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
                force_refresh=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")

        if self.api_mode == "anthropic_messages":
            self._anthropic_api_key = self.api_key
            self._anthropic_base_url = self.base_url
            self._rebuild_anthropic_client()
            return True

        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_env_client_credentials(self) -> bool:
        """Adopt ~/.hermes/.env credential/base-url edits at the turn boundary.

        A Settings save updates ``.env`` but a live worker keeps init-time values, so an open chat kept
        calling the
        old endpoint. Reacts only to env *edits* (resolved value changed since last look), never to divergence
        from
        the agent's current values — pool rotation/failover and a config ``model.base_url`` legitimately move
        the
        session and must not flap. Covers registry providers and named custom providers with ``key_env``.
        """
        if self.api_mode != "chat_completions":
            return False
        if getattr(self, "_fallback_activated", False):
            return False
        try:
            from agent.credential_pool import get_env_prefer_dotenv
            from hermes_cli.auth import PROVIDER_REGISTRY
        except ImportError:
            return False

        pconfig = PROVIDER_REGISTRY.get(self.provider)
        if (
            pconfig
            and getattr(pconfig, "auth_type", "") == "api_key"
            and getattr(pconfig, "api_key_env_vars", ())
        ):
            api_key = ""
            for env_var in pconfig.api_key_env_vars:
                api_key = get_env_prefer_dotenv(env_var).strip()
                if api_key:
                    break
            if not api_key:
                return False

            env_url = ""
            if pconfig.base_url_env_var:
                env_url = get_env_prefer_dotenv(pconfig.base_url_env_var).strip().rstrip("/")
            default_base = (pconfig.inference_base_url or "").strip().rstrip("/")
            base_url = env_url or default_base
            if self.provider == "kimi-coding":
                from hermes_cli.auth import _resolve_kimi_base_url

                base_url = _resolve_kimi_base_url(
                    api_key, pconfig.inference_base_url, env_url
                ).rstrip("/")
            elif self.provider == "zai":
                from hermes_cli.auth import _resolve_zai_base_url

                base_url = _resolve_zai_base_url(
                    api_key, pconfig.inference_base_url, env_url
                ).rstrip("/")
        elif self.provider == "custom":
            # Named custom provider (#67935): identity in config, credential in the env var named by key_env;
            # entries without key_env have nothing env-sourced to watch.
            try:
                from hermes_cli.runtime_provider import _get_named_custom_provider
            except ImportError:
                return False
            custom_provider = _get_named_custom_provider(
                getattr(self, "requested_provider", "") or ""
            )
            if not custom_provider:
                return False
            key_env = str(custom_provider.get("key_env") or "").strip()
            if not key_env:
                return False
            api_key = get_env_prefer_dotenv(key_env).strip()
            if not api_key:
                return False
            # Custom providers pin base_url in config, so only key edits are adopted here.
            default_base = str(custom_provider.get("base_url") or "").strip().rstrip("/")
            base_url = default_base
        else:
            return False

        if not base_url:
            return False

        resolved = (base_url, api_key)
        prev = getattr(self, "_env_creds_seen", None)
        current_base = (self.base_url or "").strip().rstrip("/")

        if prev is None:
            # First look — adopt only the boot-default case; anything else is unattributable on turn one.
            adopt = current_base == default_base and not (
                base_url == current_base and api_key == self.api_key
            )
            # #79156: a pool-rotated key is not a boot-time env adoption; don't stomp it.
            if (
                adopt
                and api_key != self.api_key
                and getattr(self, "_credential_pool", None) is not None
                and getattr(self, "_credential_pool_entry_id", None)
            ):
                adopt = False
        else:
            # Env unchanged → no-op; drift from self.* is rotation/failover/config precedence. Adopt only
            # while the session still runs on the registry default or the previously-seen env value.
            adopt = (
                resolved != prev
                and current_base in {default_base, prev[0]}
                and not (base_url == current_base and api_key == self.api_key)
            )

        if not adopt:
            self._env_creds_seen = resolved
            return False

        from hermes_cli.route_identity import normalize_route_base_url

        route_changed = normalize_route_base_url(self.base_url) != normalize_route_base_url(
            base_url
        )
        prior_api_key = self.api_key
        prior_base_url = self.base_url
        prior_client_kwargs = dict(self._client_kwargs)

        self.api_key = api_key
        self.base_url = base_url
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # A base-url change moves the route: recompute TLS material and default headers.
        self._reapply_route_client_config(route_changed=route_changed)

        if not self._replace_primary_openai_client(reason="env_credential_refresh"):
            # Leave the baseline un-advanced (retry next turn) and roll the agent back to match the live
            # client.
            self.api_key = prior_api_key
            self.base_url = prior_base_url
            self._client_kwargs.clear()
            self._client_kwargs.update(prior_client_kwargs)
            return False

        # Rebind the pool entry id to the adopted key, or the next 429 quarantines the wrong credential
        # (#79156).
        try:
            from agent.agent_runtime_helpers import sync_credential_pool_entry_id

            sync_credential_pool_entry_id(self)
        except Exception:
            logger.debug(
                "sync_credential_pool_entry_id after env refresh failed",
                exc_info=True,
            )

        self._env_creds_seen = resolved
        logger.info(
            "Applied updated .env credentials for %s: endpoint %s",
            self.provider,
            self.base_url,
        )
        return True

    def _try_refresh_vertex_client_credentials(self) -> bool:
        """Re-mint the Vertex OAuth2 access token (~1h TTL) and rebuild the OpenAI client.

        Long-lived gateway sessions 401 mid-session on the expired bearer. Returns True when a usable
        token+base_url were obtained.
        """
        if self.api_mode != "chat_completions" or self.provider != "vertex":
            return False

        try:
            from agent.vertex_adapter import get_vertex_config

            token, base_url = get_vertex_config()
        except Exception as exc:
            logger.debug("Vertex credential refresh failed: %s", exc)
            return False

        if not isinstance(token, str) or not token.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = token.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="vertex_credential_refresh"):
            return False

        logger.info("Vertex AI OAuth token refreshed")
        return True

    def _try_refresh_copilot_client_credentials(self) -> bool:
        """Refresh Copilot credentials and rebuild the shared OpenAI client.

        The raw GitHub token is stable, but the short-TTL *exchanged* IDE token is what authenticates and
        expires
        mid-turn (``401 IDE token expired``). Re-resolving the raw token leaves the same expired JWT on the
        wire, so
        force a fresh exchange. Caller enforces the single-shot guard.
        """
        if not self._is_copilot_provider():
            return False

        try:
            from hermes_cli.copilot_auth import (
                resolve_copilot_token,
                get_copilot_api_token,
                evict_cached_exchanged_token,
            )

            new_token, token_source = resolve_copilot_token()
        except Exception as exc:
            logger.debug("Copilot credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False

        new_token = new_token.strip()

        # Force a fresh IDE-token exchange (the cached JWT is what expired); fall back to the raw token
        # only if the exchange itself is unavailable.
        try:
            evict_cached_exchanged_token(new_token)
            api_token, enterprise_base_url = get_copilot_api_token(new_token)
            if isinstance(api_token, str) and api_token.strip():
                new_token = api_token.strip()
                if enterprise_base_url:
                    self.base_url = enterprise_base_url.rstrip("/")
        except Exception as exc:
            logger.debug("Copilot 401 re-exchange failed, using resolved token: %s", exc)

        self.api_key = new_token
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(str(self.base_url or ""))

        if not self._replace_primary_openai_client(reason="copilot_credential_refresh"):
            return False

        logger.info("Copilot credentials refreshed from %s", token_source)
        return True

    def _try_recover_stale_copilot_credential(self) -> bool:
        """Force a fresh Copilot token exchange + client rebuild after a 400.

        Copilot surfaces a stale credential as ``400 model_not_available_for_integrator`` /
        ``model_not_supported``,
        not a 401 — typically a raw ``ghu_`` token cached when the startup exchange degraded, routing to the
        restricted integrator allowlist. Evict, re-exchange, re-apply headers, rebuild. Single-shot (caller-
        guarded).
        """
        if not self._is_copilot_provider():
            return False

        try:
            from hermes_cli.copilot_auth import (
                resolve_copilot_token,
                get_copilot_api_token,
                evict_cached_exchanged_token,
            )

            raw_token, token_source = resolve_copilot_token()
            if not isinstance(raw_token, str) or not raw_token.strip():
                return False
            raw_token = raw_token.strip()

            # Drop any cached (possibly degraded/raw) exchanged token so the
            # next exchange hits the network and mints a fresh one.
            evict_cached_exchanged_token(raw_token)

            api_token, enterprise_base_url = get_copilot_api_token(raw_token)
        except Exception as exc:
            logger.debug("Copilot stale-credential recovery failed: %s", exc)
            return False

        if not isinstance(api_token, str) or not api_token.strip():
            return False

        # If the exchange STILL degraded to the raw token, a rebuild won't help
        # — don't burn the single-shot retry on an identical request.
        if api_token == raw_token and not enterprise_base_url:
            logger.warning(
                "Copilot stale-credential recovery: exchange still degraded to "
                "raw token; skipping retry (network/exchange endpoint unavailable)."
            )
            return False

        self.api_key = api_token.strip()
        if enterprise_base_url:
            self.base_url = enterprise_base_url.rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(str(self.base_url or ""))

        if not self._replace_primary_openai_client(reason="copilot_stale_credential_recovery"):
            return False

        logger.info("Copilot credentials re-exchanged after stale-credential 400 (source=%s)", token_source)
        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False
        # Azure endpoints use static API keys — OAuth token rotation doesn't apply.
        # Refreshing would pick up ~/.claude/.credentials.json OAuth token and break auth.
        _base = getattr(self, "_anthropic_base_url", "") or ""
        if base_url_host_matches(_base, "azure.com"):
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(
                new_token,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # Update the OAuth flag — only on native Anthropic; third-party Anthropic-protocol endpoints must
        # not trip OAuth paths (#1739).
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token) if self.provider == "anthropic" else False
        return True

    def _apply_client_headers_for_base_url(
        self,
        base_url: str,
        *,
        apply_user_headers: bool = True,
    ) -> None:
        for host, build in _ROUTE_DEFAULT_HEADERS:
            if base_url_host_matches(base_url, host):
                self._client_kwargs["default_headers"] = build(self, base_url)
                break
        else:
            # No URL-specific headers — check profile.default_headers before clearing.
            _ph_headers = None
            try:
                from providers import get_provider_profile as _gpf2
                _ph2 = _gpf2(self.provider)
                if _ph2 and _ph2.default_headers:
                    _ph_headers = dict(_ph2.default_headers)
            except Exception:
                pass
            if _ph_headers:
                self._client_kwargs["default_headers"] = _ph_headers
            else:
                self._client_kwargs.pop("default_headers", None)

        # User-configured overrides win over URL/profile defaults for the same
        # route. A credential swap to another endpoint must not inherit them.
        if apply_user_headers:
            self._apply_user_default_headers()

        # Per-provider extra_headers applied last so they survive credential swaps and rebuilds.
        # SECURITY: values may carry credentials — never log them.
        if self.api_mode not in ("anthropic_messages", "bedrock_converse"):
            try:
                from hermes_cli.config import (
                    apply_custom_provider_extra_headers_to_client_kwargs,
                )

                apply_custom_provider_extra_headers_to_client_kwargs(
                    self._client_kwargs, base_url,
                )
            except Exception:
                logger.debug("custom-provider extra_headers skipped", exc_info=True)

    def _apply_user_default_headers(self) -> None:
        """Merge ``model.default_headers`` from config onto the OpenAI client (user values win).

        Lets custom endpoints behind a WAF that rejects the SDK's identifying headers (``User-Agent``,
        ``X-Stainless-*``) work. Delegates to ``agent.auxiliary_client._apply_user_default_headers`` so main
        and
        auxiliary clients cannot drift. No-op for Anthropic/Bedrock modes.
        """
        if self.api_mode in ("anthropic_messages", "bedrock_converse"):
            return
        from agent.auxiliary_client import (
            _apply_user_default_headers as _merge_user_headers,
        )
        merged = _merge_user_headers(self._client_kwargs.get("default_headers"))
        if merged:
            self._client_kwargs["default_headers"] = merged

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url
        self._credential_pool_entry_id = getattr(entry, "id", None)
        from hermes_cli.route_identity import normalize_route_base_url

        route_changed = normalize_route_base_url(self.base_url) != normalize_route_base_url(
            runtime_base
        )

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
            self._anthropic_client = build_anthropic_client(
                runtime_key, self._anthropic_base_url,
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
            self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
            self.api_key = runtime_key
            self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._reapply_route_client_config(route_changed=route_changed)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _reapply_route_client_config(self, *, route_changed: bool) -> None:
        """Recompute route-derived client kwargs (TLS material, default headers) for the current
        ``self.base_url``.

        Any rebuild that may have moved ``base_url`` must call this or the new endpoint inherits the old one's
        config. Shared by pool rotation and the per-turn env refresh so they cannot drift.
        """
        self._client_kwargs.pop("ssl_verify", None)
        self._client_kwargs.pop("ssl_ca_cert", None)
        try:
            from hermes_cli.config import (
                apply_custom_provider_tls_to_client_kwargs,
                get_compatible_custom_providers,
                load_config_readonly,
            )

            apply_custom_provider_tls_to_client_kwargs(
                self._client_kwargs,
                str(self.base_url or ""),
                get_compatible_custom_providers(load_config_readonly()),
            )
        except Exception:
            logger.debug(
                "custom-provider TLS resolution skipped on credential rotation",
                exc_info=True,
            )
        self._apply_client_headers_for_base_url(
            self.base_url,
            apply_user_headers=not route_changed,
        )

    _recover_with_credential_pool = _forward("agent.agent_runtime_helpers", "recover_with_credential_pool")

    def _anthropic_messages_create(self, api_kwargs: dict, *, client: Any = None):
        # A supplied request-local client was already refreshed in _create_request_anthropic_client.
        if client is None and self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        # Strip Responses-only kwargs that leak in under an api_mode-flip race (#31673).
        from agent.anthropic_adapter import create_anthropic_message
        return create_anthropic_message(
            client or self._anthropic_client,
            api_kwargs,
            log_prefix=getattr(self, "log_prefix", ""),
            prefer_stream=not bool(getattr(self, "_disable_streaming", False)),
            # Rate-limit + credits state live in response headers, which the parsed Message drops.
            on_response=self._capture_anthropic_response_headers,
        )

    def _rebuild_anthropic_client(self) -> None:
        """Rebuild the Anthropic client after an interrupt or stale call.

        Uses the Bedrock SDK when provider is bedrock (``build_anthropic_client()`` needs a direct key).
        Honors
        ``_oauth_1m_beta_disabled`` so the rebuilt client carries the reduced beta set.
        """
        _drop_1m = bool(getattr(self, "_oauth_1m_beta_disabled", False))
        if getattr(self, "provider", None) == "bedrock":
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
            self._anthropic_client = build_anthropic_bedrock_client(region)
        else:
            from agent.anthropic_adapter import build_anthropic_client
            self._anthropic_client = build_anthropic_client(
                self._anthropic_api_key,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
                drop_context_1m_beta=_drop_1m,
            )
