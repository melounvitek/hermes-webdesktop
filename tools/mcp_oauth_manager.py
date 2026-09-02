#!/usr/bin/env python3
"""Central manager for per-server MCP OAuth state.

One instance per process. Holds per-server provider instances and coordinates:

- **Cross-process token reload** via mtime-based disk watch, so tokens
  refreshed by another process (cron, another CLI) are picked up without a
  restart (Claude Code's ``invalidateOAuthCacheIfDiskChanged`` bug class).
- **401 deduplication** via in-flight futures: N concurrent tool calls hitting
  401 with the same access_token trigger one recovery attempt.
- **Reconnect signalling** — ``MCPServerTask`` in ``mcp_tool.py`` drives the
  reconnect; the manager decides when it is warranted.

This module is the ONLY place that instantiates the SDK's ``OAuthClientProvider``
for runtime use; other code paths go through ``get_manager()``. We lean on the
SDK's lazy refresh rather than refreshing before every op: one ``stat()`` per
tool call is cheaper than an await + refresh round-trip.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _same_endpoint(a: str, b: str) -> bool:
    """Return True if two URLs target the same endpoint (ignoring query/fragment).

    Compares scheme, host (case-insensitive), and path. Used to confirm a
    rejected response actually came from the OAuth token endpoint before we
    act on an ``invalid_client`` body.
    """
    from urllib.parse import urlsplit

    try:
        pa, pb = urlsplit(a), urlsplit(b)
    except ValueError:  # pragma: no cover — malformed URL
        return False
    return (
        pa.scheme == pb.scheme
        and pa.netloc.lower() == pb.netloc.lower()
        and pa.path.rstrip("/") == pb.path.rstrip("/")
    )


# ---------------------------------------------------------------------------
# Per-server entry
# ---------------------------------------------------------------------------


@dataclass
class _ProviderEntry:
    """Per-server OAuth state. ``last_mtime_ns`` is the last-seen tokens-file
    mtime (0 = never read) for external-refresh detection; ``lock`` binds to
    whichever asyncio loop first awaits it (the MCP event loop);
    ``pending_401`` dedupes thundering-herd 401s by failed access_token."""

    server_url: str
    oauth_config: Optional[dict]
    provider: Optional[Any] = None
    last_mtime_ns: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_401: dict[str, "asyncio.Future[bool]"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HermesMCPOAuthProvider — OAuthClientProvider subclass with disk-watch
# ---------------------------------------------------------------------------


def _make_hermes_provider_class() -> Optional[type]:
    """Lazy-import the SDK base class and return our subclass (None if the
    SDK's OAuth module is unavailable, so this module still imports)."""
    try:
        from mcp.client.auth.oauth2 import OAuthClientProvider
    except ImportError:  # pragma: no cover — SDK required in CI
        return None
    from tools.mcp_oauth_provider import HermesProviderMixin

    class HermesMCPOAuthProvider(HermesProviderMixin, OAuthClientProvider):
        """OAuthClientProvider with pre-flow disk-mtime reload.

        Before every ``async_auth_flow`` the manager checks whether the tokens
        file changed on disk and, if so, resets ``_initialized`` so the next
        flow re-reads storage — making external refreshes visible to a running
        session. Token-endpoint fixes come from ``HermesProviderMixin``.
        """

        _hermes_logger = logger

        def __init__(
            self,
            *args: Any,
            server_name: str = "",
            preregistered: bool = False,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            # mcp 2.0 uses a task-owned anyio.Lock held across the yielded
            # resource request: a session-long GET blocks every concurrent
            # POST, and HTTPX may close the auth-flow generator from another
            # task. A binary semaphore keeps mutual exclusion without task
            # ownership; async_auth_flow narrows its scope around resource I/O.
            import anyio

            self.context.lock = anyio.Semaphore(1, max_value=1)
            self._hermes_server_name = server_name
            self._hermes_home = ""
            # A config-supplied (pre-registered) client_id rejected as
            # invalid_client means the *config* is wrong — re-registration
            # can't help, so only dynamically-registered clients auto-heal.
            self._hermes_preregistered = preregistered

        def _hermes_storage(self):
            """The context storage when it is a ``HermesTokenStorage``, else None."""
            from tools.mcp_oauth import HermesTokenStorage

            storage = self.context.storage
            return storage if isinstance(storage, HermesTokenStorage) else None

        async def _initialize(self) -> None:
            """Load stored state, seed ``token_expiry_time``, restore/prefetch metadata.

            The SDK's ``_initialize`` populates ``current_tokens`` but never
            calls ``update_token_expiry``, so ``is_token_valid()`` is True for
            any loaded token regardless of age and a restarted process ships
            stale Bearer tokens (some providers answer 200 with an app-level
            auth error the transport can't see). Seeding the expiry makes the
            SDK take ``can_refresh_token()`` and refresh before the first
            request; ``HermesTokenStorage`` persists absolute ``expires_at`` so
            the TTL reflects wall-clock age.

            Metadata is restored from disk, else discovered pre-flight when we
            hold tokens but no metadata: otherwise ``_refresh_token`` guesses
            ``{server_url}/token`` (wrong for split-origin providers such as
            BetterStack), the refresh 404s and we fall through to browser reauth.
            """
            await super()._initialize()
            tokens = self.context.current_tokens
            if tokens is not None and tokens.expires_in is not None:
                self.context.update_token_expiry(tokens)

            storage = self._hermes_storage()
            if storage is not None and self.context.oauth_metadata is None:
                meta = storage.load_oauth_metadata()
                if meta is not None:
                    self.context.oauth_metadata = meta
                    logger.debug(
                        "MCP OAuth '%s': restored metadata from disk "
                        "(token_endpoint=%s)",
                        self._hermes_server_name,
                        meta.token_endpoint,
                    )

            if tokens is not None and self.context.oauth_metadata is None:
                try:
                    await self._prefetch_oauth_metadata()
                except Exception as exc:  # pragma: no cover — defensive
                    # Non-fatal: the SDK's 401-branch discovery runs next request.
                    logger.debug(
                        "MCP OAuth '%s': pre-flight metadata discovery "
                        "failed (non-fatal): %s",
                        self._hermes_server_name, exc,
                    )

        async def _prefetch_oauth_metadata(self) -> None:
            """Fetch PRM + ASM from the well-known endpoints and cache on context.

            Mirrors the SDK's 401-branch discovery but runs before the first
            request. Uses the SDK's own URL builders/response handlers so we
            track whatever the pinned SDK version expects.
            """
            # The SDK's httpx flavour, not Hermes' — mcp 2.0 builds on httpx2 and
            # `create_oauth_metadata_request` returns *its* Request objects,
            # which only its own AsyncClient can send (tools.mcp_tool.sdk_httpx).
            from tools.mcp_tool import sdk_httpx
            httpx = sdk_httpx()
            if httpx is None:  # pragma: no cover — SDK import would have failed
                return
            from mcp.client.auth.utils import (
                build_oauth_authorization_server_metadata_discovery_urls,
                build_protected_resource_metadata_discovery_urls,
                create_oauth_metadata_request,
                handle_auth_metadata_response,
                handle_protected_resource_response,
            )

            server_url = self.context.server_url

            async def _send(client, url: str, label: str):
                try:
                    return await client.send(create_oauth_metadata_request(url))
                except httpx.HTTPError as exc:
                    logger.debug(
                        "MCP OAuth '%s': %s discovery to %s failed: %s",
                        self._hermes_server_name, label, url, exc,
                    )
                    return None

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Step 1: PRM discovery to learn the authorization_server URL.
                for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                    resp = await _send(client, url, "PRM")
                    if resp is None:
                        continue
                    prm = await handle_protected_resource_response(resp)
                    if prm:
                        self.context.protected_resource_metadata = prm
                        if prm.authorization_servers:
                            self.context.auth_server_url = str(prm.authorization_servers[0])
                        break

                # Step 2: ASM discovery against auth_server_url (server_url
                # fallback for legacy providers).
                for url in build_oauth_authorization_server_metadata_discovery_urls(
                    self.context.auth_server_url, server_url
                ):
                    resp = await _send(client, url, "ASM")
                    if resp is None:
                        continue
                    ok, asm = await handle_auth_metadata_response(resp)
                    if not ok:
                        break
                    if asm:
                        self.context.oauth_metadata = asm
                        # Persist now so a later cold-load skips discovery.
                        storage = self._hermes_storage()
                        if storage is not None:
                            storage.save_oauth_metadata(asm)
                        logger.debug(
                            "MCP OAuth '%s': pre-flight ASM discovered "
                            "token_endpoint=%s",
                            self._hermes_server_name, asm.token_endpoint,
                        )
                        break

        def _persist_oauth_metadata_if_changed(self) -> None:
            """Save metadata the SDK discovered lazily (401 branch) for future
            restarts; no-op when absent, not our storage, or unchanged."""
            meta = self.context.oauth_metadata
            storage = self._hermes_storage()
            if meta is None or storage is None:
                return
            existing = storage.load_oauth_metadata()
            if existing is None or str(existing.token_endpoint) != str(meta.token_endpoint):
                storage.save_oauth_metadata(meta)

        async def _maybe_flag_poisoned_client(self, response: Any) -> None:
            """Detect a dead client registration and force re-registration.

            An ``invalid_client`` rejection of our ``client_id`` at the token
            endpoint (exchange or refresh) proves the cached registration is
            dead server-side; delete ``client.json`` (+ stale metadata) so the
            SDK re-runs DCR next flow. The browser-side "Redirect URI Mismatch"
            case has no HTTP signal and is left to ``hermes mcp reauth``.

            Conservative by construction — acts ONLY when status is 400/401,
            the request hit the discovered ``token_endpoint`` (the only request
            carrying our ``client_id``), and the body carries ``invalid_client``
            as a whole word (so RFC 7591's ``invalid_client_metadata`` does not
            trip it). Pre-registered clients are never poisoned. Best-effort:
            any failure is swallowed so a miss never breaks the live flow. If
            ``token_endpoint`` was never discovered the guard returns early.
            """
            try:
                if self._hermes_preregistered:
                    return
                if getattr(response, "status_code", None) not in (400, 401):
                    return
                meta = getattr(self.context, "oauth_metadata", None)
                token_endpoint = (
                    str(meta.token_endpoint)
                    if meta is not None and getattr(meta, "token_endpoint", None)
                    else None
                )
                req = getattr(response, "request", None)
                req_url = str(req.url) if req is not None else None
                if not token_endpoint or not req_url:
                    return
                if not _same_endpoint(req_url, token_endpoint):
                    return
                body = await response.aread()
                if not re.search(rb"\binvalid_client\b", body.lower()):
                    return

                storage = self._hermes_storage()
                # If the rejected client_id was our CIMD URL, re-presenting it
                # would loop (the server already fetched and refused it). Drop
                # the URL so the retry takes DCR, and mark it on disk so the
                # next process doesn't walk back into the same refusal
                # (`hermes mcp login` clears the marker).
                cimd_url = getattr(self.context, "client_metadata_url", None)
                rejected_id = getattr(self.context.client_info, "client_id", None)
                if cimd_url and rejected_id == cimd_url:
                    logger.warning(
                        "MCP OAuth '%s': authorization server rejected our "
                        "Client ID Metadata Document (%s) with invalid_client "
                        "— falling back to dynamic client registration.",
                        self._hermes_server_name, cimd_url,
                    )
                    self.context.client_metadata_url = None
                    if storage is not None:
                        storage.mark_cimd_rejected()

                if storage is not None:
                    storage.poison_client_registration()
                # Drop the in-memory client so the SDK re-registers next flow.
                self.context.client_info = None
                self._initialized = False
            except Exception as exc:  # pragma: no cover — defensive, must not throw
                logger.debug(
                    "MCP OAuth '%s': invalid_client detection failed (non-fatal): %s",
                    self._hermes_server_name, exc,
                )

        async def async_auth_flow(self, request):  # type: ignore[override]
            # Pre-flow hook: reload from disk if it changed (non-fatal on error).
            try:
                await get_manager().invalidate_if_disk_changed(
                    self._hermes_server_name,
                    hermes_home=self._hermes_home,
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(
                    "MCP OAuth '%s': pre-flow disk-watch failed (non-fatal): %s",
                    self._hermes_server_name, exc,
                )

            # Bridge the bidirectional generator protocol by hand: httpx feeds
            # responses back via ``auth_flow.asend(response)``. A naive
            # ``async for item in inner: yield item`` DISCARDS those values, so
            # the SDK's ``response = yield request`` sees None and crashes on
            # ``response.status_code`` (tests/tools/test_mcp_oauth_bidirectional.py).
            inner = super().async_auth_flow(request)
            resource_lock_released = False
            sent_access_token = None
            retry_after_concurrent_auth = False
            try:
                outgoing = await inner.__anext__()
                while True:
                    # The SDK holds context.lock for its whole generator, even
                    # while HTTPX waits on the MCP request. Release it for that
                    # request only; discovery/refresh/registration/exchange stay
                    # serialized exactly as the SDK implements them.
                    if outgoing is request:
                        tokens = self.context.current_tokens
                        sent_access_token = (
                            tokens.access_token if tokens is not None else None
                        )
                        self.context.lock.release()
                        resource_lock_released = True
                    incoming = yield outgoing
                    if resource_lock_released:
                        await self.context.lock.acquire()
                        resource_lock_released = False
                    # Another request may have refreshed/authorized while this
                    # one was in flight: retry with that token instead of a
                    # duplicate OAuth transition from the stale 401/403.
                    tokens = self.context.current_tokens
                    if (
                        getattr(incoming, "status_code", None) in (401, 403)
                        and self.context.is_token_valid()
                        and tokens is not None
                        and tokens.access_token != sent_access_token
                    ):
                        self._add_auth_header(request)
                        await inner.aclose()
                        retry_after_concurrent_auth = True
                        break
                    # Sniff for a dead-client-registration signal (best-effort).
                    await self._maybe_flag_poisoned_client(incoming)
                    outgoing = await inner.asend(incoming)
            except StopAsyncIteration:
                # Persist metadata discovered lazily in the 401 branch.
                self._persist_oauth_metadata_if_changed()
                return
            finally:
                if resource_lock_released:
                    # Balance the SDK's surrounding ``async with`` even when
                    # HTTPX cancels/closes the flow mid-request. Shield only
                    # this local bookkeeping.
                    import anyio

                    with anyio.CancelScope(shield=True):
                        await self.context.lock.acquire()

            if retry_after_concurrent_auth:
                yield request
                self._persist_oauth_metadata_if_changed()
                return

    return HermesMCPOAuthProvider


# Cached at import time. Tested and used by :class:`MCPOAuthManager`.
_HERMES_PROVIDER_CLS: Optional[type] = _make_hermes_provider_class()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MCPOAuthManager:
    """Single source of truth for per-server MCP OAuth state.

    Thread-safe: the ``_entries`` dict is guarded by ``_entries_lock`` for
    get-or-create semantics. Per-entry state is guarded by the entry's own
    ``asyncio.Lock`` (used from the MCP event loop thread).
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _ProviderEntry] = {}
        self._entries_lock = threading.Lock()
        # Strong refs to in-flight 401 tasks so the loop's weak bookkeeping
        # cannot GC them mid-run and leave `await pending` hanging forever.
        self._inflight_tasks: set[asyncio.Task] = set()

    # -- Provider construction / caching -------------------------------------

    def get_or_build_provider(
        self,
        server_name: str,
        server_url: str,
        oauth_config: Optional[dict],
    ) -> Optional[Any]:
        """Return a cached OAuth provider for ``server_name`` or build one.

        Idempotent: repeat calls with the same name return the same instance.
        If ``server_url`` changes for a given name, the cached entry is
        discarded and a fresh provider is built.

        Returns None if the MCP SDK's OAuth support is unavailable.
        """
        key = self._key(server_name)
        with self._entries_lock:
            entry = self._entries.get(key)
            if entry is not None and entry.server_url != server_url:
                logger.info(
                    "MCP OAuth '%s': URL changed from %s to %s, discarding cache",
                    server_name, entry.server_url, server_url,
                )
                entry = None

            if entry is None:
                entry = _ProviderEntry(
                    server_url=server_url,
                    oauth_config=oauth_config,
                )
                self._entries[key] = entry

            if entry.provider is None:
                entry.provider = self._build_provider(server_name, entry)
                if entry.provider is not None:
                    entry.provider._hermes_home = key[0]

            return entry.provider

    @staticmethod
    def _key(
        server_name: str,
        hermes_home: str | Path | None = None,
    ) -> tuple[str, str]:
        from hermes_constants import get_hermes_home

        home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
        return (str(home.expanduser().resolve(strict=False)), server_name)

    def _build_provider(
        self,
        server_name: str,
        entry: _ProviderEntry,
    ) -> Optional[Any]:
        """Build a :class:`HermesMCPOAuthProvider` from the shared
        ``tools.mcp_oauth`` helpers; None if the SDK's OAuth support is unavailable."""
        if _HERMES_PROVIDER_CLS is None:
            logger.warning(
                "MCP OAuth '%s': SDK auth module unavailable", server_name,
            )
            return None

        # Local imports avoid circular deps at module import time.
        from tools.mcp_oauth import _OAUTH_AVAILABLE, OAuthNonInteractiveError, _is_interactive
        from tools.mcp_oauth_provider import build_provider_kwargs, prepare_oauth_config

        if not _OAUTH_AVAILABLE:
            return None

        cfg, storage = prepare_oauth_config(server_name, entry.server_url, entry.oauth_config)

        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

        if (
            get_dashboard_oauth_flow() is None
            and not _is_interactive()
            and not storage.has_cached_tokens()
        ):
            raise OAuthNonInteractiveError(
                "MCP OAuth for "
                f"'{server_name}': non-interactive environment and no "
                "cached tokens found. Run `hermes mcp login "
                f"{server_name}` interactively first to complete initial "
                "authorization."
            )

        return _HERMES_PROVIDER_CLS(
            server_name=server_name,
            preregistered=bool(cfg.get("client_id")),
            server_url=entry.server_url,
            **build_provider_kwargs(cfg, storage, ssh_proxy_hint=False),
        )

    def remove(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
    ) -> _ProviderEntry | None:
        """Evict the provider from cache AND delete tokens from disk.

        Called by ``hermes mcp remove <name>`` and (indirectly) by
        ``hermes mcp login <name>`` during forced re-auth.
        """
        entry = self.evict(server_name, hermes_home=hermes_home)
        from tools.mcp_oauth import remove_oauth_tokens
        remove_oauth_tokens(server_name, hermes_home=hermes_home)
        logger.info(
            "MCP OAuth '%s': evicted from cache and removed from disk",
            server_name,
        )
        return entry

    def restore_entry(
        self,
        server_name: str,
        entry: _ProviderEntry | None,
        *,
        hermes_home: str | Path | None = None,
    ) -> None:
        """Restore a provider entry removed for a failed reauthorization."""
        if entry is None:
            return
        with self._entries_lock:
            self._entries.setdefault(self._key(server_name, hermes_home), entry)

    def evict(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
    ) -> _ProviderEntry | None:
        """Drop only the in-process provider, preserving persisted OAuth state."""
        with self._entries_lock:
            return self._entries.pop(self._key(server_name, hermes_home), None)

    # -- Disk watch ----------------------------------------------------------

    async def invalidate_if_disk_changed(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
    ) -> bool:
        """Force the SDK provider to reload when the tokens file mtime changed.

        Returns True if invalidated. This is the external-refresh fix: a cron
        job writes fresh tokens and the next tool call picks them up.
        """
        from tools.mcp_oauth import _get_token_dir, _safe_filename

        entry = self._entries.get(self._key(server_name, hermes_home))
        if entry is None or entry.provider is None:
            return False

        async with entry.lock:
            tokens_path = _get_token_dir(hermes_home) / f"{_safe_filename(server_name)}.json"
            try:
                mtime_ns = tokens_path.stat().st_mtime_ns
            except (FileNotFoundError, OSError):
                return False

            if mtime_ns != entry.last_mtime_ns:
                old = entry.last_mtime_ns
                entry.last_mtime_ns = mtime_ns
                # `_initialized` is private SDK API but stable across the
                # versions we pin (>=1.26.0); resetting it forces a reload.
                if hasattr(entry.provider, "_initialized"):
                    entry.provider._initialized = False  # noqa: SLF001
                logger.info(
                    "MCP OAuth '%s': tokens file changed (mtime %d -> %d), "
                    "forcing reload",
                    server_name, old, mtime_ns,
                )
                return True
            return False

    # -- 401 handler (dedup'd) -----------------------------------------------

    async def handle_401(
        self,
        server_name: str,
        failed_access_token: Optional[str] = None,
    ) -> bool:
        """Handle a 401 from a tool call, deduplicated across concurrent callers.

        Returns:
            True  if a (possibly new) access token is now available — caller
                  should trigger a reconnect and retry the operation.
            False if no recovery path exists — caller should surface a
                  ``needs_reauth`` error to the model so it stops hallucinating
                  manual refresh attempts.

        Thundering-herd protection: if N concurrent tool calls hit 401 with
        the same ``failed_access_token``, only one recovery attempt fires.
        Others await the same future.
        """
        entry = self._entries.get(self._key(server_name))
        if entry is None or entry.provider is None:
            return False

        key = failed_access_token or "<unknown>"
        loop = asyncio.get_running_loop()

        async with entry.lock:
            pending = entry.pending_401.get(key)
            if pending is None:
                pending = loop.create_future()
                entry.pending_401[key] = pending

                async def _do_handle() -> None:
                    try:
                        # Step 1: Did disk change? Picks up external refresh.
                        disk_changed = await self.invalidate_if_disk_changed(
                            server_name
                        )
                        if disk_changed:
                            if not pending.done():
                                pending.set_result(True)
                            return

                        # Step 2: No disk change — if the SDK can refresh in
                        # place, let the caller retry (the httpx.Auth flow
                        # refreshes on the next request).
                        can_refresh_fn = getattr(
                            getattr(entry.provider, "context", None), "can_refresh_token", None
                        )
                        try:
                            can_refresh = bool(can_refresh_fn()) if callable(can_refresh_fn) else False
                        except Exception:
                            can_refresh = False
                        if not pending.done():
                            pending.set_result(can_refresh)
                    except Exception as exc:  # pragma: no cover — defensive
                        logger.warning(
                            "MCP OAuth '%s': 401 handler failed: %s",
                            server_name, exc,
                        )
                        if not pending.done():
                            pending.set_result(False)
                    finally:
                        entry.pending_401.pop(key, None)

                task = asyncio.create_task(_do_handle())
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)

        try:
            return await pending
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "MCP OAuth '%s': awaiting 401 handler failed: %s",
                server_name, exc,
            )
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_MANAGER: Optional[MCPOAuthManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> MCPOAuthManager:
    """Return the process-wide :class:`MCPOAuthManager` singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = MCPOAuthManager()
        return _MANAGER


def reset_manager_for_tests() -> None:
    """Test-only helper: drop the singleton so fixtures start clean."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
