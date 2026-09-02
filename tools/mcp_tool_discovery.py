"""Connecting and discovery for tools.mcp_tool: per-server connect cooldown, connect /
lazy-start / recycled-stdio wake-up, ``register_mcp_servers`` / ``discover_mcp_tools``
and the status / probe public API. Split from tools/mcp_tool.py; origin state
(``_servers``, ``_lock``, the loop, patchable helpers) is read through ``_core`` so
``mock.patch("tools.mcp_tool.X")`` keeps working."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")


def _record_connect_failure(server_name: str) -> None:
    """Stamp a geometric, capped retry cooldown after a failed connect (under ``_lock``)."""
    n = _core._server_connect_failures.get(server_name, 0) + 1
    _core._server_connect_failures[server_name] = n
    backoff = min(
        _core._CONNECT_RETRY_BASE_BACKOFF_SEC * (2 ** (n - 1)),
        _core._CONNECT_RETRY_MAX_BACKOFF_SEC,
    )
    _core._server_connect_retry_after[server_name] = time.monotonic() + backoff


def _clear_connect_failure(server_name: str) -> None:
    """Clear the connect-cooldown state after a successful connection."""
    _core._server_connect_failures.pop(server_name, None)
    _core._server_connect_retry_after.pop(server_name, None)


def _connect_cooldown_active(server_name: str) -> bool:
    """Return True if ``server_name`` is still within its retry cooldown."""
    deadline = _core._server_connect_retry_after.get(server_name)
    return deadline is not None and time.monotonic() < deadline


async def _connect_server(name: str, config: dict) -> _core.MCPServerTask:
    """Create an MCPServerTask, start it and return once ready.

    Tear it down with ``server.shutdown()`` on the same loop. Raises on bad
    config, missing HTTP support, or connect/initialize failure.
    """
    server = _core.MCPServerTask(name)
    claim = _core._connect_server_claim.get()
    claim_token = None
    if claim is not None:
        claim(server)
        # The run task copies this context; the claim is for this attempt
        # only, so don't retain the discovery closure for the server's life.
        claim_token = _core._connect_server_claim.set(None)
    try:
        await server.start(config)
    except asyncio.CancelledError:
        # start() already reaps server._task; a shutdown() here could
        # swallow the cancellation.
        raise
    except BaseException:
        # Discovery owns claimed tasks (recoverable park vs terminal failure);
        # standalone probes have no revival owner and must reap locally.
        if claim is None:
            try:
                await server.shutdown()
            except Exception as shutdown_exc:  # noqa: BLE001 -- best-effort reap, don't mask the real error
                logger.debug(
                    "MCP server '%s' shutdown during orphan-reap failed: %s",
                    name, shutdown_exc,
                )
        raise
    finally:
        if claim_token is not None:
            _core._connect_server_claim.reset(claim_token)
    return server


def _request_lazy_reconnect(server_name: str, server: _core.MCPServerTask) -> bool:
    """Wake a recycled stdio server and wait briefly for a fresh session."""
    if not server._is_recycled_stdio():
        return False

    with _core._lock:
        loop = _core._mcp_loop
    if loop is None or not loop.is_running():
        return False

    def _wake() -> None:
        server._ready.clear()
        server._reconnect_event.set()

    loop.call_soon_threadsafe(_wake)

    async def _await_ready() -> bool:
        deadline = time.monotonic() + _core._RECYCLED_RECONNECT_TIMEOUT
        while time.monotonic() < deadline:
            if server.session is not None and server._ready.is_set():
                return True
            await asyncio.sleep(0.05)
        return False

    try:
        return bool(_core._run_on_mcp_loop(_await_ready, timeout=_core._RECYCLED_RECONNECT_TIMEOUT))
    except Exception as exc:
        logger.warning(
            "MCP server '%s': lazy reconnect after stdio recycle failed: %s",
            server_name, exc,
        )
        return False


def _resolve_server_lazy(name: str, config: dict) -> bool:
    """True when ``mcp_servers.<name>.lazy`` defers connect to first tool use (default off)."""
    return _core._parse_boolish(config.get("lazy", False), default=False)


def _ensure_lazy_server_connected(server_name: str) -> bool:
    """Connect a lazily-registered server on demand (sync; blocks the caller).

    Honours the connect cooldown and the ``_server_connecting`` dedup set and
    routes through ``_discover_and_register_server`` so park/recycle/cooldown
    bookkeeping stays in one place. True when a live session exists after.
    """
    with _core._lock:
        server = _core._servers.get(server_name)
        if server is not None and server.session is not None:
            return True
        config = _core._lazy_server_configs.get(server_name)
        if not config:
            return False
        if _core._connect_cooldown_active(server_name):
            return False
        if server_name in _core._server_connecting:
            return False
        _core._server_connecting.add(server_name)
        _core._server_connect_errors.pop(server_name, None)

    logger.info("MCP server '%s': lazy start on first use", server_name)
    _core._ensure_mcp_loop()
    connect_timeout = config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT)

    async def _connect():
        return await _core._discover_and_register_server(server_name, config)

    try:
        _core._run_on_mcp_loop(_connect, timeout=float(connect_timeout) + 30.0)
    except BaseException as exc:
        message = _core._format_connect_error(exc)
        with _core._lock:
            _core._server_connecting.discard(server_name)
            _core._server_connect_errors[server_name] = message
            _core._record_connect_failure(server_name)
        logger.warning(
            "Lazy MCP connect failed for '%s': %s", server_name, message,
        )
        return False

    with _core._lock:
        _core._server_connecting.discard(server_name)
        _core._clear_connect_failure(server_name)
        _core._lazy_server_configs.pop(server_name, None)
        stale_fingerprint = _core._lazy_server_fingerprints.pop(server_name, None)
        cached_names = _core._lazy_server_tool_names.pop(server_name, None) or []
        server = _core._servers.get(server_name)
        live_names = set(
            getattr(server, "_registered_tool_names", []) or []
        )
    # The cached manifest may advertise tools the live server no longer
    # serves; deregister those phantoms.
    phantom_names = [n for n in cached_names if n not in live_names]
    if phantom_names:
        from tools.registry import registry

        for tool_name in phantom_names:
            registry.deregister(tool_name, scope=_core._server_registry_scope(server_name))
            _core._forget_mcp_tool_server(tool_name)
        logger.info(
            "MCP server '%s': deregistered %d phantom cached tool(s) not "
            "served live (stale schema-cache fingerprint %s): %s",
            server_name, len(phantom_names), stale_fingerprint,
            ", ".join(phantom_names),
        )
    return server is not None and server.session is not None


def _get_connected_server_for_call(server_name: str) -> Optional[_core.MCPServerTask]:
    """Return a connected server; the single first-use connect point for lazy
    servers and the wake-up point for recycled stdio ones."""
    with _core._lock:
        server = _core._servers.get(server_name)
        is_lazy = server_name in _core._lazy_server_configs
    if is_lazy and (server is None or server.session is None):
        _core._ensure_lazy_server_connected(server_name)
        with _core._lock:
            server = _core._servers.get(server_name)
        return server
    if server is not None and server.session is None and server._is_recycled_stdio():
        _core._request_lazy_reconnect(server_name, server)
        with _core._lock:
            server = _core._servers.get(server_name)
    return server


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """Connect one server, register its tools; return the registered names."""
    connect_timeout = config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT)
    # The claim callback runs inside _connect_server while this frame is
    # suspended; a list append avoids a nonlocal rebind.
    claimed: List[_core.MCPServerTask] = []

    def _claim_server(created: _core.MCPServerTask) -> None:
        claimed.append(created)

    claim_token = _core._connect_server_claim.set(_claim_server)
    try:
        server = await asyncio.wait_for(
            _core._connect_server(name, config),
            timeout=connect_timeout,
        )
    except BaseException:
        server = claimed[0] if claimed else None
        task = server._task if server is not None else None
        task_cancelling = (
            task.cancelling()
            if task is not None and hasattr(task, "cancelling")
            else 0
        )
        if (
            server is not None
            and server._error is not None
            and task is not None
            and not task.done()
            and not task_cancelling
        ):
            # Recoverable park: the run task stays alive to self-probe, so
            # adopt it for shutdown/revival.
            with _core._lock:
                _core._servers[name] = server
                _core._server_scope_keys[name] = _core._mcp_registry_scope()
        elif server is not None:
            await server.shutdown()
        raise
    finally:
        _core._connect_server_claim.reset(claim_token)

    with _core._lock:
        _core._server_connecting.discard(name)
        _core._server_connect_errors.pop(name, None)
        _core._servers[name] = server
        _core._server_scope_keys[name] = _core._mcp_registry_scope()

    registered_names = _core._register_server_tools(name, server, config)
    server._registered_tool_names = list(registered_names)

    transport_type = "HTTP" if "url" in config else "stdio"
    logger.info(
        "MCP server '%s' (%s): registered %d tool(s): %s",
        name, transport_type, len(registered_names),
        ", ".join(registered_names),
    )
    return registered_names


def register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    """Connect the given ``{name: config}`` servers and register their tools.

    Idempotent for connected names; ``enabled: false`` servers are skipped
    without disconnecting existing sessions. Returns every registered MCP
    tool name.
    """
    if not _core._ensure_mcp_sdk():
        logger.debug("MCP SDK not available -- skipping explicit MCP registration")
        return []

    servers = _core._filter_suspicious_mcp_servers(servers)
    if not servers:
        logger.debug("No explicit MCP servers provided")
        return []

    # Candidates: enabled, not connected, not connecting (dedups concurrent
    # discovery entry points), not lazily registered, not in backoff.
    with _core._lock:
        connecting = set(_core._server_connecting)
        new_servers = {
            k: v
            for k, v in servers.items()
            if k not in _core._servers
            and k not in connecting
            and k not in _core._lazy_server_configs
            and _core._parse_boolish(v.get("enabled", True), default=True)
            and not _core._connect_cooldown_active(k)
        }
        # Known servers without a live session are parked or mid-reconnect;
        # their tools are deregistered so nothing else can nudge them.
        stale_cached = [
            _core._servers[k]
            for k in servers
            if k in _core._servers and getattr(_core._servers[k], "session", None) is None
        ]
        _core._server_connecting.update(new_servers)
        for srv_name in new_servers:
            _core._server_connect_errors.pop(srv_name, None)
        # Track which servers opt-in to parallel tool calls (idempotent).
        for srv_name, srv_cfg in servers.items():
            if _core._parse_boolish(srv_cfg.get("supports_parallel_tool_calls", False), default=False):
                _core._parallel_safe_servers.add(srv_name)
            else:
                _core._parallel_safe_servers.discard(srv_name)

    for srv in stale_cached:
        _core._signal_reconnect(srv)

    if not new_servers:
        return _core._existing_tool_names()

    # ``lazy: true`` servers with a valid schema-cache entry register from
    # cache without connecting; a missing/stale entry falls back to eager.
    eager_servers: Dict[str, dict] = dict(new_servers)
    lazy_registered = 0
    lazy_server_count = 0
    try:
        from tools.mcp_schema_cache import config_fingerprint, get_cached_entry
    except Exception:  # pragma: no cover - cache module missing
        config_fingerprint = None  # type: ignore[assignment]
        get_cached_entry = None  # type: ignore[assignment]
    if config_fingerprint is not None and get_cached_entry is not None:
        for name, cfg in new_servers.items():
            if not _core._resolve_server_lazy(name, cfg):
                continue
            entry = get_cached_entry(name, config_fingerprint(cfg))
            if not entry:
                continue
            with _core._lock:
                _core._server_connecting.discard(name)
            try:
                names = _core._register_from_cache_sync(name, cfg, entry)
            except Exception as exc:
                logger.warning(
                    "Failed lazy MCP registration for '%s': %s", name, exc,
                )
                with _core._lock:
                    _core._server_connecting.add(name)
                continue
            eager_servers.pop(name, None)
            lazy_registered += len(names)
            lazy_server_count += 1
    new_servers = eager_servers

    if not new_servers:
        if lazy_registered:
            logger.info(
                "MCP: registered %d lazy tool(s) from schema cache "
                "(no processes spawned)",
                lazy_registered,
            )
        return _core._existing_tool_names()

    _core._ensure_mcp_loop()

    async def _discover_all():
        server_names = list(new_servers.keys())
        results = await asyncio.gather(
            *(_core._discover_and_register_server(name, cfg) for name, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(server_names, results):
            if isinstance(result, BaseException):
                command = new_servers.get(name, {}).get("command")
                message = _core._format_connect_error(result)
                with _core._lock:
                    _core._server_connecting.discard(name)
                    _core._server_connect_errors[name] = message
                    _core._record_connect_failure(name)
                logger.warning(
                    "Failed to connect to MCP server '%s'%s: %s",
                    name,
                    f" (command={command})" if command else "",
                    message,
                )
            else:
                with _core._lock:
                    _core._server_connecting.discard(name)
                    _core._server_connect_errors.pop(name, None)
                    _core._clear_connect_failure(name)

    # Clear a stale interrupt flag (executor threads are reused) so a prior
    # session's interrupt cannot cancel this discovery pass.
    from tools.interrupt import is_interrupted as _is_interrupted, set_interrupt as _set_interrupt
    _was_interrupted = _is_interrupted()
    if _was_interrupted:
        _set_interrupt(False)
    try:
        _core._run_on_mcp_loop(_discover_all, timeout=120)
    except (TimeoutError, InterruptedError) as _e:
        # Entries stranded in _server_connecting would block future
        # reconnect attempts.
        with _core._lock:
            stale = [n for n in new_servers if n in _core._server_connecting]
            if stale:
                logger.warning(
                    "MCP discovery %s while %d server(s) were still "
                    "connecting; clearing stale connecting set: %s",
                    "timed out" if isinstance(_e, TimeoutError) else "interrupted",
                    len(stale),
                    ", ".join(stale),
                )
                _core._server_connecting.difference_update(stale)
                for _sn in stale:
                    _core._server_connect_errors.setdefault(
                        _sn,
                        f"Connection attempt {'timed out' if isinstance(_e, TimeoutError) else 'interrupted'} during discovery",
                    )
        raise
    finally:
        if _was_interrupted:
            _set_interrupt(True)

    with _core._lock:
        connected = [
            n
            for n in new_servers
            if n in _core._servers and n not in _core._server_connect_errors
        ]
        new_tool_count = sum(
            len(getattr(_core._servers[n], "_registered_tool_names", []))
            for n in connected
        )
    failed = len(new_servers) - len(connected)
    new_tool_count += lazy_registered
    connected_count = len(connected) + lazy_server_count
    if new_tool_count or failed:
        summary = f"MCP: registered {new_tool_count} tool(s) from {connected_count} server(s)"
        if failed:
            summary += f" ({failed} failed)"
        logger.info(summary)

    return _core._existing_tool_names()


def discover_mcp_tools() -> List[str]:
    """Entry point: load config, connect servers, register tools.

    Safe without the ``mcp`` package (returns []). Idempotent: only servers
    missing from a previous call are retried. Returns all MCP tool names.
    """
    servers = _core._load_mcp_config()
    if not servers:
        logger.debug("No MCP servers configured")
        return []

    # SDK import deferred to here so a config without servers never pays it.
    if not _core._ensure_mcp_sdk():
        logger.debug("MCP SDK not available -- skipping MCP tool discovery")
        return []

    # Cross-process guard: a lock loser waits for the holder, then runs its
    # own discovery; if locking is unavailable or the wait expires, run
    # unguarded (fail-soft).
    cookie = _core._try_acquire_mcp_discovery_lock()
    if cookie is None:
        logger.debug(
            "Another process holds MCP discovery lock -- retrying with backoff"
        )
        for _ in range(_core._MCP_DISCOVERY_LOCK_MAX_RETRIES):
            time.sleep(_core._MCP_DISCOVERY_LOCK_RETRY_DELAY_S)
            cookie = _core._try_acquire_mcp_discovery_lock()
            if cookie is not None:
                break

        if cookie is None:
            logger.warning(
                "MCP discovery lock still held after %d retries -- "
                "running discovery unguarded",
                _core._MCP_DISCOVERY_LOCK_MAX_RETRIES,
            )
        elif cookie is not _core._LOCK_UNAVAILABLE:
            logger.debug("Retry succeeded -- acquired MCP discovery lock")

    try:
        with _core._lock:
            connecting = set(_core._server_connecting)
            new_server_names = [
                name
                for name, cfg in servers.items()
                if name not in _core._servers
                and name not in connecting
                and _core._parse_boolish(cfg.get("enabled", True), default=True)
            ]

        tool_names = _core.register_mcp_servers(servers)
        if not new_server_names:
            return tool_names

        with _core._lock:
            connected_server_names = [
                name
                for name in new_server_names
                if name in _core._servers and name not in _core._server_connect_errors
            ]
            new_tool_count = sum(
                len(getattr(_core._servers[name], "_registered_tool_names", []))
                for name in connected_server_names
            )

        failed_count = len(new_server_names) - len(connected_server_names)
        if new_tool_count or failed_count:
            summary = f"  MCP: {new_tool_count} tool(s) from {len(connected_server_names)} server(s)"
            if failed_count:
                summary += f" ({failed_count} failed)"
            logger.info(summary)

        return tool_names

    finally:
        if cookie not in (None, _core._LOCK_UNAVAILABLE):
            cookie.release()


def is_mcp_tool_parallel_safe(tool_name: str) -> bool:
    """True when the tool's server opted into ``supports_parallel_tool_calls``.

    Uses the provenance captured at registration, never the (ambiguous)
    ``mcp__{server}__{tool}`` string shape.
    """
    if not tool_name.startswith(_core.MCP_TOOL_NAME_PREFIX):
        return False
    with _core._lock:
        server_name = _core._mcp_tool_server_names.get(tool_name)
        return bool(server_name and server_name in _core._parallel_safe_servers)


def get_mcp_status() -> List[dict]:
    """Status of every configured server for banner/TUI display.

    Each dict has name, transport, tools, connected, disabled, status (one of
    connected / disabled / connecting / failed / configured) and, for failed,
    error. ``enabled: false`` is reported as disabled, not failed.
    """
    configured = _core._load_mcp_config()
    if not configured:
        return []

    with _core._lock:
        active_servers = dict(_core._servers)
        connecting = set(_core._server_connecting)
        connect_errors = dict(_core._server_connect_errors)

    def _entry(name: str, transport: str, status: str, **extra) -> dict:
        return {
            "name": name,
            "transport": transport,
            "tools": 0,
            "connected": False,
            "disabled": status == "disabled",
            "status": status,
            **extra,
        }

    result: List[dict] = []
    for name, cfg in configured.items():
        transport = cfg.get("transport", "http") if "url" in cfg else "stdio"
        enabled = _core._parse_boolish(cfg.get("enabled", True), default=True)
        server = active_servers.get(name)
        if server and server.session is not None:
            entry = _entry(name, transport, "connected", connected=True)
            entry["tools"] = (
                len(server._registered_tool_names)
                if hasattr(server, "_registered_tool_names")
                else len(server._tools)
            )
            if server._sampling:
                entry["sampling"] = dict(server._sampling.metrics)
        elif not enabled:
            entry = _entry(name, transport, "disabled")
        elif name in connecting:
            entry = _entry(name, transport, "connecting")
        elif name in connect_errors:
            entry = _entry(name, transport, "failed", error=connect_errors[name])
        else:
            entry = _entry(name, transport, "configured")
        result.append(entry)

    return result


def probe_mcp_server_tools() -> Dict[str, List[tuple]]:
    """Connect to each enabled server, list ``(tool_name, description)`` and
    disconnect, without registering anything. Failed servers are omitted."""
    if not _core._ensure_mcp_sdk():
        return {}

    servers_config = _core._load_mcp_config()
    if not servers_config:
        return {}

    enabled = {
        k: v for k, v in servers_config.items()
        if _core._parse_boolish(v.get("enabled", True), default=True)
    }
    if not enabled:
        return {}

    _core._ensure_mcp_loop()

    result: Dict[str, List[tuple]] = {}
    probed_servers: List[_core.MCPServerTask] = []

    async def _probe_all():
        names = list(enabled.keys())
        coros = []
        for name, cfg in enabled.items():
            ct = cfg.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT)
            coros.append(asyncio.wait_for(_core._connect_server(name, cfg), timeout=ct))

        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, Exception):
                logger.debug("Probe: failed to connect to '%s': %s", name, outcome)
                continue
            probed_servers.append(outcome)
            tools = []
            for t in outcome._tools:
                desc = getattr(t, "description", "") or ""
                tools.append((t.name, desc))
            result[name] = tools

        await asyncio.gather(
            *(s.shutdown() for s in probed_servers),
            return_exceptions=True,
        )

    try:
        _core._run_on_mcp_loop(_probe_all, timeout=120)
    except Exception as exc:
        logger.debug("MCP probe failed: %s", exc)
    finally:
        _core._stop_mcp_loop_if_idle()

    return result


def has_registered_mcp_tools() -> bool:
    """True if any MCP server has registered tools (cheap; no registry walk).

    Checks registered TOOLS, not connected servers, so the per-turn refresh
    hook stays idle for zero-tool servers.
    """
    with _core._lock:
        return bool(_core._mcp_tool_server_names)


def get_registered_mcp_server_names() -> set:
    """Server names that registered at least one tool (the live, filtered
    signal — not merely what config.yaml lists)."""
    with _core._lock:
        return set(_core._mcp_tool_server_names.values())
