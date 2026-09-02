"""Transport bring-up for MCPServerTask: stdio spawn (OSV preflight, watchdog wrap, child PID ledger), Streamable HTTP / SSE connect (preflight, identity header, client certs, OAuth), protocol negotiation and initial tool discovery. Split from tools/mcp_tool.py."""

import logging
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set
from tools.mcp_tool_config import _wrap_command_with_watchdog
from tools.mcp_tool_errors import NonMcpEndpointError, _apply_identity_header, _handshake_rejected_as_modern, _make_redirect_header_stripper, _resolve_client_cert
from tools.mcp_tool_lifecycle import _filter_mcp_children, _orphan_stdio_pid_servers, _orphan_stdio_pids, _stdio_pgids, _stdio_pids
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")

# JSON-RPC ``initialize`` body used by the content-type preflight POST.
_PROBE_INITIALIZE_BODY = (
    '{"jsonrpc":"2.0","id":"_probe",'
    '"method":"initialize",'
    '"params":{"protocolVersion":"2025-03-26",'
    '"capabilities":{},'
    '"clientInfo":{"name":"hermes-probe",'
    '"version":"0.1"}}}'
)


def _content_type_base(resp) -> str:
    """``content-type`` header of *resp* without parameters, lowercased."""
    return resp.headers.get("content-type", "").split(";")[0].strip().lower()


def _is_2xx(resp) -> bool:
    return 200 <= resp.status_code < 300


def _capture_pgids(pids: Set[int]) -> Dict[int, int]:
    """pgid per live pid. Captured while the child is alive — getpgid fails once
    it exits, and the sweep needs it to reach reparented descendants."""
    pgids: Dict[int, int] = {}
    for pid in pids:
        try:
            pgids[pid] = os.getpgid(pid)
        except (AttributeError, ProcessLookupError, OSError):
            # AttributeError: Windows; ProcessLookupError: already exited.
            pass
    return pgids


def _pgroup_alive(pgid: Optional[int]) -> bool:
    """Signal 0 to the group succeeds iff any member is alive (POSIX only)."""
    _killpg = getattr(os, "killpg", None)
    if pgid is None or _killpg is None:
        return False
    try:
        _killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


async def _osv_malware_preflight(server_name: str, command: str, args: list) -> None:
    """OSV malware preflight: off-loop (blocking HTTPS) with a wall-clock bound so
    a stalled handshake can't freeze discovery; fail-open on timeout. Must run
    against the REAL command/args — the watchdog wrap rewrites argv to the
    supervisor, which would turn the check into a no-op."""
    from tools.osv_check import check_package_for_malware
    try:
        malware_error = await asyncio.wait_for(
            asyncio.to_thread(check_package_for_malware, command, args),
            timeout=_core._OSV_MALWARE_CHECK_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "MCP server '%s': OSV malware preflight timed out after %.0fs "
            "(network slow/unreachable) — proceeding without the check.",
            server_name, _core._OSV_MALWARE_CHECK_TIMEOUT_S,
        )
        return
    if malware_error:
        raise ValueError(f"MCP server '{server_name}': {malware_error}")


class MCPServerTransportMixin:
    """Methods of :class:`tools.mcp_tool.MCPServerTask` (mixed in; relies on its attributes)."""

    __slots__ = ()

    def _advertises_tools(self) -> bool:
        """Whether the server advertises the ``tools`` capability.

        Prompt-/resource-only servers omit it, and ``tools/list`` against them
        raises ``MCPError(-32601)``. True when no capability info was captured
        (legacy fallback: keep the old always-call-list_tools behavior).
        """
        init_result = self.initialize_result
        caps = getattr(init_result, "capabilities", None) if init_result is not None else None
        if caps is None:
            return True
        return getattr(caps, "tools", None) is not None

    def _session_kwargs(self) -> dict:
        """ClientSession kwargs: sampling, elicitation, notification + logging callbacks."""
        kwargs = self._sampling.session_kwargs() if self._sampling else {}
        if self._elicitation:
            kwargs.update(self._elicitation.session_kwargs())
        if _core._MCP_NOTIFICATION_TYPES and _core._MCP_MESSAGE_HANDLER_SUPPORTED:
            kwargs["message_handler"] = self._make_message_handler()
        if _core._MCP_LOGGING_CALLBACK_SUPPORTED:
            kwargs["logging_callback"] = self._make_logging_callback()
        return kwargs

    async def _negotiate_session(self, session, connect_timeout: float):
        """Negotiate the protocol era (``initialize`` vs ``server/discover``) and return its result.

        Per-server ``protocol`` key: ``auto`` (default) tries the legacy handshake
        FIRST and falls back to ``server/discover`` only when the server signals
        modern-only (-32022 / initialize -32601) — the reverse of the SDK's
        discover-first mode, on purpose: zero extra round-trips for the handshake-era
        servers that dominate today. ``stateless`` probes discover first (one legacy
        retry on error); ``legacy`` is handshake only, no fallback. Both result
        types expose ``.capabilities``, so downstream gates work on either.
        A handshake TIMEOUT never triggers a fallback — it propagates.
        """
        def initialize():
            return asyncio.wait_for(session.initialize(), timeout=connect_timeout)

        def discover():
            return asyncio.wait_for(session.discover(), timeout=connect_timeout)

        mode = str((self._config or {}).get("protocol", "auto")).lower().strip()
        if mode in ("stateless", "modern", "2026-07-28"):
            try:
                return await discover()
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                logger.info(
                    "MCP server '%s': server/discover rejected (%s) despite "
                    "protocol=%s — falling back to the legacy handshake",
                    self.name, exc, mode,
                )
                return await initialize()
        if mode in ("legacy", "handshake"):
            return await initialize()
        if mode != "auto":
            logger.warning(
                "MCP server '%s': unknown protocol=%r — treating as 'auto' "
                "(valid: auto, stateless, legacy)", self.name, mode,
            )
        try:
            return await initialize()
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            # mcp 1.x has no server/discover client — nothing to fall back to.
            if not _handshake_rejected_as_modern(exc) or not hasattr(session, "discover"):
                raise
            logger.info(
                "MCP server '%s': legacy handshake rejected (%s) — "
                "retrying via server/discover (2026-07-28 stateless server)",
                self.name, exc,
            )
            return await discover()

    async def _serve_session(self, session, connect_timeout: float,
                             label: str = "", mark_lifecycle: bool = False) -> str:
        """Handshake, discover, publish readiness, then serve until a lifecycle event.

        Clears stale breaker state from a prior outage, but leaves the session
        UNPROVEN: a completed handshake is not proof of health (flapping
        transports handshake fine and drop moments later); only keepalive or
        tool-call success clears the reconnect budget.
        """
        self.initialize_result = await self._negotiate_session(session, connect_timeout)
        self.session = session
        if mark_lifecycle:
            self._mark_lifecycle_started()
        await self._discover_tools()
        self._ready.set()
        self._ever_connected = True
        _core._reset_server_error(self.name)
        self._session_proven = False
        reason = await self._wait_for_lifecycle_event()
        if label and reason == "reconnect":
            logger.info("MCP server '%s': reconnect requested — tearing down %s session", self.name, label)
        return reason

    async def _serve_transport(self, transport_cm, label: str, connect_timeout: float) -> str:
        """Open *transport_cm*, wrap its streams in a ClientSession and serve it.

        Streams are unpacked positionally: mcp 1.x yields ``(read, write,
        get_session_id)``, 2.x yields ``(read, write)``. A transport TaskGroup
        drop maps to ``"reconnect"`` instead of backoff/park.
        """
        try:
            async with transport_cm as _streams:
                async with _core.ClientSession(
                    _streams[0], _streams[1], **self._session_kwargs()
                ) as session:
                    return await self._serve_session(session, connect_timeout, label)
        except BaseExceptionGroup as _eg:
            return self._reconnect_or_reraise_group(_eg)

    # ------------------------------------------------------------------ stdio

    def _resolve_stdio_config(self, config: dict):
        """``(command, args, safe_env)`` from config, with the command resolved against the safe env."""
        command = config.get("command")
        if not command:
            raise ValueError(f"MCP server '{self.name}' has no 'command' in config")
        args = config.get("args", [])
        safe_env = _core._build_safe_env(config.get("env"))
        command, safe_env = _core._resolve_stdio_command(command, safe_env)
        return command, args, safe_env

    def _track_spawned_children(self, new_pids: Set[int]) -> None:
        """Ledger the freshly spawned stdio children (pids, pgids, machine spawn ledger)."""
        new_pgids = _capture_pgids(new_pids)
        with _core._lock:
            for _pid in new_pids:
                _stdio_pids[_pid] = self.name
            _stdio_pgids.update(new_pgids)
        # Machine spawn ledger so startup sweeps can reap orphans after an
        # unclean parent exit. Best-effort — never break startup.
        for _pid in new_pids:
            try:
                from hermes_cli.process_identity import register_child

                register_child(_pid, "mcp-helper")
            except Exception:
                logger.debug("spawn-ledger register_child failed for MCP helper pid %s", _pid, exc_info=True)

    def _release_spawned_children(self, new_pids: Set[int]) -> None:
        """Drop the ledger entries; any child (or its pgroup) still alive means SDK
        teardown failed (common on cancel mid-way on Linux, where setsid() children
        escape the cgroup) — mark it orphaned for the next cleanup sweep."""
        from gateway.status import _pid_exists
        with _core._lock:
            for _pid in new_pids:
                _stdio_pids.pop(_pid, None)
            for pid in new_pids:
                # ``os.kill(pid, 0)`` is NOT a no-op on Windows; use the
                # cross-platform check. Child exited but descendants may remain
                # in its pgroup.
                if _pid_exists(pid) or _pgroup_alive(_stdio_pgids.get(pid)):
                    _orphan_stdio_pids.add(pid)
                    _orphan_stdio_pid_servers[pid] = self.name
                else:
                    # Nothing to reap — drop the pgid so PID reuse can't
                    # surface stale pgroup state later.
                    _stdio_pgids.pop(pid, None)

    async def _run_stdio(self, config: dict):
        """Run the server using stdio transport."""
        if config.get("identity_header") is not None:
            # No headers on stdio — warn so a copy-pasted HTTP block doesn't mislead.
            logger.warning(
                "MCP server '%s': identity_header is only supported on "
                "HTTP/SSE transports — ignored for stdio servers", self.name,
            )
        if not _core._ensure_mcp_sdk():
            raise ImportError(
                f"MCP server '{self.name}' requires the 'mcp' Python SDK, but "
                "it is not installed. Run `hermes setup` to install MCP support, "
                "then retry."
            )
        command, args, safe_env = self._resolve_stdio_config(config)
        await _osv_malware_preflight(self.name, command, args)
        # Parent-death watchdog: an ungraceful Hermes exit (kill -9, crash) can't
        # leave the child and its descendants running. Clean-exit reaping is
        # unchanged. POSIX-only (process groups); no-op elsewhere. AFTER the OSV
        # preflight so the check inspects the real package.
        command, args = _wrap_command_with_watchdog(command, args)
        server_params = _core.StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
            cwd=config.get("cwd"),
            # Windows pipes can split non-UTF-8 bytes at chunk boundaries;
            # substitute U+FFFD instead of raising UnicodeDecodeError.
            encoding_error_handler="replace",
        )
        session_kwargs = self._session_kwargs()

        # Reap orphans from prior failed attempts before spawning, else each
        # reconnect retry piles up zombie pairs. Unscoped on purpose (also reaps
        # orphans of servers that never reconnect). Worker thread: the reaper
        # blocks up to 2s (SIGTERM → wait → SIGKILL) and would stall the loop.
        await asyncio.to_thread(_core._kill_orphaned_mcp_children)

        # Snapshot child PIDs before spawning so the new one can be identified.
        pids_before = _core._snapshot_child_pids()
        new_pids: set = set()
        # Route subprocess stderr to ~/.hermes/logs/mcp-stderr.log so server
        # banners don't land on the user's TTY and corrupt the TUI.
        _core._write_stderr_log_header(self.name)
        _errlog = _core._get_mcp_stderr_log()
        try:
            async with _core.stdio_client(server_params, errlog=_errlog) as (
                read_stream,
                write_stream,
            ):
                # Capture the new PID for force-kill cleanup, filtering non-MCP
                # children (slash_worker, LSP servers) that race into the snapshot
                # window: they share the TUI parent's pgid, so leaking them into
                # _stdio_pgids makes the shutdown killpg() kill the TUI itself.
                new_pids = _filter_mcp_children(_core._snapshot_child_pids() - pids_before)
                if new_pids:
                    self._track_spawned_children(new_pids)
                # Tracked on the connection so in-flight calls fail fast when
                # the subprocess dies.
                self._stdio_child_pids = set(new_pids)
                async with _core.ClientSession(
                    read_stream, write_stream, **session_kwargs
                ) as session:
                    # Bound the handshake: ``connect_timeout`` only bounds the
                    # caller's ``.result()`` wait, not this coroutine. A server that
                    # never answers ``initialize`` would otherwise hang here forever,
                    # the ``finally`` below would never run, and the child + pipes
                    # would leak on every retry until EMFILE.
                    connect_timeout = float(config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT))
                    return await self._serve_session(session, connect_timeout, mark_lifecycle=True)
        finally:
            # Runs on clean exit, exceptions AND cancellation.
            if new_pids:
                self._release_spawned_children(new_pids)

    # ------------------------------------------------------------------- HTTP

    async def _preflight_content_type(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        ssl_verify: bool = True,
        client_cert=None,
        timeout: float = 5.0,
    ) -> None:
        """Probe *url* for an MCP-shaped response before the SDK connects.

        A URL pointing at a plain web page makes the SDK sit out the full
        ``connect_timeout`` before an opaque ``CancelledError``; this raises
        :class:`NonMcpEndpointError` within ``timeout`` instead. Allow-list based:
        only a 2xx with a definite non-MCP content type is rejected, and only
        after a JSON-RPC ``initialize`` POST also fails to look like MCP (some
        servers serve a UI on GET but speak Streamable HTTP via POST). Missing
        content type, non-2xx, or transport errors pass silently — the real
        handshake stays the source of truth. Uses its own httpx client OUTSIDE
        the SDK's anyio task group so the error isn't wrapped in an ExceptionGroup.
        """
        try:
            import httpx as _httpx
        except ImportError:
            return  # No httpx → skip probe; SDK import would have failed first.

        client_kwargs: dict = {
            "verify": ssl_verify,
            "follow_redirects": True,
            "timeout": _httpx.Timeout(timeout),
        }
        if client_cert is not None:
            client_kwargs["cert"] = client_cert

        probe_headers = dict(headers) if headers else {}
        try:
            async with _httpx.AsyncClient(**client_kwargs) as client:
                # HEAD is cheapest; fall back to GET on 405/501.
                resp = await client.head(url, headers=probe_headers)
                if resp.status_code in (405, 501):
                    resp = await client.get(url, headers=probe_headers)
                # Non-MCP content type on HEAD/GET: try a JSON-RPC POST before
                # rejecting, so POST-only servers aren't false positives.
                ct = _content_type_base(resp)
                if ct and ct not in self._MCP_CONTENT_TYPES and _is_2xx(resp):
                    post_resp = await client.post(
                        url,
                        headers={
                            **probe_headers,
                            "Content-Type": "application/json",
                            "Accept": "application/json, text/event-stream",
                        },
                        content=_PROBE_INITIALIZE_BODY,
                    )
                    if _is_2xx(post_resp) and _content_type_base(post_resp) in self._MCP_CONTENT_TYPES:
                        resp = post_resp
        except _httpx.HTTPError:
            return  # DNS/connect/timeout/transport error — let the SDK try.

        # Only judge 2xx: a 4xx/5xx may be an auth challenge or transient error
        # the real handshake handles correctly. No content type advertised →
        # don't second-guess the SDK.
        if not _is_2xx(resp):
            return
        ct_base = _content_type_base(resp)
        if not ct_base or ct_base in self._MCP_CONTENT_TYPES:
            return
        raise NonMcpEndpointError(
            f"MCP server '{self.name}' at {url} returned Content-Type "
            f"'{ct_base}', not an MCP response (expected one of: "
            f"{', '.join(self._MCP_CONTENT_TYPES)}). The URL most likely "
            "points at a web page rather than an MCP endpoint — check it "
            "resolves to a Streamable HTTP / SSE endpoint "
            "(e.g. https://host/mcp, not https://host/)."
        )

    def _reconnect_or_reraise_group(self, eg: BaseExceptionGroup) -> str:
        """Map an SDK transport TaskGroup failure to a clean ``"reconnect"``.

        HTTP/SSE stream pumps run in an anyio TaskGroup, so a transient stream
        drop escapes as a ``BaseExceptionGroup``. Unmapped, ``run()`` would back
        off and eventually park the server for 300s (deregistering its tools)
        over a sub-second glitch; ``"reconnect"`` rebuilds the session at once.
        Re-raise when it is not a transient drop: shutdown in progress
        (``_shutdown_event`` is set before the task is cancelled), the group
        carries KeyboardInterrupt/SystemExit or a real CancelledError (must
        propagate), or no live session was reached this attempt (``_ready``
        unset — connect failures must go through backoff, not hot-loop).
        """
        if (
            self._shutdown_event.is_set()
            or eg.split((KeyboardInterrupt, SystemExit))[0] is not None
            or eg.split(asyncio.CancelledError)[0] is not None
            or not self._ready.is_set()
        ):
            raise eg
        logger.debug(
            "MCP server '%s': transport TaskGroup exited after a live session "
            "(%r) — reconnecting immediately instead of backing off",
            self.name, eg,
        )
        return "reconnect"

    def _build_oauth_auth(self, url: str, config: dict):
        """OAuth 2.1 PKCE via the central MCPOAuthManager so one provider is reused
        across reconnects and shared with config-time CLI paths. On setup failure
        (e.g. non-interactive without cached tokens) re-raise so only this server
        is reported failed."""
        if self._auth_type != "oauth":
            return None
        try:
            from tools.mcp_oauth_manager import get_manager
            return get_manager().get_or_build_provider(self.name, url, config.get("oauth"))
        except Exception as exc:
            logger.warning("MCP OAuth setup failed for '%s': %s", self.name, exc)
            raise

    def _sse_transport(self, url: str, headers: dict, connect_timeout: float,
                       ssl_verify, client_cert, oauth_auth, strict_cfg_headers: bool):
        """``sse_client`` context manager for ``transport: sse`` entries."""
        if strict_cfg_headers:
            # Fail closed: SSE cannot enforce the redirect boundary.
            raise ValueError(
                f"MCP server '{self.name}': strict_redirect_headers is "
                "not supported on the SSE transport."
            )
        if _core.sse_client is None:
            raise ImportError(
                f"MCP server '{self.name}' requires SSE transport but "
                "mcp.client.sse.sse_client is not available. "
                "Upgrade the mcp package to get SSE support."
            )
        # sse_read_timeout bounds the gap between SSE events. SSE servers
        # commonly idle for minutes, so tool_timeout (60s) would drop the
        # stream; 300s matches the Streamable HTTP read timeout.
        sse_kwargs: dict = {
            "url": url,
            "headers": headers or None,
            "timeout": float(connect_timeout),
            "sse_read_timeout": 300.0,
        }
        if oauth_auth is not None:
            # Forward OAuth to sse_client, else OAuth SSE servers 401 silently.
            sse_kwargs["auth"] = oauth_auth
        if client_cert is not None or ssl_verify is not True:
            # sse_client has no verify/cert kwargs: wrap the SDK defaults
            # (follow_redirects=True) in an httpx_client_factory, forwarding
            # the SDK's (headers, auth, timeout) and layering TLS on top. The
            # client MUST come from the SDK's own httpx module (httpx2 on
            # mcp >= 2.0) — see sdk_httpx().
            _httpx_mod = _core.sdk_httpx()

            def _mcp_http_client_factory(headers=None, timeout=None, auth=None):
                kwargs: dict = {
                    "follow_redirects": True,
                    "verify": ssl_verify,
                    "timeout": timeout if timeout is not None else _httpx_mod.Timeout(30.0, read=300.0),
                }
                if headers is not None:
                    kwargs["headers"] = headers
                if auth is not None:
                    kwargs["auth"] = auth
                if client_cert is not None:
                    kwargs["cert"] = client_cert
                return _httpx_mod.AsyncClient(**kwargs)

            sse_kwargs["httpx_client_factory"] = _mcp_http_client_factory
        return _core.sse_client(**sse_kwargs)

    def _streamable_http_transport(self, url: str, headers: dict, connect_timeout: float,
                                   ssl_verify, client_cert, oauth_auth,
                                   strict_cfg_headers: bool, configured_header_names: set):
        """Streamable HTTP context manager (mcp >= 1.24.0: caller-owned httpx client)."""
        if not _core._MCP_NEW_HTTP:
            return self._legacy_http_transport(url, headers, connect_timeout, ssl_verify,
                                               oauth_auth, strict_cfg_headers)
        # Build an explicit AsyncClient matching the SDK's create_mcp_http_client
        # defaults. It MUST come from the SDK's httpx module (httpx2 on mcp >= 2.0)
        # since the SDK sends its own Request objects through it — see sdk_httpx().
        httpx = _core.sdk_httpx()
        _strip_auth_on_cross_origin_redirect = _make_redirect_header_stripper(
            httpx.URL(url),
            strict=strict_cfg_headers,
            configured_header_names=configured_header_names,
        )
        client_kwargs: dict = {
            "follow_redirects": True,
            "timeout": httpx.Timeout(float(connect_timeout), read=300.0),
            "verify": ssl_verify,
            "event_hooks": {"response": [_strip_auth_on_cross_origin_redirect]},
        }
        if headers:
            client_kwargs["headers"] = headers
        if oauth_auth is not None:
            client_kwargs["auth"] = oauth_auth
        if client_cert is not None:
            client_kwargs["cert"] = client_cert

        @asynccontextmanager
        async def _owned_client_streams():
            # Caller owns the client lifecycle — the SDK skips cleanup when
            # http_client is provided.
            async with httpx.AsyncClient(**client_kwargs) as http_client:
                async with _core.streamable_http_client(url, http_client=http_client) as streams:
                    yield streams

        return _owned_client_streams()

    def _legacy_http_transport(self, url: str, headers: dict, connect_timeout: float,
                               ssl_verify, oauth_auth, strict_cfg_headers: bool):
        """Deprecated API (mcp < 1.24.0): the SDK owns the httpx client."""
        if strict_cfg_headers:
            # Fail closed: without an owned client we cannot hook redirects,
            # so the cross-origin header boundary cannot be enforced.
            raise ImportError(
                f"MCP server '{self.name}' requires mcp >= 1.24.0 to "
                "enforce the portable redirect-header boundary "
                "(strict_redirect_headers). Upgrade the mcp package."
            )
        http_kwargs: dict = {"headers": headers, "timeout": float(connect_timeout), "verify": ssl_verify}
        if oauth_auth is not None:
            http_kwargs["auth"] = oauth_auth
        return _core.streamablehttp_client(url, **http_kwargs)

    async def _run_http(self, config: dict):
        """Run the server using HTTP/StreamableHTTP (or SSE) transport."""
        _core._ensure_mcp_sdk()
        if not _core._MCP_HTTP_AVAILABLE:
            raise ImportError(
                f"MCP server '{self.name}' requires HTTP transport but "
                "mcp.client.streamable_http is not available. "
                "Upgrade the mcp package to get HTTP support."
            )
        url = config["url"]
        headers = dict(config.get("headers") or {})
        # Portable Agent Plugins v1 (strict_redirect_headers): configured
        # headers MUST NOT follow a redirect to a different origin. Capture the
        # configured names BEFORE client-generated headers are merged in.
        strict_cfg_headers = bool(config.get("strict_redirect_headers"))
        configured_header_names = {key.lower() for key in headers}
        # Optional per-user identity header; explicit headers of the same name win.
        headers = _apply_identity_header(self.name, config, headers)
        # Some servers require MCP-Protocol-Version on the initial request; seed
        # it (case-insensitive user override wins). Seeded from the HANDSHAKE
        # version, not the latest: the body sent by ``initialize()`` speaks the
        # handshake era, and a 2026-07-28 header would route the request onto
        # the server's per-request-envelope ladder, which rejects that body.
        # The header must agree with what the body actually speaks.
        if not any(key.lower() == "mcp-protocol-version" for key in headers):
            headers["mcp-protocol-version"] = _core.LATEST_HANDSHAKE_VERSION
        connect_timeout = config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT)
        ssl_verify = config.get("ssl_verify", True)
        client_cert = _resolve_client_cert(self.name, config)
        oauth_auth = self._build_oauth_auth(url, config)

        if config.get("transport") == "sse":
            transport = self._sse_transport(url, headers, connect_timeout, ssl_verify,
                                            client_cert, oauth_auth, strict_cfg_headers)
            label = "SSE"
        else:
            transport = self._streamable_http_transport(
                url, headers, connect_timeout, ssl_verify, client_cert, oauth_auth,
                strict_cfg_headers, configured_header_names,
            )
            label = "HTTP" if _core._MCP_NEW_HTTP else "legacy HTTP"
        return await self._serve_transport(transport, label, float(connect_timeout))

    # -------------------------------------------------------------- discovery

    async def _discover_tools(self):
        """Discover tools from the connected session.

        Capability-gated: prompt-/resource-only servers raise ``MCPError(-32601)``
        on ``tools/list``, which would abort the connection — skip the call when
        ``tools`` isn't advertised.
        """
        # Fresh transport: re-probe with cheap ``ping`` in case the server gained
        # support across the reconnect.
        self._ping_unsupported = False
        if self.session is None:
            return
        if not self._advertises_tools():
            logger.info(
                "MCP server '%s': does not advertise 'tools' capability — "
                "skipping tools/list (prompts/resources remain available)",
                self.name,
            )
            self._tools = []
            self._register_discovered_tools_if_needed()
            return
        async with self._rpc_lock:
            self._list_cache_meta = {}
            self._tools = await _core._paginate_full_list(
                self.session.list_tools, "tools", self.name,
                cache_meta_out=self._list_cache_meta,
            )
        self._register_discovered_tools_if_needed()

    def _register_discovered_tools_if_needed(self) -> None:
        """Publish freshly discovered tools for a registry-owned server if none are registered.

        Initial registration normally happens in ``_discover_and_register_server``
        after ``start()``. On reconnect, outage handling may clear ``_ready`` and
        deregister stale tools; ownership via ``_servers`` authorizes publishing
        before readiness is restored so a revival never comes back with zero
        tools. A server retained after a recoverable initial failure is likewise
        owned before its first session, which authorizes its first publication.
        """
        if self._registered_tool_names:
            return
        if not self._ready.is_set():
            with _core._lock:
                if _core._servers.get(self.name) is not self:
                    return
        self._registered_tool_names = _core._register_server_tools(self.name, self, self._config)
        # A retained initial-failure server that just published tools has
        # recovered: drop its stale connect error from status surfaces.
        with _core._lock:
            if _core._servers.get(self.name) is self:
                _core._server_connect_errors.pop(self.name, None)
