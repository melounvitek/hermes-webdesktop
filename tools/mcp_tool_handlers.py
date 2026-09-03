"""Registry-facing sync handlers for MCP tools and utility tools (resources/prompts), plus the per-call recovery ladder: trust gating, circuit breaker, auth (401) refresh, session-expired reconnect and dead-stdio respawn retry. Split from tools/mcp_tool.py."""

import logging
import asyncio
import contextvars
import inspect
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from tools.registry import tool_error
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import _exc_str, _sanitize_error, mcp_field, _core
from tools.mcp_tool_content import _MCP_HARD_RESULT_CAP_CHARS, _cache_mcp_audio_block, _cache_mcp_image_block, _render_mcp_resource_block, _strip_reserved_meta_keys, _truncate_mcp_text_result
from tools.mcp_tool_errors import _is_session_expired_error

logger = logging.getLogger("tools.mcp_tool")


# --------------------------------------------------------------- pre-call gates

def _trust_gate_check(server_name: str, tool_name: str) -> Optional[str]:
    """Approval gate for write-capable tools on ``trust: untrusted`` servers.
    None to proceed, else a ``tool_error``. Fail-closed: approval-system errors block."""
    trust = _core._server_trust_levels.get(server_name, _core._TRUST_FULL)
    if trust != _core._TRUST_UNTRUSTED or _core._tool_read_only_hints.get(server_name, {}).get(tool_name) is True:
        return None
    # Lazy import: tools.approval routes the prompt to whichever surface owns the session.
    try:
        from tools.approval import request_elicitation_consent

        answer = request_elicitation_consent(
            f"MCP tool '{tool_name}' on UNTRUSTED server '{server_name}' wants to run. This "
            f"tool is write-capable (no readOnlyHint=true annotation) and may modify external state.",
            f"Server '{server_name}' is configured 'trust: untrusted'. "
            f"Approve to run '{tool_name}' once, or deny to block it.",
            surface=f"mcp-trust/{server_name}",
        )
    except Exception as exc:
        logger.error("MCP trust gate: approval check failed for %s.%s: %s", server_name, tool_name, exc, exc_info=True)
        return tool_error(f"MCP tool '{tool_name}' on untrusted server '{server_name}' was blocked: the approval "
                          f"system was unavailable (fail-closed).")
    if answer == "accept":
        return None
    logger.info("MCP trust gate: user %s '%s' on untrusted server '%s'",
                "cancelled" if answer == "cancel" else "denied", tool_name, server_name)
    return tool_error(f"The user did not approve running write-capable MCP tool '{tool_name}' on untrusted server "
                      f"'{server_name}'. The command was NOT run. Do not retry without explicit user direction.")


def _check_circuit_breaker(server_name: str) -> Optional[str]:
    """Open-breaker error, or None when calls may proceed. After the cooldown the breaker is
    half-open: the next call probes; success resets, failure re-bumps and re-arms the cooldown."""
    failures = _core._server_error_counts.get(server_name, 0)
    if failures < _core._CIRCUIT_BREAKER_THRESHOLD:
        return None
    age = time.monotonic() - _core._server_breaker_opened_at.get(server_name, 0.0)
    if age >= _core._CIRCUIT_BREAKER_COOLDOWN_SEC:
        return None
    remaining = max(1, int(_core._CIRCUIT_BREAKER_COOLDOWN_SEC - age))
    return tool_error(f"MCP server '{server_name}' is unreachable after {failures} consecutive failures. "
                      f"Auto-retry available in ~{remaining}s. Do NOT retry this tool yet — use alternative "
                      f"approaches or ask the user to check the MCP server.")


def _acquire_call_server(server_name: str, tool_timeout: float):
    """``(server, None)`` when a call may be dispatched, else ``(None, error)``.
    No session: a reconnect may be completing (fresh session swaps in asynchronously), so wait
    briefly before charging a breaker strike. Still down → reconnecting or parked (e.g. dead
    stdio child); probing a dead transport would re-arm the breaker forever, so ask the server
    task to rebuild and return a clean "reconnecting" error — the breaker resets once the
    fresh session initializes."""
    not_connected = tool_error(f"MCP server '{server_name}' is not connected")
    server = _core._get_connected_server_for_call(server_name)
    if not server:
        _core._bump_server_error(server_name)
        return None, not_connected
    if server.session or _core._wait_for_server_session_ready(server, timeout=min(5.0, float(tool_timeout or 5.0))):
        return server, None
    _core._bump_server_error(server_name)
    if _core._signal_reconnect(server):
        return None, tool_error(f"MCP server '{server_name}' transport is down; reconnect requested. Do NOT retry this "
                                f"tool immediately — give it a few seconds to come back.")
    return None, not_connected


# ------------------------------------------------------------ breaker bookkeeping

def _result_is_error(result) -> bool:
    """True only for a JSON payload carrying an ``error`` key (non-JSON = success)."""
    try:
        return "error" in json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False


def _record_call_outcome(server_name: str, result) -> Any:
    """Breaker bookkeeping: an error payload from the tool itself still counts as a strike."""
    if _result_is_error(result):
        _core._bump_server_error(server_name)
    else:
        _core._reset_server_error(server_name)
    return result


def _strike(server_name: str, message: str, **extra) -> str:
    """Breaker strike + the ``tool_error`` payload for *message*."""
    _core._bump_server_error(server_name)
    return tool_error(message, **extra)


def _lookup_reconnectable_server(server_name: str, require_loop: bool = False):
    """The registered server object when it can be signalled to reconnect, else None.
    With *require_loop*, also None unless the MCP loop is running (nothing to wait on)."""
    with _core._lock:
        srv = _core._servers.get(server_name)
    if srv is None or not hasattr(srv, "_reconnect_event"):
        return None
    if require_loop and not _mcp_loop_running():
        return None
    return srv


def _mcp_loop_running() -> bool:
    loop = _core._mcp_loop
    return loop is not None and loop.is_running()


def _retry_once(server_name: str, retry_call, op_description: str, what: str):
    """Re-run ``retry_call`` after a recovery step. Returns the result (closing the breaker)
    when it is not an error payload; None when the retry raised or errored (caller falls through)."""
    try:
        result = retry_call()
    except Exception as retry_exc:
        logger.warning("MCP %s/%s retry after %s failed: %s", server_name, op_description, what, retry_exc)
        return None
    if _result_is_error(result):
        return None
    _core._reset_server_error(server_name)
    return result


# --------------------------------------------------------------- recovery ladder

def _handle_auth_error_and_retry(server_name: str, exc: BaseException, retry_call, op_description: str):
    """OAuth recovery + one retry; None when *exc* is not an auth error.
    ``MCPOAuthManager.handle_401`` decides whether recovery is viable; if so, signal
    ``_reconnect_event`` so the server task rebuilds the session with fresh credentials, wait
    for ready, retry once. Any failure returns the structured ``needs_reauth`` error so the
    model stops trying to refresh manually."""
    if not _core._is_auth_error(exc):
        return None
    from tools.mcp_oauth_manager import get_manager
    manager = get_manager()

    async def _recover():
        return await manager.handle_401(server_name, None)

    try:
        recovered = _core._run_on_mcp_loop(_recover, timeout=10)
    except Exception as rec_exc:
        logger.warning("MCP OAuth '%s': recovery attempt failed: %s", server_name, rec_exc)
        recovered = False
    if recovered:
        srv = _lookup_reconnectable_server(server_name)
        # OAuth recovery + reconnect is independent evidence the server is viable, so close the
        # breaker here, not only on retry success — otherwise a failing retry would leave it
        # pinned open forever. A broken server re-trips it via _bump_server_error on the retry.
        if srv is not None and _core._signal_reconnect_and_wait(
                server_name, srv, op_description=f"{op_description} after OAuth recovery", timeout=15):
            _core._reset_server_error(server_name)
        result = _retry_once(server_name, retry_call, op_description, "auth recovery")
        if result is not None:
            return result
    return _strike(
        server_name,
        f"MCP server '{server_name}' requires re-authentication. Run `hermes mcp login "
        f"{server_name}` (or delete the tokens file under ~/.hermes/mcp-tokens/ and restart). Do "
        f"NOT retry this tool — ask the user to re-authenticate.",
        needs_reauth=True, server=server_name)


def _handle_session_expired_and_retry(server_name: str, exc: BaseException, retry_call, op_description: str):
    """Transport reconnect + one retry on session expiry; None to fall through (not
    session-expired, no server record / loop, reconnect did not ready in time, retry failed).
    Skips ``handle_401`` — the token is still valid, only the server-side session is stale."""
    if not _is_session_expired_error(exc):
        return None
    srv = _lookup_reconnectable_server(server_name, require_loop=True)
    if srv is None:
        return None
    logger.info("MCP server '%s': %s failed with session-expired error (%s); "
                "signalling transport reconnect and retrying once.", server_name, op_description, exc)
    if not _core._signal_reconnect_and_wait(server_name, srv, op_description=op_description, timeout=15):
        logger.warning("MCP server '%s': reconnect did not ready within 15s after "
                       "session-expired error; falling through to error response.", server_name)
        return None
    return _retry_once(server_name, retry_call, op_description, "session reconnect")


class _StdioChildExited(RuntimeError):
    """A server's stdio subprocess was gone when (or while) a call ran.
    Deliberately NOT a TimeoutError: nothing timed out — the child was already dead
    (typically a gateway restart killed it under a live agent session)."""


def _handle_stdio_child_exited_and_retry(server_name: str, exc: Exception, retry_call, op_description: str):
    """Respawn a dead stdio child and retry once; None if not our error.
    Never spawns anything itself: it sets ``_reconnect_event`` once and waits for the server
    task to publish a fresh session, so spawn frequency stays governed by ``run()``'s
    rapid-drop budget. Single-shot: a child that dies again immediately reports and stops."""
    if not isinstance(exc, _StdioChildExited):
        return None
    reconnected = False
    srv = _lookup_reconnectable_server(server_name)
    if srv is not None:
        logger.info("MCP server '%s': %s found the stdio subprocess dead (%s); "
                    "respawning and retrying once.", server_name, op_description, exc)
        if _mcp_loop_running():
            reconnected = _core._signal_reconnect_and_wait(
                server_name, srv, op_description=op_description, timeout=_core._STDIO_RESPAWN_WAIT_SEC)
        else:
            # No MCP loop to wait on (non-async adapters, tests) — still request the respawn
            # so the next call lands on a live transport.
            _core._signal_reconnect(srv)

    if not reconnected:
        return _strike(
            server_name,
            f"MCP server '{server_name}' stdio subprocess had exited (this is not a timeout — the "
            f"call never reached the server). A respawn was requested but no fresh session came "
            f"back within {_core._STDIO_RESPAWN_WAIT_SEC:.0f}s. Wait a few seconds before retrying; "
            f"if it keeps failing the server is not starting and needs the user.")
    try:
        return _record_call_outcome(server_name, retry_call())
    except _StdioChildExited as retry_exc:
        # Died again right after respawn: broken server, not a restart artifact. Stop here —
        # run()'s budget takes it to the park.
        logger.warning("MCP server '%s': %s stdio subprocess exited again right "
                       "after respawn (%s); not retrying further.", server_name, op_description, retry_exc)
        return _strike(
            server_name,
            f"MCP server '{server_name}' respawned its stdio subprocess and it exited again "
            f"immediately. The server is not starting cleanly — do NOT retry this tool; ask the "
            f"user to check the server's command and its stderr log.")
    except Exception as retry_exc:
        logger.warning("MCP %s/%s retry after stdio respawn failed: %s", server_name, op_description, retry_exc)
        return _strike(server_name, _sanitize_error(
            f"MCP call failed after respawning the stdio subprocess for '{server_name}': "
            f"{type(retry_exc).__name__}: {_exc_str(retry_exc)}"))


def _interrupted_call_result() -> str:
    """Standardized JSON error for a user-interrupted MCP tool call."""
    return tool_error("MCP call interrupted: user sent a new message")


def _invoke_with_recovery(server_name: str, call_once: Callable[[], str], op: str,
                          recoverers, on_final_failure: Callable[[BaseException], None],
                          record_outcome: bool = False) -> str:
    """Run ``call_once``, walking the recovery ladder on failure. Each recoverer
    ``(server_name, exc, retry_call, op) -> Optional[str]`` returns None when the exception is
    not its kind; order matters: dead stdio child → auth → session expiry. Unrecovered
    exceptions go through ``on_final_failure`` (breaker strike / logging) and become the generic
    call-failed error. ``record_outcome`` applies breaker bookkeeping to the FIRST attempt only;
    retries own their bookkeeping inside the recoverers."""
    try:
        result = call_once()
        return _record_call_outcome(server_name, result) if record_outcome else result
    except InterruptedError:
        return _interrupted_call_result()
    except Exception as exc:
        for recover in recoverers:
            recovered = recover(server_name, exc, call_once, op)
            if recovered is not None:
                return recovered
        on_final_failure(exc)
        return tool_error(_sanitize_error(f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"))


# ------------------------------------------------------------- the RPC itself

def _mark_server_call_started(server: Any) -> None:
    """Record a user-visible MCP operation when the server supports it."""
    mark_tool_call = getattr(server, "mark_tool_call", None)
    if callable(mark_tool_call):
        mark_tool_call()


@asynccontextmanager
async def _track_inflight_rpc(server: Any, server_name: str, op: str):
    """Register the running RPC on the server so teardown can fail it fast.
    A deliberate reconnect/shutdown teardown (``_fail_inflight_calls`` sets ``_reconnecting``
    first) turns the cancel into a clean retryable RuntimeError; external cancels (caller
    timeout, user interrupt) propagate unchanged. Doubles without ``_inflight_tasks`` skip tracking."""
    inflight = getattr(server, "_inflight_tasks", None)
    task = asyncio.current_task()
    tracked = task is not None and inflight is not None
    if tracked:
        inflight.add(task)
    try:
        yield
    except asyncio.CancelledError:
        if getattr(server, "_reconnecting", False):
            raise RuntimeError(f"MCP {op} on '{server_name}' was aborted by a reconnect "
                               f"teardown; retry the request on the rebuilt session") from None
        raise
    finally:
        if tracked:
            inflight.discard(task)


async def _call_tool_racing_stdio_death(server, server_name: str, tool_name: str, args: dict):
    """``session.call_tool`` that fails fast when the stdio child is/gets dead.
    Pre-call: an already-dead child must not hold the slot for the full tool timeout
    (``server.session`` is stale so the transport-down path never fired). Mid-call: race the
    RPC against ``_watch_stdio_children``. Both raise :class:`_StdioChildExited` for the
    respawn-and-retry path, which owns the reconnect signal (nothing clears ``server.session``).
    callable()/``is True`` checks because MagicMock attributes return truthy Mocks."""
    _stdio_dead = getattr(server, "_stdio_children_dead", None)
    if callable(_stdio_dead) and _stdio_dead() is True:
        raise _StdioChildExited(f"MCP stdio subprocess for '{server_name}' had already exited when the call was dispatched")
    _call_coro = server.session.call_tool(tool_name, arguments=args)
    _watch_children = getattr(server, "_watch_stdio_children", None)
    if not (_watch_children is not None and inspect.iscoroutinefunction(_watch_children) and asyncio.iscoroutine(_call_coro)):
        # Stubbed sessions return a non-awaitable, or there is no child-watcher to race: plain await.
        return await _call_coro if asyncio.iscoroutine(_call_coro) else _call_coro

    rpc_task = asyncio.ensure_future(_call_coro)
    watch_task = asyncio.ensure_future(_watch_children())
    try:
        done, _pending = await asyncio.wait({rpc_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        if watch_task in done and not rpc_task.done():
            rpc_task.cancel()
            raise _StdioChildExited(f"MCP stdio subprocess for '{server_name}' exited mid-call")
        return await rpc_task
    finally:
        watch_task.cancel()
        if not rpc_task.done():
            rpc_task.cancel()
        await asyncio.gather(rpc_task, watch_task, return_exceptions=True)


# ---------------------------------------------------------- result rendering

def _error_result_text(result) -> str:
    """Concatenated text of an ``isError`` result's blocks (EmbeddedResource error payloads
    carry text under ``.resource.text``)."""
    texts = (getattr(b, "text", None) or getattr(getattr(b, "resource", None), "text", None) for b in (result.content or []))
    return "".join(str(t) for t in texts if t)


def _render_content_blocks(result, server_name: str) -> str:
    """Text blocks pass through; image/audio blocks are cached via the gateway image-cache so
    they flow out as MEDIA: tags; resource blocks (PDFs, docs, ...) are materialized rather
    than silently dropped."""
    parts: List[str] = []
    for block in (result.content or []):
        if getattr(block, "text", None):
            parts.append(strip_unicode_tags(block.text))
            continue
        rendered = _cache_mcp_image_block(block) or _cache_mcp_audio_block(block) or _render_mcp_resource_block(block, server_name)
        if rendered:
            parts.append(rendered)
            continue
        # Benign empty renders log at debug; warn only for unknown shapes.
        block_type = getattr(block, "type", None) or type(block).__name__
        if block_type in {"text", "resource", "audio", "image"}:
            logger.debug("MCP %s: content block type %r rendered empty", server_name, block_type)
        else:
            logger.warning("MCP %s: dropping unsupported content block type %r", server_name, block_type)
    # Hard-cap pathological payloads; ordinary large results pass to spillover.
    return _truncate_mcp_text_result("\n".join(parts))


def _capped_structured_content(result):
    """``structuredContent`` (or None); over the hard cap it degrades to the head+tail
    truncated JSON string (multi-MB JSON flood guard)."""
    structured = mcp_field(result, "structured_content", "structuredContent")
    if structured is None:
        return None
    try:
        as_json = json.dumps(structured, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return structured
    return _truncate_mcp_text_result(as_json) if len(as_json) > _MCP_HARD_RESULT_CAP_CHARS else structured


def _render_call_tool_result(result, server_name: str) -> str:
    """Pure: ``CallToolResult`` → the handler's JSON string. ``content`` is the primary
    (model-oriented) payload; ``structuredContent`` supplements it (or becomes ``result`` when
    there is no text). Server-level ``_meta`` is surfaced minus protocol-reserved keys.
    ``.is_error`` is ``.isError`` before mcp 2.0."""
    if mcp_field(result, "is_error", "isError", False):
        return tool_error(_sanitize_error(_truncate_mcp_text_result(_error_result_text(result) or "MCP tool returned an error")))

    text_result = _render_content_blocks(result, server_name)
    structured = _capped_structured_content(result)
    meta = _strip_reserved_meta_keys(mcp_field(result, "meta", "meta"))
    if structured is None and meta is None:
        return json.dumps({"result": text_result}, ensure_ascii=False)
    # Key order is part of the output: "result" leads when there is text, otherwise "_meta"
    # precedes the (empty) "result".
    payload: Dict[str, Any] = {"result": text_result} if text_result else {}
    if structured is not None:
        payload["structuredContent" if text_result else "result"] = structured
    if meta is not None:
        payload["_meta"] = meta
    payload.setdefault("result", text_result)
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        # Non-serializable metadata: drop the extras, keep the call.
        return json.dumps({"result": text_result}, ensure_ascii=False)


# ------------------------------------------------------------------- handlers

def _make_tool_handler(server_name: str, tool_name: str, tool_timeout: float):
    """Sync registry handler (``handler(args_dict, **kwargs) -> str``) calling an MCP tool via the background loop."""
    op = f"tools/call {tool_name}"

    def _handler(args: dict, **kwargs) -> str:
        # Security boundary: untrusted-server write tools need approval before ANY transport
        # work, including the lazy first-use spawn below.
        error = _trust_gate_check(server_name, tool_name) or _check_circuit_breaker(server_name)
        if error is not None:
            return error
        server, error = _acquire_call_server(server_name, tool_timeout)
        if server is None:
            return error

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock, _track_inflight_rpc(server, server_name, op):
                # Snapshot contextvars so an elicitation callback (fired on the MCP recv loop,
                # which doesn't inherit them) can replay them for gateway platform / session routing.
                server._pending_call_context = contextvars.copy_context()
                try:
                    result = await _call_tool_racing_stdio_death(server, server_name, tool_name, args)
                finally:
                    server._pending_call_context = None
            # Round-trip completed: transport is healthy even if the tool returned isError.
            # Clear the rapid-drop budget.
            _mark_proven = getattr(server, "_mark_session_proven", None)
            if _mark_proven is not None:
                _mark_proven()
            return _render_call_tool_result(result, server_name)

        def _on_failure(exc):
            _core._bump_server_error(server_name)
            logger.error("MCP tool %s/%s call failed: %s", server_name, tool_name, exc)

        return _invoke_with_recovery(
            server_name, lambda: _core._run_on_mcp_loop(_call, timeout=tool_timeout), op,
            (_handle_stdio_child_exited_and_retry, _handle_auth_error_and_retry, _handle_session_expired_and_retry),
            _on_failure, record_outcome=True,
        )

    return _handler


def _make_utility_handler(server_name: str, tool_timeout: float, op: str, log_label: str,
                          rpc, render, required: Optional[str] = None):
    """Shared shape of the four utility handlers (resources/prompts): ``rpc(session, args)``
    is awaited under ``_rpc_lock``, ``render(result, server_name)`` builds the JSON-able payload,
    ``required`` names a parameter validated before any transport work. The wrapper owns the
    connected check and the auth / session-expired recovery ladder."""

    def _handler(args: dict, **kwargs) -> str:
        server = _core._get_connected_server_for_call(server_name)
        if not server or not server.session:
            return tool_error(f"MCP server '{server_name}' is not connected")
        if required and not args.get(required):
            return tool_error(f"Missing required parameter '{required}'")

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock:
                result = await rpc(server.session, args)
            return json.dumps(render(result, server_name), ensure_ascii=False)

        def _on_failure(exc):
            logger.error("MCP %s/%s failed: %s", server_name, log_label, exc)

        return _invoke_with_recovery(
            server_name, lambda: _core._run_on_mcp_loop(_call, timeout=tool_timeout), op,
            (_handle_auth_error_and_retry, _handle_session_expired_and_retry), _on_failure,
        )

    return _handler


def _pick(obj, *specs) -> dict:
    """``{out_key: value}`` for each ``(out_key, attr[, truthy])`` present on *obj*.
    ``hasattr`` (not a default) so SDK models and test stubs behave alike; with ``truthy``
    the field is also skipped when falsy. Output key order = spec order."""
    entry = {}
    for out_key, attr, *truthy in specs:
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        if value or not (truthy and truthy[0]):
            entry[out_key] = value
    return entry


def _render_resource_list(all_resources, server_name: str) -> dict:
    resources = []
    for r in all_resources:
        entry = _pick(r, ("uri", "uri"), ("name", "name"), ("description", "description", True))
        if "uri" in entry:
            entry["uri"] = str(entry["uri"])
        # Key stays camelCase — this is the tool's own JSON output shape.
        _mime = mcp_field(r, "mime_type", "mimeType")
        if _mime:
            entry["mimeType"] = _mime
        resources.append(entry)
    return {"resources": resources}


def _render_read_resource(result, server_name: str) -> dict:
    parts: List[str] = []
    for block in getattr(result, "contents", []):
        if getattr(block, "text", None) is not None:
            parts.append(strip_unicode_tags(block.text))
        elif getattr(block, "blob", None) is not None:
            # Materialize binary contents into the document cache (same contract as
            # EmbeddedResource blocks in tool results).
            rendered = _render_mcp_resource_block(SimpleNamespace(type="resource", resource=block), server_name)
            parts.append(rendered or f"[binary data, {len(block.blob)} bytes]")
    return {"result": "\n".join(parts)}


def _render_prompt_list(all_prompts, server_name: str) -> dict:
    prompts = []
    for p in all_prompts:
        entry = _pick(p, ("name", "name"), ("description", "description", True))
        if getattr(p, "arguments", None):
            entry["arguments"] = [
                {"name": a.name, **_pick(a, ("description", "description", True), ("required", "required"))}
                for a in p.arguments
            ]
        prompts.append(entry)
    return {"prompts": prompts}


def _render_get_prompt(result, server_name: str) -> dict:
    messages = []
    for msg in getattr(result, "messages", []):
        entry = _pick(msg, ("role", "role"))
        if hasattr(msg, "content"):
            content = msg.content
            entry["content"] = strip_unicode_tags(content.text if hasattr(content, "text") else str(content))
        messages.append(entry)
    resp = {"messages": messages}
    if getattr(result, "description", None):
        resp["description"] = result.description
    return resp


def _make_list_resources_handler(server_name: str, tool_timeout: float):
    """Sync handler that lists resources from an MCP server."""
    return _make_utility_handler(server_name, tool_timeout, "resources/list", "list_resources",
                                 lambda session, args: _core._paginate_full_list(session.list_resources, "resources", server_name),
                                 _render_resource_list)


def _make_read_resource_handler(server_name: str, tool_timeout: float):
    """Sync handler that reads a resource by URI from an MCP server."""
    return _make_utility_handler(server_name, tool_timeout, "resources/read", "read_resource",
                                 lambda session, args: session.read_resource(args["uri"]), _render_read_resource, required="uri")


def _make_list_prompts_handler(server_name: str, tool_timeout: float):
    """Sync handler that lists prompts from an MCP server."""
    return _make_utility_handler(server_name, tool_timeout, "prompts/list", "list_prompts",
                                 lambda session, args: _core._paginate_full_list(session.list_prompts, "prompts", server_name),
                                 _render_prompt_list)


def _make_get_prompt_handler(server_name: str, tool_timeout: float):
    """Sync handler that gets a prompt by name from an MCP server."""
    return _make_utility_handler(server_name, tool_timeout, "prompts/get", "get_prompt",
                                 lambda session, args: session.get_prompt(args["name"], arguments=args.get("arguments", {})),
                                 _render_get_prompt, required="name")


def _make_check_fn(server_name: str):
    """Check function that verifies the MCP connection is alive."""

    def _check() -> bool:
        with _core._lock:
            server = _core._servers.get(server_name)
            if server is not None and (server.session is not None or server._is_recycled_stdio()):
                return True
            # Lazy (schema-cache registered) servers count as available: the first real
            # call spawns/connects them.
            return server_name in _core._lazy_server_configs

    return _check
