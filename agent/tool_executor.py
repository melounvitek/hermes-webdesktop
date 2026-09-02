"""Tool-call execution: sequential and concurrent dispatch, extracted from AIAgent.

Functions take the parent ``AIAgent`` first; ``run_agent`` keeps thin wrappers, and
tests that patch ``run_agent._set_interrupt`` still work because we reach it via ``_ra()``.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    build_tool_label as _build_tool_label,
    get_cute_tool_message as _get_cute_tool_message_impl,
    get_tool_emoji as _get_tool_emoji,
    redact_tool_args_for_display as _redact_tool_args_for_display,
    _detect_tool_failure,
)
from agent.message_sanitization import coalesce_tool_call_id
from agent.inline_tool_executors import (
    INLINE_TOOL_EXECUTORS,
    InlineToolContext,
    emit_terminal_post_tool_call,
    tool_hook_ids,
)
from agent.tool_dispatch_helpers import (
    _NEVER_PARALLEL_TOOLS,
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,
    _plan_tool_batch_segments,
    make_tool_result_message,
)
from tools.terminal_tool import (
    get_active_env,
)
from tools.thread_context import propagate_context_to_thread
from tools.tool_result_storage import (
    maybe_persist_tool_result,
    enforce_turn_budget,
    extract_persisted_path,
)
from tools.budget_config import BudgetConfig, DEFAULT_BUDGET, budget_for_context_window

logger = logging.getLogger(__name__)


def _pairing_tool_call_id(tool_call: Any) -> str:
    """Return the canonical id used by the persisted assistant message."""
    return coalesce_tool_call_id(tool_call)


def _record_persisted_path_for_stub(agent, tool_call_id: str, function_result) -> None:
    """Record the spillover file path so a later result-reference stub can't dangle.

    Best-effort: bookkeeping never breaks tool execution.
    """
    try:
        if not isinstance(function_result, str):
            return
        path = extract_persisted_path(function_result)
        if path:
            agent._tool_guardrails.record_persisted_result(tool_call_id, path)
    except Exception as exc:
        logger.debug("persisted-path record for result stub failed: %s", exc)


def _ensure_file_checkpoint(
    agent,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
) -> None:
    """Checkpoint the same workspace path that the file tool will mutate."""
    file_path = function_args.get("path", "")
    if not file_path:
        return

    # File tools resolve relative paths against the task's live cwd (differs from the
    # process cwd in Docker); resolve the same way before locating the project root.
    from tools.file_tools import _resolve_path_for_task

    resolved_path = _resolve_path_for_task(file_path, effective_task_id or "default")
    work_dir = agent._checkpoint_mgr.get_working_dir_for_path(str(resolved_path))
    agent._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")


def _budget_for_agent(agent) -> BudgetConfig:
    """Resolve a tool-result BudgetConfig scaled to the agent's context window.

    Small-context models get a proportional budget so one large result can't overflow
    the request (#23767); falls back to the default when context length is unknown.
    """
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        # budget_for_context_window(None), not DEFAULT_BUDGET, so the MCP threshold
        # override still applies when the context length isn't resolvable.
        return budget_for_context_window(int(ctx) if ctx else None)
    except Exception:
        return DEFAULT_BUDGET

# Maximum number of concurrent worker threads for parallel tool execution.
# Mirrors the constant in ``run_agent`` for tests/imports that look here.
_MAX_TOOL_WORKERS = 8
_DEFAULT_IMAGE_PARALLEL_REQUESTS = 4
# Generous ceiling for slow-but-valid tool work (large page fetches, slow
# remote backends) so the batch guard does not preempt a legitimate attempt.
_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0
# Start-order gate wait bound: long enough for an approval round-trip, short enough
# that one wedged dispatch cannot starve the batch.
_START_ORDER_GATE_TIMEOUT_S = 120.0
# Fallback authorization-gate lock bound; the effective bound derives from
# approvals.timeout (see _authorization_gate_lock_timeout) since overstaying it means wedged.
_AUTHORIZATION_GATE_LOCK_TIMEOUT_S = 360.0


def _authorization_gate_lock_timeout() -> float:
    """Bound for the authorization serialization lock: approval timeout + margin.

    Delegates to ``tools.approval.human_wait_ceiling`` so the two bounds can't drift:
    never break serialization while an approval prompt is answerable, but never let a
    wedged holder park other workers forever (#79719). Resolved once per batch.
    """
    try:
        from tools.approval import human_wait_ceiling

        # Safety-capped so a huge approvals.timeout can't overflow Lock.acquire (#83220);
        # deliberately NOT min()'d with the fallback so the gate never gives up early (#79719).
        return human_wait_ceiling()
    except Exception:
        return _AUTHORIZATION_GATE_LOCK_TIMEOUT_S


class _BatchAbandoned(BaseException):
    """Raised inside a worker when the batch was abandoned before dispatch.

    BaseException so ``except Exception`` handlers in the middleware chain can't swallow it.
    """


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
    """Parse model-emitted arguments without repairing or coercing them."""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        return arguments, None
    return {}, json.dumps(
        {
            "error": "Invalid tool arguments",
            "message": (
                "Tool arguments must be a valid JSON object; tool was not executed."
            ),
        },
        ensure_ascii=False,
    )


def _resolve_concurrent_tool_timeout() -> float | None:
    """Resolve the per-batch concurrent tool deadline via the unified resolver (#85125).

    ``timeouts.tools.concurrent_batch`` wins; ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` is the
    legacy bridge; ``0``/negative disables the bound.
    """
    from agent.deadline import resolve_timeout

    return resolve_timeout(
        "tools.concurrent_batch",
        default=_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S,
        env_var="HERMES_CONCURRENT_TOOL_TIMEOUT_S",
    )


def _flush_session_db_after_tool_progress(
    agent,
    messages: list,
    *,
    stage: str,
) -> bool:
    """Flush tool-call progress to the session DB before projecting it to any UI.

    Tool side effects can kill/restart the process before turn-end persistence runs.
    """
    try:
        persisted = agent._flush_messages_to_session_db(messages) is not False
        if not persisted:
            agent._incremental_persistence_failed = True
            # Flush recorded any classified cause at the catch site; only default
            # to 'unknown' when nothing more specific exists.
            if getattr(agent, "_last_persistence_error_cause", None) is None:
                agent._last_persistence_error_cause = "unknown"
        return persisted
    except Exception as exc:
        agent._incremental_persistence_failed = True
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning("Incremental tool-call persistence failed after %s: %s", stage, exc)
        return False


def _image_generate_parallel_limit() -> int:
    """Return the configured image-generation parallelism cap (conservative default;
    backend bursts hit TTFB/rate-limit failures).
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        image_gen = cfg.get("image_gen") if isinstance(cfg, dict) else None
        value = (
            image_gen.get("max_parallel_requests")
            if isinstance(image_gen, dict)
            else None
        )
    except Exception:
        value = None

    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = _DEFAULT_IMAGE_PARALLEL_REQUESTS
    return max(1, min(limit, _MAX_TOOL_WORKERS))


def _max_workers_for_tool_batch(runnable_calls) -> int:
    """Return the worker cap for a concurrent tool batch."""
    if not runnable_calls:
        return 0
    max_workers = _MAX_TOOL_WORKERS
    if any(
        (call[2] if len(call) >= 3 else None) == "image_generate"
        for call in runnable_calls
    ):
        max_workers = min(max_workers, _image_generate_parallel_limit())
    return min(len(runnable_calls), max_workers)


def _ra():
    """Lazy reference to ``run_agent`` so patches like ``run_agent._set_interrupt`` work."""
    import run_agent
    return run_agent


def _is_interpreter_shutdown_submit_error(exc: RuntimeError) -> bool:
    """Shutdown-race predicate; delegates to ``tools.interpreter_shutdown`` so every site
    recognizes both CPython shutdown-message variants (#55924/#58720).
    """
    from tools.interpreter_shutdown import interpreter_shutting_down

    return interpreter_shutting_down(exc)


_emit_terminal_post_tool_call = emit_terminal_post_tool_call


def _cancelled_tool_result(reason: str = "user interrupt") -> str:
    return json.dumps(
        {
            "error": f"Tool execution cancelled by {reason}",
            "status": "cancelled",
        },
        ensure_ascii=False,
    )


def _emit_cancelled_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    start_time: float,
    reason: str = "user interrupt",
    error_type: str = "keyboard_interrupt",
    middleware_trace: Optional[list[dict[str, Any]]] = None,
) -> str:
    result = _cancelled_tool_result(reason)
    _emit_terminal_post_tool_call(
        agent,
        function_name=function_name,
        function_args=function_args,
        result=result,
        effective_task_id=effective_task_id,
        tool_call_id=tool_call_id,
        duration_ms=int((time.time() - start_time) * 1000),
        status="cancelled",
        error_type=error_type,
        error_message=f"Tool execution cancelled by {reason}",
        middleware_trace=list(middleware_trace or []),
    )
    return result


def _tool_search_scoped_names(agent) -> frozenset:
    """Return the deferrable tool names the session may invoke via tool_call.

    The Tool Search unwrap bypasses the bridge's scope check in
    ``model_tools.handle_function_call``, so restricted sessions are validated against
    this set. Cached on the agent; refreshed when the registry generation changes.
    """
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:
        return frozenset()

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    cache_key = (
        _registry.current_scope_key(),
        getattr(_registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    try:
        scoped_defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
        names = _ts.scoped_deferrable_names(scoped_defs)
    except Exception:
        names = frozenset()
    try:
        agent._tool_search_scope_cache = (cache_key, names)
    except Exception:
        pass
    return names


def _canonical_tool_name(function_name: str) -> str:
    """Map legacy tool-name aliases (2026-08 renames) BEFORE agent-loop dispatch."""
    from model_tools import _LEGACY_TOOL_ALIASES as _lta

    return _lta.get(function_name, function_name)


def _unwrap_tool_search_call(
    agent, function_name: str, function_args: dict, *, flatten_probe: bool = False
) -> tuple[str, dict, Optional[str]]:
    """Peel the ``tool_call`` bridge so downstream hooks see the underlying tool.

    Checkpointing, guardrails, plugin hooks and the activity feed must observe the real
    tool name, not the bridge. ``tool_call.function`` stays untouched for the transcript
    and tool_call_id pairing. The unwrap bypasses
    handle_function_call's scope check, so session toolset scope is enforced HERE.
    Returns ``(name, args, scope_block)``; ``scope_block`` is the block message when
    the underlying tool is out of scope or its args fail the deferred-schema probe
    (``flatten_probe`` collapses the probe's JSON payload to one plain string for
    callers that wrap the message in ``{"error": ...}``).
    """
    scope_block: Optional[str] = None
    try:
        from tools import tool_search as _ts
        if function_name == _ts.TOOL_CALL_NAME:
            underlying, underlying_args, err = _ts.resolve_underlying_call(function_args)
            if not err and underlying:
                if underlying in _tool_search_scoped_names(agent):
                    # Validate before unwrapping: the generic bridge hides the concrete
                    # parameter schema from provider-native tool-call validation.
                    probe_err = _ts.validate_deferred_call_args(underlying, underlying_args)
                    if probe_err is None:
                        return underlying, underlying_args, None
                    scope_block = probe_err
                    if flatten_probe:
                        try:
                            probe = json.loads(probe_err)
                            scope_block = (
                                f"{probe.get('error', '')} Parameters schema: "
                                f"{json.dumps(probe.get('parameters', {}), ensure_ascii=False)}. "
                                f"{probe.get('hint', '')}"
                            ).strip()
                        except Exception:
                            scope_block = probe_err
                else:
                    scope_block = (
                        f"'{underlying}' is not available in this session. "
                        "Use tool_search to find tools you can call."
                    )
    except Exception:
        pass
    return function_name, function_args, scope_block


@dataclass
class _ManagedToolResult:
    result: Any
    args: dict[str, Any]
    middleware_trace: list[dict[str, Any]]
    blocked: bool
    dispatched: bool


class _ToolTimeoutResult(str):
    """Marker for a synthesized sequential-tool timeout result."""


class _ToolCancelledResult(str):
    """Marker for a synthesized sequential-tool user-interrupt result.

    The terminal post_tool_call event was already emitted (status=cancelled), so a
    late-finishing abandoned worker must not report success.
    """


class _ConcurrentToolAuthorizationGate:
    """Serialize policy prompts and exclude human approval waits from batch deadlines.

    The acquire is BOUNDED: on expiry the worker prompts unserialized rather than
    starving the batch behind a wedged plugin/approval client (#79705).

    Deadline exclusion is measured at the SOURCE of the human wait
    (``tools.approval.human_wait_seconds``), NOT as gate residency: residency-based
    exclusion let a wedged plugin keep the deadline from ever firing (#79719).
    """

    def __init__(
        self,
        *,
        lock_timeout: float | None = None,
        session_key: str | None = None,
    ) -> None:
        self._serialization_lock = threading.Lock()
        self._lock_timeout = (
            _authorization_gate_lock_timeout()
            if lock_timeout is None
            else lock_timeout
        )
        self._session_key = session_key
        if self._session_key is None:
            try:
                from tools.approval import get_current_session_key

                # Snapshot on the SUBMITTING thread: excluded_seconds() is polled
                # from the batch wait loop, whose context may differ from workers'.
                self._session_key = get_current_session_key()
            except Exception:
                logger.debug(
                    "authorization gate could not snapshot the session key; "
                    "human-wait exclusion will re-resolve it at poll time",
                    exc_info=True,
                )
        self._baseline_wait_seconds = self._human_wait_seconds()

    def _human_wait_seconds(self) -> float:
        try:
            from tools.approval import human_wait_seconds

            return human_wait_seconds(self._session_key)
        except Exception:
            return 0.0

    def run(self, callback):
        acquired = self._serialization_lock.acquire(timeout=self._lock_timeout)
        if not acquired:
            logger.warning(
                "authorization gate lock not acquired after %.1fs "
                "(holder wedged in a pre_tool_call plugin or approval "
                "round-trip?); running prompt unserialized",
                self._lock_timeout,
            )
            return callback()
        try:
            return callback()
        finally:
            self._serialization_lock.release()

    def excluded_seconds(self) -> float:
        """Return human-approval wait seconds accrued since the batch started."""
        return max(0.0, self._human_wait_seconds() - self._baseline_wait_seconds)


def _managed_values(
    outcome: _ManagedToolResult,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], bool, bool]:
    return (
        outcome.result,
        outcome.args,
        outcome.middleware_trace,
        outcome.blocked,
        outcome.dispatched,
    )


# Heartbeat cadence; must stay far below the gateway turn-inactivity timeout
# (default 1800s) so a silent-but-healthy tool never looks idle.
_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S = 30.0


def _run_tool_activity_heartbeat(
    agent,
    stop_event: threading.Event,
    label: str,
    interval: float = _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S,
) -> None:
    """Daemon thread that stamps ``agent._touch_activity`` every ``interval`` seconds
    until ``stop_event`` is set.

    Keeps the gateway turn-inactivity watchdog (default 30 min) from abandoning a turn
    whose tool runs silently. Wedged tools stay bounded by the tool layer's own timeouts.
    """

    try:
        while not stop_event.wait(interval):
            agent._touch_activity(label)
    except Exception:
        # A heartbeat must never break the agent loop.
        pass


def _run_agent_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
    scope_block: str | None = None,
    display_index: int | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
    begin_execution=None,
    authorization_gate: _ConcurrentToolAuthorizationGate | None = None,
) -> _ManagedToolResult:
    """Run Relay rewrites before Hermes policy and dispatch exactly once."""
    from agent import relay_tools
    from hermes_cli.middleware import (
        apply_tool_request_middleware,
        run_tool_execution_middleware,
    )

    trace = middleware_trace if middleware_trace is not None else []
    state = {
        "args": function_args,
        "middleware_trace": trace,
        "blocked": False,
        "dispatched": False,
    }
    dispatch_lock = threading.Lock()

    def _authorized_dispatch(final_args: dict[str, Any]) -> Any:
        with dispatch_lock:
            if state["dispatched"]:
                raise RuntimeError(
                    "Hermes tool execution callback invoked more than once"
                )
            state["dispatched"] = True
            state["blocked"] = False
            state["args"] = final_args

        def _begin() -> None:
            _begin_tool_execution(
                agent,
                function_name=function_name,
                function_args=final_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                display_index=display_index,
            )

        def _advance_start_order(callback=None) -> None:
            if begin_execution is None:
                if callback is not None:
                    callback()
                return
            begin_execution(callback)

        block_message = scope_block
        block_error_type = "tool_scope_block"
        if block_message is None:
            block_error_type = "plugin_block"

            def _resolve_pre_tool_block():
                nonlocal final_args
                try:
                    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

                    block_msg, modified_args = _dispatch_pre_tool_call_hooks(
                        function_name,
                        final_args,
                        **tool_hook_ids(agent, effective_task_id, tool_call_id),
                        middleware_trace=list(state["middleware_trace"]),
                    )
                    if modified_args is not None:
                        final_args = modified_args
                        state["args"] = modified_args
                    return block_msg
                except Exception:
                    return None

            block_message = (
                _resolve_pre_tool_block()
                if authorization_gate is None
                else authorization_gate.run(_resolve_pre_tool_block)
            )

        guardrail_decision = None
        if block_message is None:
            guardrail_decision = agent._tool_guardrails.before_call(
                function_name, final_args
            )
            if guardrail_decision.allows_execution:
                guardrail_decision = None

        if block_message is not None or guardrail_decision is not None:
            _advance_start_order()
            state["blocked"] = True
            if block_message is not None:
                result = json.dumps({"error": block_message}, ensure_ascii=False)
                error_type = block_error_type
                error_message = block_message
            else:
                result = agent._guardrail_block_result(guardrail_decision)
                error_type = "guardrail_block"
                error_message = (
                    getattr(guardrail_decision, "message", None)
                    or "Tool blocked by guardrail policy"
                )
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=final_args,
                result=result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="blocked",
                error_type=error_type,
                error_message=error_message,
                middleware_trace=list(state["middleware_trace"]),
            )
            return result

        if function_name == "memory":
            agent._turns_since_memory = 0
        elif function_name == "skill_manage":
            agent._iters_since_skill = 0

        _advance_start_order(_begin)

        # Heartbeat while the tool is in flight so the gateway inactivity watchdog
        # doesn't abandon a silent-but-live turn (#84491); covers both executor paths.
        _hb_stop = threading.Event()
        _hb_thread = threading.Thread(
            target=_run_tool_activity_heartbeat,
            args=(agent, _hb_stop, f"tool running: {function_name}"),
            kwargs={"interval": _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S},
            daemon=True,
            name=f"tool-activity-hb-{function_name[:24]}",
        )
        _hb_thread.start()
        try:
            return execute(final_args)
        finally:
            _hb_stop.set()
            _hb_thread.join(timeout=2.0)

    def _hermes_pipeline(relay_args: dict[str, Any]) -> Any:
        request_result = apply_tool_request_middleware(
            function_name,
            relay_args,
            skip_relay=True,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
        )
        request_args = (
            request_result.payload
            if isinstance(request_result.payload, dict)
            else relay_args
        )
        trace.clear()
        trace.extend(request_result.trace)
        return run_tool_execution_middleware(
            function_name,
            request_args,
            lambda next_args: _authorized_dispatch(
                next_args if isinstance(next_args, dict) else request_args
            ),
            original_args=function_args,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
        )

    result, _relay_args = relay_tools.execute(
        function_name,
        function_args,
        _hermes_pipeline,
        session_id=str(getattr(agent, "session_id", "") or ""),
        metadata={
            "task_id": effective_task_id or "",
            "turn_id": getattr(agent, "_current_turn_id", "") or "",
            "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
            "tool_call_id": tool_call_id or "",
        },
    )
    return _ManagedToolResult(
        result=result,
        args=state["args"],
        middleware_trace=state["middleware_trace"],
        blocked=bool(state["blocked"]),
        dispatched=bool(state["dispatched"]),
    )


# Sequential wait-loop interrupt poll cadence: /stop lands within ~1s even when
# the tool never polls is_interrupted().
_SEQUENTIAL_INTERRUPT_POLL_SECONDS = 1.0


def _resolve_sequential_tool_timeout() -> float | None:
    """Deadline for one sequential tool call (#85125 Phase 2a).

    ``timeouts.tools.sequential_call`` wins; unset inherits the concurrent batch deadline
    so the two paths can't drift. ``0``/negative disables the bound.

    Deliberately NOT ``agent.deadline.run_bounded_sync``: both executors extend their
    deadline while an approval prompt is open (MUST-preserve), which the fixed-deadline
    primitive can't express.
    """
    from agent.deadline import resolve_timeout

    return resolve_timeout(
        "tools.sequential_call",
        default=_resolve_concurrent_tool_timeout(),
    )


def _run_sequential_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
    scope_block: str | None = None,
    display_index: int | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
) -> _ManagedToolResult:
    """Run one sequential call with the concurrent executor's deadline.

    Interactive tools (``clarify``) own their wait via ``agent.clarify_timeout``; the
    generic deadline would report ``tool_timeout`` while the prompt is still live.
    """
    timeout_s = _resolve_sequential_tool_timeout()
    kwargs = {
        "function_name": function_name,
        "function_args": function_args,
        "effective_task_id": effective_task_id,
        "tool_call_id": tool_call_id,
        "execute": execute,
        "scope_block": scope_block,
        "display_index": display_index,
        "middleware_trace": middleware_trace,
    }
    if function_name in _NEVER_PARALLEL_TOOLS:
        return _run_agent_tool_execution_middleware(agent, **kwargs)

    from tools.daemon_pool import DaemonThreadPoolExecutor

    authorization_gate = _ConcurrentToolAuthorizationGate()
    worker_tid: list[int] = []

    def _run() -> _ManagedToolResult:
        tid = threading.current_thread().ident
        worker_tid.append(tid)
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(tid)
        try:
            return _run_agent_tool_execution_middleware(
                agent, authorization_gate=authorization_gate, **kwargs
            )
        finally:
            with agent._tool_worker_threads_lock:
                agent._tool_worker_threads.discard(tid)
            try:
                _ra()._set_interrupt(False, tid)
            except Exception:
                pass

    executor = DaemonThreadPoolExecutor(max_workers=1)
    future = executor.submit(propagate_context_to_thread(_run))
    # Disabled timeout still runs on the worker: this wait loop is what makes a
    # non-cooperative tool interruptible, so no deadline must not mean no interrupt checks.
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    started = time.monotonic()
    timed_out = False
    interrupted = False
    _last_heartbeat = 0
    try:
        while True:
            wait_slice = _SEQUENTIAL_INTERRUPT_POLL_SECONDS
            if deadline is not None:
                remaining = (
                    deadline + authorization_gate.excluded_seconds() - time.monotonic()
                )
                if remaining <= 0:
                    timed_out = True
                    break
                wait_slice = min(wait_slice, remaining)
            try:
                return future.result(timeout=wait_slice)
            except concurrent.futures.TimeoutError:
                if agent._interrupt_requested:
                    interrupted = True
                    break
                elapsed = int(time.monotonic() - started)
                if elapsed - _last_heartbeat >= 30:
                    _last_heartbeat = elapsed
                    agent._touch_activity(
                        f"sequential tool running ({elapsed}s): {function_name}"
                    )

        if interrupted:
            # Belt-and-braces: interrupt() already fans out to tracked worker
            # tids, but the worker may have registered after the fan-out ran.
            for tid in worker_tid:
                try:
                    _ra()._set_interrupt(
                        True,
                        tid,
                        reason=getattr(agent, "_tool_interrupt_reason", None),
                    )
                except Exception:
                    pass
            # Grace for a cooperative tool to notice its interrupt bit (mirrors the
            # concurrent path's 3s).
            concurrent.futures.wait([future], timeout=3.0)
            if future.done() and not future.cancelled():
                return future.result()
            timed_out = True  # reuse the abandon-shutdown path in finally
            future.cancel()
            interrupt_reason = (
                getattr(agent, "_tool_interrupt_reason", None)
                or "interrupt requested"
            )
            message = (
                f"[Tool execution cancelled — {function_name} was abandoned: "
                f"{interrupt_reason}]"
            )
            logger.info(
                "sequential tool %s abandoned due to %s (%.1fs elapsed)",
                function_name, interrupt_reason, time.monotonic() - started,
            )
            trace = middleware_trace if middleware_trace is not None else []
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=message,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                duration_ms=int((time.monotonic() - started) * 1000),
                status="cancelled",
                error_type="tool_interrupted",
                error_message=f"Tool execution cancelled: {interrupt_reason}",
                middleware_trace=list(trace),
            )
            return _ManagedToolResult(
                result=_ToolCancelledResult(message),
                args=function_args,
                middleware_trace=trace,
                blocked=False,
                dispatched=True,
            )

        # Only reachable when a deadline exists (interrupted returns above).
        assert timeout_s is not None
        message = (
            f"Error executing tool '{function_name}': "
            f"timed out after {timeout_s:.1f}s"
        )
        logger.warning(
            "sequential tool %s timed out after %.1fs", function_name, timeout_s
        )
        future.cancel()
        for tid in worker_tid:
            try:
                _ra()._set_interrupt(True, tid)
            except Exception:
                pass
        trace = middleware_trace if middleware_trace is not None else []
        _emit_terminal_post_tool_call(
            agent,
            function_name=function_name,
            function_args=function_args,
            result=message,
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            duration_ms=int(timeout_s * 1000),
            status="timeout",
            error_type="tool_timeout",
            error_message=message,
            middleware_trace=list(trace),
        )
        return _ManagedToolResult(
            result=_ToolTimeoutResult(message),
            args=function_args,
            middleware_trace=trace,
            blocked=False,
            dispatched=True,
        )
    finally:
        # Never join a wedged worker. DaemonThreadPoolExecutor also keeps it out
        # of the stdlib atexit join, matching the concurrent timeout path.
        executor.shutdown(wait=not timed_out, cancel_futures=timed_out)


def _begin_tool_execution(
    agent,
    *,
    function_name: str,
    function_args: dict[str, Any],
    effective_task_id: str,
    tool_call_id: str,
    display_index: int | None,
) -> None:
    """Run user-visible and checkpoint preflight on final tool arguments."""
    if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
        display_args = (
            _redact_tool_args_for_display(function_name, function_args) or function_args
        )
        args_str = json.dumps(display_args, ensure_ascii=False)
        prefix = f"Tool {display_index}" if display_index is not None else "Tool"
        if agent.verbose_logging:
            print(f"  📞 {prefix}: {function_name}({list(display_args.keys())})")
            print(
                agent._wrap_verbose(
                    "Args: ", json.dumps(display_args, indent=2, ensure_ascii=False)
                )
            )
        else:
            args_preview = (
                args_str[: agent.log_prefix_chars] + "..."
                if len(args_str) > agent.log_prefix_chars
                else args_str
            )
            print(
                f"  📞 {prefix}: {function_name}({list(function_args.keys())}) - "
                f"{args_preview}"
            )

    agent._current_tool = function_name
    agent._touch_activity(f"executing tool: {function_name}")
    try:
        from tools.environments.base import set_activity_callback

        set_activity_callback(agent._touch_activity)
    except Exception:
        pass

    if agent.tool_progress_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            preview = _build_tool_preview(function_name, display_args)
            agent.tool_progress_callback(
                "tool.started", function_name, preview, display_args
            )
        except Exception as callback_error:
            logging.debug("Tool progress callback error: %s", callback_error)

    if agent.tool_start_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            agent.tool_start_callback(
                tool_call_id, function_name, display_args
            )
        except Exception as callback_error:
            logging.debug("Tool start callback error: %s", callback_error)

    if function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
        try:
            _ensure_file_checkpoint(
                agent,
                function_name,
                function_args,
                effective_task_id,
            )
        except Exception:
            pass

    if function_name == "terminal" and agent._checkpoint_mgr.enabled:
        try:
            command = function_args.get("command", "")
            if _is_destructive_command(command):
                cwd = function_args.get("workdir") or os.getenv(
                    "TERMINAL_CWD", os.getcwd()
                )
                agent._checkpoint_mgr.ensure_checkpoint(
                    cwd, f"before terminal: {command[:60]}"
                )
        except Exception:
            pass


def _append_finalized_tool_result(
    agent,
    messages: list,
    *,
    function_name: str,
    function_args: dict,
    function_result,
    tool_call_id: str,
    effective_task_id: str,
    budget: BudgetConfig,
    effect_disposition=None,
):
    """Persist/spill, hint, wrap and append one tool result; flush the session DB.

    Returns ``(function_result, tool_message, risk_metadata)`` — ``function_result`` is the
    persisted/hinted content — or ``None`` when the incremental flush failed (the caller
    must stop the batch).
    """
    if not _is_multimodal_tool_result(function_result):
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=function_name,
            tool_use_id=tool_call_id,
            env=get_active_env(effective_task_id),
            config=budget,
        )
    _record_persisted_path_for_stub(agent, tool_call_id, function_result)

    subdir_hints = agent._subdirectory_hints.check_tool_call(function_name, function_args)
    if subdir_hints:
        if _is_multimodal_tool_result(function_result):
            # Append the hint to the text summary part so the model still sees it;
            # don't touch the image blocks.
            _append_subdir_hint_to_multimodal(function_result, subdir_hints)
        else:
            function_result += subdir_hints

    # Unwrap _multimodal dicts to an OpenAI-style content list; text-only servers
    # get a string-safe fallback so a rejected image result never poisons history.
    _tool_content = agent._tool_result_content_for_active_model(function_name, function_result)
    tool_message = make_tool_result_message(
        function_name,
        _tool_content,
        tool_call_id,
        effect_disposition=effect_disposition,
    )
    messages.append(tool_message)
    if not _flush_session_db_after_tool_progress(
        agent,
        messages,
        stage=f"tool result {function_name}",
    ):
        return None
    return function_result, tool_message, tool_message.get("_tool_output_risk")


def _emit_tool_completed_progress(agent, function_name: str, *, duration: float, is_error: bool, result) -> None:
    """``tool.completed`` UI projection; downstream of the canonical append so resume
    can reconstruct the result even if the UI bridge dies mid-projection."""
    if not agent.tool_progress_callback:
        return
    try:
        agent.tool_progress_callback(
            "tool.completed", function_name, None, None,
            duration=duration, is_error=is_error, result=result,
        )
    except Exception as cb_err:
        logging.debug("Tool progress callback error: %s", cb_err)


def _emit_tool_complete_and_risk(
    agent, *, function_name: str, function_args: dict, tool_call_id: str, result, risk_metadata, blocked: bool
) -> None:
    """Fire ``tool_complete_callback`` (unless blocked) then the ``tool.output_risk`` projection."""
    if not blocked and agent.tool_complete_callback:
        try:
            display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
            agent.tool_complete_callback(tool_call_id, function_name, display_args, result)
        except Exception as cb_err:
            logging.debug("Tool complete callback error: %s", cb_err)

    if (
        risk_metadata is not None
        and risk_metadata.get("risk") != "low"
        and agent.tool_progress_callback
    ):
        try:
            agent.tool_progress_callback(
                "tool.output_risk",
                function_name,
                None,
                None,
                tool_call_id=tool_call_id,
                risk_metadata=risk_metadata,
            )
        except Exception as cb_err:
            logging.debug("Tool output risk callback error: %s", cb_err)


def execute_tool_calls_concurrent(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls concurrently; results are appended in original call order.

    ``finalize=False`` skips end-of-batch budget enforcement and /steer injection (the
    segmented dispatcher owns turn-end work).
    """
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)

    # Resolve the context-scaled tool-output budget once per turn, not per result.
    _tool_budget = _budget_for_agent(agent)

    # ── Pre-flight: interrupt check ──────────────────────────────────
    if agent._interrupt_requested:
        print(f"{agent.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
        for tc in tool_calls:
            cancelled_result = (
                f"[Tool execution cancelled — {tc.function.name} was skipped "
                "due to user interrupt]"
            )
            tool_call_id = _pairing_tool_call_id(tc)
            messages.append(make_tool_result_message(
                tc.function.name,
                cancelled_result,
                tool_call_id,
                effect_disposition="none",
            ))
            _emit_terminal_post_tool_call(
                agent,
                function_name=tc.function.name,
                function_args={},
                result=cancelled_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="cancelled",
                error_type="user_interrupt",
                error_message="Tool execution skipped due to user interrupt",
            )
            _flush_session_db_after_tool_progress(
                agent,
                messages,
                stage=f"cancelled tool result {tc.function.name}",
            )
        return

    # ── Parse args + pre-execution bookkeeping ────────────────────────────
    # (tool call, name, args, middleware trace, parse error, tool-search scope block)
    parsed_calls = []
    for tool_call in tool_calls:
        function_name = _canonical_tool_name(tool_call.function.name)
        function_args, malformed_args_result = _parse_tool_arguments(
            tool_call.function.arguments
        )

        if malformed_args_result is not None:
            parsed_calls.append(
                (
                    tool_call,
                    function_name,
                    function_args,
                    [],
                    malformed_args_result,
                    None,
                )
            )
            continue

        function_name, function_args, _ts_scope_block = _unwrap_tool_search_call(
            agent, function_name, function_args
        )

        parsed_calls.append(
            (tool_call, function_name, function_args, [], None, _ts_scope_block)
        )

    # ── Logging / callbacks ──────────────────────────────────────────
    tool_names_str = ", ".join(name for _, name, _, _, _, _ in parsed_calls)
    if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
        print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")

    # ── Concurrent execution ─────────────────────────────────────────
    # Each slot holds (function_name, function_args, function_result, duration, error_flag, blocked_flag, middleware_trace)
    results = [None] * num_tools
    for i, (tc, name, args, middleware_trace, block_result, _scope_block) in enumerate(parsed_calls):
        if block_result is not None:
            results[i] = (name, args, block_result, 0.0, True, True, middleware_trace)

    start_condition = threading.Condition()
    next_start_order = 0
    # Set once the batch is abandoned so gate-parked workers exit instead of
    # dispatching a tool the turn already reported as timed out.
    batch_abandoned = threading.Event()
    authorization_gate = _ConcurrentToolAuthorizationGate()

    def _abandon_batch() -> None:
        """Release every gate-parked worker so none dispatches post-abandon."""
        batch_abandoned.set()
        with start_condition:
            start_condition.notify_all()

    # The gate bound must sit UNDER the batch deadline, else parked workers are falsely
    # reported timed out without starting. A disabled deadline keeps the stock bound.
    def _start_order_gate_timeout(batch_timeout: float | None) -> float:
        if batch_timeout is None:
            return _START_ORDER_GATE_TIMEOUT_S
        return min(_START_ORDER_GATE_TIMEOUT_S, batch_timeout / 2)

    def _begin_in_order(
        order: int, callback=None, *, tool_name: str = "", gate_timeout: float | None = None
    ) -> bool:
        """Serialize dispatch by submit order. Returns False if abandoned."""
        nonlocal next_start_order
        with start_condition:
            # Bounded wait so one wedged dispatch can't starve/leak later-ordered workers;
            # on expiry proceed out of order (interleaved prompts beat starvation).
            # >= (not ==) releases every skipped worker at once; batch_abandoned short-circuits.
            in_order = start_condition.wait_for(
                lambda: next_start_order >= order or batch_abandoned.is_set(),
                timeout=(
                    _START_ORDER_GATE_TIMEOUT_S if gate_timeout is None else gate_timeout
                ),
            )
            if batch_abandoned.is_set():
                # Do not run the callback or advance the counter: the turn has
                # already synthesized this tool's result and moved on.
                return False
            if not in_order:
                logger.warning(
                    "start-order gate timed out for %s (order=%d next=%d); "
                    "proceeding out of order",
                    tool_name or "tool",
                    order,
                    next_start_order,
                )
            try:
                if callback is not None:
                    callback()
            finally:
                next_start_order = max(next_start_order, order + 1)
                start_condition.notify_all()
        return True

    # Resolved before the workers are defined so the start-order gate can clamp
    # its own bound against the batch deadline it must stay under.
    timeout_s = _resolve_concurrent_tool_timeout()
    gate_timeout_s = _start_order_gate_timeout(timeout_s)

    # Touch activity before launching workers so the gateway knows
    # we're executing tools (not stuck).
    agent._current_tool = tool_names_str
    agent._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

    def _run_tool(
        index,
        tool_call,
        function_name,
        function_args,
        middleware_trace,
        scope_block,
        start_order,
    ):
        """Worker function executed in a thread."""
        # Register this worker tid for interrupt fan-out (AIAgent.interrupt()); must be
        # first and paired with discard + clear in finally.
        _worker_tid = threading.current_thread().ident
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(_worker_tid)
        # Race: interrupt may have fanned out before our registration; apply it
        # to our own tid now.
        if agent._interrupt_requested:
            try:
                _ra()._set_interrupt(
                    True,
                    _worker_tid,
                    reason=getattr(agent, "_tool_interrupt_reason", None),
                )
            except Exception:
                pass
        # Activity callback is thread-local; set it on THIS worker so
        # _wait_for_process heartbeats fire.
        try:
            from tools.environments.base import set_activity_callback
            set_activity_callback(agent._touch_activity)
        except Exception:
            pass
        # Approval/sudo callbacks and turn ContextVars are propagated by
        # propagate_context_to_thread() at submit (GHSA-qg5c-hvr5-hjgr, #13617).
        start = time.time()
        tool_call_id = _pairing_tool_call_id(tool_call)
        blocked = False
        dispatched = False
        start_advanced = False

        def _advance_start(callback=None) -> None:
            nonlocal start_advanced
            if start_advanced:
                return
            try:
                proceed = _begin_in_order(
                    start_order,
                    callback,
                    tool_name=function_name,
                    gate_timeout=gate_timeout_s,
                )
            finally:
                start_advanced = True
            if not proceed:
                # Batch already abandoned: the turn synthesized this tool's
                # result and moved on. Abort instead of dispatching late.
                raise _BatchAbandoned(function_name)

        try:
            try:
                def _execute(next_args: dict[str, Any]) -> Any:
                    return agent._invoke_tool(
                        function_name,
                        next_args,
                        effective_task_id,
                        tool_call_id,
                        messages=messages,
                        pre_tool_block_checked=True,
                        skip_tool_request_middleware=True,
                        skip_tool_execution_middleware=True,
                        tool_request_middleware_trace=list(middleware_trace),
                    )

                managed = _run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=scope_block,
                    display_index=index + 1,
                    middleware_trace=middleware_trace,
                    begin_execution=_advance_start,
                    authorization_gate=authorization_gate,
                )
                result = managed.result
                function_args = managed.args
                middleware_trace = managed.middleware_trace
                blocked = managed.blocked
                dispatched = managed.dispatched
            except _BatchAbandoned:
                # Abandoned at the start-order gate: the main thread already synthesized
                # this result, so write/emit nothing (would double-report the tool_call_id).
                logger.info(
                    "tool %s abandoned at start-order gate; skipping dispatch",
                    function_name,
                )
                return
            except KeyboardInterrupt:
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
                result = _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    start_time=start,
                    middleware_trace=list(middleware_trace),
                )
                duration = time.time() - start
                logger.info("tool %s cancelled (%.2fs)", function_name, duration)
                results[index] = (
                    function_name,
                    function_args,
                    result,
                    duration,
                    True,
                    False,
                    middleware_trace,
                )
                return
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            if not blocked and not dispatched:
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    result=result,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    duration_ms=int(duration * 1000),
                    middleware_trace=list(middleware_trace),
                )
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (
                function_name,
                function_args,
                result,
                duration,
                is_error,
                blocked,
                middleware_trace,
            )
        finally:
            # Teardown advance keeps later-ordered workers moving; never let the
            # abandonment signal escape here.
            try:
                _advance_start()
            except _BatchAbandoned:
                pass
            # Tear down tid tracking and clear any interrupt bit so a recycled tid starts
            # clean. MUST be in finally: BaseException subclasses bypass ``except Exception``.
            with agent._tool_worker_threads_lock:
                agent._tool_worker_threads.discard(_worker_tid)
            try:
                _ra()._set_interrupt(False, _worker_tid)
            except Exception:
                pass

    # Start spinner for CLI mode (skip when TUI handles tool progress)
    spinner = None
    if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
        face = random.choice(KawaiiSpinner.get_waiting_faces())
        spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=agent._print_fn)
        spinner.start()

    try:
        runnable_calls = [
            (i, tc, name, args, scope_block)
            for i, (tc, name, args, _trace, parse_error, scope_block) in enumerate(
                parsed_calls
            )
            if parse_error is None
        ]
        futures = []
        future_to_index = {}
        timed_out_indices: set[int] = set()
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        if runnable_calls:
            max_workers = _max_workers_for_tool_batch(runnable_calls)
            # Daemon workers: stdlib ThreadPoolExecutor's atexit join would let one
            # wedged tool thread block interpreter exit forever.
            from tools.daemon_pool import DaemonThreadPoolExecutor
            executor = DaemonThreadPoolExecutor(max_workers=max_workers)
            abandon_executor = False
            try:
                for submit_index, (i, tc, name, args, scope_block) in enumerate(
                    runnable_calls
                ):
                    # Propagate turn ContextVars and thread-local approval/sudo
                    # callbacks into the worker; clears callbacks on exit.
                    try:
                        f = executor.submit(
                            propagate_context_to_thread(_run_tool),
                            i,
                            tc,
                            name,
                            args,
                            parsed_calls[i][3],
                            scope_block,
                            submit_index,
                        )
                    except RuntimeError as submit_error:
                        if not _is_interpreter_shutdown_submit_error(submit_error):
                            raise
                        skipped_calls = runnable_calls[submit_index:]
                        logger.warning(
                            "interpreter shutdown while scheduling concurrent tools; "
                            "skipping %d unsubmitted tool(s)",
                            len(skipped_calls),
                        )
                        for (
                            skipped_i,
                            _tc,
                            skipped_name,
                            skipped_args,
                            _scope_block,
                        ) in skipped_calls:
                            if results[skipped_i] is None:
                                middleware_trace = parsed_calls[skipped_i][3]
                                result = (
                                    f"Error executing tool '{skipped_name}': "
                                    "Python interpreter is shutting down; tool was not started"
                                )
                                results[skipped_i] = (
                                    skipped_name,
                                    skipped_args,
                                    result,
                                    0.0,
                                    True,
                                    False,
                                    middleware_trace,
                                )
                        break
                    futures.append(f)
                    future_to_index[f] = i

                # Wait with periodic heartbeats (gateway inactivity monitor) and
                # interrupt checks (/stop or a new message).
                _conc_start = time.time()
                _interrupt_logged = False
                while True:
                    wait_timeout = 5.0
                    if deadline is not None:
                        effective_deadline = (
                            deadline + authorization_gate.excluded_seconds()
                        )
                        remaining = effective_deadline - time.monotonic()
                        if remaining <= 0:
                            done, not_done = set(), {
                                f for f in futures if not f.done()
                            }
                        else:
                            wait_timeout = min(wait_timeout, remaining)
                            done, not_done = concurrent.futures.wait(
                                futures, timeout=wait_timeout,
                            )
                    else:
                        done, not_done = concurrent.futures.wait(
                            futures, timeout=wait_timeout,
                        )
                    if not not_done:
                        break

                    if (
                        deadline is not None
                        and time.monotonic()
                        >= deadline + authorization_gate.excluded_seconds()
                    ):
                        abandon_executor = True
                        timed_out_indices = {
                            future_to_index[f]
                            for f in not_done
                            if f in future_to_index
                        }
                        _still_running = [
                            parsed_calls[i][1]
                            for i in timed_out_indices
                        ]
                        logger.warning(
                            "concurrent tool batch timed out after %.1fs; "
                            "%d tool(s) still running: %s",
                            timeout_s,
                            len(timed_out_indices),
                            ", ".join(_still_running[:5]),
                        )
                        for f in not_done:
                            f.cancel()
                        # Release gate-parked workers before interrupt fan-out so none
                        # later dispatches a tool just reported as timed out.
                        _abandon_batch()
                        with agent._tool_worker_threads_lock:
                            worker_tids = list(agent._tool_worker_threads)
                        for tid in worker_tids:
                            try:
                                _ra()._set_interrupt(True, tid)
                            except Exception:
                                pass
                        break

                    # Tools without interrupt checks (web_search, read_file) run to
                    # completion; cancel unstarted futures so we don't block on them.
                    if agent._interrupt_requested:
                        abandon_executor = True
                        if not _interrupt_logged:
                            _interrupt_logged = True
                            agent._vprint(
                                f"{agent.log_prefix}⚡ Interrupt: cancelling "
                                f"{len(not_done)} pending concurrent tool(s)",
                                force=True,
                            )
                        for f in not_done:
                            f.cancel()
                        # Release gate-parked workers so they abort instead of
                        # dispatching after the turn was already interrupted.
                        _abandon_batch()
                        # Give already-running tools a moment to notice the
                        # per-thread interrupt signal and exit gracefully.
                        concurrent.futures.wait(not_done, timeout=3.0)
                        break

                    _conc_elapsed = int(time.time() - _conc_start)
                    # Heartbeat every ~30s (6 × 5s poll intervals)
                    if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                        _still_running = [
                            parsed_calls[future_to_index[f]][1]
                            for f in not_done
                            if f in future_to_index
                        ]
                        agent._touch_activity(
                            f"concurrent tools running ({_conc_elapsed}s, "
                            f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                        )
            finally:
                # Any abandoning exit from the wait loop (including the exception
                # path) must release gate-parked workers.
                if abandon_executor:
                    _abandon_batch()
                # On abandon do NOT join hung workers: a wedged thread is left detached
                # rather than deadlocking the batch. Normal completion joins.
                executor.shutdown(
                    wait=not abandon_executor,
                    cancel_futures=abandon_executor,
                )
    finally:
        if spinner:
            completed = sum(1 for r in results if r is not None)
            total_dur = sum(r[3] for r in results if r is not None)
            spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

    # ── Post-execution: display per-tool results ─────────────────────
    for i, (tc, name, args, middleware_trace, _parse_error, _scope_block) in enumerate(
        parsed_calls
    ):
        r = results[i]
        tool_call_id = _pairing_tool_call_id(tc)
        blocked = False
        is_error = True
        progress_function_name = name
        # A worker may finish between the deadline snapshot and this loop;
        # prefer its real result over a fabricated timeout.
        effect_disposition = None
        if i in timed_out_indices and r is None:
            suffix = f"{timeout_s:.1f}s" if timeout_s is not None else "the configured timeout"
            function_result = f"Error executing tool '{name}': timed out after {suffix}"
            effect_disposition = "unknown"
            _emit_terminal_post_tool_call(
                agent,
                function_name=name,
                function_args=args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                duration_ms=int((timeout_s or 0.0) * 1000),
                status="timeout",
                error_type="tool_timeout",
                error_message=function_result,
                middleware_trace=list(middleware_trace),
            )
            tool_duration = float(timeout_s or 0.0)
        elif r is None:
            # Tool was cancelled (interrupt) or thread didn't return
            if agent._interrupt_requested:
                function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=name,
                    function_args=args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    status="cancelled",
                    error_type="keyboard_interrupt",
                    error_message="Tool execution cancelled by user interrupt",
                    middleware_trace=list(middleware_trace),
                )
            else:
                function_result = f"Error executing tool '{name}': thread did not return a result"
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=name,
                    function_args=args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    status="error",
                    error_type="thread_missing_result",
                    error_message=function_result,
                    middleware_trace=list(middleware_trace),
                )
            tool_duration = 0.0
        else:
            function_name, function_args, function_result, tool_duration, is_error, blocked, middleware_trace = r
            name = function_name
            args = function_args
            progress_function_name = function_name
            if _parse_error is not None:
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    status="error",
                    error_type="invalid_tool_arguments",
                    error_message="Tool arguments must be a valid JSON object",
                    middleware_trace=list(middleware_trace),
                )
            if blocked:
                effect_disposition = "none"

            if not blocked:
                function_result = agent._append_guardrail_observation(
                    function_name,
                    function_args,
                    function_result,
                    failed=is_error,
                    tool_call_id=tool_call_id,
                )

            if is_error:
                _err_text = _multimodal_text_summary(function_result)
                result_preview = _err_text[:200] if len(_err_text) > 200 else _err_text
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

            # Track file-mutation outcome for the turn-end verifier; blocked calls
            # never ran, so they count as neither failure nor success.
            if not blocked:
                try:
                    agent._record_file_mutation_result(
                        function_name, function_args, function_result, is_error,
                    )
                except Exception as _ver_err:
                    logging.debug("file-mutation verifier record failed: %s", _ver_err)

            if agent.verbose_logging:
                logging.debug("Tool %s completed in %.2fs", function_name, tool_duration)
                logging.debug("Tool result (%d chars): %s", len(function_result), function_result)

        agent._current_tool = None
        _status_suffix = " (error)" if is_error else ""
        agent._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s){_status_suffix}")

        display_function_result = function_result
        finalized = _append_finalized_tool_result(
            agent,
            messages,
            function_name=name,
            function_args=args,
            function_result=function_result,
            tool_call_id=tool_call_id,
            effective_task_id=effective_task_id,
            budget=_tool_budget,
            effect_disposition=effect_disposition,
        )
        if finalized is None:
            return
        function_result, _tool_message, risk_metadata = finalized

        if not blocked:
            _emit_tool_completed_progress(
                agent, progress_function_name,
                duration=tool_duration, is_error=is_error, result=display_function_result,
            )

        if agent._should_emit_quiet_tool_messages():
            cute_msg = _get_cute_tool_message_impl(
                name, args, tool_duration, result=display_function_result,
            )
            agent._safe_print(f"  {cute_msg}")
        elif not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
            _preview_str = _multimodal_text_summary(display_function_result)
            if agent.verbose_logging:
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                print(agent._wrap_verbose("Result: ", _preview_str))
            else:
                response_preview = _preview_str[:agent.log_prefix_chars] + "..." if len(_preview_str) > agent.log_prefix_chars else _preview_str
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

        _emit_tool_complete_and_risk(
            agent,
            function_name=name,
            function_args=args,
            tool_call_id=tool_call_id,
            result=display_function_result,
            risk_metadata=risk_metadata,
            blocked=blocked,
        )

    # ── Per-turn aggregate budget enforcement ──────────────────────────
    # Keep /steer pending until the post-budget drain: an early drain could be
    # discarded when budget enforcement replaces that tool result.
    num_tools = len(parsed_calls)
    if finalize and num_tools > 0:
        turn_tool_msgs = messages[-num_tools:]
        enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id), config=_tool_budget)

    # ── /steer injection ────────────────────────────────────────────────
    # AFTER budget enforcement so the steer marker is never truncated; see steer().
    if finalize and num_tools > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools)



def _append_cancelled_tool_results(messages: list, tool_calls, *, reason: str) -> None:
    """Append a cancelled ``tool`` result for each call so a hard interrupt never leaves
    the assistant tool-call turn without matching results (role-alternation violation).
    """
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", "") or "tool"
        messages.append(make_tool_result_message(
            name,
            f"[Tool execution cancelled — {name} was skipped due to {reason}]",
            _pairing_tool_call_id(tc),
            effect_disposition="none",
        ))


def _start_quiet_tool_spinner(agent, function_name: str, function_args: dict, *, gate: bool = True):
    """Start the quiet-mode kawaii spinner for one tool call, or return None.

    ``gate=False`` skips ``_should_start_quiet_spinner`` (context-engine tools always spin).
    """
    if not agent._should_emit_quiet_tool_messages():
        return None
    if gate and not agent._should_start_quiet_spinner():
        return None
    face = random.choice(KawaiiSpinner.get_waiting_faces())
    emoji = _get_tool_emoji(function_name)
    display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
    preview = _build_tool_label(function_name, display_args) or function_name
    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
    spinner.start()
    return spinner


def _finish_quiet_tool_spinner(agent, spinner, function_name: str, function_args: dict, tool_duration: float, result) -> None:
    """Stop the spinner with the cute completion line, or print it when no spinner ran."""
    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=result)
    if spinner:
        spinner.stop(cute_msg)
    elif agent._should_emit_quiet_tool_messages():
        agent._vprint(f"  {cute_msg}")


def execute_tool_calls_sequential(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls sequentially (single calls or interactive tools).

    ``finalize=False`` skips end-of-batch budget enforcement and /steer injection (the
    segmented dispatcher owns turn-end work).
    """
    # Resolve the context-scaled tool-output budget once per turn, not per result.
    _tool_budget = _budget_for_agent(agent)

    # One bounded execution funnel for every runtime-tool branch; no duplicated
    # timeout policy in the callbacks below.
    def _run_agent_tool_execution_middleware(agent, **kwargs):
        return _run_sequential_tool_execution_middleware(agent, **kwargs)

    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
        tool_call_id = _pairing_tool_call_id(tool_call)
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        # SAFETY: check interrupt BEFORE each tool so a "stop" during the previous
        # tool skips all remaining ones.
        if agent._interrupt_requested:
            remaining_calls = assistant_message.tool_calls[i-1:]
            if remaining_calls:
                agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
            for skipped_tc in remaining_calls:
                skipped_name = skipped_tc.function.name
                cancelled_result = (
                    f"[Tool execution cancelled — {skipped_name} was skipped "
                    "due to user interrupt]"
                )
                messages.append(make_tool_result_message(
                    skipped_name,
                    cancelled_result,
                    _pairing_tool_call_id(skipped_tc),
                    effect_disposition="none",
                ))
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=skipped_name,
                    function_args={},
                    result=cancelled_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(skipped_tc, "id", "") or "",
                    status="cancelled",
                    error_type="user_interrupt",
                    error_message="Tool execution skipped due to user interrupt",
                )
                if not _flush_session_db_after_tool_progress(
                    agent,
                    messages,
                    stage=f"cancelled tool result {skipped_name}",
                ):
                    return
            break

        function_name = _canonical_tool_name(tool_call.function.name)
        function_args, malformed_args_result = _parse_tool_arguments(
            tool_call.function.arguments
        )
        if malformed_args_result is not None:
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=malformed_args_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="error",
                error_type="invalid_tool_arguments",
                error_message="Tool arguments must be a valid JSON object",
            )
            messages.append(
                make_tool_result_message(
                    function_name,
                    malformed_args_result,
                    tool_call_id,
                )
            )
            if not _flush_session_db_after_tool_progress(
                agent,
                messages,
                stage=f"invalid tool arguments {function_name}",
            ):
                return
            continue

        function_name, function_args, _ts_scope_block = _unwrap_tool_search_call(
            agent, function_name, function_args, flatten_probe=True
        )

        middleware_trace: list[dict[str, Any]] = []
        _execution_blocked = False
        _execution_dispatched = False

        tool_start_time = time.time()

        if function_name != "delegate_task" and function_name in INLINE_TOOL_EXECUTORS:
            # Agent-level tools that need live AIAgent state; table shared with invoke_tool.
            inline_executor = INLINE_TOOL_EXECUTORS[function_name]
            inline_ctx = InlineToolContext(
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                messages=messages,
            )

            def _execute(next_args: dict) -> Any:
                return inline_executor(agent, next_args, inline_ctx)
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl(function_name, function_args, tool_duration, result=function_result)}")
        elif function_name == "delegate_task":
            _action_arg = str(function_args.get("action") or "").strip().lower()
            tasks_arg = function_args.get("tasks")
            if _action_arg in ("list", "steer", "stop"):
                spinner_label = f"🔀 subagent {_action_arg}"
            elif tasks_arg and isinstance(tasks_arg, list):
                spinner_label = f"🔀 delegating {len(tasks_arg)} tasks · (/agents to monitor)"
            else:
                goal_preview = (function_args.get("goal") or "")[:30]
                spinner_label = (
                    f"🔀 {goal_preview} · (/agents to monitor)"
                    if goal_preview
                    else "🔀 delegating · (/agents to monitor)"
                )
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            agent._delegate_spinner = spinner
            _delegate_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent._dispatch_delegate_task(next_args)
                function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=_ts_scope_block,
                    display_index=i,
                ))
                _delegate_result = function_result
            finally:
                agent._delegate_spinner = None
                tool_duration = time.time() - tool_start_time
                _finish_quiet_tool_spinner(agent, spinner, 'delegate_task', function_args, tool_duration, _delegate_result)
        elif agent._context_engine_tool_names and function_name in agent._context_engine_tool_names:
            # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
            spinner = _start_quiet_tool_spinner(agent, function_name, function_args, gate=False)
            _ce_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages)
                function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=_ts_scope_block,
                    display_index=i,
                ))
                _ce_result = function_result
            except Exception as tool_error:
                function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                _finish_quiet_tool_spinner(agent, spinner, function_name, function_args, tool_duration, _ce_result)
        elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
            # Memory provider tools (hindsight_retain, honcho_search, etc.)
            # These are not in the tool registry — route through MemoryManager.
            spinner = _start_quiet_tool_spinner(agent, function_name, function_args)
            _mem_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent._memory_manager.handle_tool_call(function_name, next_args)
                function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=_ts_scope_block,
                    display_index=i,
                ))
                _mem_result = function_result
            except Exception as tool_error:
                function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                _finish_quiet_tool_spinner(agent, spinner, function_name, function_args, tool_duration, _mem_result)
        else:
            # Registry tools: post hook is owned by this executor (inner observer suppressed).
            spinner = _start_quiet_tool_spinner(agent, function_name, function_args) if agent.quiet_mode else None
            _spinner_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    from model_tools import suppress_post_tool_call_hook

                    with suppress_post_tool_call_hook():
                        return _ra().handle_function_call(
                            function_name,
                            next_args,
                            effective_task_id,
                            tool_call_id=tool_call_id,
                            session_id=agent.session_id or "",
                            turn_id=getattr(agent, "_current_turn_id", "") or "",
                            api_request_id=getattr(agent, "_current_api_request_id", "")
                            or "",
                            enabled_tools=(
                                list(agent.valid_tool_names)
                                if agent.valid_tool_names
                                else None
                            ),
                            skip_pre_tool_call_hook=True,
                            skip_tool_request_middleware=True,
                            skip_tool_execution_middleware=True,
                            tool_request_middleware_trace=list(middleware_trace),
                            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                        )

                (
                    function_result,
                    function_args,
                    middleware_trace,
                    _execution_blocked,
                    _execution_dispatched,
                ) = _managed_values(
                    _run_agent_tool_execution_middleware(
                        agent,
                        function_name=function_name,
                        function_args=function_args,
                        effective_task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                        execute=_execute,
                        scope_block=_ts_scope_block,
                        display_index=i,
                        middleware_trace=middleware_trace,
                    )
                )
                _spinner_result = function_result
            except KeyboardInterrupt:
                function_result = _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    start_time=tool_start_time,
                    middleware_trace=list(middleware_trace),
                )
                _spinner_result = function_result
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
                # Emit results for THIS and every remaining call before re-raising so
                # the tool-call turn keeps matching results (alternation).
                _append_cancelled_tool_results(
                    messages,
                    assistant_message.tool_calls[i - 1:],
                    reason="keyboard interrupt",
                )
                raise
            except Exception as tool_error:
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                if agent.quiet_mode:
                    _finish_quiet_tool_spinner(agent, spinner, function_name, function_args, tool_duration, _spinner_result)

        _execution_timed_out = isinstance(
            function_result, (_ToolTimeoutResult, _ToolCancelledResult)
        )
        if isinstance(function_result, str):
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
            _result_len = len(function_result)
        else:
            # Multimodal dict result (_multimodal=True) — not sliceable as string
            result_preview = function_result
            _result_len = len(str(function_result))

        # Log tool errors to the persistent error log so [error] tags
        # in the UI always have a corresponding detailed entry on disk.
        _is_error_result, _ = _detect_tool_failure(function_name, function_result)
        # Inline-dispatched runtime tools never reach handle_function_call, so the
        # executor owns the one terminal post_tool_call per tool_call_id (the inner
        # observer is suppressed); also stops an abandoned timeout worker reporting late.
        _executor_must_emit_post_hook = (
            not _execution_blocked
            and not _execution_timed_out
        )
        if _executor_must_emit_post_hook:
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                duration_ms=int(tool_duration * 1000),
                middleware_trace=list(middleware_trace),
            )
        if not _execution_blocked:
            function_result = agent._append_guardrail_observation(
                function_name,
                function_args,
                function_result,
                failed=_is_error_result,
                tool_call_id=tool_call_id,
            )
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
        if _is_error_result:
            logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

        # Track file-mutation outcome for the turn-end verifier; both paths feed
        # the same state so the footer reflects every tool call.
        if not _execution_blocked:
            try:
                agent._record_file_mutation_result(
                    function_name, function_args, function_result, _is_error_result,
                )
            except Exception as _ver_err:
                logging.debug("file-mutation verifier record failed: %s", _ver_err)

        agent._current_tool = None
        _status_suffix = " (error)" if _is_error_result else ""
        agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s){_status_suffix}")

        if agent.verbose_logging:
            logging.debug("Tool %s completed in %.2fs", function_name, tool_duration)
            _log_result = _multimodal_text_summary(function_result)
            logging.debug("Tool result (%d chars): %s", len(_log_result), _log_result)

        display_function_result = function_result
        finalized = _append_finalized_tool_result(
            agent,
            messages,
            function_name=function_name,
            function_args=function_args,
            function_result=function_result,
            tool_call_id=tool_call_id,
            effective_task_id=effective_task_id,
            budget=_tool_budget,
            effect_disposition="unknown" if _execution_timed_out else None,
        )
        if finalized is None:
            return
        function_result, _tool_message, risk_metadata = finalized

        if not _execution_blocked:
            _emit_tool_completed_progress(
                agent, function_name,
                duration=tool_duration, is_error=_is_error_result, result=display_function_result,
            )
        _emit_tool_complete_and_risk(
            agent,
            function_name=function_name,
            function_args=function_args,
            tool_call_id=tool_call_id,
            result=display_function_result,
            risk_metadata=risk_metadata,
            blocked=_execution_blocked,
        )

        if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
            if agent.verbose_logging:
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                print(agent._wrap_verbose("Result: ", function_result))
            else:
                _fr_str = function_result if isinstance(function_result, str) else str(function_result)
                response_preview = _fr_str[:agent.log_prefix_chars] + "..." if len(_fr_str) > agent.log_prefix_chars else _fr_str
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

        if agent._interrupt_requested and i < len(assistant_message.tool_calls):
            remaining = len(assistant_message.tool_calls) - i
            agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
            for skipped_tc in assistant_message.tool_calls[i:]:
                skipped_name = skipped_tc.function.name
                messages.append(make_tool_result_message(
                    skipped_name,
                    f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                    _pairing_tool_call_id(skipped_tc),
                    effect_disposition="none",
                ))
                if not _flush_session_db_after_tool_progress(
                    agent,
                    messages,
                    stage=f"skipped tool result {skipped_name}",
                ):
                    return
            break

    # ── Per-turn aggregate budget enforcement ──────────────────────────
    # Keep /steer pending until the post-budget drain: an early drain could be
    # discarded when budget enforcement replaces a tool result.
    num_tools_seq = len(assistant_message.tool_calls)
    if finalize and num_tools_seq > 0:
        enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id), config=_tool_budget)

    # ── /steer injection ────────────────────────────────────────────────
    # See the concurrent path for rationale.
    if finalize and num_tools_seq > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools_seq)




def execute_tool_calls_segmented(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, segments=None) -> None:
    """Execute a mixed batch as ordered parallel/sequential segments.

    ``segments`` is the ``(kind, calls)`` plan from ``_plan_tool_batch_segments``;
    contiguous segments preserve per-call result order and barrier boundaries exactly
    as fully-sequential execution. Turn-end work (budget + /steer) runs once here;
    segment executors run with ``finalize=False``. Each segment executor checks the
    interrupt flag up front, so an interrupt drains later segments with one result per call.
    """
    from types import SimpleNamespace

    if segments is None:
        _active_env = get_active_env(effective_task_id)
        _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
        segments = _plan_tool_batch_segments(assistant_message.tool_calls, execution_cwd=_exec_cwd)

    for kind, calls in segments:
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        segment_message = SimpleNamespace(tool_calls=list(calls))
        if kind == "parallel":
            execute_tool_calls_concurrent(
                agent, segment_message, messages, effective_task_id, api_call_count,
                finalize=False,
            )
        else:
            execute_tool_calls_sequential(
                agent, segment_message, messages, effective_task_id, api_call_count,
                finalize=False,
            )

        if getattr(agent, "_incremental_persistence_failed", False):
            return

    # ── Whole-turn finalize (budget + /steer) ─────────────────────────
    total_tools = len(assistant_message.tool_calls)
    if total_tools > 0:
        _tool_budget = _budget_for_agent(agent)
        enforce_turn_budget(
            messages[-total_tools:],
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        )
        agent._apply_pending_steer_to_tool_results(messages, total_tools)


__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
    "execute_tool_calls_segmented",
]
