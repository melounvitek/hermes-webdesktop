"""Trace state and observation lifecycle for the Langfuse plugin.

Holds no process-global state: the live ``TraceState`` registry, its lock and
the client cache live in the package ``__init__`` (tests patch them there).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ._capture import _capture_mode, _debug, _extract_last_user_message

try:
    from langfuse import propagate_attributes
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    propagate_attributes = None


@dataclass
class TraceState:
    trace_id: str
    root_ctx: Any
    root_span: Any
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    turn_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Keyed by child_session_id: subagent_stop carries no child_subagent_id.
    subagents: Dict[str, Any] = field(default_factory=dict)
    # Fingerprints of MoA fan-outs already recorded: the client holds its last
    # fan-out until the next one, so tool-loop turns would re-emit advisors.
    moa_emitted: set = field(default_factory=set)
    last_updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_updated_at = time.time()

    def pop_tool(self, tool_call_id: str, tool_name: str) -> Any:
        """Detach the open tool observation: by id, else FIFO by name for id-less callers."""
        observation = self.tools.pop(tool_call_id, None) if tool_call_id else None
        if observation is None:
            queue = self.pending_tools_by_name.get(tool_name)
            if queue:
                observation = queue.pop(0)
                if not queue:
                    self.pending_tools_by_name.pop(tool_name, None)
        return observation

    def record_tool_output(self, tool_call_id: str, output: Any) -> None:
        """Backfill the generation's tool_call record so it carries the result alongside arguments."""
        for tool_call in reversed(self.turn_tool_calls):
            if tool_call.get("id") == tool_call_id:
                tool_call["output"] = output
                function_payload = tool_call.get("function")
                if isinstance(function_payload, dict):
                    function_payload["output"] = output
                return


# ---------------------------------------------------------------------------
# Trace keys
# ---------------------------------------------------------------------------

def _scope_prefix(task_id: str, session_id: str) -> str:
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _trace_key(task_id: str, session_id: str, *, turn_id: str = "", api_request_id: str = "") -> str:
    """Stable in-process trace scope key for one agent turn.

    ``turn_id``/``api_request_id`` scope state so concurrent requests sharing a
    task/session never collide. ``turn_id`` wins over ``api_request_id`` so the
    turn-level post_llm_call hook (no api_request_id) resolves to the same key
    as request-level hooks. Legacy shape: bare ``task_id`` (no ``task:`` prefix),
    kept for keys minted before turn/request scoping existed.
    """
    if turn_id:
        return f"{_scope_prefix(task_id, session_id)}:turn:{turn_id}"
    if api_request_id:
        return f"{_scope_prefix(task_id, session_id)}:api:{api_request_id}"
    if task_id:
        return task_id
    return _scope_prefix(task_id, session_id)


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def _start_root_trace(task_key: str, *, task_id: str, session_id: str, platform: str, provider: str, model: str,
                      api_mode: str, messages: Any, client: Any,
                      turn_id: str = "", api_request_id: str = "") -> TraceState:
    trace_id = client.create_trace_id(seed=f"{session_id or 'sessionless'}::{task_id or task_key}")
    trace_input = _extract_last_user_message(messages)
    metadata = {
        "source": "hermes", "task_id": task_id, "turn_id": turn_id, "api_request_id": api_request_id,
        "platform": platform, "provider": provider, "model": model, "api_mode": api_mode,
        "capture_mode": _capture_mode(),
    }
    # session_id must be in trace_context for Langfuse session grouping.
    trace_ctx: Dict[str, Any] = {"trace_id": trace_id}
    if session_id:
        trace_ctx["session_id"] = session_id

    def open_root():
        ctx = client.start_as_current_observation(
            trace_context=trace_ctx, name="Hermes turn", as_type="chain",
            input=trace_input, metadata=metadata, end_on_exit=False,
        )
        return ctx, ctx.__enter__()

    root_ctx = root_span = None
    if propagate_attributes is not None:
        try:
            with propagate_attributes(session_id=session_id or task_key, trace_name="Hermes turn",
                                      tags=["hermes", "langfuse"]):
                root_ctx, root_span = open_root()
        except Exception:
            root_ctx = None
    if root_ctx is None:
        root_ctx, root_span = open_root()

    # SDK v3 uses update_trace(); failures must never block the turn.
    try:
        root_span.update_trace(input=trace_input)
    except Exception as exc:
        _debug(f"update_trace(input) failed: {exc}")

    _debug(f"started trace {trace_id} for {task_key}")
    return TraceState(trace_id=trace_id, root_ctx=root_ctx, root_span=root_span)


def _start_child_observation(state: TraceState, *, name: str, as_type: str, input_value: Any,
                             metadata: Optional[dict] = None, model: Optional[str] = None,
                             model_parameters: Optional[dict] = None) -> Any:
    return state.root_span.start_observation(
        name=name, as_type=as_type, input=input_value, metadata=metadata or {},
        model=model, model_parameters=model_parameters,
    )


def _end_observation(observation: Any, *, output: Any = None, metadata: Optional[dict] = None,
                     usage_details: Optional[dict] = None, cost_details: Optional[dict] = None) -> None:
    if observation is None:
        return
    try:
        update_kwargs: Dict[str, Any] = {}
        if output is not None:
            update_kwargs["output"] = output
        for key, val in (("metadata", metadata), ("usage_details", usage_details), ("cost_details", cost_details)):
            if val:
                update_kwargs[key] = val
        if update_kwargs:
            observation.update(**update_kwargs)
        observation.end()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"end observation failed: {exc}")


def _end_children(state: TraceState, *, include_subagents: bool = False) -> None:
    for observation in (*state.generations.values(), *state.tools.values()):
        _end_observation(observation)
    for queue in state.pending_tools_by_name.values():
        for observation in queue:
            _end_observation(observation)
    if include_subagents:
        for observation in state.subagents.values():
            _end_observation(observation)


def _close_root(state: TraceState, *, label: str, output: Any = None,
                children: bool = True, include_subagents: bool = False) -> None:
    """End ``state``'s root: children first, then trace output, root ``end()``, context unwind.

    Never raises. Neither output update may prevent ``end()``, else children
    export without a root. The root context manager is unwound now, while
    ``opentelemetry.trace.Span`` is still a real type: a GC-driven close at
    interpreter teardown raises TypeError inside ``use_span``'s isinstance check.
    """
    try:
        if children:
            _end_children(state, include_subagents=include_subagents)
        if output is not None:
            # update_trace sets TRACE-level I/O (SDK v3); root I/O via update().
            for method, what in (("update_trace", "update_trace(output)"), ("update", "root update(output)")):
                try:
                    getattr(state.root_span, method)(output=output)
                except Exception as exc:
                    _debug(f"{what} failed: {exc}")
        try:
            state.root_span.end()
        except Exception as exc:
            _debug(f"root end() failed: {exc}")
        if state.root_ctx is not None:
            try:
                state.root_ctx.__exit__(None, None, None)
            except Exception:  # pragma: no cover - fail-open
                pass
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"{label} failed: {exc}")
        # Last-chance end so an unexpected error still exports the root.
        try:
            state.root_span.end()
        except Exception:
            pass


def _merge_trace_output(output: Any, state: TraceState) -> Any:
    if not state.turn_tool_calls:
        return output
    merged = dict(output) if isinstance(output, dict) else {"content": output}
    merged["tool_calls"] = list(state.turn_tool_calls)
    return merged
