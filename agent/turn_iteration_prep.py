"""Per-iteration transcript preparation for the conversation turn loop, run before the request
is assembled: the ``agent:step`` callback, skill-nudge counter, pre-API ``/steer`` drain into
the newest tool result (never a user message — alternation), run-budget wrap-up notice,
tool_call argument sanitization, legacy interrupt-scaffold ghost-row drop and role-alternation
repair. Extracted from ``run_conversation``; nothing here imports
``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger("agent.conversation_loop")


@dataclass
class IterationPrep:
    """Always ``action == "fallthrough"``. ``messages`` is the (possibly filtered) transcript
    and ``request_logger`` the per-request logger the caller keeps using."""

    action: str
    messages: Any
    request_logger: Any


def prepare_iteration(
    agent: Any,
    *,
    messages: Any,
    api_call_count: Any,
) -> IterationPrep:
    """Prepare ``messages`` for this iteration in the original order. Every mutation here is
    cache-safe by construction: steer text lands in the newest tool result, the ghost-row
    filter only drops hidden scaffold placeholders, and repair runs BEFORE the request build."""
    from agent.conversation_loop import (
        _INTERRUPT_SCAFFOLD_MARKER,
        _maybe_inject_run_budget_wrapup,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> IterationPrep:
        return IterationPrep(
            action=action,
            messages=messages,
            request_logger=request_logger,

        )

    # Fire step_callback for gateway hooks (agent:step event)
    if agent.step_callback is not None:
        try:
            prev_tools = []
            for _idx, _m in enumerate(reversed(messages)):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _fwd_start = len(messages) - _idx
                    _results_by_id = {}
                    for _tm in messages[_fwd_start:]:
                        if _tm.get("role") != "tool":
                            break
                        _tcid = _tm.get("tool_call_id")
                        if _tcid:
                            _results_by_id[_tcid] = _tm.get("content", "")
                    prev_tools = [
                        {
                            "name": tc["function"]["name"],
                            "result": _results_by_id.get(tc.get("id")),
                            "arguments": tc["function"].get("arguments"),
                        }
                        for tc in _m["tool_calls"]
                        if isinstance(tc, dict)
                    ]
                    break
            agent.step_callback(api_call_count, prev_tools)
        except Exception as _step_err:
            logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

    # Track tool-calling iterations for skill nudge.
    # Counter resets whenever skill_manage is actually used.
    if (agent._skill_nudge_interval > 0
            and "skill_manage" in agent.valid_tool_names):
        agent._iters_since_skill += 1

    # ── Pre-API-call /steer drain ──────────────────────────────────
    # Drain a /steer sent during the last API call into the newest tool message so
    # it lands THIS iteration. Never put in a user message (breaks alternation).
    _pre_api_steer = agent._drain_pending_steer()
    if _pre_api_steer:
        _injected = False
        for _si in range(len(messages) - 1, -1, -1):
            _sm = messages[_si]
            if isinstance(_sm, dict) and _sm.get("role") == "tool":
                from agent.prompt_builder import format_steer_marker
                marker = format_steer_marker(_pre_api_steer)
                existing = _sm.get("content", "")
                if isinstance(existing, str):
                    _sm["content"] = existing + marker
                else:
                    # Multimodal content blocks — append text block
                    try:
                        blocks = list(existing) if existing else []
                        blocks.append({"type": "text", "text": marker})
                        _sm["content"] = blocks
                    except Exception:
                        pass
                _injected = True
                logger.debug(
                    "Pre-API-call steer drain: injected into tool msg at index %d",
                    _si,
                )
                break
        if not _injected:
            # No tool message to inject into — put it back so
            # the post-tool-execution drain picks it up later.
            _lock = getattr(agent, "_pending_steer_lock", None)
            if _lock is not None:
                with _lock:
                    if agent._pending_steer:
                        agent._pending_steer = agent._pending_steer + "\n" + _pre_api_steer
                    else:
                        agent._pending_steer = _pre_api_steer
            else:
                existing = getattr(agent, "_pending_steer", None)
                agent._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

    # ── Wall-clock run-budget wrap-up notice ───────────────────────
    # One-shot at 80% of agent.run_budget_seconds: ask the model to wrap up via the
    # same cache-safe channel as /steer (newest tool result); off with no budget.
    if getattr(agent, "run_budget_seconds", None):
        _maybe_inject_run_budget_wrapup(agent, messages)

    # Reasoning lives in content via <think> tags for trajectory storage, but some
    # providers (Moonshot) also need a 'reasoning_content' field; handle both here.
    request_logger = getattr(agent, "logger", None) or logger  # same name as the origin module
    # Per-agent validation cursor skips re-parsing tool_call args already validated.
    # Identity-keyed; a rewritten list breaks the prefix match and forces a re-scan.
    _sanitize_cursor = getattr(agent, "_sanitize_args_cursor", None)
    if _sanitize_cursor is None:
        _sanitize_cursor = {}
        try:
            agent._sanitize_args_cursor = _sanitize_cursor
        except Exception:
            pass
    repaired_tool_calls = agent._sanitize_tool_call_arguments(
        messages,
        logger=request_logger,
        session_id=agent.session_id,
        cursor=_sanitize_cursor,
    )
    if repaired_tool_calls > 0:
        request_logger.info(
            "Sanitized %s corrupted tool_call arguments before request (session=%s)",
            repaired_tool_calls,
            agent.session_id or "-",
        )

    # Drop legacy hidden assistant placeholders carrying the raw interrupt scaffold
    # before repair: replayed, the model echoes/self-replicates (#81841).
    messages = [
        msg for msg in messages
        if not (
            msg.get("display_kind") == "hidden"
            and msg.get("role") == "assistant"
            and (
                (
                    isinstance(msg.get("content"), str)
                    and msg["content"].strip() == _INTERRUPT_SCAFFOLD_MARKER
                )
                or (
                    isinstance(msg.get("api_content"), str)
                    and msg["api_content"].strip() == _INTERRUPT_SCAFFOLD_MARKER
                )
            )
        )
    ]

    # Repair malformed role alternation (tool→user / user→user tails): providers
    # return empty content on them and the empty-retry loop spins. The _with_cursor
    # variant also recomputes the SessionDB flush cursor after compaction (#44837).
    from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
    repaired_seq = repair_message_sequence_with_cursor(agent, messages)
    if repaired_seq > 0:
        request_logger.info(
            "Repaired %s message-alternation violations before request (session=%s)",
            repaired_seq,
            agent.session_id or "-",
        )
    return _verdict("fallthrough")
