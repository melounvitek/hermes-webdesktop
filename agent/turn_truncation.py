"""Truncation recovery (``finish_reason == "length"``) for the conversation turn loop.

Extracted from ``run_conversation``. Handles thinking-budget exhaustion, repetition-
dominated truncation (#86581), content-filter stream stalls escalated to the fallback
chain (#32421), text continuation nudges (up to 4, with the ceiling exit that drops the
fragment trail), truncated tool-call retries with max_tokens boosts, and the final
roll-back. Nothing here imports ``agent.conversation_loop`` at module level (cycle);
loop-internal helpers are imported lazily so tests patching them on the loop keep working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.error_classifier import FailoverReason
from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence
from agent.repetition_guard import is_repetition_dominated
from agent.turn_retry_state import TurnRetryState
from hermes_constants import PARTIAL_STREAM_STUB_ID

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class TruncationVerdict:
    """Outcome of ``recover_from_truncation``.

    ``action``: ``"return"`` (end the turn with ``result``), ``"break"`` (a
    ``_retry.restart_with_*`` flag is set — restart the API call), ``"continue"``
    (re-issue the same call immediately) or ``"fallthrough"`` (unreachable in practice:
    every path exits, kept for the contract). The remaining fields are the loop locals
    the handler may have rebound."""

    action: str
    result: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    length_continue_retries: int
    truncated_response_parts: List[str]
    truncated_tool_call_retries: int
    retry_count: int
    compression_attempts: int


def recover_from_truncation(
    agent: Any, response: Any, finish_reason: str, _retry: TurnRetryState, *,
    messages: List[Dict[str, Any]], conversation_history: Any, api_kwargs: Any, api_call_count: int,
    effective_task_id: Any, current_turn_user_idx: Any, length_continue_retries: int,
    truncated_response_parts: List[str], truncated_tool_call_retries: int, retry_count: int,
    compression_attempts: int,
) -> TruncationVerdict:
    """Recover from a truncated response. Order is load-bearing: thinking exhaustion and
    repetition abort BEFORE any continuation; a content-filter stall escalates to the
    fallback chain BEFORE the primary is retried; text continuation (no tool calls) then
    truncated tool-call retry; finally roll back to the last complete assistant turn.
    Never appends an interim assistant row with NO visible content (strict providers
    reject it with 400) — only the continuation nudge."""
    from agent.conversation_loop import _get_continuation_prompt, _join_truncated_parts

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> TruncationVerdict:
        return TruncationVerdict(
            action=action, result=result, messages=messages,
            length_continue_retries=length_continue_retries,
            truncated_response_parts=truncated_response_parts,
            truncated_tool_call_retries=truncated_tool_call_retries, retry_count=retry_count,
            compression_attempts=compression_attempts,
        )

    if getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Response truncated — stream "
            f"ended before completion",
            force=True,
        )
    else:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Response truncated "
            f"(finish_reason='length') - model hit max output tokens",
            force=True,
        )

    # Normalize to one OpenAI-style message so continuation and tool-
    # call retry work across transports (Anthropic reuses the loop's
    # adapter).
    _trunc_msg = None
    _trunc_transport = agent._get_transport()
    if agent.api_mode == "anthropic_messages":
        _trunc_result = _trunc_transport.normalize_response(
            response, strip_tool_prefix=agent._is_anthropic_oauth
        )
    else:
        _trunc_result = _trunc_transport.normalize_response(response)
    _trunc_msg = _trunc_result

    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

    # ── Detect thinking-budget exhaustion ──────────────
    # Only when reasoning blocks exist with no visible text after them;
    # content=None from non-<think> models is normal truncation.
    _has_think_tags = bool(
        _trunc_content and re.search(
            r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
            _trunc_content,
            re.IGNORECASE,
        )
    )
    _thinking_exhausted = (
        not _trunc_has_tool_calls
        and _has_think_tags
        and (
            (_trunc_content is not None and not agent._has_content_after_think_block(_trunc_content))
            or _trunc_content is None
        )
    )

    if _thinking_exhausted:
        _exhaust_error = (
            "Model used all output tokens on reasoning with none left "
            "for the response. Try lowering reasoning effort or "
            "increasing max_tokens."
        )
        agent._vprint(
            f"{agent.log_prefix}💭 Reasoning exhausted the output token budget — "
            f"no visible response was produced.",
            force=True,
        )
        # Return a user-friendly message as the response so CLI and
        # gateway display it.
        _exhaust_response = (
            "⚠️ **Thinking Budget Exhausted**\n\n"
            "The model used all its output tokens on reasoning "
            "and had none left for the actual response.\n\n"
            "To fix this:\n"
            "→ Lower reasoning effort: `/reasoning low` or `/reasoning minimal`\n"
            "→ Or switch to a larger/non-reasoning model with `/model`"
        )
        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", {
            "final_response": _exhaust_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": False,
            "partial": True,
            "error": _exhaust_error,
        })

    # ── Detect repetition-dominated truncation (#86581) ──
    # A repetition loop can burn the whole budget on one fragment; abort
    # like _thinking_exhausted (reasoning stripped first).
    _visible_trunc = (
        agent._strip_think_blocks(_trunc_content)
        if isinstance(_trunc_content, str)
        else _trunc_content
    )
    _repetition_dominated = (
        not _trunc_has_tool_calls
        and bool(_visible_trunc)
        and is_repetition_dominated(_visible_trunc)
    )
    if _repetition_dominated:
        _rep_error = (
            "Model output entered a repetition loop and was "
            "truncated mid-loop; refusing to continue a "
            "degenerate response."
        )
        agent._vprint(
            f"{agent.log_prefix}🔁 Response dominated by "
            f"repeated text — stopping instead of "
            f"continuing a degenerate response.",
            force=True,
        )
        _rep_response = (
            "⚠️ **Response Stopped — Repetition Detected**\n\n"
            "The model fell into a repetition loop while "
            "writing this response, so continuing would only "
            "produce more repeated text. The partial response "
            "was discarded.\n\n"
            "→ Switch to a different model with `/model`\n"
            "→ Or resend your message (your conversation "
            "history is preserved)"
        )
        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", {
            "final_response": _rep_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": False,
            "partial": True,
            "error": _rep_error,
        })

    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
        assistant_message = _trunc_msg
        # ── Content-filter stream stall → fallback (#32421) ──
        # ``_content_filter_terminated`` is content-deterministic;
        # escalate to the fallback before retrying the primary.
        _cf_terminated = getattr(
            response, "_content_filter_terminated", False
        )
        if (
            _cf_terminated and agent._fallback_index < len(agent._fallback_chain)
        ):
            agent._vprint(
                f"{agent.log_prefix}🛡️  Content filter terminated "
                f"stream — activating fallback provider...",
                force=True,
            )
            agent._emit_status(
                "Content filter terminated stream; switching to fallback..."
            )
            if agent._try_activate_fallback():
                # Roll partial content back to the last clean turn so
                # the fallback gets a coherent continuation point.
                if truncated_response_parts:
                    messages = agent._get_messages_up_to_last_assistant(messages)
                # Unmark survivors: their text left the stitched partial.
                for _frag in messages:
                    if isinstance(_frag, dict):
                        _frag.pop("_length_continuation_fragment", None)
                        _frag.pop("_length_continuation_nudge", None)
                agent._session_messages = messages
                length_continue_retries = 0
                truncated_response_parts = []
                retry_count = 0
                compression_attempts = 0
                _retry.primary_recovery_attempted = False
                _retry.restart_with_rebuilt_messages = True
                return _verdict("break")
            # No fallback available — fall through to normal
            # continuation (best-effort, may loop).
            agent._vprint(
                f"{agent.log_prefix}⚠️  No fallback provider "
                f"configured — retrying with same provider "
                f"(may re-hit filter)...",
                force=True,
            )
        if assistant_message is not None and not _trunc_has_tool_calls:
            length_continue_retries += 1
            # Never append an interim assistant message with NO visible
            # content: strict providers reject it (HTTP 400), poisoning
            # history. Append only the nudge.
            _interim_content = getattr(assistant_message, "content", None)
            _is_empty_partial_stub = (
                getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID and not _interim_content
            )
            if not _interim_content and not _is_empty_partial_stub:
                # Thinking-only truncation: continuing with thinking ON
                # re-burns the budget, so drop thinking for one request.
                agent._ephemeral_reasoning_off = True
            if _interim_content:
                interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                # Marked so the ceiling exit can drop the fragment trail.
                interim_msg["_length_continuation_fragment"] = True
                append_message(messages, interim_msg)
                truncated_response_parts.append(_interim_content)

            if length_continue_retries < 4:
                _is_partial_stream_stub = (
                    getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                )
                _dropped_tools = getattr(
                    response, "_dropped_tool_names", None
                )

                if _is_partial_stream_stub and _dropped_tools:
                    _tool_list = ", ".join(_dropped_tools[:3])
                    agent._vprint(
                        f"{agent.log_prefix}↻ Stream interrupted mid "
                        f"tool-call ({_tool_list}) — requesting "
                        f"chunked retry "
                        f"({length_continue_retries}/4)..."
                    )
                elif _is_partial_stream_stub:
                    agent._vprint(
                        f"{agent.log_prefix}↻ Stream interrupted — "
                        f"requesting continuation "
                        f"({length_continue_retries}/4)..."
                    )
                else:
                    agent._vprint(
                        f"{agent.log_prefix}↻ Requesting continuation "
                        f"({length_continue_retries}/4)..."
                    )

                _continue_content = _get_continuation_prompt(
                    _is_partial_stream_stub, _dropped_tools
                )
                continue_msg = {
                    "role": "user", "content": _continue_content, "_length_continuation_nudge": True
                }
                append_message(messages, continue_msg)
                agent._session_messages = messages
                _retry.restart_with_length_continuation = True
                return _verdict("break")

            partial_response = agent._strip_think_blocks(_join_truncated_parts(truncated_response_parts)).strip()
            # The one-shot reasoning-off override must not leak into the
            # next turn when the ceiling exit skips the consuming call.
            agent._ephemeral_reasoning_off = False
            if partial_response:
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Response still truncated "
                    f"after {length_continue_retries} continuation attempts — keeping the "
                    f"partial response received so far.",
                    force=True,
                )
                _ceiling_final = partial_response
            else:
                # Every fragment was empty (e.g. reasoning-only model):
                # return an actionable message, not a bare None.
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Response still truncated "
                    f"after {length_continue_retries} continuation attempts — no visible "
                    f"text was produced.",
                    force=True,
                )
                _ceiling_final = (
                    "⚠️ **No visible answer was produced.** The "
                    "model hit its output-token limit on every "
                    "continuation attempt — its reasoning "
                    "consumed the entire budget each time.\n\n"
                    "To fix this:\n"
                    "→ Lower reasoning effort: `/reasoning low` "
                    "or `/reasoning none`\n"
                    "→ Or raise max_tokens for this model"
                )
            # Unanswered continue nudges made every later turn re-truncate.
            _turn_start = (
                current_turn_user_idx + 1
                if isinstance(current_turn_user_idx, int)
                and current_turn_user_idx >= 0
                else 0
            )
            messages[_turn_start:] = [
                m for m in messages[_turn_start:]
                if not (
                    isinstance(m, dict)
                    and (
                        m.get("_length_continuation_fragment")
                        or m.get("_length_continuation_nudge")
                    )
                )
            ]
            if partial_response:
                append_message(messages, {
                    "role": "assistant", "content": partial_response, "finish_reason": "length"
                })
            agent._session_messages = messages
            agent._cleanup_task_resources(effective_task_id)
            agent._persist_session(messages, conversation_history)
            return _verdict("return", {
                "final_response": _ceiling_final,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "partial": True,
                "error": "Response remained truncated after 4 continuation attempts",
            })

    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
        assistant_message = _trunc_msg
        if assistant_message is not None and _trunc_has_tool_calls:
            _is_stub_stall = (
                getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
            )
            if truncated_tool_call_retries < 4:
                truncated_tool_call_retries += 1
                if _is_stub_stall:
                    # Stream broke mid tool-call (network), not a real
                    # output cap — say so.
                    agent._buffer_vprint(
                        f"⚠️  Stream interrupted mid tool-call — "
                        f"retrying ({truncated_tool_call_retries}/4)..."
                    )
                else:
                    agent._buffer_vprint(
                        f"⚠️  Truncated tool call detected — "
                        f"retrying API call "
                        f"({truncated_tool_call_retries}/4)..."
                    )
                # Boost max_tokens per retry: a real output-cap
                # truncation needs it; harmless for a stall.
                _tc_boost_base = agent.max_tokens if agent.max_tokens else 4096
                _tc_boost = _tc_boost_base * (2 ** truncated_tool_call_retries)
                _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
                if _tc_requested_cap is not None:
                    _tc_boost = max(_tc_boost, _tc_requested_cap)
                _tc_boost_cap = max(32768, _tc_requested_cap or 0)
                agent._ephemeral_max_output_tokens = min(_tc_boost, _tc_boost_cap)
                # Don't append the broken response; re-run the same call
                # from current state.
                return _verdict("continue")
            agent._flush_status_buffer()
            if _is_stub_stall:
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Stream kept dropping mid tool-call after 4 retries — the action was not executed.",
                    force=True,
                )
            else:
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                    force=True,
                )
            agent._cleanup_task_resources(effective_task_id)
            _final_response = (
                "Stream repeatedly dropped mid tool-call (network); "
                "the tool was not executed"
                if _is_stub_stall
                else "Response truncated due to output length limit"
            )
            # Prior tool batches can leave a tool-result tail; this path
            # never reaches finalize_turn (#48879).
            close_interrupted_tool_sequence(messages, _final_response)
            agent._persist_session(messages, conversation_history)
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "partial": True,
                "error": _final_response,
            })

    # If we have prior messages, roll back to last complete state
    if len(messages) > 1:
        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)

        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)

        return _verdict("return", {
            "final_response": "Response truncated due to output length limit",
            "messages": rolled_back_messages,
            "api_calls": api_call_count,
            "completed": False,
            "partial": True,
            "error": "Response truncated due to output length limit"
        })
    else:
        # First message was truncated - mark as failed
        agent._flush_status_buffer()
        agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", {
            "final_response": "First response truncated due to output length limit",
            "messages": messages,
            "api_calls": api_call_count,
            "completed": False,
            "failed": True,
            "error": "First response truncated due to output length limit"
        })
    return _verdict("fallthrough")


def continue_codex_incomplete(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int,
) -> Optional[Dict[str, Any]]:
    """Codex Responses ``status=incomplete`` continuation (max 3 per turn).

    Appends the interim assistant message (deduped on visible content only — opaque
    provider state drifts per continuation, #52711; ``codex_reasoning_items`` are merged,
    not overwritten, because the earlier response holds the only native-compaction
    checkpoint) and, when a bare retry would be byte-identical, a user-role nudge — only
    after an assistant row, to preserve role alternation. Returns ``None`` to continue
    the turn loop, or the terminal ``partial`` result once retries are exhausted."""
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE

    agent._codex_incomplete_retries += 1

    interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
    interim_has_content = bool((interim_msg.get("content") or "").strip())
    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
    interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

    if (
        interim_has_content
        or interim_has_reasoning
        or interim_has_codex_reasoning
        or interim_has_codex_message_items
    ):
        last_msg = messages[-1] if messages else None
        # Dedup on visible content only (content + reasoning): opaque
        # provider state drifts per continuation and would defeat dedup
        # (#52711).
        last_interim_visible = (
            agent._interim_assistant_visible_text(last_msg) if isinstance(last_msg, dict) else ""
        )
        current_interim_visible = agent._interim_assistant_visible_text(interim_msg)
        if last_interim_visible or current_interim_visible:
            same_visible_output = last_interim_visible == current_interim_visible
        else:
            # Preserve the existing reasoning-only behavior when
            # neither response has text eligible for interim delivery.
            same_visible_output = (
                (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
            ) if isinstance(last_msg, dict) else False
        visible_duplicate = (
            isinstance(last_msg, dict)
            and last_msg.get("role") == "assistant"
            and last_msg.get("finish_reason") == "incomplete"
            and same_visible_output
        )
        if visible_duplicate:
            # Update replay state in-place: keep the latest provider payload
            # without re-emitting identical user-visible commentary.
            for _key in (
                "content",
                "reasoning",
                "reasoning_content",
                "reasoning_details",
                "codex_reasoning_items",
                "codex_message_items",
            ):
                if _key in interim_msg:
                    if _key == "codex_reasoning_items":
                        # Merge, don't overwrite: the earlier response's
                        # native compaction checkpoint is the only copy. See
                        # merge_interim_reasoning_items.
                        from agent.native_compaction import (
                            merge_interim_reasoning_items,
                        )
                        last_msg[_key] = merge_interim_reasoning_items(
                            last_msg.get(_key), interim_msg[_key]
                        )
                    else:
                        last_msg[_key] = interim_msg[_key]
        else:
            append_message(messages, interim_msg)
            agent._emit_interim_assistant_message(interim_msg)

    if agent._codex_incomplete_retries < 3:
        # If the interim has nothing the Responses converter will replay, a
        # bare retry is byte-identical and fails identically; append a
        # user-role nudge so the retry differs and asks for the answer.
        interim_replayable = (
            interim_has_content or interim_has_codex_reasoning or interim_has_codex_message_items
        )
        # Replayable ≠ different: an interim holding only a ``compaction``
        # checkpoint in ``codex_reasoning_items`` is replayable yet re-sends
        # identically. One bare retry, then always nudge.
        if not interim_replayable or agent._codex_incomplete_retries >= 2:
            _last_msg = messages[-1] if messages else None
            _already_nudged = (
                isinstance(_last_msg, dict)
                and _last_msg.get("role") == "user"
                and _last_msg.get("content") == _CODEX_INCOMPLETE_NUDGE
            )
            # Alternation guard: the user-role nudge may only follow an
            # assistant message; after a too-empty interim it would create
            # user→user / tool→user.
            _last_is_assistant = (
                isinstance(_last_msg, dict) and _last_msg.get("role") == "assistant"
            )
            if not _already_nudged and _last_is_assistant:
                append_message(messages, {
                    "role": "user", "content": _CODEX_INCOMPLETE_NUDGE
                })
        if not agent.quiet_mode:
            agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({agent._codex_incomplete_retries}/3)")
        # Show the continuation on the spinner/status line and gateway
        # heartbeat; these retries can take minutes and otherwise look like
        # infinite thinking (#64434).
        agent._emit_wait_notice(
            f"↻ model returned reasoning with no final answer — "
            f"asking it to continue "
            f"({agent._codex_incomplete_retries}/3)"
        )
        agent._session_messages = messages
        return None

    agent._codex_incomplete_retries = 0
    agent._persist_session(messages, conversation_history)
    return {
        "final_response": "Codex response remained incomplete after 3 continuation attempts",
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "partial": True,
        "error": "Codex response remained incomplete after 3 continuation attempts",
    }


@dataclass
class RefusalVerdict:
    """Outcome of ``handle_content_policy_refusal``: ``"break"`` (fallback activated —
    restart armed on ``_retry``; caller resets retry/compression counters) or
    ``"return"`` (the typed content-policy result in ``result``). ``active_system_prompt``
    is the possibly re-synced system prompt."""

    action: str
    result: Optional[Dict[str, Any]]
    active_system_prompt: Any


def handle_content_policy_refusal(
    agent: Any, response: Any, _retry: TurnRetryState, *, thinking_spinner: Any,
    messages: List[Dict[str, Any]], api_messages: Any, api_kwargs: Any, active_system_prompt: Any,
    conversation_history: Any, api_call_count: int, effective_task_id: Any, turn_id: Any,
    api_request_id: Any, api_start_time: float, retry_count: int, max_retries: int,
) -> RefusalVerdict:
    """HTTP-200 refusal (``finish_reason`` ``content_filter`` / ``guardrail_intervened``).
    Deterministic for the unchanged prompt — never retried: one configured-fallback try,
    else surface the refusal (explanation may live only in the reasoning channel). The
    caller stops its spinner reference; this stops the spinner object."""
    from agent.conversation_loop import (
        _CONTENT_POLICY_RECOVERY_HINT, _arm_fallback_restart, _content_policy_blocked_result
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> RefusalVerdict:
        return RefusalVerdict(action=action, result=result, active_system_prompt=active_system_prompt)

    _refusal_transport = agent._get_transport()
    if agent.api_mode == "anthropic_messages":
        _refusal_result = _refusal_transport.normalize_response(
            response, strip_tool_prefix=agent._is_anthropic_oauth
        )
    else:
        _refusal_result = _refusal_transport.normalize_response(response)
    _refusal_text = (getattr(_refusal_result, "content", None) or "").strip()
    # Some refusals carry the explanation only in the reasoning
    # channel; fall back to it so the user sees *something*.
    if not _refusal_text:
        _refusal_text = (agent._extract_reasoning(_refusal_result) or "").strip()

    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type="ContentPolicyBlocked",
        error_message=_refusal_text or "model declined to respond (content_filter)",
        status_code=None, retry_count=retry_count, max_retries=max_retries, retryable=False,
        reason=FailoverReason.content_policy_blocked.value,
    )

    if thinking_spinner:
        thinking_spinner.stop("")
    if agent.thinking_callback:
        agent.thinking_callback("")

    # Deterministic for the unchanged prompt — never retry. Try a
    # configured fallback once; otherwise surface the refusal.
    if agent._has_pending_fallback():
        agent._buffer_status(
            "⚠️ Model declined to respond (safety refusal) — trying fallback..."
        )
    if agent._try_activate_fallback():
        active_system_prompt = _arm_fallback_restart(
            agent, api_messages, active_system_prompt, _retry)
        return _verdict("break")

    agent._flush_status_buffer()
    _refusal_log = (
        _refusal_text[:500] + "..." if len(_refusal_text) > 500 else _refusal_text
    )
    logger.warning(
        "%sModel declined to respond (finish_reason=content_filter). "
        "model=%s provider=%s refusal=%s",
        agent.log_prefix, agent.model, agent.provider,
        _refusal_log or "(no text)",
    )
    agent._emit_status(
        "⚠️ The model declined to respond to this request (safety refusal)."
    )

    _refusal_detail = (
        f"Model's explanation: {_refusal_text}"
        if _refusal_text
        else "The model returned no explanation."
    )
    _refusal_response = (
        "⚠️  The model declined to respond to this request "
        "(safety refusal — not a Hermes/gateway failure).\n\n"
        f"{_refusal_detail}\n\n"
        f"{_CONTENT_POLICY_RECOVERY_HINT}"
    )

    agent._cleanup_task_resources(effective_task_id)
    agent._persist_session(messages, conversation_history)
    return _verdict("return", _content_policy_blocked_result(
        messages, api_call_count, final_response=_refusal_response,
        error_detail=_refusal_text or "model declined (content_filter)",
    ))
