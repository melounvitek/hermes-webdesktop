"""Empty / thinking-only final-response recovery ladder for the conversation turn loop.

Extracted from ``run_conversation``. Runs when the model returned no visible text after
``<think>`` blocks. Ladder order is load-bearing: partial-stream recovery → reuse prior
turn content (housekeeping tools only) → one post-tool-call nudge (#9400) → thinking-only
prefill continuation (×2) → empty-response retries (budgeted, deterministic-empty
short-circuit) → fallback provider → terminal ``(empty)`` sentinel. Nothing here imports
``agent.conversation_loop`` at module level (cycle); loop-internal helpers resolve lazily.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent import empty_response_guard as _empty_guard
from agent.message_metadata import append_message
from agent.turn_recovery import interruptible_backoff_sleep

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class EmptyResponseVerdict:
    """Outcome of ``recover_empty_response``.

    ``action``: ``"break"`` (turn is done — ``final_response`` is set), ``"continue"``
    (re-enter the OUTER turn loop: a nudge/prefill row was appended, a retry wait
    elapsed, or a fallback was activated and preflight must re-run), ``"return"``
    (interrupted during a retry wait — return ``result``) or ``"fallthrough"``
    (unreachable: every path exits; kept for the contract)."""

    action: str
    result: Optional[Dict[str, Any]]
    final_response: Any
    turn_exit_reason: Any
    active_system_prompt: Any
    preflight_compression_blocked: bool


def recover_empty_response(
    agent: Any,
    assistant_message: Any,
    response: Any,
    finish_reason: str,
    *,
    final_response: Any,
    messages: List[Dict[str, Any]],
    api_messages: Any,
    conversation_history: Any,
    active_system_prompt: Any,
    api_call_count: int,
    turn_exit_reason: Any,
    preflight_compression_blocked: bool,
) -> EmptyResponseVerdict:
    """Recover from a final response with no visible content (see module docstring for
    the ladder). Role alternation is preserved: the post-tool nudge appends the empty
    assistant row BEFORE the user-level hint (APIs reject tool→user). Reasoning is
    surfaced only at the terminal step, for delivery — the persisted row keeps the
    ``(empty)`` sentinel."""
    from agent.conversation_loop import (
        _EMPTY_TOOL_RESPONSE_NUDGE,
        _sync_failover_system_message,
        jittered_backoff,
    )

    _turn_exit_reason = turn_exit_reason
    _preflight_compression_blocked = preflight_compression_blocked

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> EmptyResponseVerdict:
        return EmptyResponseVerdict(
            action=action,
            result=result,
            final_response=final_response,
            turn_exit_reason=_turn_exit_reason,
            active_system_prompt=active_system_prompt,
            preflight_compression_blocked=_preflight_compression_blocked,
        )

    # Partial stream recovery: content streamed before the connection
    # died becomes the final response instead of fallback or retries.
    _partial_streamed = (
        getattr(agent, "_current_streamed_assistant_text", "") or ""
    )
    if agent._has_content_after_think_block(_partial_streamed):
        _turn_exit_reason = "partial_stream_recovery"
        _recovered = agent._strip_think_blocks(_partial_streamed).strip()
        logger.info(
            "Partial stream content delivered (%d chars) "
            "— using as final response",
            len(_recovered),
        )
        agent._emit_status(
            "↻ Stream interrupted — using delivered content "
            "as final response"
        )
        final_response = _recovered
        # A streamed fragment isn't a confirmed preview: keep
        # response_previewed false so gateway fallback delivery can
        # send the text plus the abnormal-turn explanation.
        agent._response_was_previewed = False
        return _verdict("break")

    # Prior turn had real content + ONLY housekeeping tools: model is
    # done, reuse it. With substantive tools it was mid-task narration
    # and the empty reply is a choke; let the post-tool nudge handle it.
    fallback = getattr(agent, '_last_content_with_tools', None)
    if fallback and getattr(agent, '_last_content_tools_all_housekeeping', False):
        _turn_exit_reason = "fallback_prior_turn_content"
        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
        agent._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        agent._empty_content_retries = 0
        # Do NOT modify the assistant message content (injected text
        # poisoned history); use the fallback as the response and break.
        final_response = agent._strip_think_blocks(fallback).strip()
        agent._response_was_previewed = True
        return _verdict("break")

    # ── Post-tool-call empty response nudge ───────────
    # Empty after tool results (no prior content, or only mid-task
    # narration): nudge once via a user-level hint. (#9400)
    _prior_was_tool = any(
        m.get("role") == "tool"
        for m in messages[-5:]  # check recent messages
    )
    # Ollama puts <think> in content, not reasoning_content, so
    # _has_structured misses it; detect here to route to prefill.
    _has_inline_thinking = bool(
        re.search(
            r'<think>|<thinking>|<reasoning>',
            final_response or "",
            re.IGNORECASE,
        )
    )
    if (
        _prior_was_tool
        and not getattr(agent, "_post_tool_empty_retried", False)
        and not _has_inline_thinking  # thinking model still working — let prefill handle
    ):
        agent._post_tool_empty_retried = True
        # Clear stale narration so it doesn't resurface
        # on a later empty response after the nudge.
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        logger.info(
            "Empty response after tool calls — nudging model "
            "to continue processing"
        )
        agent._buffer_status(
            "⚠️ Model returned empty after tool calls — "
            "nudging to continue"
        )
        # Append the empty assistant first so the sequence stays valid:
        # tool → assistant("(empty)") → user (APIs reject tool→user).
        _nudge_msg = agent._build_assistant_message(assistant_message, finish_reason)
        _nudge_msg["content"] = "(empty)"
        _nudge_msg["_empty_recovery_synthetic"] = True
        append_message(messages, _nudge_msg)
        append_message(messages, {
            "role": "user",
            "content": _EMPTY_TOOL_RESPONSE_NUDGE,
            "_empty_recovery_synthetic": True,
        })
        return _verdict("continue")

    # ── Thinking-only prefill continuation ──────────
    # Reasoning but no text: append as-is and continue so the model sees
    # its own reasoning and writes text. Covers _has_inline_thinking.
    _has_structured = bool(
        getattr(assistant_message, "reasoning", None)
        or getattr(assistant_message, "reasoning_content", None)
        or getattr(assistant_message, "reasoning_details", None)
        or _has_inline_thinking
    )
    if _has_structured and agent._thinking_prefill_retries < 2:
        agent._thinking_prefill_retries += 1
        logger.info(
            "Thinking-only response (no visible content) — "
            "prefilling to continue (%d/2)",
            agent._thinking_prefill_retries,
        )
        agent._buffer_status(
            f"↻ Thinking-only response — prefilling to continue "
            f"({agent._thinking_prefill_retries}/2)"
        )
        interim_msg = agent._build_assistant_message(
            assistant_message, "incomplete"
        )
        interim_msg["_thinking_prefill"] = True
        append_message(messages, interim_msg)
        agent._session_messages = messages
        return _verdict("continue")

    # ── Empty response retry ──────────────────────
    # Retry up to 3 times before fallback; covers truly empty replies
    # AND reasoning-only replies after prefill exhaustion.
    _truly_empty = not agent._strip_think_blocks(
        final_response
    ).strip()
    _prefill_exhausted = (
        _has_structured
        and agent._thinking_prefill_retries >= 2
    )
    _empty_candidate = _truly_empty and (
        not _has_structured or _prefill_exhausted
    )
    if _empty_candidate:
        # Each empty attempt re-bills the full input; record its
        # signature so deterministic empties stop burning paid retries.
        # Fails open: missing usage or any output keeps the budget.
        _empty_guard.record_empty_attempt(
            agent,
            finish_reason=finish_reason,
            response=response,
        )
    _empty_retry_budget = (
        _empty_guard.empty_retry_budget(agent, response)
        if _empty_candidate
        else _empty_guard.DEFAULT_EMPTY_RETRY_BUDGET
    )
    _deterministic_empty = _empty_candidate and (
        _empty_guard.deterministic_empty(agent)
    )
    if (
        _empty_candidate
        and agent._empty_content_retries < _empty_retry_budget
        and not _deterministic_empty
    ):
        agent._empty_content_retries += 1
        wait_time = jittered_backoff(
            agent._empty_content_retries,
            base_delay=5.0,
            max_delay=60.0,
        )
        logger.warning(
            "Empty response (no content or reasoning) — "
            "retry %d/%d in %.1fs (model=%s)",
            agent._empty_content_retries,
            _empty_retry_budget, wait_time, agent.model,
        )
        _budget_note = (
            " — high-cost request, reduced retry budget"
            if _empty_retry_budget < _empty_guard.DEFAULT_EMPTY_RETRY_BUDGET
            else ""
        )
        agent._buffer_status(
            f"⚠️ Empty response from model — retrying "
            f"({agent._empty_content_retries}/{_empty_retry_budget}) "
            f"in {wait_time:.0f}s{_budget_note}"
        )
        _interrupted = interruptible_backoff_sleep(
            agent, wait_time, None,
            messages=messages,
            conversation_history=conversation_history,
            api_call_count=api_call_count,
            abort_message="Interrupt detected during empty-response retry wait, aborting.",
            interrupt_text=(
                f"Operation interrupted: retrying empty response from model "
                f"(retry {agent._empty_content_retries}/{_empty_retry_budget})."
            ),
            activity_label=f"empty response retry backoff ({agent._empty_content_retries}/{_empty_retry_budget})",
        )
        if _interrupted is not None:
            return _verdict("return", _interrupted)
        return _verdict("continue")

    if _truly_empty and _deterministic_empty:
        logger.warning(
            "Deterministic empty response detected "
            "(consecutive zero-output completions, "
            "model=%s provider=%s finish_reason=%s) — "
            "skipping remaining retries",
            agent.model, agent.provider, finish_reason,
        )
        agent._buffer_status(
            "⚠️ Model is deterministically returning empty "
            "(zero output tokens) — skipping further retries "
            "to avoid repeat charges"
        )

    # ── Exhausted retries — try fallback provider ──
    # Before "(empty)", switch to the next provider in the chain.
    if _truly_empty and agent._fallback_chain:
        logger.warning(
            "Empty response after %d retries — "
            "attempting fallback (model=%s, provider=%s)",
            agent._empty_content_retries, agent.model,
            agent.provider,
        )
        agent._buffer_status(
            "⚠️ Model returning empty responses — "
            "switching to fallback provider..."
        )
        if agent._try_activate_fallback():
            active_system_prompt = _sync_failover_system_message(
                agent, api_messages, active_system_prompt)
            agent._empty_content_retries = 0
            agent._buffer_status(
                f"↻ Switched to fallback: {agent.model} "
                f"({agent.provider})"
            )
            logger.info(
                "Fallback activated after empty responses: "
                "now using %s on %s",
                agent.model, agent.provider,
            )
            # OUTER loop: `continue` re-runs preflight against the
            # fallback's window; `break` would end the turn without
            # calling the fallback. Clear the preflight block. (#84733)
            _preflight_compression_blocked = False
            return _verdict("continue")

    # Retries and fallback exhausted — fall through to "(empty)".
    # Surface the buffered retry trace and, if known, what the empty
    # streak cost (each attempt re-billed the full input).
    _streak_cost = _empty_guard.streak_cost_usd(agent)
    if _streak_cost is not None:
        agent._buffer_status(
            f"ℹ️ Estimated cost of these empty attempts: "
            f"~${_streak_cost:.2f} (input tokens are billed "
            f"per attempt even when no answer is produced)"
        )
    agent._flush_status_buffer()
    _turn_exit_reason = "empty_response_exhausted"
    reasoning_text = agent._extract_reasoning(assistant_message)
    agent._drop_trailing_empty_response_scaffolding(messages)
    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
    assistant_msg["content"] = "(empty)"
    # Gateway failure sentinel, not content: persisting it lets later
    # "continue" turns replay assistant("(empty)") and loop on empties.
    assistant_msg["_empty_terminal_sentinel"] = True
    append_message(messages, assistant_msg)

    if reasoning_text:
        reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
        logger.warning(
            "Reasoning-only response (no visible content) "
            "after exhausting retries and fallback. "
            "Reasoning: %s", reasoning_preview,
        )
        agent._emit_status(
            "⚠️ Model produced reasoning but no visible "
            "response after all retries. Returning empty."
        )
    else:
        logger.warning(
            "Empty response (no content or reasoning) "
            "after %d retries. No fallback available. "
            "model=%s provider=%s",
            agent._empty_content_retries, agent.model,
            agent.provider,
        )
        agent._emit_status(
            "❌ Model returned no content after all retries"
            + (" and fallback attempts." if agent._fallback_chain else
               ". No fallback providers configured.")
        )

    # Delivery-only: show labeled reasoning instead of bare "(empty)"
    # when the model thought but wrote no text. The persisted row keeps
    # the sentinel; reasoning is never promoted earlier in the ladder.
    if reasoning_text:
        final_response = (
            "⚠️ The model produced only internal reasoning and "
            "no final answer, despite retries"
            + (" and fallback" if agent._fallback_chain else "")
            + ". Its last reasoning, which may contain the "
            "answer:\n\n" + reasoning_preview
        )
    else:
        final_response = "(empty)"
    return _verdict("break")
    return _verdict("fallthrough")
