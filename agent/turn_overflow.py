"""Overflow recovery for the conversation turn loop: 413 payload-too-large and
context-length errors after ``classify_api_error``.

Extracted from ``run_conversation``'s ``except`` branch. Each path either compresses and
signals a restart, defers softly (compression lock / transient block), or ends the turn
with a typed result. Nothing here imports ``agent.conversation_loop`` at module level
(cycle); loop-internal helpers and the token estimators that tests patch on the loop
module are imported lazily inside the handler so they keep resolving through the loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.conversation_compression import (
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE,
    compression_blocked_transiently,
    compression_skipped_due_to_lock,
    context_compression_timed_out,
)
from agent.error_classifier import FailoverReason
from agent.message_sanitization import serialized_messages_bytes
from agent.model_metadata import (
    get_context_length_from_provider_error,
    is_output_cap_error,
    parse_available_output_tokens_from_error,
)
from agent.turn_retry_state import TurnRetryState
from utils import base_url_host_matches

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class OverflowVerdict:
    """Outcome of ``recover_from_overflow``.

    ``action`` is one of ``"return"`` (end the turn with ``result``), ``"break"``
    (restart the API call — a ``_retry.restart_with_*`` flag is set), ``"continue"``
    (retry the call immediately) or ``"fallthrough"`` (not an overflow error, or
    overflow recovery declined — continue generic error handling). The remaining
    fields are the loop locals the handler may have rebound."""

    action: str
    result: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    active_system_prompt: Any
    conversation_history: Any
    approx_tokens: int
    compression_attempts: int
    provider_overflow_recovery_pending: bool
    is_context_length_error: bool


def recover_from_overflow(
    agent: Any,
    api_error: Exception,
    classified: Any,
    _retry: TurnRetryState,
    *,
    status_code: Optional[int],
    error_msg: str,
    wrapped_output_cap_budget: Optional[int],
    messages: List[Dict[str, Any]],
    api_messages: Any,
    system_message: Any,
    active_system_prompt: Any,
    conversation_history: Any,
    approx_tokens: int,
    compression_attempts: int,
    max_compression_attempts: int,
    api_call_count: int,
    effective_task_id: Any,
) -> OverflowVerdict:
    """413 payload-too-large and context-length recovery (compress + retry, output-cap
    clamp, provider-reported context limit, GitHub Models free-tier hint). Order is
    load-bearing: 413 is checked BEFORE the generic 4xx handler, and context-length
    errors (incl. relay-wrapped output-cap 429s) BEFORE non-retryable client errors.
    Compression progress is scored in payload BYTES for 413 (never the byte-blind token
    estimate) and in tokens/message count for context overflow."""
    # Token estimators + loop-internal helpers resolve through the loop module so
    # existing ``patch("agent.conversation_loop.X")`` mocks keep intercepting.
    from agent.conversation_loop import (
        _COMPRESSION_TIMEOUT_FINAL_RESPONSE,
        _compression_deferred_result,
        conversation_history_after_compression,
        estimate_messages_tokens_rough,
        estimate_request_tokens_rough,
        save_context_length,
    )

    _provider_overflow_recovery_pending = False
    is_context_length_error = False
    _wrapped_output_cap_budget = wrapped_output_cap_budget

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> OverflowVerdict:
        return OverflowVerdict(
            action=action,
            result=result,
            messages=messages,
            active_system_prompt=active_system_prompt,
            conversation_history=conversation_history,
            approx_tokens=approx_tokens,
            compression_attempts=compression_attempts,
            provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
            is_context_length_error=is_context_length_error,
        )

    is_payload_too_large = (
        classified.reason == FailoverReason.payload_too_large
    )

    # GitHub Models free tier caps requests at 8K tokens, under the system
    # prompt + tool schema floor; compression can't help, so say so.
    if (
        status_code == 413
        and isinstance(agent.base_url, str)
        and base_url_host_matches(agent.base_url, "models.inference.ai.azure.com")
    ):
        agent._vprint(
            f"{agent.log_prefix}   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      exceeds that floor, so this endpoint cannot run an agentic loop.",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      Use the `copilot` provider with a Copilot subscription token (`hermes",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      setup` → GitHub Copilot), or pick any other provider.",
            force=True,
        )

    if is_payload_too_large:
        compression_attempts += 1
        if compression_attempts > max_compression_attempts:
            # Terminal — surface the buffered retry trace.
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
            logger.error("%s413 compression failed after %d attempts.", agent.log_prefix, max_compression_attempts)
            agent._persist_session(messages, conversation_history)
            _final_response = f"Request payload too large: max compression attempts ({max_compression_attempts}) reached."
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "completed": False,
                "api_calls": api_call_count,
                "error": _final_response,
                "partial": True,
                "failed": True,
                "compression_exhausted": True,
            })
        agent._buffer_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

        original_len = len(messages)
        # A 413 is a BYTE-size error: score progress in payload bytes,
        # never the token estimate, which is deliberately byte-blind to
        # images and wedged sessions on "no progress" (#88960 / #47339).
        original_bytes = serialized_messages_bytes(messages)
        _overflow_input = messages
        # Option A (LCM issue 441): overhead-aware request size so recovery arms on the
        # true request (msgs + tools + system), not the tool-blind message count.
        messages, active_system_prompt = agent._compress_context(
            messages, system_message,
            approx_tokens=estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
            task_id=effective_task_id,
            # Provider proved the request doesn't fit: ignore the
            # summary-failure cooldown for this ONE attempt (#100661).
            bypass_cooldown=True,
        )
        if messages is _overflow_input and compression_skipped_due_to_lock(agent):
            # Lock-skip: another path holds the compression lock. A
            # temporary defer, not exhaustion — refund the attempt and
            # end softly so the gateway does NOT auto-reset (#69870).
            compression_attempts -= 1
            agent._persist_session(messages, conversation_history)
            return _verdict("return", _compression_deferred_result(
                agent, messages, api_call_count
            ))
        if messages is _overflow_input and compression_blocked_transiently(agent):
            # Transient-block: a timed guard no-oped compression. A
            # defer, never compression_exhausted (auto-reset) (#97488).
            compression_attempts -= 1
            agent._persist_session(messages, conversation_history)
            return _verdict("return", _compression_deferred_result(
                agent, messages, api_call_count,
                reason="transient_block",
            ))
        conversation_history = conversation_history_after_compression(
            agent, messages, conversation_history
        )

        # Re-measure: same-count compression and media aging can shrink
        # the request without shrinking the array. Bytes are the yardstick
        # for a 413; tokens only for status display.
        new_tokens = estimate_messages_tokens_rough(messages)
        approx_tokens = new_tokens  # update for downstream logging
        new_bytes = serialized_messages_bytes(messages)

        made_progress = (
            len(messages) < original_len
            or (new_bytes > 0 and new_bytes < original_bytes * 0.95)
        )
        if made_progress:
            if len(messages) < original_len:
                agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
            else:
                agent._buffer_status(
                    f"🗜️ Compressed {original_bytes:,} → {new_bytes:,} "
                    f"payload bytes, retrying..."
                )
            time.sleep(2)  # Brief pause between compression retries
            _retry.restart_with_compressed_messages = True
            return _verdict("break")
        else:
            if agent._try_strip_image_parts_from_tool_messages(
                api_messages,
                remember_model=False,
            ):
                agent._buffer_status(
                    "📐 Compression could not reduce the request further — "
                    "removed retained vision payloads and retrying..."
                )
                return _verdict("continue")

            # Terminal — surface buffered context so the user
            # sees what compression attempts were made.
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Payload too large and cannot compress further.", force=True)
            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
            logger.error("%s413 payload too large. Cannot compress further.", agent.log_prefix)
            agent._persist_session(messages, conversation_history)
            _final_response = "Request payload too large (413). Cannot compress further."
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "completed": False,
                "api_calls": api_call_count,
                "error": _final_response,
                "partial": True,
                "failed": True,
                "compression_exhausted": True,
            })

    # Check context-length errors BEFORE the generic 4xx handler; the
    # classifier also covers 400/disconnect + large-session heuristics.
    is_context_length_error = (
        classified.reason == FailoverReason.context_overflow
        # Relay-wrapped output-cap 429s (parsed above) go to the clamp
        # below, not failover or generic retries (#72281).
        or _wrapped_output_cap_budget is not None
    )

    if is_context_length_error:
        compressor = agent.context_compressor
        old_ctx = compressor.context_length

        # Two errors: "prompt too long" = INPUT overflows the window (shrink
        # context_length + compress); "max_tokens too large" = input fits
        # but input + max_tokens > window (shrink OUTPUT cap only).
        available_out = parse_available_output_tokens_from_error(error_msg)
        if available_out is not None:
            # Output-cap error: provider available_tokens is the
            # authoritative bound; also estimate the real request shape
            # (API-only content), use the smaller minus a margin.
            request_input_estimate = estimate_request_tokens_rough(
                api_messages, tools=agent.tools or None,
            )
            local_available_out = old_ctx - request_input_estimate
            if local_available_out > 0:
                safe_out = max(1, min(available_out, local_available_out) - 64)
            else:
                # Local estimate can overshoot; fall back to the
                # authoritative provider-reported budget.
                safe_out = max(1, available_out - 64)
            agent._ephemeral_max_output_tokens = safe_out
            agent._buffer_vprint(
                f"⚠️  Output cap too large for current prompt — "
                f"retrying with max_tokens={safe_out:,} "
                f"(provider_available={available_out:,}, "
                f"estimated_request_tokens={request_input_estimate:,}; "
                f"context_length unchanged at {old_ctx:,})"
            )
            # Still count against compression_attempts so we don't
            # loop forever if the error keeps recurring.
            compression_attempts += 1
            if compression_attempts > max_compression_attempts:
                agent._flush_status_buffer()
                agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                logger.error("%sContext compression failed after %d attempts.", agent.log_prefix, max_compression_attempts)
                agent._persist_session(messages, conversation_history)
                _final_response = f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached."
                return _verdict("return", {
                    "final_response": _final_response,
                    "messages": messages,
                    "completed": False,
                    "api_calls": api_call_count,
                    "error": _final_response,
                    "partial": True,
                    "failed": True,
                    "compression_exhausted": True,
                })
            # Also compress history so the output-cap retry doesn't spin on
            # max_tokens alone; dropping the middle window makes the total
            # fit. (#55546)
            try:
                original_len = len(messages)
                original_tokens = estimate_messages_tokens_rough(messages)
                _overflow_input = messages
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message,
                    approx_tokens=request_input_estimate,
                    task_id=effective_task_id,
                    bypass_cooldown=True,  # #100661 provider-proven overflow
                )
                if messages is _overflow_input and compression_skipped_due_to_lock(agent):
                    compression_attempts -= 1
                    agent._persist_session(messages, conversation_history)
                    return _verdict("return", _compression_deferred_result(
                        agent, messages, api_call_count
                    ))
                if messages is _overflow_input and compression_blocked_transiently(agent):
                    # #97488: timed transient guard — defer, never
                    # exhaustion (gateway auto-reset).
                    compression_attempts -= 1
                    agent._persist_session(messages, conversation_history)
                    return _verdict("return", _compression_deferred_result(
                        agent, messages, api_call_count,
                        reason="transient_block",
                    ))
                conversation_history = conversation_history_after_compression(
                    agent, messages, conversation_history
                )
                new_tokens = estimate_messages_tokens_rough(messages)
                if len(messages) < original_len:
                    agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
                elif new_tokens > 0 and new_tokens < original_tokens * 0.95:
                    agent._buffer_status(COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=original_tokens, after=new_tokens))
            except Exception:
                # Compression must never turn an output-cap error
                # fatal — fall through and retry on max_tokens alone.
                logger.warning(
                    "%sOutput-cap compression hit an error; retrying on max_tokens only.",
                    agent.log_prefix,
                )
            _retry.restart_with_compressed_messages = True
            return _verdict("break")

        # Output-cap error with unparseable budget: compression can't help
        # (input already fits) and would death-loop on the same 400. Fail
        # fast. (#55546)
        if is_output_cap_error(error_msg):
            agent._flush_status_buffer()
            agent._vprint(
                f"{agent.log_prefix}❌ The provider rejected the request because "
                f"max_tokens exceeds its output cap for this model.",
                force=True,
            )
            agent._vprint(
                f"{agent.log_prefix}   💡 Lower model.max_tokens in your config.yaml to "
                f"at or below the model's max-output limit. "
                f"(This is an output-cap error, not a context overflow — "
                f"compression cannot fix it.)",
                force=True,
            )
            logger.error(
                f"{agent.log_prefix}Output-cap error not routed into compression "
                f"(max_tokens over provider cap): {error_msg[:200]}"
            )
            agent._persist_session(messages, conversation_history)
            _final_response = (
                "max_tokens exceeds the provider's output cap for this model. "
                "Lower model.max_tokens in config.yaml."
            )
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "completed": False,
                "api_calls": api_call_count,
                "error": _final_response,
                "partial": True,
                "failed": True,
            })

        # Input too large: shrink context_length only when the provider
        # reports the real limit; else keep the window and compress. Guessed
        # probe tiers can turn a configured 1M window into 256K/128K/64K.
        new_ctx = get_context_length_from_provider_error(error_msg, old_ctx)
        _provider_lower = (getattr(agent, "provider", "") or "").lower()
        _base_lower = (getattr(agent, "base_url", "") or "").rstrip("/").lower()
        is_minimax_provider = (
            _provider_lower in {"minimax", "minimax-cn"}
            or _base_lower.startswith((
                "https://api.minimax.io/anthropic",
                "https://api.minimaxi.com/anthropic",
            ))
        )
        minimax_delta_only_overflow = (
            is_minimax_provider
            and new_ctx is None
            and "context window exceeds limit (" in error_msg
        )

        if new_ctx is not None:
            agent._buffer_vprint(f"Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})")
            compressor.update_model(
                model=agent.model,
                context_length=new_ctx,
                base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),
                provider=agent.provider,
                api_mode=agent.api_mode,
            )
            # Persist the provider-reported limit before compression/retry:
            # rate limit, missing usage, or restart must not lose confirmed
            # metadata. Probe flags remain a fallback if this write fails.
            save_context_length(agent.model, agent.base_url, new_ctx)
            # Probe flags only on the built-in compressor (plugin engines
            # manage their own); provider-sourced value, so safe to cache.
            if hasattr(compressor, "_context_probed"):
                compressor._context_probed = True
                compressor._context_probe_persistable = True
            agent._buffer_vprint(f"⚠️  Context length exceeded — using provider limit: {old_ctx:,} → {new_ctx:,} tokens")
        elif minimax_delta_only_overflow:
            agent._buffer_vprint(
                f"Provider reported overflow amount only; "
                f"keeping context_length at {old_ctx:,} tokens and compressing."
            )
        else:
            agent._buffer_vprint(
                f"⚠️  Context length exceeded, but provider did not report a max context length; "
                f"keeping context_length at {old_ctx:,} tokens and compressing."
            )

        compression_attempts += 1
        if compression_attempts > max_compression_attempts:
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
            logger.error("%sContext compression failed after %d attempts.", agent.log_prefix, max_compression_attempts)
            agent._persist_session(messages, conversation_history)
            _final_response = f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached."
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "completed": False,
                "api_calls": api_call_count,
                "error": _final_response,
                "partial": True,
                "failed": True,
                "compression_exhausted": True,
            })
        agent._buffer_status(COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(tokens=approx_tokens, attempt=compression_attempts, cap=max_compression_attempts))

        original_len = len(messages)
        original_tokens = estimate_messages_tokens_rough(messages)
        _overflow_input = messages
        # Pass the OVERHEAD-AWARE size (msgs + tool schemas + system) so LCM
        # forced-overflow recovery arms on the TRUE request; approx_tokens
        # stays for status. See hermes-lcm _should_force_overflow_recovery.
        messages, active_system_prompt = agent._compress_context(
            messages, system_message,
            approx_tokens=estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
            task_id=effective_task_id,
            # Provider proved the request doesn't fit: ignore the
            # summary-failure cooldown for this ONE attempt (bounded by
            # max_compression_attempts). (#100661)
            bypass_cooldown=True,
        )
        if messages is _overflow_input and compression_skipped_due_to_lock(agent):
            # Lock-skip: another path holds the compression lock, so this
            # pass no-oped. Temporary defer, not exhaustion — refund the
            # attempt, end the turn softly, no auto-reset. (#69870)
            compression_attempts -= 1
            agent._persist_session(messages, conversation_history)
            return _verdict("return", _compression_deferred_result(
                agent, messages, api_call_count
            ))
        if messages is _overflow_input and compression_blocked_transiently(agent):
            # Transient block: a timed guard (host-timeout cooldown /
            # structural backoff) no-oped this pass — defer softly, never
            # compression_exhausted (auto-reset). (#97488)
            compression_attempts -= 1
            agent._persist_session(messages, conversation_history)
            return _verdict("return", _compression_deferred_result(
                agent, messages, api_call_count,
                reason="transient_block",
            ))
        if context_compression_timed_out(agent):
            # Host timeout: recovery spent its wait budget with no committed
            # summary. Re-sending would hit the same overflow; end the turn
            # via the typed recovery contract. (#98722)
            agent._persist_session(messages, conversation_history)
            _final_response = _COMPRESSION_TIMEOUT_FINAL_RESPONSE
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "completed": False,
                "api_calls": api_call_count,
                "error": _final_response,
                "partial": True,
                "failed": True,
                "compression_exhausted": True,
                "turn_exit_reason": "context_compression_timeout",
            })
        conversation_history = conversation_history_after_compression(
            agent, messages, conversation_history
        )

        # Re-estimate after compression: same-message-count compression
        # (tool-result pruning, in-place summarization) can shrink the
        # request. (#39550)
        new_tokens = estimate_messages_tokens_rough(messages)
        approx_tokens = new_tokens  # update for downstream logging

        if len(messages) < original_len or (new_tokens > 0 and new_tokens < original_tokens * 0.95) or (new_ctx and new_ctx < old_ctx):
            if len(messages) < original_len:
                agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
            elif new_tokens > 0 and new_tokens < original_tokens * 0.95:
                agent._buffer_status(COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=original_tokens, after=new_tokens))
            time.sleep(2)  # Brief pause between compression retries
            # Rebuild the full request and force normal preflight to honor
            # it; message count alone doesn't prove system/tool-inclusive
            # pressure fell.
            _provider_overflow_recovery_pending = True
            _retry.restart_with_compressed_messages = True
            return _verdict("break")
        else:
            # Can't compress further and already at minimum tier
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
            agent._vprint(f"{agent.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
            logger.error("%sContext length exceeded: %s tokens. Cannot compress further.", agent.log_prefix, f"{new_tokens:,}")
            agent._persist_session(messages, conversation_history)
            _final_response = f"Context length exceeded ({new_tokens:,} tokens). Cannot compress further."
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "completed": False,
                "api_calls": api_call_count,
                "error": _final_response,
                "partial": True,
                "failed": True,
                "compression_exhausted": True,
            })
    return _verdict("fallthrough")
