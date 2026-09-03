"""Pre-API pressure gate for the conversation turn loop: the Ollama runtime-context floor,
the provider-overflow re-check arming, the insufficient-progress blocker (compares fully
assembled requests, not raw ``messages``) and the call into
``turn_preflight.run_preflight_compression``. Extracted from ``run_conversation``; nothing
here imports ``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from agent.message_metadata import append_message
from agent.turn_context import _compression_warrants_another_preflight_pass
from agent.turn_preflight import run_preflight_compression

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class PreflightGateVerdict:
    """``action``: ``"fallthrough"`` (make the API call), ``"continue"`` (compaction/window
    growth — re-measure), ``"break"`` (turn ends) or ``"return"`` (``result`` is the turn's
    result dict). The other fields are the loop locals the gate rebinds."""

    action: str
    pending_moa_prepared_request: Any
    messages: Any
    active_system_prompt: Any
    conversation_history: Any
    api_call_count: Any
    compression_attempts: Any
    final_response: Any
    failed: Any
    _turn_exit_reason: Any
    _compression_timeout_exhausted: Any
    _preflight_compression_blocked: Any
    _provider_overflow_recovery_pending: Any
    _last_preflight_pressure: Any
    result: Optional[Dict[str, Any]] = None


def run_preflight_gate(
    agent: Any, *, request_pressure_tokens: Any, _moa_prepared_request: Any,
    pending_moa_prepared_request: Any, messages: Any, system_message: Any, user_message: Any,
    active_system_prompt: Any, conversation_history: Any, api_call_count: Any,
    compression_attempts: Any, max_compression_attempts: Any, effective_task_id: Any,
    final_response: Any, failed: Any, _turn_exit_reason: Any, _compression_timeout_exhausted: Any,
    _preflight_compression_blocked: Any, _provider_overflow_recovery_pending: Any,
    _last_preflight_pressure: Any,
) -> PreflightGateVerdict:
    """Run the pre-API guard chain in the original order (#11529). ``_last_preflight_pressure``
    is consumed here (set to None) and re-armed only by a compression pass, so a blocked
    preflight never compares against a stale figure."""
    from agent.conversation_loop import (
        _ollama_context_limit_error,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> PreflightGateVerdict:
        return PreflightGateVerdict(
            action=action, pending_moa_prepared_request=pending_moa_prepared_request,
            messages=messages, active_system_prompt=active_system_prompt,
            conversation_history=conversation_history, api_call_count=api_call_count,
            compression_attempts=compression_attempts, final_response=final_response, failed=failed,
            _turn_exit_reason=_turn_exit_reason,
            _compression_timeout_exhausted=_compression_timeout_exhausted,
            _preflight_compression_blocked=_preflight_compression_blocked,
            _provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
            _last_preflight_pressure=_last_preflight_pressure, result=result,
        )

    _runtime_context_error = _ollama_context_limit_error(
        agent, request_pressure_tokens
    )
    if _runtime_context_error:
        final_response = _runtime_context_error
        failed = True
        _turn_exit_reason = "ollama_runtime_context_too_small"
        append_message(messages, {"role": "assistant", "content": final_response})
        agent._emit_status("❌ Ollama runtime context is too small for Hermes tool use")
        api_call_count -= 1
        agent._api_call_count = api_call_count
        try:
            agent.iteration_budget.refund()
        except Exception:
            pass
        return _verdict("break")

    # Pre-API pressure check: tool results grow a turn and last_prompt_tokens lags
    # them. Mirror the turn-prologue guard chain: defer on noisy estimate, skip in
    # failure cooldown, then should_compress() (#11529).
    _compressor = agent.context_compressor
    _preflight_threshold = int(
        getattr(_compressor, "threshold_tokens", 0) or 0
    )
    _provider_overflow_preflight = (
        _provider_overflow_recovery_pending
        and (
            _preflight_threshold <= 0 or request_pressure_tokens >= _preflight_threshold
        )
    )
    if (
        _provider_overflow_recovery_pending and not _provider_overflow_preflight
    ):
        # The outer-loop rebuild includes system prompt, request-only injections and
        # tool schemas; only that full request with output runway may be sent.
        _provider_overflow_recovery_pending = False
    # Compare fully assembled requests, not raw ``messages`` (which omit
    # api_content, plugin injections, prefills, MoA context, ephemeral system text).
    _previous_preflight_pressure = _last_preflight_pressure
    _last_preflight_pressure = None
    if (
        _previous_preflight_pressure is not None
        and request_pressure_tokens >= _preflight_threshold
        and not _compression_warrants_another_preflight_pass(
            _previous_preflight_pressure, request_pressure_tokens, _preflight_threshold
        )
    ):
        # Stop proactive retries this turn without consuming the shared overflow-
        # recovery budget; the provider's error handler may still compact.
        _preflight_compression_blocked = True
        logger.warning(
            "Pre-API compression made insufficient progress: ~%s -> "
            "~%s request tokens; skipping additional preflight passes",
            f"{_previous_preflight_pressure:,}",
            f"{request_pressure_tokens:,}",
        )
    _defer_preflight = getattr(
        _compressor, "should_defer_preflight_to_real_usage", lambda _t: False
    )
    _pf = run_preflight_compression(
        agent, compressor=_compressor, request_pressure_tokens=request_pressure_tokens,
        provider_overflow_preflight=_provider_overflow_preflight,
        preflight_compression_blocked=_preflight_compression_blocked,
        defer_preflight=_defer_preflight, moa_prepared_request=_moa_prepared_request,
        pending_moa_prepared_request=pending_moa_prepared_request, messages=messages,
        system_message=system_message, user_message=user_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        api_call_count=api_call_count, compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, effective_task_id=effective_task_id,
        final_response=final_response, failed=failed,
        compression_timeout_exhausted=_compression_timeout_exhausted,
        turn_exit_reason=_turn_exit_reason,
    )
    messages = _pf.messages
    active_system_prompt = _pf.active_system_prompt
    conversation_history = _pf.conversation_history
    api_call_count = _pf.api_call_count
    compression_attempts = _pf.compression_attempts
    pending_moa_prepared_request = _pf.pending_moa_prepared_request
    final_response = _pf.final_response
    failed = _pf.failed
    _compression_timeout_exhausted = _pf.compression_timeout_exhausted
    _turn_exit_reason = _pf.turn_exit_reason
    if _pf.last_preflight_pressure is not None:
        _last_preflight_pressure = _pf.last_preflight_pressure
    if _pf.action == "return":
        return _verdict("return", _pf.result)
    if _pf.action == "break":
        return _verdict("break")
    if _pf.action == "continue":
        return _verdict("continue")
    return _verdict("fallthrough")
