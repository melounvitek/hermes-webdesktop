"""Pre-API preflight compression gate for the conversation turn loop.

Extracted from ``run_conversation``. Runs once per API call after the request pressure
is measured: grow a managed llama.cpp window (last resort), or compress when over
threshold (deferring on noisy estimates, in failure cooldown, or when the review fork's
first request is pending), handle the provider-proven overflow re-check fail-closed,
and emit the deduped blocked/uncompressed overflow warnings. Nothing here imports
``agent.conversation_loop`` at module level (cycle); loop-internal helpers resolve lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.context_engine import automatic_compaction_status_message
from agent.conversation_compression import (
    PRE_API_COMPRESSION_STATUS_TEMPLATE,
    compression_blocked_transiently,
    compression_skipped_due_to_lock,
    context_compression_timed_out,
    conversation_history_after_compression,
)
from agent.turn_context import _review_fork_first_request_pending

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class PreflightVerdict:
    """Outcome of ``run_preflight_compression``.

    ``action``: ``"proceed"`` (make the API call), ``"continue"`` (window grown or
    history compacted — the call/budget was refunded, re-enter the turn loop and
    re-measure), ``"break"`` (turn ends: compression timeout or non-actionable
    compaction handoff — ``final_response``/``failed``/``turn_exit_reason`` set) or
    ``"return"`` (typed deferred/exhausted result in ``result``). The remaining fields
    are the loop locals the gate may have rebound."""

    action: str
    result: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    active_system_prompt: Any
    conversation_history: Any
    api_call_count: int
    compression_attempts: int
    pending_moa_prepared_request: Any
    last_preflight_pressure: Optional[int]
    final_response: Any
    failed: bool
    compression_timeout_exhausted: bool
    turn_exit_reason: Any


def run_preflight_compression(
    agent: Any,
    *,
    compressor: Any,
    request_pressure_tokens: int,
    provider_overflow_preflight: bool,
    preflight_compression_blocked: bool,
    defer_preflight: Any,
    moa_prepared_request: Any,
    pending_moa_prepared_request: Any,
    messages: List[Dict[str, Any]],
    system_message: Any,
    user_message: Any,
    active_system_prompt: Any,
    conversation_history: Any,
    api_call_count: int,
    compression_attempts: int,
    max_compression_attempts: int,
    effective_task_id: Any,
    final_response: Any,
    failed: bool,
    compression_timeout_exhausted: bool,
    turn_exit_reason: Any,
) -> PreflightVerdict:
    """Mirror of the turn-prologue guard chain (defer on noisy estimate → skip in failure
    cooldown → ``should_compress``), #11529. A compression pass that never reaches the
    provider refunds the call/budget in every branch (skip, re-run, timeout) so
    ``api_call_count`` never over-reports; a lock/transient skip refunds the attempt and
    leaves the progress blocker unarmed (#69870, #97488). A forced provider-overflow
    preflight that any gate blocks fails closed (llama.cpp may silently truncate)."""
    from agent.conversation_loop import (
        _COMPRESSION_TIMEOUT_FINAL_RESPONSE,
        _HANDOFF_SKIP_FINAL_RESPONSE,
        _compression_deferred_result,
        _maybe_grow_local_window,
        _provider_overflow_exhausted_result,
        _should_skip_model_call_for_reference_handoff,
    )

    _compressor = compressor
    _provider_overflow_preflight = provider_overflow_preflight
    _preflight_compression_blocked = preflight_compression_blocked
    _defer_preflight = defer_preflight
    _moa_prepared_request = moa_prepared_request
    _last_preflight_pressure = None
    _compression_timeout_exhausted = compression_timeout_exhausted
    _turn_exit_reason = turn_exit_reason

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> PreflightVerdict:
        return PreflightVerdict(
            action=action,
            result=result,
            messages=messages,
            active_system_prompt=active_system_prompt,
            conversation_history=conversation_history,
            api_call_count=api_call_count,
            compression_attempts=compression_attempts,
            pending_moa_prepared_request=pending_moa_prepared_request,
            last_preflight_pressure=_last_preflight_pressure,
            final_response=final_response,
            failed=failed,
            compression_timeout_exhausted=_compression_timeout_exhausted,
            turn_exit_reason=_turn_exit_reason,
        )

    _compression_cooldown = getattr(
        _compressor, "get_active_compression_failure_cooldown", lambda: None
    )()
    if (
        agent.compression_enabled
        and not _review_fork_first_request_pending(agent)
        and len(messages) > 1
        and compression_attempts < max_compression_attempts
        and (
            not _preflight_compression_blocked
            or _provider_overflow_preflight
        )
        and (
            not _defer_preflight(request_pressure_tokens)
            or _provider_overflow_preflight
        )
        and not _compression_cooldown
        and _compressor.should_compress(request_pressure_tokens)
    ):
        # Managed local runtime: grow the context window before compressing (last
        # resort). Only for a llamacpp provider at the supervised base_url.
        _grown_window = _maybe_grow_local_window(
            agent, _compressor, request_pressure_tokens
        )
        if _grown_window:
            # Bigger window granted: recalibrate the compressor and skip compression
            # this pass.
            _compressor.update_model(
                agent.model,
                _grown_window,
                base_url=getattr(agent, "base_url", "") or "",
                api_key=getattr(agent, "api_key", "") or "",
                provider=getattr(agent, "provider", "") or "",
                api_mode=getattr(agent, "api_mode", "") or "",
            )
            agent._buffer_status(
                f"📈 Context window grown to {_grown_window // 1024}K "
                f"(local model; conversation continues uncompressed)"
            )
            # Never reached the provider — refund the call/budget like the
            # compression path does before its continue.
            api_call_count -= 1
            agent._api_call_count = api_call_count
            agent.iteration_budget.refund()
            return _verdict("continue")
        if _moa_prepared_request is not None:
            pending_moa_prepared_request = _moa_prepared_request
        compression_attempts += 1
        # Compression is running: reset the blocked-overflow warning dedup so a
        # later blocked turn warns again (#62625). getattr: test doubles lack it.
        _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
        if callable(_clear_warn):
            _clear_warn()
        logger.info(
            "Pre-API compression: ~%s request tokens >= %s threshold "
            "(context=%s, attempt=%s/%s)",
            f"{request_pressure_tokens:,}",
            f"{int(getattr(_compressor, 'threshold_tokens', 0) or 0):,}",
            f"{int(getattr(_compressor, 'context_length', 0) or 0):,}"
            if getattr(_compressor, "context_length", 0) else "unknown",
            compression_attempts,
            max_compression_attempts,
        )
        _pre_api_status = automatic_compaction_status_message(
            _compressor,
            phase="pre_api",
            default_message=PRE_API_COMPRESSION_STATUS_TEMPLATE.format(
                tokens=request_pressure_tokens
            ),
            approx_tokens=request_pressure_tokens,
            threshold_tokens=int(
                getattr(_compressor, "threshold_tokens", 0) or 0
            ),
            context_length=int(
                getattr(_compressor, "context_length", 0) or 0
            ),
            model=agent.model,
            attempt=compression_attempts,
            max_attempts=max_compression_attempts,
        )
        if _pre_api_status:
            agent._emit_status(_pre_api_status)
        _last_preflight_pressure = request_pressure_tokens
        _pre_api_input = messages
        messages, active_system_prompt = agent._compress_context(
            messages,
            system_message,
            approx_tokens=request_pressure_tokens,
            task_id=effective_task_id,
        )
        if context_compression_timed_out(agent):
            # Progress-aware timeout (#98722): never reached the provider — refund
            # the call/budget and stop; an overflow retry would only re-compress.
            api_call_count -= 1
            agent._api_call_count = api_call_count
            agent.iteration_budget.refund()
            final_response = _COMPRESSION_TIMEOUT_FINAL_RESPONSE
            failed = True
            _compression_timeout_exhausted = True
            _turn_exit_reason = "context_compression_timeout"
            return _verdict("break")
        if messages is _pre_api_input and (
            compression_skipped_due_to_lock(agent)
            or compression_blocked_transiently(agent)
        ):
            # Temporary DEFER (lock held / cooldown), not evidence about
            # compressibility: refund the attempt, leave the progress blocker
            # unarmed and proceed (#69870, #97488).
            compression_attempts -= 1
            _last_preflight_pressure = None
            if pending_moa_prepared_request is _moa_prepared_request:
                pending_moa_prepared_request = None
        else:
            # Reset retry/empty-response state so the compacted request gets a fresh
            # chance.
            agent._empty_content_retries = 0
            agent._thinking_prefill_retries = 0
            agent._last_content_with_tools = None
            agent._last_content_tools_all_housekeeping = False
            agent._mute_post_response = False
            # Re-baseline the flush cursor: rotation returns None (child flushes
            # whole); in-place returns list(messages) — None would re-append
            # persisted rows. See conversation_history_after_compression().
            conversation_history = conversation_history_after_compression(
                agent, messages, conversation_history
            )
            # Never reaches the provider on skip or re-run — refund the call/budget
            # in BOTH cases, else budget leaks and api_call_count over-reports.
            api_call_count -= 1
            agent._api_call_count = api_call_count
            agent.iteration_budget.refund()
            if _should_skip_model_call_for_reference_handoff(
                messages, user_message
            ):
                # Reference-only handoff must not become the active turn
                # after a completed assistant response (#80622).
                logger.info(
                    "Skipping post-compaction model call: reference-only "
                    "handoff would be the sole active user turn (#80622)"
                )
                if not final_response:
                    final_response = _HANDOFF_SKIP_FINAL_RESPONSE
                _turn_exit_reason = "compaction_handoff_not_actionable"
                return _verdict("break")
            return _verdict("continue")
    elif _provider_overflow_preflight and _compression_cooldown:
        # Provider proved the request cannot fit and the compressor is unavailable:
        # don't resend; let the next user turn retry after cooldown.
        agent._persist_session(messages, conversation_history)
        return _verdict("return", _compression_deferred_result(
            agent,
            messages,
            api_call_count,
            reason="transient_block",
        ))
    elif (
        _provider_overflow_preflight
        and compression_attempts >= max_compression_attempts
    ):
        # All recovery passes consumed and still over threshold: fail closed —
        # llama.cpp may silently truncate an oversized retry.
        return _verdict("return", _provider_overflow_exhausted_result(
            agent,
            messages,
            conversation_history,
            api_call_count,
            request_pressure_tokens,
            max_compression_attempts,
        ))
    elif (
        agent.compression_enabled
        and len(messages) > 1
        and compression_attempts < max_compression_attempts
        and not _defer_preflight(request_pressure_tokens)
        and _compression_cooldown
    ):
        # Summary-LLM cooldown blocks compression: deduped warning only when over
        # threshold (should_compress_info reason is None below it) (#62625).
        _block_reason = None
        try:
            _block_reason = _compressor.should_compress_info(
                request_pressure_tokens
            )[1]
        except Exception:
            _block_reason = None
        if _block_reason:
            agent._warn_context_overflow_blocked(
                _block_reason,
                request_pressure_tokens,
                int(getattr(_compressor, "threshold_tokens", 0) or 0),
            )
    elif not agent.compression_enabled and len(messages) > 1:
        # Uncompressed session guard (#89297): compression is disabled, so warn
        # (deduped) when the request exceeds the context window; the turn-context
        # preflight re-arms the dedup.
        _ctx_len = getattr(
            getattr(agent, "context_compressor", None), "context_length", None
        )
        if (
            isinstance(_ctx_len, int)
            and _ctx_len > 0
            and request_pressure_tokens > _ctx_len
        ):
            _warn_fn = getattr(
                agent, "_warn_uncompressed_context_overflow", None
            )
            if callable(_warn_fn):
                _warn_fn(request_pressure_tokens, _ctx_len)

    if _provider_overflow_preflight:
        # Any other gate blocking the forced preflight (e.g. uncompressible one-
        # message request) must fail closed: the request is proven not to fit.
        return _verdict("return", _provider_overflow_exhausted_result(
            agent,
            messages,
            conversation_history,
            api_call_count,
            request_pressure_tokens,
            max_compression_attempts,
        ))
    return _verdict("proceed")


@dataclass
class PostToolCompressionVerdict:
    """``end_turn`` True → a reference-only compaction handoff would be the sole active
    user turn (#80622): stop without another model call (``final_response`` /
    ``turn_exit_reason`` set)."""

    end_turn: bool
    messages: List[Dict[str, Any]]
    active_system_prompt: Any
    conversation_history: Any
    compression_attempts: int
    final_response: Any
    turn_exit_reason: Any


def compress_after_tool_results(
    agent: Any,
    *,
    messages: List[Dict[str, Any]],
    system_message: Any,
    user_message: Any,
    active_system_prompt: Any,
    conversation_history: Any,
    compression_attempts: int,
    max_compression_attempts: int,
    effective_task_id: Any,
    final_response: Any,
    turn_exit_reason: Any,
) -> PostToolCompressionVerdict:
    """Post-tool-call compression decision. Pressure comes from API-reported
    ``prompt_tokens`` (a tight lower bound; thinking models inflate completion tokens,
    #12026), ``0`` right after compression (no real count yet), else the route-aware
    overhead-inclusive estimate (#14695). Over threshold but blocked → deduped warning
    (#62625) plus the deterministic tool-result-only prune, committed only when the
    engine returns a NEW list (never rebuild ``conversation_history`` for it)."""
    from agent.conversation_loop import (
        _HANDOFF_SKIP_FINAL_RESPONSE,
        _midturn_request_pressure_tokens,
        _should_skip_model_call_for_reference_handoff,
        estimate_request_tokens_rough,
    )

    _turn_exit_reason = turn_exit_reason

    def _verdict(end_turn: bool) -> PostToolCompressionVerdict:
        return PostToolCompressionVerdict(
            end_turn=end_turn,
            messages=messages,
            active_system_prompt=active_system_prompt,
            conversation_history=conversation_history,
            compression_attempts=compression_attempts,
            final_response=final_response,
            turn_exit_reason=_turn_exit_reason,
        )

    # Decide compression from API-reported prompt tokens (tight lower bound;
    # tool results get counted on the next call). If last_prompt_tokens is 0
    # (disconnect / no usage data) fall back to a rough estimate. (#2153)
    _compressor = agent.context_compressor
    if _compressor.last_prompt_tokens > 0:
        # Only prompt_tokens: thinking models inflate completion_tokens with
        # reasoning that uses no context → premature compression. (#12026)
        _real_tokens = _compressor.last_prompt_tokens
    elif _compressor.last_prompt_tokens == -1:
        # Compression just ran, no API prompt count yet: don't treat a rough
        # schema-heavy post-compression estimate as real context pressure.
        _real_tokens = 0
    else:
        # Include tool schemas (20-30K tokens the messages-only estimate
        # misses) and stay route-aware: on a compacted native-Codex session
        # the generic durable-history figure would false-trigger. (#14695)
        _real_tokens = _midturn_request_pressure_tokens(
            agent,
            messages,
            active_system_prompt or "",
            estimate_request_tokens_rough(
                messages, tools=agent.tools or None
            ),
        )

    if (
        agent.compression_enabled
        and compression_attempts < max_compression_attempts
        and _compressor.should_compress(_real_tokens)
    ):
        compression_attempts += 1
        # Compression is running: reset blocked-overflow warning dedup so a
        # future blocked turn can warn again. getattr: test doubles lack it.
        _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
        if callable(_clear_warn):
            _clear_warn()
        agent._safe_print("  ⟳ compacting context…")
        _post_tool_input = messages
        # Pass overhead-aware _real_tokens, not last_prompt_tokens (0 in
        # the no-usage fallback), so the overflow guard sees the true size.
        messages, active_system_prompt = agent._compress_context(
            messages, system_message,
            approx_tokens=_real_tokens,
            task_id=effective_task_id,
        )
        if (
            messages is _post_tool_input
            and compression_skipped_due_to_lock(agent)
        ):
            # Lock-skip no-op is a temporary defer, not evidence about
            # compressibility: refund so a lock-loser loop doesn't burn the
            # budget toward compression_exhausted. (#69870)
            compression_attempts -= 1
        else:
            conversation_history = conversation_history_after_compression(
                agent, messages, conversation_history
            )
            if _should_skip_model_call_for_reference_handoff(
                messages, user_message
            ):
                logger.info(
                    "Skipping post-tool compaction model call: "
                    "reference-only handoff would be the sole "
                    "active user turn (#80622)"
                )
                if not final_response:
                    final_response = _HANDOFF_SKIP_FINAL_RESPONSE
                _turn_exit_reason = "compaction_handoff_not_actionable"
                return _verdict(True)
    elif agent.compression_enabled:
        # Over threshold but compression blocked (cooldown/anti-thrash):
        # deduped warning so context can't silently overflow. (#62625)
        _block_reason = None
        _info = getattr(_compressor, "should_compress_info", None)
        if _info is not None:
            try:
                _block_reason = _info(_real_tokens)[1]
            except Exception:
                _block_reason = None
        if _block_reason:
            agent._warn_context_overflow_blocked(
                _block_reason,
                _real_tokens,
                int(getattr(_compressor, "threshold_tokens", 0) or 0),
            )
        # Proactive tool-result prune (deterministic, no LLM, keeps tail):
        # no-op unless proactive_prune_tokens is exceeded; commits only past
        # proactive_prune_min_reclaim_tokens so cache breaks stay episodic.
        _prune = getattr(_compressor, "prune_tool_results_only", None)
        if callable(_prune):
            try:
                _pruned_msgs, _pruned_n = _prune(
                    messages, current_tokens=_real_tokens
                )
            except Exception:
                logger.debug(
                    "proactive tool-result prune failed; skipping",
                    exc_info=True,
                )
                _pruned_msgs, _pruned_n = messages, 0
            # Standard no-op caller contract: only commit when the
            # engine returned a NEW list object with a non-zero count.
            if _pruned_n and _pruned_msgs is not messages:
                # Do NOT rebuild conversation_history: rows already carry
                # _DB_PERSISTED_MARKER, and on a stale in-place flag the
                # helper could seed unpersisted rows into history_ids.
                messages = _pruned_msgs
    return _verdict(False)
