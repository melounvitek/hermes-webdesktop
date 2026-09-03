"""Post-loop turn finalization for ``run_conversation``.

Budget summary, trajectory save, persist, diagnostics, response transforms, result
assembly, steer drain, memory/skill review. Synchronous, single return. ``logger`` is
imported lazily from ``agent.conversation_loop`` (no cycle, same logger name)."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional, Tuple

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_content import flatten_message_text
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.message_sanitization import _sanitize_surrogates

# Verification-continuation nudges (verify-on-stop / pre_verify) must be stripped from
# returned/live history to avoid role-alternation breaks; the assistant response is
# real content and is not flagged. (#65919)
_VERIFICATION_CONTINUATION_FLAGS = (
    "_verification_stop_synthetic", "_pre_verify_synthetic"
)

_SENTENCE_END = {".", "!", "?", "。", "！", "？", "`", ")"}


def _assistant_row_missing_visible_text(msg: dict) -> bool:
    """True when an assistant row has no visible text (blank final or tool-only)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return False
    return not flatten_message_text(msg.get("content")).strip()


def _is_pure_tool_call_tail(msg: dict) -> bool:
    """Assistant row with ``tool_calls`` but no visible text of its own."""
    if not isinstance(msg, dict) or not msg.get("tool_calls"):
        return False
    return _assistant_row_missing_visible_text(msg)


def _fill_assistant_tail_content(agent, tail: dict, final_response) -> None:
    """Write delivered text onto an already-persisted blank assistant row."""
    tail["content"] = final_response
    stamp_message_timestamp(tail)
    tail.pop(_DB_PERSISTED_MARKER, None)
    agent._db_flush_scan_prefix = None


def _record_kanban_budget_exhausted(
    kanban_task: str, api_call_count: int, max_iterations: int, logger: logging.Logger
) -> None:
    """Record a terminal ``timed_out`` outcome for a kanban worker out of budget.

    Routed via ``_record_task_failure`` (not ``kanban_block``) so it counts toward the
    consecutive-failure circuit breaker (#29747). Idempotent via the ``_end_run`` CAS
    (``WHERE ended_at IS NULL``): a no-op if another path already closed the run, so
    safe from multiple exit paths (#87096)."""
    try:
        from hermes_cli import kanban_db as _kb
        _conn = _kb.connect()
        try:
            _kb._record_task_failure(
                _conn,
                kanban_task,
                error=(
                    f"Iteration budget exhausted "
                    f"({api_call_count}/{max_iterations}) — "
                    "task could not complete within the allowed "
                    "iterations"
                ),
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                event_payload_extra={
                    "budget_used": api_call_count, "budget_max": max_iterations
                },
            )
        finally:
            try:
                _conn.close()
            except Exception:
                pass
    except Exception:
        logger.warning(
            "Failed to record budget-exhausted failure for task %s", kanban_task, exc_info=True
        )


def _drop_verification_continuation_scaffolding(messages) -> None:
    """Remove verification-continuation nudge messages from *messages* in place.

    Only the synthetic nudges carry these flags, so the real attempted-final-answer
    persisted to state.db survives."""
    messages[:] = [
        m for m in messages
        if not (isinstance(m, dict) and any(m.get(f) for f in _VERIFICATION_CONTINUATION_FLAGS))
    ]


def _clone_background_review_messages(messages):
    """Copy the review input without aliasing the live transcript."""
    # Import lazily: conversation_loop imports this module during turn
    # finalization, so a module-level import would create a cycle.
    from agent.conversation_loop import _clone_message_for_send

    return [_clone_message_for_send(message) for message in messages]


def _invoke_hook_safely(name: str, logger: logging.Logger, **kwargs) -> list:
    """Fire a lifecycle plugin hook; a failing hook is logged, never fatal."""
    try:
        from hermes_cli.lifecycle import invoke_hook
        return invoke_hook(name, **kwargs)
    except Exception as exc:
        logger.warning("%s hook failed: %s", name, exc)
        return []


def _guarded_cleanup(label: str, fn: Callable[[], Any], errors: List[str], logger) -> None:
    """Post-loop cleanup must never lose the response: each step is guarded
    independently and errors surface via ``cleanup_errors`` (#8049)."""
    try:
        fn()
    except Exception as err:
        errors.append(f"{label}: {err}")
        logger.error("finalize_turn: _%s failed: %s", label, err, exc_info=True)


def _resolve_budget_fallback(
    agent, *, final_response, api_call_count, interrupted, failed, messages, _turn_exit_reason,
    _pending_verification_response, _pending_verification_response_previewed, logger,
) -> Tuple[Any, Any, bool]:
    """Iteration-budget exhaustion. Returns ``(final_response, _turn_exit_reason,
    preserved_verification_fallback)``."""
    budget_exhausted = (
        api_call_count >= agent.max_iterations or agent.iteration_budget.remaining <= 0
    )
    budget_fallback_eligible = (
        budget_exhausted
        and not interrupted
        and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    )
    preserved_verification_fallback = False
    if final_response is None and budget_fallback_eligible:
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        if _pending_verification_response:
            # A verification gate withheld a composed answer, then the budget ran out:
            # preserve it rather than make another fallible call. The explicit pending
            # value is the provenance guard; unrelated error exits never enter here.
            final_response = _pending_verification_response
            # Previewed only if the reused candidate was actually streamed as interim.
            if _pending_verification_response_previewed:
                agent._response_was_previewed = True
            preserved_verification_fallback = True
        else:
            # _handle_max_iterations makes one extra toolless request for a summary.
            agent._emit_status(
                f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— asking model to summarise"
            )
            if not agent.quiet_mode:
                agent._safe_print(
                    f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                    "— requesting summary..."
                )
            final_response = agent._handle_max_iterations(messages, api_call_count)

    if budget_exhausted:
        # A kanban worker must record a terminal outcome whether or not a fallback
        # path was eligible, so the dispatcher learns the worker could not complete.
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            _record_kanban_budget_exhausted(
                _kanban_task, api_call_count, agent.max_iterations, logger,
            )
    return final_response, _turn_exit_reason, preserved_verification_fallback


def _rollback_interrupted_preflight_display(agent, interrupted) -> None:
    """Roll back the preflight-seeded display count only when an interrupt wins before
    any provider response; compaction state (incl. ``-1``) stays with the real-usage
    path. Type-pinned guards keep MagicMock/SimpleNamespace doubles inert."""
    _preflight_snapshot = getattr(agent, "_turn_preflight_display_snapshot", None)
    if (
        interrupted is True
        and isinstance(_preflight_snapshot, int)
        and not isinstance(_preflight_snapshot, bool)
        and getattr(agent, "_turn_received_provider_response", False) is not True
        and getattr(agent, "context_compressor", None) is not None
    ):
        _rollback_fn = getattr(
            agent.context_compressor, "rollback_interrupted_preflight_display_tokens", None
        )
        if callable(_rollback_fn):
            _rollback_fn(_preflight_snapshot)


def _close_transcript_tail(agent, messages, final_response, interrupted, failed):
    """Shape the transcript tail before the durable snapshot; returns the (possibly
    stream-recovered) ``final_response``."""
    # Strip private retry scaffolding first, or a later "continue" replays
    # assistant("(empty)") / recovery nudges into the same empty-response loop. Only
    # the synthetic verification nudges go; the assistant candidate persists (#65919).
    agent._drop_trailing_empty_response_scaffolding(messages)
    _drop_verification_continuation_scaffolding(messages)

    # An empty terminal completion is not authoritative when the stream already
    # delivered text; recover before persist so a blank tail isn't frozen (#95514).
    _recovered_from_stream = False
    if not interrupted and not failed:
        _streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
        _streamed = _streamed.strip() if isinstance(_streamed, str) else ""
        _final_visible = flatten_message_text(final_response).strip() if final_response else ""
        if not _final_visible and _streamed:
            final_response = _streamed
            _recovered_from_stream = True

    # An interrupt can leave a tool result as the tail (no scaffolding flag rewinds
    # it); close the sequence so strict providers don't see ``tool → user``. An
    # explicit placeholder is used since final_response is usually empty (#48879).
    if interrupted:
        from agent.message_sanitization import close_interrupted_tool_sequence
        close_interrupted_tool_sequence(messages, final_response)

    # Recovery ``break`` sites can return a final_response with no closing assistant
    # row; enforce "delivered final_response ⇒ assistant row" here. Compare content,
    # not role, so a matching verification candidate isn't dup'd.
    if final_response and not interrupted:
        try:
            _tail = messages[-1] if messages else None
        except Exception:
            _tail = None
        _tail_role = _tail.get("role") if isinstance(_tail, dict) else None
        if _tail_role != "assistant":
            # Append so the durable turn closes with the answer (#43849/#44100).
            append_message(messages, {"role": "assistant", "content": final_response})
        elif (
            isinstance(_tail, dict)
            and _tail.get("content") != final_response
            and (
                _is_pure_tool_call_tail(_tail)
                or (_recovered_from_stream and _assistant_row_missing_visible_text(_tail))
            )
        ):
            # Pure tool-call turn or stream-recovered blank (#95514): fill its content
            # rather than append a second row.
            _fill_assistant_tail_content(agent, _tail, final_response)

    # Request is complete, so replace API-local voice/model/skill guidance with the
    # clean user input before the durable snapshot; earlier flushes used the DB-only
    # override as their messages were still needed (#48677 / #63766).
    _apply_override = getattr(agent, "_apply_persist_user_message_override", None)
    if callable(_apply_override):
        _apply_override(messages)
    return final_response


def _micro_compact_after_turn(agent, messages, final_response, logger) -> None:
    """Post-turn micro-compaction: absorb the oldest uncompacted exchange into the
    rolling summary before persist, amortizing compression across turns."""
    try:
        _compressor = getattr(agent, "context_compressor", None)
        # Strict `is True` + isinstance gates: plugin context engines and MagicMock
        # compressors pass duck checks and would wipe the transcript.
        if (
            _compressor
            and getattr(_compressor, '_micro_compact_enabled', False) is True
            and callable(getattr(_compressor, '_micro_compact', None))
            and final_response
            # No checkpoint hook, so never run while compression.checkpoint_required
            # is armed.
            and getattr(agent, "compression_checkpoint_required", False) is not True
            # Persistence-isolated agents (background review fork) must not
            # micro-compact: it burns an aux-LLM call on a throwaway transcript and
            # could archive_and_compact the CANONICAL session rows.
            and not getattr(agent, "_persist_disabled", False)
        ):
            _before = len(messages)
            _compacted = _compressor._micro_compact(messages)
            # Defrag rewrites the newest MICRO marker in place and pops _db_persisted;
            # the compressor flags us to invalidate the flush-scan cursor, else the
            # rewritten row is identity-skipped (stale).
            if getattr(_compressor, "_flush_scan_cursor_invalidated", False):
                _compressor._flush_scan_cursor_invalidated = False
                agent._db_flush_scan_prefix = None
            if isinstance(_compacted, list) and _compacted:
                messages[:] = _compacted
            _after = len(messages)
            if _before != _after:
                logger.info("Micro-compaction: %d -> %d messages", _before, _after)
    except Exception as _mc_err:
        logger.info("Micro-compaction failed: %s", _mc_err)


def _log_turn_exit(agent, messages, final_response, api_call_count, _turn_exit_reason, interrupted, logger) -> None:
    """Always INFO so agent.log captures WHY every turn ended; WARNING when the last
    message is a tool result (the "just stops" scenario)."""
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to the assistant message with the tool call.
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        agent.iteration_budget.used if agent.iteration_budget else 0,
        agent.iteration_budget.max_total if agent.iteration_budget else 0,
        _turn_tool_count, _last_msg_role, len(final_response) if final_response else 0,
        agent.session_id or "none",
    )
    if _last_msg_role == "tool" and not interrupted:
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)


def _append_file_mutation_footer(agent, final_response, logger):
    """If ``write_file`` / ``patch`` calls failed and were never superseded by a
    successful write to the same path, append an advisory so over-claiming is
    surfaced."""
    try:
        _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
        if _failed and agent._file_mutation_verifier_enabled():
            footer = agent._format_file_mutation_failure_footer(_failed)
            if footer:
                final_response = final_response.rstrip() + "\n\n" + footer
    except Exception as _ver_err:
        logger.debug("file-mutation verifier footer failed: %s", _ver_err)
    return final_response


def _explain_abnormal_exit(agent, final_response, _turn_exit_reason, preserved_verification_fallback, logger):
    """Turn-completion explainer: on abnormal exits, surface one explanation from
    ``_turn_exit_reason``. Only acts when no usable reply exists (empty, "(empty)",
    or a short unpunctuated fragment); ``text_response(...)`` exits stay silent."""
    try:
        if not agent._turn_completion_explainer_enabled():
            return final_response
        _stripped = (final_response or "").strip()
        _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
        # A short fragment not from a text_response exit and lacking sentence-ending
        # punctuation is treated as a truncated partial (#34452).
        _is_partial_fragment = (
            not _is_empty_terminal
            and not preserved_verification_fallback
            and not str(_turn_exit_reason).startswith("text_response")
            and len(_stripped) <= 24
            and _stripped[-1:] not in _SENTENCE_END
        )
        if (
            _is_empty_terminal
            or _is_partial_fragment
            or str(_turn_exit_reason) == "partial_stream_recovery"
        ):
            _explanation = agent._format_turn_completion_explanation(
                _turn_exit_reason, getattr(agent, "_last_persistence_error_cause", None)
            )
            if _explanation:
                # Replace the bare sentinel; keep a partial fragment and append why.
                final_response = _explanation if _is_empty_terminal else _stripped + "\n\n" + _explanation
    except Exception as _exp_err:
        logger.debug("turn-completion explainer failed: %s", _exp_err)
    return final_response


def _last_turn_reasoning(messages) -> Optional[Any]:
    """Reasoning from the CURRENT turn only: stop at this turn's user message (#17055),
    but take the most recent non-empty reasoning since many providers emit it on the
    tool-call step and leave the final step with reasoning=None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return None  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            return msg["reasoning"]
    return None


def finalize_turn(
    agent, *, final_response, api_call_count, interrupted, failed, messages, conversation_history,
    effective_task_id, turn_id, user_message, original_user_message, _should_review_memory,
    _turn_exit_reason, _pending_verification_response=None,
    _pending_verification_response_previewed=False,
):
    """Run the post-loop finalization and return the turn ``result`` dict."""
    from agent.conversation_loop import logger

    final_response, _turn_exit_reason, preserved_verification_fallback = _resolve_budget_fallback(
        agent, final_response=final_response, api_call_count=api_call_count,
        interrupted=interrupted, failed=failed, messages=messages,
        _turn_exit_reason=_turn_exit_reason,
        _pending_verification_response=_pending_verification_response,
        _pending_verification_response_previewed=_pending_verification_response_previewed,
        logger=logger,
    )

    completed = (
        final_response is not None
        and not failed
        and (
            api_call_count < agent.max_iterations
            or str(_turn_exit_reason).startswith("text_response(")
        )
    )

    _rollback_interrupted_preflight_display(agent, interrupted)

    _cleanup_errors: List[str] = []
    # ``user_message`` may be a multimodal list of parts; the trajectory format wants
    # a plain string.
    _guarded_cleanup(
        "save_trajectory",
        lambda: agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed),
        _cleanup_errors, logger,
    )
    _guarded_cleanup(
        "cleanup_task_resources",
        lambda: agent._cleanup_task_resources(effective_task_id),
        _cleanup_errors, logger,
    )
    # Persist only after the transcript tail is shaped and scaffolding removed.
    try:
        final_response = _close_transcript_tail(agent, messages, final_response, interrupted, failed)
        if not interrupted and not failed:
            _micro_compact_after_turn(agent, messages, final_response, logger)
        agent._persist_session(messages, conversation_history)
    except Exception as _persist_err:
        _cleanup_errors.append(f"persist_session: {_persist_err}")
        logger.error("finalize_turn: _persist_session failed: %s", _persist_err, exc_info=True)

    # Keep the gateway's separate in-memory history snapshot current even on
    # cleanup error, so a later prompt isn't sent with a pre-turn snapshot.
    try:
        agent._session_messages = messages
    except Exception:
        pass

    _log_turn_exit(agent, messages, final_response, api_call_count, _turn_exit_reason, interrupted, logger)

    # Response transforms apply only to real, uninterrupted responses.
    if final_response and not interrupted:
        final_response = _append_file_mutation_footer(agent, final_response, logger)
    if not interrupted:
        final_response = _explain_abnormal_exit(
            agent, final_response, _turn_exit_reason, preserved_verification_fallback, logger,
        )

    _platform = getattr(agent, "platform", None) or ""
    _response_transformed = False
    _pre_transform_response = None
    if final_response and not interrupted:
        # Plugin hook: transform_llm_output — fired once per turn after the tool loop.
        # First hook to return a string wins; None/empty leaves the text unchanged.
        for _hook_result in _invoke_hook_safely(
            "transform_llm_output", logger,
            response_text=final_response,
            session_id=agent.session_id or "",
            model=agent.model,
            platform=_platform,
        ):
            if isinstance(_hook_result, str) and _hook_result:
                _pre_transform_response = final_response
                final_response = _hook_result
                _response_transformed = True
                break
        # Plugin hook: post_llm_call (e.g. sync conversation data to an external
        # memory system).
        _invoke_hook_safely(
            "post_llm_call", logger,
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            assistant_response=final_response,
            conversation_history=list(messages),
            model=agent.model,
            platform=_platform,
        )

    # Context engine observation hook (complements per-request select_context()):
    # notify the engine the turn finished with the finalized transcript. Fail-open.
    try:
        from agent.conversation_loop import _notify_context_engine_turn_complete
        # ``_last_turn_usage`` holds the last API response's canonical usage dict, or
        # ``None`` on turns that never reached a provider response — by contract.
        _notify_context_engine_turn_complete(
            agent, messages, usage=getattr(agent, "_last_turn_usage", None), logger=logger,
            turn_id=turn_id, task_id=effective_task_id, api_call_count=api_call_count,
            interrupted=interrupted, failed=failed, turn_exit_reason=_turn_exit_reason,
        )
    except Exception as exc:
        logger.warning("on_turn_complete notification failed: %s", exc)

    # Surrogate chokepoint: ``final_response`` may be RAW SDK content, and a lone UTF-16
    # surrogate crashes downstream consumers (stdout, Telegram ``utf16_len``, JSON).
    # Scrub once where model text leaves the loop (#80366).
    if isinstance(final_response, str):
        final_response = _sanitize_surrogates(final_response)

    result = {
        "final_response": final_response,
        "last_reasoning": _last_turn_reasoning(messages),
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "pre_transform_response": _pre_transform_response,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        # Requested service tier (from request_overrides.extra_body), for billing
        # audits by callers like `hermes -z --usage-file`.
        "service_tier": (
            (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
        ).get("service_tier"),
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # Persistence failures already set failed=True; also stamp `error` so the gateway
    # surfaces status="error" (and desktop can toast) instead of a quiet complete frame.
    if failed and str(_turn_exit_reason) == "session_persistence_failed":
        result["error"] = final_response or (
            "session storage could not be written — check the state database "
            "health (`hermes doctor`), then send your message again"
        )
        # Machine-readable cause for the gateway/desktop, exactly
        # 'session_persistence_failed:<locked|compression|turn_lease|corrupt|...>'.
        # Never clobber a failure_reason another path already stamped.
        if "failure_reason" not in result:
            _cause = getattr(agent, "_last_persistence_error_cause", None)
            result["failure_reason"] = "session_persistence_failed:" + (_cause or "unknown")
    # Surface post-loop cleanup failures so the caller can tell a clean turn from one
    # whose teardown raised; the response is returned either way (#8049).
    if _cleanup_errors:
        result["cleanup_errors"] = _cleanup_errors
    # A /steer landing after the final assistant turn has no tool batch to drain into;
    # hand it back so it becomes the next user turn instead of being lost.
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message
    agent.clear_interrupt()
    # Clear stream callback so it doesn't leak into future calls.
    agent._stream_callback = None

    # Skill trigger is checked NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message, final_response=final_response,
        interrupted=interrupted, messages=messages,
    )

    # Background memory/skill review runs AFTER delivery so it never competes with the
    # user's task. Suppressed by skip_background_review (e.g. cron): the fork costs
    # ~30K tokens / event with no human-in-the-loop benefit.
    if (
        final_response
        and not interrupted
        and not getattr(agent, "skip_background_review", False)
        and (_should_review_memory or _should_review_skills)
    ):
        try:
            # _spawn_background_review clones the snapshot structurally so the fork's
            # in-place sanitizers can't reach the live transcript.
            agent._spawn_background_review(
                messages_snapshot=list(messages), review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # Memory provider on_session_end()/shutdown_all() are NOT called here:
    # run_conversation() runs once per message; CLI/gateway own session-end cleanup.
    _invoke_hook_safely(
        "on_session_end", logger,
        session_id=agent.session_id,
        task_id=effective_task_id,
        turn_id=turn_id,
        completed=completed,
        failed=failed,
        interrupted=interrupted,
        turn_exit_reason=_turn_exit_reason,
        model=agent.model,
        platform=_platform,
    )

    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    return result
