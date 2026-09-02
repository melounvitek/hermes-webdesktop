"""Post-loop turn finalization for ``run_conversation``.

Lifted verbatim: budget summary, trajectory save, persist, diagnostics, response
transforms, result assembly, steer drain, memory/skill review. Synchronous, single
return. ``logger`` is imported lazily from ``agent.conversation_loop`` (no cycle,
same logger name)."""

from __future__ import annotations

import logging
import os

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_content import flatten_message_text
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.message_sanitization import _sanitize_surrogates


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


# Verification-continuation nudges (verify-on-stop / pre_verify) must be stripped from
# returned/live history to avoid role-alternation breaks; the assistant response is
# real content and is not flagged. (#65919)
_VERIFICATION_CONTINUATION_FLAGS = (
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
)


def _record_kanban_budget_exhausted(
    kanban_task: str,
    api_call_count: int,
    max_iterations: int,
    logger: logging.Logger,
) -> None:
    """Record a terminal ``timed_out`` outcome for a kanban worker out of budget.

    Idempotent via the ``_end_run`` CAS (``WHERE ended_at IS NULL``): a no-op if
    another path already closed the run, so safe from multiple exit paths (#87096)."""
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
                    "budget_used": api_call_count,
                    "budget_max": max_iterations,
                },
            )
        finally:
            try:
                _conn.close()
            except Exception:
                pass
    except Exception:
        logger.warning(
            "Failed to record budget-exhausted failure for task %s",
            kanban_task,
            exc_info=True,
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


def finalize_turn(
    agent,
    *,
    final_response,
    api_call_count,
    interrupted,
    failed,
    messages,
    conversation_history,
    effective_task_id,
    turn_id,
    user_message,
    original_user_message,
    _should_review_memory,
    _turn_exit_reason,
    _pending_verification_response=None,
    _pending_verification_response_previewed=False,
):
    """Run the post-loop finalization and return the turn ``result`` dict."""
    from agent.conversation_loop import logger

    budget_exhausted = (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    )
    budget_fallback_eligible = (
        budget_exhausted
        and not interrupted
        and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    )
    continuation_budget_exhausted = (
        final_response is None
        and bool(_pending_verification_response)
        and budget_fallback_eligible
    )

    iteration_limit_fallback = False
    preserved_verification_fallback = False
    if continuation_budget_exhausted:
        # A verification gate withheld a composed answer, then the budget ran out:
        # preserve it rather than make another fallible call. The explicit pending
        # value is the provenance guard; unrelated error exits never enter here.
        final_response = _pending_verification_response
        # Previewed only if the reused candidate was actually streamed as interim.
        if _pending_verification_response_previewed:
            agent._response_was_previewed = True
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        iteration_limit_fallback = True
        preserved_verification_fallback = True
    elif final_response is None and budget_fallback_eligible:
        # Budget exhausted: _handle_max_iterations makes one extra toolless request
        # for a summary.
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
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
        iteration_limit_fallback = True

    if iteration_limit_fallback:
        # Kanban worker: signal the dispatcher the worker could not complete. Route
        # via ``_record_task_failure(outcome="timed_out")`` (not ``kanban_block``) so
        # it counts toward the consecutive-failure circuit breaker (#29747).
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            _record_kanban_budget_exhausted(
                _kanban_task, api_call_count, agent.max_iterations, logger,
            )
    elif budget_exhausted:
        # Bounded fallback: budget exhausted with no eligible fallback path. A kanban
        # worker must still record a terminal outcome; the ``_end_run`` CAS makes it
        # idempotent if another path already closed the run (#87096).
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            _record_kanban_budget_exhausted(
                _kanban_task, api_call_count, agent.max_iterations, logger,
            )

    # Determine if conversation completed successfully
    normal_text_response = str(_turn_exit_reason).startswith("text_response(")
    completed = (
        final_response is not None
        and not failed
        and (
            api_call_count < agent.max_iterations
            or normal_text_response
        )
    )

    # Roll back the preflight-seeded display count only when an interrupt wins
    # before any provider response; compaction state (incl. ``-1``) stays with the
    # real-usage path. Type-pinned guards keep MagicMock/SimpleNamespace doubles inert.
    _preflight_snapshot = getattr(
        agent, "_turn_preflight_display_snapshot", None
    )
    if (
        interrupted is True
        and isinstance(_preflight_snapshot, int)
        and not isinstance(_preflight_snapshot, bool)
        and getattr(agent, "_turn_received_provider_response", False) is not True
        and getattr(agent, "context_compressor", None) is not None
    ):
        _rollback_fn = getattr(
            agent.context_compressor,
            "rollback_interrupted_preflight_display_tokens",
            None,
        )
        if callable(_rollback_fn):
            _rollback_fn(_preflight_snapshot)

    # Post-loop cleanup must never lose the response: trajectory save, teardown,
    # and session persist are guarded independently and errors surface via
    # ``cleanup_errors`` rather than killing the turn (#8049).
    _cleanup_errors = []

    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    try:
        agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)
    except Exception as _save_err:
        _cleanup_errors.append(f"save_trajectory: {_save_err}")
        logger.error("finalize_turn: _save_trajectory failed: %s", _save_err, exc_info=True)

    # Clean up VM and browser for this task after conversation completes
    try:
        agent._cleanup_task_resources(effective_task_id)
    except Exception as _cleanup_err:
        _cleanup_errors.append(f"cleanup_task_resources: {_cleanup_err}")
        logger.error("finalize_turn: _cleanup_task_resources failed: %s", _cleanup_err, exc_info=True)

    # Persist only after private retry scaffolding is removed, or a later "continue"
    # replays assistant("(empty)") / recovery nudges into the same empty-response loop.
    try:
        agent._drop_trailing_empty_response_scaffolding(messages)

        # Strip only the synthetic verification nudges before the tail-assistant
        # check; the assistant candidate persists in state.db. (#65919)
        _drop_verification_continuation_scaffolding(messages)

        # An empty terminal completion is not authoritative when the stream already
        # delivered text; recover before persist so a blank tail isn't frozen (#95514).
        _recovered_from_stream = False
        if not interrupted and not failed:
            _streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
            if isinstance(_streamed, str):
                _streamed = _streamed.strip()
            else:
                _streamed = ""
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

        # Recovery ``break`` sites can return a final_response with no closing
        # assistant row; enforce "delivered final_response ⇒ assistant row" here.
        # Compare content, not role, so a matching verification candidate isn't dup'd.
        if final_response and not interrupted:
            try:
                _tail = messages[-1] if messages else None
            except Exception:
                _tail = None
            _tail_role = _tail.get("role") if isinstance(_tail, dict) else None
            if _tail_role != "assistant":
                # Tail is not an assistant row — append the final response
                # so the durable turn closes with the answer (#43849/#44100).
                append_message(
                    messages,
                    {"role": "assistant", "content": final_response},
                )
            elif (
                isinstance(_tail, dict)
                and _tail.get("content") != final_response
                and (
                    _is_pure_tool_call_tail(_tail)
                    or (
                        _recovered_from_stream
                        and _assistant_row_missing_visible_text(_tail)
                    )
                )
            ):
                # Tail is an assistant row (pure tool-call turn or stream-recovered
                # blank, #95514): fill its content rather than append a second row.
                _fill_assistant_tail_content(agent, _tail, final_response)

        # Request is complete, so replace API-local voice/model/skill guidance with
        # the clean user input before the durable snapshot; earlier flushes used the
        # DB-only override as their messages were still needed (#48677 / #63766).
        _apply_override = getattr(agent, "_apply_persist_user_message_override", None)
        if callable(_apply_override):
            _apply_override(messages)

        # ── Post-turn micro-compaction ────────────────────────────
        # Absorb the oldest uncompacted exchange into the rolling summary before
        # persist, amortizing compression across turns instead of one big pause.
        if not interrupted and not failed:
            try:
                _compressor = getattr(agent, "context_compressor", None)
                # Strict `is True` + isinstance gates: plugin context engines and
                # MagicMock compressors pass duck checks and would wipe the transcript.
                if (
                    _compressor
                    and getattr(_compressor, '_micro_compact_enabled', False) is True
                    and callable(getattr(_compressor, '_micro_compact', None))
                    and final_response
                    # Micro-compaction has no checkpoint hook, so it must never run
                    # while compression.checkpoint_required is armed.
                    and getattr(
                        agent, "compression_checkpoint_required", False
                    ) is not True
                    # Persistence-isolated agents (background review fork) must not
                    # micro-compact: it burns an aux-LLM call on a throwaway transcript
                    # and could archive_and_compact the CANONICAL session rows.
                    and not getattr(agent, "_persist_disabled", False)
                ):
                    _before = len(messages)
                    _compacted = _compressor._micro_compact(messages)
                    # Defrag rewrites the newest MICRO marker in place and pops
                    # _db_persisted; the compressor flags us to invalidate the flush-
                    # scan cursor, else the rewritten row is identity-skipped (stale).
                    if getattr(
                        _compressor, "_flush_scan_cursor_invalidated", False
                    ):
                        _compressor._flush_scan_cursor_invalidated = False
                        agent._db_flush_scan_prefix = None
                    if isinstance(_compacted, list) and _compacted:
                        messages[:] = _compacted
                    _after = len(messages)
                    if _before != _after:
                        logger.info(
                            "Micro-compaction: %d -> %d messages",
                            _before, _after,
                        )
            except Exception as _mc_err:
                logger.info("Micro-compaction failed: %s", _mc_err)

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

    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always INFO so agent.log captures WHY every turn ended; WARNING when the last
    # message is a tool result (the "just stops" scenario).
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
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
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # File-mutation verifier footer: if ``write_file`` / ``patch`` calls failed and
    # were never superseded by a successful write to the same path, append an
    # advisory so over-claiming is surfaced. Only on real, uninterrupted responses.
    if final_response and not interrupted:
        try:
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    # Turn-completion explainer: on abnormal exits, surface one explanation from
    # ``_turn_exit_reason``. Only acts when no usable reply exists (empty, "(empty)",
    # or a short unpunctuated fragment); ``text_response(...)`` exits stay silent.
    if not interrupted:
        try:
            if agent._turn_completion_explainer_enabled():
                _stripped = (final_response or "").strip()
                _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
                # A short fragment not from a text_response exit and lacking sentence-
                # ending punctuation is treated as a truncated partial (#34452).
                _is_partial_fragment = (
                    not _is_empty_terminal
                    and not preserved_verification_fallback
                    and not str(_turn_exit_reason).startswith("text_response")
                    and len(_stripped) <= 24
                    and _stripped[-1:] not in {".", "!", "?", "。", "！", "？", "`", ")"}
                )
                _is_partial_stream_recovery = (
                    str(_turn_exit_reason) == "partial_stream_recovery"
                )
                if (
                    _is_empty_terminal
                    or _is_partial_fragment
                    or _is_partial_stream_recovery
                ):
                    _explanation = agent._format_turn_completion_explanation(
                        _turn_exit_reason,
                        getattr(agent, "_last_persistence_error_cause", None),
                    )
                    if _explanation:
                        if _is_empty_terminal:
                            # Replace the bare "(empty)"/blank sentinel with
                            # the actionable explanation.
                            final_response = _explanation
                        else:
                            # Keep the partial fragment and append why it stopped.
                            final_response = (
                                _stripped + "\n\n" + _explanation
                            )
        except Exception as _exp_err:
            logger.debug("turn-completion explainer failed: %s", _exp_err)

    _response_transformed = False
    _pre_transform_response = None

    # Plugin hook: transform_llm_output — fired once per turn after the tool loop.
    # First hook to return a string wins; None/empty leaves the text unchanged.
    if final_response and not interrupted:
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    _pre_transform_response = final_response
                    final_response = _hook_result
                    _response_transformed = True
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # Plugin hook: post_llm_call — fired once per turn after the tool loop (e.g. sync
    # conversation data to an external memory system).
    if final_response and not interrupted:
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                task_id=effective_task_id,
                turn_id=turn_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # Context engine observation hook (complements per-request select_context()):
    # notify the engine the turn finished with the finalized transcript. Fail-open.
    try:
        from agent.conversation_loop import _notify_context_engine_turn_complete
        # ``_last_turn_usage`` holds the last API response's canonical usage dict, or
        # ``None`` on turns that never reached a provider response — by contract.
        _turn_usage = getattr(agent, "_last_turn_usage", None)
        _notify_context_engine_turn_complete(
            agent,
            messages,
            usage=_turn_usage,
            logger=logger,
            turn_id=turn_id,
            task_id=effective_task_id,
            api_call_count=api_call_count,
            interrupted=interrupted,
            failed=failed,
            turn_exit_reason=_turn_exit_reason,
        )
    except Exception as exc:
        logger.warning("on_turn_complete notification failed: %s", exc)

    # Reasoning from the CURRENT turn only: stop at this turn's user message
    # (#17055), but take the most recent non-empty reasoning since many providers
    # emit it on the tool-call step and leave the final step with reasoning=None.
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # Surrogate chokepoint: ``final_response`` may be RAW SDK content, and a lone UTF-16
    # surrogate crashes downstream consumers (stdout, Telegram ``utf16_len``, JSON).
    # Scrub once where model text leaves the loop (#80366).
    if isinstance(final_response, str):
        final_response = _sanitize_surrogates(final_response)

    # Build result with interrupt info if applicable
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
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
        # Requested service tier (from request_overrides.extra_body), for
        # billing audits by callers like `hermes -z --usage-file`.
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
            result["failure_reason"] = (
                "session_persistence_failed:" + (_cause or "unknown")
            )
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

    # Include interrupt message if one triggered the interrupt
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # Clear interrupt state after handling
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    agent._stream_callback = None

    # Check skill trigger NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
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
            # _spawn_background_review clones the snapshot structurally so
            # the fork's in-place sanitizers can't reach the live transcript.
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # Memory provider on_session_end()/shutdown_all() are NOT called here:
    # run_conversation() runs once per message; CLI/gateway own session-end cleanup.

    # Plugin hook: on_session_end — fired at the end of every run_conversation call.
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            completed=completed,
            failed=failed,
            interrupted=interrupted,
            turn_exit_reason=_turn_exit_reason,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False

    return result
