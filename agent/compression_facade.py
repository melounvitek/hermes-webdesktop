"""Host-side ``AIAgent._compress_context`` wrapper.

Publishes the commit fence ``hard_interrupt()`` reads, runs the compressor on a snapshot under the progress
timeout, mirrors ``_DB_PERSISTED_MARKER`` stamps back onto the live lists and rebinds the session context.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import copy
import threading

from agent.session_activity import ActivityProvenance

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class CompressionFacadeMixin:
    """``_compress_context`` (see module docstring)."""

    def _compress_context(
        self,
        messages: list,
        system_message: str,
        *,
        approx_tokens: int = None,
        task_id: str = "default",
        focus_topic: str = None,
        force: bool = False,
        bypass_cooldown: bool = False,
        defer_context_engine_notification: bool = False,
        commit_fence=None,
    ) -> tuple:
        """Forwarder — see ``agent.conversation_compression.compress_context``.

        ``force=True`` (manual /compress) bypasses the summary-failure cooldown; ``bypass_cooldown=True``
        (provider-proven overflow recovery) runs one real attempt while the cooldown stays armed.
        """
        # Per-attempt timeout signal for turn-start preflight and in-loop consumers (#98424): a stalled
        # compression must not be mistaken for a structural no-op. Thread-local + per-agent lock (#98741).
        from agent.conversation_compression import (
            CompressionCommitFence,
            compress_context,
            mark_context_compression_timed_out,
            reset_context_compression_timeout_outcome,
            resolve_context_compression_timeouts,
            run_compress_context_with_progress_timeout,
        )
        reset_context_compression_timeout_outcome(self)
        from agent.portal_tags import (
            get_affinity_scope,
            get_conversation_context,
            reset_affinity_scope,
            reset_conversation_context,
            set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        # Out-of-turn compaction (/compact, gateway /compress, partial head compression) runs outside
        # run_conversation's ambient scope; publish the root as a fallback so the summarizer's call carries
        # the conversation tag. No-op for in-turn callers.
        token = None
        if get_conversation_context() is None:
            root = self._conversation_root_id()
            if root:
                token = set_conversation_context(root)
        # Same fallback for the ROUTING scope, only when the host declared one (pre-#96811 fallback
        # otherwise).
        affinity_token = None
        if get_affinity_scope() is None:
            declared = declared_conversation_scope_safe(self)
            if declared:
                affinity_token = set_affinity_scope(declared)
        # Every compression has a fence; hard_interrupt() uses this exact instance to serialize cancel
        # admission against begin_commit().
        active_fence = commit_fence or CompressionCommitFence()
        # Serialize fence publication so overlapping automatic/manual entrypoints cannot replace the
        # fence of the attempt currently committing.
        fence_registration_lock = vars(self).setdefault(
            "_compression_commit_fence_lock", threading.RLock()
        )
        with fence_registration_lock:
            missing_fence = object()
            previous_fence = vars(self).get(
                "_active_compression_commit_fence", missing_fence
            )
            self._active_compression_commit_fence = active_fence
        try:
            def _run(fence=None, target_messages=None):
                return compress_context(
                    self,
                    target_messages if target_messages is not None else messages,
                    system_message,
                    approx_tokens=approx_tokens, task_id=task_id,
                    focus_topic=focus_topic,
                    force=force,
                    bypass_cooldown=bypass_cooldown,
                    defer_context_engine_notification=(
                        defer_context_engine_notification
                    ),
                    commit_fence=fence,
                )

            # Callers that already own a progress-aware wait (gateway session
            # hygiene) pass commit_fence and must not be double-wrapped.
            direct_path = commit_fence is not None
            idle_timeout = total_ceiling = None
            if not direct_path:
                idle_timeout, total_ceiling = resolve_context_compression_timeouts()
                if idle_timeout <= 0:
                    direct_path = True

            if direct_path:
                result = _run(active_fence)
            else:
                def _snapshot_worker(fence=None):
                    # #76354 F3: the pooled worker must NEVER share the caller's live transcript — a late
                    # engine after a host timeout could rewrite it. Deep-snapshot on the worker; results
                    # publish only via an ADMITTED commit.
                    snapshot = copy.deepcopy(messages)
                    result_msgs, result_prompt = _run(
                        fence, target_messages=snapshot
                    )
                    if result_msgs is snapshot:
                        # No-op/abort returned the snapshot unchanged: hand back the ORIGINAL list so
                        # identity-based semantics keep working.
                        return messages, result_prompt
                    return result_msgs, result_prompt

                # Resolve the fallback prompt lazily: an eager rebuild would raise before compress_context
                # runs when _cached_system_prompt is unset and _build_system_prompt fails.
                def _fallback_prompt():
                    cached = getattr(self, "_cached_system_prompt", None)
                    if cached:
                        return cached
                    try:
                        return self._build_system_prompt(system_message)
                    except Exception:
                        logger.debug(
                            "compress_context timeout fallback prompt rebuild "
                            "failed; using raw system_message",
                            exc_info=True,
                        )
                        return system_message or ""

                timeout_cause = {
                    "total_exhausted": False,
                    "progress_observed": False,
                }

                def _on_timeout_cause(total_exhausted, progress_observed):
                    timeout_cause["total_exhausted"] = total_exhausted
                    timeout_cause["progress_observed"] = progress_observed

                def _on_timeout(idle, waited, since_progress):
                    mark_context_compression_timed_out(self)
                    total_exhausted = timeout_cause["total_exhausted"]
                    progress_observed = timeout_cause["progress_observed"]
                    if total_exhausted:
                        logger.warning(
                            "Context compression reached its total ceiling "
                            "after %.1fs (progress observed=%s); continuing "
                            "without compression",
                            waited,
                            progress_observed,
                        )
                    else:
                        logger.warning(
                            "Context compression made no progress for %.1fs "
                            "(total wait %.1fs, ceiling %.1fs); continuing "
                            "without compression",
                            since_progress,
                            waited,
                            total_ceiling,
                        )
                    touch = getattr(self, "_touch_activity", None)
                    if callable(touch):
                        try:
                            touch(
                                "context compression timed out",
                                provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
                            )
                        except Exception:
                            logger.debug(
                                "compress_context timeout activity touch failed",
                                exc_info=True,
                            )
                    # Same timeout cooldown ladder as summary-LLM timeouts
                    # (#62452): avoid re-burning the full idle budget every turn.
                    compressor = getattr(self, "context_compressor", None)
                    if compressor is not None:
                        record = getattr(compressor, "record_timeout_failure", None)
                        if callable(record):
                            try:
                                reason = (
                                    "host compress_context total ceiling "
                                    "exhausted"
                                    if total_exhausted
                                    else "host compress_context timeout "
                                    "(no summary progress)"
                                )
                                record(
                                    reason,
                                    failure_kind=(
                                        "ceiling_exhausted"
                                        if total_exhausted
                                        else "stalled"
                                    ),
                                )
                            except Exception:
                                logger.debug(
                                    "failed to record compress_context timeout "
                                    "cooldown",
                                    exc_info=True,
                                )
                    emit = getattr(self, "_emit_warning", None)
                    if callable(emit):
                        if total_exhausted:
                            progress = (
                                " after summary output was observed"
                                if progress_observed
                                else ""
                            )
                            emit(
                                "⚠ Context compression reached its total ceiling "
                                f"after {waited:.1f}s{progress}. No messages were "
                                "dropped — continuing without compression. Run "
                                "/compress to retry or /new for a clean session."
                            )
                        else:
                            emit(
                                "⚠ Context compression timed out "
                                f"after {idle:.1f}s with no output from the summary "
                                "model. No messages were dropped — continuing "
                                "without compression. Run /compress to retry, /new "
                                "for a clean session, or check "
                                "auxiliary.compression."
                            )

                def _on_commit_overrun(waited, ceiling):
                    # Commit-phase ceiling breach: the SessionDB mutation must complete, so this only surfaces
                    # the overrun.
                    emit = getattr(self, "_emit_warning", None)
                    if callable(emit):
                        emit(
                            "⚠ Context compression commit is taking unusually "
                            f"long ({waited:.0f}s, ceiling {ceiling:.0f}s). "
                            "Waiting for it to finish safely — if this persists, "
                            "check SessionDB health (disk / lock contention)."
                        )

                def _publish_new_fence():
                    # The stall-fallback retry (#78981) needs a fence the aborted attempt cannot veto; publish
                    # it on the slot hard_interrupt() reads. The finally restores the caller's fence.
                    retry_fence = CompressionCommitFence()
                    with fence_registration_lock:
                        self._active_compression_commit_fence = retry_fence
                    return retry_fence

                result = run_compress_context_with_progress_timeout(
                    worker=_snapshot_worker,
                    messages=messages,
                    system_prompt_fallback=_fallback_prompt,
                    idle_timeout_seconds=idle_timeout,
                    total_ceiling_seconds=total_ceiling,
                    on_timeout=_on_timeout,
                    on_timeout_cause=_on_timeout_cause,
                    on_commit_overrun=_on_commit_overrun,
                    fence=active_fence,
                    telemetry_agent=self,
                    new_fence=_publish_new_fence,
                )
            # Imported UNCONDITIONALLY: a silent fallback literal would split the stamping key from the
            # flush's and resurrect the duplicate-row bug.
            from agent.context_compressor import _DB_PERSISTED_MARKER
            from agent.conversation_compression import (
                _messages_match_scoped_identity,

            )

            def _sync_persisted_markers(target_messages, source_messages):
                if not isinstance(target_messages, list) or not isinstance(
                    source_messages, list
                ):
                    return
                # Stamps land on the worker's snapshot first; mirror them onto the live lists by scoped
                # identity. Timestamp-less repeated content is ambiguous, so every scoped match is stamped.
                for source_message in source_messages:
                    if not (
                        isinstance(source_message, dict)
                        and source_message.get(_DB_PERSISTED_MARKER)
                    ):
                        continue
                    source_timestamp = source_message.get("timestamp")
                    matched_exact_timestamp = False
                    if source_timestamp is not None:
                        for target_message in target_messages:
                            if not isinstance(target_message, dict):
                                continue
                            if target_message.get(_DB_PERSISTED_MARKER):
                                continue
                            if not _messages_match_scoped_identity(
                                target_message, source_message
                            ):
                                continue
                            if target_message.get("timestamp") != source_timestamp:
                                continue
                            target_message[_DB_PERSISTED_MARKER] = True
                            matched_exact_timestamp = True
                        if matched_exact_timestamp:
                            continue
                    for target_message in target_messages:
                        if not isinstance(target_message, dict):
                            continue
                        if target_message.get(_DB_PERSISTED_MARKER):
                            continue
                        if not _messages_match_scoped_identity(
                            target_message, source_message
                        ):
                            continue
                        target_message[_DB_PERSISTED_MARKER] = True

            if isinstance(result, tuple) and result:
                result_messages = result[0]
                if isinstance(result_messages, list):
                    # Direct-path callers bypass the snapshot worker but still need the post-publish mirror.
                    if direct_path or result_messages is not messages:
                        _sync_persisted_markers(messages, result_messages)
                    session_messages = getattr(self, "_session_messages", None)
                    if (
                        isinstance(session_messages, list)
                        and session_messages is not messages
                    ):
                        # Durable-parent adoption can leave `_session_messages` on the pre-adoption list; sync
                        # both.
                        _sync_persisted_markers(session_messages, result_messages)
            # The worker thread rotated hermes_logging's thread-local session id; propagate to this thread
            # (#34089).
            try:
                from hermes_logging import set_session_context
                set_session_context(self.session_id)
            except Exception:
                pass
            # #76354 F5: rebind the session ContextVar in the CALLER's context so post-compression tools
            # resolve HERMES_SESSION_ID to the child id (idempotent when no rotation happened).
            try:
                from gateway.session_context import set_current_session_id
                if self.session_id:
                    set_current_session_id(self.session_id)
            except Exception:
                logger.debug(
                    "post-compression session ContextVar rebind failed",
                    exc_info=True,
                )
            return result
        finally:
            with fence_registration_lock:
                if previous_fence is missing_fence:
                    vars(self).pop("_active_compression_commit_fence", None)
                else:
                    self._active_compression_commit_fence = previous_fence
            # Restore whatever the caller had, so a compaction never leaks its
            # tag into the surrounding scope.
            if token is not None:
                reset_conversation_context(token)
            if affinity_token is not None:
                reset_affinity_scope(affinity_token)
