"""Context compression: feasibility probe, warning replay, compress, image fix.

Thread-safety contract for extension points
--------------------------------------------
With ``compression.context_timeout_seconds > 0`` (default) the whole pass,
context engines and memory providers included, runs on a pooled daemon thread.
* Calls may arrive on any pooled thread; never rely on thread-affinity/locals.
* The message list is a private deep snapshot; in-place mutation is allowed
  but invisible to the live conversation unless the pass commits.
* State is published ONLY on an admitted :class:`CompressionCommitFence`
  commit; work of an engine still running after a host timeout is discarded.
* One pass per session at a time (durable lock), but different sessions may
  run concurrently, so shared engine/provider instances must be thread-safe.
"""

from __future__ import annotations

import concurrent.futures
import copy
import inspect
import json
import logging
import math
import os
import tempfile
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_engine import (
    automatic_compaction_status_message,
    sanitize_memory_context,
)
from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)
from agent.session_activity import ActivityProvenance, normalize_activity_provenance

logger = logging.getLogger(__name__)

# Terminal outcomes from host/hygiene timeout or cooldown writers. Detached
# heartbeat workers must not clobber these (timeout unobservable). Seeing one
# latches the heartbeat silent so a later UNKNOWN rewrite can't re-arm a zombie.
_TERMINAL_COMPRESSION_PROVENANCES = frozenset(
    {
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
    }
)

# Split failures are usually transient lease/DB conditions, so use the FIRST
# timeout-ladder rung (60s), not the 600s summary-provider cooldown.
_SPLIT_FAILURE_COOLDOWN_SECONDS = 60

# Marker tui_gateway/server.py::_status_update matches to tag kind="compacting"
# for drivers' "Summarizing…" UI. Keep the phrase intact when rewording. Idle/
# preflight/retry lines lack it; is_compaction_progress_status covers those.
COMPACTION_STATUS_MARKER = "Compacting context"
COMPACTION_STATUS = (
    f"🗜️ {COMPACTION_STATUS_MARKER} — summarizing earlier conversation so I can continue..."
)

COMPACTION_DONE_STATUS = "✓ Context compaction complete — continuing turn..."


def _strip_marker_for_comparison(msgs: Any) -> Any:
    """Copy ``msgs`` with the ``_db_persisted`` marker removed for no-op comparison.

    Live dicts carry the marker while ``compress()`` output is swept, so a raw
    ``==`` would misclassify an identical no-op copy as progress. Non-list inputs
    and non-dict entries pass through unchanged.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    if not isinstance(msgs, list):
        return msgs
    return [
        {k: v for k, v in m.items() if k != _DB_PERSISTED_MARKER}
        if isinstance(m, dict)
        else m
        for m in msgs
    ]


def _emit_compaction_done(agent: Any) -> None:
    """Emit the structured terminal edge for a started compaction."""
    status_callback = getattr(agent, "status_callback", None)
    if not status_callback:
        return
    try:
        status_callback("compacted", COMPACTION_DONE_STATUS)
    except Exception:
        logger.debug("status_callback error in compaction completion", exc_info=True)


# Every ROUTINE compression status line lives here: suppressed on chat platforms
# by _TELEGRAM_NOISY_STATUS_RE (gateway/run.py); update that regex + telegram
# noise test when rewording. Failure notices and /compress feedback: NOT here.
PRE_API_COMPRESSION_STATUS_TEMPLATE = (
    "📦 Pre-API compression: ~{tokens:,} tokens "
    "near the context/output limit. Compacting before the next model call."
)
PREFLIGHT_COMPRESSION_STATUS_TEMPLATE = (
    "📦 Preflight compression: ~{tokens:,} tokens "
    ">= {threshold:,} threshold. This may take a moment."
)
IDLE_COMPACTION_STATUS_TEMPLATE = (
    "💤 Resumed after {idle_seconds}s idle — compacting "
    "~{tokens:,} tokens before continuing."
)
COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE = (
    "🗜️ Context too large (~{tokens:,} tokens) — compressing ({attempt}/{cap})..."
)
COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE = (
    "🗜️ Compressed {before} → {after} messages, retrying..."
)
COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE = (
    "🗜️ Compressed ~{before:,} → ~{after:,} tokens, retrying..."
)
COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE = (
    "🗜️ Context reduced to {new_ctx:,} tokens (was {old_ctx:,}), retrying..."
)

# FAILURE-class notice: compression blocked, so the session grows until the
# provider limit kills it. Must stay visible on gateways: never add it to
# ROUTINE_COMPRESSION_STATUS_SAMPLES or _TELEGRAM_NOISY_STATUS_RE.
CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE = (
    "⚠ Context is over the compression threshold "
    "(~{tokens:,} tokens >= {threshold:,}) "
    "but compression is currently blocked ({reason}). "
    "The model may stop responding. Run /new to start a fresh "
    "session or /compress to retry immediately."
)

# Formatted from the same constants the emission sites use, so noise-filter
# tests exercise the ACTUAL wording.
ROUTINE_COMPRESSION_STATUS_SAMPLES = (
    COMPACTION_STATUS,
    COMPACTION_DONE_STATUS,
    PRE_API_COMPRESSION_STATUS_TEMPLATE.format(tokens=123456),
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(tokens=120000, threshold=100000),
    IDLE_COMPACTION_STATUS_TEMPLATE.format(idle_seconds=3600, tokens=120000),
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(tokens=250000, attempt=1, cap=3),
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=30, after=12),
    COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=250000, after=120000),
    COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(
        new_ctx=120000, old_ctx=250000
    ),
)


def is_compaction_progress_status(text: str | None) -> bool:
    """True for in-progress auto-compaction lifecycle lines (not the done edge).

    The gateway re-tags matches as ``kind="compacting"`` for the whole pause;
    matching only the marker left idle/preflight/retry lines looking hung.
    ``COMPACTION_DONE_STATUS`` is emitted as ``kind="compacted"`` and must not
    match here.
    """
    if not isinstance(text, str):
        return False
    body = text.strip()
    if not body:
        return False
    if COMPACTION_STATUS_MARKER in body:
        return True
    if body == COMPACTION_DONE_STATUS:
        return False
    lowered = body.lower()
    if "compaction complete" in lowered:
        return False
    # Failure-class overflow warning mentions compression but is a blocked
    # notice, not progress — keep it lifecycle so chat gateways stay loud.
    if "compression is currently blocked" in lowered:
        return False
    return (
        "compact" in lowered
        or "compress" in lowered
        or "context reduced to" in lowered
    )


def _builtin_memory_prompt_snapshot(agent: Any) -> Optional[Tuple[str, str]]:
    """Return the built-in memory text that can affect a system prompt.

    Rendered after ``load_from_disk()`` so compression can retain a cached prompt
    that already embeds current memory. ``None`` when unreadable, so callers take
    the conservative rebuild path.
    """
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return "", ""
    try:
        memory = (
            store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_memory_enabled", False)
            else ""
        )
        user = (
            store.format_for_system_prompt("user") or ""
            if getattr(agent, "_user_profile_enabled", False)
            else ""
        )
    except Exception:
        return None
    return memory, user


def _refresh_agent_tool_definitions(agent) -> bool:
    """Rebuild agent.tools at the compaction commit boundary.

    Forever-sessions never restart, so this is the only moment config changes
    reach the frozen dynamic tool schemas; the prompt cache is already invalid.
    Delegates to refresh_agent_mcp_tools in content_aware mode (swaps on schema
    CONTENT change). Returns True when tools were added. Never raises.
    """
    from tools.mcp_tool import refresh_agent_mcp_tools

    added = refresh_agent_mcp_tools(agent, content_aware=True)
    if added:
        logger.info(
            "Compaction tool refresh added tools: %s", sorted(added),
        )
    return bool(added)


def _cached_prompt_reflects_builtin_memory(agent: Any, cached_prompt: str) -> bool:
    """Whether the cached system prompt already embeds current built-in memory.

    Do NOT compare snapshots before/after reload: a DB-restored prompt can predate
    memory writes the fresh store already holds, latching stale memory. Instead
    verify current blocks appear verbatim and no stale block header remains.
    """
    snapshot = _builtin_memory_prompt_snapshot(agent)
    if snapshot is None:
        return False
    try:
        from tools.memory_tool import MEMORY_BLOCK_HEADERS
    except Exception:
        return False
    for target, block in zip(("memory", "user"), snapshot):
        block = block.strip()
        if block:
            # The rendered prompt embeds the block verbatim (incl. usage header), so any
            # entry/char-count change breaks containment → rebuild.
            if block not in cached_prompt:
                return False
        elif MEMORY_BLOCK_HEADERS[target] in cached_prompt:
            # The prompt still carries a block for a target that is now
            # empty/disabled — stale; rebuild.
            return False
    return True


_COMPRESSOR_ATTEMPT_STATE_FIELDS = (
    "_previous_summary",
    "_summary_has_user_turn",
    "compression_count",
    "_last_compression_savings_pct",
    "_ineffective_compression_count",
    "_anti_thrash_recovery_deadline",
    "_fallback_compression_streak",
    "_verify_compaction_cleared_threshold",
    "_last_compression_made_progress",
    "_summary_failure_cooldown_until",
    "_cooldown_persist_failed",
    "_last_summary_error",
    "_consecutive_timeout_failures",
    "_last_summary_dropped_count",
    "_last_summary_fallback_used",
    "_last_compress_aborted",
    "_last_summary_auth_failure",
    "_last_summary_network_failure",
    "_last_summary_empty_content_failure",
    "_last_summary_truncated_failure",
    "_last_aux_model_failure_error",
    "_last_aux_model_failure_model",
    "_summary_model_fallen_back",
    "summary_model",
    "_last_compression_telemetry",
    "_active_compression_telemetry",
    "_compression_telemetry_seed",
    "_proactive_prune_rearm_tokens",
)

_COMPRESSOR_COOLDOWN_STATE_FIELDS = (
    "_summary_failure_cooldown_until",
    "_last_summary_error",
    "_cooldown_persist_failed",
)


def _snapshot_compressor_attempt_state(compressor: Any) -> dict[str, Any]:
    """Copy only the mutable bookkeeping owned by one compression attempt.

    The allow-list avoids copying clients, DB handles, locks and plugin resources;
    missing fields are ignored so legacy/third-party compressors keep working.
    """
    try:
        values = vars(compressor)
    except TypeError:
        return {}
    selected = {
        name: values[name]
        for name in _COMPRESSOR_ATTEMPT_STATE_FIELDS
        if name in values
    }
    # Copy the collection as one object so aliases between fields (notably
    # _active_compression_telemetry and _last_compression_telemetry) survive.
    return copy.deepcopy(selected)


# Attempt ownership: stall-fallback detaches a timed-out worker and reuses the
# compressor, so its late unwind could restore a stale snapshot or clear the
# fallback's cancel check. Generation guards ATTRIBUTE writes; fence, COMMITs.

_COMPRESSOR_ATTEMPT_LOCK = threading.Lock()


def _claim_compressor_attempt(compressor: Any) -> int:
    """Claim the compressor for a new attempt; return its monotonic generation id.

    Restores or cancelled-check mutations stamped with an OLDER generation no-op,
    so a detached late attempt cannot clobber its successor's state.
    """
    with _COMPRESSOR_ATTEMPT_LOCK:
        generation = int(getattr(compressor, "_compression_attempt_generation", 0) or 0) + 1
        try:
            compressor._compression_attempt_generation = generation
        except Exception:
            # Slotted/frozen compressor: gen 0 disables the guard. Per-compressor, so gen-0
            # and gen>0 attempts can never coexist on one instance.
            return 0
        return generation


def _compressor_attempt_is_current(compressor: Any, generation: int) -> bool:
    """True when *generation* still owns the compressor (or guard disabled)."""
    if not generation:
        return True
    with _COMPRESSOR_ATTEMPT_LOCK:
        return (
            int(getattr(compressor, "_compression_attempt_generation", 0) or 0)
            == generation
        )


def _install_compression_cancelled_check(
    compressor: Any, check: Any, generation: int
) -> None:
    """Install the F4 cancellation consult, stamped with its owner attempt."""
    with _COMPRESSOR_ATTEMPT_LOCK:
        try:
            compressor._compression_cancelled_check = check
            compressor._compression_cancelled_check_owner = generation
        except Exception:
            pass


def _clear_compression_cancelled_check_if_owner(
    compressor: Any, generation: int
) -> bool:
    """Clear the cancellation consult only when *generation* installed it.

    Prevents a detached late primary from tearing down a newer fallback's
    callback. Returns True when cleared.
    """
    with _COMPRESSOR_ATTEMPT_LOCK:
        owner = getattr(compressor, "_compression_cancelled_check_owner", None)
        if owner is not None and generation and owner != generation:
            return False
        try:
            compressor._compression_cancelled_check = None
            compressor._compression_cancelled_check_owner = None
        except Exception:
            pass
        return True


def _restore_compressor_attempt_state(
    compressor: Any,
    snapshot: dict[str, Any],
    *,
    durable_cooldown_authoritative: Optional[bool] = None,
    durable_cooldown_state: Optional[dict[str, Any]] = None,
    attempt_generation: Optional[int] = None,
) -> None:
    """Restore the per-attempt snapshot after a pre-commit hard cancel.

    A restore stamped with a stale ``attempt_generation`` no-ops so a timed-out
    primary's late unwind cannot roll back state owned by the fallback attempt.
    """
    if attempt_generation is not None and not _compressor_attempt_is_current(
        compressor, attempt_generation
    ):
        logger.warning(
            "Skipping stale compressor attempt-state restore: attempt "
            "generation %s no longer owns the compressor (current: %s). A "
            "newer (stall-fallback) attempt's state is preserved.",
            attempt_generation,
            getattr(compressor, "_compression_attempt_generation", None),
        )
        return
    # Success clears the durable cooldown pre-commit; recreate/clear that row BEFORE
    # restoring in-memory values or the next refresh overwrites the rollback. Never
    # turn unknown durable state / unpersisted local cooldowns into DB writes.
    if (
        "_summary_failure_cooldown_until" in snapshot
        and durable_cooldown_authoritative is not False
        and (
            durable_cooldown_authoritative is True
            or not bool(snapshot.get("_cooldown_persist_failed", False))
        )
    ):
        session_db = vars(compressor).get("_session_db")
        session_id = vars(compressor).get("_session_id")
        if session_db is not None and session_id:
            if durable_cooldown_authoritative is True:
                restorer = getattr(
                    type(session_db),
                    "restore_compression_failure_cooldown_row",
                    None,
                )
                if not callable(restorer) or durable_cooldown_state is None:
                    raise RuntimeError(
                        "exact compression cooldown rollback API is unavailable"
                    )
                # This API restores raw columns (including expired and null
                # combinations), verifies the read-back, and propagates failure.
                restorer(
                    session_db,
                    session_id,
                    copy.deepcopy(durable_cooldown_state),
                )
            else:
                try:
                    deadline = float(
                        snapshot["_summary_failure_cooldown_until"] or 0.0
                    )
                    remaining = max(0.0, deadline - time.monotonic())
                    durable_deadline = time.time() + remaining
                    durable_error = snapshot.get("_last_summary_error")
                    if remaining > 0:
                        recorder = getattr(
                            type(session_db),
                            "record_compression_failure_cooldown",
                            None,
                        )
                        if callable(recorder):
                            recorder(
                                session_db,
                                session_id,
                                durable_deadline,
                                durable_error,
                            )
                    else:
                        clearer = getattr(
                            type(session_db),
                            "clear_compression_failure_cooldown",
                            None,
                        )
                        if callable(clearer):
                            clearer(session_db, session_id)
                except Exception:
                    # Legacy/third-party compatibility path: its existing APIs
                    # do not provide a verifiable transaction contract.
                    logger.debug(
                        "compression cooldown persistence rollback failed",
                        exc_info=True,
                    )
    restored = copy.deepcopy(snapshot)
    # Re-validate under the claim lock: the slow durable rollback above leaves a
    # window where a fallback may have claimed; stale writes must not interleave.
    # The rollback itself is safe: landing after a fallback needs a prior claim.
    with _COMPRESSOR_ATTEMPT_LOCK:
        if attempt_generation is not None and attempt_generation and (
            int(getattr(compressor, "_compression_attempt_generation", 0) or 0)
            != attempt_generation
        ):
            logger.warning(
                "Skipping stale compressor attempt-state restore at write "
                "time: attempt generation %s lost the compressor mid-restore.",
                attempt_generation,
            )
            return
        for name, value in restored.items():
            setattr(compressor, name, value)


def _capture_authoritative_cooldown_under_lease(
    compressor: Any,
    attempt_snapshot: dict[str, Any],
) -> tuple[Optional[bool], Optional[dict[str, Any]]]:
    """Refresh and snapshot built-in durable cooldown state under the lease.

    Third-party compressors are not invoked: plugin code must not run under the
    lease. Returns ``False`` on durable read failure (rollback must not mistake
    unknown state for an empty row) and ``None`` when the legacy API is absent.
    """
    try:
        from agent.context_compressor import ContextCompressor

        if not isinstance(compressor, ContextCompressor):
            return None, None
        values = vars(compressor)
        session_db = values.get("_session_db")
        session_id = values.get("_session_id")
        raw_reader = (
            getattr(
                type(session_db), "get_compression_failure_cooldown_row", None
            )
            if session_db is not None
            else None
        )
        if session_db is None or not session_id:
            # Unbound compressors have no durable row to mutate or restore.
            return None, None
        if not callable(raw_reader):
            return False, None
        # Read the raw persisted row: the active getter filters expired rows and is not
        # a lossless rollback snapshot.
        durable_state = raw_reader(session_db, session_id)
        if not isinstance(durable_state, dict):
            raise TypeError("raw compression cooldown snapshot must be a mapping")
        ContextCompressor.get_active_compression_failure_cooldown(
            compressor,
            refresh=True,
        )
    except Exception as exc:
        logger.debug("authoritative compression cooldown capture failed: %s", exc)
        return False, None
    authoritative = getattr(
        compressor, "_last_cooldown_refresh_was_authoritative", None
    )
    if authoritative is not True:
        return authoritative, None

    values = vars(compressor)
    for name in _COMPRESSOR_COOLDOWN_STATE_FIELDS:
        if name in values:
            attempt_snapshot[name] = copy.deepcopy(values[name])
    return True, copy.deepcopy(durable_state)


class CompressionCommitFence:
    """Fence timeout cancellation against post-summary session mutation.

    The sync worker thread cannot be killed; the fence makes the commit boundary
    deterministic: cancellation wins before mutation starts, or waits for an
    already-started commit to finish completely.
    """

    def __init__(self, total_ceiling_seconds: float | None = None) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._commit_started = False
        # begin_commit holds self._lock until finish_commit, so this Event is readable
        # WITHOUT the lock: hosts can see a hung commit and fire the overrun warning.
        self._commit_phase = threading.Event()
        # Set on ANY host unwind without the fence lock, so a host that cannot block
        # behind an in-flight commit still blocks FUTURE commits. bool store is atomic.
        self._admission_revoked = False
        # Worker publishes a holder-scoped release once it owns the durable lock; a
        # timed-out host frees the lease without racing a NEW holder (no ABA).
        self._lock_release_guard = threading.Lock()
        self._cancelled_lock_release: Optional[Callable[[], None]] = None
        self._cancelled_lock_release_requested = False
        # Touched per streamed summary token; waiters distinguish SLOW-but-alive from
        # HUNG so slow models are not killed by a fixed wall-clock deadline.
        self._last_progress = time.monotonic()
        self._progress_observed = False
        self._deadline: float | None = None
        self._retain_cancelled_lock_until_worker_done = False
        # Set once the commit path captured the active-row watermark: later rows survive
        # as concurrent tail, so hosts may KEEP a detached worker's commit admission.
        self._commit_watermark_fenced = False
        if total_ceiling_seconds is not None:
            self.set_total_ceiling_seconds(total_ceiling_seconds)

    def set_total_ceiling_seconds(self, seconds: float) -> None:
        """Arm the wall-clock deadline shared by the host and worker."""
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("total compression ceiling must be positive")
        self._deadline = time.monotonic() + seconds

    def touch_progress(self) -> None:
        """Record forward progress (e.g. a streamed summary token arriving).

        Called from the worker thread, read by waiters via ``seconds_since_progress``;
        a bare float store is atomic in CPython so no lock is needed.
        """
        self._last_progress = time.monotonic()
        self._progress_observed = True

    @property
    def progress_observed(self) -> bool:
        """Whether semantic provider progress was reported for this attempt."""
        return self._progress_observed

    @property
    def deadline_exceeded(self) -> bool:
        deadline = self._deadline
        return deadline is not None and time.monotonic() >= deadline

    @property
    def deadline_monotonic(self) -> float | None:
        """The armed deadline as an absolute ``time.monotonic()`` instant.

        Published so the worker's stream consumer can stop exactly when the host
        stops waiting (see ``auxiliary_client.aux_stream_deadline``).
        """
        return self._deadline

    def seconds_since_progress(self) -> float:
        """Seconds since the worker last reported forward progress."""
        return max(0.0, time.monotonic() - self._last_progress)

    def cancel_before_commit(self, cancel_event: Any = None) -> bool:
        """Cancel a pending commit, or wait for an active commit to finish.

        Returns ``True`` when cancellation won before the commit boundary; ``False``
        after blocking until an already-started commit fully completed.
        """
        with self._lock:
            if self._commit_started:
                if cancel_event is not None:
                    cancel_event.set()
                return False
            self._cancelled = True
            if cancel_event is not None:
                cancel_event.set()
            return True

    def try_cancel_before_commit(self) -> Optional[bool]:
        """Non-blocking form of :meth:`cancel_before_commit`.

        Returns ``None`` while an active commit owns the fence so an async caller can
        yield instead of blocking its event loop.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._commit_started:
                return False
            self._cancelled = True
            return True
        finally:
            self._lock.release()

    def begin_commit(self, cancel_event: Any = None) -> bool:
        """Atomically admit commit unless a hard cancellation already won."""
        self._lock.acquire()
        if (
            self.is_cancelled
            or self._admission_revoked
            or (cancel_event is not None and bool(cancel_event.is_set()))
        ):
            self._cancelled = True
            self._lock.release()
            if self._admission_revoked:
                # A revoke that lost the fence-lock race deferred its lease release; commit was
                # refused, so releasing now is safe (idempotent with holder-qualified cleanup).
                self.release_cancelled_compression_lock()
            return False
        self._commit_started = True
        # Set while the fence lock is held so observers can never see
        # commit_in_flight=True for a commit that lost to cancellation.
        self._commit_phase.set()
        return True

    def finish_commit(self) -> None:
        """Leave a commit boundary entered by :meth:`begin_commit`."""
        self._commit_phase.clear()
        self._lock.release()
        if self._admission_revoked:
            # A revoke during THIS commit deferred its lease release (no freeing under an
            # active SessionDB mutation); commit is done, release now. Holder-qualified.
            self.release_cancelled_compression_lock()

    @property
    def commit_in_flight(self) -> bool:
        """Lock-free read: an admitted commit has begun and not yet finished.

        Safe while the worker holds the fence lock for a hung commit; lets hosts reach
        the overrun-warning loop instead of spinning on ``try_cancel_before_commit``.
        """
        return self._commit_phase.is_set()

    @property
    def is_cancelled(self) -> bool:
        """True after cancellation won before the commit boundary."""
        return self._cancelled or self._admission_revoked or self.deadline_exceeded

    def retain_compression_lock_until_worker_done(self) -> None:
        """Prevent a timed-out live worker from overlapping a retry."""
        self._retain_cancelled_lock_until_worker_done = True

    def mark_commit_watermark_fenced(self) -> None:
        """Record that this attempt's commit is bounded by a start watermark.

        A watermark-fenced commit archives only rows at or below the watermark and
        clones later rows as live tail, so a detached worker may keep its admission.
        """
        self._commit_watermark_fenced = True

    @property
    def commit_watermark_fenced(self) -> bool:
        """Lock-free read: the worker's commit is watermark-bounded."""
        return self._commit_watermark_fenced

    def allow_cancelled_lock_release(self) -> None:
        """Undo :meth:`retain_compression_lock_until_worker_done`.

        Called after a bounded join confirmed the timed-out worker exited, so the
        durable lease may be released and a fallback attempt can proceed.
        """
        self._retain_cancelled_lock_until_worker_done = False

    def revoke_commit_admission(self) -> None:
        """Revoke FUTURE commit admission without blocking on the fence lock.

        An in-flight commit is never abandoned, but ``begin_commit`` re-checks the
        flag under the lock so no new commit is admitted. The lease release must not
        run mid-commit (a second compressor could interleave): released now if the
        lock is free, else deferred to ``finish_commit``/refusal (holder-qualified).
        """
        self._admission_revoked = True
        if self._lock.acquire(blocking=False):
            try:
                self.release_cancelled_compression_lock()
            finally:
                self._lock.release()
        # else: deferred — finish_commit()/begin_commit() re-check _admission_revoked
        # and release once no commit can be mid-mutation.

    # ── Holder-qualified durable-lease cancellation: release is DELETE WHERE
    # holder = ?, so a stale release can never free a NEW holder's lease (no ABA).

    def begin_lock_setup(self) -> bool:
        """Fence durable-lock acquisition and release-hook publication.

        The caller holds the fence until the holder-qualified release hook is
        published (or no lock was taken), so a timeout cannot win in that gap.
        """
        self._lock.acquire()
        if self.is_cancelled or self._admission_revoked:
            self._lock.release()
            return False
        return True

    def finish_lock_setup(self) -> None:
        """Leave a lock setup boundary entered by :meth:`begin_lock_setup`."""
        self._lock.release()

    def register_cancelled_lock_release(
        self, release: Callable[[], None]
    ) -> bool:
        """Publish the timed-out worker's holder-qualified lock release.

        Returns whether cleanup was already requested; in that race the release runs
        synchronously before returning.
        """
        with self._lock_release_guard:
            self._cancelled_lock_release = release
            requested = self._cancelled_lock_release_requested
        if requested:
            release()
        return requested

    def clear_cancelled_lock_release(self, release: Callable[[], None]) -> None:
        """Forget ``release`` after the worker's normal cleanup finishes."""
        with self._lock_release_guard:
            if self._cancelled_lock_release is release:
                self._cancelled_lock_release = None

    def release_cancelled_compression_lock(self) -> None:
        """Release the cancelled worker's lock without finalizing its clients.

        Only valid after cancellation won. A request racing ahead of hook publication
        is retained and fulfilled when the worker publishes the hook.
        """
        if self._retain_cancelled_lock_until_worker_done:
            return
        with self._lock_release_guard:
            self._cancelled_lock_release_requested = True
            release = self._cancelled_lock_release
        if release is not None:
            release()


# Defaults for the in-agent (non-hygiene) progress-aware compress_context wrap.
# Mirror hermes_cli.config.DEFAULT_CONFIG["compression"] keys of the same name.
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS = 600.0

# Unlike explicit_interrupt: a /stop after the stall window arms the durable
# backoff so the next automatic turn does not re-enter the stalled strategy.
STALL_INTERRUPTED_FAILURE_CLASS = "stall_interrupted"

# Daemon pool so a fence-cancelled hung worker cannot block interpreter exit
# via the atexit join. Never shut down per call (workers may still be winding).
_compress_timeout_executor = None
_compress_timeout_executor_lock = threading.Lock()

# Overrun waits proceed in bounded slices so each window logs (escalating)
# instead of one silent future.result(). Clamped to ceiling for tiny test values
_COMMIT_OVERRUN_WAIT_SLICE_SECONDS = 30.0

# A worker exiting within the grace proves no provider call is in flight, so the
# lease can be released even on the total-ceiling path. One that doesn't exit is
# orphaned behind the poison fence and keeps its lease so no attempt overlaps.
_CANCELLED_WORKER_TEARDOWN_GRACE_SECONDS = 5.0


def _join_cancelled_worker(future: Any, grace_seconds: float) -> bool:
    """Best-effort bounded join of a fence-cancelled compression worker.

    Returns True when the future settled within ``grace_seconds`` (thread provably
    exited); False for a still-running worker, which the caller must treat as an
    orphan behind the poison fence.
    """
    try:
        grace = max(float(grace_seconds), 0.0)
    except (TypeError, ValueError):
        grace = 0.0
    try:
        future.result(timeout=grace)
        return True
    except concurrent.futures.TimeoutError:
        return False
    except concurrent.futures.CancelledError:
        # Never started; nothing can be in flight.
        return True
    except Exception:
        # Exception swallowed: the host already chose the fallback result and the fence
        # keeps the failed attempt from touching session state.
        logger.debug(
            "cancelled compression worker exited with an exception",
            exc_info=True,
        )
        return True


# Executor queue is unbounded: a queued job would wait out its timeout unstarted
# and run stale later. Cap admission at worker count; fail fast (warn, continue
# uncompressed). Slots free via done-callback; a never-returning worker loses 1.
_COMPRESS_EXECUTOR_MAX_WORKERS = 4
_compress_admission_lock = threading.Lock()
_compress_admitted_count = 0


class CompressionExecutorSaturatedError(RuntimeError):
    """All compression pool slots are occupied; submission was refused."""


def _try_admit_compression_job() -> bool:
    """Reserve one bounded compression-pool admission slot (F6)."""
    global _compress_admitted_count
    with _compress_admission_lock:
        if _compress_admitted_count >= _COMPRESS_EXECUTOR_MAX_WORKERS:
            return False
        _compress_admitted_count += 1
        return True


def _release_compression_admission(_future=None) -> None:
    """Free an admission slot (future done-callback or failed submit)."""
    global _compress_admitted_count
    with _compress_admission_lock:
        if _compress_admitted_count > 0:
            _compress_admitted_count -= 1


def _get_compress_timeout_executor():
    """Return the process-wide compress-timeout DaemonThreadPoolExecutor."""
    global _compress_timeout_executor
    executor = _compress_timeout_executor
    if executor is not None:
        return executor
    from tools.daemon_pool import DaemonThreadPoolExecutor

    with _compress_timeout_executor_lock:
        if _compress_timeout_executor is None:
            # Small pool: compress is rare/heavy; sized for live compress + cancelled
            # workers still winding down, not asyncio's min(32, cpu+4).
            _compress_timeout_executor = DaemonThreadPoolExecutor(
                max_workers=_COMPRESS_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="compress-ctx-timeout",
            )
        return _compress_timeout_executor


def resolve_context_compression_timeouts(
    compression_cfg: Optional[dict] = None,
) -> Tuple[float, float]:
    """Return ``(idle_timeout_seconds, total_ceiling_seconds)``.

    ``idle_timeout_seconds <= 0`` disables the progress-aware wrapper. The ceiling
    is clamped to at least one idle window when the idle budget is positive.
    """
    idle = DEFAULT_CONTEXT_TIMEOUT_SECONDS
    ceiling = DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
    cfg = compression_cfg
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            raw = load_config()
            maybe = raw.get("compression", {}) if isinstance(raw, dict) else {}
            cfg = maybe if isinstance(maybe, dict) else {}
        except Exception:
            cfg = {}
    if isinstance(cfg, dict):
        raw_idle = cfg.get("context_timeout_seconds")
        if raw_idle is not None:
            try:
                parsed = float(raw_idle)
                # Explicit 0/negative disables; positive values win.
                idle = parsed
            except (TypeError, ValueError):
                pass
        raw_ceiling = cfg.get("context_total_ceiling_seconds")
        if raw_ceiling is not None:
            try:
                parsed = float(raw_ceiling)
                if parsed > 0:
                    ceiling = parsed
            except (TypeError, ValueError):
                pass
    if idle > 0:
        ceiling = max(ceiling, idle)
    return idle, ceiling


def compression_attempt_stalled(
    *,
    commit_fence: Optional[CompressionCommitFence],
    started_at: float,
    idle_timeout_seconds: Optional[float] = None,
) -> bool:
    """Return whether a pre-commit cancel landed after the stall window.

    An early ``/stop`` stays cooldown-neutral; an interrupt after the inactivity
    budget counts as a stall so the next automatic turn does not blindly retry.
    """
    idle = idle_timeout_seconds
    if idle is None:
        idle, _ceiling = resolve_context_compression_timeouts()
    try:
        idle = float(idle)
    except (TypeError, ValueError):
        return False
    if idle <= 0:
        return False
    if commit_fence is not None:
        try:
            return float(commit_fence.seconds_since_progress()) >= idle
        except Exception:
            return False
    try:
        return (time.monotonic() - float(started_at)) >= idle
    except (TypeError, ValueError):
        return False


def _stall_source_fingerprint(
    agent: Any,
    messages: Any,
    approx_tokens: Optional[int],
) -> str:
    """Identity of the stalled source context + summary strategy."""
    compressor = getattr(agent, "context_compressor", None)
    model = (
        getattr(compressor, "summary_model", None)
        or getattr(agent, "model", None)
        or ""
    )
    n_messages = len(messages) if isinstance(messages, list) else 0
    try:
        tokens = int(approx_tokens or 0)
    except (TypeError, ValueError):
        tokens = 0
    return f"msgs={n_messages}:tokens={tokens}:model={model}"


def _record_stall_interrupted_backoff(
    agent: Any,
    *,
    commit_fence: Optional[CompressionCommitFence],
    started_at: float,
    messages: Any,
    approx_tokens: Optional[int],
) -> bool:
    """Persist a stall-interrupted cooldown after snapshot restore.

    Must run *after* ``_restore_compressor_attempt_state`` so rollback cannot wipe
    the new row. Returns True when the backoff was recorded.
    """
    if not compression_attempt_stalled(
        commit_fence=commit_fence, started_at=started_at
    ):
        return False
    compressor = getattr(agent, "context_compressor", None)
    record = getattr(compressor, "record_timeout_failure", None)
    if not callable(record):
        return False
    error = (
        f"{STALL_INTERRUPTED_FAILURE_CLASS}:"
        f"{_stall_source_fingerprint(agent, messages, approx_tokens)}"
    )
    try:
        record(error, failure_kind="stall_interrupted")
    except Exception:
        logger.debug(
            "stall-interrupted compression cooldown persist failed",
            exc_info=True,
        )
        return False
    logger.info(
        "Recorded stall-interrupted compression backoff (session=%s, %s)",
        getattr(agent, "session_id", None) or "none",
        error,
    )
    return True


def resolve_compression_fallback_route() -> Optional[dict]:
    """Return the first usable ``auxiliary.compression.fallback_chain`` entry.

    The aux client applies the chain only from its exception handler, so a silent
    stall never reaches it; this pins the route onto one bounded retry instead.
    Only the first complete entry: if it errors, the aux client's own exception
    path walks the rest. ``None`` when none is usable (skip compression).
    """
    try:
        from agent.auxiliary_client import (
            _fallback_entry_api_key,
            _get_auxiliary_task_config,
        )

        chain = _get_auxiliary_task_config("compression").get("fallback_chain")
    except Exception:
        logger.debug("compression fallback_chain lookup failed", exc_info=True)
        return None
    if not isinstance(chain, list):
        return None

    for index, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        # Both are required to name a route. _resolve_fallback_entry applies
        # the same rule when the aux client walks this chain itself.
        if not provider or not model:
            continue
        try:
            api_key = _fallback_entry_api_key(entry)
        except Exception:
            logger.debug(
                "compression fallback_chain[%d] api key resolution failed",
                index,
                exc_info=True,
            )
            api_key = None
        from agent.auxiliary_client import _coerce_positive_timeout

        timeout = _coerce_positive_timeout(entry.get("timeout"))
        return {
            "label": f"fallback_chain[{index}]({provider})",
            "provider": provider,
            "model": model,
            "base_url": str(entry.get("base_url") or "").strip() or None,
            "api_key": api_key or None,
            "api_mode": str(
                entry.get("api_mode") or entry.get("transport") or ""
            ).strip() or None,
            "timeout": timeout,
        }
    return None


def _retry_compression_on_fallback_chain(
    *,
    worker: Callable[[CompressionCommitFence], Tuple[list, str]],
    messages: list,
    system_prompt_fallback: Any,
    idle_timeout_seconds: float,
    total_ceiling_seconds: float,
    on_commit_overrun: Optional[Callable[[float, float], None]] = None,
    on_timeout_cause: Optional[Callable[[bool, bool], None]] = None,
    telemetry_agent: Any = None,
    new_fence: Optional[Callable[[], CompressionCommitFence]] = None,
) -> Optional[Tuple[list, str]]:
    """Re-run an aborted compression once with the summary route pinned.

    Returns ``(messages, system_prompt)`` on real compression, else ``None`` and
    the caller degrades as before. The entry's ``timeout`` sets the idle window.
    Re-runs the whole worker, so pre-compression callbacks must be idempotent.
    """
    # An explicit stop is not a stalled route. The retry worker would abort on
    # the same event anyway, but starting one at all makes /stop look ignored.
    hard_cancel = getattr(telemetry_agent, "_hard_interrupt_requested", None)
    if callable(getattr(hard_cancel, "is_set", None)) and hard_cancel.is_set():
        return None

    route = resolve_compression_fallback_route()
    if route is None:
        return None

    # The aborted fence refuses all commits; mint a fresh one via the host factory
    # so a /stop during the retry serializes against THIS attempt's commit boundary.
    retry_fence = None
    if new_fence is not None:
        try:
            retry_fence = new_fence()
        except Exception:
            logger.warning(
                "compression stall-fallback fence factory failed; the retry "
                "will run on an unpublished fence (a /stop mid-retry cannot "
                "serialize against its commit boundary)",
                exc_info=True,
            )
    if not isinstance(retry_fence, CompressionCommitFence):
        logger.warning(
            "compression stall-fallback retry running on an unpublished fence; "
            "hard-interrupt admission will read the aborted attempt's fence "
            "rather than the retry's commit boundary",
        )
        retry_fence = CompressionCommitFence()
    idle = float(route.get("timeout") or idle_timeout_seconds)
    ceiling = max(float(total_ceiling_seconds), idle)
    logger.warning(
        "Context compression stalled on the configured summary route — "
        "retrying once on %s (%s) before continuing without compression",
        route["label"],
        route["model"],
    )
    try:
        from agent.context_compressor import pin_summary_route

        with pin_summary_route(route):
            result_msgs, result_prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=messages,
                system_prompt_fallback=system_prompt_fallback,
                idle_timeout_seconds=idle,
                total_ceiling_seconds=ceiling,
                on_commit_overrun=on_commit_overrun,
                on_timeout_cause=on_timeout_cause,
                fence=retry_fence,
                telemetry_agent=telemetry_agent,
                stall_fallback=False,
            )
    except Exception:
        # The primary already failed; a failing fallback must degrade, never
        # turn "continue without compression" into a raised turn.
        logger.warning(
            "Context compression fallback attempt on %s failed",
            route["label"],
            exc_info=True,
        )
        return None
    if result_msgs is messages:
        # Aborted or no-op: the worker hands back the caller's own list.
        logger.warning(
            "Context compression fallback attempt on %s produced no "
            "compression; continuing without compression",
            route["label"],
        )
        return None
    logger.info(
        "Context compression recovered on %s after the primary summary route "
        "stalled",
        route["label"],
    )
    return result_msgs, result_prompt


def run_compress_context_with_progress_timeout(
    *,
    worker: Callable[[CompressionCommitFence], Tuple[list, str]],
    messages: list,
    system_prompt_fallback: Any,
    idle_timeout_seconds: float,
    total_ceiling_seconds: float,
    on_timeout: Optional[Callable[[float, float, float], None]] = None,
    on_timeout_cause: Optional[Callable[[bool, bool], None]] = None,
    on_commit_overrun: Optional[Callable[[float, float], None]] = None,
    fence: Optional[CompressionCommitFence] = None,
    telemetry_agent: Any = None,
    stall_fallback: bool = True,
    new_fence: Optional[Callable[[], CompressionCommitFence]] = None,
) -> Tuple[list, str]:
    """Run ``worker(fence)`` under a sync progress-aware (idle + ceiling) timeout.

    Budgets bound the PRE-commit phase only: an admitted commit always completes
    (overrun logged, surfaced once via ``on_commit_overrun``). A pre-commit cancel
    returns ``(messages, system_prompt_fallback)`` (lazy callable), detaching the
    worker; a stall first retries the chain once on ``new_fence``, then on_timeout
    """
    if idle_timeout_seconds <= 0:
        raise ValueError(
            "run_compress_context_with_progress_timeout requires "
            "idle_timeout_seconds > 0; call compress_context directly to disable"
        )

    def _resolve_fallback_prompt() -> str:
        if callable(system_prompt_fallback):
            return system_prompt_fallback()
        return system_prompt_fallback

    ceiling = max(float(total_ceiling_seconds), float(idle_timeout_seconds))
    idle = float(idle_timeout_seconds)
    fence = fence if fence is not None else CompressionCommitFence()
    fence.set_total_ceiling_seconds(ceiling)
    # Sync mirror of gateway hygiene's run_in_executor + wait_for loop: offload,
    # poll idle budget + ceiling, fence-cancel on timeout so no late commit lands.
    from tools.thread_context import propagate_context_to_thread

    executor = _get_compress_timeout_executor()
    # Refuse rather than queue when the pool is full: a queued job would wait out
    # its budget unstarted and run stale later. Skip compression this cycle.
    if not _try_admit_compression_job():
        logger.warning(
            "Context compression pool saturated (%d workers busy) — "
            "refusing new compression this cycle and continuing without "
            "compression. Wedged workers are fence-cancelled and free their "
            "slot when they return; if this persists, check the summary "
            "provider health.",
            _COMPRESS_EXECUTOR_MAX_WORKERS,
        )
        # Saturation refusals must hit the same telemetry stream as other failures, or
        # a wedged pool looks like compression simply stopped being attempted.
        if telemetry_agent is not None:
            _emit_compression_attempt_telemetry(
                telemetry_agent,
                started_at=time.monotonic(),
                commit_status="aborted",
                split_status="aborted",
                failure_class="pool_saturated",
            )
        return messages, _resolve_fallback_prompt()

    def _fence_gated_worker(worker_fence: CompressionCommitFence):
        # An admitted job may start after the host stopped waiting; check the fence
        # BEFORE summary work so a stale job never burns an LLM call.
        if worker_fence.deadline_exceeded:
            raise concurrent.futures.TimeoutError(
                "compression deadline expired before worker start"
            )
        if worker_fence.is_cancelled:
            logger.info(
                "Skipping stale compression job: fence cancelled before start"
            )
            return messages, ""
        return worker(worker_fence)

    # Bare pool workers start with an empty ContextVar map; propagate the
    # parent conversation/approval context into the worker.
    try:
        future = executor.submit(
            propagate_context_to_thread(_fence_gated_worker), fence
        )
    except BaseException:
        _release_compression_admission()
        raise
    future.add_done_callback(_release_compression_admission)
    wait_started = time.monotonic()
    # EVERY host unwind must revoke commit admission or a detached worker could
    # later mutate durable state; handled_exit marks paths that settle it themselves
    handled_exit = False
    try:
        while True:
            waited = time.monotonic() - wait_started
            remaining_ceiling = ceiling - waited
            if remaining_ceiling <= 0:
                break
            # Charge idle budget from LAST PROGRESS, not slice start, or silence could
            # approach 2x the budget.
            since_progress = fence.seconds_since_progress()
            wait_slice = min(
                max(idle - since_progress, 0.005), remaining_ceiling
            )
            try:
                result = future.result(timeout=wait_slice)
                handled_exit = True
                return result
            except concurrent.futures.TimeoutError:
                waited = time.monotonic() - wait_started
                since_progress = fence.seconds_since_progress()
                if (
                    not fence.deadline_exceeded
                    and since_progress < idle
                    and waited < ceiling
                ):
                    logger.info(
                        "Context compression still streaming after %.0fs "
                        "(last progress %.1fs ago) — extending wait "
                        "(ceiling %.0fs)",
                        waited,
                        since_progress,
                        ceiling,
                    )
                    continue
                break

        # F6: a not-yet-started future must not linger as a stale queued job.
        # cancel() is a no-op for a running worker (fence handles that path).
        future.cancel()

        total_exhausted = (
            time.monotonic() - wait_started >= ceiling or fence.deadline_exceeded
        )
        if total_exhausted:
            # A total-ceiling candidate may be unwinding a healthy provider call; keep its
            # lease until it exits so no other attempt overlaps the unchanged source.
            fence.retain_compression_lock_until_worker_done()

        if on_timeout_cause is not None:
            try:
                on_timeout_cause(total_exhausted, fence.progress_observed)
            except Exception:
                logger.debug(
                    "compress_context timeout-cause callback failed",
                    exc_info=True,
                )

        cancelled: Optional[bool] = None
        while cancelled is None:
            # begin_commit holds the fence lock until finish_commit, so try_cancel spins
            # forever on a hung commit; lock-free marker makes the overrun loop reachable.
            if fence.commit_in_flight:
                cancelled = False
                break
            cancelled = fence.try_cancel_before_commit()
            if cancelled is None:
                # Fence is held only transiently here, but that window rides SessionDB write
                # patience (seconds). 25ms keeps sub-tick latency without a 1kHz spin.
                time.sleep(0.025)
        if not cancelled:
            # begin_commit won the race: SessionDB mutation cannot be fence-cancelled, so
            # wait in bounded slices, logging (escalating) + surfacing once via
            # on_commit_overrun WHILE the commit hangs. Never silently hung or abandoned.
            overrun_surfaced = False
            overrun_reports = 0
            while True:
                waited = time.monotonic() - wait_started
                remaining = ceiling - waited
                if remaining <= 0:
                    # Bounded increments so each overrun window is visible in logs rather than one
                    # silent unbounded block.
                    remaining = min(
                        _COMMIT_OVERRUN_WAIT_SLICE_SECONDS,
                        max(ceiling, 0.05),
                    )
                    overrun_reports += 1
                    log = (
                        logger.warning if overrun_reports <= 2 else logger.error
                    )
                    log(
                        "Context compression SessionDB commit still running "
                        "%.1fs past the total ceiling (waited %.1fs, ceiling "
                        "%.1fs); commit cannot be abandoned mid-flight — "
                        "continuing to wait (check SessionDB health if this "
                        "persists)",
                        waited - ceiling,
                        waited,
                        ceiling,
                    )
                    if not overrun_surfaced and on_commit_overrun is not None:
                        overrun_surfaced = True
                        try:
                            on_commit_overrun(waited, ceiling)
                        except Exception:
                            logger.debug(
                                "compress_context commit-overrun callback "
                                "failed",
                                exc_info=True,
                            )
                try:
                    result = future.result(timeout=remaining)
                    handled_exit = True
                    return result
                except concurrent.futures.TimeoutError:
                    # Commit-phase progress is informative only — the commit must complete; loop
                    # and re-report with the updated overrun window.
                    continue

        # Idle-timeout: cancel won pre-commit. Also free the worker's durable lease via
        # the holder-qualified hook so a NEW compressor can acquire at once (no ABA).
        handled_exit = True
        # Total-ceiling only: bounded grace for the worker to exit (it checks the fence
        # between provider phases; an uninterruptible call is orphaned). Idle-stall
        # skips the join: worker is hung, fallback needs a prompt return, fence guards.
        if total_exhausted:
            worker_exited = _join_cancelled_worker(
                future,
                min(_CANCELLED_WORKER_TEARDOWN_GRACE_SECONDS, ceiling),
            )
            if worker_exited:
                # Worker provably exited: no provider call can outlive this attempt, so lease
                # retention is unneeded and a retry cannot overlap.
                fence.allow_cancelled_lock_release()
            else:
                logger.warning(
                    "Cancelled compression worker did not exit within %.1fs "
                    "grace — orphaning it behind the poison fence (late "
                    "result will be discarded); retaining the session "
                    "compression lease until it exits so no new attempt "
                    "overlaps it",
                    min(_CANCELLED_WORKER_TEARDOWN_GRACE_SECONDS, ceiling),
                )
        fence.release_cancelled_compression_lock()
        waited = time.monotonic() - wait_started
        since_progress = fence.seconds_since_progress()
        # Lease is free, so run the fallback BEFORE on_timeout: that callback records
        # the summary-failure cooldown, which would no-op the retry's summary call.
        if stall_fallback:
            recovered = _retry_compression_on_fallback_chain(
                worker=worker,
                messages=messages,
                system_prompt_fallback=system_prompt_fallback,
                idle_timeout_seconds=idle,
                total_ceiling_seconds=ceiling,
                on_commit_overrun=on_commit_overrun,
                on_timeout_cause=on_timeout_cause,
                telemetry_agent=telemetry_agent,
                new_fence=new_fence,
            )
            if recovered is not None:
                return recovered
        if on_timeout is not None:
            try:
                on_timeout(idle, waited, since_progress)
            except Exception:
                logger.debug(
                    "compress_context timeout callback failed",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Context compression made no progress for %.1fs "
                "(total wait %.1fs, ceiling %.1fs); continuing without "
                "compression",
                since_progress,
                waited,
                ceiling,
            )
        # Leave the future on the shared pool: fence cancel won, so a late
        # commit cannot land (same detachment model as gateway hygiene).
        return messages, _resolve_fallback_prompt()
    finally:
        if not handled_exit:
            # Any unwind while waiting: revoke commit admission and release the worker's
            # lease before the host unwinds, so the detached worker can never publish.
            fence.revoke_commit_admission()


class CompressionCheckpointUnavailable(RuntimeError):
    """Raised when required durable pre-compress checkpointing is unavailable."""


def _checkpoint_blocked(reason: str) -> CompressionCheckpointUnavailable:
    return CompressionCheckpointUnavailable(
        "BLOCKED_MISSING_PREREQUISITE: required pre-compress checkpoint "
        f"unavailable: {reason}"
    )


def _lock_api_is_absent_on_session_db(lock_db: Any) -> bool:
    """Whether the live in-memory SessionDB class structurally predates locks.

    Only the exact old ``hermes_state.SessionDB`` class (hot-reload skew) may fail
    open; proxies, lookalikes, non-callables and descriptor failures fail closed.
    """
    try:
        from hermes_state import SessionDB

        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(
                SessionDB, "try_acquire_compression_lock", missing
            ) is missing
        )
    except Exception:
        return False


def _refresh_persisted_compression_guards(
    compressor: Any,
    *,
    include_cooldown: bool = True,
) -> None:
    """Refresh durable automatic-compression guards on a built-in compressor."""
    method_calls = [
        ("_load_fallback_compression_streak", {}),
        ("_load_ineffective_compression_count", {}),
    ]
    if include_cooldown:
        method_calls.insert(
            0,
            ("get_active_compression_failure_cooldown", {"refresh": True}),
        )
    for method_name, kwargs in method_calls:
        method = getattr(type(compressor), method_name, None)
        if not callable(method):
            continue
        try:
            method(compressor, **kwargs)
        except Exception as exc:
            logger.debug("compression guard refresh failed (%s): %s", method_name, exc)


def _session_was_rotated_by_compression(session_db: Any, session_id: str) -> bool:
    """Return whether another path already rotated this compression parent."""
    getter = getattr(type(session_db), "get_session", None)
    if not callable(getter):
        return False
    session = getter(session_db, session_id)
    return bool(
        session
        and session.get("ended_at") is not None
        and session.get("end_reason") == "compression"
    )


def _emit_compression_attempt_telemetry(
    agent: Any,
    *,
    started_at: float,
    commit_status: str,
    split_status: str,
    failure_class: str | None = None,
    commit_started_at: float | None = None,
) -> None:
    """Emit one content-free JSON log line for a compression attempt."""
    try:
        telemetry = getattr(agent.context_compressor, "_last_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            telemetry = {}
        payload = dict(telemetry)
        payload.setdefault("event", "compression_attempt")
        payload.setdefault("attempt_id", getattr(agent, "_compression_attempt_id", "") or uuid.uuid4().hex)
        payload.setdefault("session_id", getattr(agent, "session_id", "") or "")
        payload["total_duration_ms"] = int((time.monotonic() - started_at) * 1000)
        payload["commit_status"] = commit_status
        payload["split_status"] = split_status
        if commit_started_at is not None:
            commit_ms = max(0, int((time.monotonic() - commit_started_at) * 1000))
            telemetry["commit_ms"] = commit_ms
            payload["commit_ms"] = commit_ms
        if failure_class:
            payload["failure_class"] = failure_class
        payload.setdefault("chunking", False)
        payload.setdefault("chunk_count", 0)
        payload["fallback_used"] = bool(
            payload.get("fallback_used")
            or getattr(agent.context_compressor, "_last_summary_fallback_used", False)
            or getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
        )
        logger.info(
            "context compression attempt telemetry: %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:
        logger.debug("failed to emit compression attempt telemetry: %s", exc)


def _existing_system_prompt(agent: Any, system_message: str) -> str:
    """Cached system prompt, or a fresh build when nothing is cached (abort paths)."""
    existing = getattr(agent, "_cached_system_prompt", None)
    if not existing:
        existing = agent._build_system_prompt(system_message)
    return existing


def _emit_aborted_attempt_telemetry(
    agent: Any, started_at: float, failure_class: str | None
) -> None:
    _emit_compression_attempt_telemetry(
        agent,
        started_at=started_at,
        commit_status="aborted",
        split_status="aborted",
        failure_class=failure_class,
    )


def _restore_messages_snapshot(messages: list, snapshot: Optional[list]) -> None:
    """Put the pre-compression deep snapshot back into the live list if it drifted."""
    if snapshot is not None and messages != snapshot:
        messages[:] = copy.deepcopy(snapshot)


def _restore_prune_rearm_tokens(compressor: Any, snapshot: dict) -> None:
    """Restore ONLY the prune runway from the attempt snapshot.

    compress() zeroes it in memory while the durable copy only clears on a
    successful commit; a kept transcript keeps its cached prefix, and 0 would let
    the next prune break that cache.
    """
    if "_proactive_prune_rearm_tokens" in snapshot:
        compressor._proactive_prune_rearm_tokens = snapshot["_proactive_prune_rearm_tokens"]


def compression_skipped_due_to_lock(agent: Any) -> bool:
    """Type-pinned read of the per-session lock-skip signal.

    ``agent._compression_skipped_due_to_lock`` is a holder string or ``True`` when
    a pass no-oped because the lock was held, ``None`` otherwise. Pinning avoids
    MagicMock auto-attributes hijacking mocked agents into the lock-skip branch.
    """
    _sig = getattr(agent, "_compression_skipped_due_to_lock", None)
    return _sig is True or isinstance(_sig, str)


def _get_context_compression_timeout_state(
    agent: Any,
    *,
    create: bool,
) -> Optional[Tuple[Any, Optional[threading.local]]]:
    """Return the stable lock and thread-local timeout state for an agent."""
    try:
        attributes = vars(agent)
    except TypeError:
        return None

    lock = attributes.setdefault(
        "_context_compression_timeout_state_lock",
        threading.Lock(),
    )
    with lock:
        state = attributes.get("_context_compression_timeout_state")
        if create and not isinstance(state, threading.local):
            state = threading.local()
            attributes["_context_compression_timeout_state"] = state
        return lock, state if isinstance(state, threading.local) else None


def reset_context_compression_timeout_outcome(agent: Any) -> None:
    """Clear the current thread's owned-compression timeout outcome.

    The ``agent._last_compression_timed_out`` mirror stays authoritative for
    minimal agent doubles that do not support ``vars()``.
    """
    locked_state = _get_context_compression_timeout_state(agent, create=True)
    if locked_state is None or locked_state[1] is None:
        agent._last_compression_timed_out = False
        return
    lock, state = locked_state
    with lock:
        state.timed_out = False
        agent._last_compression_timed_out = False


def mark_context_compression_timed_out(agent: Any) -> None:
    """Mark the current owned compression as host-timed-out."""
    locked_state = _get_context_compression_timeout_state(agent, create=True)
    if locked_state is None or locked_state[1] is None:
        agent._last_compression_timed_out = True
        return
    lock, state = locked_state
    with lock:
        state.timed_out = True
        agent._last_compression_timed_out = True


def context_compression_timed_out(agent: Any) -> bool:
    """Return whether this thread's owned compression hit its host timeout.

    Thread-local so overlapping automatic/manual entrypoints cannot hide each
    other's timeout; attribute fallback for minimal doubles; reads type-pinned.
    """
    locked_state = _get_context_compression_timeout_state(agent, create=False)
    if locked_state is not None:
        lock, state = locked_state
        with lock:
            if isinstance(state, threading.local):
                return getattr(state, "timed_out", None) is True
    return getattr(agent, "_last_compression_timed_out", None) is True


def _automatic_gate_blocked(
    blocked: Any, compressor: Any, bypass_cooldown: bool
) -> bool:
    """Evaluate the automatic breaker gate, optionally ignoring the cooldown.

    Engines whose gate predates ``bypass_cooldown`` are called with the legacy
    no-argument shape.
    """
    if bypass_cooldown:
        try:
            accepts = "ignore_cooldown" in inspect.signature(blocked).parameters
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            return bool(blocked(compressor, ignore_cooldown=True))
    return bool(blocked(compressor))


def compression_blocked_transiently(agent: Any) -> bool:
    """Type-pinned read of the transient-block signal.

    Set when an automatic pass no-ops on a TRANSIENT guard (summary-failure
    cooldown or structural backoff). Consumers must defer, not count it toward
    ``compression_exhausted``, or an overflow auto-reset wipes a session that was
    merely cooling down. The permanent ``ineffective`` breaker never sets it.
    """
    _sig = getattr(agent, "_compression_blocked_transient", None)
    return isinstance(_sig, str) and bool(_sig)


def _mark_compression_blocked_transient(agent: Any, compressor: Any) -> None:
    """Publish the transient-block signal when the active guard is transient.

    Classification comes from ``_compression_block_reason``: ``cooldown:*`` and
    ``structural_backoff:*`` are transient; ``ineffective`` stays unmarked.
    """
    reason_fn = getattr(compressor, "_compression_block_reason", None)
    reason = None
    if callable(reason_fn):
        try:
            reason = reason_fn()
        except Exception:
            logger.debug("compression block-reason read failed", exc_info=True)
    if isinstance(reason, str) and (
        reason.startswith("cooldown") or reason.startswith("structural_backoff")
    ):
        logger.info(
            "Skipping automatic compression re-entry: transient guard "
            "active (%s, session=%s, last failure: %s) — will retry after "
            "the backoff lapses; /compress forces an immediate retry",
            reason,
            getattr(agent, "session_id", None) or "none",
            getattr(compressor, "_last_summary_error", None) or "unknown",
        )
        try:
            agent._compression_blocked_transient = reason
        except Exception:
            pass


def _adopt_live_compression_child(
    agent: Any,
    session_db: Any,
    parent_session_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Move a stale compression contender onto the live continuation tip.

    Resolve and load first, then mutate the agent, so ambiguous lineage or an
    unreadable handoff fails closed. Uses the transitive ``get_compression_tip``
    walk; a tip is adopted only while its row is still live.
    """
    resolver = getattr(type(session_db), "get_compression_tip", None)
    row_getter = getattr(type(session_db), "get_session", None)
    loader = getattr(type(session_db), "get_messages_as_conversation", None)
    if not callable(resolver) or not callable(row_getter) or not callable(loader):
        return None
    tip = resolver(session_db, parent_session_id)
    if not tip or str(tip) == str(parent_session_id):
        return None
    child_session_id = str(tip)
    child = row_getter(session_db, child_session_id)
    if not isinstance(child, dict) or child.get("ended_at") is not None:
        return None
    recovered = loader(session_db, child_session_id)
    if not isinstance(recovered, list) or not recovered:
        return None
    # Revalidate after loading: the tip may have rotated or a competing
    # continuation may have appeared between the two DB reads.
    confirmed = resolver(session_db, parent_session_id)
    if not confirmed or str(confirmed) != child_session_id:
        return None

    agent.session_id = child_session_id
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(child_session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = child_session_id
    try:
        from hermes_logging import set_session_context

        set_session_context(child_session_id)
    except Exception:
        pass

    agent._session_db_created = True
    if child.get("system_prompt"):
        agent._cached_system_prompt = child["system_prompt"]
    agent._last_flushed_db_idx = len(recovered)
    agent._flushed_db_message_session_id = child_session_id
    agent._flushed_db_message_ids = {
        id(message) for message in recovered if isinstance(message, dict)
    }

    on_session_start = getattr(agent.context_compressor, "on_session_start", None)
    if callable(on_session_start):
        try:
            on_session_start(
                child_session_id,
                boundary_reason="compression",
                old_session_id=parent_session_id,
                session_db=session_db,
                platform=getattr(agent, "platform", None) or "cli",
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
        except Exception as exc:
            logger.debug("context engine compression-child adoption failed: %s", exc)
    else:
        bind_state = getattr(agent.context_compressor, "bind_session_state", None)
        if callable(bind_state):
            try:
                bind_state(session_db=session_db, session_id=child_session_id)
            except Exception:
                pass
    try:
        if agent._memory_manager:
            agent._memory_manager.on_session_switch(
                child_session_id,
                parent_session_id=parent_session_id,
                reset=False,
                reason="compression",
            )
    except Exception as exc:
        logger.debug("memory manager compression-child adoption failed: %s", exc)

    return recovered


def recover_rotated_compression_session(
    agent: Any,
) -> Optional[List[Dict[str, Any]]]:
    """Recover a stale live agent before a new turn writes to its old parent."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None) or ""
    if session_db is None or not session_id:
        return None
    try:
        if not _session_was_rotated_by_compression(session_db, session_id):
            return None
        # Rotation holds the parent lease until the child handoff is durable; wait
        # briefly rather than observe the parent-ended/child-empty intermediate state.
        holder_getter = getattr(session_db, "get_compression_lock_holder", None)
        for attempt in range(21):
            recovered = _adopt_live_compression_child(agent, session_db, session_id)
            if recovered is not None:
                return recovered
            holder = holder_getter(session_id) if callable(holder_getter) else None
            if not holder or attempt == 20:
                if not holder:
                    orphan_reopener = getattr(
                        type(session_db),
                        "reopen_orphaned_compression_session",
                        None,
                    )
                    if callable(orphan_reopener):
                        try:
                            if orphan_reopener(session_db, session_id):
                                logger.warning(
                                    "compression recovery: reopened orphaned "
                                    "session=%s with no continuation",
                                    session_id,
                                )
                        except Exception as exc:
                            logger.warning(
                                "orphaned compression session reopen failed "
                                "for %s: %s",
                                session_id,
                                exc,
                            )
                return None
            time.sleep(0.05)
        return None
    except Exception as exc:
        logger.warning(
            "compression session recovery failed for session=%s (%s: %s)",
            session_id,
            type(exc).__name__,
            exc,
        )
        return None


def _compression_lock_holder(agent: Any) -> str:
    """Build a unique lock holder id: ``pid:tid:agent-instance:uuid``.

    pid+tid tell crashed holders apart in diagnostics; instance id and per-acquire
    uuid disambiguate co-resident agents on one thread or pooled compressions.
    """
    import threading
    return (
        f"pid={os.getpid()}"
        f":tid={threading.get_ident()}"
        f":agent={id(agent):x}"
        f":nonce={uuid.uuid4().hex[:8]}"
    )


def _supported_compression_kwargs(
    compress_fn: Any,
    *,
    current_tokens: Optional[int],
    focus_topic: Optional[str],
    force: bool,
    memory_context: str,
    bypass_cooldown: bool = False,
) -> dict:
    """Return only compression kwargs accepted by an engine callable.

    Inspecting first keeps older plugin signatures compatible without catching
    ``TypeError`` and running a stateful compressor twice.
    """
    candidates = {
        "current_tokens": current_tokens,
        "focus_topic": focus_topic,
        "force": force,
    }
    if bypass_cooldown:
        candidates["bypass_cooldown"] = True
    if memory_context:
        candidates["memory_context"] = memory_context
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # current_tokens has always been in the ContextEngine ABC; use the oldest call
        # shape when the callable has no inspectable signature.
        return {"current_tokens": current_tokens}

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


class _CompressionActivityHeartbeat:
    """Refresh the agent inactivity tracker while compression blocks in an aux call."""

    def __init__(
        self,
        agent: Any,
        interval_seconds: float | None = None,
        commit_fence: Optional[CompressionCommitFence] = None,
    ) -> None:
        self._agent = agent
        self._commit_fence = commit_fence
        # Latched once host cancel/timeout wins or a terminal stamp is observed,
        # so a later UNKNOWN rewrite cannot re-arm a detached zombie heartbeat.
        self._suppressed = False
        if interval_seconds is None:
            interval_seconds = getattr(agent, "_compression_activity_heartbeat_interval", 60.0)
        try:
            interval_seconds = float(interval_seconds or 60.0)
        except (TypeError, ValueError):
            interval_seconds = 60.0
        if not math.isfinite(interval_seconds):
            interval_seconds = 60.0
        self._interval_seconds = max(0.1, interval_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-activity-heartbeat",
            daemon=True,
        )

    def start(self) -> "_CompressionActivityHeartbeat":
        # A new compression episode always republishes agent.compression even
        # if a prior timeout/cooldown stamp is still on the agent.
        self._suppressed = False
        self._touch("context compression started", allow_terminal_overwrite=True)
        self._thread.start()
        return self

    def stop(self, desc: str = "context compression completed") -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        # Host timeout already owns the terminal stamp; a detached worker's
        # late stop must not republish agent.compression / "completed".
        if self._should_suppress():
            return
        # Force persist: /compress never hits run_conversation's turn-end clear, so
        # durable labels would stay "in progress" for the 60s persist window.
        self._touch(desc, force_persist=True)

    def _fence_cancelled(self) -> bool:
        fence = self._commit_fence
        return fence is not None and fence.is_cancelled

    def _should_suppress(self) -> bool:
        if self._suppressed:
            return True
        if self._fence_cancelled():
            self._suppressed = True
            return True
        return False

    def _touch(
        self,
        desc: str,
        *,
        allow_terminal_overwrite: bool = False,
        force_persist: bool = False,
    ) -> None:
        try:
            if not allow_terminal_overwrite:
                if self._should_suppress():
                    return
                current = normalize_activity_provenance(
                    getattr(self._agent, "_last_activity_provenance", None)
                )
                if current in _TERMINAL_COMPRESSION_PROVENANCES:
                    self._suppressed = True
                    return
            touch = getattr(self._agent, "_touch_activity", None)
            if callable(touch):
                # Re-check after reading provenance: host may cancel/stamp
                # TIMEOUT between the earlier guard and the write.
                if not allow_terminal_overwrite and self._should_suppress():
                    return
                touch(
                    desc,
                    provenance=ActivityProvenance.AGENT_COMPRESSION,
                    force_persist=force_persist,
                )
        except Exception:
            logger.debug("compression activity heartbeat touch failed", exc_info=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            if self._should_suppress():
                return
            self._touch("context compression in progress")


def _direct_messages_for_pre_compress_memory(messages: Any) -> list[dict[str, Any]]:
    """Return direct user/assistant evidence safe for memory checkpointing.

    Summaries, tool rows and system messages are omitted; assistant prose is kept
    with ``tool_calls`` stripped, and pure tool-call wrappers are dropped.
    """
    # Deferred import: context_compressor → turn_context → this module would form
    # an import cycle.
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

    direct_messages: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = message.get("content")
            has_prose = bool(
                content.strip() if isinstance(content, str) else content
            )
            if not has_prose:
                continue
            message = {k: v for k, v in message.items() if k != "tool_calls"}
        direct_messages.append(message)
    return direct_messages


class _CompressionLockLeaseRefresher:
    def __init__(
        self,
        db: Any,
        session_id: str,
        holder: str,
        ttl_seconds: float,
        refresh_interval_seconds: float | None = None,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = ttl_seconds
        if refresh_interval_seconds is None:
            refresh_interval_seconds = max(1.0, min(60.0, ttl_seconds / 2.0))
        self._refresh_interval_seconds = max(0.1, float(refresh_interval_seconds))
        # Tolerate transient refresh failures for at most one TTL so the lease cannot
        # outlive its TTL; floor 1 so interval >= ttl still tolerates one blip.
        self._max_consecutive_failures = max(
            1, int(self._ttl_seconds / self._refresh_interval_seconds)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-lock-refresh",
            daemon=True,
        )

    def start(self) -> "_CompressionLockLeaseRefresher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # join() timing out mid-UPDATE is safe: daemon thread, and a late refresh on a
        # released lock is a rowcount-0 no-op. stop() does not guarantee quiescence.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # A single falsy refresh (transient DB blip) must not kill the lease; only
        # ttl/interval consecutive failures do, so a stuck refresher never outlives TTL.
        consecutive_failures = 0
        # Refresh immediately: work between try_acquire() and start() is charged to the
        # first lease, so on a short TTL it could expire before tick #1.
        first = True
        while first or not self._stop.wait(self._refresh_interval_seconds):
            if first:
                first = False
                if self._stop.is_set():
                    break
            try:
                refreshed = self._db.refresh_compression_lock(
                    self._session_id,
                    self._holder,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as exc:
                logger.debug("compression lock refresh raised: %s", exc)
                refreshed = False
            if refreshed:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                logger.debug(
                    "compression lock refresh failed %d times in a row; "
                    "stopping lease refresher for session %s",
                    consecutive_failures, self._session_id,
                )
                break


def check_compression_model_feasibility(agent: Any) -> None:
    """Warn at session start if the aux compression context is below the threshold.

    Called from ``AIAgent.__init__`` (CLI sees it via ``_vprint``); the gateway
    wires ``status_callback`` later, so ``replay_compression_warning`` resends it.
    """
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model,
            _try_configured_fallback_for_unavailable_client,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import (
            MINIMUM_CONTEXT_LENGTH,
            get_model_context_length,
        )

        # Provider may be "auto"; fall back to the client's base_url hostname so the
        # user can tell where the compression model is actually called.
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        client, aux_model = get_text_auxiliary_client(
            "compression",
            main_runtime=agent._current_main_runtime(),
        )
        if client is None or not aux_model:
            fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                "compression",
                _aux_cfg_provider,
            )
            if fb_client is not None and fb_model:
                client, aux_model = fb_client, fb_model
                if "(" in fb_label and fb_label.endswith(")"):
                    _aux_cfg_provider = fb_label.rsplit("(", 1)[1][:-1]
        if client is None or not aux_model:
            if _aux_cfg_provider and _aux_cfg_provider != "auto":
                msg = (
                    "⚠ Configured auxiliary compression provider "
                    f"'{_aux_cfg_provider}' is unavailable — context "
                    "compression will drop middle turns without a summary. "
                    "Check auxiliary.compression in config.yaml and "
                    "reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "No auxiliary LLM provider for compression — "
                "summaries will be unavailable."
            )
            return

        aux_base_url = str(getattr(client, "base_url", ""))
        # client.api_key may be a callable (Entra bearer); the resolver only needs a key
        # for live catalogue probes, so pass "" rather than mint a JWT for a lookup.
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")

        aux_context = get_model_context_length(
            aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            # Resolve each model with its own provider so provider-specific paths (Bedrock
            # table, OpenRouter API) hit the correct client, not the main model's.
            provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")),
            custom_providers=agent._custom_providers,
        )

        # Aux model must meet MINIMUM_CONTEXT_LENGTH like the main model, else it cannot
        # summarise a full threshold-sized window.
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Auxiliary compression model {aux_model} has a context "
                f"window of {aux_context:,} tokens, which is below the "
                f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                f"Agent.  Choose a compression model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                f"auxiliary.compression.model in config.yaml), or set "
                f"auxiliary.compression.context_length to override the "
                f"detected value if it is wrong."
            )

        threshold = agent.context_compressor.threshold_tokens
        if aux_context < threshold:
            # Lower the live threshold so compression works this session. The summariser
            # sends one user prompt (no system/tools), so threshold == aux_context is safe.
            old_threshold = threshold
            new_threshold = aux_context
            agent.context_compressor.threshold_tokens = new_threshold
            # tail_token_budget derives from the threshold; keep it in lockstep (as
            # update_model does) or the 1.5x tail ceiling exceeds the trigger and re-fires.
            summary_target_ratio = getattr(
                agent.context_compressor, "summary_target_ratio", None
            )
            if isinstance(summary_target_ratio, (int, float)):
                agent.context_compressor.tail_token_budget = int(
                    new_threshold * summary_target_ratio
                )
            # Keep threshold_percent in sync so update_model re-derives from a sensible
            # value rather than the original too-high one.
            main_ctx = agent.context_compressor.context_length
            if main_ctx:
                agent.context_compressor.threshold_percent = (
                    new_threshold / main_ctx
                )
            safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
            # Mirror the compressor's threshold math (percent floor, output reservation,
            # 64K floor): a suggestion it would override is silently ignored and this
            # warning reappears every session. External engines own policy: keep it plain.
            from agent.context_compressor import ContextCompressor as _CC

            recomputed_threshold = None
            if main_ctx and isinstance(agent.context_compressor, _CC):
                recomputed_threshold = _CC._compute_threshold_tokens(
                    main_ctx,
                    _CC._effective_threshold_percent(main_ctx, safe_pct / 100),
                    getattr(agent.context_compressor, "max_tokens", None),
                )
            threshold_suggestion_viable = (
                recomputed_threshold is None or recomputed_threshold <= aux_context
            )
            # "model (provider)" labels for both sides; empty/"auto" provider falls back to
            # the client's base_url hostname.
            _main_model = getattr(agent, "model", "") or "?"
            _main_provider = getattr(agent, "provider", "") or ""
            _aux_provider_label = (
                _aux_cfg_provider
                if _aux_cfg_provider and _aux_cfg_provider != "auto"
                else ""
            )
            if not _aux_provider_label:
                try:
                    from urllib.parse import urlparse
                    _aux_provider_label = (
                        urlparse(aux_base_url).hostname or aux_base_url
                    )
                except Exception:
                    _aux_provider_label = aux_base_url or "auto"
            _main_label = (
                f"{_main_model} ({_main_provider})"
                if _main_provider
                else _main_model
            )
            _aux_label = f"{aux_model} ({_aux_provider_label})"
            msg = (
                f"⚠ Compression model {_aux_label} context is "
                f"{aux_context:,} tokens, but the main model "
                f"{_main_label}'s compression threshold was "
                f"{old_threshold:,} tokens. "
                f"Auto-lowered this session's threshold to "
                f"{new_threshold:,} tokens so compression can run.\n"
            )
            if threshold_suggestion_viable:
                msg += (
                    f"  To make this permanent, edit config.yaml — either:\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
            else:
                msg += (
                    f"  To make this permanent, use a larger compression "
                    f"model in config.yaml:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  (Lowering compression.threshold cannot help here — "
                    f"with {_main_label}'s {main_ctx:,}-token window, "
                    f"Hermes's small-context floor and output reservation "
                    f"would recompute the trigger to "
                    f"{recomputed_threshold:,} tokens, still above the "
                    f"compression model's {aux_context:,}.)"
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression model %s has %d token context, "
                "below the main model's compression threshold of %d "
                "tokens — auto-lowered session threshold to %d to "
                "keep compression working.",
                aux_model,
                aux_context,
                old_threshold,
                new_threshold,
            )
    except ValueError:
        # Hard rejections (aux below minimum context) must propagate
        # so the session refuses to start.
        raise
    except Exception as exc:
        logger.debug(
            "Compression feasibility check failed (non-fatal): %s", exc
        )


def replay_compression_warning(agent: Any) -> None:
    """Re-send the stored compression warning through ``status_callback``.

    Called once at the start of the first ``run_conversation()``, when the gateway
    callback (absent during ``__init__``) is finally wired.
    """
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        try:
            agent.status_callback("lifecycle", msg)
        except Exception:
            pass


def conversation_history_after_compression(
    agent: Any,
    messages: list,
    previous_history: Optional[list] = None,
) -> Optional[list]:
    """Return the correct flush baseline after a compression boundary.

    Session rotation returns ``None`` so the child gets the full compacted list.
    In-place compaction returns a shallow copy of the already-persisted rows (else
    the identity flush re-appends them). Aborted/no-op attempts keep the baseline:
    marking all persisted drops unflushed turns; clearing re-appends rows.
    """
    if bool(getattr(agent, "_last_compression_attempt_recorded", False)):
        attempt_in_place = getattr(agent, "_last_compression_attempt_in_place", None)
        if attempt_in_place is True:
            return list(messages)
        if attempt_in_place is False:
            return None
        return previous_history
    if bool(getattr(agent, "_last_compaction_in_place", False)):
        return list(messages)
    return None


_SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
)


def _is_real_user_message(message: Any) -> bool:
    """Distinguish human intent from user-role runtime scaffolding.

    A compaction summary flipped to ``role="user"`` for alternation is scaffolding
    and must not short-circuit anchor restoration.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return False
    text = _message_text(message).strip()
    if not text:
        return False
    if text.startswith(_SYNTHETIC_USER_PREFIXES):
        return False
    from agent.context_compressor import ContextCompressor

    return not ContextCompressor._is_synthetic_compression_user_turn(message)


def _message_contains_busy_steer(message: Any) -> bool:
    """Return whether *message* carries a busy-steer marker.

    Steer follow-ups live as markers inside ``role=tool`` results, so they carry
    user intent that ``_is_real_user_message`` alone would miss.
    """
    text = _message_text(message)
    if not text:
        return False
    try:
        from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN

        return STEER_MARKER_OPEN in text and STEER_MARKER_CLOSE in text
    except Exception:
        return "[OUT-OF-BAND USER MESSAGE" in text and "[/OUT-OF-BAND USER MESSAGE]" in text


def _extract_steer_text_from_message(message: Any) -> Optional[str]:
    """Extract the inner user text from a steer marker, or None."""
    text = _message_text(message)
    if not text:
        return None
    try:
        from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN

        open_marker = STEER_MARKER_OPEN
        close_marker = STEER_MARKER_CLOSE
    except Exception:
        open_marker = "[OUT-OF-BAND USER MESSAGE"
        close_marker = "[/OUT-OF-BAND USER MESSAGE]"
    start = text.find(open_marker)
    if start == -1:
        # Fallback: marker wording may evolve; look for the stable prefix.
        fallback_open = "[OUT-OF-BAND USER MESSAGE"
        start = text.find(fallback_open)
        if start == -1:
            return None
        # Skip to end of the opening line.
        nl = text.find("\n", start)
        if nl != -1:
            start = nl + 1
        else:
            start += len(fallback_open)
    else:
        start += len(open_marker)
    end = text.find(close_marker, start)
    if end == -1:
        end = text.find("[/OUT-OF-BAND USER MESSAGE]", start)
        if end == -1:
            return None
    extracted = text[start:end].strip()
    return extracted if extracted else None


def _compressed_has_busy_steer(messages: list) -> bool:
    """Whether *messages* already carries a steer marker in a ``role=tool`` row.

    Only tool rows count, so a summary merely quoting the marker text is not
    mistaken for live intent.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if _message_contains_busy_steer(msg):
            return True
    return False


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.

    Snapshots are appended to the trailing user turn, so a surviving header is
    stale; stripping before re-injection prevents accumulation across boundaries.
    """
    from tools.todo_tool import TODO_INJECTION_HEADER

    if isinstance(content, str):
        idx = content.find(TODO_INJECTION_HEADER)
        if idx == -1:
            return content
        return content[:idx].rstrip()
    if isinstance(content, list):
        cleaned = []
        for part in content:
            if not isinstance(part, dict):
                cleaned.append(part)
                continue
            if part.get("type") == "text":
                text = str(part.get("text") or "")
                idx = text.find(TODO_INJECTION_HEADER)
                if idx != -1:
                    stripped = text[:idx].rstrip()
                    if stripped:
                        p = dict(part)
                        p["text"] = stripped
                        cleaned.append(p)
                else:
                    cleaned.append(part)
            else:
                cleaned.append(part)
        return cleaned
    return content


def _todo_snapshot_is_only_content(content: Any, stripped: Any) -> bool:
    """Return whether stripping the snapshot leaves no structured content.

    Text snapshots trail a string; structured ones occupy their own text part, so
    only an empty remainder proves the row was scaffolding alone. Text extraction
    is deliberately not used: image, audio and other non-text parts must survive.
    """
    if isinstance(content, str) and isinstance(stripped, str):
        return not stripped.strip()
    if isinstance(content, list) and isinstance(stripped, list):
        return not stripped
    return False


def _replace_message_content(message: dict, content: Any) -> None:
    """Rewrite message content without allowing an old API sidecar to replay."""
    from agent.turn_context import drop_stale_api_content

    message["content"] = content
    drop_stale_api_content(message)


# Compaction re-injects the todo list verbatim but prunes skills to markers, so
# couple them: tell the model to reload pruned skills BEFORE acting on tasks.
# Lives after TODO_INJECTION_HEADER so it strips with the snapshot next time.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = (
    "[Skills pruned during compression — reload before acting on these tasks]"
)


def _pruned_skill_reload_notice(compressed: list) -> str:
    """Reload notice for skills whose bodies were pruned, or ``""``.

    Scans ``[SKILL_PRUNED: ...]`` markers in the post-compression transcript;
    first-seen order, deduplicated, capped at ``_MAX_PRUNED_SKILL_MARKERS``.
    """
    from agent.context_compressor import (
        _MAX_PRUNED_SKILL_MARKERS,
        _extract_pruned_skill_names,
    )

    names: list = []
    for message in compressed:
        if not isinstance(message, dict):
            continue
        for name in _extract_pruned_skill_names(_message_text(message)):
            if name not in names:
                names.append(name)
    del names[_MAX_PRUNED_SKILL_MARKERS:]
    if not names:
        return ""
    calls = "; ".join(f"skill_view(name='{name}')" for name in names)
    return (
        f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n"
        "The task list above crossed the compression boundary verbatim, but "
        "the skill instructions that governed it were pruned. Before "
        f"executing any preserved task that depends on these skills, reload "
        f"them first: {calls}. After reloading, re-check that each pending "
        "task is still justified — findings recorded before the boundary may "
        "have invalidated it."
    )


def _merge_anchor_into_user_message(target: dict, anchor: dict) -> None:
    """Fold the human anchor into an existing user-role scaffolding turn.

    Used only when any insertion would create consecutive user turns. Anchor text
    leads, scaffolding follows, and synthetic flags are cleared.
    """
    anchor_content = anchor.get("content")
    target_content = target.get("content")
    if isinstance(anchor_content, list) or isinstance(target_content, list):
        anchor_parts = (
            list(anchor_content)
            if isinstance(anchor_content, list)
            else [{"type": "text", "text": str(anchor_content or "")}]
        )
        target_parts = (
            list(target_content)
            if isinstance(target_content, list)
            else [{"type": "text", "text": str(target_content or "")}]
        )
        _replace_message_content(target, anchor_parts + target_parts)
    else:
        merged = f"{anchor_content or ''}\n\n{target_content or ''}".strip()
        _replace_message_content(target, merged)
    for flag in _SYNTHETIC_USER_FLAGS:
        target.pop(flag, None)


CompressedUserTurnOutcome = Literal[
    "inserted",
    "merged",
    "already_present",
    "placeholder_appended",
]


def _insert_real_user_anchor(messages: list, anchor: dict) -> CompressedUserTurnOutcome:
    """Insert the latest human turn without breaking role alternation."""
    from agent.context_compressor import _DB_PERSISTED_MARKER

    def _role(msg: Any) -> Optional[str]:
        return msg.get("role") if isinstance(msg, dict) else None

    # Preferred anchor: the summary boundary — first assistant message not preceded
    # by a user turn. Left neighbour is then non-user, right is an assistant.
    for index, message in enumerate(messages):
        if _role(message) != "assistant":
            continue
        previous_role = _role(messages[index - 1]) if index > 0 else None
        if previous_role != "user":
            anchor[_DB_PERSISTED_MARKER] = True
            messages.insert(index, anchor)
            return "inserted"
    # Every assistant is user-preceded (or there are none). Appending is
    # safe whenever the transcript does not already end with a user turn.
    if not messages or _role(messages[-1]) != "user":
        anchor[_DB_PERSISTED_MARKER] = True
        messages.append(anchor)
        return "inserted"
    # The transcript ends with a user-role message and no slot avoids
    # user/user adjacency.
    from agent.context_compressor import ContextCompressor

    if ContextCompressor._is_context_summary_content(
        _message_text(messages[-1])
    ):
        # Never merge into a summary: its prefix must stay at message start for summary
        # detection; repair_message_sequence merges adjacent user turns summary-first.
        anchor[_DB_PERSISTED_MARKER] = True
        messages.append(anchor)
        return "inserted"
    # Trailing user-role scaffolding (e.g. the todo snapshot): merge instead
    # of inserting a consecutive same-role message (#55677 strict templates).
    _merge_anchor_into_user_message(messages[-1], anchor)
    messages[-1][_DB_PERSISTED_MARKER] = True
    return "merged"


def _ensure_compressed_has_user_turn(
    original_messages: list, compressed: list
) -> CompressedUserTurnOutcome:
    """Preserve human intent, not merely a synthetic user-role placeholder."""
    if any(_is_real_user_message(message) for message in compressed):
        return "already_present"
    if _compressed_has_busy_steer(compressed):
        return "already_present"
    from agent.context_compressor import (
        COMPRESSION_CONTINUATION_USER_CONTENT,
        _fresh_compaction_message_copy,
    )

    # One reversed scan over BOTH kinds: scanning steer then user would let an older
    # consumed steer outrank a newer real user request and replay it.
    for message in reversed(original_messages):
        if _is_real_user_message(message):
            return _insert_real_user_anchor(
                compressed,
                _fresh_compaction_message_copy(message),
            )
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        steer_text = _extract_steer_text_from_message(message)
        if steer_text:
            return _insert_real_user_anchor(
                compressed,
                {"role": "user", "content": steer_text},
            )
    from agent.message_metadata import append_message

    append_message(
        compressed,
        {
            "role": "user",
            "content": COMPRESSION_CONTINUATION_USER_CONTENT,
        },
    )
    return "placeholder_appended"


def _messages_match_scoped_identity(left: Any, right: Any) -> bool:
    """Compare the live turn identity we care about for rotation stamping."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.get("role") != right.get("role"):
        return False
    if left.get("content") != right.get("content"):
        return False
    left_timestamp = left.get("timestamp")
    right_timestamp = right.get("timestamp")
    if left_timestamp is not None and right_timestamp is not None:
        return left_timestamp == right_timestamp
    return True


_PENDING_CONTEXT_ENGINE_NOTIFICATION = (
    "_pending_context_engine_compression_notification"
)


def _notify_context_engine_compression_complete(
    agent: Any,
    *,
    new_session_id: str,
    old_session_id: str,
) -> bool:
    """Notify the active context engine after a durable compression commit."""
    # Opt-in relay session-span segmentation. Observer semantics — failure must
    # never undo or delay the committed compression.
    try:
        from agent import relay_runtime

        relay_runtime.SESSION_COORDINATOR.notify_session_compacted(
            profile_key=relay_runtime.current_profile_key(),
            session_id=new_session_id,
            old_session_id=old_session_id,
        )
    except Exception:
        logger.debug("relay segment rotation notification failed", exc_info=True)
    callback = getattr(agent.context_compressor, "on_session_start", None)
    if not callable(callback):
        return False
    try:
        callback(
            new_session_id,
            boundary_reason="compression",
            old_session_id=old_session_id,
            platform=getattr(agent, "platform", None) or "cli",
            conversation_id=getattr(agent, "_gateway_session_key", None),
        )
    except Exception:
        # Context-engine hooks are observers. A callback failure must not undo
        # history that the core or an outer host transaction already committed.
        logger.debug(
            "context engine on_session_start (compression) failed",
            exc_info=True,
        )
        return False
    return True


def _queue_context_engine_compression_notification(
    agent: Any,
    *,
    new_session_id: str,
    old_session_id: str,
) -> None:
    """Stage exactly one existing hook call for an outer host transaction."""
    if callable(getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)):
        raise RuntimeError("a compression notification is already pending")

    def _notify() -> bool:
        return _notify_context_engine_compression_complete(
            agent,
            new_session_id=new_session_id,
            old_session_id=old_session_id,
        )

    setattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, _notify)


def finalize_context_engine_compression_notification(
    agent: Any,
    *,
    committed: bool,
) -> bool:
    """Emit or discard a deferred notification; repeated calls are no-ops."""
    pending = getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)
    setattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)
    if not committed or not callable(pending):
        return False
    return bool(pending())


class _CompactionLifecycle:
    """Owns the one-shot terminal edge of the compaction status lifecycle.

    ``commit_status`` is rebound to "committed" only on success and read at
    ``complete()`` time, so abort paths keep the terminal edge suppressed.
    """

    def __init__(self, agent: Any, status_emitted: bool) -> None:
        self._agent = agent
        self._status_emitted = status_emitted
        self._done_emitted = False
        self.commit_status = "aborted"

    def complete(self, *, force_terminal: bool = False) -> None:
        if self._done_emitted:
            return
        self._done_emitted = True
        # Suppressed start → no terminal edge. Non-compacting aborts (lock contender,
        # cancelled fence) opt in via force_terminal so clients can retire their phase.
        # Failure warnings go through _emit_warning and are never suppressed here.
        if self._status_emitted and (
            self.commit_status == "committed" or force_terminal
        ):
            _emit_compaction_done(self._agent)


class _CompressionLease:
    """The per-attempt durable compression lock plus its lifecycle plumbing.

    ``holder`` is None when no durable lock is owned (legacy DB, no session db);
    ``watermark`` is MAX(id) of active rows at lease start (None = archive
    everything, no concurrent-tail preservation this cycle).
    """

    def __init__(
        self,
        agent: Any,
        *,
        db: Any,
        sid: str,
        ttl: float,
        refresh_interval: Any,
        commit_fence: Optional[CompressionCommitFence],
        lifecycle: _CompactionLifecycle,
    ) -> None:
        self._agent = agent
        self.db = db
        self.sid = sid
        self.ttl = ttl
        self._refresh_interval = refresh_interval
        self._commit_fence = commit_fence
        self._lifecycle = lifecycle
        self.holder: Optional[str] = None
        self.watermark: Optional[int] = None
        self._refresher: Optional[_CompressionLockLeaseRefresher] = None
        self._released = False
        self._release_guard = threading.Lock()
        # Fence lock acquisition + release-hook publication together so a host timeout
        # cannot win between acquiring the lock and having a way to release it.
        self._lock_setup_entered = False

    def begin_lock_setup(self) -> bool:
        if self._commit_fence is None:
            return True
        self._lock_setup_entered = self._commit_fence.begin_lock_setup()
        return self._lock_setup_entered

    def finish_lock_setup(self) -> None:
        if not self._lock_setup_entered or self._commit_fence is None:
            return
        self._lock_setup_entered = False
        self._commit_fence.finish_lock_setup()

    def start_refresher(self) -> None:
        if self.holder is None:
            return
        candidate = _CompressionLockLeaseRefresher(
            self.db, self.sid, self.holder, self.ttl, self._refresh_interval
        )
        # Cancellation may release the holder between hook publication and this
        # start; serialize with the release path so no refresher starts on a freed lock.
        with self._release_guard:
            if not self._released:
                self._refresher = candidate
                self._refresher.start()

    def release_holder_only(self) -> None:
        """Stop this holder's refresher and release only its durable lock.

        Holder-qualified and idempotent: safe for the host after a timeout because a
        newer holder's lease can never be deleted by this stale release.
        """
        with self._release_guard:
            if self._released:
                return
            self._released = True
            if getattr(self._agent, "_active_compression_lock_holder", None) == self.holder:
                self._agent._active_compression_lock_holder = None
            if self._refresher is not None:
                try:
                    self._refresher.stop()
                except Exception as _stop_err:
                    logger.debug("compression lock refresher stop failed: %s", _stop_err)
            if self.db is not None and self.sid and self.holder:
                try:
                    self.db.release_compression_lock(self.sid, self.holder)
                except Exception as _rel_err:
                    logger.debug("compression lock release failed: %s", _rel_err)

    def release(self) -> None:
        """Finish lifecycle cleanup and release the OLD session lock once."""
        try:
            self._lifecycle.complete()
        finally:
            try:
                self.release_holder_only()
            finally:
                try:
                    if self._commit_fence is not None:
                        self._commit_fence.clear_cancelled_lock_release(
                            self.release_holder_only
                        )
                finally:
                    self.finish_lock_setup()


def _acquire_compression_lease(
    agent: Any,
    *,
    commit_fence: Optional[CompressionCommitFence],
    lifecycle: _CompactionLifecycle,
    system_message: str,
    approx_tokens: Optional[int],
    attempt_started_at: float,
) -> Tuple[Optional[_CompressionLease], Optional[str]]:
    """Take the per-session compression lock; ``(None, prompt)`` means sit out.

    Two AIAgents sharing a session_id (e.g. background review fork) would both
    rotate and orphan a child. Keyed on the OLD id (what rivals read from
    SessionEntry). Loser sits out: messages unchanged, caller sees no-op.
    Only structural absence of the lock API (version skew) fails open; once
    resolved, any exception fails closed since unlocked runs can fork lineage.
    """
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    _try_acquire_lock = None
    _lock_lookup_error: Optional[Exception] = None
    _legacy_session_db_without_lock_api = False
    # Clear stale lock-skip so this call's outcome alone is visible; else a manual
    # /compress after an auto lock-skip falsely reports "already in progress".
    agent._compression_skipped_due_to_lock = None
    if _lock_db is not None:
        try:
            _legacy_session_db_without_lock_api = _lock_api_is_absent_on_session_db(
                _lock_db
            )
        except Exception as exc:
            _lock_lookup_error = exc
        if _lock_lookup_error is None and not _legacy_session_db_without_lock_api:
            try:
                _try_acquire_lock = _lock_db.try_acquire_compression_lock
                if not callable(_try_acquire_lock):
                    _lock_lookup_error = TypeError(
                        "compression lock API is present but not callable"
                    )
            except Exception as exc:
                _lock_lookup_error = exc
    try:
        _lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)
    except (TypeError, ValueError):
        _lock_ttl = 300.0
    lease = _CompressionLease(
        agent,
        db=_lock_db,
        sid=_lock_sid,
        ttl=_lock_ttl,
        refresh_interval=getattr(agent, "_compression_lock_refresh_interval", None),
        commit_fence=commit_fence,
        lifecycle=lifecycle,
    )

    if _lock_db is not None and _lock_sid:
        lease.holder = _compression_lock_holder(agent)
        if _lock_lookup_error is not None:
            # Attribute lookup itself failed for a reason other than a missing
            # lock API. It is unsafe to proceed without a lock in that case.
            lease.holder = None
            logger.warning(
                "compression lock lookup raised unexpectedly for session=%s "
                "(%s: %s) — skipping compression this cycle",
                _lock_sid, type(_lock_lookup_error).__name__, _lock_lookup_error,
            )
            _lock_acquired = False
        elif _try_acquire_lock is None:
            # Lock API absent on this instance: log once, proceed unlocked so version skew
            # cannot stall the outer auto-compression loop forever.
            lease.holder = None
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s "
                    "— proceeding without lock. This usually means a stale "
                    "in-memory module after an update; restart the process "
                    "(or `hermes update`) to resync.",
                    _lock_sid,
                )
            _lock_acquired = True  # acquired-but-unlocked compatibility path
        else:
            if not lease.begin_lock_setup():
                logger.info(
                    "Compression commit cancelled before lock acquisition "
                    "(session=%s).",
                    agent.session_id or "none",
                )
                agent._last_compaction_in_place = False
                _existing_sp = _existing_system_prompt(agent, system_message)
                _emit_aborted_attempt_telemetry(agent, attempt_started_at, "commit_fence_cancelled")
                lifecycle.complete(force_terminal=True)
                return None, _existing_sp
            try:
                _lock_acquired = _try_acquire_lock(
                    _lock_sid, lease.holder, ttl_seconds=_lock_ttl
                )
                if _lock_acquired:
                    # Watermark = MAX(id) of active rows at START. Appends aren't blocked during
                    # summary; later rows are concurrent tail that archive_and_compact re-sequences.
                    try:
                        lease.watermark = _lock_db.get_active_message_watermark(
                            _lock_sid
                        )
                        # A captured watermark makes the commit safe against later rows on BOTH commit
                        # paths; tell the fence so a host may keep this attempt's admission.
                        if commit_fence is not None:
                            try:
                                commit_fence.mark_commit_watermark_fenced()
                            except AttributeError:
                                pass  # test doubles without the method
                    except Exception as _wm_err:
                        # Watermark capture is safety-additive (fallback archives everything), so
                        # failure here must not abort compression.
                        logger.warning(
                            "compression watermark capture failed for "
                            "session=%s (%s) — concurrent appends this cycle "
                            "will be archived with the snapshot",
                            _lock_sid, _wm_err,
                        )
                        lease.watermark = None
            except Exception as _lock_err:
                # Method entered but failed: not version skew, fail closed. Acquire may have
                # committed, so release holder-qualified best-effort (safe if never acquired).
                try:
                    _lock_db.release_compression_lock(_lock_sid, lease.holder)
                except Exception as _release_err:
                    logger.debug(
                        "compression lock cleanup after failed acquire failed: %s",
                        _release_err,
                    )
                lease.holder = None
                logger.warning(
                    "compression lock acquisition raised unexpectedly for "
                    "session=%s (%s: %s) — skipping compression this cycle",
                    _lock_sid, type(_lock_err).__name__, _lock_err,
                )
                _lock_acquired = False
        if not _lock_acquired:
            lease.finish_lock_setup()
            try:
                existing = _lock_db.get_compression_lock_holder(_lock_sid)
            except Exception:
                existing = None
            logger.warning(
                "compression skipped: another path is compressing session=%s "
                "(holder=%s) — returning messages unchanged to avoid session fork",
                _lock_sid, existing,
            )
            lease.holder = None  # don't release a lock we don't own
            # Distinguish lock-contention no-op from "nothing to compress" so manual
            # /compress can show a clear status instead of "No changes".
            agent._compression_skipped_due_to_lock = existing or True
            # Surface to the user once — quiet for downstream auto-compress loops
            if getattr(agent, "_last_compression_lock_warning_sid", None) != _lock_sid:
                agent._last_compression_lock_warning_sid = _lock_sid
                try:
                    agent._emit_warning(
                        "⚠ Skipping concurrent compression — another path "
                        "is already compressing this session. Will retry "
                        "after it finishes."
                    )
                except Exception:
                    pass
            _existing_sp = _existing_system_prompt(agent, system_message)
            try:
                if hasattr(agent.context_compressor, "_begin_compression_telemetry"):
                    agent.context_compressor._begin_compression_telemetry(current_tokens=approx_tokens)
            except Exception:
                pass
            _emit_aborted_attempt_telemetry(agent, attempt_started_at, "lock_contended")
            lifecycle.complete(force_terminal=True)
            return None, _existing_sp

    if lease.holder is not None:
        agent._active_compression_lock_holder = lease.holder
        if (
            commit_fence is not None
            and commit_fence.register_cancelled_lock_release(
                lease.release_holder_only
            )
        ):
            # Cancellation won during lock setup (hook ran synchronously, lease gone):
            # abort before any summary work.
            logger.info(
                "Compression commit cancelled before summary dispatch "
                "(session=%s).",
                agent.session_id or "none",
            )
            agent._last_compaction_in_place = False
            _existing_sp = _existing_system_prompt(agent, system_message)
            _emit_aborted_attempt_telemetry(agent, attempt_started_at, "commit_fence_cancelled")
            lease.release()
            return None, _existing_sp
    return lease, None


def _adopt_if_parent_rotated(
    agent: Any, lease: _CompressionLease, messages: list, system_message: str
) -> Optional[Tuple[list, str]]:
    """Sit out (or adopt the live child) when the parent was already rotated.

    A late contender can take the parent lock after the winner released it and
    rotated; holding the lock does not prove this agent still owns a live parent.
    Returns the ``compress_context`` result to hand back, or None to proceed.
    """
    if lease.db is None or not lease.sid:
        return None
    try:
        _parent_already_rotated = _session_was_rotated_by_compression(
            lease.db, lease.sid
        )
    except Exception as _session_err:
        logger.warning(
            "compression session ownership lookup failed for session=%s "
            "(%s: %s) - skipping compression this cycle",
            lease.sid,
            type(_session_err).__name__,
            _session_err,
        )
        lease.release()
        return messages, _existing_system_prompt(agent, system_message)
    if not _parent_already_rotated:
        return None
    recovered_messages = _adopt_live_compression_child(agent, lease.db, lease.sid)
    lease.release()
    _existing_sp = _existing_system_prompt(agent, system_message)
    if recovered_messages is not None:
        logger.warning(
            "compression recovery: stale session=%s adopted live child=%s",
            lease.sid,
            agent.session_id,
        )
        return recovered_messages, _existing_sp
    logger.warning(
        "compression skipped: session=%s was already rotated by "
        "another compression path, but no unique live child could be adopted",
        lease.sid,
    )
    return messages, _existing_sp


def _adopt_grown_durable_parent(
    agent: Any, lease: _CompressionLease, messages: list
) -> Optional[list]:
    """Return the durable parent transcript when it outgrew the in-memory snapshot.

    Rotation only (in-place never loses rows). The snapshot predates the lease: if
    durable grew, a writer committed a turn — ADOPT it (aborting wedged busy
    sessions forever). Length check only: in-memory edits of past turns are legal.
    """
    if lease.db is None or not lease.sid:
        return None
    durable_loader = getattr(type(lease.db), "get_messages_as_conversation", None)
    if not callable(durable_loader):
        return None
    durable_parent = durable_loader(lease.db, lease.sid)
    if not (isinstance(durable_parent, list) and len(durable_parent) > len(messages)):
        return None
    # In-memory carries this turn's un-persisted user tail; flush it via the normal
    # rotation-boundary path before adopting, else skip adoption (would drop input).
    _preflush_idx = getattr(agent, "_persist_user_message_idx", None)
    _preflush_ok = False
    if isinstance(_preflush_idx, int) and 0 <= _preflush_idx < len(messages):
        try:
            _preflush_ok = agent._flush_messages_to_session_db(
                messages,
                conversation_history=messages[:_preflush_idx],
            )
        except Exception:
            _preflush_ok = False
    else:
        # No un-persisted tail: transcript is fully durable, so adopting the longer
        # parent cannot drop live input — adopt directly.
        _preflush_ok = True
    if not _preflush_ok:
        logger.warning(
            "compression: session=%s grew before lease "
            "(%d → %d msgs) but the pre-adoption flush of the "
            "live tail failed; skipping durable-snapshot "
            "adoption so un-persisted user input is kept",
            lease.sid,
            len(messages),
            len(durable_parent),
        )
        return None
    # Re-read after the flush so the adopted snapshot carries the just-persisted tail.
    durable_parent = durable_loader(lease.db, lease.sid)
    if not (isinstance(durable_parent, list) and len(durable_parent) > len(messages)):
        return None
    logger.info(
        "compression: session=%s grew before lease "
        "(%d → %d msgs); adopting durable snapshot",
        lease.sid,
        len(messages),
        len(durable_parent),
    )
    return durable_parent


def _pre_compress_memory_context(
    agent: Any, messages: list, checkpoint_required: bool
) -> str:
    """Provider ``on_pre_compress()`` insights to surface in the summary ("" if none).

    Raw messages stay the API v1 provider contract; normalized evidence goes only
    to API v2+ checkpoint providers inside MemoryManager.on_pre_compress().
    Raises :class:`CompressionCheckpointUnavailable` when a required checkpoint
    cannot be taken.
    """
    memory_context = ""
    memory_manager = getattr(agent, "_memory_manager", None)
    evidence_messages = _direct_messages_for_pre_compress_memory(messages)
    if checkpoint_required:
        supports_checkpoint = getattr(
            memory_manager, "supports_pre_compress_checkpoint", None
        )
        if memory_manager is None or not callable(supports_checkpoint):
            raise _checkpoint_blocked(
                f"no active provider implements checkpoint API "
                f"v{PRE_COMPRESS_CHECKPOINT_API_VERSION}"
            )
        try:
            compatible = bool(
                supports_checkpoint(PRE_COMPRESS_CHECKPOINT_API_VERSION)
            )
        except Exception as exc:
            raise _checkpoint_blocked("provider capability probe failed") from exc
        if not compatible:
            raise _checkpoint_blocked(
                f"active provider does not implement checkpoint API "
                f"v{PRE_COMPRESS_CHECKPOINT_API_VERSION}"
            )
        try:
            _maybe_ctx = memory_manager.on_pre_compress(
                messages,
                evidence_messages=evidence_messages,
                require_checkpoint=True,
                checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
            )
        except Exception as exc:
            logger.warning(
                "Required pre-compress checkpoint failed (%s)",
                type(exc).__name__,
            )
            raise _checkpoint_blocked(
                f"provider checkpoint API v{PRE_COMPRESS_CHECKPOINT_API_VERSION} failed"
            ) from exc
        if isinstance(_maybe_ctx, str):
            memory_context = sanitize_memory_context(_maybe_ctx)
    elif memory_manager:
        try:
            _maybe_ctx = memory_manager.on_pre_compress(
                messages, evidence_messages=evidence_messages
            )
            if isinstance(_maybe_ctx, str):
                memory_context = sanitize_memory_context(_maybe_ctx)
        except Exception:
            pass
    return memory_context


def _resolve_compress_call(
    agent: Any,
    *,
    approx_tokens: Optional[int],
    focus_topic: Optional[str],
    force: bool,
    memory_context: str,
    bypass_cooldown: bool,
) -> Tuple[Callable[..., Any], dict[str, Any]]:
    """Bind ``compress()`` and only the kwargs its signature accepts."""
    compress_fn = agent.context_compressor.compress
    compress_kwargs = _supported_compression_kwargs(
        compress_fn,
        current_tokens=approx_tokens,
        focus_topic=focus_topic,
        force=force,
        memory_context=memory_context,
        bypass_cooldown=bypass_cooldown,
    )
    if memory_context.strip() and "memory_context" not in compress_kwargs:
        engine_name = getattr(
            agent.context_compressor,
            "name",
            type(agent.context_compressor).__name__,
        )
        if (
            getattr(agent, "_last_memory_context_unsupported_engine", None)
            != engine_name
        ):
            agent._last_memory_context_unsupported_engine = engine_name
            logger.warning(
                "context engine %s does not accept memory_context; continuing "
                "without provider-supplied summary context",
                engine_name,
            )
    return compress_fn, compress_kwargs


def _run_summary_dispatch(
    agent: Any,
    messages: list,
    compress_fn: Callable[..., Any],
    compress_kwargs: dict[str, Any],
    *,
    commit_fence: Optional[CompressionCommitFence],
    attempt_generation: Any,
    hard_cancel_event: Any,
) -> list:
    """Run the compressor under the fence's progress hook, deadline and interrupt guard."""
    # Publish progress to the commit fence so hosts extend deadlines while tokens
    # flow. Any active hook (even no-op) selects the streamed path: the timeout is
    # inactivity-based and a byte-trickling provider hits the stream total ceiling.
    from agent.auxiliary_client import (
        aux_interrupt_protection,
        aux_progress_hook,
        aux_stream_deadline,
    )
    _progress_hook = (
        commit_fence.touch_progress if commit_fence is not None
        else (lambda: None)
    )
    # Return leg: cancel frees the owner but the provider daemon streams on to its
    # own larger ceiling; share the host deadline so orphan streams stop with it.
    _host_stream_deadline = (
        commit_fence.deadline_monotonic if commit_fence is not None else None
    )
    # A LATE successful summary must not undo the host's timeout cooldown: the
    # compressor checks cancellation before clearing; removed in finally (no leak).
    if commit_fence is not None:
        _install_compression_cancelled_check(
            agent.context_compressor,
            lambda: commit_fence.is_cancelled,
            attempt_generation,
        )

    def _compression_cancel_requested() -> bool:
        return bool(
            (
                hard_cancel_event is not None
                and hard_cancel_event.is_set()
            )
            or (
                commit_fence is not None
                and commit_fence.is_cancelled
            )
        )

    try:
        # F6: never start expensive summary work for an already-cancelled
        # fence (a stale queued job admitted after host departure).
        if commit_fence is not None and commit_fence.is_cancelled:
            logger.info(
                "Compression cancelled before summary dispatch "
                "(session=%s) — skipping summary work.",
                agent.session_id or "none",
            )
            compressed = messages
        else:
            with aux_progress_hook(_progress_hook), aux_stream_deadline(
                _host_stream_deadline
            ), aux_interrupt_protection(
                cancel_check=_compression_cancel_requested
            ):
                compressed = compress_fn(messages, **compress_kwargs)
                # Freeze a hard stop that arrived after the last provider attempt but before
                # session state rotates.
                if (
                    hard_cancel_event is not None
                    and hard_cancel_event.is_set()
                ):
                    raise AuxiliaryExplicitCancellation()
    finally:
        if commit_fence is not None:
            _clear_compression_cancelled_check_if_owner(
                agent.context_compressor, attempt_generation
            )
    return compressed


def _fold_todo_snapshot(agent: Any, compressed: list) -> None:
    """Strip stale todo snapshots from ``compressed`` and fold the live one in (in place)."""
    todo_snapshot = agent._todo_store.format_for_injection()
    # Non-empty store (even all done) is authoritative: drop the old snapshot. A
    # truly empty store may be un-rehydrated post-compaction: keep the snapshot.
    _todo_has_items = getattr(agent._todo_store, "has_items", None)
    try:
        _todo_store_is_authoritative = bool(
            _todo_has_items()
        ) if callable(_todo_has_items) else False
    except Exception:
        # Store may implement only format_for_injection(); unknown authority must
        # preserve the pending snapshot rather than risk deleting it.
        _todo_store_is_authoritative = False
    if _todo_store_is_authoritative:
        for _todo_idx in range(len(compressed) - 1, -1, -1):
            _todo_message = compressed[_todo_idx]
            if not isinstance(_todo_message, dict) or _todo_message.get("role") != "user":
                continue
            _todo_content = _todo_message.get("content")
            _todo_stripped = _strip_stale_todo_snapshot(_todo_content)
            if _todo_stripped == _todo_content:
                continue
            if (
                _todo_message.get("_todo_snapshot_synthetic")
                and _todo_snapshot_is_only_content(
                    _todo_content, _todo_stripped
                )
            ):
                compressed.pop(_todo_idx)
                if _todo_idx < len(compressed):
                    # A standalone snapshot can drift from the tail; deleting it may expose two
                    # assistant rows, so use the normal replay repair to keep metadata consistent.
                    agent._repair_message_sequence(compressed)
            else:
                _replace_message_content(_todo_message, _todo_stripped)
                # No longer todo-only scaffolding; other synthetic flags stay authoritative and
                # _is_real_user_message() recomputes provenance from content + flags.
                _todo_message.pop("_todo_snapshot_synthetic", None)
            break
    if todo_snapshot:
        # If this boundary pruned skill bodies, the policy behind the todos is gone:
        # add a reload notice after TODO_INJECTION_HEADER so both strip together.
        _reload_notice = _pruned_skill_reload_notice(compressed)
        if _reload_notice:
            todo_snapshot = f"{todo_snapshot}\n\n{_reload_notice}"
        # Fold the snapshot into a trailing REAL user msg (no synthetic user/user pair);
        # strip old snapshots first. Scaffolding tails must not absorb it (provenance).
        from agent.context_compressor import _append_text_to_content

        merged = False
        _tail = (
            compressed[-1]
            if compressed and isinstance(compressed[-1], dict)
            else None
        )
        if _tail is not None and _tail.get("role") == "user":
            _stripped = _strip_stale_todo_snapshot(_tail.get("content"))
            _probe = {
                key: value for key, value in _tail.items() if key != "content"
            }
            _probe["content"] = _stripped
            if _is_real_user_message(_probe):
                _snapshot_text = (
                    f"\n\n{todo_snapshot}"
                    if isinstance(_stripped, str) and _stripped
                    else todo_snapshot
                )
                _replace_message_content(
                    _tail,
                    _append_text_to_content(_stripped, _snapshot_text),
                )
                merged = True
            elif _stripped != _tail.get("content") and not _message_text(
                {"role": "user", "content": _stripped}
            ).strip():
                # The tail was nothing but an earlier snapshot row —
                # refresh it in place instead of stacking a duplicate.
                _replace_message_content(_tail, todo_snapshot)
                _tail["_todo_snapshot_synthetic"] = True
                merged = True
        if not merged:
            compressed.append({
                "role": "user",
                "content": todo_snapshot,
                "_todo_snapshot_synthetic": True,
            })


def _rebuild_system_prompt_at_boundary(agent: Any, system_message: str) -> str:
    """Refresh tool schemas and rebuild the system prompt at the commit boundary."""
    cached_system_prompt = agent._cached_system_prompt
    agent._invalidate_system_prompt()

    # Refresh tool schemas at the commit boundary: forever-sessions never restart,
    # so config reaches agent.tools here. Keep list identity if byte-equal (cache).
    try:
        _refresh_agent_tool_definitions(agent)
    except Exception:  # noqa: BLE001
        logger.warning(
            "compaction tool-definition refresh failed; keeping the "
            "session's existing tool snapshot",
            exc_info=True,
        )

    # ALWAYS rebuild the prompt here: keeping old bytes meant prompt-builder changes
    # never reached long sessions. Equal bytes keep KV; preserve object identity.
    rebuilt_system_prompt = agent._build_system_prompt(system_message)
    if cached_system_prompt is not None and rebuilt_system_prompt == cached_system_prompt:
        new_system_prompt = cached_system_prompt
        agent._cached_system_prompt = cached_system_prompt
        from agent.system_prompt import reconstruct_static_prefix

        reconstruct_static_prefix(
            agent,
            system_message=system_message,
            log_label="compression keep-prompt",
        )
    else:
        new_system_prompt = rebuilt_system_prompt
        agent._cached_system_prompt = new_system_prompt
        if cached_system_prompt is not None:
            logger.info(
                "Compaction rebuilt a drifted system prompt "
                "(session=%s, %d -> %d chars): builder output changed "
                "since the stored snapshot (update, config change, or "
                "memory/skills growth)",
                agent.session_id or "none",
                len(cached_system_prompt),
                len(new_system_prompt),
            )
    return new_system_prompt


def _salvage_or_refuse_grown_transcript(
    agent: Any,
    messages: list,
    compressed: list,
    *,
    system_message: str,
    attempt_started_at: float,
    attempt_snapshot: dict,
) -> Tuple[Optional[list], Optional[str]]:
    """Anti-growth guard at the COMMIT SITE (in-place commits before the gateway can inspect).

    Compares like-for-like rough estimates; on growth tries one mechanical salvage
    pass, else treats the attempt as a refused no-op. Returns ``(compressed, None)``
    to proceed or ``(None, prompt)`` when refused (caller releases the lease).
    """
    # Anti-growth guard at the COMMIT SITE: in-place commits here before the gateway
    # can inspect. Compare like-for-like rough estimates; on growth treat as no-op.
    _rough_in = estimate_messages_tokens_rough(messages)
    _rough_out = estimate_messages_tokens_rough(compressed)
    if _rough_out > _rough_in:
        # Todo refresh and user-turn anchoring run after the compressor's own size check
        # and can tip a break-even candidate; give it one mechanical salvage pass.
        from agent.context_compressor import salvage_grown_transcript

        _salvaged = salvage_grown_transcript(
            messages, compressed, budget=_rough_in
        )
        if _salvaged is not None:
            _salv_est = estimate_messages_tokens_rough(_salvaged)
            if _salv_est < _rough_in:
                logger.info(
                    "Compression salvage recovered a shrinking "
                    "transcript (session=%s, ~%s -> ~%s tokens)",
                    agent.session_id or "none",
                    f"{_rough_in:,}",
                    f"{_salv_est:,}",
                )
                compressed = _salvaged
                _rough_out = _salv_est
    if _rough_out > _rough_in:
        logger.warning(
            "Compression refused: compressed transcript would be "
            "larger than the original (session=%s, ~%s -> ~%s "
            "tokens); keeping the original transcript unchanged",
            agent.session_id or "none",
            f"{_rough_in:,}",
            f"{_rough_out:,}",
        )
        # Flag the refusal on compressor state so /compress feedback reports it instead
        # of comparing list lengths (adoption can change the count), claiming success.
        try:
            agent.context_compressor._last_compress_refused_would_grow = True
        except Exception:
            pass
        try:
            agent._emit_warning(
                "⚠️ Compression refused: the generated summary "
                "would have GROWN the conversation instead of "
                "shrinking it. No messages were dropped — "
                "conversation continues unchanged."
            )
        except Exception:
            pass
        _existing_sp = _existing_system_prompt(agent, system_message)
        _emit_aborted_attempt_telemetry(agent, attempt_started_at, "would_grow")
        # Count the refusal as an ineffective-compaction strike so the anti-thrash
        # breaker latches; otherwise auto-compress retries the same summary every turn.
        try:
            agent.context_compressor.record_rejected_compaction()
        except Exception:
            logger.debug(
                "could not record rejected-compaction strike",
                exc_info=True,
            )
        _restore_prune_rearm_tokens(agent.context_compressor, attempt_snapshot)
        return None, _existing_sp
    return compressed, None


def _publish_rotated_compaction(
    agent: Any,
    messages: list,
    compressed: list,
    *,
    new_system_prompt: str,
    lease: _CompressionLease,
    old_session_id: str,
    compressed_user_turn_outcome: str,
) -> None:
    """Rotate the session: flush the parent, publish the child, re-point the agent.

    Flushes current-turn msgs to the OLD session, passing the durable prefix
    (messages[:persist idx]) so preflight, which runs before rows are
    marker-stamped, can't re-append them.
    """
    current_idx = getattr(agent, "_persist_user_message_idx", None)
    persisted_history = (
        messages[:current_idx]
        if isinstance(current_idx, int)
        and 0 <= current_idx <= len(messages)
        else None
    )
    # The flush is durable and NOT rolled back on abort: a deliberately-ended parent
    # fails publish forever, so check that before writing. Automatic end stamps are
    # healed by publish (don't abort); the lease is re-acquirable (don't check it).
    _parent_row_reader = getattr(agent._session_db, "get_session", None)
    _parent_already_ended = False
    if callable(_parent_row_reader):
        try:
            from hermes_state_common import is_automatic_end_reason

            _parent_row = _parent_row_reader(old_session_id) or {}
            _parent_already_ended = (
                _parent_row.get("ended_at") is not None
                and not is_automatic_end_reason(
                    _parent_row.get("end_reason")
                )
            )
        except Exception:
            # Fail OPEN: an unreadable row must not turn a cheap
            # guard into a new way to lose compression.
            _parent_already_ended = False
    if _parent_already_ended:
        raise RuntimeError(
            f"Compression parent already ended: {old_session_id}"
        )
    # Foreign-tail ceiling: the flush below writes OUR rows (already in handoff);
    # rows above the start watermark up to this MAX(id) are foreign appends.
    try:
        _foreign_tail_ceiling = (
            agent._session_db.get_active_message_watermark(
                agent.session_id
            )
        )
    except Exception:
        # No trustworthy ceiling: the clone could duplicate the handoff, so skip tail
        # preservation this rotation.
        _foreign_tail_ceiling = None
    try:
        agent._flush_messages_to_session_db(
            messages,
            conversation_history=persisted_history,
        )
    except Exception:
        pass  # best-effort — don't block compression on a flush error
    # Publish closure + child + handoff in one transaction so no reader sees an
    # empty child. Child stays on the parent's profile ("default" persists as NULL);
    # publish also COALESCEs from the parent row for threads lacking HERMES_HOME.
    try:
        from hermes_cli.profiles import get_active_profile_name

        _profile_for_child = get_active_profile_name()
        if _profile_for_child == "default":
            _profile_for_child = None
    except Exception:
        _profile_for_child = None
    old_title = agent._session_db.get_session_title(agent.session_id)
    new_session_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    from agent.context_compressor import _DB_PERSISTED_MARKER
    agent._session_db.publish_compression_child(
        parent_session_id=old_session_id,
        child_session_id=new_session_id,
        source=agent.platform
        or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        model=agent.model,
        model_config=agent._session_init_model_config,
        system_prompt=new_system_prompt,
        messages=compressed,
        cwd=getattr(agent, "working_directory", None),
        profile_name=_profile_for_child,
        compression_lock_holder=lease.holder,
        require_compression_lease=lease.holder is not None,
        require_lease_refresh=lease.holder is not None,
        lease_ttl_seconds=lease.ttl,
        watermark=(
            lease.watermark
            if _foreign_tail_ceiling is not None
            else None
        ),
        watermark_ceiling=_foreign_tail_ceiling,
    )
    # `already_present` stamping is done by run_agent's _sync_persisted_markers;
    # this branch covers inserted/merged only; direct callers must use that wrapper.
    if compressed_user_turn_outcome in {"inserted", "merged"}:
        # Stamp the anchor source row itself, not the (drifted, possibly out-of-range)
        # persist index; don't match the HANDOFF row — for `merged` it is a superset.
        _compressed_anchor_source = None
        for _reversed_message in reversed(messages):
            if _is_real_user_message(_reversed_message):
                _compressed_anchor_source = _reversed_message
                break
        if isinstance(_compressed_anchor_source, dict):
            _compressed_anchor_source[_DB_PERSISTED_MARKER] = True
            _session_messages = getattr(
                agent, "_session_messages", None
            )
            if (
                isinstance(_session_messages, list)
                and _session_messages is not messages
            ):
                # Adoption may leave _session_messages on the pre-adoption list with an out-of-
                # range idx; stamp every scoped twin against the ANCHOR SOURCE, as the wrapper.
                _anchor_timestamp = _compressed_anchor_source.get(
                    "timestamp"
                )
                _found_exact_timestamp_candidate = False
                if _anchor_timestamp is not None:
                    for _twin_message in _session_messages:
                        if (
                            isinstance(_twin_message, dict)
                            and _twin_message.get("timestamp")
                            == _anchor_timestamp
                            and _messages_match_scoped_identity(
                                _twin_message,
                                _compressed_anchor_source,
                            )
                        ):
                            # Count an exact scoped twin REGARDLESS of marker: an already-stamped twin must
                            # still suppress the broad fallback or a content-equal old dup gets stamped.
                            _found_exact_timestamp_candidate = True
                            if not _twin_message.get(
                                _DB_PERSISTED_MARKER
                            ):
                                _twin_message[
                                    _DB_PERSISTED_MARKER
                                ] = True
                if not _found_exact_timestamp_candidate:
                    # No exact twin anywhere (or timestamp-less anchor): stamp every scoped match.
                    # An already-stamped exact hit never opens this branch.
                    for _twin_message in _session_messages:
                        if (
                            isinstance(_twin_message, dict)
                            and not _twin_message.get(
                                _DB_PERSISTED_MARKER
                            )
                            and _messages_match_scoped_identity(
                                _twin_message,
                                _compressed_anchor_source,
                            )
                        ):
                            _twin_message[
                                _DB_PERSISTED_MARKER
                            ] = True
    for _handoff_message in compressed:
        if isinstance(_handoff_message, dict):
            _handoff_message[_DB_PERSISTED_MARKER] = True
    agent.session_id = new_session_id
    agent._db_flush_scan_prefix = None
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(agent.session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = agent.session_id
    try:
        from hermes_logging import set_session_context

        set_session_context(agent.session_id)
    except Exception:
        pass
    agent._session_db_created = True
    # Carry /goal to the child: load_goal is a flat per-session lookup with no
    # parent walk, so the goal would silently die at the boundary.
    try:
        from hermes_cli.goals import migrate_goal_to_session
        migrate_goal_to_session(old_session_id, agent.session_id, reason="compression")
    except Exception as _goal_err:
        logger.debug("Could not migrate goal on compression: %s", _goal_err)
    # Same boundary hazard for /heartbeat state — carry it too.
    try:
        from hermes_cli.heartbeat import migrate_heartbeat_to_session
        migrate_heartbeat_to_session(old_session_id, agent.session_id)
    except Exception as _hb_err:
        logger.debug("Could not migrate heartbeat on compression: %s", _hb_err)
    # Same hazard for a persistent /loop: carry it so recurring wakeups survive.
    try:
        from hermes_cli.loops import migrate_loop_to_session
        migrate_loop_to_session(old_session_id, agent.session_id, reason="compression")
    except Exception as _loop_err:
        logger.debug("Could not migrate loop on compression: %s", _loop_err)
    # Carry the title unchanged: renumbering per rotation made one session look
    # like many. Uniqueness holds: _set_session_title transfers off the ancestor.
    if old_title:
        # Read provenance BEFORE the write: the transfer clears the ancestor's row, so
        # a later read is None and the child would be frozen as "user".
        _src = None
        try:
            _src = agent._session_db.get_session_title_source(
                old_session_id
            )
        except Exception as _src_err:
            logger.debug(
                "Could not read title provenance: %s", _src_err
            )
        try:
            agent._session_db.set_session_title(
                agent.session_id, old_title
            )
        except (ValueError, Exception) as e:
            logger.debug("Could not propagate title on compression: %s", e)
        else:
            # set_session_title() records "user"; restore the original authority so an
            # inherited auto-title stays upgradeable and a manual one stays pinned.
            if _src is not None:
                try:
                    agent._session_db.set_session_title_source(
                        agent.session_id, _src
                    )
                except Exception as _src_err:
                    logger.debug(
                        "Could not propagate title provenance: %s",
                        _src_err,
                    )


def _warn_summary_or_aux_fallback(agent: Any) -> None:
    """Surface a failed summary, or a recovered-but-broken aux compression model, once."""
    summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
    if summary_error:
        if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
            agent._last_compression_summary_warning = summary_error
            agent._emit_warning(
                f"⚠ Compression summary failed: {summary_error}. "
                "Inserted a fallback context marker."
            )
    else:
        # Aux model may have errored and been recovered on main; tell the user their
        # auxiliary.compression.model is broken even though compression succeeded.
        _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
        _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
        if _aux_fail_model:
            # Dedup on (model, error) so we don't spam on every compaction
            _aux_key = (_aux_fail_model, _aux_fail_err)
            if getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
                agent._last_aux_fallback_warning_key = _aux_key
                agent._emit_warning(
                    f"ℹ Configured compression model '{_aux_fail_model}' failed "
                    f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                    "check auxiliary.compression.model in config.yaml."
                )


def _finish_compaction_boundary(
    agent: Any,
    compressed: list,
    *,
    new_system_prompt: str,
    old_session_id: Optional[str],
    in_place: bool,
    compacted_in_place: bool,
    session_commit_succeeded: bool,
    defer_context_engine_notification: bool,
    compression_made_progress: bool,
    compression_used_fallback: bool,
    compression_feasibility_skip: bool,
    task_id: str,
) -> int:
    """Post-commit bookkeeping: notify engines/providers/hooks, re-arm usage tracking.

    Returns the rough post-compression token estimate (diagnostics only).
    """
    # old_session_id is bound only on rotation; _boundary_parent is the id the
    # boundary notifications attribute prior state to (old id, or same id in-place).
    _old_sid = old_session_id
    _is_boundary = bool(_old_sid) or in_place
    _context_engine_boundary_committed = session_commit_succeeded and (
        bool(_old_sid) or compacted_in_place
    )
    _boundary_parent = _old_sid or agent.session_id or ""

    # The heartbeat's terminal stamp landed on the PARENT before the id re-pointed;
    # clear labels (keep last_activity_at) so the archived row isn't falsely fresh.
    if _old_sid and session_commit_succeeded:
        try:
            _labels_db = getattr(agent, "_session_db", None)
            _clear_labels = getattr(
                type(_labels_db) if _labels_db is not None else None,
                "clear_session_activity_labels",
                None,
            )
            if callable(_clear_labels):
                _clear_labels(_labels_db, _old_sid)
        except Exception:
            logger.debug(
                "failed to clear archived compression parent's activity "
                "labels (ignored)",
                exc_info=True,
            )

    # Plugin engines use boundary_reason="compression" to keep lineage/checkpoint
    # state. Fires in BOTH modes: in-place passes the same id, the boundary is real.
    if _context_engine_boundary_committed:
        if defer_context_engine_notification:
            _queue_context_engine_compression_notification(
                agent,
                new_session_id=agent.session_id or "",
                old_session_id=_boundary_parent,
            )
        else:
            _notify_context_engine_compression_complete(
                agent,
                new_session_id=agent.session_id or "",
                old_session_id=_boundary_parent,
            )

    # Providers refresh cached per-session state; reset=False, conversation goes on.
    # Fires in BOTH modes so buffers don't double-count dropped turns in-place.
    try:
        if _is_boundary and agent._memory_manager:
            agent._memory_manager.on_session_switch(
                agent.session_id or "",
                parent_session_id=_boundary_parent,
                reset=False,
                reason="compression",
            )
    except Exception as _me_err:
        logger.debug("memory manager on_session_switch (compression): %s", _me_err)

    # Route via _emit_status so the warning reaches gateway platforms; store it on
    # _compression_warning so a late-bound status_callback can replay it.
    _cc = agent.context_compressor.compression_count
    if _cc >= 2:
        _cc_msg = (
            f"{agent.log_prefix}⚠️  Session compressed {_cc} times — "
            f"accuracy may degrade. Consider /new to start fresh."
        )
        agent._compression_warning = _cc_msg
        agent._emit_status(_cc_msg)

    # session:compress lets hooks ingest the old session before it's lost;
    # in_place=True tells them the same id was compacted rather than rotated.
    if getattr(agent, "event_callback", None):
        try:
            agent.event_callback("session:compress", {
                "platform": agent.platform or "",
                "session_id": agent.session_id,
                "old_session_id": _old_sid or "",
                "in_place": in_place,
                "compression_count": agent.context_compressor.compression_count,
            })
        except Exception as e:
            logger.debug("event_callback error on session:compress: %s", e)

    # Rotation-independent flag: the gateway uses it (not an id diff) to re-baseline
    # transcript handling (history_offset=0 + rewrite on the same id) in-place.
    agent._last_compression_attempt_in_place = compacted_in_place
    agent._last_compaction_in_place = compacted_in_place

    # Diagnostics only, not provider usage: schema-heavy rough estimates can stay
    # above threshold even after the next real request fits.
    _compressed_est = estimate_request_tokens_rough(
        compressed,
        system_prompt=new_system_prompt or "",
        tools=agent.tools or None,
    )
    agent.context_compressor.last_compression_rough_tokens = _compressed_est
    agent.context_compressor.last_prompt_tokens = -1
    agent.context_compressor.last_completion_tokens = 0
    agent.context_compressor.awaiting_real_usage_after_compression = True
    # Transcript rewritten: invalidate the usage anchor's base snapshot explicitly
    # (its structural check would fail closed anyway); estimate until re-anchored.
    agent._usage_anchor = None
    agent._turn_base_usage_anchor = None
    # Arm the effectiveness verdict only after a completed rewrite crosses the
    # boundary so later usage isn't charged to an attempt that changed nothing.
    if compression_made_progress:
        record_boundary = getattr(
            type(agent.context_compressor),
            "record_completed_compaction",
            None,
        )
        if callable(record_boundary):
            record_boundary(
                agent.context_compressor,
                used_fallback=compression_used_fallback,
                feasibility_skip=compression_feasibility_skip,
            )
        else:
            agent.context_compressor._verify_compaction_cleared_threshold = True

    # Clear file-read dedup cache: original read content was summarized away, so a
    # re-read needs full content, not a "file unchanged" stub.
    try:
        from tools.file_tools import reset_file_dedup
        reset_file_dedup(task_id)
    except Exception:
        pass
    # Same for the skill_view repeat-view dedup: a post-compression
    # re-view must return the full skill content again.
    try:
        from tools.skills_tool import reset_skill_view_dedup
        reset_skill_view_dedup(task_id)
    except Exception:
        pass
    return _compressed_est


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
    bypass_cooldown: bool = False,
    defer_context_engine_notification: bool = False,
    commit_fence: Optional[CompressionCommitFence] = None,
) -> Tuple[list, str]:
    """Compress conversation context and split the session in SQLite.

    ``force`` (manual /compress) clears the summary-failure cooldown;
    ``bypass_cooldown`` (provider-proven overflow) skips it once, breakers still
    apply. ``commit_fence`` stops a timed-out worker mutating session state.
    Returns ``(messages, system_prompt)``; on abort input is unchanged, NOT split.
    """
    _compressor_attempt_snapshot = _snapshot_compressor_attempt_state(
        agent.context_compressor
    )
    # Claim attempt ownership so a late-unwinding sibling (stall-fallback overlap)
    # cannot restore its snapshot over ours or clear our cancellation consult.
    _attempt_generation = _claim_compressor_attempt(agent.context_compressor)
    _durable_cooldown_authoritative: Optional[bool] = None
    _durable_cooldown_state: Optional[dict[str, Any]] = None
    if (
        defer_context_engine_notification
        and callable(getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None))
    ):
        raise RuntimeError("a compression notification is already pending")

    # Per-attempt outcome for conversation_history_after_compression(); None means
    # aborted/no boundary, so the previous flush baseline stays authoritative.
    agent._last_compression_attempt_recorded = True
    agent._last_compression_attempt_in_place = None
    # Clear at the VERY TOP, before codex/breaker early-returns: a stale value must
    # not make a later no-op look like lock contention to automatic-path consumers.
    agent._compression_skipped_due_to_lock = None
    # Per-attempt transient-block signal, set when a cooldown/backoff guard no-ops
    # this pass.
    agent._compression_blocked_transient = None

    _attempt_started_at = time.monotonic()
    _attempt_id = uuid.uuid4().hex
    _trigger_source = "manual" if force else "auto"
    try:
        agent._compression_attempt_id = _attempt_id
        setattr(agent.context_compressor, "_compression_telemetry_seed", {
            "attempt_id": _attempt_id,
            "session_id": agent.session_id or "",
            "trigger_source": _trigger_source,
        })
    except Exception:
        pass

    # Codex owns the real thread; route compaction to its own compact (config
    # compression.codex_app_server_auto). Memory handoff is Hermes-only: no native
    # summary prompt to inject into. `is True`: MagicMock attributes are truthy.
    checkpoint_required = (
        getattr(agent, "compression_checkpoint_required", False) is True
    )
    if getattr(agent, "api_mode", None) == "codex_app_server":
        if checkpoint_required:
            raise _checkpoint_blocked(
                "codex_app_server owns the authoritative thread and does not "
                "expose a truthful pre-compaction transcript boundary"
            )
        _codex_fence_entered = False
        if commit_fence is not None:
            _codex_fence_entered = commit_fence.begin_commit(
                getattr(agent, "_hard_interrupt_requested", None)
            )
            if not _codex_fence_entered:
                _restore_compressor_attempt_state(
                    agent.context_compressor, _compressor_attempt_snapshot,
                    attempt_generation=_attempt_generation,
                )
                existing_prompt = _existing_system_prompt(agent, system_message)
                return messages, existing_prompt
        try:
            return _compress_context_via_codex_app_server(
                agent,
                messages,
                system_message,
                approx_tokens=approx_tokens,
                task_id=task_id,
                force=force,
            )
        finally:
            if _codex_fence_entered:
                commit_fence.finish_commit()

    # All automatic entrypoints honor compressor cooldown/breaker state; hygiene's
    # fresh AIAgent loads the persisted streak via bind_session_state() first.
    if not force:
        _refresh_persisted_compression_guards(agent.context_compressor)
        blocked = getattr(
            type(agent.context_compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and _automatic_gate_blocked(
            blocked, agent.context_compressor, bypass_cooldown
        ):
            _mark_compression_blocked_transient(agent, agent.context_compressor)
            existing_prompt = _existing_system_prompt(agent, system_message)
            return messages, existing_prompt

    # Lazy feasibility probe (~400ms cold) on first attempt, not __init__; it sets
    # _compression_warning so status replay still surfaces the warning.
    if not getattr(agent, "_compression_feasibility_checked", False):
        # Mark checked only after the probe completes; a raise leaves it unset
        # harmlessly, transient failures are swallowed inside so it sets next pass.
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    _pre_msg_count = len(messages)
    # In-place keeps the SAME session_id (no rotation/child/renumber/re-sync). A
    # missing attribute must default True, not rotation, which can wedge sessions.
    in_place = bool(getattr(agent, "compression_in_place", True))
    # Set True once the in-place DB write actually completes (the DB block can
    # raise and skip it). Surfaced to the gateway via agent._last_compaction_in_place.
    compacted_in_place = False
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
        agent.session_id or "none", _pre_msg_count,
        f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model,
        focus_topic,
    )
    _compaction_status = COMPACTION_STATUS
    if not force:
        _compaction_status = automatic_compaction_status_message(
            agent.context_compressor,
            phase="compress",
            default_message=_compaction_status,
            approx_tokens=approx_tokens,
            message_count=_pre_msg_count,
            model=agent.model,
            focus_topic=focus_topic,
        )
    _compaction_status_emitted = bool(_compaction_status)
    if _compaction_status:
        agent._emit_status(_compaction_status)
    lifecycle = _CompactionLifecycle(agent, _compaction_status_emitted)

    lease, _abort_prompt = _acquire_compression_lease(
        agent,
        commit_fence=commit_fence,
        lifecycle=lifecycle,
        system_message=system_message,
        approx_tokens=approx_tokens,
        attempt_started_at=_attempt_started_at,
    )
    if lease is None:
        return messages, _abort_prompt

    # Publish the holder-qualified release hook before a timeout can win the
    # fence. If no durable lock was acquired there is no hook to publish.
    lease.finish_lock_setup()

    _adopted = _adopt_if_parent_rotated(agent, lease, messages, system_message)
    if _adopted is not None:
        return _adopted

    # Snapshot durable cooldown only once we own the lease. Runs for force=True
    # too but skips the automatic breaker gate: manual compression retries now.
    _durable_cooldown_authoritative, _durable_cooldown_state = (
        _capture_authoritative_cooldown_under_lease(
            agent.context_compressor,
            _compressor_attempt_snapshot,
        )
    )
    if _durable_cooldown_authoritative is False:
        # Durable cooldown read failed under a built-in compressor: force=True could
        # clear an unknown newer row before cancellation could restore it. Abort.
        lease.release()
        existing_prompt = _existing_system_prompt(agent, system_message)
        return messages, existing_prompt

    # Another path may have compacted this session in place since construction;
    # re-read breaker state under the lock, not the bind_session_state() snapshot.
    if not force:
        compressor = agent.context_compressor
        _refresh_persisted_compression_guards(
            compressor,
            include_cooldown=False,
        )
        blocked = getattr(
            type(compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and _automatic_gate_blocked(
            blocked, compressor, bypass_cooldown
        ):
            _mark_compression_blocked_transient(agent, compressor)
            lease.release()
            existing_prompt = _existing_system_prompt(agent, system_message)
            return messages, existing_prompt

    _activity_heartbeat: Optional[_CompressionActivityHeartbeat] = None
    messages_before_compression = None
    try:
        lease.start_refresher()

        if not in_place:
            _adopted_parent = _adopt_grown_durable_parent(agent, lease, messages)
            if _adopted_parent is not None:
                messages = _adopted_parent
                _pre_msg_count = len(messages)
                # Estimate was for the stale snapshot; force re-derivation from adopted rows.
                approx_tokens = 0
                # Adopted list is fully durable: re-anchor persist idx at the end so the post-
                # compression flush skips it; run_agent marker sync realigns _session_messages.
                agent._persist_user_message_idx = len(messages)

        memory_context = _pre_compress_memory_context(agent, messages, checkpoint_required)

        compress_fn, compress_kwargs = _resolve_compress_call(
            agent,
            approx_tokens=approx_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
            bypass_cooldown=bypass_cooldown,
        )

        messages_before_compression = copy.deepcopy(messages)
        _activity_heartbeat = _CompressionActivityHeartbeat(
            agent, commit_fence=commit_fence
        ).start()
        # Interrupts/redirects must not tear a summary in half. Use the explicit stop
        # Event (message fields race) + fence timeout so pool slots free promptly.
        _hard_cancel_event = getattr(agent, "_hard_interrupt_requested", None)
        compressed = _run_summary_dispatch(
            agent,
            messages,
            compress_fn,
            compress_kwargs,
            commit_fence=commit_fence,
            attempt_generation=_attempt_generation,
            hard_cancel_event=_hard_cancel_event,
        )
    except AuxiliaryExplicitCancellation:
        try:
            _restore_compressor_attempt_state(
                agent.context_compressor,
                _compressor_attempt_snapshot,
                durable_cooldown_authoritative=_durable_cooldown_authoritative,
                durable_cooldown_state=_durable_cooldown_state,
                attempt_generation=_attempt_generation,
            )
        except BaseException as _rollback_exc:
            # Compensation failure must surface, but it must not strand the
            # session lease or retain an in-memory transcript mutation.
            _restore_messages_snapshot(messages, messages_before_compression)
            if _activity_heartbeat is not None:
                _activity_heartbeat.stop("context compression rollback failed")
                _activity_heartbeat = None
            lease.release()
            _emit_aborted_attempt_telemetry(agent, _attempt_started_at, f"rollback:{type(_rollback_exc).__name__}")
            raise
        _restore_messages_snapshot(messages, messages_before_compression)
        # Record after restore so rollback cannot wipe a stall backoff, and
        # while the lease is still held so the next turn cannot race it.
        _stall_backoff = _record_stall_interrupted_backoff(
            agent,
            commit_fence=commit_fence,
            started_at=_attempt_started_at,
            messages=messages,
            approx_tokens=approx_tokens,
        )
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression cancelled")
            _activity_heartbeat = None
        lease.release()
        _emit_aborted_attempt_telemetry(
            agent,
            _attempt_started_at,
            (
                STALL_INTERRUPTED_FAILURE_CLASS
                if _stall_backoff
                else "explicit_interrupt"
            ),
        )
        _existing_sp = _existing_system_prompt(agent, system_message)
        return messages, _existing_sp
    except BaseException as _compress_exc:
        # Any failure after lock acquisition must release it or the session is
        # permanently blocked from compression.
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression failed")
            _activity_heartbeat = None
        lease.release()
        _emit_aborted_attempt_telemetry(agent, _attempt_started_at, f"exception:{type(_compress_exc).__name__}")
        raise
    finally:
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression completed")

    _commit_fence_entered = False
    try:
        # Capture the verdict before rotation callbacks: lifecycle hooks may reset
        # compressor fields on rebind; record only after the full boundary commits.
        _compression_made_progress = bool(
            getattr(agent.context_compressor, "_last_compression_made_progress", False)
        )
        _compression_used_fallback = bool(
            getattr(agent.context_compressor, "_last_summary_fallback_used", False)
        )
        _compression_feasibility_skip = bool(
            getattr(agent.context_compressor, "_last_feasibility_skip", False)
        )

        # Aborted compression returns input unchanged: surface the error, skip rotation
        # (no session ended); auto-compress callers detect no-op via equal lengths.
        if getattr(agent.context_compressor, "_last_compress_aborted", False):
            try:
                _err = getattr(agent.context_compressor, "_last_summary_error", None) or "unknown error"
                if getattr(agent, "_last_compression_summary_warning", None) != _err:
                    agent._last_compression_summary_warning = _err
                    agent._emit_warning(
                        f"⚠ Compression aborted: {_err}. "
                        "No messages were dropped — conversation continues unchanged. "
                        "Run /compress to retry, or /new to start a fresh session."
                    )
                _existing_sp = _existing_system_prompt(agent, system_message)
                _emit_aborted_attempt_telemetry(
                    agent,
                    _attempt_started_at,
                    (
                        getattr(agent.context_compressor, "_last_summary_error", None)
                        and "summary_generation_aborted"
                    ),
                )
                return messages, _existing_sp
            finally:
                lease.release()

        # Compare semantic state, not identity: engines may return an equal copy or
        # mutate the live list. ``==`` first (subclass __eq__), then marker-insensitive.
        if compressed == messages_before_compression or (
            _strip_marker_for_comparison(compressed)
            == _strip_marker_for_comparison(messages_before_compression)
        ):
            if messages != messages_before_compression:
                messages[:] = copy.deepcopy(messages_before_compression)
            logger.info(
                "Compression made no progress (session=%s) — skipping boundary rewrite.",
                agent.session_id or "none",
            )
            # Unchanged output would fail identically next turn; arm structural backoff so
            # auto-compress stops re-firing each turn (success lifts it, force overrides).
            try:
                _no_progress_recorder = getattr(
                    agent.context_compressor, "_record_structural_no_op", None
                )
                if callable(_no_progress_recorder):
                    _no_progress_recorder(
                        "compaction returned the transcript unchanged "
                        "(no_progress)"
                    )
            except Exception:
                logger.debug(
                    "no-progress backoff arm failed", exc_info=True
                )
            _existing_sp = _existing_system_prompt(agent, system_message)
            _emit_aborted_attempt_telemetry(agent, _attempt_started_at, "no_progress")
            lease.release()
            return messages, _existing_sp

        if not compressed:
            logger.error(
                "context compression returned an empty transcript; refusing to "
                "rotate session=%s so the parent remains resumable",
                agent.session_id or "none",
            )
            try:
                agent._emit_warning(
                    "⚠ Compression returned an empty transcript. "
                    "No session split was performed; conversation continues unchanged."
                )
            except Exception:
                pass
            _existing_sp = _existing_system_prompt(agent, system_message)
            lease.release()
            return messages, _existing_sp

        # A newer attempt claiming this compressor supersedes us; discard the late
        # candidate. Fence poison alone misses a successor that minted its own fence.
        _attempt_superseded = not _compressor_attempt_is_current(
            agent.context_compressor, _attempt_generation
        )
        if _attempt_superseded:
            logger.warning(
                "Discarding late compression candidate: attempt generation "
                "%s was superseded by a newer attempt (current: %s) "
                "(session=%s).",
                _attempt_generation,
                getattr(
                    agent.context_compressor,
                    "_compression_attempt_generation",
                    None,
                ),
                agent.session_id or "none",
            )
            _restore_messages_snapshot(messages, messages_before_compression)
            agent._last_compaction_in_place = False
            _existing_sp = _existing_system_prompt(agent, system_message)
            _emit_aborted_attempt_telemetry(agent, _attempt_started_at, "attempt_superseded")
            lease.release()
            return messages, _existing_sp

        if commit_fence is not None:
            _commit_fence_entered = commit_fence.begin_commit(_hard_cancel_event)
            if not _commit_fence_entered:
                _restore_compressor_attempt_state(
                    agent.context_compressor,
                    _compressor_attempt_snapshot,
                    durable_cooldown_authoritative=_durable_cooldown_authoritative,
                    durable_cooldown_state=_durable_cooldown_state,
                    attempt_generation=_attempt_generation,
                )
                _restore_messages_snapshot(messages, messages_before_compression)
                logger.info(
                    "Compression commit cancelled before session mutation "
                    "(session=%s).",
                    agent.session_id or "none",
                )
                agent._last_compaction_in_place = False
                _stall_backoff = _record_stall_interrupted_backoff(
                    agent,
                    commit_fence=commit_fence,
                    started_at=_attempt_started_at,
                    messages=messages,
                    approx_tokens=approx_tokens,
                )
                _existing_sp = _existing_system_prompt(agent, system_message)
                _emit_aborted_attempt_telemetry(
                    agent,
                    _attempt_started_at,
                    (
                        STALL_INTERRUPTED_FAILURE_CLASS
                        if _stall_backoff
                        else "commit_fence_cancelled"
                    ),
                )
                lease.release()
                return messages, _existing_sp

        _warn_summary_or_aux_fallback(agent)

        _fold_todo_snapshot(agent, compressed)
        compressed_user_turn_outcome = _ensure_compressed_has_user_turn(
            messages, compressed
        )

        new_system_prompt = _rebuild_system_prompt_at_boundary(agent, system_message)

        _session_commit_succeeded = False
        _commit_started_at = time.monotonic()
        split_status = "not_applicable"
        old_session_id: Optional[str] = None  # bound only once rotation begins
        if agent._session_db:
            split_status = "pending"
            try:
                # Memory extraction runs in BOTH modes: pre-compaction turns are summarized
                # away whether or not the id rotates.
                agent.commit_memory_session(messages)

                # Pop _compaction_tail tags before the size estimate / rotation: they must not
                # inflate anti-growth or reach the provider. Track ids: salvage may subset list.
                _tail_tagged_ids = {
                    id(m)
                    for m in compressed
                    if isinstance(m, dict) and m.pop("_compaction_tail", None)
                }

                compressed, _refused_sp = _salvage_or_refuse_grown_transcript(
                    agent,
                    messages,
                    compressed,
                    system_message=system_message,
                    attempt_started_at=_attempt_started_at,
                    attempt_snapshot=_compressor_attempt_snapshot,
                )
                if compressed is None:
                    lease.release()
                    return messages, _refused_sp

                if in_place:
                    # In-place compaction: same session_id; soft-archive old turns (active=0, still
                    # searchable) + insert `compressed` atomically; no pre-flush (tail already in).
                    from agent.context_compressor import (
                        PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY,
                    )

                    # Tail rows tagged by compress() are archived as superseded duplicates, not
                    # compacted=1. Count against the FINAL list — salvage may have dropped rows.
                    _tail_count = sum(
                        1 for m in compressed if id(m) in _tail_tagged_ids
                    )
                    agent._session_db.archive_and_compact(
                        agent.session_id,
                        compressed,
                        model_config_patch={
                            PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: None,
                        },
                        watermark=lease.watermark,
                        lock_holder=lease.holder,
                        tail_count=_tail_count,
                    )
                    split_status = "in_place_committed"
                    # compress() returned marker-swept copies; stamp them as persisted or the next
                    # flush re-INSERTs the whole compacted transcript, doubling the live set.
                    from agent.context_compressor import (
                        stamp_db_persisted_markers,
                    )

                    stamp_db_persisted_markers(compressed)
                    # Reset flush identity set so next turn diffs against the COMPACTED transcript:
                    # only genuinely new messages append (no summary dup, no resurrected turns).
                    agent._flushed_db_message_ids = set()
                    # Rotation-independent signal; the gateway reads this (not an id diff) to
                    # re-baseline transcript handling.
                    compacted_in_place = True
                else:
                    # Bind old_session_id first: it is the rollback key in the handler below.
                    old_session_id = agent.session_id
                    _publish_rotated_compaction(
                        agent,
                        messages,
                        compressed,
                        new_system_prompt=new_system_prompt,
                        lease=lease,
                        old_session_id=old_session_id,
                        compressed_user_turn_outcome=compressed_user_turn_outcome,
                    )
                    split_status = "rotated_committed"

                # In-place mode still updates/replaces the current row here.
                # Rotation already published prompt + compacted handoff atomically.
                if in_place:
                    agent._session_db.update_system_prompt(
                        agent.session_id, new_system_prompt
                    )
                    agent._last_flushed_db_idx = 0
                else:
                    agent._last_flushed_db_idx = len(compressed)
                    agent._flushed_db_message_session_id = agent.session_id
                _session_commit_succeeded = True
            except Exception as e:
                if (
                    not in_place
                    and old_session_id
                    and agent.session_id == old_session_id
                ):
                    # Atomic publication failed (including lease loss): keep the
                    # parent live and discard the stale compacted snapshot.
                    old_session_id = None
                    # _db_flush_scan_prefix is intentionally NOT cleared: the scan is identity-based
                    # and the deepcopy replaces every row. A failed parent flush clears its own; the
                    # snapshot path leaves the live list untouched. Recheck both before adding one.
                    messages[:] = copy.deepcopy(messages_before_compression)
                    compressed = messages
                    _compression_made_progress = False
                    # Only the runway rolls back: the full snapshot restore is for pre-commit
                    # cancels (telemetry keeps failed values).
                    _restore_prune_rearm_tokens(agent.context_compressor, _compressor_attempt_snapshot)
                elif (
                    in_place
                    and split_status != "in_place_committed"
                    and messages_before_compression is not None
                ):
                    # In-place rollback: archive_and_compact is atomic so old rows stay active, but
                    # marker-swept `compressed` would re-INSERT on top of them (doubling each try).
                    # Gate on split_status (set right after commit); deepcopy keeps markers/identity
                    messages[:] = copy.deepcopy(messages_before_compression)
                    compressed = messages
                    _compression_made_progress = False
                    _restore_prune_rearm_tokens(agent.context_compressor, _compressor_attempt_snapshot)
                split_status = (
                    "aborted"
                    if old_session_id is None and not in_place
                    else "failed_not_indexed"
                )
                # If rotation rolled back to the parent, agent.session_id is the indexed parent
                # and old_session_id was cleared: recovery, not an un-indexed orphan.
                if old_session_id is None and not in_place:
                    logger.warning(
                        "Compression rotation aborted and rolled back to the "
                        "parent session (%s): %s", agent.session_id or "?", e,
                    )
                else:
                    logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)
                # Arm the failure cooldown so the next turn can't rerun the doomed compression;
                # try/except so a stub compressor can't mask the original error in this handler.
                try:
                    agent.context_compressor._record_compression_failure_cooldown(
                        _SPLIT_FAILURE_COOLDOWN_SECONDS,
                        f"session_split_failed: {e}",
                    )
                except Exception:
                    logger.debug(
                        "could not record split-failure cooldown",
                        exc_info=True,
                    )

        _compressed_est = _finish_compaction_boundary(
            agent,
            compressed,
            new_system_prompt=new_system_prompt,
            old_session_id=old_session_id,
            in_place=in_place,
            compacted_in_place=compacted_in_place,
            session_commit_succeeded=_session_commit_succeeded,
            defer_context_engine_notification=defer_context_engine_notification,
            compression_made_progress=_compression_made_progress,
            compression_used_fallback=_compression_used_fallback,
            compression_feasibility_skip=_compression_feasibility_skip,
            task_id=task_id,
        )

        logger.info(
            "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=true",
            agent.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        lifecycle.commit_status = "committed" if split_status in {"not_applicable", "in_place_committed", "rotated_committed"} else "aborted"
        _emit_compression_attempt_telemetry(
            agent,
            started_at=_attempt_started_at,
            commit_status=lifecycle.commit_status,
            split_status=split_status,
            failure_class=(
                "session_split_failed"
                if split_status in {"failed_not_indexed", "aborted"}
                else None
            ),
            commit_started_at=_commit_started_at,
        )
        return compressed, new_system_prompt
    finally:
        # Release the OLD session's lock only after rotation and all post-rotation
        # bookkeeping; a waking contender then sees the NEW id and acquires on that.
        try:
            lease.release()
        finally:
            if _commit_fence_entered:
                commit_fence.finish_commit()


def _codex_compaction_cooldown_remaining(agent: Any) -> float:
    """Seconds left on this session's compaction-failure cooldown (0 = clear)."""
    compressor = getattr(agent, "context_compressor", None)
    getter = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if not callable(getter):
        return 0.0
    try:
        state = getter(refresh=True)
    except Exception:
        logger.debug("codex compaction cooldown lookup failed", exc_info=True)
        return 0.0
    if not state:
        return 0.0
    try:
        return max(0.0, float(state.get("remaining_seconds") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _record_codex_compaction_failure(agent: Any, error: str) -> None:
    """Arm the shared compression-failure cooldown after a failed codex compaction.

    The codex path returns the transcript unchanged, so without a cooldown the
    still-over-threshold session would retry every turn.
    """
    from agent.context_compressor import _SUMMARY_FAILURE_COOLDOWN_SECONDS

    compressor = getattr(agent, "context_compressor", None)
    recorder = getattr(compressor, "_record_compression_failure_cooldown", None)
    if not callable(recorder):
        return
    try:
        recorder(_SUMMARY_FAILURE_COOLDOWN_SECONDS, error)
    except Exception:
        logger.debug("codex compaction cooldown persist failed", exc_info=True)


def _compress_context_via_codex_app_server(
    agent: Any,
    messages: list,
    system_message: Optional[str],
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    force: bool = False,
) -> Tuple[list, str]:
    """Route compaction to Codex app-server for Codex-owned threads.

    Rewriting the local transcript would not shrink the Codex thread, so Codex
    compacts its own thread and Hermes' transcript is left unchanged.
    """
    auto_mode = str(
        getattr(agent, "codex_app_server_auto_compaction", "native") or "native"
    ).lower()
    if auto_mode not in {"native", "hermes", "off"}:
        auto_mode = "native"
    if not force and auto_mode != "hermes":
        logger.info(
            "codex app-server compaction skipped: mode=%s force=false "
            "(session=%s messages=%d tokens=~%s)",
            auto_mode,
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = _existing_system_prompt(agent, system_message)
        return messages, existing_prompt

    # Automatic entrypoints honor the compressor-owned cooldown: a recent compaction
    # failed, and retrying every turn is what thrashes.
    if not force:
        _cooldown_remaining = _codex_compaction_cooldown_remaining(agent)
        if _cooldown_remaining > 0:
            logger.info(
                "codex app-server compaction skipped: failure cooldown active "
                "for %.0fs (session=%s messages=%d tokens=~%s)",
                _cooldown_remaining,
                getattr(agent, "session_id", None) or "none",
                len(messages),
                f"{approx_tokens:,}" if approx_tokens else "unknown",
            )
            existing_prompt = _existing_system_prompt(agent, system_message)
            return messages, existing_prompt

    codex_session = getattr(agent, "_codex_session", None)
    if codex_session is None:
        logger.info(
            "codex app-server compaction skipped: no active codex thread "
            "(session=%s messages=%d tokens=~%s)",
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = _existing_system_prompt(agent, system_message)
        return messages, existing_prompt

    logger.info(
        "codex app-server compaction started: session=%s messages=%d tokens=~%s",
        getattr(agent, "session_id", None) or "none",
        len(messages),
        f"{approx_tokens:,}" if approx_tokens else "unknown",
    )
    try:
        agent._emit_status(COMPACTION_STATUS)
    except Exception:
        pass

    _activity_heartbeat: Optional[_CompressionActivityHeartbeat] = None
    try:
        _activity_heartbeat = _CompressionActivityHeartbeat(agent).start()
        result = codex_session.compact_thread()
    except BaseException:
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression failed")
        raise

    if getattr(result, "interrupted", False) or getattr(result, "error", None):
        _activity_heartbeat.stop("context compression failed")
    else:
        _activity_heartbeat.stop("context compression completed")

    if getattr(result, "should_retire", False):
        try:
            codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    if getattr(result, "interrupted", False) or getattr(result, "error", None):
        try:
            agent._emit_warning(
                f"⚠ Codex app-server compaction failed: {result.error}"
            )
        except Exception:
            pass
        # The transcript is returned unchanged, so the session is still over
        # threshold. Without a brake the next turn retries immediately.
        _record_codex_compaction_failure(
            agent,
            str(getattr(result, "error", None) or "compaction interrupted"),
        )
        existing_prompt = _existing_system_prompt(agent, system_message)
        return messages, existing_prompt

    try:
        from agent.codex_runtime import (
            _record_codex_app_server_compaction,
            _record_codex_app_server_usage,
        )

        _record_codex_app_server_compaction(
            agent,
            result,
            approx_tokens=approx_tokens,
            force=True,
        )
        # An empty usage report must consume the pending verdict, not leave deferral
        # armed until a later turn; minimal test engines may lack update_from_response.
        if hasattr(agent.context_compressor, "update_from_response"):
            _record_codex_app_server_usage(agent, result)
    except Exception:
        logger.debug("codex compaction bookkeeping failed", exc_info=True)

    try:
        from tools.file_tools import reset_file_dedup

        reset_file_dedup(task_id)
    except Exception:
        pass

    logger.info(
        "codex app-server compaction done: session=%s thread=%s turn=%s",
        getattr(agent, "session_id", None) or "none",
        getattr(result, "thread_id", None) or "",
        getattr(result, "turn_id", None) or "",
    )
    existing_prompt = _existing_system_prompt(agent, system_message)
    # Terminal edge only on success — failure/interrupt paths above return
    # without it, matching the main compress_context() gating.
    _emit_compaction_done(agent)
    return messages, existing_prompt


def try_shrink_image_parts_in_messages(
    api_messages: list,
    *,
    max_dimension: int = 8000,
) -> bool:
    """Re-encode oversized native image parts to recover from image-too-large errors.

    Mutates ``api_messages`` in place. Returns True if any part was replaced,
    False if nothing to shrink or Pillow could not help. Targets data-URL parts
    over 4 MB or ``max_dimension``; http(s) image URLs are left untouched.
    """
    if not api_messages:
        return False

    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False

    # 4 MB leaves headroom under Anthropic's 5 MB; shrinking loses quality but only
    # runs after a confirmed provider rejection, so the alternative is failure.
    target_bytes = 4 * 1024 * 1024
    # Anthropic also caps per-side pixels (8000, or lower in many-image requests);
    # the caller passes the parsed ceiling when the rejection includes it.
    changed_count = 0
    # Track over-target parts that could not be shrunk: if any remain, a retry
    # re-sends the same payload and wastes the single retry budget.
    unshrinkable_oversized = 0

    def _decode_pixels(data_url: str) -> Optional[tuple]:
        """Return ``(width, height)`` of a base64 data URL, or None on failure.

        None when Pillow is missing or the payload is corrupt; caller falls back to a
        bytes-only check.
        """
        try:
            import base64 as _b64_dim
            import io as _io_dim
            header_d, _, data_d = data_url.partition(",")
            if not data_d or not data_url.startswith("data:"):
                return None
            from PIL import Image as _PILImage
            with _PILImage.open(_io_dim.BytesIO(_b64_dim.b64decode(data_d))) as _img:
                return _img.size
        except Exception:
            return None

    def _shrink_data_url(url: str) -> tuple:
        """Return ``(resized_url, unshrinkable)`` for a data URL.

        ``resized_url`` is None when no rewrite applied. ``unshrinkable`` is True only
        when the image violated a constraint and resizing failed to satisfy that same
        constraint, so the caller knows a retry is pointless.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, False

        # The accept gate MUST use the axis that triggered the shrink: a pixel downscale
        # can re-encode to MORE bytes (PNG non-monotonic); byte-only reject wedges.
        needs_shrink = len(url) > target_bytes  # over byte budget
        triggered_by = "bytes" if needs_shrink else None
        if not needs_shrink:
            # Bytes fine; check pixels against the provider cap (tiny bytes, huge pixels).
            dims = _decode_pixels(url)
            if dims is None:
                # Pillow missing or corrupt data — fall back to byte-only.
                return None, False
            if max(dims) <= max_dimension:
                return None, False  # both bytes and pixels are within limits
            needs_shrink = True
            triggered_by = "dimension"

        try:
            header, _, data = url.partition(",")
            mime = "image/jpeg"
            if header.startswith("data:"):
                mime_part = header[len("data:"):].split(";", 1)[0].strip()
                if mime_part.startswith("image/"):
                    mime = mime_part
            import base64 as _b64
            raw = _b64.b64decode(data)
            suffix = {
                "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
            }.get(mime, ".jpg")
            tmp = tempfile.NamedTemporaryFile(
                prefix="hermes_shrink_", suffix=suffix, delete=False,
            )
            try:
                tmp.write(raw)
                tmp.close()
                resized = _resize_image_for_vision(
                    Path(tmp.name),
                    mime_type=mime,
                    max_base64_bytes=target_bytes,
                    max_dimension=max_dimension,
                )
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            if not resized:
                # Resize returned nothing — Pillow couldn't help.
                return None, True
            if triggered_by == "bytes":
                # Byte budget is the binding constraint — bytes must shrink.
                if len(resized) >= len(url):
                    return None, True  # re-encode made it bigger
                # The resizer may return an over-cap blob (long side freezes at the 64px short-
                # side floor); still over cap → re-400, so unshrinkable. Undecodable dims: skip.
                new_dims = _decode_pixels(resized)
                if new_dims is not None and max(new_dims) > max_dimension:
                    return None, True
                return resized, False
            # Dimension cap is binding: accept a byte-larger re-encode if now within cap.
            new_dims = _decode_pixels(resized)
            if new_dims is not None:
                if max(new_dims) <= max_dimension:
                    return resized, False
                # Still over the per-side cap — the resize didn't satisfy it.
                return None, True
            # Can't verify dimensions: fall back to the bytes-must-shrink gate so we never
            # accept an unverifiable byte-larger blob.
            if len(resized) >= len(url):
                return None, True
            return resized, False
        except Exception as exc:
            logger.warning("image-shrink recovery: re-encode failed — %s", exc)
            return None, triggered_by is not None

    def _source_to_data_url(source: Any) -> Optional[str]:
        if not isinstance(source, dict) or source.get("type") != "base64":
            return None
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        media_type = str(source.get("media_type") or "image/jpeg").strip()
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        return f"data:{media_type};base64,{data}"

    def _write_data_url_to_source(source: dict, data_url: str) -> dict:
        """Return a NEW source dict carrying the re-encoded payload.

        Copy-on-write: parts may be shared with the persistent history, so mutating
        in place would store the degraded image; the caller replaces the part.
        """
        header, _, data = data_url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            candidate = header[len("data:"):].split(";", 1)[0].strip()
            if candidate.startswith("image/"):
                media_type = candidate
        return {
            **source,
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Copy-on-write: part/source dicts can alias stored history, so build a new
        # content list and reassign msg["content"] on the per-call copy.
        new_content: list | None = None
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                source = part.get("source")
                url = _source_to_data_url(source)
                resized, unshrinkable = _shrink_data_url(url or "")
                if resized and isinstance(source, dict):
                    if new_content is None:
                        new_content = list(content)
                    new_content[part_idx] = {
                        **part,
                        "source": _write_data_url_to_source(source, resized),
                    }
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
                continue
            if ptype not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url")
            # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
            # OpenAI Responses: {"image_url": "data:..."}
            if isinstance(image_value, dict):
                url = image_value.get("url", "")
                resized, unshrinkable = _shrink_data_url(url)
                if resized:
                    if new_content is None:
                        new_content = list(content)
                    new_content[part_idx] = {
                        **part,
                        "image_url": {**image_value, "url": resized},
                    }
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
            elif isinstance(image_value, str):
                resized, unshrinkable = _shrink_data_url(image_value)
                if resized:
                    if new_content is None:
                        new_content = list(content)
                    new_content[part_idx] = {**part, "image_url": resized}
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
        if new_content is not None:
            msg["content"] = new_content

    if changed_count:
        logger.info(
            "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
            changed_count, target_bytes / (1024 * 1024),
        )
    if unshrinkable_oversized:
        # An unshrinkable oversized image makes retry pointless; signal no progress even
        # if others shrank so the caller surfaces the original error.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)",
            unshrinkable_oversized, target_bytes / (1024 * 1024),
        )
        return False
    return changed_count > 0


__all__ = [
    "COMPACTION_STATUS",
    "COMPACTION_DONE_STATUS",
    "COMPACTION_STATUS_MARKER",
    "is_compaction_progress_status",
    "check_compression_model_feasibility",
    "replay_compression_warning",
    "compress_context",
    "try_shrink_image_parts_in_messages",
]
