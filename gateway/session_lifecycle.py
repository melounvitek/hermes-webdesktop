"""SessionStore reset/expiry policy and crash-recovery markers: idle/daily reset
evaluation, expiry finalization, active-turn tokens, resume_pending,
suspension and pruning.

Mixin split out of ``gateway/session.py``; bound onto ``SessionStore`` via the MRO.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")


class SessionLifecycleMixin:
    """SessionStore reset/expiry policy and crash-recovery markers: idle/daily
    reset evaluation, expiry finalization, active-turn tokens, resume_pending,
    suspension and pruning.
    """

    def set_expiry_finalized(
        self, entry: SessionEntry, *, clear_model_override: bool = True
    ) -> None:
        """Mark a session entry expiry-finalized in memory, sessions.json, AND state.db.

        Single write-path for the expiry watcher so the durable flag survives
        sessions.json loss. ``clear_model_override=False`` = flag only.
        """
        with self._lock:
            entry.expiry_finalized = True
            if clear_model_override:
                # Finalization is a conversation boundary: drop the persisted
                # /model override so a later message cannot rehydrate it.
                entry.model_override = None
            self._save()
        # Background caller never entered ``_profile_runtime_scope``: resolve
        # the store from the key, not the ambient scope.
        _db = self._db_for_key(entry.session_key)
        if _db:
            setter = getattr(_db, "set_expiry_finalized", None)
            if callable(setter):
                try:
                    setter(entry.session_id, True)
                except Exception as exc:
                    logger.debug("Session DB expiry_finalized write failed for %s: %s", entry.session_id, exc)
            try:
                # Without a durable ``session_reset`` end_reason, later agent
                # cleanup ends the row as ``agent_close``, which stale-route
                # recovery treats as resumable. Promotion only upgrades live/
                # agent_close rows; explicit boundaries are preserved.
                _db.promote_to_session_reset(entry.session_id)
            except Exception as exc:
                logger.debug("Session DB promote_to_session_reset failed for %s: %s", entry.session_id, exc)

    @staticmethod
    def _policy_reset_reason(policy, updated_at: datetime) -> Optional[str]:
        """Return "idle"/"daily" when *updated_at* is overdue under *policy*, else None."""
        from gateway.session import _now
        if policy.mode == "none":
            return None
        now = _now()
        if policy.mode in {"idle", "both"} and now > updated_at + timedelta(minutes=policy.idle_minutes):
            return "idle"
        if policy.mode in {"daily", "both"}:
            today_reset = now.replace(
                hour=policy.at_hour, minute=0, second=0, microsecond=0,
            )
            if now.hour < policy.at_hour:
                today_reset -= timedelta(days=1)
            if updated_at < today_reset:
                return "daily"
        return None

    def _is_session_expired(self, entry: SessionEntry) -> bool:
        """Whether the entry's reset policy has expired it (entry alone, no source).

        Used by the background expiry watcher. Sessions with active
        background processes are never considered expired.
        """
        if self._has_active_processes_safe(entry.session_key, context="expiry"):
            logger.debug("Session %s not expired — active background processes", entry.session_key)
            return False
        policy = self.config.get_reset_policy(
            platform=entry.platform,
            session_type=entry.chat_type,
        )
        return self._policy_reset_reason(policy, entry.updated_at) is not None

    def is_session_finalizable(self, entry: SessionEntry) -> bool:
        """True if the expiry watcher will *ever* finalize this session.

        A ``mode == "none"`` session never expires, so the agent-cache idle
        sweep must reap its agent itself instead of deferring to the watcher
        (deferring would pin the agent for the gateway's lifetime). Policy
        resolution errors count as "not finalizable" (sweep reaps — safe).
        """
        try:
            policy = self.config.get_reset_policy(
                platform=entry.platform,
                session_type=entry.chat_type,
            )
            return policy.mode != "none"
        except Exception:
            return False

    def _is_session_ended_in_db(self, session_id: str) -> bool:
        """True iff state.db has this session with a non-null end_reason.

        Same staleness test as ``_prune_stale_sessions_locked`` (no DB, no
        row, or DB error -> False, keep). Used by ``get_or_create_session``
        to self-heal at routing time, since the startup prune cannot see a
        session ended while the gateway stays alive. Store resolved from the
        row's owning profile, not the ambient scope.
        """
        db = self._db_for_session_id(session_id)
        if not db or not session_id:
            return False
        try:
            row = db.get_session(session_id)
        except Exception:
            return False
        return bool(row is not None and row.get("end_reason") is not None)

    def _should_reset(self, entry: SessionEntry, source: SessionSource) -> Optional[str]:
        """Return the reset reason ("idle"/"daily") if policy says reset, else None.

        Sessions with active background processes are never reset.
        """
        session_key = self._generate_session_key(source)
        if self._has_active_processes_safe(session_key, context="reset"):
            logger.debug("Session reset skipped for %s — active background processes", session_key)
            return None
        policy = self.config.get_reset_policy(
            platform=source.platform,
            session_type=source.chat_type
        )
        return self._policy_reset_reason(policy, entry.updated_at)

    def _route_reset_reason(
        self, entry: SessionEntry, source: SessionSource, now: datetime
    ) -> Optional[str]:
        """Reset decision for an existing route (no lock; DB/config I/O).

        ``suspended`` always resets. Otherwise the reset policy decides; a
        still-pending resume marker is additionally freshness-gated — but
        ``session_reset.mode: none`` (user opted out of ALL automatic resets)
        makes an expired marker fall through to a normal resume, never a
        silent fresh session.
        """
        from gateway.session import auto_continue_freshness_window
        if entry.suspended:
            return "suspended"
        reason = self._should_reset(entry, source)
        if reason or not entry.resume_pending:
            return reason
        policy = self.config.get_reset_policy(
            platform=source.platform, session_type=source.chat_type,
        )
        if policy.mode == "none":
            return None
        window = auto_continue_freshness_window()
        ref_time = entry.last_resume_marked_at or entry.updated_at
        if window > 0 and (now - ref_time).total_seconds() > window:
            return "resume_pending_expired"
        return None

    def _update_entry(self, session_key: str, mutate) -> bool:
        """Apply ``mutate(entry)`` under ``_lock`` and full-save; False when the
        entry is missing or *mutate* returned False (nothing to persist)."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or mutate(entry) is False:
                return False
            self._save()
            return True

    def suspend_session(self, session_key: str) -> bool:
        """Mark a session suspended so it auto-resets on next access (/stop).
        Returns True if the session existed."""
        return self._update_entry(session_key, lambda e: setattr(e, "suspended", True))

    def mark_turn_active(self, session_key: str) -> Optional[str]:
        """Persist exact ownership of the agent turn running for *session_key*.

        The opaque token is returned to the caller and must be supplied to
        :meth:`clear_turn_active`.  Re-marking replaces the previous token so
        a stale asynchronous unwind cannot clear a newer turn.
        """
        from gateway.session import _now
        token = uuid.uuid4().hex
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            now = _now()
            candidate = entry.to_dict()
            candidate["active_turn_token"] = token
            candidate["active_turn_started_at"] = now.isoformat()
            # Keeps the legacy 120s startup heuristic working for an older
            # binary during a rolling downgrade/upgrade window.
            candidate["updated_at"] = now.isoformat()

            # Persist before publishing in memory so a failed write cannot
            # leak an unowned token through a later unrelated save.
            self._save_entry(session_key, entry_data=candidate, lock_held=True)
            entry.active_turn_token = token
            entry.active_turn_started_at = now
            entry.updated_at = now
        return token

    def clear_turn_active(self, session_key: str, token: str) -> bool:
        """Compare-and-swap clear an active-turn marker.

        Returns ``False`` when the entry disappeared or a newer turn owns it.
        """
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or entry.active_turn_token != token:
                return False
            candidate = entry.to_dict()
            candidate["active_turn_token"] = None
            candidate["active_turn_started_at"] = None

            # Keep the live token until the clear is durable (retryable).
            self._save_entry(session_key, entry_data=candidate, lock_held=True)
            entry.active_turn_token = None
            entry.active_turn_started_at = None
        return True

    def recover_interrupted_turns(
        self,
        max_age_seconds: int = 60 * 60,
    ) -> int:
        """Promote crash-left turn markers into ``resume_pending`` (unclean startup only).

        Old/invalid markers are cleared without resuming; suspended sessions
        are never re-armed. Returns the number of newly promoted sessions.
        """
        from gateway.session import _now
        now = _now()
        max_age = timedelta(seconds=max(0, max_age_seconds))
        promoted = 0
        changed = False

        with self._lock:
            self._ensure_loaded_locked()
            for entry in self._entries.values():
                if not entry.active_turn_token:
                    continue

                started_at = entry.active_turn_started_at
                try:
                    marker_is_stale = (
                        started_at is None
                        or (max_age_seconds > 0 and now - started_at > max_age)
                    )
                except TypeError:
                    # Mixed aware/naive timestamps: clear rather than risk an
                    # unsafe old resume.
                    marker_is_stale = True

                if not marker_is_stale and not entry.suspended:
                    if entry.resume_pending:
                        # A drain-timeout marker is more specific; keep it.
                        if entry.last_resume_marked_at is None:
                            entry.last_resume_marked_at = now
                    else:
                        entry.resume_pending = True
                        entry.resume_reason = "restart_interrupted"
                        # Freshness starts at discovery, not turn start.
                        entry.last_resume_marked_at = now
                        promoted += 1

                entry.active_turn_token = None
                entry.active_turn_started_at = None
                changed = True

            if changed:
                self._save()

        return promoted

    def discard_active_turn_markers(self) -> int:
        """Clear orphan turn markers after a verified clean shutdown."""
        cleared = 0
        with self._lock:
            self._ensure_loaded_locked()
            for entry in self._entries.values():
                if not entry.active_turn_token and entry.active_turn_started_at is None:
                    continue
                entry.active_turn_token = None
                entry.active_turn_started_at = None
                cleared += 1
            if cleared:
                self._save()
        return cleared

    def mark_resume_pending(self, session_key: str, reason: str = "restart_timeout") -> bool:
        """Mark a session resumable after a restart interruption (keeps the
        session_id/transcript, unlike ``suspend_session``). True if marked."""
        from gateway.session import _now
        def _apply(entry: SessionEntry):
            # Never override an explicit ``suspended`` (hard forced-wipe).
            if entry.suspended:
                return False
            entry.resume_pending = True
            entry.resume_reason = reason
            entry.last_resume_marked_at = _now()

        return self._update_entry(session_key, _apply)

    def clear_resume_pending(self, session_key: str) -> bool:
        """Clear the resume-pending flag after a successful resumed turn.
        Returns True if a flag was cleared."""
        def _apply(entry: SessionEntry):
            if not entry.resume_pending:
                return False
            entry.resume_pending = False
            entry.resume_reason = None
            entry.last_resume_marked_at = None

        return self._update_entry(session_key, _apply)

    def prune_old_entries(self, max_age_days: int) -> int:
        """Drop routing entries idle (by ``updated_at``) for more than max_age_days.

        Suspended entries and entries with active background processes are
        kept. The SQLite transcript stays; only the key -> session_id mapping
        is dropped. ``max_age_days <= 0`` disables. Returns the count removed.
        """
        from gateway.session import _now
        if max_age_days is None or max_age_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=max_age_days)
        removed_keys: list[str] = []

        with self._lock:
            self._ensure_loaded_locked()
            for key, entry in list(self._entries.items()):
                if entry.suspended:
                    continue
                # The callback is keyed by session_key, NOT session_id.
                if self._has_active_processes_safe(entry.session_key, context="prune"):
                    continue
                if entry.updated_at < cutoff:
                    removed_keys.append(key)
            for key in removed_keys:
                self._entries.pop(key, None)
            if removed_keys:
                self._save()

        if removed_keys:
            logger.info(
                "SessionStore pruned %d entries older than %d days",
                len(removed_keys), max_age_days,
            )
        return len(removed_keys)

    def suspend_recently_active(self, max_age_seconds: int = 120) -> int:
        """Mark sessions active within *max_age_seconds* as ``resume_pending``
        after a crash/fast restart (already-pending and suspended entries are
        skipped). Returns the number marked."""
        from gateway.session import _now
        cutoff = _now() - timedelta(seconds=max_age_seconds)
        count = 0
        with self._lock:
            self._ensure_loaded_locked()
            for entry in self._entries.values():
                if entry.resume_pending:
                    continue
                if not entry.suspended and entry.updated_at >= cutoff:
                    entry.resume_pending = True
                    entry.resume_reason = "restart_interrupted"
                    entry.last_resume_marked_at = _now()
                    count += 1
            if count:
                self._save()
        return count
