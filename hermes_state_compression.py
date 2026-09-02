"""Compression lineage, cooldown/streak counters, locks and turn leases for SessionDB.

Mixin bound onto ``SessionDB`` via the MRO, built on its ``_read_ctx`` /
``_execute_write`` / ``_write_sql`` / ``_read_one`` primitives."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

from hermes_state_common import _sql_session_last_active, is_automatic_end_reason

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


class SessionCompressionMixin:
    """Compression lineage, cooldown/streak counters, locks and turn leases."""

    def find_live_compression_child(
        self, parent_session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the unique live direct child of a compression-ended session.

        A stale agent whose parent was rotated elsewhere may recover only when the
        lineage names exactly one live continuation; more than one fails closed
        rather than guessing which transcript owns later messages."""
        if not parent_session_id:
            return None
        with self._read_ctx() as conn:
            parent = conn.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (parent_session_id,),
            ).fetchone()
            if (
                parent is None
                or parent["ended_at"] is None
                or parent["end_reason"] != "compression"
            ):
                return None
            rows = conn.execute(
                """
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
                WHERE s.parent_session_id = ?
                  AND s.ended_at IS NULL
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="s.")
                + """
                ORDER BY s.started_at ASC
                LIMIT 2
                """,
                (parent_session_id, parent_session_id, parent_session_id),
            ).fetchall()
        return self._session_row_dict(rows[0]) if len(rows) == 1 else None

    def reopen_orphaned_compression_session(self, session_id: str) -> bool:
        """Reopen a compression parent only when no continuation was published.

        Publication is atomic now, but older builds could leave a closed parent
        after an interrupted handoff. Conservative by design: an active lease or
        any canonical child means another path owns the lineage — fail closed."""
        if not session_id:
            return False

        def _do(conn):
            parent = conn.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if (
                parent is None
                or parent["ended_at"] is None
                or parent["end_reason"] != "compression"
            ):
                return False

            # Any non-branch/non-delegate/non-tool child is a continuation, ended
            # or not; reopening past it could give one lineage a second live head.
            child = conn.execute(
                """
                SELECT 1
                FROM sessions
                WHERE parent_session_id = ?
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="")
                + """
                LIMIT 1
                """,
                (session_id, session_id, session_id),
            ).fetchone()
            if child is not None:
                return False

            # refresh_compression_lock() lets an owner revive its own expired
            # row, so reclaim it inside this write transaction before reopening:
            # refresh-first makes the lease active and aborts recovery;
            # recovery-first deletes the holder so a later refresh can't resurrect it.
            now = time.time()
            lock_row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if lock_row is not None:
                expires_at = lock_row["expires_at"]
                if expires_at is None or float(expires_at) >= now:
                    return False
                deleted = conn.execute(
                    "DELETE FROM compression_locks "
                    "WHERE session_id = ? AND holder = ? AND expires_at = ?",
                    (session_id, lock_row["holder"], expires_at),
                )
                if deleted.rowcount != 1:
                    return False

            updated = conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ? AND ended_at IS NOT NULL "
                "AND end_reason = 'compression'",
                (session_id,),
            )
            # rowcount==1 is guaranteed by the parent SELECT in this same BEGIN
            # IMMEDIATE transaction. If a False return is ever added past this
            # point, raise instead: _execute_write commits the lease DELETE above unless _do raises.
            return updated.rowcount == 1

        return bool(self._execute_write(_do))

    def publish_compression_child(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        source: str,
        messages: List[Dict[str, Any]],
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        cwd: str = None,
        profile_name: str = None,
        compression_lock_holder: str = None,
        require_compression_lease: bool = True,
        require_lease_refresh: bool = False,
        lease_ttl_seconds: float = 300.0,
        watermark: Optional[int] = None,
        watermark_ceiling: Optional[int] = None,
    ) -> None:
        """Atomically close a parent and publish its durable compression child.

        Closure, child row, and handoff commit in one transaction: readers see
        the live parent or a complete child, never an ended parent with a
        missing/empty child.

        *watermark* (parent's ``get_active_message_watermark`` at compression
        start): parent rows with ``id > watermark`` — appends landed during the
        slow summary call — are column-cloned into the child AFTER the handoff
        so they survive rotation. *watermark_ceiling* bounds the clone: the
        rotation path flushes its OWN transcript to the parent just before
        publishing and those rows are already in the handoff, so the caller
        captures ``MAX(id)`` right BEFORE that flush and only
        ``(watermark, watermark_ceiling]`` is foreign tail. ``None`` = unbounded.

        *require_lease_refresh* + *compression_lock_holder* refreshes the lease
        on the same ``conn`` before the expiry check (no TOCTOU window), so a
        refresher that died on transient DB errors gets one last chance."""
        from hermes_state import CompressionSessionBusyError
        def _do(conn):
            if require_lease_refresh and compression_lock_holder:
                conn.execute(
                    "UPDATE compression_locks SET expires_at = ? "
                    "WHERE session_id = ? AND holder = ?",
                    (time.time() + lease_ttl_seconds, parent_session_id,
                     compression_lock_holder),
                )
            lock_row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                (parent_session_id,),
            ).fetchone()
            if require_compression_lease and (
                lock_row is None
                or not compression_lock_holder
                or lock_row["holder"] != compression_lock_holder
                or float(lock_row["expires_at"]) <= time.time()
            ):
                raise CompressionSessionBusyError(
                    f"Compression lease lost before publication: {parent_session_id}"
                )
            parent = conn.execute(
                """SELECT ended_at, end_reason, cwd, git_branch, git_repo_root,
                          user_id, session_key, chat_id, chat_type,
                          thread_id, display_name, origin_json, profile_name
                   FROM sessions WHERE id = ?""",
                (parent_session_id,),
            ).fetchone()
            if parent is None:
                raise RuntimeError(f"Compression parent not found: {parent_session_id}")
            if parent["ended_at"] is not None:
                # An ended stamp from AUTOMATIC cleanup (tui_shutdown, ws_disconnect,
                # orphan reap, idle/LRU evict) is stale by construction — this lease
                # holder is still continuing the conversation. Left alone it wedges
                # rotation forever (every attempt aborts here; each pre-publish flush
                # re-grows the parent until the provider rejects it). Clear it; the
                # closure UPDATE below re-stamps end_reason='compression'. Deliberate
                # boundaries (compression, session_reset, explicit close) still fail
                # closed — another path owns the lineage.
                if is_automatic_end_reason(parent["end_reason"]):
                    conn.execute(
                        "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                        "WHERE id = ?",
                        (parent_session_id,),
                    )
                else:
                    raise RuntimeError(
                        f"Compression parent already ended: {parent_session_id}"
                    )
            if not messages:
                raise RuntimeError("Compression child handoff must not be empty")
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)

            conn.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt,
                   system_prompt_hash,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, started_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    child_session_id,
                    source,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    cwd or parent["cwd"],
                    parent["git_branch"],
                    parent["git_repo_root"],
                    # Same contract as _insert_session_row's compression-fork backfill:
                    # child stays on the parent's profile and keeps gateway routing/
                    # origin columns so peer recovery works after a boundary crash. No
                    # owner on either side (legacy NULL parent) → stamp this store's
                    # profile so the child doesn't extend the unowned lineage.
                    profile_name
                    or parent["profile_name"]
                    or self._own_profile_name(),
                    parent["user_id"],
                    parent["session_key"],
                    parent["chat_id"],
                    parent["chat_type"],
                    parent["thread_id"],
                    parent["display_name"],
                    parent["origin_json"],
                    time.time(),
                ),
            )
            total_messages, total_tool_calls = self._insert_message_rows(
                conn, child_session_id, messages
            )
            if watermark is not None:
                # Clone the parent's concurrent tail (see docstring) into the
                # child after the handoff: column-exact except id/session_id;
                # originals stay in the closed parent for lineage recovery.
                _ceiling_clause = ""
                _params: list = [parent_session_id, int(watermark)]
                if watermark_ceiling is not None:
                    _ceiling_clause = " AND id <= ?"
                    _params.append(int(watermark_ceiling))
                tail_rows = conn.execute(
                    "SELECT id, tool_calls FROM messages "
                    "WHERE session_id = ? AND active = 1 AND id > ?"
                    f"{_ceiling_clause} ORDER BY id",
                    _params,
                ).fetchall()
                if tail_rows:
                    tail_ids = [int(r["id"]) for r in tail_rows]
                    placeholders = ",".join("?" for _ in tail_ids)
                    clone_cols = [
                        c for c in self._message_column_names(conn)
                        if c not in ("id", "session_id", "active", "compacted")
                    ]
                    col_list = ", ".join(clone_cols)
                    conn.execute(
                        f"INSERT INTO messages ({col_list}, session_id, active, compacted) "
                        f"SELECT {col_list}, ?, 1, 0 FROM messages "
                        f"WHERE id IN ({placeholders}) ORDER BY id",
                        [child_session_id, *tail_ids],
                    )
                    total_messages += len(tail_ids)
                    for r in tail_rows:
                        raw = r["tool_calls"]
                        if raw:
                            try:
                                parsed = json.loads(raw) if isinstance(raw, str) else raw
                                total_tool_calls += len(parsed) if isinstance(parsed, list) else 0
                            except (TypeError, ValueError):
                                pass
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, child_session_id),
            )
            updated = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'compression' "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), parent_session_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    f"Compression parent changed during publication: {parent_session_id}"
                )

        self._execute_write(_do)

    def record_compression_failure_cooldown(
        self,
        session_id: str,
        cooldown_until: float,
        error: Optional[str] = None,
    ) -> None:
        """Persist the active compression-failure cooldown for a session."""
        if not session_id:
            return

        try:
            # Merge-max with any longer live deadline so a later shorter write
            # can't reopen the thrash window; error always takes the latest diagnostic.
            self._write_sql(
                "UPDATE sessions SET compression_failure_cooldown_until = CASE "
                "WHEN compression_failure_cooldown_until IS NOT NULL "
                " AND compression_failure_cooldown_until > ? "
                "THEN compression_failure_cooldown_until ELSE ? END, "
                "compression_failure_error = ? WHERE id = ?",
                (cooldown_until, cooldown_until, error, session_id),
            )
        except sqlite3.Error as exc:
            logger.warning(
                "record_compression_failure_cooldown(%s) failed: %s",
                session_id, exc,
            )

    def get_compression_failure_cooldown(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the active compression-failure cooldown for ``session_id``."""
        if not session_id:
            return None
        now = time.time()
        row = self._read_one(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        )
        if row is None:
            return None
        cooldown_until = row[0]
        if cooldown_until is None:
            return None
        cooldown_until = float(cooldown_until)
        if cooldown_until <= now:
            return None
        error = row[1]
        return {
            "cooldown_until": cooldown_until,
            "remaining_seconds": cooldown_until - now,
            "error": error,
        }

    def get_compression_failure_cooldown_row(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Exact stored cooldown columns, no expiry filtering. Compression
        cancellation uses this under its session lease so rollback preserves an
        expired, partially-null, or absent row exactly instead of coercing it
        through the active-cooldown API."""
        if not session_id:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        row = self._read_one(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        )
        if row is None:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        cooldown_until = row[0]
        error = row[1]
        return {
            "session_exists": True,
            "cooldown_until": (
                float(cooldown_until) if cooldown_until is not None else None
            ),
            "error": error,
        }

    def restore_compression_failure_cooldown_row(
        self,
        session_id: str,
        snapshot: Dict[str, Any],
    ) -> None:
        """Restore and verify an exact cooldown-row snapshot. Unlike record/clear,
        this rollback API propagates write and verification failures: cancellation
        must not be reported mutation-free when compensation failed."""
        expected_exists = bool(snapshot.get("session_exists", False))
        if not expected_exists:
            actual = self.get_compression_failure_cooldown_row(session_id)
            if actual.get("session_exists", False):
                raise RuntimeError(
                    "cannot restore absent compression cooldown row: session now exists"
                )
            return

        deadline = snapshot.get("cooldown_until")
        error = snapshot.get("error")

        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (deadline, error, session_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"compression cooldown rollback session missing: {session_id}"
                )

        self._execute_write(_do)
        actual = self.get_compression_failure_cooldown_row(session_id)
        expected = {
            "session_exists": True,
            "cooldown_until": float(deadline) if deadline is not None else None,
            "error": error,
        }
        if actual != expected:
            raise RuntimeError(
                f"compression cooldown rollback verification failed: "
                f"expected={expected!r}, actual={actual!r}"
            )

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        """Clear any persisted compression-failure cooldown for a session."""
        if not session_id:
            return

        try:
            self._write_sql(
                "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
                "compression_failure_error = NULL WHERE id = ?",
                (session_id,),
            )
        except sqlite3.Error as exc:
            logger.warning(
                "clear_compression_failure_cooldown(%s) failed: %s",
                session_id, exc,
            )

    def _read_session_number(self, column: str, session_id: str, cast: type, zero: Any) -> Any:
        """Read one numeric ``sessions`` column clamped at ``zero``; a missing
        session, NULL, or unparsable value also reads as ``zero``."""
        if not session_id:
            return zero
        row = self._read_one(
            f"SELECT {column} FROM sessions WHERE id = ?", (session_id,)
        )
        if row is None:
            return zero
        try:
            return max(zero, cast(row[0] or zero))
        except (TypeError, ValueError):
            return zero

    def get_compression_fallback_streak(self, session_id: str) -> int:
        """Return the persisted deterministic-fallback streak."""
        return self._read_session_number("compression_fallback_streak", session_id, int, 0)

    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None:
        """Persist the deterministic-fallback streak for one session."""
        if session_id:
            self._write_sql(
                "UPDATE sessions SET compression_fallback_streak = ? WHERE id = ?",
                (max(0, int(streak)), session_id),
            )

    def get_compression_ineffective_count(self, session_id: str) -> int:
        """Persisted ineffective-compaction strike count: the durable half of
        the built-in compressor's anti-thrash guard, so a fresh compressor bound
        to a resumed session inherits an armed/tripped guard across restarts."""
        return self._read_session_number("compression_ineffective_count", session_id, int, 0)

    def set_compression_ineffective_count(self, session_id: str, count: int) -> None:
        """Persist the ineffective-compaction strike count for one session."""
        if session_id:
            self._write_sql(
                "UPDATE sessions SET compression_ineffective_count = ? WHERE id = ?",
                (max(0, int(count)), session_id),
            )

    def get_compression_recovery_deadline(self, session_id: str) -> float:
        """Persisted anti-thrash recovery deadline (epoch; ``0.0`` = not armed).
        Durable because the gateway rebuilds the compressor every turn / cache
        eviction: a process-local deadline restarted on each rebuild, so a
        tripped session never earned its probe."""
        return self._read_session_number("compression_recovery_deadline", session_id, float, 0.0)

    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None:
        """Persist the anti-thrash recovery deadline; ``0`` / ``None`` disarms it."""
        if not session_id:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        stored = normalized if normalized > 0.0 else None

        self._write_sql(
            "UPDATE sessions SET compression_recovery_deadline = ? WHERE id = ?",
            (stored, session_id),
        )

    def refresh_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Extend the compression lock lease if ``holder`` still owns it.

        Ownership is decided by ``holder`` alone, deliberately NOT ``expires_at``:
        a live owner whose refresher stalled past its TTL (GC pause, loaded CI
        runner, slow write escaping ``_execute_write``'s retry budget) must be
        able to revive its still-unclaimed row. Requiring ``expires_at >= now``
        made such a stall permanent — every later refresh matched 0 rows and the
        owner kept compressing/rotating with no lease, exactly the window in
        which a competing path can fork the lineage.

        It cannot resurrect a lock someone else took: SQLite serialises writes,
        so :meth:`try_acquire_compression_lock`'s reclaim (DELETE-expired +
        INSERT-or-IGNORE) never interleaves with this UPDATE. Reclaim-first
        replaces ``holder`` and this matches nothing; refresh-first pushes
        ``expires_at`` forward and the reclaimer's DELETE matches nothing."""
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        try:
            return self._write_rowcount(
                "UPDATE compression_locks SET expires_at = ? "
                "WHERE session_id = ? AND holder = ?",
                (expires_at, session_id, holder),
            ) > 0
        except sqlite3.Error as exc:
            logger.warning(
                "refresh_compression_lock(%s) failed: %s",
                session_id, exc,
            )
            return False

    def try_acquire_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Try to atomically acquire the compression lock for ``session_id``.

        ``True``: caller owns the lock and must :meth:`release_compression_lock`.
        ``False``: another holder owns a live lock and the caller MUST NOT
        compress — its rotation would race the holder's and split the lineage.
        Expired locks and structured holders whose local ``pid=`` is dead are
        reclaimed transparently, so a gateway killed mid-compression doesn't
        stall its replacement for the full TTL. Single-transaction DELETE-expired
        + INSERT-or-IGNORE + SELECT-to-confirm; SQLite serialises writes, so it's atomic."""
        from hermes_state import _compression_lock_holder_process_is_dead
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        def _do(conn):
            reclaimed_holder = None
            row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                current_holder = (
                    row[0]
                )
                current_expires_at = (
                    row[1]
                )
                if (
                    current_expires_at < now
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    conn.execute(
                        "DELETE FROM compression_locks "
                        "WHERE session_id = ? AND holder = ?",
                        (session_id, current_holder),
                    )
                    reclaimed_holder = current_holder
            # INSERT OR IGNORE gives no rowcount signal — verify ownership via SELECT.
            conn.execute(
                "INSERT OR IGNORE INTO compression_locks "
                "(session_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, holder, now, expires_at),
            )
            row = conn.execute(
                "SELECT holder FROM compression_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            acquired = row is not None and (
                row[0]
            ) == holder
            return acquired, reclaimed_holder

        try:
            acquired, reclaimed_holder = self._execute_write(_do)
            if reclaimed_holder:
                logger.warning(
                    "Reclaimed stale compression lock for session=%s "
                    "(holder=%s)",
                    session_id,
                    reclaimed_holder,
                )
            return bool(acquired)
        except sqlite3.Error as exc:
            logger.warning(
                "try_acquire_compression_lock(%s) failed: %s",
                session_id, exc,
            )
            # False makes the caller skip compression — the safe behaviour
            # when the lock subsystem is broken.
            return False

    def release_compression_lock(self, session_id: str, holder: str) -> None:
        """Release the compression lock for ``session_id`` iff we own it. Idempotent
        when the lock is gone or reclaimed; the ``holder`` check stops a late
        compressor clobbering someone else's fresh lock."""
        if not session_id:
            return

        try:
            self._write_sql(
                "DELETE FROM compression_locks "
                "WHERE session_id = ? AND holder = ?",
                (session_id, holder),
            )
        except sqlite3.Error as exc:
            logger.warning(
                "release_compression_lock(%s) failed: %s",
                session_id, exc,
            )

    def _session_turn_lease_key_on_conn(self, conn, session_id: str) -> str:
        """Walk compression parents on ``conn`` to the conversation lease key.

        Must share the connection of the lease INSERT/UPDATE/DELETE: a failed
        ``get_session`` must not yield a child id the write then persists
        (refresh would walk to the parent and fail-close). Markers bind to
        ``parent_session_id`` (as in ``_NON_CONTINUATION_CHILD_FILTER_SQL``).
        Lock errors propagate so ``_execute_write`` / ``acquire_session_turn_lease`` can retry."""
        if not session_id:
            return session_id

        def _row(sid: str):
            row = conn.execute(
                "SELECT id, parent_session_id, source, model_config, end_reason "
                "FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            return dict(row) if row else None

        current = _row(session_id)
        seen = {session_id}
        while current:
            parent_id = current.get("parent_session_id")
            if (
                not parent_id
                or parent_id in seen
                or self._is_explicit_fork_child_row(current)
            ):
                break
            parent = _row(parent_id)
            if not parent or parent.get("end_reason") != "compression":
                break
            seen.add(parent_id)
            current = parent
        return str(current.get("id") or session_id) if current else session_id

    def _session_turn_lease_key(self, session_id: str) -> str:
        """Return the stable serialization key for every compression segment.

        Acquire/refresh/release resolve this inside their write transaction; this
        is for tests/diagnostics. It does not swallow lock errors — a swallowed
        walk plus a later successful write was the fail-open that replayed the
        post-rotation refresh miss."""
        if not session_id:
            return session_id
        with self._read_ctx() as conn:
            return self._session_turn_lease_key_on_conn(conn, session_id)

    def try_acquire_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
        patience_s: Optional[float] = None,
    ) -> bool:
        """Atomically acquire the cross-process turn lease for a conversation.
        Compression rotates a session into child segments, so the durable key is
        the lineage root, not the current segment id. The walk, the INSERT, and
        reclaim of expired or dead-local-PID leases share one write transaction."""
        from hermes_state import _compression_lock_holder_process_is_dead
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))

        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            row = conn.execute(
                "SELECT holder, expires_at FROM session_turn_leases "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is not None:
                current_holder = row["holder"]
                if (
                    float(row["expires_at"]) <= now
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    conn.execute(
                        "DELETE FROM session_turn_leases "
                        "WHERE conversation_id = ? AND holder = ?",
                        (conversation_id, current_holder),
                    )
            conn.execute(
                "INSERT OR IGNORE INTO session_turn_leases "
                "(conversation_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, holder, now, expires_at),
            )
            owner = conn.execute(
                "SELECT holder FROM session_turn_leases WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return owner is not None and owner["holder"] == holder

        return bool(self._execute_write(_do, patience_s=patience_s))

    def acquire_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
        wait_seconds: float = 1800.0,
        poll_interval_seconds: float = 1.0,
        on_wait=None,
        wait_notice_interval_seconds: float = 15.0,
        should_abort=None,
        acquire_patience_s: float = 0.5,
    ) -> bool:
        """Wait for a cross-process turn lease without holding a SQLite lock.

        ``on_wait(elapsed)`` is best-effort: called when the first attempt fails
        (elapsed ~0) and about every ``wait_notice_interval_seconds`` after, so
        UIs can show another process holds the conversation. ``should_abort()``
        True (e.g. ``/stop``) returns False at once, not after ``wait_seconds``."""
        from hermes_state import classify_persistence_error
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        wait_started = None
        last_notice_at = None
        notice_every = max(0.0, float(wait_notice_interval_seconds))
        while True:
            if should_abort is not None:
                try:
                    if should_abort():
                        return False
                except Exception:
                    logger.debug(
                        "session turn lease should_abort callback failed",
                        exc_info=True,
                    )
            try:
                if self.try_acquire_session_turn_lease(
                    session_id,
                    holder,
                    ttl_seconds=ttl_seconds,
                    patience_s=acquire_patience_s,
                ):
                    return True
            except sqlite3.Error as exc:
                # Long holder transactions (compression publish, large flushes)
                # can exhaust one write-patience budget; keep polling until
                # wait_seconds or should_abort.
                if classify_persistence_error(exc) != "locked":
                    raise
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False
            if wait_started is None:
                wait_started = now
            if on_wait is not None and (
                last_notice_at is None
                or notice_every == 0.0
                or (now - last_notice_at) >= notice_every
            ):
                try:
                    on_wait(max(0.0, now - wait_started))
                except Exception:
                    logger.debug(
                        "session turn lease on_wait callback failed",
                        exc_info=True,
                    )
                last_notice_at = now
            time.sleep(min(max(0.01, float(poll_interval_seconds)), remaining))

    def refresh_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Extend a turn lease only while ``holder`` still owns it."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + max(0.1, float(ttl_seconds))

        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            cursor = conn.execute(
                "UPDATE session_turn_leases SET expires_at = ? "
                "WHERE conversation_id = ? AND holder = ?",
                (expires_at, conversation_id, holder),
            )
            return cursor.rowcount > 0

        return bool(self._execute_write(_do))

    def release_session_turn_lease(self, session_id: str, holder: str) -> None:
        """Release a turn lease iff ``holder`` still owns it; idempotent."""
        if not session_id or not holder:
            return

        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            conn.execute(
                "DELETE FROM session_turn_leases "
                "WHERE conversation_id = ? AND holder = ?",
                (conversation_id, holder),
            )

        self._execute_write(_do)

    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Return the current (non-expired) holder for ``session_id``, or None.
        Diagnostic only — not part of the locking protocol."""
        if not session_id:
            return None
        now = time.time()
        row = self._read_one(
            "SELECT holder FROM compression_locks "
            "WHERE session_id = ? AND expires_at >= ?",
            (session_id, now),
        )
        if row is None:
            return None
        return row[0]

    def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuations (parent ended by compression;
        child has messages, no end_reason/ended_at, api_call_count=0) as
        ``orphaned_compression``. Non-destructive: messages are preserved."""
        cutoff = time.time() - 604800  # 7 days

        def _do(conn):
            now = time.time()
            result = conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?,
                    end_reason = 'orphaned_compression'
                WHERE api_call_count = 0
                  AND end_reason IS NULL
                  AND ended_at IS NULL
                  AND started_at < ?
                  AND parent_session_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM sessions p
                      WHERE p.id = sessions.parent_session_id
                        AND p.end_reason = 'compression'
                        AND p.ended_at IS NOT NULL
                  )
                  AND EXISTS (
                      SELECT 1 FROM messages m
                      WHERE m.session_id = sessions.id
                  )
                """,
                (now, cutoff),
            )
            return result.rowcount

        return self._execute_write(_do) or 0

    def get_compression_chain(self, session_id: str) -> List[str]:
        """Walk the compression-continuation chain forward and return every id.

        Root-first, ending at the tip; ``[session_id]`` when no continuation
        exists. ``get_compression_tip`` is this walk's last element — one
        implementation so the two can never disagree.

        A continuation is a child of a session with ``end_reason='compression'``.
        Older builds also required ``child.started_at >= parent.ended_at``;
        too brittle — gateway + compression races can insert the real
        continuation before the parent's ``ended_at`` is written while a stale
        websocket later creates a sibling that passes the timestamp test, so
        desktop resume followed the sibling and recent messages looked "lost".
        Instead: follow only children of compression-ended parents, exclude
        explicit branch/delegate/tool children, and prefer children that continue
        the chain (``end_reason='compression'``) or are still live over stale
        closed siblings such as ``ws_orphan_reap``."""
        current = session_id
        chain = [current] if current else []
        seen = {current} if current else set()
        # Defensive bound; chains this deep are pathological.
        for _ in range(100):
            with self._read_ctx() as conn:
                cursor = conn.execute(
                    f"""
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = ?
                      AND parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                    ORDER BY
                      CASE
                        WHEN child.end_reason = 'compression' THEN 0
                        WHEN child.ended_at IS NULL THEN 1
                        ELSE 2
                      END,
                      {_sql_session_last_active("child")} DESC,
                      child.started_at DESC,
                      child.id DESC
                    LIMIT 1
                    """,
                    (current,),
                )
                row = cursor.fetchone()
            if row is None:
                return chain
            child_id = row["id"]
            if not child_id or child_id in seen:
                return chain
            seen.add(child_id)
            current = child_id
            chain.append(child_id)
        return chain

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Live tip of a compression chain (walk semantics: ``get_compression_chain``);
        the input id when no continuation exists."""
        chain = self.get_compression_chain(session_id)
        return chain[-1] if chain else session_id

    def _is_compression_child_row(self, child: Dict[str, Any]) -> bool:
        parent_id = child.get("parent_session_id")
        if not parent_id or self._is_explicit_fork_child_row(child):
            return False
        parent = self.get_session(parent_id)
        return bool(parent and parent.get("end_reason") == "compression")

    def get_compression_lineage(self, session_id: str) -> List[str]:
        """Return compression ancestors through tip in chronological order."""
        session = self.get_session(session_id)
        if not session or self._is_explicit_fork_child_row(session):
            return [session_id] if session else []

        root = session
        ancestors = {root["id"]}
        while self._is_compression_child_row(root):
            parent = self.get_session(root["parent_session_id"])
            if not parent or parent["id"] in ancestors:
                break
            root = parent
            ancestors.add(root["id"])

        lineage = [root["id"]]
        seen = {root["id"]}
        current = root
        while current.get("end_reason") == "compression":
            rows = self._read_all(
                """
                SELECT * FROM sessions
                WHERE parent_session_id = ?
                ORDER BY started_at ASC
                """,
                (current["id"],),
            )
            next_child = None
            for row in rows:
                candidate = dict(row)
                if self._is_compression_child_row(candidate):
                    next_child = candidate
                    break
            if not next_child or next_child["id"] in seen:
                break
            lineage.append(next_child["id"])
            seen.add(next_child["id"])
            current = next_child
            if current["id"] == session_id:
                # Later tips are included only when the requested session itself was compacted.
                continue
        return lineage if session_id in lineage else [session_id]
