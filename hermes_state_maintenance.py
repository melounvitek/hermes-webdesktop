"""Retention pruning, stale-session archiving and VACUUM policy mixin for
SessionDB."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_state_common import (
    AUTO_VACUUM_MIN_FREELIST_RATIO,
    _sql_session_last_active,
    escape_like as _escape_like,
)

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

_LAST_ACTIVE_SQL = """COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m
                        WHERE m.session_id = s.id),
                       s.started_at
                   )"""
_TOKENS_SQL = "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0))"
_COST_SQL = "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0)"


def _like(value: str) -> str:
    return f"%{_escape_like(value.lower())}%"


def _cwd_prefix_filter(value: str) -> Tuple[List[str], list]:
    from hermes_state import _cwd_prefix_clause
    clause, params = _cwd_prefix_clause(value)
    return [clause], list(params)


def _one(clause: str, conv=None):
    return lambda v: ([clause], [conv(v) if conv else v])


# Prune/archive filters in evaluation order: (kwarg, applies-when, builder).
# ``applies-when`` is "notnone" (numeric/time bounds; 0 is a real bound) or
# "truthy" (strings; "" means unset).  Builders return (clauses, params).
_PRUNE_FILTERS = (
    # Orphan-swept rows age from the sweep, not their old activity, or the
    # next prune pass deletes them before the user can recover.
    ("last_active_before", "notnone", lambda v: (
        [_LAST_ACTIVE_SQL + " < ?",
         "(COALESCE(s.end_reason, '') != 'startup_orphan_reap' OR s.ended_at < ?)"],
        [v, v])),
    ("last_active_after", "notnone", _one(_LAST_ACTIVE_SQL + " >= ?")),
    ("started_before", "notnone", _one("s.started_at < ?")),
    ("started_after", "notnone", _one("s.started_at >= ?")),
    ("source", "truthy", _one("s.source = ?")),
    ("title_like", "truthy", _one("LOWER(COALESCE(s.title, '')) LIKE ? ESCAPE '\\'", _like)),
    ("end_reason", "truthy", _one("s.end_reason = ?")),
    ("cwd_prefix", "truthy", _cwd_prefix_filter),
    ("min_messages", "notnone", _one("s.message_count >= ?")),
    ("max_messages", "notnone", _one("s.message_count <= ?")),
    ("model_like", "truthy", _one("LOWER(COALESCE(s.model, '')) LIKE ? ESCAPE '\\'", _like)),
    ("provider", "truthy", _one("LOWER(COALESCE(s.billing_provider, '')) = ?", str.lower)),
    ("user_id", "truthy", _one("s.user_id = ?")),
    ("chat_id", "truthy", _one("s.chat_id = ?")),
    ("chat_type", "truthy", _one("s.chat_type = ?")),
    ("branch_like", "truthy", _one("LOWER(COALESCE(s.git_branch, '')) LIKE ? ESCAPE '\\'", _like)),
    ("min_tokens", "notnone", _one(_TOKENS_SQL + " >= ?")),
    ("max_tokens", "notnone", _one(_TOKENS_SQL + " <= ?")),
    ("min_cost", "notnone", _one(_COST_SQL + " >= ?")),
    ("max_cost", "notnone", _one(_COST_SQL + " <= ?")),
    ("min_tool_calls", "notnone", _one("COALESCE(s.tool_call_count, 0) >= ?")),
    ("max_tool_calls", "notnone", _one("COALESCE(s.tool_call_count, 0) <= ?")),
)
_PRUNE_FILTER_NAMES = frozenset(name for name, _, _ in _PRUNE_FILTERS) | {"archived", "include_pinned"}


class SessionMaintenanceMixin:
    """Retention pruning, stale-session archiving and VACUUM policy for SessionDB."""

    def prune_empty_ghost_sessions(self, sessions_dir: "Optional[Path]" = None) -> int:
        """Remove empty TUI ghost sessions (no messages, no title, >24hr old)."""
        cutoff = time.time() - 86400

        def _do(conn):
            rows = conn.execute("""
                SELECT id FROM sessions
                WHERE source = 'tui'
                  AND title IS NULL
                  AND ended_at IS NOT NULL
                  AND started_at < ?
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
            """, (cutoff,)).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", ids)
                self._delete_unreferenced_system_prompts(conn)
            return ids

        removed_ids = self._execute_write(_do) or []
        if sessions_dir and removed_ids:
            for sid in removed_ids:
                self._remove_session_files(sessions_dir, sid)
        return len(removed_ids)

    def sweep_orphaned_sessions(
        self, *, max_idle_seconds: float,
        sources: Tuple[str, ...] = ("tui", "desktop", "subagent"),
        exclude_ids: Tuple[str, ...] = (), exclude_pinned: bool = False,
        heartbeat_staleness_seconds: Optional[float] = None,
        heartbeat_ownership_grace_seconds: Optional[float] = None,
        respect_gateway_heartbeats: bool = True,
    ) -> List[str]:
        """Close session rows orphaned by a dead gateway process.

        The TUI/desktop gateway reaps disconnected sessions with an in-process
        grace timer; a restart destroys the timer and leaves ``ended_at IS
        NULL`` forever.  Closes rows for ``sources`` whose ``started_at`` AND
        canonical last activity are both older than ``max_idle_seconds`` with
        ``end_reason='startup_orphan_reap'`` (the separate ``started_at``
        predicate protects fresh compression/branch children whose copied
        activity is old).  Only pass sources whose lifecycle the caller owns —
        never messaging platforms like ``telegram`` (ending those triggers a
        routing loop).  ``exclude_ids`` spares rows this process still holds in
        memory.  Non-destructive: messages are kept and the row stays
        resumable; first-reason-wins via ``ended_at IS NULL``.

        Cross-backend liveness: with ``respect_gateway_heartbeats``, a row is
        reaped only when stale AND no live backend (heartbeat within
        ``heartbeat_staleness_seconds``, default ``2 * max_idle_seconds``) could
        own it, where backend B owns session S if ``B.started_at <= S.started_at
        + heartbeat_ownership_grace_seconds`` (default = staleness).  The grace
        covers a migrating backend whose sessions predate its first heartbeat
        but is bounded so a PID-reuse respawn cannot protect rows forever.
        Disable the gate only for sources owned by state.db itself.

        SELECT, live-lease validation and UPDATE run in one ``BEGIN IMMEDIATE``
        transaction; active turn leases / compression locks spare the row, and
        expired guards are removed so their former owner is fenced.
        """
        from hermes_state import SessionCompressionInProgressError, SessionTurnLeaseLostError
        srcs = tuple(s for s in sources if s)
        if max_idle_seconds <= 0 or not srcs:
            return []
        hb_staleness = (
            heartbeat_staleness_seconds
            if heartbeat_staleness_seconds and heartbeat_staleness_seconds > 0
            else max_idle_seconds * 2
        )
        hb_grace = (
            heartbeat_ownership_grace_seconds
            if heartbeat_ownership_grace_seconds is not None and heartbeat_ownership_grace_seconds >= 0
            else hb_staleness
        )
        now = time.time()
        cutoff = now - max_idle_seconds
        placeholders = ",".join("?" for _ in srcs)
        pin_scope = " AND COALESCE(pinned, 0) = 0" if exclude_pinned else ""
        orphan_predicate = f"started_at < ? AND {_sql_session_last_active('sessions')} < ?"
        heartbeat_params: Tuple[float, ...] = ()
        if respect_gateway_heartbeats:
            orphan_predicate += (
                " AND NOT EXISTS ("
                "SELECT 1 FROM gateway_heartbeats h"
                " WHERE h.last_heartbeat >= ?"
                " AND h.started_at <= sessions.started_at + ?"
                ")"
            )
            heartbeat_params = (now - hb_staleness, hb_grace)
        scope_sql = f" AND source IN ({placeholders}){pin_scope} AND {orphan_predicate}"
        scope_params = (*srcs, cutoff, cutoff, *heartbeat_params)

        def _do(conn):
            rows = conn.execute(
                f"SELECT id FROM sessions WHERE ended_at IS NULL{scope_sql}", scope_params
            ).fetchall()
            excluded = {str(x) for x in exclude_ids if x}
            victims = []
            for row in rows:
                sid = str(row["id"])
                if sid in excluded:
                    continue
                try:
                    self._check_transcript_write_guards(
                        conn, sid, compression_lock_holder=None, turn_lease_holder=None,
                        reject_active_turn_lease=True, reject_active_compression_lock=True,
                    )
                except (SessionCompressionInProgressError, SessionTurnLeaseLostError):
                    continue
                victims.append(sid)
            if not victims:
                return []
            marks = ",".join("?" for _ in victims)
            # Re-apply every predicate under the write lock.
            conn.execute(
                f"UPDATE sessions SET ended_at = ?, end_reason = 'startup_orphan_reap'"
                f" WHERE id IN ({marks}) AND ended_at IS NULL{scope_sql}",
                (time.time(), *victims, *scope_params),
            )
            return victims

        return self._execute_write(_do) or []

    @staticmethod
    def _prune_filter_where(
        *, archived: Optional[bool] = None, include_pinned: bool = False, **filters
    ) -> Tuple[str, list]:
        """Shared WHERE clause for bulk prune/archive selection (alias ``s``).

        Filters (see ``_PRUNE_FILTERS``) AND together; only ended sessions are
        ever candidates.  ``archived`` is tri-state (None = both).  ``*_like``
        filters are case-insensitive substrings; the rest are exact (provider
        case-insensitive).  Token bounds use input+output; cost bounds use
        ``COALESCE(actual_cost_usd, estimated_cost_usd)``.
        """
        unknown = set(filters) - _PRUNE_FILTER_NAMES
        if unknown:
            raise TypeError(
                "SessionMaintenanceMixin._prune_filter_where() got an unexpected "
                f"keyword argument {sorted(unknown)[0]!r}"
            )
        clauses = ["s.ended_at IS NOT NULL"]
        params: list = []
        for name, applies, build in _PRUNE_FILTERS:
            value = filters.get(name)
            if (value is not None) if applies == "notnone" else bool(value):
                new_clauses, new_params = build(value)
                clauses.extend(new_clauses)
                params.extend(new_params)
        if archived is True:
            clauses.append("s.archived = 1")
        elif archived is False:
            clauses.append("s.archived = 0")
        # Pinned is a durable "keep" flag: bulk prune/delete/archive exclude
        # pinned rows unless the caller explicitly opts in.
        if not include_pinned:
            clauses.append("COALESCE(s.pinned, 0) = 0")
        return " AND ".join(clauses), params

    @staticmethod
    def _apply_prune_age_filter(older_than_days: Optional[float], filters: Dict[str, Any]) -> None:
        """Translate the legacy age window into the shared activity filter."""
        if (
            filters.get("last_active_before") is None
            and filters.get("started_before") is None
            and older_than_days is not None
        ):
            filters["last_active_before"] = time.time() - (older_than_days * 86400)

    def _prune_where(self, older_than_days, source, filters) -> Tuple[str, list]:
        self._apply_prune_age_filter(older_than_days, filters)
        return self._prune_filter_where(source=source, **filters)

    def list_prune_candidates(
        self, older_than_days: Optional[float] = None, source: str = None, **filters
    ) -> List[Dict[str, Any]]:
        """Sessions a matching prune/archive would touch (dry-run), oldest
        first.  Same filters as :meth:`_prune_filter_where`; ``older_than_days``
        is an inactivity threshold (latest message, else ``started_at``)."""
        where, params = self._prune_where(older_than_days, source, filters)
        rows = self._read_all(
            f"""SELECT s.id, s.source, s.title, s.model, s.started_at,
                           COALESCE(
                               (SELECT MAX(m.timestamp) FROM messages m
                                WHERE m.session_id = s.id),
                               s.started_at
                           ) AS last_active,
                           s.ended_at, s.message_count, s.archived
                    FROM sessions s WHERE {where}
                    ORDER BY last_active ASC, s.started_at ASC""",
            params,
        )
        return [dict(row) for row in rows]

    def count_prune_matches(
        self, older_than_days: Optional[float] = None, source: str = None, **filters
    ) -> int:
        """Count-only variant of :meth:`list_prune_candidates` (the CLI uses it
        to report how many pinned sessions are spared)."""
        where, params = self._prune_where(older_than_days, source, filters)
        return int(self._read_one(f"SELECT COUNT(*) FROM sessions s WHERE {where}", params)[0])

    def count_open_prune_matches(
        self, older_than_days: Optional[float] = None, source: str = None, **filters
    ) -> int:
        """Count open sessions a matching prune skips: every normal filter with
        only the ``ended_at`` guard inverted.  Visibility-only; live sessions
        never become prune-eligible."""
        where, params = self._prune_where(older_than_days, source, filters)
        ended_guard = "s.ended_at IS NOT NULL"
        if not where.startswith(ended_guard):
            raise RuntimeError("prune filter lost its ended-session safety guard")
        open_where = f"s.ended_at IS NULL{where[len(ended_guard):]}"
        return int(self._read_one(f"SELECT COUNT(*) FROM sessions s WHERE {open_where}", params)[0])

    def archive_stale_sessions(self, idle_days: float, *, exclude_pinned: bool = True) -> int:
        """Archive every session untouched for ``idle_days`` (real recency:
        freshest of ``last_activity_at`` / latest message / ``started_at``).
        Unlike :meth:`archive_sessions`, this can archive unended sessions.

        Guards: ``pinned = 0`` when ``exclude_pinned``; ``archived = 0`` so
        repeats are no-ops; only lineage tips (``end_reason <> 'compression'``)
        are candidates — a stale tip archives its chain via
        :meth:`set_session_archived`, so an old compressed-away root with a
        recent continuation is never matched.  Returns the count archived.
        """
        if idle_days is None or idle_days < 0:
            return 0
        cutoff = time.time() - float(idle_days) * 86400.0
        pin_clause = "AND s.pinned = 0" if exclude_pinned else ""
        rows = self._read_all(
            f"""
            SELECT s.id FROM sessions s
            WHERE s.archived = 0
              AND COALESCE(s.end_reason, '') <> 'compression'
              {pin_clause}
              AND {_sql_session_last_active("s")} < ?
            ORDER BY s.started_at ASC
            """,
            (cutoff,),
        )
        ids = [r[0] for r in rows]
        for sid in ids:
            self.set_session_archived(sid, True)
        return len(ids)

    def prune_sessions(
        self, older_than_days: Optional[float] = 90, source: str = None,
        sessions_dir: Optional[Path] = None, exclude_active_write_guards: bool = False,
        **filters,
    ) -> int:
        """Delete ended sessions matching the filters; returns the count.

        Default: inactive for ``older_than_days`` (latest message, else
        ``started_at``), optionally by ``source``.  Extra keyword filters are
        those of :meth:`_prune_filter_where`; an explicit ``started_before`` /
        ``last_active_before`` overrides the ``older_than_days`` cutoff
        (pass ``older_than_days=None`` for no implicit age bound).

        Children outside the window are orphaned (parent NULLed), not cascade-
        deleted.  With *sessions_dir*, on-disk transcript files are removed
        outside the DB transaction.  ``exclude_active_write_guards`` (automatic
        maintenance) skips rows under a live turn lease or compression lock,
        while expired/dead holders are reclaimed and fenced in the same write.
        """
        from hermes_state import SessionCompressionInProgressError, SessionTurnLeaseLostError
        where, where_params = self._prune_where(older_than_days, source, filters)
        removed_ids: list[str] = []

        def _do(conn):
            cursor = conn.execute(f"SELECT s.id FROM sessions s WHERE {where}", where_params)
            session_ids = {row["id"] for row in cursor.fetchall()}
            if exclude_active_write_guards:
                protected = set()
                for sid in session_ids:
                    try:
                        self._check_transcript_write_guards(
                            conn, sid, compression_lock_holder=None, turn_lease_holder=None,
                            reject_active_turn_lease=True, reject_active_compression_lock=True,
                            allow_closed_compression_parent=True,
                        )
                    except (SessionCompressionInProgressError, SessionTurnLeaseLostError):
                        protected.add(sid)
                session_ids.difference_update(protected)
            if not session_ids:
                return 0
            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )
            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)

        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    def _page_pragmas(self, *names: str) -> Optional[list]:
        """Read integer PRAGMAs over the existing connection (never a byte probe
        of the live file); None if the connection is closed or a pragma fails."""
        with self._read_ctx() as conn:
            if self._conn is None:
                return None
            return [conn.execute(f"PRAGMA {name}").fetchone()[0] for name in names]

    def logical_size_bytes(self) -> Optional[int]:
        """``page_count * page_size``: the main-file size once the WAL is
        checkpointed back in.  Prefer over ``os.path.getsize`` when reporting a
        VACUUM: in WAL mode the rewrite lands in ``-wal`` and the checkpoint is
        refused while another connection holds a read-mark, so a stat() delta
        understates the win and can go negative.  None if pragmas fail.
        """
        try:
            values = self._page_pragmas("page_count", "page_size")
            if values is None:
                return None
            page_count, page_size = values
            return int(page_count) * int(page_size)
        except Exception as exc:
            logger.debug("Could not read logical DB size: %s", exc)
            return None

    def _freelist_ratio(self) -> Optional[float]:
        """Reclaimable fraction (``freelist_count / page_count``); gates VACUUM
        in :meth:`maybe_auto_prune_and_vacuum`.  None if pragmas fail (callers
        then fall back to the time throttle alone)."""
        try:
            values = self._page_pragmas("page_count", "freelist_count")
            if values is None:
                return None
            page_count, freelist = int(values[0]), int(values[1])
            if page_count <= 0:
                return 0.0
            return freelist / page_count
        except Exception as exc:
            logger.debug("Could not read freelist ratio: %s", exc)
            return None

    def vacuum(self) -> int:
        """VACUUM to reclaim space after large deletes (SQLite never shrinks
        the file on its own).

        Rewrites the whole DB, cannot run inside a transaction, and takes an
        exclusive lock — callers must ensure no other writers are active (safe
        at startup before serving traffic).  FTS5 segments are merged first via
        :meth:`optimize_fts` so the VACUUM reclaims those pages too.  Returns
        the number of FTS indexes optimized (0 on merge failure / no FTS).
        """
        optimized = 0
        try:
            optimized = self.optimize_fts()  # manages its own lock
        except Exception as exc:
            logger.warning("FTS optimize before VACUUM failed: %s", exc)
        with self._lock:
            # PASSIVE, not TRUNCATE: a manual `hermes sessions vacuum` runs in
            # a transient CLI process, and a TRUNCATE reset here would race a
            # live gateway writer and tear B-tree pages.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception as exc:
                logger.debug("WAL checkpoint (PASSIVE) before VACUUM failed: %s", exc)
            self._conn.execute("VACUUM")
            # VACUUM rewrites every page THROUGH the WAL; without this TRUNCATE
            # a 3 GB database leaves a 3 GB -wal behind.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                logger.debug("WAL checkpoint (TRUNCATE) after VACUUM failed: %s", exc)
            # TRUNCATE may replace the WAL inode; adopt the new sidecars so the
            # write-path generation guard does not halt this connection.
            self._record_db_file_identity()
        return optimized

    def maybe_auto_prune_and_vacuum(
        self, retention_days: int = 90, min_interval_hours: int = 24, vacuum: bool = True,
        sessions_dir: Optional[Path] = None, min_vacuum_interval_days: int = 30,
        min_vacuum_freelist_ratio: float = AUTO_VACUUM_MIN_FREELIST_RATIO,
    ) -> Dict[str, Any]:
        """Idempotent startup auto-maintenance: prune inactive sessions,
        reap stale open state-owned rows, optional VACUUM.  Never raises.

        Runs at most once per ``min_interval_hours`` (state_meta).  VACUUM has
        its own ``min_vacuum_interval_days`` throttle and additionally requires
        ``freelist_count / page_count`` > ``min_vacuum_freelist_ratio`` so a
        small prune on a dense multi-GB database never triggers a full rewrite.
        With *sessions_dir*, pruned transcripts are removed from disk too.

        Stale-open reconciliation: cron/kanban/subagent/one-shot CLI rows never
        set ``ended_at`` when their process dies, and prune only deletes ended
        rows.  After pruning, open rows from :attr:`_AUTO_PRUNE_STALE_OPEN_SOURCES`
        older than ``retention_days`` are closed (``startup_orphan_reap``); they
        stay resumable and age from their close, so they get one more full
        retention window.  Messaging and UI sources are never touched.

        Returns ``{"skipped", "pruned", "closed", "vacuumed"}`` plus
        ``"freelist_ratio"`` when a VACUUM was considered and ``"error"`` on
        failure.
        """
        from hermes_state import _release_auto_maintenance_lock, _try_acquire_auto_maintenance_lock
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False}
        maintenance_lock = _try_acquire_auto_maintenance_lock(self.db_path)
        if maintenance_lock is None:
            result["skipped"] = True
            return result
        try:
            last_raw = self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    if now - float(last_raw) < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta; treat as no prior run

            # Prune first: orphans closed below get a full retention window.
            pruned = self.prune_sessions(
                older_than_days=retention_days, sessions_dir=sessions_dir,
                exclude_active_write_guards=True,
            )
            result["pruned"] = pruned
            closed = self.sweep_orphaned_sessions(
                max_idle_seconds=float(retention_days) * 86400.0,
                sources=self._AUTO_PRUNE_STALE_OPEN_SOURCES, exclude_pinned=True,
                respect_gateway_heartbeats=False,  # state-owned lifecycles, not gateway heartbeats
            )
            result["closed"] = len(closed)
            # VACUUM only if rows were freed, the time throttle passed AND the
            # freelist ratio passed — it holds an exclusive lock for a full rewrite.
            last_vacuum_raw = self.get_meta("last_vacuum")
            vacuum_due = True
            if last_vacuum_raw:
                try:
                    vacuum_due = (now - float(last_vacuum_raw)) >= min_vacuum_interval_days * 86400
                except (TypeError, ValueError):
                    vacuum_due = True
            if vacuum and pruned > 0 and vacuum_due:
                ratio = self._freelist_ratio()
                result["freelist_ratio"] = ratio
                if ratio is None or ratio > min_vacuum_freelist_ratio:
                    try:
                        self.vacuum()
                        result["vacuumed"] = True
                        self.set_meta("last_vacuum", str(now))
                    except Exception as exc:
                        logger.warning("state.db VACUUM failed: %s", exc)
                else:
                    logger.debug(
                        "state.db auto-maintenance: skipping VACUUM, only "
                        "%.1f%% of pages reclaimable (threshold %.0f%%)",
                        ratio * 100.0, min_vacuum_freelist_ratio * 100.0,
                    )
            # Record even when pruned == 0 so the throttle holds.
            self.set_meta("last_auto_prune", str(now))
            if closed or pruned > 0:
                logger.info(
                    "state.db auto-maintenance: closed %d stale open session(s), "
                    "pruned %d session(s) inactive for %d days%s",
                    len(closed), pruned, retention_days, " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            # Maintenance must never block startup.
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)
        finally:
            _release_auto_maintenance_lock(maintenance_lock)
        return result
