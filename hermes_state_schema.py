"""Schema creation, column reconciliation, and FTS DDL management for SessionDB.

Plain mixin consumed by ``hermes_state.SessionDB``: no ``__init__``, no state
of its own; methods use host attributes established by ``SessionDB.__init__``.
Must never import hermes_state (cycle) — shared constants live in
hermes_state_common.
"""

import datetime
import hashlib
import logging
import json
import os
import sqlite3
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Sequence


from hermes_constants import get_hermes_home
from hermes_startup_watchdog import report_startup_progress
from hermes_state_common import (
    DEFERRED_INDEX_SQL, FTS_CJK_STALE_KEY, FTS_REBUILD_DEFERRAL_KEY, FTS_STALE_KEY, FTS_SQL,
    FTS_STORAGE_VERSION, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, SCHEMA_SQL,
    SCHEMA_VERSION, _FTS_CJK_TRIGGERS, _FTS_TRIGGERS, _ephemeral_child_sql, fts_rebuild_admission,
)

# Keep the pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")

_FTS_HOLDER_ESCALATE_ATTEMPTS = 3
_FTS_HOLDER_ESCALATE_SECONDS = 60.0
# In-process retry cadence for a deferred stale-FTS rebuild
# (``retry_deferred_fts_recovery``): startup paid the full admission wait once;
# later retries are non-blocking probes so a live holder never stalls a
# long-lived writer. Each failed retry doubles the spacing up to the cap, so a
# permanent holder costs one deferral warning per hour, not per minute.
_FTS_STALE_RETRY_SECONDS = 60.0
_FTS_STALE_RETRY_MAX_SECONDS = 3600.0

# schema_read_probe_statements() cache — deriving it parses SCHEMA_SQL in an
# in-memory SQLite database, so do it once per process.
_READ_PROBE_STATEMENTS: Optional[tuple] = None

# The trigram triggers come ONLY from FTS_TRIGRAM_SQL / LEGACY_FTS_TRIGRAM_SQL,
# whose CREATE VIRTUAL TABLE needs the trigram tokenizer (SQLite >= 3.34);
# without it _ensure_fts_schema soft-fails that DDL and "all six present" is
# permanently unsatisfiable. Split the set so a trigger's absence is only
# measured against the DDL that can create it. Exhaustive and disjoint by
# construction; pinned by test_fts_trigger_subsets_match_the_ddl.
_FTS_TRIGRAM_TRIGGERS = tuple(n for n in _FTS_TRIGGERS if "_trigram_" in n)
_FTS_BASE_TRIGGERS = tuple(n for n in _FTS_TRIGGERS if n not in _FTS_TRIGRAM_TRIGGERS)

_LEGACY_INLINE_CONCAT_SQL = (
    "COALESCE(content, '') || ' ' || "
    "COALESCE(tool_name, '') || ' ' || "
    "COALESCE(tool_calls, '') "
)
_SESSION_MODEL_USAGE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model)",
)
_SESSION_MODEL_USAGE_HEAL_DDL = """CREATE TABLE session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
)"""
# Same table as emitted by the v22 migration: column lines at 35 spaces, closing
# paren at 31 (statement text is pinned by the SQL trace harness).
_SESSION_MODEL_USAGE_V22_DDL = "\n".join(
    [_SESSION_MODEL_USAGE_HEAL_DDL.splitlines()[0]]
    + [" " * 35 + ln.strip() for ln in _SESSION_MODEL_USAGE_HEAL_DDL.splitlines()[1:-1]]
    + [" " * 31 + ")"]
)
# Statement text pinned by the SQL trace harness (whitespace included).
_SESSION_MODEL_USAGE_V20_SEED_SQL = """INSERT OR IGNORE INTO session_model_usage (
                               session_id, model, billing_provider,
                               billing_base_url, billing_mode,
                               api_call_count, input_tokens,
                               output_tokens, cache_read_tokens,
                               cache_write_tokens, reasoning_tokens,
                               estimated_cost_usd, actual_cost_usd,
                               cost_status, cost_source, first_seen, last_seen
                           )
                           SELECT id, COALESCE(model, 'unknown'),
                                  COALESCE(billing_provider, ''),
                                  COALESCE(billing_base_url, ''),
                                  COALESCE(billing_mode, ''),
                                  COALESCE(api_call_count, 0),
                                  COALESCE(input_tokens, 0),
                                  COALESCE(output_tokens, 0),
                                  COALESCE(cache_read_tokens, 0),
                                  COALESCE(cache_write_tokens, 0),
                                  COALESCE(reasoning_tokens, 0),
                                  COALESCE(estimated_cost_usd, 0),
                                  COALESCE(actual_cost_usd, 0),
                                  cost_status, cost_source,
                                  started_at, COALESCE(ended_at, started_at)
                           FROM sessions
                           WHERE COALESCE(input_tokens, 0)
                                 + COALESCE(output_tokens, 0)
                                 + COALESCE(cache_read_tokens, 0)
                                 + COALESCE(cache_write_tokens, 0)
                                 + COALESCE(reasoning_tokens, 0) > 0"""
_TITLE_UNIQUE_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
    "ON sessions(title) WHERE title IS NOT NULL"
)


def _q(ident: str) -> str:
    """Double-quote an SQL identifier."""
    return '"' + ident.replace('"', '""') + '"'


def schema_read_probe_statements() -> tuple:
    """SELECT statements that fail iff a live store is behind SCHEMA_SQL.

    Read-only opens skip ``_reconcile_columns()`` by design (no DDL against
    another profile's live DB), so healing callers (``_open_session_db_at_path``
    in the web server) run these probes after a read-only open: a missing
    table/column raises at prepare time. Derived from SCHEMA_SQL so a column
    added there is covered automatically (a hand-maintained list went stale
    within days); ``LIMIT 0`` so zero rows are read. Column references are
    table-qualified: an unqualified double-quoted identifier that fails to
    resolve silently degrades to a string literal (SQLite misfeature), which
    would make the probe pass on exactly the stale store it exists to catch.
    """
    global _READ_PROBE_STATEMENTS
    if _READ_PROBE_STATEMENTS is None:
        tables = SessionSchemaMixin._parse_schema_columns(SCHEMA_SQL)
        _READ_PROBE_STATEMENTS = tuple(
            "SELECT {} FROM {} LIMIT 0".format(", ".join(f"{_q(table)}.{_q(col)}" for col in cols), _q(table))
            for table, cols in sorted(tables.items())
        )
    return _READ_PROBE_STATEMENTS


class SessionSchemaMixin:
    """See module docstring — mixin for SessionDB (Schema cluster)."""

    def _dedupe_legacy_system_prompts(self, cursor: sqlite3.Cursor) -> None:
        """Move inline prompt snapshots into the shared content-addressed table.

        Contention-safe: any ``OperationalError`` mid-loop returns instead of
        raising. Partial migration is safe — the legacy ``system_prompt``
        column stays a read fallback and the next schema init resumes.
        Propagating the error aborted schema init, left the version below
        25, and re-entered this migration on every open (gateway crash loop).
        """
        try:
            rows = cursor.execute(
                "SELECT id, system_prompt FROM sessions "
                "WHERE system_prompt IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for session_id, prompt in rows:
            try:
                prompt_hash = self._store_system_prompt(cursor, prompt)
                cursor.execute(
                    "UPDATE sessions "
                    "SET system_prompt_hash = ?, system_prompt = NULL "
                    "WHERE id = ?",
                    (prompt_hash, session_id),
                )
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "v25 prompt dedupe paused after contention (%s); "
                    "unmigrated rows keep the legacy inline prompt and the "
                    "next schema init resumes the migration.",
                    exc,
                )
                return

    def _sqlite_supports_fts5(self, cursor: sqlite3.Cursor) -> bool:
        try:
            cursor.execute("CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)")
            cursor.execute("DROP TABLE temp._hermes_fts5_probe")
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            self._warn_fts5_unavailable(exc)
            return False

    def _drop_all_fts_triggers(self, cursor: sqlite3.Cursor) -> None:
        self._drop_fts_triggers(cursor)
        for trigger in _FTS_CJK_TRIGGERS:
            try:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.OperationalError:
                pass

    @staticmethod
    def _fts_trigger_count(cursor: sqlite3.Cursor, names: Sequence[str] = _FTS_TRIGGERS) -> int:
        """Count how many of *names* currently exist as triggers (pass
        _FTS_BASE_TRIGGERS / _FTS_TRIGRAM_TRIGGERS to check one half)."""
        if not names:
            return 0  # "name IN ()" is a SQLite syntax error
        placeholders = ",".join("?" for _ in names)
        row = cursor.execute(
            f"SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            tuple(names),
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _fts_update_trigger_needs_narrowing(sql: Optional[str]) -> bool:
        """True when trigger SQL is a broad AFTER UPDATE (missing ``OF``)."""
        if not sql:
            return False
        compact = " ".join(sql.split()).upper()  # multi-line DDL still matches
        return "AFTER UPDATE OF " not in compact and "AFTER UPDATE ON " in compact

    def _migrate_broad_fts_update_triggers(self, cursor: sqlite3.Cursor) -> int:
        """Replace broad AFTER UPDATE FTS triggers with AFTER UPDATE OF variants.

        ``CREATE TRIGGER IF NOT EXISTS`` never replaces an existing broad
        trigger, so it would keep firing on every messages row touch. Drop
        still-broad UPDATE triggers and re-apply the current DDL. No FTS
        rebuild: correctness was already gated by WHEN clauses; OF only skips
        unnecessary trigger evaluation. Returns the number dropped.
        """
        # CJK is v23-only. Decide the layout before selecting destructive
        # candidates so the legacy branch never drops a trigger it won't recreate.
        legacy_layout = self._db_has_legacy_inline_fts(cursor)
        update_names = ("messages_fts_update", "messages_fts_trigram_update")
        if not legacy_layout:
            update_names += ("messages_fts_cjk_update",)
        placeholders = ", ".join("?" for _ in update_names)
        rows = cursor.execute(
            "SELECT name, sql FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            update_names,
        ).fetchall()
        to_drop = [name for name, sql in rows if self._fts_update_trigger_needs_narrowing(sql)]
        if not to_drop:
            return 0
        for name in to_drop:
            # Names come from the literal allowlist above — interpolation-safe.
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")

        # Re-apply current DDL (legacy vs v23 chosen as _init_schema does) so
        # CREATE TRIGGER installs the OF variants.
        if legacy_layout:
            self._ensure_fts_schema(cursor, "messages_fts", LEGACY_FTS_SQL)
            self._ensure_fts_schema(cursor, "messages_fts_trigram", LEGACY_FTS_TRIGRAM_SQL)
        else:
            self._ensure_fts_schema(cursor, "messages_fts", FTS_SQL)
            self._ensure_fts_schema(cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL)
            # Only recreate the CJK trigger this migration actually dropped.
            # ``_ensure_fts_cjk_schema`` soft-fails OperationalError by clearing
            # availability (never raises), so afterwards require a narrowed CJK
            # UPDATE trigger or durable quarantine (stale breadcrumb + unavailable).
            if "messages_fts_cjk_update" in to_drop:
                try:
                    self._ensure_fts_cjk_schema(cursor)
                except Exception:
                    self._quarantine_cjk_after_update_of_migration(cursor)
                    logger.exception("CJK FTS re-ensure after UPDATE OF migration failed")
                    raise
                if not self._cjk_update_trigger_is_narrowed(cursor):
                    self._quarantine_cjk_after_update_of_migration(cursor)
                    logger.warning(
                        "CJK FTS UPDATE trigger missing or still broad after "
                        "UPDATE OF migration; marked stale and unavailable"
                    )
        logger.info(
            "Migrated %d broad FTS UPDATE trigger(s) to AFTER UPDATE OF " "(no rebuild required)",
            len(to_drop),
        )
        return len(to_drop)

    def _cjk_update_trigger_is_narrowed(self, cursor: sqlite3.Cursor) -> bool:
        """True when messages_fts_cjk_update exists with AFTER UPDATE OF."""
        row = cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name = ?",
            ("messages_fts_cjk_update",),
        ).fetchone()
        return bool(row) and not self._fts_update_trigger_needs_narrowing(row[0])

    def _quarantine_cjk_after_update_of_migration(self, cursor: sqlite3.Cursor) -> None:
        """Fail closed after dropping the CJK UPDATE trigger mid-migration:
        clear availability, persist ``fts_cjk_stale``, drop any residual CJK
        UPDATE trigger so a later open cannot IF-NOT-EXISTS over a gap."""
        self._fts_cjk_available = False
        try:
            self.set_meta(FTS_CJK_STALE_KEY, "1", cursor=cursor)
        except Exception:
            logger.debug("Could not persist CJK FTS stale breadcrumb", exc_info=True)
        try:
            cursor.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        except Exception:
            logger.debug("Could not drop residual CJK UPDATE trigger after quarantine", exc_info=True)

    @staticmethod
    def _rebuild_fts_indexes(cursor: sqlite3.Cursor, *, include_trigram: bool = True) -> None:
        """v23+ external-content tables: 'rebuild' repopulates the inverted
        index from the content source (messages / messages_fts_trigram_src).
        'rebuild' indexes EVERY row, so the deferred-backfill markers are
        cleared or the worker would re-insert covered rows (duplicates)."""
        cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        if include_trigram:
            cursor.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')")
        cursor.execute(
            "DELETE FROM state_meta WHERE key IN "
            "('fts_rebuild_high_water', 'fts_rebuild_progress')"
        )

    @staticmethod
    def _rebuild_legacy_fts_indexes(cursor: sqlite3.Cursor, *, include_trigram: bool = True) -> None:
        """Rebuild the LEGACY inline (pre-v23) FTS indexes from messages.
        Inline tables have no external-content 'rebuild' source, so DELETE +
        reinsert the concatenated content the legacy triggers produced.
        Never touches the v23 shape."""
        tables = ("messages_fts", "messages_fts_trigram") if include_trigram else ("messages_fts",)
        for tbl in tables:
            cursor.execute(f"DELETE FROM {tbl}")
            cursor.execute(f"INSERT INTO {tbl}(rowid, content) SELECT id, {_LEGACY_INLINE_CONCAT_SQL}FROM messages")

    def _fts_table_probe(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
        """True = queryable, False = absent, None = FTS module/tokenizer missing
        or content undecodable (index degraded, store accessible).

        Invalid UTF-8 in FTS content surfaces as a bare UnicodeDecodeError on
        some builds and as OperationalError("Could not decode to UTF-8 ...") on
        others; both are caught so the probe never raises into writable-init /
        recovery flows. Anything else (malformed schema, corrupt vtable) re-raises.
        """
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            return True
        except (sqlite3.OperationalError, UnicodeDecodeError) as exc:
            if isinstance(exc, sqlite3.OperationalError):
                if self._is_fts5_unavailable_error(exc):
                    # A missing trigram tokenizer only affects trigram search;
                    # only a missing FTS5 module disables FTS entirely.
                    if self._is_trigram_unavailable_error(exc):
                        self._warn_trigram_unavailable(exc)
                    else:
                        self._warn_fts5_unavailable(exc)
                    return None
                if "no such table" in str(exc).lower():
                    return False
                if "decode to utf-8" not in str(exc).lower():
                    raise
            logger.warning(
                "%s probe encountered invalid UTF-8 in FTS content; "
                "search may return incomplete results until FTS is rebuilt: %s",
                table_name,
                exc,
            )
            return None

    # ── Stale-FTS recovery ─────────────────────────────────────────────────

    def _defer_stale_fts_for_holders(self, cursor: sqlite3.Cursor, foreign_holders) -> bool:
        """Record a deferral diagnostic for the foreign processes holding the
        DB and decide whether to defer; True = defer (holders remain).

        After ``_FTS_HOLDER_ESCALATE_ATTEMPTS`` deferrals spanning
        ``_FTS_HOLDER_ESCALATE_SECONDS``, provably inactive orphan Desktop
        backends are reaped and the holders re-checked.
        """
        now = time.time()
        record = None
        try:
            row = cursor.execute(
                "SELECT value FROM state_meta WHERE key = ? LIMIT 1",
                (FTS_REBUILD_DEFERRAL_KEY,),
            ).fetchone()
            if row:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    record = parsed
        except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            record = None
        try:
            first_seen = float((record or {}).get("first_seen", now))
            attempts = int((record or {}).get("attempts", 0)) + 1
        except (TypeError, ValueError):
            first_seen = now
            attempts = 1
        if first_seen > now or first_seen < 0:
            first_seen = now
        diagnostic = {
            "first_seen": first_seen, "last_seen": now, "attempts": attempts,
            "holder_pids": sorted({pid for pid, _path in foreign_holders if pid > 0}),
        }
        cursor.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (FTS_REBUILD_DEFERRAL_KEY, json.dumps(diagnostic, sort_keys=True)),
        )
        if attempts >= _FTS_HOLDER_ESCALATE_ATTEMPTS and now - first_seen >= _FTS_HOLDER_ESCALATE_SECONDS:
            reaped = self._reap_inactive_orphan_desktop_holders(
                foreign_holders, min_age_seconds=_FTS_HOLDER_ESCALATE_SECONDS,
            )
            if reaped:
                logger.error(
                    "Reaped inactive orphan Desktop backend(s) %s after %d "
                    "state.db FTS rebuild deferrals; checking holders again.",
                    reaped,
                    attempts,
                )
                foreign_holders = self._foreign_state_db_holders()
            if foreign_holders:
                logger.error(
                    "state.db FTS repair remains blocked after %d deferrals "
                    "by holder(s) %s. Stop the listed processes, then run "
                    "`hermes sessions optimize-storage` with the gateway stopped. "
                    "`hermes doctor` reports this degraded state.",
                    attempts,
                    foreign_holders,
                )
        if not foreign_holders:
            return False
        logger.warning(
            "Deferred stale state.db FTS rebuild while foreign processes "
            "hold the database or WAL sidecars (%s); canonical writes and "
            "LIKE search remain available (deferral %d).",
            foreign_holders,
            attempts,
        )
        return True

    def _recover_stale_fts(self, cursor: sqlite3.Cursor, *, legacy: bool, timeout_seconds=None) -> bool:
        """Atomically rebuild stale base/trigram indexes and resume syncing.

        *timeout_seconds* bounds the cross-process admission wait; None uses
        the full startup budget, ``0`` is the non-blocking in-process retry.
        Fails closed: foreign holders or a lost admission race leave the
        breadcrumb set and defer to a later retry.
        """
        foreign_holders = self._foreign_state_db_holders()
        if foreign_holders and self._defer_stale_fts_for_holders(cursor, foreign_holders):
            return False
        with fts_rebuild_admission(self.db_path, timeout_seconds=timeout_seconds) as admitted:
            if not admitted:
                logger.warning(
                    "Deferred stale state.db FTS rebuild: another process "
                    "holds the rebuild authority; canonical writes and LIKE "
                    "search remain available."
                )
                return False
            return self._recover_stale_fts_locked(cursor, legacy=legacy)

    def retry_deferred_fts_recovery(self) -> bool:
        """Retry a deferred stale-FTS rebuild on this open SessionDB (gateway
        housekeeping tick). ``_recover_stale_fts`` fails closed at open when
        holders or the rebuild lock are busy, leaving ``_fts_stale`` set and
        search on LIKE; live write/search paths must never start a full
        rebuild, and a gateway opens state.db once for days, so "next open"
        never comes. Bounded backoff (``_FTS_STALE_RETRY_SECONDS`` doubling to
        the max), non-blocking admission (``timeout=0``), no new thread.
        Returns True only when the index was rebuilt and sync triggers
        restored. Never raises."""
        if not self._fts_stale or self.read_only or self._conn is None:
            return False
        now = time.monotonic()
        if now < getattr(self, "_fts_stale_retry_after", 0.0):
            return False
        interval = float(getattr(self, "_fts_stale_retry_interval", 0.0))
        if interval <= 0.0:
            interval = _FTS_STALE_RETRY_SECONDS
        self._fts_stale_retry_after = now + interval
        self._fts_stale_retry_interval = min(
            max(interval, _FTS_STALE_RETRY_SECONDS, 1.0) * 2.0, _FTS_STALE_RETRY_MAX_SECONDS,
        )
        try:
            with self._lock:
                if self._conn is None or not self._fts_stale:
                    return False
                cursor = self._conn.cursor()
                legacy = self._db_has_legacy_inline_fts(cursor)
                recovered = self._recover_stale_fts(cursor, legacy=legacy, timeout_seconds=0.0)
                if recovered:
                    # CJK was detached alongside the base indexes; its own
                    # ensure path decides when it comes back online.
                    self._ensure_fts_cjk_schema(cursor)
                    self._fts_stale_retry_interval = 0.0
                try:
                    self._conn.commit()
                except sqlite3.Error:
                    pass
                return recovered
        except Exception:  # noqa: BLE001 - background retry must never raise
            logger.warning(
                "In-process retry of the deferred stale state.db FTS rebuild "
                "failed; will retry later.",
                exc_info=True,
            )
            return False

    def _recover_stale_fts_locked(self, cursor: sqlite3.Cursor, *, legacy: bool) -> bool:
        """Body of :meth:`_recover_stale_fts`; caller holds rebuild authority.

        One write transaction closes the dangerous gap: no canonical writer
        can slip between the full rebuild and trigger restoration.
        """
        try:
            trigram_status = self._fts_table_probe(cursor, "messages_fts_trigram")
        except (sqlite3.DatabaseError, UnicodeDecodeError):
            # A corrupt vtable may fail even a LIMIT 0 probe; it must still be
            # included in the drop-and-recreate below.
            trigram_status = True
        include_trigram = trigram_status is True

        drop_sql = "".join(f"DROP TRIGGER IF EXISTS {trigger};" for trigger in _FTS_TRIGGERS)
        if include_trigram:
            drop_sql += "DROP TABLE IF EXISTS messages_fts_trigram;"
        drop_sql += "DROP VIEW IF EXISTS messages_fts_trigram_src;"
        drop_sql += "DROP TABLE IF EXISTS messages_fts;"

        if legacy:
            schema_sql = LEGACY_FTS_SQL
            if include_trigram:
                schema_sql += LEGACY_FTS_TRIGRAM_SQL
            rebuild_sql = schema_sql + """
                INSERT INTO messages_fts(rowid, content)
                SELECT id,
                       COALESCE(content, '') || ' ' ||
                       COALESCE(tool_name, '') || ' ' ||
                       COALESCE(tool_calls, '')
                FROM messages;
            """
            if include_trigram:
                rebuild_sql += """
                    DELETE FROM messages_fts_trigram;
                    INSERT INTO messages_fts_trigram(rowid, content)
                    SELECT id,
                           COALESCE(content, '') || ' ' ||
                           COALESCE(tool_name, '') || ' ' ||
                           COALESCE(tool_calls, '')
                    FROM messages;
                """
        else:
            schema_sql = FTS_SQL
            if include_trigram:
                schema_sql += FTS_TRIGRAM_SQL
            rebuild_sql = schema_sql + "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
            if include_trigram:
                rebuild_sql += "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild');"
            rebuild_sql += "DELETE FROM state_meta WHERE key IN ('fts_rebuild_high_water', 'fts_rebuild_progress');"

        recovery_sql = (
            "BEGIN IMMEDIATE;"
            + drop_sql
            + rebuild_sql
            + "DELETE FROM state_meta WHERE key IN "
            + f"('{FTS_STALE_KEY}', '{FTS_REBUILD_DEFERRAL_KEY}');"
            + "COMMIT;"
        )
        try:
            cursor.executescript(recovery_sql)
        except sqlite3.DatabaseError as exc:
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass
            # Stale indexes must stay detached even on SQLite builds whose DDL
            # transaction behavior differs.
            self._drop_all_fts_triggers(cursor)
            self._conn.commit()
            logger.error(
                "Automatic rebuild of stale FTS indexes failed (%s); "
                "canonical writes remain enabled with FTS detached.",
                exc,
            )
            return False

        self._fts_stale = False
        self._fts_enabled = True
        self._trigram_available = include_trigram
        logger.warning(
            "Rebuilt stale state.db FTS indexes from canonical messages and "
            "restored sync triggers."
        )
        return True

    # ── Declarative column reconciliation ──────────────────────────────────

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Expected columns per table, parsed from SCHEMA_SQL.

        Executes the DDL in an in-memory SQLite database and reads PRAGMA
        table_info, so SQLite handles every syntax edge case (no regex).
        The result is memoized on disk keyed by a hash of the DDL (~85ms per
        startup otherwise; a pure function of the DDL text). Only the
        reference-side parse is cached — diffing the LIVE database still runs
        every startup. A corrupt or stale cache degrades to recomputation.
        """
        cache_path = None
        schema_hash = hashlib.sha256(schema_sql.encode("utf-8")).hexdigest()
        try:
            # Late import: resolves a test-patched hermes_constants.get_hermes_home.
            from hermes_constants import get_hermes_home as _home
            cache_path = _home() / "cache" / "schema_columns.json"
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(blob, dict)
                and blob.get("schema_hash") == schema_hash
                and isinstance(blob.get("tables"), dict)
            ):
                tables = blob["tables"]
                if all(
                    isinstance(cols, dict) and all(isinstance(v, str) for v in cols.values())
                    for cols in tables.values()
                ):
                    return tables
        except Exception:
            pass  # missing/corrupt cache → recompute below

        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
                for _cid, col_name, col_type, notnull, default, pk in ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                ).fetchall():
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
        finally:
            ref.close()

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(cache_path.parent), prefix=".schema_columns.")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump({"schema_hash": schema_hash, "tables": table_columns}, fh)
                os.replace(tmp, cache_path)
            except Exception:
                pass  # cache write is best-effort
        return table_columns

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """ADD every SCHEMA_SQL column missing from the live tables.

        Beets/sqlite-utils pattern: SCHEMA_SQL is the single source of truth;
        column additions are declarative and need no version-gated migration.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            try:
                rows = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            except sqlite3.OperationalError:
                continue  # Table doesn't exist yet (shouldn't happen after executescript)
            # PRAGMA table_info rows: (cid, name, type, notnull, dflt_value, pk)
            live_cols = {row[1] for row in rows}
            for col_name, col_type in declared_cols.items():
                if col_name in live_cols:
                    continue
                try:
                    cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN {_q(col_name)} {col_type}')
                except sqlite3.OperationalError as exc:
                    message = str(exc).lower()
                    if "duplicate column" in message:
                        # A sibling process won the ADD race; store is correct.
                        logger.debug("reconcile %s.%s: %s", table_name, col_name, exc)
                        continue
                    if "locked" in message or "busy" in message:
                        # Swallowing lock contention left the store half-reconciled
                        # ("no such column" on every read). Re-raise so
                        # _connect_and_init_with_lock_patience retries the WHOLE
                        # init (idempotent) with backoff.
                        raise
                    # Anything else permanently strands the store behind SCHEMA_SQL — be loud.
                    logger.warning(
                        "reconcile %s.%s failed; store remains behind "
                        "SCHEMA_SQL: %s", table_name, col_name, exc,
                    )

    @staticmethod
    def _live_pk_columns(cursor: sqlite3.Cursor, table: str) -> Optional[List[str]]:
        """PRIMARY KEY column names of *table* in key order; None when the
        table is missing or has no columns (SCHEMA_SQL creates it correctly)."""
        try:
            rows = cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        # row: (cid, name, type, notnull, dflt_value, pk)
        return [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]

    def _heal_gateway_routing_pk(self, cursor: sqlite3.Cursor) -> None:
        """Rebuild ``gateway_routing`` when its PRIMARY KEY predates scoping.

        Early builds used ``session_key TEXT PRIMARY KEY``; the reconciler ADDs
        ``scope`` but SQLite cannot ALTER a PK, so the composite key never
        lands and every routing write fails (ON CONFLICT mismatch / UNIQUE
        violation across scopes) with per-save warning spam. Rebuild once,
        preserving rows; on a cross-scope session_key collision the newest
        row wins.
        """
        pk_cols = self._live_pk_columns(cursor, "gateway_routing")
        if pk_cols is None or pk_cols == ["scope", "session_key"]:
            return
        logger.info(
            "gateway_routing has legacy primary key %r; rebuilding with "
            "composite (scope, session_key) key",
            pk_cols,
        )
        cursor.execute("ALTER TABLE gateway_routing RENAME TO gateway_routing_legacy_pk")
        cursor.execute(
            """CREATE TABLE gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
)"""
        )
        # INSERT OR REPLACE in updated_at order: newest row per key wins.
        cursor.execute(
            "INSERT OR REPLACE INTO gateway_routing "
            "(scope, session_key, entry_json, updated_at) "
            "SELECT COALESCE(scope, ''), session_key, entry_json, updated_at "
            "FROM gateway_routing_legacy_pk ORDER BY updated_at ASC"
        )
        cursor.execute("DROP TABLE gateway_routing_legacy_pk")

    def _heal_session_model_usage_pk(self, cursor: sqlite3.Cursor) -> None:
        """Rebuild ``session_model_usage`` when its PRIMARY KEY lacks ``task``.

        Installs already at v22+ when ``task`` landed carry the 5-column PK;
        the reconciler ADDs ``task`` as a bare nullable but SQLite cannot
        ALTER a PK, and the version-gated v22 rebuild is unreachable there.
        Every ``_record_model_usage()`` upsert then fails (ON CONFLICT
        mismatch), aborting the write transaction and silently zeroing token
        and cost accounting. Idempotent; no-op on healthy databases.

        FK-off window: INSERT OR IGNORE does NOT suppress foreign-key
        violations, so an orphaned usage row (partial prune while accounting
        was broken) would abort the whole rebuild. PRAGMA foreign_keys is a
        no-op inside a transaction — fine here, _init_schema runs on an
        isolation_level=None connection with no transaction open.
        """
        pk_cols = self._live_pk_columns(cursor, "session_model_usage")
        if pk_cols is None or "task" in pk_cols:
            return
        logger.info(
            "session_model_usage has legacy primary key %r (missing task); "
            "rebuilding with composite 6-column key",
            sorted(pk_cols),
        )
        cursor.execute("PRAGMA foreign_keys=OFF")
        try:
            cursor.execute("ALTER TABLE session_model_usage RENAME TO session_model_usage_legacy_pk")
            cursor.execute(_SESSION_MODEL_USAGE_HEAL_DDL)
            # OR IGNORE: COALESCE(task, '') on legacy NULL rows can collide
            # with a genuine ''-task row — keep the first rather than fail.
            cursor.execute(
                """INSERT OR IGNORE INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   )
                   SELECT session_id, model,
                          COALESCE(billing_provider, ''),
                          COALESCE(billing_base_url, ''),
                          COALESCE(billing_mode, ''),
                          COALESCE(task, ''),
                          api_call_count, input_tokens,
                          output_tokens, cache_read_tokens, cache_write_tokens,
                          reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                          cost_status, cost_source, first_seen, last_seen
                   FROM session_model_usage_legacy_pk"""
            )
            cursor.execute("DROP TABLE session_model_usage_legacy_pk")
            for sql in _SESSION_MODEL_USAGE_INDEX_SQL:
                cursor.execute(sql)
        except sqlite3.OperationalError as exc:
            logger.debug("session_model_usage PK heal skipped: %s", exc)
        finally:
            cursor.execute("PRAGMA foreign_keys=ON")

    # ── _init_schema ───────────────────────────────────────────────────────

    def _init_schema(self):
        """Create tables and FTS if missing, reconcile columns, run data migrations.

        SCHEMA_SQL is the single source of truth: column additions are
        declarative via _reconcile_columns(), so reordered migrations can
        never skip a column. schema_version remains for data migrations
        (row transforms) that cannot be expressed declaratively.
        """
        # Startup-watchdog progress lease: on multi-GB state.db files the
        # reconciliation + data migrations are legitimately slow and I/O-bound
        # (near-zero CPU), which the watchdog's CPU fallback would misread as
        # a parked deadlock. Single lease is deliberate (clamped to
        # _MAX_LEASE_S=900): a genuinely wedged init delays supervisor respawn
        # by up to the lease; per-chunk renewal isn't worth the complexity.
        report_startup_progress(600.0, phase="state_db_init_schema")
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)

        # Idempotent, self-healing column reconciliation, then the two
        # table-shape repairs ADD COLUMN cannot express (PK rebuilds).
        self._reconcile_columns(cursor)
        self._heal_gateway_routing_pk(cursor)
        self._heal_session_model_usage_pk(cursor)

        # Indexes referencing reconciler-added columns must be created AFTER
        # _reconcile_columns — in SCHEMA_SQL the initial executescript would
        # fail on legacy DBs (WHERE references a not-yet-existing column).
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
                "ON messages(session_id, platform_message_id) "
                "WHERE platform_message_id IS NOT NULL"
            )
        except sqlite3.OperationalError as exc:
            logger.debug("idx_messages_platform_msg_id create skipped: %s", exc)
        cursor.executescript(DEFERRED_INDEX_SQL)  # same ordering constraint (``active``)

        # Heal NULL ``active`` rows on every startup: older reconciler builds
        # added ``active`` without its NOT NULL DEFAULT 1, so INSERTs omitting
        # it wrote NULL and ``WHERE active = 1`` loaders hid whole histories.
        # Unconditional because a ``current_version < 12`` gate never re-ran
        # for already-v12+ databases.
        try:
            cursor.execute("UPDATE messages SET active = 1 WHERE active IS NULL")
        except sqlite3.OperationalError:
            pass

        fts5_available = self._sqlite_supports_fts5(cursor)
        self._fts_stale = cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1",
            (FTS_STALE_KEY,),
        ).fetchone() is not None
        if self._fts_stale:
            # A prior process detached FTS after corruption; keep every FTS
            # writer detached until a full rebuild succeeds.
            self._drop_all_fts_triggers(cursor)
        if not fts5_available:
            # Existing FTS triggers would still fire on messages writes even
            # though this runtime cannot read their targets. Drop only the
            # triggers so persistence continues; a future FTS5 runtime's
            # _ensure_fts_schema() recreates them.
            self._drop_fts_triggers(cursor)

        row = cursor.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            # Store provenance so fresh vs wiped stores are distinguishable.
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            instance_id = str(uuid.uuid4())
            cursor.executemany(
                "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                [("store_instance_id", instance_id), ("store_created_at_utc", now_iso)],
            )
        else:
            self._run_data_migrations(cursor, row[0], fts5_available)

        self._ensure_unique_title_index(cursor)
        if fts5_available:
            self._init_fts(cursor)
        self._conn.commit()

    def _run_data_migrations(self, cursor: sqlite3.Cursor, current_version: int, fts5_available: bool) -> None:
        """Version-gated chain for DATA migrations only (row backfills,
        version-specific index changes); column additions never belong here.
        Advances schema_version at the end unless FTS work could not complete."""
        # Renew the lease: the chain can rewrite whole tables on large DBs.
        report_startup_progress(600.0, phase="state_db_data_migrations")
        fts_migrations_complete = True
        if current_version < 10 and SCHEMA_VERSION == 10:
            # v10: one-time trigram backfill. Only when v10 itself is the
            # target: v11+ drops and rebuilds both FTS tables, so the
            # backfill would only burn startup time and WAL space.
            if fts5_available:
                _fts_trigram_exists = self._fts_table_probe(cursor, "messages_fts_trigram")
                if _fts_trigram_exists is False:
                    if self._ensure_fts_schema(cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL):
                        cursor.execute(
                            "INSERT INTO messages_fts_trigram(rowid, content) "
                            "SELECT id, content FROM messages WHERE content IS NOT NULL"
                        )
                    else:
                        fts_migrations_complete = False
                elif _fts_trigram_exists is None:
                    fts_migrations_complete = False
            else:
                fts_migrations_complete = False
        # (v11 inline FTS re-index was superseded by v23 and removed.)
        if current_version < 16:
            # v16: tag delegate subagent rows so pickers stay clean after
            # parent deletes orphan them. The shared predicate excludes
            # user-visible reset children.
            try:
                cursor.execute(
                    "UPDATE sessions SET model_config = json_set("
                    "COALESCE(model_config, '{}'), '$._delegate_from', parent_session_id) "
                    f"WHERE parent_session_id IS NOT NULL "
                    "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                    f"AND {_ephemeral_child_sql('sessions')}"
                )
                cursor.execute(
                    "UPDATE sessions SET model_config = json_set("
                    "COALESCE(model_config, '{}'), '$._delegate_from', '__orphaned__') "
                    "WHERE parent_session_id IS NULL "
                    "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                    "AND json_extract(COALESCE(model_config, '{}'), '$._branched_from') IS NULL "
                    "AND title IS NULL "
                    "AND message_count <= 25 "
                    "AND EXISTS (SELECT 1 FROM messages m "
                    "            WHERE m.session_id = sessions.id AND m.role = 'tool') "
                    "AND NOT EXISTS (SELECT 1 FROM sessions ch "
                    "                WHERE ch.parent_session_id = sessions.id)"
                )
            except sqlite3.OperationalError:
                pass
        if current_version < 18:
            # v18: backfill gateway metadata from sessions.json. Best-effort:
            # consumers fall back to sessions.json until the gateway rewrites.
            try:
                self._backfill_gateway_metadata_from_sessions_json(cursor)
            except Exception as exc:
                logger.debug("v18 gateway metadata backfill skipped: %s", exc)
        if current_version < 20:
            # v20: seed one session_model_usage row per historical session
            # from the sessions aggregates. INSERT OR IGNORE: a row newer
            # code already wrote wins over the stale aggregate.
            try:
                cursor.execute(_SESSION_MODEL_USAGE_V20_SEED_SQL)
            except sqlite3.OperationalError:
                pass
        if current_version < 22:
            self._migrate_v22_session_model_usage(cursor)
        # v23: FTS storage redesign (external-content tables; inline v11
        # tables were ~75% of state.db on heavy installs). OPT-IN, NOT
        # AUTOMATIC: the transition is disk-heavy (~2x transient) and long
        # (hours on a 25 GB DB), so an existing install only gets a flag
        # advertising it; `hermes sessions optimize-storage` performs it as
        # a deliberate foreground operation. DECOUPLED VERSIONING: the FTS
        # layout is tracked by the independent `fts_storage_version`
        # marker, so schema_version still advances here and future
        # migrations land for legacy-FTS users too.
        if current_version < 23 and fts5_available and self._db_has_legacy_inline_fts(cursor):
            self.set_meta("fts_optimize_available", "1", cursor=cursor)
        if current_version < 25:
            # v25: de-duplicate system prompt snapshots into the shared
            # content-addressed table; the old column stays a read fallback
            # for partially migrated or externally written rows.
            self._dedupe_legacy_system_prompts(cursor)

        # Stamp the FTS layout version (fresh/optimized DBs) so the main
        # version can always advance; a legacy DB keeps its absent/0 marker
        # until optimize-storage runs. An INTERRUPTED optimize (rebuild
        # markers, trash tables, or an empty external index against
        # non-empty messages) is NOT stamped: the marker is the source of
        # truth for "fully optimized" and keeps the resume offer alive.
        if (
            fts5_available
            and not self._db_has_legacy_inline_fts(cursor)
            and cursor.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone() is None
            and not self._has_fts_trash(cursor)
            and not self._fts_external_index_empty_with_messages(cursor)
        ):
            self.set_meta("fts_storage_version", str(FTS_STORAGE_VERSION), cursor=cursor)

        # Advance schema_version — deliberately NOT gated on the FTS opt-in
        # (that would block every future migration for a user who never
        # optimizes). FTS5 unavailable is the one skip: we can't have
        # created the current FTS objects, so claiming current would lie.
        if current_version < SCHEMA_VERSION and fts_migrations_complete and fts5_available:
            cursor.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def _migrate_v22_session_model_usage(self, cursor: sqlite3.Cursor) -> None:
        """v22: ``task`` joins the session_model_usage PRIMARY KEY ('' = main
        loop; 'vision'/'compression'/... = aux calls). SQLite cannot ALTER a
        PK, so rebuild; existing rows are main-loop accounting → task=''."""
        try:
            legacy_pk = cursor.execute(
                "SELECT COUNT(*) FROM pragma_table_info('session_model_usage') "
                "WHERE name = 'task' AND pk > 0"
            ).fetchone()[0]
            if legacy_pk:
                return
            cursor.execute("ALTER TABLE session_model_usage RENAME TO session_model_usage_v21")
            cursor.execute(_SESSION_MODEL_USAGE_V22_DDL)
            cursor.execute(
                """INSERT INTO session_model_usage (
                                   session_id, model, billing_provider, billing_base_url,
                                   billing_mode, task, api_call_count, input_tokens,
                                   output_tokens, cache_read_tokens, cache_write_tokens,
                                   reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                                   cost_status, cost_source, first_seen, last_seen
                               )
                               SELECT session_id, model, billing_provider, billing_base_url,
                                      billing_mode, '', api_call_count, input_tokens,
                                      output_tokens, cache_read_tokens, cache_write_tokens,
                                      reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                                      cost_status, cost_source, first_seen, last_seen
                               FROM session_model_usage_v21"""
            )
            cursor.execute("DROP TABLE session_model_usage_v21")
            for sql in _SESSION_MODEL_USAGE_INDEX_SQL:
                cursor.execute(sql)
        except sqlite3.OperationalError as exc:
            logger.debug("v22 session_model_usage rebuild skipped: %s", exc)

    def _ensure_unique_title_index(self, cursor: sqlite3.Cursor) -> None:
        """Unique title index. Older DBs may hold duplicate aliases from before
        the constraint; keep every session, the newest retains the alias. The
        index must never abort opening the DB, so the repair is guarded too."""
        try:
            cursor.execute(_TITLE_UNIQUE_INDEX_SQL)
        except sqlite3.IntegrityError:
            try:
                cursor.execute(
                    """UPDATE sessions AS older
                       SET title = NULL
                       WHERE title IS NOT NULL
                         AND EXISTS (
                             SELECT 1 FROM sessions AS newer
                             WHERE newer.title = older.title
                               AND newer.rowid > older.rowid
                         )"""
                )
                logger.warning(
                    "Cleared %d duplicate session title(s) while restoring the unique index",
                    cursor.rowcount,
                )
                cursor.execute(_TITLE_UNIQUE_INDEX_SQL)
            except sqlite3.Error:
                logger.exception("Could not repair duplicate session titles; unique title index not created")
        except sqlite3.OperationalError:
            pass  # Index already exists

    def _init_fts(self, cursor: sqlite3.Cursor) -> None:
        """Create/repair the FTS objects on an FTS5-capable runtime.

        The DDL runs even when the vtable exists so CREATE TRIGGER IF NOT
        EXISTS repairs trigger-only degradation from a no-FTS5 runtime.
        OPT-IN v23 boundary: a legacy v22 inline install must keep its inline
        schema + triggers (the v23 external-content DDL would create the
        trigram source VIEW and leave a mixed state), so it gets the legacy
        DDL only; fresh/opted-in DBs get v23.
        """
        legacy_fts = self._db_has_legacy_inline_fts(cursor)
        if self._fts_stale:
            if self._recover_stale_fts(cursor, legacy=legacy_fts):
                # CJK was detached alongside the base indexes and has its own
                # stale marker; its ensure path decides when it returns.
                self._ensure_fts_cjk_schema(cursor)
            else:
                self._fts_enabled = False
                self._trigram_available = False
                self._fts_cjk_available = False
        else:
            base_sql, trigram_sql, rebuild = (
                (LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, self._rebuild_legacy_fts_indexes)
                if legacy_fts
                else (FTS_SQL, FTS_TRIGRAM_SQL, self._rebuild_fts_indexes)
            )
            # Measure BEFORE the DDL below runs (pre-repair state). Whether the
            # trigram half is even creatable is only known AFTER
            # _ensure_fts_schema, which is why the halves combine at the `if`.
            base_triggers_missing = self._fts_trigger_count(cursor, _FTS_BASE_TRIGGERS) < len(_FTS_BASE_TRIGGERS)
            trigram_triggers_missing = (
                self._fts_trigger_count(cursor, _FTS_TRIGRAM_TRIGGERS) < len(_FTS_TRIGRAM_TRIGGERS)
            )
            self._fts_enabled = self._ensure_fts_schema(cursor, "messages_fts", base_sql)
            if self._fts_enabled:
                # Trigram is optional relative to the main table; without it
                # CJK search falls back to LIKE.
                trigram_enabled = self._ensure_fts_schema(cursor, "messages_fts_trigram", trigram_sql)
                self._trigram_available = trigram_enabled
                if base_triggers_missing or (trigram_enabled and trigram_triggers_missing):
                    self._run_admitted_startup_rebuild(
                        cursor, lambda: rebuild(cursor, include_trigram=trigram_enabled),
                    )
                if not legacy_fts:
                    # CJK-bigram index: strictly additive, gated on the loadable tokenizer.
                    self._ensure_fts_cjk_schema(cursor)
        # IF NOT EXISTS cannot rewrite pre-existing broad AFTER UPDATE triggers.
        if self._fts_enabled:
            self._migrate_broad_fts_update_triggers(cursor)

    def _run_admitted_startup_rebuild(self, cursor, rebuild_fn) -> None:
        """Run a full trigger-repair FTS rebuild under cross-process admission.

        Reached when the sync triggers were missing and the DDL just recreated
        them: the index has a gap of unknown extent. Two processes opening the
        same DB after an update commonly hit this simultaneously (the
        interleaving that structurally corrupted state.db in production), so
        this admits through ``fts_rebuild_admission`` and FAILS CLOSED. On
        deferral the just-repaired triggers are dropped again and the stale
        breadcrumb persisted — triggers must never be live over an unrebuilt
        gap (``_enter_fts_fail_open``'s ordering contract); the winner's
        rebuild, ``retry_deferred_fts_recovery`` or ``_recover_stale_fts`` at
        next startup restores index and triggers."""
        with fts_rebuild_admission(self.db_path) as admitted:
            if admitted:
                rebuild_fn()
                return
        logger.warning(
            "Deferred startup FTS rebuild: another process holds the "
            "rebuild authority for this state.db; detaching FTS sync "
            "until the stale-index recovery path rebuilds it."
        )
        cursor.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (FTS_STALE_KEY,),
        )
        self._drop_all_fts_triggers(cursor)
        self._fts_stale = True
        self._fts_enabled = False
        self._trigram_available = False
        self._fts_cjk_available = False

    def _backfill_gateway_metadata_from_sessions_json(self, cursor: sqlite3.Cursor) -> None:
        """One-time v18 backfill of gateway metadata from sessions.json.
        Only fills NULL columns — never overwrites data written by newer code."""
        sessions_file = get_hermes_home() / "sessions" / "sessions.json"
        if not sessions_file.exists():
            return
        with open(sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        for key, entry in data.items():
            if str(key).startswith("_") or not isinstance(entry, dict):
                continue
            session_id = entry.get("session_id")
            if not session_id:
                continue
            origin = entry.get("origin")
            origin_dict = origin if isinstance(origin, dict) else None
            cursor.execute(
                """UPDATE sessions
                   SET session_key = COALESCE(session_key, ?),
                       chat_id = COALESCE(chat_id, ?),
                       chat_type = COALESCE(chat_type, ?),
                       thread_id = COALESCE(thread_id, ?),
                       display_name = COALESCE(display_name, ?),
                       origin_json = COALESCE(origin_json, ?),
                       expiry_finalized = CASE
                           WHEN COALESCE(expiry_finalized, 0) = 0 AND ? = 1 THEN 1
                           ELSE expiry_finalized
                       END
                   WHERE id = ?""",
                (
                    entry.get("session_key") or key,
                    origin_dict.get("chat_id") if origin_dict is not None else None,
                    entry.get("chat_type"),
                    origin_dict.get("thread_id") if origin_dict is not None else None,
                    entry.get("display_name"),
                    json.dumps(origin) if origin_dict is not None else None,
                    1 if entry.get("expiry_finalized") or entry.get("memory_flushed") else 0,
                    str(session_id),
                ),
            )
