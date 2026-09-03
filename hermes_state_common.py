"""Shared constants and helpers for the SessionDB family of modules.

Lives outside hermes_state so the mixin modules can import it without a cycle;
hermes_state re-exports every name for backward compatibility.
"""

import contextlib
import errno
import json
import logging
import os
import sys
import time
from typing import Any

from agent.skill_commands import SKILL_EXCERPT_JOINT, SKILL_SCAFFOLD_SQL_LIKE, describe_skill_invocation
from agent.context_compressor import (
    LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER)


# Session preview = head of the first user message, shown wherever a session has no
# title.  A /skill invocation embeds the whole skill body, so its plain head would
# preview the SKILL's prose; scaffolded rows carry a wider excerpt (whole message under
# budget, else head + tail where the typed instruction lands) so ``_shape_preview`` can
# recover ``/work — fix the title leak``.
_PREVIEW_HEAD_CHARS = 63
_PREVIEW_SCAFFOLD_WINDOW = 400
_PREVIEW_MAX_CHARS = 60


def escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``) so derived text matches literally; pair with
    ``ESCAPE '\\'``.  ``_`` is common in branch names, titles and paths, and a documented
    substring/prefix match must not silently widen."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_PREVIEW_CONTENT_SQL = "REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' ')"
_PREVIEW_SCAFFOLDED_SQL = f"m.content LIKE '{SKILL_SCAFFOLD_SQL_LIKE}'"
_SQL_WHITESPACE = "CHAR(9) || CHAR(10) || CHAR(13) || CHAR(32)"


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _sql_ltrim_whitespace(expression: str) -> str:
    return f"LTRIM({expression}, {_SQL_WHITESPACE})"


def _sql_trim_whitespace(expression: str) -> str:
    return f"TRIM({expression}, {_SQL_WHITESPACE})"


def _sql_starts_with(expression: str, prefixes: tuple[str, ...]) -> str:
    trimmed = _sql_ltrim_whitespace(expression)
    return "(" + " OR ".join(f"SUBSTR({trimmed}, 1, {len(p)}) = {_sql_literal(p)}" for p in prefixes) + ")"


def _sql_after_marker(marker: str) -> str:
    """``m.content`` after the first occurrence of *marker*."""
    return f"SUBSTR(m.content, INSTR(m.content, {_sql_literal(marker)}) + {len(marker)})"


# Current and legacy long-form prefixes share this whole introduction; matching all of
# it keeps an ordinary message that merely starts with the bracketed label from
# counting as a compaction carrier.
_PREVIEW_LONG_FORM_PREFIX = SUMMARY_PREFIX.split("Do NOT answer", 1)[0]
_PREVIEW_SUMMARY_PREFIXES = (_PREVIEW_LONG_FORM_PREFIX, LEGACY_SUMMARY_PREFIX)
_PREVIEW_STANDALONE_SUMMARY_SQL = _sql_starts_with("m.content", _PREVIEW_SUMMARY_PREFIXES)
_PREVIEW_MERGED_AFTER_SQL = _sql_after_marker(_MERGED_SUMMARY_DELIMITER)
_PREVIEW_MERGED_SUMMARY_SQL = (
    f"(INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) > 0"
    f" AND {_sql_starts_with(_PREVIEW_MERGED_AFTER_SQL, _PREVIEW_SUMMARY_PREFIXES)})"
)
_PREVIEW_MERGED_PRIOR_SQL = _sql_trim_whitespace(
    f"SUBSTR(m.content, 1, INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) - 1)"
)
_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL = _sql_ltrim_whitespace(_PREVIEW_MERGED_PRIOR_SQL)
_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL = (
    f"CASE WHEN SUBSTR({_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL}, 1,"
    f" {len(_MERGED_PRIOR_CONTEXT_HEADER)}) = {_sql_literal(_MERGED_PRIOR_CONTEXT_HEADER)}"
    f" THEN {_sql_ltrim_whitespace(f'SUBSTR({_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL}, {len(_MERGED_PRIOR_CONTEXT_HEADER) + 1})')}"
    f" ELSE {_PREVIEW_MERGED_PRIOR_SQL} END"
)
_PREVIEW_FORCE_USER_REMAINDER_SQL = _sql_after_marker(_SUMMARY_END_MARKER)

# Pure compaction rows are ineligible for previews; force-user-leading and merged
# carriers are eligible only when authentic content survives.
_PREVIEW_ELIGIBLE_SQL = (
    f"((NOT {_PREVIEW_STANDALONE_SUMMARY_SQL} AND NOT {_PREVIEW_MERGED_SUMMARY_SQL})"
    f" OR ({_PREVIEW_STANDALONE_SUMMARY_SQL}"
    f" AND INSTR(m.content, {_sql_literal(_SUMMARY_END_MARKER)}) > 0"
    f" AND LENGTH({_sql_trim_whitespace(_PREVIEW_FORCE_USER_REMAINDER_SQL)}) > 0)"
    f" OR ({_PREVIEW_MERGED_SUMMARY_SQL}"
    f" AND LENGTH({_sql_trim_whitespace(_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL)}) > 0))"
)

# Shared ``_preview_raw`` SELECT expression for every listing query (scaffolded rows:
# head + tail spliced around SKILL_EXCERPT_JOINT when over budget).
_PREVIEW_RAW_SELECT = (
    f"CASE WHEN {_PREVIEW_STANDALONE_SUMMARY_SQL}"
    f" THEN {_PREVIEW_FORCE_USER_REMAINDER_SQL}"
    f" WHEN {_PREVIEW_MERGED_SUMMARY_SQL}"
    f" THEN {_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL}"
    f" WHEN {_PREVIEW_SCAFFOLDED_SQL}"
    f" AND LENGTH(m.content) > {_PREVIEW_SCAFFOLD_WINDOW * 2}"
    f" THEN SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW})"
    f" || '{SKILL_EXCERPT_JOINT}'"
    f" || SUBSTR({_PREVIEW_CONTENT_SQL}, -{_PREVIEW_SCAFFOLD_WINDOW})"
    f" WHEN {_PREVIEW_SCAFFOLDED_SQL}"
    f" THEN SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW * 2})"
    f" ELSE SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_HEAD_CHARS}) END"
)


def _shape_preview(raw: Any) -> str:
    """Turn a ``_preview_raw`` column into the short preview callers show."""
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    described = describe_skill_invocation(text)
    text = described if described is not None else text.split(SKILL_EXCERPT_JOINT)[0]
    return text[:_PREVIEW_MAX_CHARS] + "..." if len(text) > _PREVIEW_MAX_CHARS else text


# Correlated ``_preview_raw`` column for a ``sessions s`` row.
_PREVIEW_RAW_SUBQUERY_SQL = (
    f"COALESCE((SELECT {_PREVIEW_RAW_SELECT} FROM messages m"
    f" WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL"
    f" AND {_PREVIEW_ELIGIBLE_SQL}"
    f" ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw"
)

# ── Session lineage predicates ({a} = sessions alias) ───────────────────────

# A /branch child (kept visible, never cascade-deleted): stable marker OR the legacy
# end_reason heuristic.
_BRANCH_CHILD_SQL = (
    "json_extract(COALESCE({a}.model_config, '{{}}'), '$._branched_from') IS NOT NULL"
    " OR EXISTS (SELECT 1 FROM sessions p            WHERE p.id = {a}.parent_session_id"
    "            AND p.end_reason = 'branched'            AND {a}.started_at >= p.ended_at)"
)
_COMPRESSION_CHILD_SQL = (
    "EXISTS (SELECT 1 FROM sessions p        WHERE p.id = {a}.parent_session_id"
    "        AND p.end_reason = 'compression')"
)

_RESET_END_REASONS = (
    "session_reset",
    # switch_session() creates no child row, but pre-marker DBs hold legacy reset
    # children whose parent later ended 'session_switch'.  Must stay identical to the
    # recovery fence in find_latest_gateway_session_for_peer (interpolates the SQL form).
    "session_switch",
    "idle",
    "daily",
    "suspended",
    "resume_pending_expired",
)
_RESET_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RESET_END_REASONS)

# Accidental end reasons recovery treats as resumable (docs/session-lifecycle.md).
# Single source of truth: interpolated into recovery SQL AND exposed as
# SessionDB.RECOVERABLE_END_REASONS.
_RECOVERABLE_END_REASONS = (
    "agent_close",
    "ws_orphan_reap",
    # Stale sentinel-parked runtime superseded by a fresh session.resume.
    "superseded_by_resume",
    # Startup sweep of rows orphaned by a dead gateway process: same accident class as
    # ws_orphan_reap, kept distinct for forensics.
    "startup_orphan_reap",
)
_RECOVERABLE_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RECOVERABLE_END_REASONS)

# End reasons written by AUTOMATIC cleanup (shutdown, orphan reapers, idle/LRU eviction),
# not by a deliberate conversation boundary: "some runtime went away", NOT "this
# conversation ended", so a writer that can prove liveness (e.g. a compression rotation
# holding the lease) may clear it.  Superset of the recoverable set plus the TUI
# gateway's automatic reasons.
_AUTOMATIC_END_REASONS = frozenset(_RECOVERABLE_END_REASONS) | {
    "tui_shutdown", "ws_disconnect", "idle_timeout", "lru_evict"}


def is_automatic_end_reason(reason) -> bool:
    """True when *reason* is an automatic-cleanup end stamp.  Single owner of the
    accidental-vs-deliberate predicate; compression-liveness sites must call this."""
    return isinstance(reason, str) and reason in _AUTOMATIC_END_REASONS


def _legacy_reset_child_sql(alias: str, reasons_sql: str) -> str:
    """Pre-marker reset-continuation heuristic: child rides its parent's exact non-empty routing key and the
    parent ended at a reset boundary.  Shared by ``_RESET_CHILD_SQL`` and ``reopen_session()``'s
    marker-stamping UPDATE so the two cannot drift; ``reasons_sql`` is a literal or placeholder list."""
    return (
        f"EXISTS (SELECT 1 FROM sessions p"
        f"            WHERE p.id = {alias}.parent_session_id"
        f"            AND p.end_reason IN ({reasons_sql})"
        f"            AND {alias}.session_key IS NOT NULL"
        f"            AND {alias}.session_key != ''"
        f"            AND {alias}.session_key = p.session_key)"
    )


# A reset starts a separate user-visible conversation though rows keep parent_session_id
# for lineage.  Stable marker, or the same-key fallback for pre-marker rows (the
# exact-key requirement keeps subagent children out).
_RESET_CHILD_SQL = (
    "json_extract(COALESCE({a}.model_config, '{{}}'), '$._reset_from') IS NOT NULL"
    " OR " + _legacy_reset_child_sql("{a}", _RESET_END_REASONS_SQL)
)

# Picker-visible rows: roots + branch/reset children (not subagent runs or compression
# continuations).
_LISTABLE_CHILD_SQL = (
    f"(s.parent_session_id IS NULL OR {_BRANCH_CHILD_SQL.format(a='s')}"
    f" OR {_RESET_CHILD_SQL.format(a='s')})"
)


def _ephemeral_child_sql(alias: str = "s") -> str:
    """Subagent runs, not branch, reset, or compression children."""
    return (
        f"({alias}.parent_session_id IS NOT NULL"
        f" AND NOT ({_BRANCH_CHILD_SQL.format(a=alias)})"
        f" AND NOT ({_COMPRESSION_CHILD_SQL.format(a=alias)})"
        f" AND NOT ({_RESET_CHILD_SQL.format(a=alias)}))"
    )


def _sql_freshest_of(activity: str, session_id_expr: str, started: str) -> str:
    """Freshest of *activity* and the latest message timestamp for *session_id_expr*,
    else *started*.  Heartbeats are rate-limited (~60s) so ``last_activity_at`` can lag
    a newer message; never prefer it alone."""
    msg_max = f"(SELECT MAX(_act_m.timestamp) FROM messages _act_m WHERE _act_m.session_id = {session_id_expr})"
    return (
        f"COALESCE("
        f"(SELECT MAX(_act_v.v) FROM ("
        f"SELECT {activity} AS v "
        f"UNION ALL "
        f"SELECT {msg_max}"
        f") _act_v), "
        f"{started})"
    )


def _sql_session_last_active(alias: str = "s") -> str:
    """Session recency expression for a ``sessions {alias}`` row."""
    return _sql_freshest_of(f"{alias}.last_activity_at", f"{alias}.id", f"{alias}.started_at")


def _sql_session_last_active_by_id(session_id_expr: str) -> str:
    """Same freshest-of expression keyed by a session-id SQL expression."""
    return _sql_freshest_of(
        f"(SELECT last_activity_at FROM sessions _act_s WHERE _act_s.id = {session_id_expr})",
        session_id_expr,
        f"(SELECT started_at FROM sessions _act_s WHERE _act_s.id = {session_id_expr})")


SCHEMA_VERSION = 28

# Auto-maintenance VACUUMs only when at least this fraction of pages is on the freelist;
# below it a full rewrite costs more I/O than it returns.
AUTO_VACUUM_MIN_FREELIST_RATIO = 0.25

# FTS storage layout, tracked INDEPENDENTLY of SCHEMA_VERSION (state_meta
# ``fts_storage_version``): schema version advances freely on open, the FTS layout only
# changes when a DB is born fresh or explicitly optimized via ``hermes sessions
# optimize-storage``.  Legacy DBs sit at 0 (marker absent) with a working inline index;
# 1 = v23 external-content layout.
FTS_STORAGE_VERSION = 1

# Cap on user-controlled FTS5 query input before sanitizer processing.
MAX_FTS5_QUERY_CHARS = 2_048


def stat_db_file_identity(path) -> "tuple[int, int] | None":
    """``(st_dev, st_ino)`` for *path*, or None.  st_ino=0 (Windows, some network FS)
    would false-positive every replaced-file check, so it counts as unknown."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino) if st.st_dev and st.st_ino else None


# Row probes shared by the messages / compression mixins.
_ENDED_ROW_SQL = "SELECT ended_at, end_reason FROM sessions WHERE id = ?"
_COMPRESSION_LOCK_ROW_SQL = "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?"


def _ended_by_compression(row) -> bool:
    return row is not None and row["ended_at"] is not None and row["end_reason"] == "compression"


def _placeholders(items) -> str:
    """``?,?,?`` for one bound parameter per element of *items* (a sequence or an int count)."""
    return ",".join("?" for _ in range(items if isinstance(items, int) else len(items)))


_FTS_TRIGGERS = (
    "messages_fts_insert", "messages_fts_delete", "messages_fts_update",
    "messages_fts_trigram_insert", "messages_fts_trigram_delete", "messages_fts_trigram_update")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    compression_recovery_deadline REAL,
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    tool_names TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
);

CREATE TABLE IF NOT EXISTS session_model_usage (
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
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);

CREATE TABLE IF NOT EXISTS gateway_hygiene_state (
    session_key TEXT PRIMARY KEY,
    failure_streak INTEGER NOT NULL DEFAULT 0
);

-- Monotonic conversation generation per routing peer (#96811).
--
-- A host-declared conversation key (X-Hermes-Session-Key / build_session_key)
-- is per-CHAT and outlives any single conversation on it, so the prompt-cache
-- affinity scope derived from it must be qualified by which conversation is
-- currently live. Deriving that from the session rows themselves
-- (COUNT/MAX over _RESET_END_REASONS boundaries) cannot prove non-reuse:
-- delete_session() and bulk pruning remove ended rows, so an aggregate can
-- return a pair it already emitted and hand a new conversation a retired
-- affinity identity.
--
-- This counter lives outside prunable session history and only ever
-- increments, once per boundary actually written, so a generation can never
-- be reused for a peer even if every session row behind it is deleted.
--
-- These rows are deliberately NEVER garbage-collected, including when every
-- session row for the peer is gone. Collecting one resets that peer to "no
-- generation", so its next boundary writes generation = 1 again and re-issues
-- a gwk_ scope a retired conversation already used — exactly the ABA this
-- table exists to close. Do not add it to delete_session()'s cascade or to any
-- prune sweep. One (TEXT, TEXT, INTEGER) row per routing peer is the intended,
-- bounded cost.
CREATE TABLE IF NOT EXISTS conversation_generations (
    source TEXT NOT NULL,
    session_key TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, session_key)
);

-- Per-backend liveness heartbeat (#94895). Each serve / tui_gateway process
-- registers a row at startup and refreshes ``last_heartbeat`` periodically.
-- The startup orphan sweep (sessions.startup_orphan_reap) consults this
-- table to avoid reaping rows whose owning backend is still alive but
-- just idle (multi-backend state.db shared by isolated serve processes).
-- A backend whose ``last_heartbeat`` is older than the heartbeat staleness
-- window is treated as dead; rows without ANY matching heartbeat fall back
-- to the original staleness predicate so legacy deployments keep working.
CREATE TABLE IF NOT EXISTS gateway_heartbeats (
    backend_id TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    started_at REAL NOT NULL,
    last_heartbeat REAL NOT NULL,
    profile TEXT NOT NULL DEFAULT '',
    host TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at REAL NOT NULL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivered_at REAL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    task_json TEXT,
    delivery_claim TEXT,
    delivery_claimed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
-- Partial index for the Insights assistant tool-call scan
-- (agent/insights.py _get_tool_usage / _get_skill_usage): those queries filter
-- messages by role='assistant' AND tool_calls IS NOT NULL, a small fraction of
-- rows on a large state.db. role and tool_calls are base columns, so this can
-- live in SCHEMA_SQL rather than DEFERRED_INDEX_SQL.
CREATE INDEX IF NOT EXISTS idx_messages_assistant_calls_by_session
    ON messages(session_id)
    WHERE role = 'assistant' AND tool_calls IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_turn_leases_expires ON session_turn_leases(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model);
CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
"""

# Indexes on columns added in later schema versions must run AFTER
# _reconcile_columns() adds them, or executescript fails on legacy DBs.
DEFERRED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash
    ON sessions(system_prompt_hash);
"""

# ── Deferred FTS rebuild bookkeeping ──
# While a background rebuild is pending, two state_meta keys define which rows
# are IN the FTS indexes: H = fts_rebuild_high_water (MAX(messages.id) when the
# old indexes were dropped), P = fts_rebuild_progress (highest backfilled id).
# A row is indexed iff id <= P OR id > H (AUTOINCREMENT ids: post-drop rows are
# indexed live by the insert triggers); rows in (P, H] are not.  Every trigger
# gates on that predicate: an external-content 'delete' for a row NOT in the
# index corrupts it, and skipping one for an indexed row leaves a stale entry.
# With no rebuild pending both keys are absent and COALESCE makes it a tautology.
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

-- UPDATE OF skips the trigger entirely for non-content column writes
-- (status/compacted/observed/etc.), which is stronger than the WHEN gate
-- alone and avoids FTS I/O saturation on large state.db (#68858 / #73639).
CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
"""

# Trigram FTS5 table for CJK substring search (unicode61 splits CJK into single
# tokens, breaking phrase matching).  The trigram index is ~2.6x the text it
# covers and ``role='tool'`` rows are ~90% of message bytes of machine noise,
# so it reads through the ``messages_fts_trigram_src`` view, which excludes
# tool rows; those remain searchable via ``messages_fts``, and
# ``search_messages`` routes CJK queries filtered on role='tool' to LIKE.
FTS_TRIGRAM_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""

_FTS_CJK_TRIGGERS = (
    "messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update")

# Set when a tokenizer-less process dropped the cjk triggers to keep writes
# alive: the cjk index is missing rows and must not serve reads until
# `hermes sessions optimize-storage` rebuilds it on a capable host.
FTS_CJK_STALE_KEY = "fts_cjk_stale"

# Set when a base/trigram FTS index was detached after runtime corruption.
# While present, startup must rebuild the complete index before reinstalling
# sync triggers: rows written while they were absent leave an unknown gap.
FTS_STALE_KEY = "fts_stale"

# Durable diagnostic for stale FTS recovery blocked across process restarts.
FTS_REBUILD_DEFERRAL_KEY = "fts_rebuild_deferral"

# ── Legacy (v22 / inline-content) FTS DDL ──────────────────────────────
# Used ONLY to keep a pre-v23 install's search working and its triggers
# repairable until `optimize_fts_storage()` migrates it: inline copies of
# content || tool_name || tool_calls, trigram over every row.  Never created
# on a fresh install.  Handing a legacy DB the v23 DDL would create the
# external-content trigram VIEW and leave it in a mixed, broken state.
LEGACY_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

LEGACY_FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

# ── Cross-process full-FTS-rebuild admission (single authority) ──────────────
# Several Hermes processes share one state.db; a full structural FTS rebuild
# (FTS5 'rebuild' or the drop/recreate in `_recover_stale_fts`) must run in ONE
# of them at a time — concurrent rebuilds structurally corrupted state.db in
# production.  Single authority for `rebuild_fts()`, `_rebuild_fts_indexes()`
# and `_recover_stale_fts()`; the chunked backfill (`fts_rebuild_step`) is
# deliberately NOT routed through it (it claims progress under SQLite
# transaction authority and is multi-process).
# Semantics mirror `hermes_state._cross_process_repair_lock`: portable (msvcrt
# on Windows, flock elsewhere), bounded wait, FAIL CLOSED.  flock rides the open
# file description, so a forked child that inherited the fd holds it forever
# after the holder dies; the holder's pid + start time are recorded under the
# lock and a provably-dead holder's lock is broken by unlinking and retaking on
# a fresh inode.  Indeterminate liveness still defers.  `<db>.fts_rebuild.lock`
# is distinct from `<db>.repair.lock` (schema surgery on an EXCLUSIVE offline
# connection, minutes in VACUUM).  Lives here because mixins cannot import
# hermes_state (cycle).

logger = logging.getLogger("hermes_state")

_FTS_REBUILD_LOCK_TIMEOUT_SECONDS = 120.0
_FTS_REBUILD_LOCK_POLL_SECONDS = 0.1
_IS_WINDOWS = sys.platform == "win32"

# Post-break re-acquire budget: the fresh inode is contended only by live
# processes, so a short wait suffices — never re-enter the full timeout.
_LOCK_BREAK_REACQUIRE_SECONDS = 5.0

# "Another process holds the lock": flock → EWOULDBLOCK/EAGAIN, msvcrt.locking
# → EACCES (EDEADLK when its retry gives up).  Anything else (ESTALE, ENOTSUP,
# ENOLCK, EIO) is a persistent environment failure that polling cannot fix;
# treating it as contention burned the full timeout on every attempt.
_LOCK_CONTENTION_ERRNOS = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}
if hasattr(errno, "EDEADLK"):
    _LOCK_CONTENTION_ERRNOS.add(errno.EDEADLK)


def is_advisory_lock_contention(exc: BaseException) -> bool:
    """True when *exc* means another process holds the advisory lock.  For any
    other ``OSError`` callers must fail closed IMMEDIATELY: retrying cannot succeed."""
    return isinstance(exc, BlockingIOError) or (isinstance(exc, OSError) and exc.errno in _LOCK_CONTENTION_ERRNOS)


def _proc_start_ticks(pid: int):
    """Kernel start time of *pid* (field 22 of ``/proc/<pid>/stat``), which with the PID uniquely identifies
    a process; None off Linux or on any failure — callers must treat None as unknowable and FAIL CLOSED."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read()
        # comm (field 2) may contain spaces/parens; split after the LAST ')'.
        return int(stat.rsplit(b")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _read_lock_holder_record(handle):
    """Best-effort parse of the holder metadata JSON in a lock file."""
    try:
        handle.seek(0)
        raw = handle.read(4096)
        record = json.loads(raw.decode("utf-8", "replace")) if raw else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _rewrite_lock_file(handle, payload: bytes) -> None:
    """Best-effort truncate-and-write of *payload* at offset 0."""
    try:
        handle.seek(0)
        handle.truncate()
        if payload:
            handle.write(payload)
        handle.flush()
    except (OSError, ValueError):
        pass


def _write_lock_holder_record(handle) -> None:
    """Record this process as holder (best effort), written under the flock so timed-out
    contenders can tell an orphaned-fd holder from a live wedged one."""
    record = {"pid": os.getpid(), "start_ticks": _proc_start_ticks(os.getpid()), "acquired_at": time.time()}
    _rewrite_lock_file(handle, json.dumps(record, sort_keys=True).encode("utf-8"))


def _clear_lock_holder_record(handle) -> None:
    """Erase holder metadata before a normal release, so a surviving record always
    means an ABNORMAL exit — the only condition allowing a break."""
    _rewrite_lock_file(handle, b"")


def _lock_holder_provably_dead(record) -> bool:
    """True ONLY when the recorded holder is provably dead or PID-recycled. Anything
    indeterminate (no/malformed record, PID owned by another user, /proc unavailable)
    is False — the caller must FAIL CLOSED and defer."""
    if not isinstance(record, dict):
        return False
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # PermissionError et al.: PID exists (or unknowable) — closed
    recorded_ticks = record.get("start_ticks")
    if recorded_ticks is None:
        return False
    current_ticks = _proc_start_ticks(pid)
    # Same PID, different start time: recycled by an unrelated process.
    return current_ticks is not None and current_ticks != recorded_ticks


def _acquire_db_flock(lock_path, handle, timeout_seconds, poll_seconds, description):
    """Bounded POSIX flock acquire with orphaned-holder staleness break.

    Returns ``(acquired, handle)``; *handle* may have been re-opened and the caller
    closes whichever comes back. *acquired* is True, False (a holder kept the lock past
    the deadline), or None (non-contention ``OSError``, already logged; callers treat it
    as not acquired without the held-by-another-process warning).

    ``flock`` belongs to the open file DESCRIPTION, which ``fork()`` duplicates: a holder
    that forks then dies leaves the lock held forever by a child that never releases.
    When the process that ACQUIRED is provably dead yet the flock is held, the file is
    unlinked and retaken on a fresh inode; the orphan's flock stays on the old inode
    blocking nobody. Every successful acquire verifies its inode still names
    *lock_path*, so a racer that locked a dead inode retries instead of running
    alongside the breaker. Indeterminate liveness defers."""
    import fcntl

    deadline = time.monotonic() + timeout_seconds
    broke_lock = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if not is_advisory_lock_contention(exc):
                # Not a holder and polling cannot fix it: defer NOW.
                logger.warning(
                    "Could not acquire %s %s (%s) — deferring rather than "
                    "waiting out the %.0fs holder timeout on a non-contention error.",
                    description, lock_path, exc, timeout_seconds)
                return None, handle
            if time.monotonic() < deadline:
                time.sleep(poll_seconds)
                continue
            if broke_lock:
                return False, handle
            record = _read_lock_holder_record(handle)
            if not _lock_holder_provably_dead(record):
                return False, handle
            logger.warning(
                "%s %s is held by an orphaned file descriptor (recorded "
                "holder pid %s is dead — a forked child inherited the lock "
                "fd); breaking the stale lock and retaking it on a fresh file.",
                description, lock_path, (record or {}).get("pid"))
            try:
                os.unlink(lock_path)
                handle.close()
                handle = open(lock_path, "a+b")
            except OSError as exc:
                logger.warning("Could not break stale %s %s (%s) — deferring.", description, lock_path, exc)
                return False, handle
            broke_lock = True
            deadline = time.monotonic() + _LOCK_BREAK_REACQUIRE_SECONDS
            continue
        # Verify the path still names our inode: a breaker may have replaced
        # the file while we waited, and a lock on a dead inode excludes nobody.
        try:
            fd_stat = os.fstat(handle.fileno())
            path_stat = os.stat(lock_path)
            same_file = fd_stat.st_dev == path_stat.st_dev and fd_stat.st_ino == path_stat.st_ino
        except OSError:
            same_file = False
        if same_file:
            _write_lock_holder_record(handle)
            return True, handle
        try:
            handle.close()
            handle = open(lock_path, "a+b")
        except OSError:
            return False, handle
        if time.monotonic() >= deadline:
            return False, handle


def _describe_lock_holder(record) -> str:
    """Human-readable holder identity for deferral warnings."""
    if not isinstance(record, dict) or "pid" not in record:
        return "unknown (no holder record; pre-fix writer or non-Hermes)"
    age = ""
    try:
        if record.get("acquired_at") is not None:
            age = f", acquired {time.time() - float(record['acquired_at']):.0f}s ago"
    except (TypeError, ValueError):
        pass
    return f"pid {record.get('pid')}{age}"


def _acquire_msvcrt_lock(lock_path, handle, timeout):
    """Windows counterpart of ``_acquire_db_flock`` (no orphan break); same True / False / None contract."""
    import msvcrt

    deadline = time.monotonic() + timeout
    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (BlockingIOError, OSError) as exc:
            if not is_advisory_lock_contention(exc):
                logger.warning(
                    "Could not acquire FTS rebuild lock %s (%s) — deferring on a non-contention error.",
                    lock_path, exc)
                return None
            if time.monotonic() >= deadline:
                return False
            time.sleep(_FTS_REBUILD_LOCK_POLL_SECONDS)


@contextlib.contextmanager
def fts_rebuild_admission(db_path, *, timeout_seconds=None):
    """Serialize full structural FTS rebuilds on *db_path* across processes.

    Yields True when this process holds the authority, False when the bounded acquire timed out or the lock
    file could not be opened. On False the caller must NOT rebuild (fail closed); the stale breadcrumb
    guarantees a retry. ``db_path`` None (in-memory DB) yields True. Opportunistic in-process retries pass
    ``timeout_seconds=0`` so a live holder never stalls a long-lived writer; the orphan break still applies.
    """
    if db_path is None:
        yield True
        return
    timeout = _FTS_REBUILD_LOCK_TIMEOUT_SECONDS if timeout_seconds is None else max(float(timeout_seconds), 0.0)
    lock_path = f"{db_path}.fts_rebuild.lock"
    try:
        handle = open(lock_path, "a+b")
    except OSError as exc:
        # Fail closed like a timed-out acquire: an unopenable lock file means the FS is
        # out of space/inodes/descriptors, and a sibling that opened its handle earlier
        # may still be rebuilding — yielding True gave every process on a full disk a
        # concurrent rebuild of the same DB. Deferring costs nothing (the breadcrumb
        # retries, and the rebuild's own writes could not have committed either).
        logger.warning(
            "Could not open FTS rebuild lock %s (%s) — deferring this rebuild "
            "rather than running it without cross-process authority.",
            lock_path, exc)
        yield False
        return
    acquired = False
    try:
        if _IS_WINDOWS:
            acquired = _acquire_msvcrt_lock(lock_path, handle, timeout)
        else:
            acquired, handle = _acquire_db_flock(
                lock_path, handle, timeout, _FTS_REBUILD_LOCK_POLL_SECONDS, "FTS rebuild lock")
        if acquired is None:
            # Already logged with the real errno; "held by another process" would be a lie.
            acquired = False
        elif not acquired:
            record = None if _IS_WINDOWS else _read_lock_holder_record(handle)
            if timeout <= 0:
                # Non-blocking probe from an in-process retry: keep it quiet.
                logger.info(
                    "FTS rebuild lock %s is busy — deferring this retry "
                    "(the stale-FTS breadcrumb keeps it retryable). Recorded holder: %s.",
                    lock_path, _describe_lock_holder(record))
            else:
                logger.warning(
                    "FTS rebuild lock %s held by another process for more than "
                    "%.0fs — deferring this rebuild to avoid racing the holder "
                    "(the stale-FTS breadcrumb keeps it retryable). Recorded holder: %s.",
                    lock_path, timeout, _describe_lock_holder(record))
        yield acquired
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    _clear_lock_holder_record(handle)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - best effort release
            pass
        finally:
            handle.close()
