#!/usr/bin/env python3
"""SQLite state store for Hermes Agent: session metadata, message history, model
config, FTS5 search. WAL mode (concurrent readers + one writer); compression
splits sessions via parent_session_id chains; sessions are source-tagged
('cli', 'telegram', ...). Batch-runner / RL trajectories live elsewhere.
"""

import asyncio
import atexit
import errno
import hashlib
import json
import logging
import os
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import uuid
import weakref
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from agent.session_activity import ActivityProvenance
from agent.message_sanitization import _sanitize_surrogates
# Known-durable message marker, shared with agent.context_compressor. run_agent
# keeps its own copy (cannot import hermes_state: circular), guarded by
# test_marker_constant_in_sync.
from agent.context_compressor import (  # noqa: F401  (re-exported; tests import it from here)
    _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY,
)
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, cast

from hermes_state_common import (  # noqa: F401  (re-exported for back-compat)
    AUTO_VACUUM_MIN_FREELIST_RATIO, _BRANCH_CHILD_SQL, _COMPRESSION_CHILD_SQL, _FTS_CJK_TRIGGERS,
    _FTS_TRIGGERS, _LISTABLE_CHILD_SQL, _PREVIEW_ELIGIBLE_SQL, _PREVIEW_RAW_SELECT,
    _RECOVERABLE_END_REASONS, _RECOVERABLE_END_REASONS_SQL, is_automatic_end_reason,
    _RESET_END_REASONS, _RESET_END_REASONS_SQL, _ephemeral_child_sql, _legacy_reset_child_sql,
    _shape_preview, _sql_session_last_active, _sql_session_last_active_by_id,
    escape_like as _escape_like, DEFERRED_INDEX_SQL, FTS_CJK_STALE_KEY, FTS_REBUILD_DEFERRAL_KEY,
    FTS_SQL, FTS_STALE_KEY, FTS_STORAGE_VERSION, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL, MAX_FTS5_QUERY_CHARS, SCHEMA_SQL, SCHEMA_VERSION, _PREVIEW_CONTENT_SQL,
    _PREVIEW_HEAD_CHARS, _PREVIEW_MAX_CHARS, _PREVIEW_SCAFFOLD_WINDOW, _PREVIEW_SCAFFOLDED_SQL,
    _acquire_db_flock, _clear_lock_holder_record, _describe_lock_holder, _read_lock_holder_record,
    is_advisory_lock_contention, stat_db_file_identity as _stat_db_file_identity,
)
from hermes_state_portability import SessionPortabilityMixin
from hermes_state_telegram import SessionTelegramTopicsMixin, _normalize_telegram_topic_profile_name  # noqa: F401  (re-exported for back-compat)
from hermes_state_schema import SessionSchemaMixin
from hermes_state_dbfile import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _HEADER_PROBE_FDS, _HEADER_PROBE_LOCK, _HERMES_CMDLINE_MARKERS, _RETIRED_HEADER_PROBE_FDS,
    _canonical_sqlite_path, _concrete_state_db_holder_pids, _connect_tracked_db,
    _is_inactive_orphan_desktop_holder, _looks_like_hermes, _pread_db_header, _read_proc_cmdline,
    _read_sqlite_application_id, _stat_sqlite_sidecar_identity, _watched_sqlite_sidecar_paths,
    collect_state_db_stats, count_db_holders, is_zeroed_state_db,
    iter_deleted_sqlite_sidecar_holders, quarantine_cross_process_lock, quarantine_zeroed_state_db,
    refuse_deleted_wal_generation,
)
from hermes_state_messages import SessionMessagesMixin
from hermes_state_wal import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    WalUnsupportedError, _SYNCHRONOUS_FULL, _SYNCHRONOUS_LEVELS, _SYNCHRONOUS_NAMES,
    _WAL_INCOMPAT_MARKERS, _WAL_SIZE_LIMIT_BYTES, _apply_delete_for_wal_reset_bug,
    _apply_macos_checkpoint_barrier, _apply_synchronous_pragma, _apply_wal_size_limit,
    _database_has_content, _delete_overridden_warned_lock, _delete_overridden_warned_paths,
    _enforce_macos_synchronous_full, _journal_upgrade_warned_lock, _journal_upgrade_warned_paths,
    _log_configured_delete_overridden_once, _log_journal_mode_upgrade_once, _log_wal_fallback_once,
    _log_wal_reset_bug_once, _on_disk_journal_mode, _set_journal_mode_no_wait,
    _wal_fallback_warned_lock, _wal_fallback_warned_paths, _wal_reset_bug_warned_lock,
    _wal_reset_bug_warned_paths, _wal_reset_repair_hint, apply_database_pragmas,
    apply_wal_with_fallback, is_sqlite_wal_reset_vulnerable, resolve_journal_mode,
    resolve_synchronous_level, sqlite_source_id,
)
from hermes_state_repair import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _DB_SIDECAR_SUFFIXES, _FINGERPRINT_SAMPLE_BYTES, _FINGERPRINT_VOLATILE_HEADER_RANGES,
    _MAX_MALFORMED_BACKUPS, _MAX_PERSISTENT_REPAIR_ATTEMPTS, _REPAIR_BACKUP_FREE_FRACTION,
    _REPAIR_BACKUP_MIN_FREE_BYTES, _REPAIR_LOCK_POLL_SECONDS,
    _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND, _backup_content_identity, _backup_db_file,
    _bump_schema_cookie, _claim_repair_attempt, _connect_repair_durable, _copy_database_snapshot,
    _cross_process_repair_lock, _db_fingerprint, _db_opens_cleanly, _exclusive_repair_db_guard,
    _existing_malformed_backups, _live_writer_holds_db, _mask_volatile_header,
    _persistent_repair_attempts_exhausted, _persistent_repair_exhausted_error,
    _probe_journal_mode_for_repair, _prune_malformed_backups, _read_repair_ledger,
    _reapply_durability_barriers, _record_repair_outcome, _release_auto_maintenance_lock,
    _repair_backup_headroom_bytes, _repair_failure_consumes_attempt, _repair_ledger_path,
    _repair_scratch_space_error, _repair_snapshot_timeout_seconds, _repair_state_db_schema_locked,
    _restore_journal_mode_after_repair, _run_repair_strategies, _try_acquire_auto_maintenance_lock,
    _unlink_db_triple, apply_durability_barriers, preflight_db_writability, repair_state_db_schema,
)
from hermes_state_titles import SessionTitlesMixin
from hermes_state_usage import SessionUsageMixin
from hermes_state_maintenance import SessionMaintenanceMixin
from hermes_state_gateway import SessionGatewayMixin
from hermes_state_compression import SessionCompressionMixin
from hermes_state_search import SessionSearchMixin

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MAX_SAFE_RESUME_MESSAGES = 20_000
MAX_SAFE_EXPORT_MESSAGES = 20_000


def _configured_transcript_limit(key: str, fallback: int) -> int:
    """``sessions.<key>`` from config.yaml (lazy import: circular at load), else
    *fallback*. 0 disables the guard. Not cached: load_config_readonly is
    mtime-cached already, and fresh resolution keeps monkeypatching tests working."""
    try:
        from hermes_cli.config import load_config_readonly
        sessions_cfg = load_config_readonly().get("sessions") or {}
        value = sessions_cfg.get(key)
        if value is None:
            return fallback
        limit = int(value)
        return limit if limit >= 0 else fallback
    except Exception:
        return fallback


def resolved_max_resume_messages() -> int:
    """Config-resolved resume guard limit (0 disables the guard)."""
    return _configured_transcript_limit("max_resume_messages", MAX_SAFE_RESUME_MESSAGES)


def resolved_max_export_messages() -> int:
    """Config-resolved in-memory export guard limit (0 disables the guard)."""
    return _configured_transcript_limit("max_export_messages", MAX_SAFE_EXPORT_MESSAGES)


class SessionResumeTooLargeError(ValueError):
    def __init__(
        self, message_count: int, limit: int = MAX_SAFE_RESUME_MESSAGES,
        scope: str = "across its lineage",
    ):
        self.message_count = message_count
        self.limit = limit
        super().__init__(
            f"session has at least {message_count} active messages {scope}; "
            f"safe resume limit is {limit}. Export the session instead, or set "
            "sessions.max_resume_messages: 0 in config.yaml to disable the guard."
        )


class SessionExportTooLargeError(ValueError):
    def __init__(self, session_id: str, message_count: int, limit: int = MAX_SAFE_EXPORT_MESSAGES):
        self.session_id = session_id
        self.message_count = message_count
        self.limit = limit
        super().__init__(
            f"session '{session_id}' has at least {message_count} active messages; "
            f"safe in-memory export limit is {limit}"
        )


_COMPRESSION_LOCK_HOLDER_PID_RE = re.compile(r"(?:^|:)pid=(\d+)(?::|$)")


def _system_prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """True only when a ``pid=<n>`` lock holder's local PID is provably gone.

    A process killed mid-compression cannot release its lease and every new
    turn would re-attempt compaction until TTL expiry. Reclaim only on kernel
    proof; unstructured/same-process holders and any probe doubt stay protected
    (PID reuse must never steal a live lease; a wrongly-kept lease self-heals via TTL).
    """
    match = _COMPRESSION_LOCK_HOLDER_PID_RE.search(holder or "")
    if match is None:
        return False
    try:
        pid = int(match.group(1))
    except (TypeError, ValueError):
        return False
    # Same-process holder (another thread's live lease): never self-reclaim —
    # the lease refresher and release path own it.
    if pid <= 0 or pid == os.getpid():
        return False
    if psutil is not None:
        try:
            # Canonical cross-platform liveness answer; recycled PIDs read as alive (conservative).
            return not psutil.pid_exists(pid)
        except Exception:
            return False  # any doubt → keep the lease until TTL expiry
    # psutil-less fallback is POSIX-only: os.kill(pid, 0) is NOT a no-op probe on
    # Windows (sig=0 maps to CTRL_C_EVENT and can kill the target's console group).
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — nt early-returns just above
    except ProcessLookupError:
        return True
    except (PermissionError, OSError, OverflowError):
        return False
    return False


def _scrub_surrogates(value: Any) -> Any:
    """Replace lone surrogates in text (sqlite3 raises UnicodeEncodeError on them,
    aborting the whole write); pass anything else through."""
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def workspace_key(row: Dict[str, Any]) -> Optional[str]:
    """Workspace grouping key: git repo root when known, else cwd, else None.
    Branch is deliberately excluded so a checkout doesn't fragment history."""
    return (row.get("git_repo_root") or "").strip() or (row.get("cwd") or "").strip() or None


def _delegate_from_json(col: str = "model_config") -> str:
    return f"json_extract(COALESCE({col}, '{{}}'), '$._delegate_from')"


# _merge_model_config_json's "no such row" result — distinct from the legal None
# ("merged config is empty → store NULL").
_MODEL_CONFIG_ROW_MISSING = object()


def _parse_model_config(raw: Any) -> Dict[str, Any]:
    """Tolerant ``model_config`` decode: JSON text or dict -> dict copy; anything else -> {}."""
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}

# Billing buckets that aren't a routable provider identity. A session that
# persisted only one of these (never ran /model) falls back to the config
# default rather than restoring a bare bucket. Shared by session_gateway_runtime
# and tui_gateway.server so the two consumers cannot drift.
_BARE_BILLING_PROVIDERS = frozenset({"auto", "custom"})


def _cwd_prefix_clause(cwd_prefix: str) -> Tuple[str, List[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    # ``_``/``%`` are LIKE wildcards but ordinary path characters (``my_project``):
    # unescaped, a prefix also matches sibling directories. The ``=`` arm is an
    # exact compare and keeps the raw prefix; the Windows separator backslash
    # in the LIKE pattern needs escaping too.
    esc = _escape_like(prefix)
    return (
        "(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\' OR s.cwd LIKE ? ESCAPE '\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )


def _workspace_key_clause(key: str) -> Tuple[str, List[str]]:
    """WHERE for ``workspace_key(row) == key``: git_repo_root equals ``key``, or
    (rows predating per-session git metadata) cwd is at/under ``key``. Used by
    ``hermes -c``/``--resume`` to pick the current workspace's MRU, not the global one."""
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


# First user message of a session, shaped by _shape_preview() in Python. The
# indentation is part of the list_sessions_rich SQL text.
_PREVIEW_COL_SQL = f"""COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw"""


def _session_filter_where(
    *, exclude_children: bool = False, source: str = None, sources: List[str] = None,
    session_key: str = None, exclude_sources: List[str] = None, cwd_prefix: str = None,
    min_message_count: int = 0, archived_only: bool = False, include_archived: bool = False,
) -> Tuple[List[str], List[Any]]:
    """Shared ``sessions s`` WHERE builder for list_sessions_rich / session_count /
    session_count_by_source, so the count always lines up with the listed rows.

    ``exclude_children`` keeps roots and user-visible branch/reset sessions while
    hiding sub-agent runs and compression continuations: all four carry
    parent_session_id, so ``_LISTABLE_CHILD_SQL`` classifies the edge from the
    stable ``_branched_from`` marker (survives the parent being re-ended with a
    different end_reason) OR'd with the legacy parent-ended-'branched' heuristic
    for pre-marker rows; delegate children are excluded by their own marker.
    Clause order is part of the SQL text contract.
    """
    where: List[str] = []
    params: List[Any] = []
    if exclude_children:
        where.append(_LISTABLE_CHILD_SQL)
        where.append(f"{_delegate_from_json('s.model_config')} IS NULL")
    include_sources = [source] if source else list(sources or [])
    if include_sources:
        where.append(f"s.source IN ({','.join('?' for _ in include_sources)})")
        params.extend(include_sources)
    if session_key:
        where.append("s.session_key = ?")
        params.append(session_key)
    if exclude_sources:
        where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
        params.extend(exclude_sources)
    if cwd_prefix:
        clause, clause_params = _cwd_prefix_clause(cwd_prefix)
        where.append(clause)
        params.extend(clause_params)
    if min_message_count > 0:
        where.append("s.message_count >= ?")
        params.append(min_message_count)
    if archived_only:
        where.append("s.archived = 1")
    elif not include_archived:
        where.append("s.archived = 0")
    return where, params


def _collect_delegate_child_ids(conn, parent_ids: List[str]) -> List[str]:
    """Delegate-subagent ids (``_delegate_from`` marker) to cascade-delete with
    *parent_ids*; untagged children keep the orphan-don't-delete contract.
    Walks marker chains recursively so an orchestrator's own delegates go too."""
    df = _delegate_from_json()
    seeds = {sid for sid in parent_ids if sid}
    # Seed visited with the parents: a marker chain can loop back onto a parent
    # (cycle, or a parent that is another parent's delegate child in one batch)
    # and it would be collected as its own descendant and cascade-deleted.
    # Callers delete parents separately; never return them as children.
    found: set[str] = set(seeds)
    frontier = list(seeds)
    while frontier:
        ph = ",".join("?" * len(frontier))
        cursor = conn.execute(
            f"SELECT id FROM sessions WHERE {df} IN ({ph}) "
            f"OR (parent_session_id IN ({ph}) AND {df} IS NOT NULL)",
            frontier + frontier,
        )
        frontier = [row["id"] for row in cursor.fetchall() if row["id"] not in found]
        found.update(frontier)
    return [sid for sid in found if sid not in seeds]


def _delete_delegate_children(conn, parent_ids: List[str]) -> List[str]:
    ids = _collect_delegate_child_ids(conn, parent_ids)
    if ids:
        ph = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", ids)
        # FK safety: orphan any untagged stragglers pointing at a doomed row.
        conn.execute(
            f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", ids,
        )
        conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", ids)
    return ids

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

# Back off from read-only opens for this long after one fails: long enough
# that an unreadable file isn't retried per query, short enough that transient
# fd pressure doesn't strand the read pool.
_READ_OPEN_RETRY_SECONDS = 60.0

# Transient SQLITE_IOERR retry budget for READ-ONLY opens. A WAL writer's
# checkpoint / reset / frame flush can surface "disk I/O error" to a concurrent
# mode=ro reader for a millisecond-wide window (ro cannot do the -shm recovery
# the read needs). NOT attempted on writable opens: a writer owns the
# transition, so an IOERR there is a real storage/fd problem.
_READ_ONLY_IOERR_RETRY_ATTEMPTS = 3
_READ_ONLY_IOERR_RETRY_BACKOFF_S = 0.05

# Ceiling on read-only connections ALIVE at once against one database FILE
# (idle pooled + checked out, summed over every SessionDB on that file). One
# constant for both the pool maxsize and the permit count: a LifoQueue only caps
# how many are *returned*; with open-on-miss, N readers hitting an empty pool
# all open and peak at N, and EMFILE is a peak-instant condition. So a
# connection holds a permit for its whole lifetime (_get_read_conn ->
# _close_read_conn); once permits are gone reads degrade to the locked writer
# connection — slower, but not a process-wide wedge the supervisor can't see.
_READ_POOL_MAX = 8

# Ceiling on read-only connections ALIVE in this PROCESS across every state.db
# (a multiplexed gateway opens one per profile, so a per-file cap still scales
# with profile count). Three profiles' worth; past it readers degrade to the
# writer connection for the same reason as _READ_POOL_MAX.
_READ_POOL_PROCESS_MAX = 24

# Warn past this many SessionDB handles on one file in one process. Diagnostic
# only: writer connections cannot be rationed the way read connections can.
_HANDLES_PER_PATH_WARN = 4

# Descriptors kept in reserve for everything that is NOT this module (httpx
# sockets, terminal pipes, log files): SQLite's share is only part of the fd
# table, and the EMFILE it pushes over surfaces elsewhere (terminal_tool).
_FD_HEADROOM_RESERVE = 64

# The fd count is a directory listing; cache it briefly so a read burst isn't a
# syscall per query. Staleness lets through at most the ceiling's worth of opens.
_FD_USAGE_CACHE_SECONDS = 0.25

_process_read_permits = threading.BoundedSemaphore(_READ_POOL_PROCESS_MAX)

# Read opens refused for low descriptor headroom — the only visible signal the
# guard fires. Guarded by _read_budgets_lock.
_read_open_denied_fd_headroom = 0

_fd_usage_lock = threading.Lock()
_fd_usage_cache: "tuple[float, Optional[int]]" = (0.0, None)


def _proc_fd_targets(pid: int) -> Iterator[str]:
    """readlink() of every entry in /proc/<pid>/fd (unreadable links skipped).
    Raises OSError when the fd directory itself cannot be listed."""
    fd_dir = f"/proc/{pid}/fd"
    for fd in os.listdir(fd_dir):
        try:
            yield os.readlink(f"{fd_dir}/{fd}")
        except OSError:
            continue


def _open_fd_count() -> Optional[int]:
    """Open descriptors in THIS process; None when unmeasurable (Windows: no fd
    dir and no RLIMIT_NOFILE, correctly inert — its limit is thousands); -1 when
    the probe itself hit EMFILE/ENFILE (that IS the answer: no headroom)."""
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError as exc:
            if exc.errno in (errno.EMFILE, errno.ENFILE):
                return -1
    return None


def _fd_soft_limit() -> Optional[int]:
    """The process's soft RLIMIT_NOFILE, or None when there is no usable one."""
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    if soft in (resource.RLIM_INFINITY, -1):
        return None
    return int(soft)


def _fd_headroom_ok() -> bool:
    """Can the process spare a descriptor for a new read connection?
    Fails OPEN when unmeasurable (refusing every read there would be a
    self-inflicted convoy); fails CLOSED only on evidence (measured shortfall,
    or a probe that couldn't get a descriptor itself)."""
    soft = _fd_soft_limit()
    if soft is None:
        return True
    global _fd_usage_cache
    now = time.monotonic()
    with _fd_usage_lock:
        stamp, cached = _fd_usage_cache
        fresh = cached is not None and (now - stamp) < _FD_USAGE_CACHE_SECONDS
    if not fresh:
        cached = _open_fd_count()
        with _fd_usage_lock:
            _fd_usage_cache = (now, cached)
    if cached is None:
        return True
    return cached >= 0 and (soft - cached) > _FD_HEADROOM_RESERVE


def _reclaim_idle_read_conn_anywhere() -> bool:
    """Close one idle read connection on ANY path: the process ceiling is shared
    across files, so a quiet profile must not hold descriptors a busy one needs."""
    with _read_budgets_lock:
        budgets = list(_read_budgets.values())
    return any(budget.reclaim_idle() for budget in budgets)


class _PathReadBudget:
    """Read-connection permits for ONE database file, shared process-wide.

    Per-instance semaphores bounded the wrong noun: descriptors are spent on a
    *file*, so N SessionDB objects on one state.db peaked at N x (1 + MAX) —
    a gateway holds at least two per profile path — and walked into EMFILE.
    A pooled idle connection keeps its permit, so the first instance to warm up
    would pin all eight and demote every later instance to the writer lock;
    hence a permit miss first reclaims an IDLE connection from a peer on the
    same path (idle descriptors are transferable, in-use ones are not).
    """

    def __init__(self) -> None:
        self.permits = threading.BoundedSemaphore(_READ_POOL_MAX)
        self._lock = threading.Lock()
        # Weak: a SessionDB dropped without close() must not pin peers' budget.
        self._members: "weakref.WeakSet[SessionDB]" = weakref.WeakSet()
        self._duplicate_handles_warned = False

    def register(self, db: "SessionDB") -> None:
        with self._lock:
            self._members.add(db)
            handles = len(self._members)
            warn = (handles > _HANDLES_PER_PATH_WARN and not self._duplicate_handles_warned)
            if warn:
                self._duplicate_handles_warned = True
        if warn:
            # Writer connections cannot be capped (a SessionDB without one cannot
            # write); the only bound is not opening redundant handles. Make the
            # next duplicate visible before it becomes an incident.
            logger.warning(
                "%d live SessionDB handles on %s in this process; each holds "
                "its own writer connection (read connections are capped at %d "
                "for the file). A long-lived process should share one handle per path.",
                handles,
                db.db_path,
                _READ_POOL_MAX,
            )

    def acquire(self, requester: "SessionDB") -> bool:
        """Take a permit for a new read connection, or refuse (caller then reads
        via the locked writer connection — slower, never an error). Gates,
        broadest first: fd headroom, process-wide ceiling, this file's ceiling."""
        if not _fd_headroom_ok():
            global _read_open_denied_fd_headroom
            with _read_budgets_lock:
                _read_open_denied_fd_headroom += 1
            return False
        if not self._acquire_process_permit():
            return False
        if self._acquire_path_permit(requester):
            return True
        _process_read_permits.release()
        return False

    def release(self) -> None:
        """Return one connection's permits. Pairs with a successful acquire()."""
        self.permits.release()
        _process_read_permits.release()

    def _acquire_process_permit(self) -> bool:
        # Another thread may take a freed permit first; that is a legitimate
        # loss, and the caller degrades to the writer lock rather than looping.
        return _process_read_permits.acquire(blocking=False) or (
            _reclaim_idle_read_conn_anywhere() and _process_read_permits.acquire(blocking=False)
        )

    def _acquire_path_permit(self, requester: "SessionDB") -> bool:
        return self.permits.acquire(blocking=False) or (
            self.reclaim_idle(exclude=requester) and self.permits.acquire(blocking=False)
        )

    def reclaim_idle(self, exclude: "Optional[SessionDB]" = None) -> bool:
        """Close one idle pooled connection held by a member; True if one went.
        Its release() returns both permits, so both ceilings reclaim through here."""
        with self._lock:
            members = [db for db in self._members if db is not exclude]
        return any(member._evict_one_idle_read_conn() for member in members)


# canonical db path -> permits for that file. Weak values: the budget lives as
# long as some SessionDB on the path holds it, so tmp_path churn can't grow this.
_read_budgets: "weakref.WeakValueDictionary[str, _PathReadBudget]" = (weakref.WeakValueDictionary())
_read_budgets_lock = threading.Lock()


def _read_budget_key(db_path) -> str:
    """Canonicalise a db path so two spellings share one budget."""
    try:
        return str(Path(db_path).resolve())
    except OSError:
        return str(db_path)


def _read_budget_for(db_path) -> _PathReadBudget:
    key = _read_budget_key(db_path)
    with _read_budgets_lock:
        budget = _read_budgets.get(key)
        if budget is None:
            budget = _PathReadBudget()
            _read_budgets[key] = budget
        return budget


# Import-time snapshot so _default_db_path() can detect a deliberately
# re-pointed DEFAULT_DB_PATH (tests monkeypatch the constant directly).
_IMPORT_DEFAULT_DB_PATH = DEFAULT_DB_PATH


def _default_db_path() -> Path:
    """Default state DB path at CALL time. A re-pointed ``DEFAULT_DB_PATH`` (the
    test escape hatch) wins; otherwise ``get_hermes_home()`` is resolved fresh so
    a runtime HERMES_HOME redirect works regardless of import order (the frozen
    import-time value pointed every default SessionDB() at the real state.db)."""
    if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH
    return get_hermes_home() / "state.db"


# ---------------------------------------------------------------------------
# Live-DB test-isolation guard
# ---------------------------------------------------------------------------
# Forensic evidence on a live developer machine: pytest fixture rows landed in
# the production ~/.hermes/state.db and a pytest-spawned process flipped the
# journal mode under the WAL-mode gateway writer, destroying committed
# transcripts. The hermetic conftest redirects HERMES_HOME per test, but any
# escape (session-scoped fixture before the autouse one, a subprocess child
# without HERMES_HOME, a stale worktree, a shell exporting HERMES_HOME to the
# real home) silently fell through to the real database.
#
# This is the single choke point: EVERY SessionDB construction resolves its
# path here, so under pytest a production state.db fails hard. Env-based
# (PYTEST_CURRENT_TEST / PYTEST_VERSION inherit into children), so it also
# protects children that never import the conftest.

#: Escape hatch for tests that genuinely need the real DB (conftest sets it for
#: ``@pytest.mark.live_system_guard_bypass``); scripts may set it explicitly.
_STATE_DB_GUARD_BYPASS = False

#: Env twin of ``_STATE_DB_GUARD_BYPASS`` for child processes (a module global
#: cannot cross a process boundary, and ancestry arms the guard there).
_STATE_DB_GUARD_BYPASS_ENV = "HERMES_STATE_DB_GUARD_BYPASS"

#: Extra production roots to refuse; conftest injects the pre-sandbox root so
#: custom-HERMES_HOME deployments are covered too.
_STATE_DB_GUARD_EXTRA_DENY_ROOTS: Tuple[Path, ...] = ()


def _real_platform_state_root() -> Optional[Path]:
    """The REAL platform-default Hermes root. Avoids ``Path.home()`` /
    ``hermes_constants``: tests monkeypatch Path.home to a tempdir while this
    module is imported lazily, which would misidentify the hermetic home as
    production or miss the real one. ``expanduser`` reads HOME/passwd, which the
    conftest never rewrites."""
    try:
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", "").strip()
            root = (
                Path(base) / "hermes"
                if base
                else Path(os.path.expanduser("~")) / "AppData" / "Local" / "hermes"
            )
        else:
            root = Path(os.path.expanduser("~")) / ".hermes"
        return root.resolve()
    except Exception:
        return None


#: Exported by the hermetic conftest alongside the HERMES_HOME redirect (value:
#: the isolation root). Unlike PYTEST_* (scrubbed by tests that rebuild a child
#: env) it is OURS and inherits by default, so a child carrying it that resolves
#: a production DB is by definition an isolation escape.
_TEST_ISOLATION_MARKER_ENV = "HERMES_TEST_ISOLATION"


def _running_under_pytest() -> bool:
    """True when this process (or a parent test process) is a pytest run."""
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("PYTEST_VERSION")
        or os.environ.get(_TEST_ISOLATION_MARKER_ENV)
    )


#: pytest launcher names, matched against each argv token's *basename* so
#: ``/tmp/pytest-of-dev/...`` paths cannot false-positive.
_PYTEST_LAUNCHER_NAMES = frozenset({"pytest", "py.test", "pytest.exe", "py.test.exe"})

#: Memoised ancestry answer: the tree above us doesn't change; keep the hot path free.
_PYTEST_ANCESTOR: Optional[bool] = None


def _process_looks_like_pytest(proc: Any) -> bool:
    """True when *proc*'s command line is a pytest invocation (``pytest ...`` or
    ``python -m pytest``). Unreadable cmdline => not pytest: guessing the other
    way would refuse production opens for unrelated reasons."""
    try:
        cmdline = proc.cmdline() or []
    except Exception:
        return False
    for arg in cmdline:
        try:
            # Split on both separators on every host: os.path.basename is
            # POSIX-only under Linux and would leave a Windows-style path
            # intact, making the matcher's answer depend on the platform.
            name = str(arg).strip('"').strip("'").replace("\\", "/").rsplit("/", 1)[-1].lower()
        except Exception:
            continue
        if name in _PYTEST_LAUNCHER_NAMES:
            return True
    return False


def _has_pytest_ancestor() -> bool:
    """True when an ancestor process is a pytest run. A child spawned with a
    rebuilt env loses PYTEST_* and the HERMES_HOME redirect together — aiming at
    production AND disarming the guard in one step; ancestry survives that.
    Fails open without psutil / on walk errors (never block real user runs)."""
    global _PYTEST_ANCESTOR
    if _PYTEST_ANCESTOR is not None:
        return _PYTEST_ANCESTOR
    found = False
    if psutil is not None:
        try:
            found = any(_process_looks_like_pytest(p) for p in psutil.Process().parents())
        except Exception:
            found = False
    _PYTEST_ANCESTOR = found
    return found


def _in_test_context() -> bool:
    """Test run by environment or ancestry. Env first (two dict lookups); the
    memoised ancestry walk runs at most once per real ``hermes`` invocation."""
    return _running_under_pytest() or _has_pytest_ancestor()


def _production_state_roots() -> List[Path]:
    roots: List[Path] = []
    real_root = _real_platform_state_root()
    if real_root is not None:
        roots.append(real_root)
    for extra in _STATE_DB_GUARD_EXTRA_DENY_ROOTS:
        try:
            roots.append(Path(extra).expanduser().resolve())
        except Exception:
            continue
    return roots


def _is_production_state_db(resolved: Path, root: Path) -> bool:
    """*resolved* is ``<root>/state.db`` or ``<root>/profiles/<name>/state.db``.
    Deeper scratch paths (repo worktrees under ~/.hermes/hermes-agent/...) are
    deliberately NOT matched so hermetic tests cannot false-positive."""
    if resolved.parent == root:
        return True
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return False
    return len(parts) == 3 and parts[0] == "profiles"


def _ensure_test_isolation(db_path: Path) -> None:
    """Raise RuntimeError before any connection/mkdir/pragma/byte probe when a
    pytest-context process (env OR ancestry, see :func:`_in_test_context`)
    resolves a production DB. No-op outside pytest and for hermetic paths."""
    if _STATE_DB_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV):
        return
    if not _in_test_context():
        return
    try:
        resolved = Path(db_path).expanduser().resolve()
    except Exception:
        return
    for root in _production_state_roots():
        if _is_production_state_db(resolved, root):
            raise RuntimeError(
                "live-system guard: test attempted to open production "
                f"state.db at {resolved} (under real Hermes root {root}). "
                "Tests must run against a temporary HERMES_HOME — pass an "
                "explicit tmp db_path or let the hermetic conftest redirect "
                "HERMES_HOME. If this test genuinely needs the live database, mark it with "
                "@pytest.mark.live_system_guard_bypass — or, for a spawned "
                f"child process, export {_STATE_DB_GUARD_BYPASS_ENV}=1 in "
                "its environment."
            )


# Last SessionDB() init error, per-process; surfaced by /resume-style slash
# commands so users know WHY. Only SessionDB.__init__ writes it (kanban_db
# failures are reported via their own callers, by design).
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear with None) the most recent state.db init failure.
    __init__ only SETs on failure and never clears on success: a concurrent
    successful open would erase the cause another thread's /resume is about to format."""
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Most recent state.db init failure (None if none/never attempted)."""
    return _last_init_error


# Openings of the background-review harness prompts (agent/background_review.py),
# matched case-sensitively against leading user/system content.
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: Dict[str, Any]) -> bool:
    """Persisted background-review harness prompt (older builds wrote the forked
    curator's turns into real sessions; replaying them hijacks the session)."""
    if not isinstance(msg, dict) or msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(_REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop harness messages and the curator-mode assistant reply that
    immediately followed each; everything else passes through in order."""
    if not messages:
        return messages
    out: List[Dict[str, Any]] = []
    skip_next_assistant = False
    for msg in messages:
        if _is_background_review_harness_message(msg):
            skip_next_assistant = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                continue  # the curator-mode reply to the harness prompt
        out.append(msg)
    return out


# Matches a bare protocol/tool-name marker such as "[memory]" or "[skill_manage]".
_STALE_TOOL_CALL_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")


def _is_stale_tool_call_marker_message(msg: Dict[str, Any]) -> bool:
    """Assistant tool-call turn whose content is a bare ``[marker]`` — an older
    conversation_loop cached a local template's marker and persisted it as the
    "final response"; sessions written before the fix still carry these rows."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return False
    content = msg.get("content")
    return isinstance(content, str) and bool(_STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()))


def _strip_stale_tool_call_markers(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Blank stale ``[marker]`` assistant content: replaying it teaches the model
    to keep emitting the marker. Only ``content`` is blanked; tool_call/result
    pairing stays intact."""
    repaired = 0
    for msg in filter(_is_stale_tool_call_marker_message, messages):
        msg["content"] = ""
        repaired += 1
    if repaired:
        logger.info(
            "Cleared %d stale tool-call marker message(s) while restoring session (#78148)",
            repaired,
        )
    return messages


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    """User-facing "session DB unavailable" message with the captured init cause
    (e.g. "locking protocol" from NFS/SMB, with a WAL-docs hint)."""
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    hint = ""
    if any(marker in cause.lower() for marker in _WAL_INCOMPAT_MARKERS):
        hint = " (state.db may be on NFS/SMB/FUSE/ZFS — see https://www.sqlite.org/wal.html)"
    return f"{prefix}: {cause}{hint}."


# ---------------------------------------------------------------------------
# Malformed-schema recovery
# ---------------------------------------------------------------------------
# Nastier than a malformed FTS inverted index: ``sqlite_master`` itself is
# inconsistent (typically a DUPLICATE object, e.g. two ``CREATE VIRTUAL TABLE
# messages_fts`` rows). SQLite parses the whole schema while preparing the FIRST
# statement, so EVERY statement raises — including ``PRAGMA journal_mode``
# (hence it trips in apply_wal_with_fallback during __init__, before
# _init_schema), ``PRAGMA integrity_check`` and plain ``DROP TABLE``. Only
# ``PRAGMA writable_schema=ON`` + direct sqlite_master surgery still work.
# Symptom: "malformed database schema (messages_fts) - table messages_fts
# already exists" while Desktop shows "no sessions". Canonical sessions /
# messages are intact; recovery rebuilds only the FTS layer.
_MALFORMED_SCHEMA_MARKERS = ("malformed database schema",)
_MALFORMED_DB_MARKERS = (*_MALFORMED_SCHEMA_MARKERS, "database disk image is malformed")

# Auto-repair at most once per DB path per process (no repair loops; serialises
# concurrent web_server / gateway opens on the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()


def is_malformed_db_error(exc: BaseException) -> bool:
    """Malformed-schema OR generic corrupt-image error. Diagnostics / offline
    recovery only — runtime repair must use :func:`is_malformed_schema_error`."""
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in str(exc).lower() for marker in _MALFORMED_DB_MARKERS
    )


# SQLITE_IOERR as a substring (wrapped strings still classify); shared by the
# read-only open retry and the write-path BEGIN retry.
_DISK_IO_ERROR_MARKER = "disk i/o error"

# "Store BUSY, not gone" — HTTP callers map these to 503 instead of 500.
# Corruption deliberately absent: a malformed store must surface, not be
# retried into a timeout.
_TRANSIENT_SQLITE_MARKERS = (
    _DISK_IO_ERROR_MARKER, "database is locked", "database table is locked", "busy",
)


def _is_no_more_rows(exc: sqlite3.Error) -> bool:
    """Transient engine error on contended WAL appends; the identical write succeeds
    standalone, so it retries like locked/busy. Message-scoped because some builds
    raise it as InterfaceError (outside DatabaseError)."""
    return "no more rows available" in str(exc).lower()


def is_transient_sqlite_error(exc: BaseException) -> bool:
    """"Busy right now", not "damaged". One predicate so the read-only open
    retry and the HTTP 503-vs-500 split cannot drift apart."""
    return isinstance(exc, sqlite3.OperationalError) and any(
        marker in str(exc).lower() for marker in _TRANSIENT_SQLITE_MARKERS
    )


def _is_transient_read_only_ioerr(exc: sqlite3.OperationalError, *, attempt: int) -> bool:
    """Retry a read-only open? See _READ_ONLY_IOERR_RETRY_ATTEMPTS: a
    persistent IOERR still exhausts the budget and propagates."""
    return attempt < _READ_ONLY_IOERR_RETRY_ATTEMPTS and _DISK_IO_ERROR_MARKER in str(exc).lower()


def is_malformed_schema_error(exc: BaseException) -> bool:
    """Only SQLite's explicit malformed-schema text. A generic "disk image is
    malformed" (SQLITE_CORRUPT) may be any B-tree/freelist page and does not
    prove canonical rows intact, so runtime repair must fail closed on it."""
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in str(exc).lower() for marker in _MALFORMED_SCHEMA_MARKERS
    )


# "Filesystem cannot accept another write" substrings (OSError, sqlite3, and
# wrapped RPC strings all match the same helper).
_DISK_FULL_MARKERS = (
    "no space left on device",
    "not enough space",
    "database or disk is full",  # SQLITE_FULL
    "disk full",
    "full disk",
    "enospc",
)


def is_disk_full_error(exc: BaseException | str | None) -> bool:
    """Disk-full / ENOSPC: OSError(ENOSPC), SQLITE_FULL, or matching strings."""
    if exc is None:
        return False
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    lowered = (exc if isinstance(exc, str) else str(exc)).lower()
    return any(marker in lowered for marker in _DISK_FULL_MARKERS)


# Every classify_persistence_error bucket; consumers enumerate this tuple so a
# new bucket can never silently desynchronize them.
PERSISTENCE_ERROR_CAUSES = (
    "locked", "compression", "compression_closed", "turn_lease", "corrupt", "replaced", "disk",
    "unknown",
)


# "Database FILE structurally damaged" substrings. NOTE: "database disk image is
# malformed" contains "disk", so this check MUST run before the disk bucket in
# classify_persistence_error or B-tree corruption reads as "free some disk space".
_DB_CORRUPTION_MARKERS = (
    "malformed",              # "database disk image is malformed" (SQLITE_CORRUPT)
    "file is not a database", # SQLITE_NOTADB (also connection-level poisoning)
    "not a database",
    "database corruption",
)


def classify_persistence_error(exc_or_str) -> str:
    """Coarse cause bucket for a session-persistence failure (PERSISTENCE_ERROR_CAUSES).

    Fast-failing the turn is deliberate (the transcript would be lost on
    restart), but the user's guidance must match the cause: "locked" (write-lock
    contention: storage busy, retry) vs "disk" (full / read-only / permissions).
    "compression" = a live lease refused the write; "compression_closed" = the
    target was rotated by compression and the client must adopt the new id;
    "turn_lease" = fail-fast fencing, not a storage fault; "corrupt" = file
    damage (freeing space cannot help — repair path); "replaced" = the path no
    longer names the opened file (writes on the live handle must stop).
    """
    if exc_or_str is None:
        return "unknown"
    # Lease refusals contain neither "locked" nor "busy": match by type, then by
    # phrase for strings that survived RPC wrapping.
    if isinstance(exc_or_str, SessionTurnLeaseLostError):
        return "turn_lease"
    if isinstance(exc_or_str, CompressionSessionClosedError):
        return "compression_closed"
    if isinstance(exc_or_str, CompressionSessionBusyError):
        return "compression"
    if isinstance(exc_or_str, StateDbReplacedError):  # incl. DeletedWalGenerationError
        return "replaced"
    if isinstance(exc_or_str, StateDbCorruptError):
        return "corrupt"
    text = str(exc_or_str).lower()
    if "turn lease" in text:
        return "turn_lease"
    if "closed by compression" in text:
        return "compression_closed"
    if "being compressed" in text or "compression lease" in text:
        return "compression"
    if "was replaced underneath" in text:
        return "replaced"
    if "deleted state.db-wal" in text or "deleted state.db-shm" in text:
        return "replaced"
    # Corruption BEFORE the lock/disk buckets: "disk image is malformed"
    # contains "disk" and some wrapped strings mention "locked" recovery.
    if any(marker in text for marker in _DB_CORRUPTION_MARKERS):
        return "corrupt"
    if "locked" in text or "busy" in text:
        return "locked"
    if is_disk_full_error(exc_or_str) or "disk" in text or "readonly" in text or "read-only" in text:
        return "disk"
    return "unknown"


# Cross-process serialisation for schema surgery: ``_repair_attempt_lock`` only
# covers threads in ONE interpreter, but gateway, Desktop's ``hermes serve``,
# CLI sessions and the TUI slash worker all share state.db and each used to run
# the full writable_schema surgery + VACUUM on top of the winner's. Timeout
# sized for the slowest legitimate holder (VACUUM over a multi-GB DB) — the
# losing caller previously spent the same minutes on its own surgery.
_REPAIR_LOCK_TIMEOUT_SECONDS = 120.0
_IS_WINDOWS = sys.platform == "win32"


# Repair-loop bounding + dead-backup hygiene: ``_claim_repair_attempt`` bounds
# the loop only WITHIN one process. Unhealable b-tree damage failed repair on
# every process start and each pass took a fresh ~900MB forensic backup (105
# attempts / 89GB of identical copies). Two persistent bounds: a sidecar attempt
# ledger (``<db>.repair-attempts.json``, keyed by size + content-sample
# fingerprint, reset by any successful repair/replacement) refusing surgery after
# _MAX_PERSISTENT_REPAIR_ATTEMPTS; and backup dedupe + retention cap
# (_MAX_MALFORMED_BACKUPS) in ``_backup_db_file``.

# ── CJK-bigram FTS index (replaces the trigram index when available) ────
#
# The trigram tokenizer needs >=3 chars per term, so 1-2 char CJK terms (일본,
# 项目, ...) fell through to a LIKE full-table scan — 3-6s CPU per query on
# multi-GB installs. ``cjk_unicode61`` (native/fts5_cjk/, a small loadable FTS5
# tokenizer) wraps unicode61 and re-emits maximal CJK runs as overlapping
# character bigrams (Lucene CJKAnalyzer semantics); FTS5 phrase semantics then
# give exact substring matching down to 2 chars at index speed.
#
# Same v23 storage discipline as the trigram table: external-content over a
# tool-row-excluding view (tool rows stay searchable via messages_fts), triggers
# gated on a DEDICATED marker pair (fts_cjk_rebuild_high_water /
# fts_cjk_rebuild_progress) so a cjk-only backfill never gates the complete
# messages_fts index's triggers.
#
# The table exists ONLY when the loadable tokenizer is available
# (~/.hermes/lib/libfts5_cjk.so, built by native/fts5_cjk/build.sh). A process
# that cannot load it self-heals by dropping the cjk triggers (writes keep
# working; the index goes stale and is rebuilt by the next optimize-storage).
#
# Split DDL: the table/view part is safe to ensure any time; the triggers are
# created ONLY while the index is complete-or-marker-gated. A stale index
# (trigger gap of unknown extent) must keep its triggers DROPPED — an
# external-content 'delete' for a rowid the index never held is the canonical
# FTS5 index-corruption hazard the v23 marker gating exists to prevent.
FTS_CJK_TABLE_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_cjk_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_cjk USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_cjk_src',
    content_rowid='id',
    tokenize='cjk_unicode61'
);
"""

FTS_CJK_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""

def fts5_cjk_so_path() -> Path:
    """Location of the cjk_unicode61 loadable extension."""
    env = os.getenv("HERMES_FTS5_CJK_SO")
    if env:
        return Path(env).expanduser()
    return get_hermes_home() / "lib" / "libfts5_cjk.so"


def _cjk_fts_config_enabled() -> bool:
    """config.yaml ``sessions.cjk_fts`` (default on), via its env bridge."""
    return os.getenv("HERMES_CJK_FTS", "1").strip().lower() not in ("0", "false", "off", "no")


def load_fts5_cjk_extension(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer. False (never raises)
    when the .so is absent, ``sessions.cjk_fts`` is off, or extension loading
    is compiled out — callers then behave as before the cjk index existed."""
    if not _cjk_fts_config_enabled():
        return False
    path = fts5_cjk_so_path()
    if not path.exists():
        return False
    try:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(path))
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        logger.warning("fts5_cjk extension load failed (%s)", path, exc_info=True)
        return False


class CompressionSessionClosedError(RuntimeError):
    """A durable write targeted a parent already closed by compression."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} is closed by compression; "
            "adopt its live continuation before appending messages"
        )


class CompressionSessionBusyError(RuntimeError):
    """A non-owner tried to write while compression owns the session."""


class SessionCompressionInProgressError(CompressionSessionBusyError):
    """A concurrent writer collided with a *live* compression lock — transient
    (the compressor publishes in seconds; ``_execute_write`` waits), unlike the
    parent class's other case (a compressor whose own lease is gone: permanent,
    fail fast). Subclassing keeps every existing handler working."""


class SessionTurnLeaseLostError(RuntimeError):
    """A transcript write presented a turn-lease holder that no longer owns it.
    Fail-fast fencing (no ``_execute_write`` retry): a later writer may already
    be persisting a newer turn, and landing this one would interleave a stale reply."""


class StateDbReplacedError(RuntimeError):
    """The state.db path no longer names the file this SessionDB opened
    (out-of-band cp/mv/restore). In-place FTS repair and fail-open trigger
    dropping cannot fix a generation mismatch; they amplify it."""


class DeletedWalGenerationError(StateDbReplacedError):
    """A live process holds a deleted state.db-wal / -shm generation. Opening or
    writing through this handle would mint a second WAL inode (split-brain ->
    intermittent SQLITE_CORRUPT / IOERR). Stop the writers; never unlink the WAL
    yourself. Subclasses StateDbReplacedError so every consumer that diverts
    transcripts on a replaced store handles this identically."""


# SQLite header application_id (offset 68). Distinct from inode: ``cp`` onto the
# same path keeps st_ino and truncates+rewrites.
_STATE_DB_APPLICATION_ID_OFFSET = 68
_STATE_DB_GENERATION_KEY = "db_file_generation"
_STATE_DB_REPLACED_MSG = (
    "FATAL: state.db was replaced underneath the gateway; refusing further "
    "writes to this file. Divert transcripts to sessions/<id>.jsonl (and the "
    "gateway pending_messages spool) and restore or reopen after operator intervention."
)
_DELETED_WAL_GENERATION_MSG = (
    "FATAL: a live process holds a deleted state.db-wal or state.db-shm "
    "inode while the path names a different (or missing) generation. "
    "Refusing to open or write so a second WAL cannot be minted. "
    "Stop the gateway, dashboard, and cron writers that hold the deleted "
    "sidecar, then reopen. Do not delete the WAL yourself. "
    "database.journal_mode: delete is operator containment, not a new default."
)


class StateDbCorruptError(sqlite3.DatabaseError):
    """A live SessionDB observed structural (non-FTS, non-replaced) corruption
    and is quarantined. Subclasses sqlite3.DatabaseError so every ``except
    sqlite3.Error`` degrade path keeps working; sqlite_errorcode/name are copied.

    Sticky for the handle's life: later writes fail fast, no reopen after
    close(), and close() skips its WAL checkpoint. Field evidence: a handle that
    kept writing ~50 minutes after the first structural error checkpointed 15
    pages under the wrong page numbers on shutdown, turning a readable file into
    "file is not a database". SQLite's own close-time checkpoint is disabled via
    SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE on Python 3.12+ (WAL survives for
    forensics); on 3.11 it is unavoidable but can only carry pre-corruption
    frames. Recovery boundary: process restart on a repaired/restored file.
    """


_STATE_DB_CORRUPT_MSG = (
    "FATAL: state.db reported structural corruption (database disk image is "
    "malformed outside the FTS shadow tables) on a live handle; refusing further "
    "writes, automatic reopen, and the close-time WAL checkpoint on this file. "
    "Stop the gateway, then run `hermes sessions recover --source <state.db> "
    "--inspect-only` or restore a snapshot. Unwritten transcripts are diverted to "
    "sessions/<id>.jsonl (and the gateway pending_messages spool)."
)


def divert_session_transcript_jsonl(session_id: str, messages) -> "Optional[Path]":
    """Append pending messages to HERMES_HOME/sessions/<id>.jsonl (state.db was
    replaced under a live process). Returns the path, or None if nothing to write."""
    sid = str(session_id or "").strip()
    if not sid or not messages:
        return None
    sessions_dir = get_hermes_home() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for msg in messages:
            if isinstance(msg, dict):
                handle.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
            elif msg is not None:
                handle.write(json.dumps({"content": str(msg)}, ensure_ascii=False) + "\n")
    return path


# ── Process-wide shared SessionDB registry ──
# Lives in hermes_state_registry.py; re-exported here for the historical import
# path. Long-lived in-process callers (gateway, tui_gateway, cron, in-process
# tools) share ONE writer connection per resolved path via
# get_shared_session_db(); CLI one-shots, recovery flows and read-only
# cross-profile opens use SessionDB() directly with their own close().
from hermes_state_registry import (  # noqa: F401  (re-export)
    close_shared_session_dbs, get_shared_session_db, release_or_close, release_shared_session_db,
)

# Lifecycle statuses surfaced by session pickers; classified from the final
# message row ONLY (role, tool_calls, finish_reason) so it stays O(1) per session.
SESSION_STATUS_COMPLETE = "complete"
SESSION_STATUS_INTERRUPTED = "interrupted"
SESSION_STATUS_ERROR = "error"
SESSION_STATUS_EMPTY = "empty"

# finish_reason values meaning the turn ended in a provider/agent error.
_ERROR_FINISH_REASONS = frozenset({"error", "agent_error", "content_filter"})


def classify_session_status(
    role: Optional[str], has_tool_calls: bool, finish_reason: Optional[str],
) -> str:
    """Lifecycle from the final message: error finish → ``error``; assistant
    with pending tool_calls (result never landed), or a trailing user/tool row →
    ``interrupted``; normal assistant finish or unknown shape → ``complete``
    (benign default; pickers must not alarm on unknown shapes)."""
    if (finish_reason or "").strip().lower() in _ERROR_FINISH_REASONS:
        return SESSION_STATUS_ERROR
    r = (role or "").strip().lower()
    if r == "assistant":
        return SESSION_STATUS_INTERRUPTED if has_tool_calls else SESSION_STATUS_COMPLETE
    if r in {"user", "tool"}:
        return SESSION_STATUS_INTERRUPTED
    return SESSION_STATUS_COMPLETE


# Parent→child profile_name inheritance fence: keyless rows (CLI / subagent)
# inherit freely; two ``agent:<ns>:...`` keyed rows must agree on the namespace
# so a default child forked from a sibling profile's row isn't mislabelled.
_SAME_KEY_NAMESPACE_SQL = (
    "p.session_key IS NULL OR sessions.session_key IS NULL"
    " OR substr(p.session_key, 1, instr(substr(p.session_key, 7), ':') + 6)"
    "  = substr(sessions.session_key, 1, instr(substr(sessions.session_key, 7), ':') + 6)"
)


class SessionDB(
    SessionSearchMixin, SessionSchemaMixin, SessionPortabilityMixin, SessionTelegramTopicsMixin,
    SessionCompressionMixin, SessionGatewayMixin, SessionMaintenanceMixin, SessionUsageMixin,
    SessionTitlesMixin, SessionMessagesMixin,
):
    """SQLite-backed session storage with FTS5 search. Thread-safe for the gateway
    pattern (many reader threads, one writer via WAL); each method opens its own cursor."""

    # Only these state-owned producers join automatic stale-open reconciliation;
    # messaging/UI sources have their own lifecycle owners; unknown sources fail closed.
    _AUTO_PRUNE_STALE_OPEN_SOURCES: Tuple[str, ...] = (
        "cli", "cron", "kanban", "acp", "api_server", "subagent", "tool",
    )

    # ── Write-contention tuning ──
    # Many hermes processes share one state.db; SQLite's deterministic busy
    # handler convoys under concurrency, so the SQLite timeout stays short (1s)
    # and retries happen at the application level with random jitter.
    #
    # Patience is TIME-based, not attempt-based: a sibling legitimately holds
    # the lock for multi-second stretches (TRUNCATE checkpoint at close on a
    # large WAL, VACUUM after auto-prune, offline recovery, an older
    # still-running process whose FTS maintenance predates bounded merges). An
    # attempt-counted budget silently lost that race and destroyed the turn as
    # session_persistence_failed although the store was merely busy.
    #
    # Two budgets: routine writes give up after _WRITE_PATIENCE_S; transcript
    # writes (append_message / session-row creation — failure aborts the user's
    # turn) ride out anything shorter than _TRANSCRIPT_WRITE_PATIENCE_S. Jitter
    # stays small for the first _WRITE_RETRY_SLOW_AFTER_S, then backs off so a
    # long hold isn't hammered with BEGIN IMMEDIATE attempts.
    _WRITE_PATIENCE_S = 20.0
    _TRANSCRIPT_WRITE_PATIENCE_S = 60.0
    # Observation-only activity heartbeat/label writes sit on the response-
    # critical path: sub-second budget; a skipped write retries next window.
    _ACTIVITY_WRITE_PATIENCE_S = 0.5
    # A live compression lock gets a short budget: compression publishes in a
    # couple of seconds, so a brief wait saves most concurrent turns — but the
    # lease is a correctness boundary, so a writer still locked out afterwards
    # must be refused rather than land a stale turn in a wedged compression.
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    _WRITE_RETRY_SLOW_AFTER_S = 2.0
    _WRITE_RETRY_SLOW_MIN_S = 0.250  # 250ms
    _WRITE_RETRY_SLOW_MAX_S = 1.000  # 1s
    # PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50
    # FTS maintenance every 1000 writes uses bounded ``'merge'`` commands (a
    # positive rank is an output-page budget: milliseconds of write lock each)
    # instead of unbounded ``'optimize'`` (9-18s per index on a 10GB DB — longer
    # than a competing writer's patience). Up to _FTS_MERGE_COMMANDS_PER_PASS
    # per index, stopping early on the no-progress signal; ``usermerge`` is
    # lowered to 2 so levels with >= 2 segments merge (default 4 never converges).
    _FTS_MERGE_EVERY_N_WRITES = 1000
    _FTS_MERGE_MAX_PAGES_PER_INDEX = 500
    _FTS_MERGE_COMMANDS_PER_PASS = 4
    # Imports cap lower than exports: an import holds one BEGIN IMMEDIATE, so
    # bounded batches avoid starving live writers (one dashboard file at a time).
    _IMPORT_MAX_SESSIONS = 500
    _IMPORT_MAX_MESSAGES_PER_SESSION = 10_000
    _IMPORT_MAX_TOTAL_MESSAGES = 50_000
    _IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
    _IMPORT_MAX_TOTAL_BYTES = 25 * 1024 * 1024
    # Accounting workers retire when idle so a bound-method target can't keep an
    # abandoned SessionDB (and its descriptors) alive; a later enqueue restarts one.
    _TOKEN_WRITER_IDLE_SECONDS = 30.0

    @staticmethod
    def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = _system_prompt_hash(system_prompt)
        conn.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    def _delete_unreferenced_system_prompts(conn) -> None:
        conn.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash)"
        )

    @staticmethod
    def _session_row_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if "_system_prompt_resolved" in data:
            resolved = data.pop("_system_prompt_resolved")
            if "system_prompt" in data:
                data["system_prompt"] = resolved
        return data

    @staticmethod
    def _close_connection_quietly(conn: Optional[sqlite3.Connection]) -> None:
        """Close a partially initialized connection without masking its error."""
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close a SessionDB connection", exc_info=True)

    def __init__(self, db_path: Path = None, read_only: bool = False):
        self.db_path = db_path or _default_db_path()
        _ensure_test_isolation(self.db_path)  # before any connection/pragma/mkdir
        self.read_only = read_only
        self._lock = threading.Lock()
        # Read-path split (WAL only): reads borrow a read-only connection from a
        # BOUNDED pool so they never queue behind writer flushes on self._lock
        # (see _read_ctx). The old per-thread scheme pinned one connection (two
        # fds) per SessionDB x anyio worker thread for the process lifetime until
        # a 256 RLIMIT_NOFILE service hit EMFILE while staying alive, so the
        # supervisor's restart-on-exit never fired.
        self._read_pool: "queue.LifoQueue[sqlite3.Connection]" = queue.LifoQueue(maxsize=_READ_POOL_MAX)
        # Permits bound PEAK descriptors (the pool bounds only the idle set) and
        # are shared per DATABASE PATH (see _PathReadBudget). Acquired
        # non-blocking on purpose: a reader without a permit degrades to the
        # writer lock — blocking would turn fd exhaustion into a stall.
        self._read_budget = _read_budget_for(self.db_path)
        self._read_budget.register(self)
        # Bound to the semaphore itself so every release site is unchanged.
        self._read_permits = self._read_budget.permits
        # Reads that fell back to the writer connection — the only visible
        # signal that the ceiling is being reached (diagnostic, not load-bearing).
        self._read_permit_exhausted = 0
        self._read_conns_lock = threading.Lock()
        # Set when close() begins; a reader still in flight then closes its own
        # connection instead of re-populating a pool nobody will drain again.
        self._read_conns_closed = False
        # "read-only opens are failing" backoff stamp — a TIMESTAMP, not a sticky
        # bool: the likeliest trigger is transient EMFILE, and a permanent flag
        # would demote every reader (the gateway shares one SessionDB across all
        # agents) to the writer lock forever. Expires after _READ_OPEN_RETRY_SECONDS.
        self._read_open_failed_at = 0.0
        self._wal_active = False
        self._write_count = 0
        # File identity of the opened state.db, compared on every write (and
        # before FTS fail-open / reopen) so an out-of-band replace cannot limp
        # through in-place surgery. Inode catches mv/new-file; application_id
        # catches cp onto the same path (same inode, truncate+rewrite).
        self._db_file_identity: Optional[tuple] = None
        self._db_file_application_id: int = 0
        self._db_file_generation_token: str = ""
        self._db_sidecar_identity: Dict[str, tuple] = {}
        self._db_replaced = self._db_wal_generation_lost = False
        # Sticky quarantine (see StateDbCorruptError); never cleared.
        self._db_corrupt = False
        self._db_corrupt_reason = ""
        self._fts_usermerge_floor_applied = False  # one-shot usermerge-floor write guard
        self._fts_enabled = self._fts_stale = self._trigram_available = False
        # _fts_cjk_loaded: tokenizer extension present on the writer connection;
        # _fts_cjk_available: messages_fts_cjk is queryable AND not marked stale.
        self._fts_cjk_loaded = self._fts_cjk_available = self._fts_unavailable_warned = False
        self._conn = None
        # Async token accounting (queue_token_counts). Distinct from self._lock
        # so enqueue/flush bookkeeping never contends with SQLite writes.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None
        # Opened via get_shared_session_db(): close() releases a refcount instead.
        self._shared_registry_owned = False
        initialization_complete = False
        try:
            if read_only:
                self._open_read_only()
                self._record_db_file_identity()
                initialization_complete = True
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # Read-only file/sidecar preflight: repair-or-refuse BEFORE the first
            # connection, for an actionable message instead of an opaque "attempt
            # to write a readonly database" from deep inside _init_schema.
            if not read_only:
                preflight_db_writability(self.db_path, db_label="state.db")
            # Serialize zero-byte check, quarantine, connect and schema commit so
            # concurrent openers don't race the absent-path -> schema-commit window.
            needs_startup_guard = not read_only and (
                not self.db_path.exists() or is_zeroed_state_db(self.db_path)
            )
            try:
                self._open_with_optional_startup_guard(needs_startup_guard)
            except sqlite3.DatabaseError as exc:
                # Malformed schema fails on the very first statement (before
                # _init_schema), so it can't be caught at the FTS-rebuild layer:
                # repair sqlite_master in place (backup first) and reopen once.
                if not is_malformed_schema_error(exc) or not _claim_repair_attempt(self.db_path):
                    raise
                logger.error(
                    "state.db schema is malformed (%s) — attempting automatic "
                    "repair (a backup copy is made first).", exc,
                )
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                report = repair_state_db_schema(self.db_path)
                if not report.get("repaired"):
                    raise
                self._connect_and_init_with_lock_patience()
            # The v23 FTS optimization is OPT-IN (`hermes db optimize`), never
            # auto-started on open: no background worker racing session
            # lifecycle, no surprise disk/latency cost on an unattended open.
            self._ensure_db_file_generation()
            self._record_db_file_identity()
            initialization_complete = True
        except Exception as exc:
            # Surface WHY via /resume and friends; deliberately never cleared on
            # success (see _set_last_init_error). Callers keep their
            # ``self._session_db = None`` degradation path.
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not initialization_complete:
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)

    def _open_read_only(self) -> None:
        """Read-only attach for cross-profile aggregation (SELECT-only).

        Skips schema init entirely (no DDL, no FTS probe, no column reconcile)
        and takes NO write lock, so polling another profile's live DB on every
        sidebar refresh never contends with that profile's backend. The DB must
        already exist + be initialised (callers guard on ``db_path.exists()``);
        a SELECT against an empty file raises and the caller degrades.

        FTS capability flags are probed with SELECTs only. The connection is
        closed on ANY probe failure (malformed schema raises DatabaseError, not
        the OperationalError the probe handles) so a leaked tracked connection
        cannot block ``_backup_db_file``'s raw copy — the writable heal that
        follows would then repair WITHOUT its forensic backup.
        """
        open_attempt = 0
        while True:
            try:
                self._conn = conn = _connect_tracked_db(
                    f"file:{self.db_path}?mode=ro", tracking_path=self.db_path, uri=True,
                    check_same_thread=False, timeout=1.0, isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                try:
                    apply_database_pragmas(conn, db_label="state.db")
                    cursor = conn.cursor()
                    self._fts_enabled = self._fts_table_probe(cursor, "messages_fts") is True
                    if self._fts_enabled:
                        self._trigram_available = (
                            self._fts_table_probe(cursor, "messages_fts_trigram") is True
                        )
                except BaseException:
                    self._conn = None
                    try:
                        conn.close()
                    except Exception:
                        pass
                    raise
                return
            except sqlite3.OperationalError as ioerr:
                # A WAL checkpoint / reset / frame-flush in flight on the writer
                # side can surface SQLITE_IOERR to a concurrent mode=ro reader
                # (it cannot perform the -shm recovery the read needs). The
                # transition closes in milliseconds; retry a bounded number of
                # times before classifying the store as failed.
                if not _is_transient_read_only_ioerr(ioerr, attempt=open_attempt):
                    raise
                open_attempt += 1
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)

    def _handle_quarantine_if_zeroed(self, already_locked: bool = False) -> None:
        """Quarantine a zero-byte/headerless state.db so a fresh one can open.

        If quarantine failed, do not open the zeroed file (it would fail
        opaquely or risk further damage) — raise with the clear message.
        """
        if not (self.db_path.exists() and is_zeroed_state_db(self.db_path)):
            return
        try:
            zsize = self.db_path.stat().st_size
        except OSError:
            zsize = -1
        qpath = quarantine_zeroed_state_db(self.db_path, already_locked=already_locked)
        msg = (
            f"state.db looks ZEROED ({zsize} bytes, no SQLite header). "
            f"Preserved at {qpath or '(quarantine failed — file left in place)'}. "
            f"Restore from {self.db_path.parent / 'state-snapshots'} via `hermes snapshot list` / "
            f"`hermes snapshot restore <id>` if available. "
            "Opening a fresh empty database so the agent can start."
        )
        logger.error(msg)
        _set_last_init_error(msg)
        if qpath is None and self.db_path.exists() and is_zeroed_state_db(self.db_path):
            raise sqlite3.DatabaseError(msg)

    def _connect_and_init(self) -> None:
        # Refuse before sqlite3.connect (under the startup lock) so we cannot
        # mint a replacement WAL while a live writer still holds a deleted
        # sidecar inode.
        refuse_deleted_wal_generation(self.db_path)
        self._conn = _connect_tracked_db(
            str(self.db_path),
            check_same_thread=False,
            # Short timeout — application-level jittered retry handles
            # contention instead of SQLite's internal busy handler (up to 30s).
            timeout=1.0,
            # None = we manage transactions ourselves (explicit BEGIN IMMEDIATE).
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._wal_active = apply_wal_with_fallback(self._conn, db_label="state.db") == "wal"
        apply_database_pragmas(self._conn, db_label="state.db")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._fts_cjk_loaded = load_fts5_cjk_extension(self._conn)
        self._init_schema()

    def _connect_and_init_with_lock_patience(self) -> None:
        """Open + init, waiting out a sibling's write lock with jittered patience.

        ``_init_schema``'s DDL/reconcile statements run on a 1s-timeout
        connection with no retry, so a sibling process holding the write lock
        (VACUUM, TRUNCATE checkpoint at close, a long FTS pass from an older
        install) used to fail the ENTIRE open — callers then disable
        persistence for the whole run. The store is healthy; wait it out with
        the write path's patience. Non-lock errors (including the malformed
        class) propagate immediately.
        """
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        while True:
            try:
                self._connect_and_init()
                return
            except sqlite3.OperationalError as exc:
                err = str(exc).lower()
                if "locked" not in err and "busy" not in err:
                    raise
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                now = time.monotonic()
                if now >= deadline:
                    raise
                time.sleep(min(
                    random.uniform(self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S),
                    max(deadline - now, 0.001),
                ))

    def _open_with_optional_startup_guard(self, needs_startup_guard: bool) -> None:
        if needs_startup_guard:
            with quarantine_cross_process_lock(self.db_path) as lock_acquired:
                if not lock_acquired:
                    logger.warning(
                        "startup quarantine lock for %s not acquired within 5s; proceeding",
                        self.db_path,
                    )
                self._handle_quarantine_if_zeroed(already_locked=lock_acquired)
                self._connect_and_init_with_lock_patience()
        else:
            self._handle_quarantine_if_zeroed(already_locked=False)
            self._connect_and_init_with_lock_patience()

    # ── Read-path split ──

    def _get_read_conn(self) -> Optional[sqlite3.Connection]:
        """Open a fresh read-only connection, or None when unavailable (callers
        return it to self._read_pool; this opens, it does not track).

        WAL only: WAL readers never block on the writer, so reads skip
        self._lock; under DELETE journal mode (NFS fallback) readers hit
        SQLITE_BUSY storms, so the legacy locked path stays. Autocommit reads
        see everything committed so far (read-your-writes for flush-then-search).
        """
        if not self._wal_active or self.read_only:
            return None
        with self._read_conns_lock:
            if self._read_conns_closed:
                return None
            if (
                self._read_open_failed_at
                and time.monotonic() - self._read_open_failed_at
                < _READ_OPEN_RETRY_SECONDS
            ):
                return None
        # Permit BEFORE the open: openers race for permits, not descriptors.
        if not self._read_budget.acquire(self):
            with self._read_conns_lock:
                self._read_permit_exhausted += 1
            logger.debug(
                "read pool at capacity (%d) for %s; serving this read from the "
                "locked writer connection",
                _READ_POOL_MAX,
                self.db_path,
            )
            return None
        conn = None  # bound before the try so the handlers can close a half-open one
        try:
            conn = _connect_tracked_db(
                f"file:{self.db_path}?mode=ro",
                tracking_path=self.db_path,
                uri=True,
                # Pooled connections are borrowed by whichever thread reads next
                # (sqlite3 otherwise refuses cross-thread use, including close()
                # — how the old per-thread connections leaked their fds).
                # Exclusive ownership is enforced by pool checkout, not sqlite3.
                check_same_thread=False,
                timeout=5.0,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            apply_database_pragmas(conn, db_label="state.db")
            # The tokenizer registers in the connection's in-memory registry,
            # not the file, so mode=ro is fine.
            if self._fts_cjk_loaded:
                load_fts5_cjk_extension(conn)
        except sqlite3.Error:
            # A half-open connection (open ok, extension load failed) is a live
            # tracked descriptor — the leak shape this pool exists to fix.
            self._discard_partial_read_conn(conn)
            with self._read_conns_lock:
                self._read_open_failed_at = time.monotonic()
            logger.debug("read-only connection open failed for %s", self.db_path, exc_info=True)
            self._read_budget.release()
            return None
        except BaseException:
            # A stranded permit permanently shrinks the read path by one slot.
            self._discard_partial_read_conn(conn)
            self._read_budget.release()
            raise
        return conn

    def _evict_one_idle_read_conn(self) -> bool:
        """Close one idle pooled connection (a peer on the same file wants its
        permit). Only the idle set is reachable — never pulls a connection out
        from under a live reader. Returns whether a permit was released."""
        try:
            conn = self._read_pool.get_nowait()
        except queue.Empty:
            return False
        self._close_read_conn(conn)
        return True

    def _discard_partial_read_conn(self, conn) -> None:
        """Close a connection that failed between open and hand-off; unlike
        _close_read_conn this does NOT release a permit (callers release their own)."""
        if conn is None:
            return
        try:
            conn.close()
        except Exception as exc:
            logger.warning("partially-opened read conn close failed for %s: %s", self.db_path, exc)

    def _close_read_conn(self, conn) -> None:
        """Close a pooled read connection and release its permit.

        A failing close leaks a tracked fd, so it is logged, never swallowed.
        The permit is released even then: withholding it would turn one leaked
        fd into a permanently narrower read path. Pairs with _get_read_conn();
        over-releasing the BoundedSemaphore raises ValueError rather than
        silently widening the ceiling.
        """
        try:
            conn.close()
        except Exception as exc:
            logger.warning("read-conn close failed for %s: %s", self.db_path, exc)
        finally:
            self._read_budget.release()

    def _checkout_read_conn(self) -> Optional[sqlite3.Connection]:
        """Borrow a read connection, opening on a miss; None when the read path
        is unavailable. The single acquisition seam: a pool hit costs no permit
        (the connection already holds one), only _get_read_conn() takes one, so
        peak live connections stay bounded however many threads miss at once."""
        if not self._wal_active or self.read_only:
            return None
        try:
            return self._read_pool.get_nowait()
        except queue.Empty:
            return self._get_read_conn()

    @contextmanager
    def _read_ctx(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for read-only statements: a pooled read-only
        connection with NO lock under WAL (the writer lock was a global choke
        point), checked out for the block; otherwise (non-WAL, open failure,
        ceiling reached) the writer connection under self._lock — the deliberate
        degradation: slower than EMFILE, which the supervisor cannot see."""
        conn = self._checkout_read_conn()
        if conn is not None:
            try:
                yield conn
            finally:
                returned = False
                with self._read_conns_lock:
                    if not self._read_conns_closed:
                        try:
                            self._read_pool.put_nowait(conn)
                            returned = True
                        except queue.Full:
                            pass
                if not returned:
                    # close() drained the pool: this connection is surplus.
                    # queue.Full is unreachable while permits == maxsize, but the
                    # branch is load-bearing if they ever drift apart (a leak).
                    self._close_read_conn(conn)
            return
        with self._lock:
            if self._conn is None:  # close() raced a still-unwinding reader
                self._reopen_after_close_locked(context="read")
            yield cast(sqlite3.Connection, self._conn)

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        """Reopen the writer connection after ``close()`` raced a live caller.

        A teardown owner (cron run_job's finally, a delegate timeout owner,
        agent close()) sets ``_conn = None`` while an unwinding worker still has
        one transcript flush to land; the next write died on ``NoneType`` and
        the turn's tail was silently dropped. The transcript append is THE
        critical write, so self-heal: loud (WARNING names the race), bounded
        (only after an explicit close(); the constructor never leaves a None
        handle), and ``__del__`` still releases the reopened connection.
        Caller holds ``self._lock``. A failed reopen raises OperationalError
        naming the teardown race instead of the opaque attribute error.
        """
        if self.read_only:
            raise sqlite3.ProgrammingError(
                f"SessionDB for {self.db_path} was closed (read-only handle); "
                f"cannot serve a {context} after close()"
            )
        # A reopen resolves the PATH again: a replaced file would be written
        # through stale WAL/shm assumptions; a quarantined handle must never
        # hand a fresh connection (and its close-time checkpoint) to a damaged file.
        if self._db_replaced or self._db_file_was_replaced():
            self._halt_db_replaced()
        if self._db_corrupt:
            raise self._corrupt_error(
                f"state.db connection for {self.db_path} is quarantined after "
                f"structural corruption; refusing to reopen for a {context} "
                "after close(). "
            )
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()
        logger.warning(
            "state.db connection for %s was closed while a %s was still in "
            "flight — reopening (teardown/worker race, #94736)",
            self.db_path,
            context,
        )
        try:
            conn = _connect_tracked_db(
                str(self.db_path), check_same_thread=False, timeout=1.0, isolation_level=None,
            )
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen failed: {exc}"
            ) from exc
        try:
            conn.row_factory = sqlite3.Row
            self._wal_active = (apply_wal_with_fallback(conn, db_label="state.db") == "wal")
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn)
        except Exception as exc:
            self._close_connection_quietly(conn)
            raise sqlite3.OperationalError(
                f"state.db reopen after close() succeeded but connection setup failed: {exc}"
            ) from exc
        # Schema was initialised by the original open; no _init_schema here (no
        # DDL races with siblings during teardown).
        self._conn = conn

    # ── Core write helper ──

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        # Builds with FTS5 but without the optional trigram tokenizer raise
        # "no such tokenizer: trigram" instead of "no such module"; the loadable
        # cjk_unicode61 tokenizer shows the same capability-error shape. Scoped
        # to those two tokenizers so unrelated tokenizer errors aren't masked.
        err = str(exc).lower()
        return ("no such module" in err and "fts5" in err) or SessionDB._is_trigram_unavailable_error(exc)

    @staticmethod
    def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """Only an optional tokenizer is missing (trigram needs SQLite >= 3.34;
        cjk_unicode61 is loadable): "this one index can't be served", never "disable FTS"."""
        err = str(exc).lower()
        return ("no such tokenizer: trigram" in err or "no such tokenizer: cjk_unicode61" in err)

    @staticmethod
    def _db_has_legacy_inline_fts(cursor: sqlite3.Cursor) -> bool:
        """messages_fts exists in ANY pre-v23 shape. v23 is external-content over
        content/tool_name/tool_calls; every legacy shape (inline single-column
        v11..v22, or the v10-era external single-column) lacks tool_name, so
        "stored CREATE lacks tool_name" catches both. False when absent (fresh DB)."""
        row = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        return row is not None and "tool_name" not in (row[0] or "")

    def _warn_trigram_unavailable(self, exc: sqlite3.OperationalError) -> None:
        """Log once that the trigram tokenizer is missing; base FTS5 stays enabled."""
        if getattr(self, "_trigram_unavailable_warned", False):
            return
        self._trigram_unavailable_warned = True
        logger.info(
            "SQLite trigram tokenizer unavailable for %s "
            "(requires SQLite >= 3.34, this build is %s); "
            "CJK/substring search will fall back to LIKE: %s",
            self.db_path,
            sqlite3.sqlite_version,
            exc,
        )

    def _warn_fts5_unavailable(self, exc: sqlite3.OperationalError) -> None:
        self._fts_enabled = False
        if self._fts_unavailable_warned:
            return
        self._fts_unavailable_warned = True
        logger.warning(
            "SQLite FTS5 unavailable for %s; full-text session search "
            "disabled. Run `hermes update` to rebuild the venv with a "
            "current Python (managed uv guarantees FTS5). (underlying error: %s)",
            self.db_path,
            exc,
        )

    def _ensure_fts_cjk_schema(self, cursor) -> None:
        """Create / repair / self-heal the CJK-bigram index surface (v23 DBs with
        healthy base FTS). ``cursor`` may be a Cursor or Connection. Sets
        ``_fts_cjk_available``; never raises — every failure degrades to
        trigram/LIKE routing.

        tokenizer loaded, table absent → create; a populated DB gets the cjk
        backfill markers so the id-gated triggers stay correct and the index is
        NOT served until optimize-storage backfills. tokenizer loaded, table
        present → ensure triggers, honour the stale breadcrumb. tokenizer NOT
        loaded, live triggers → drop them so INSERTs don't fail at trigger time,
        leave the breadcrumb; the table stays for a capable open to rebuild.
        """
        try:
            cjk_present = bool(cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_cjk'"
            ).fetchone())
            if not self._fts_cjk_loaded:
                if cjk_present:
                    live = [r[0] for r in cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                        f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)})",
                        _FTS_CJK_TRIGGERS,
                    ).fetchall()]
                    if live:
                        # Breadcrumb FIRST (a crash between the two statements
                        # is merely conservative), then drop.
                        logger.warning(
                            "messages_fts_cjk triggers present but the "
                            "cjk_unicode61 tokenizer is unavailable (%s) — "
                            "dropping the cjk triggers so message writes keep "
                            "working. CJK search falls back to trigram/LIKE; "
                            "run `hermes sessions optimize-storage` on a host "
                            "with the extension to rebuild.",
                            fts5_cjk_so_path(),
                        )
                        cursor.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = '1'",
                            (FTS_CJK_STALE_KEY,),
                        )
                        for trig in live:
                            cursor.execute(f"DROP TRIGGER IF EXISTS {trig}")
                self._fts_cjk_available = False
                return
        except sqlite3.OperationalError:
            logger.warning(
                "messages_fts_cjk presence check failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
            self._fts_cjk_available = False
            return
        try:
            cursor.executescript(FTS_CJK_TABLE_SQL)
            if not cjk_present:
                # Any old stale breadcrumb refers to a table that no longer exists.
                cursor.execute("DELETE FROM state_meta WHERE key = ?", (FTS_CJK_STALE_KEY,))
                if cursor.execute("SELECT COUNT(*) FROM messages WHERE role <> 'tool'").fetchone()[0] > 0:
                    hw = cursor.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
                    for k, v in (
                        ("fts_cjk_rebuild_high_water", str(hw)), ("fts_cjk_rebuild_progress", "0"),
                    ):
                        cursor.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (k, v),
                        )
            if cursor.execute("SELECT 1 FROM state_meta WHERE key = ?", (FTS_CJK_STALE_KEY,)).fetchone():
                # Gap of unknown extent: do NOT reinstall triggers (an
                # external-content 'delete' for an unindexed rowid corrupts the
                # index); the next optimize-storage rebuilds from scratch.
                self._fts_cjk_available = False
                return
            cursor.executescript(FTS_CJK_TRIGGER_SQL)
            backfill_pending = cursor.execute(
                "SELECT 1 FROM state_meta WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
            ).fetchone()
            self._fts_cjk_available = not backfill_pending
        except sqlite3.OperationalError:  # incl. "no such tokenizer" after a failed registration
            logger.warning(
                "messages_fts_cjk ensure failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
            self._fts_cjk_available = False

    @staticmethod
    def _drop_fts_triggers(cursor: sqlite3.Cursor) -> None:
        for trigger in _FTS_TRIGGERS:
            try:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.OperationalError:
                pass

    def _ensure_fts_schema(self, cursor: sqlite3.Cursor, table_name: str, ddl: str) -> bool:
        status = self._fts_table_probe(cursor, table_name)
        if status is None:
            return False
        try:
            # Run even when the table exists: recreates triggers a no-FTS5 runtime dropped.
            cursor.executescript(ddl)
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            # A missing tokenizer disables only that table; the base FTS5 table is fine.
            if self._is_trigram_unavailable_error(exc):
                self._warn_trigram_unavailable(exc)
            else:
                self._warn_fts5_unavailable(exc)
            return False

    def _execute_write(
        self, fn: Callable[[sqlite3.Connection], T], patience_s: Optional[float] = None,
    ) -> T:
        """Run *fn(conn)* inside BEGIN IMMEDIATE with jittered lock retry; commit
        is handled here (callers must not commit). Returns *fn*'s result.

        BEGIN IMMEDIATE takes the WAL write lock up front so contention surfaces
        immediately; on locked/busy the Python lock is released, a random jitter
        slept, and the WHOLE callback retried (see the class tuning comment for
        the two patience budgets and the jitter schedule). *fn* must therefore
        stay idempotent under retry.
        """
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        # Set on the first compression-busy collision: the short wait is
        # measured from then, not from the start of the write.
        compression_deadline: Optional[float] = None
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself: the
        # callback has not run, so nothing is replayed (exactly-once safe). Once
        # it has started, an IOERR leaves settlement unknown and must propagate —
        # this helper owns non-idempotent transcript/counter mutations.
        ioerr_begin_retried = False
        while True:
            self._raise_if_db_corrupt()
            self._raise_if_db_replaced()
            fn_started = False
            try:
                with self._lock:
                    if self._conn is None:  # close() raced this writer
                        self._reopen_after_close_locked(context="write")
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        fn_started = True
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                # Success — periodic best-effort checkpoint + FTS merge.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                if self._write_count % self._FTS_MERGE_EVERY_N_WRITES == 0:
                    self._try_incremental_merge_fts()
                return result
            except SessionCompressionInProgressError:
                # Transient (see _COMPRESSION_BUSY_WAIT_S): without a wait, a
                # steer landing mid-compression aborts the turn and sends the
                # operator hunting disk space that was never the problem.
                if compression_deadline is None:
                    compression_deadline = min(
                        time.monotonic() + self._COMPRESSION_BUSY_WAIT_S, deadline
                    )
                if self._sleep_before_write_retry(
                    compression_deadline, self._COMPRESSION_BUSY_WAIT_S
                ):
                    continue
                raise
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    if self._sleep_before_write_retry(deadline, patience_s):
                        continue
                    # Say what actually happened, not disk/permission damage.
                    raise sqlite3.OperationalError(
                        f"database is locked (another Hermes process held the "
                        f"state.db write lock for over {patience_s:.0f}s — "
                        "likely a long maintenance operation such as VACUUM, "
                        "a large WAL checkpoint, or an older pre-update "
                        "process; the database itself is healthy)"
                    ) from exc
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                if (
                    _DISK_IO_ERROR_MARKER in err_msg
                    and not fn_started
                    and not ioerr_begin_retried
                    and self._sleep_before_write_retry(deadline, patience_s)
                ):
                    # Retry on the SAME connection. Never close()+reopen to
                    # "heal": close() cancels this process's POSIX locks on the
                    # file for every sibling connection (howtocorrupt §2.2).
                    ioerr_begin_retried = True
                    continue
                raise  # non-lock error, callback already ran, or patience exhausted
            except sqlite3.DatabaseError as exc:
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                # An out-of-band replace surfaces as this same corruption class;
                # in-file repair on a NEW generation amplifies the damage.
                if (
                    "not a database" in str(exc).lower()
                    or is_malformed_db_error(exc)
                    or self._is_fts_write_corruption_error(exc)
                ):
                    self._raise_if_db_replaced()
                # Corrupt FTS shadow tables fail every write via the sync
                # triggers while canonical rows are intact. Never rebuild FTS
                # from this live path (minutes of writer lock on a multi-GB DB):
                # detach the derived indexes atomically and retry the write.
                if self._enter_fts_fail_open(exc):
                    continue
                # What survives both checks is structural damage: quarantine.
                if self._is_structural_corruption_error(exc):
                    self._halt_db_corrupt(exc)
                raise
            except sqlite3.Error as exc:
                # Builds raising 'no more rows' as InterfaceError (sibling of
                # DatabaseError); anything else propagates untouched.
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                raise

    def _write_sql(
        self, sql: str, params: Any = (), *, many: bool = False, patience_s: Optional[float] = None,
    ) -> None:
        """Run one INSERT/UPDATE/DELETE through ``_execute_write``."""
        def _do(conn):
            (conn.executemany if many else conn.execute)(sql, params)
        self._execute_write(_do, patience_s=patience_s)

    def _write_rowcount(
        self, sql: str, params: Any = (), *, patience_s: Optional[float] = None
    ) -> int:
        """Run one UPDATE/DELETE through ``_execute_write``; return rows changed
        (``SELECT changes()`` when the driver reports None / negative)."""
        def _do(conn):
            rowcount = conn.execute(sql, params).rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        return self._execute_write(_do, patience_s=patience_s)

    def _read_one(self, sql: str, params: Any = ()) -> Optional[sqlite3.Row]:
        """``fetchone()`` of one read-only statement via ``_read_ctx``."""
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchone()

    def _read_all(self, sql: str, params: Any = ()) -> List[sqlite3.Row]:
        """``fetchall()`` of one read-only statement via ``_read_ctx``."""
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchall()

    def _ensure_db_file_generation(self) -> None:
        """Mint a once-per-file generation stamp (state_meta + application_id).
        First opener wins (INSERT OR IGNORE); application_id is written only while
        0 so racers converge. PASSIVE checkpoint only — never TRUNCATE."""
        if self.read_only or self._conn is None:
            return
        token = uuid.uuid4().hex
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                    (_STATE_DB_GENERATION_KEY, token),
                )
                row = self._conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?", (_STATE_DB_GENERATION_KEY,),
                ).fetchone()
                if row and row[0]:
                    token = str(row[0])
                self._db_file_generation_token = token
                pragma_row = self._conn.execute("PRAGMA application_id").fetchone()
                current = int(pragma_row[0] or 0) if pragma_row else 0
                if current == 0:
                    current = (int(token[:8], 16) & 0x7FFFFFFF) or 1
                    self._conn.execute(f"PRAGMA application_id={current}")
                self._db_file_application_id = current
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
        except sqlite3.Error as exc:
            logger.debug("state.db generation stamp skipped: %s", exc)

    def _record_db_file_identity(self) -> None:
        """Snapshot inode plus the on-disk generation header when present."""
        self._db_file_identity = _stat_db_file_identity(self.db_path)
        self._db_sidecar_identity = _stat_sqlite_sidecar_identity(self.db_path)
        disk_id = _read_sqlite_application_id(self.db_path)
        if disk_id:
            self._db_file_application_id = disk_id
        elif self._conn is not None and not self._db_file_application_id:
            try:
                pragma_row = self._read_one("PRAGMA application_id")
            except sqlite3.Error:
                pragma_row = None
            if pragma_row and pragma_row[0]:
                self._db_file_application_id = int(pragma_row[0])

    def _db_file_was_replaced(self) -> bool:
        """True when the path no longer names the file this instance opened."""
        recorded = self._db_file_identity
        if recorded is not None:
            current = _stat_db_file_identity(self.db_path)
            if current is None or current != recorded:
                return True
        recorded_app = int(self._db_file_application_id or 0)
        if recorded_app:
            disk_app = _read_sqlite_application_id(self.db_path)
            # Header 0 = WAL not yet checkpointed, not a replace.
            if disk_app and disk_app != recorded_app:
                return True
        return False

    def _halt_db_replaced(self) -> None:
        """Stop writes and raise; do not run in-file repair on a new generation."""
        self._db_replaced = True
        logger.error(_STATE_DB_REPLACED_MSG)
        raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this handle opened is gone.

        Recorded generation: pure stat (missing/replaced inode = split); no
        /proc walk on healthy writes. Empty identity (fresh DB whose WAL appears
        after open, or cleared by a clean close()): probe /proc/self/fd for
        deleted sidecars and adopt the current ones once clean. The full
        /proc/*/fd walk is reserved for refuse_deleted_wal_generation on open.
        """
        recorded = self._db_sidecar_identity or {}
        base = os.fspath(self.db_path)
        if recorded:
            return any(
                _stat_db_file_identity(Path(base + suffix)) != ident for suffix, ident in recorded.items()
            )
        if not self._wal_active:  # no sidecar generation to lose; keep /proc off the hot path
            return False
        if sys.platform.startswith("linux"):
            watched = _watched_sqlite_sidecar_paths(self.db_path)
            try:
                for target in _proc_fd_targets(os.getpid()):
                    if " (deleted)" in target and _canonical_sqlite_path(target) in watched:
                        return True
            except OSError:
                return False
        # Probe clean (or unavailable): adopt the current sidecar generation.
        current_identity = _stat_sqlite_sidecar_identity(self.db_path)
        if current_identity:
            self._db_sidecar_identity = current_identity
        return False

    def _halt_deleted_wal_generation(self) -> None:
        """Stop writes; do not mint or keep committing on a split WAL."""
        self._db_wal_generation_lost = True
        logger.error(_DELETED_WAL_GENERATION_MSG)
        raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)

    def _halt_if_db_generation_changed(self) -> None:
        """Halt (logging) when the file or its WAL generation is no longer ours."""
        if self._db_replaced or self._db_file_was_replaced():
            self._halt_db_replaced()
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()

    def _raise_if_db_replaced(self) -> None:
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost:
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)
        self._halt_if_db_generation_changed()

    @classmethod
    def _is_structural_corruption_error(cls, exc: BaseException) -> bool:
        """Bare SQLITE_CORRUPT/NOTADB with no FTS provenance: canonical B-tree /
        schema / freelist damage, never repairable from the live write path."""
        return (
            isinstance(exc, sqlite3.DatabaseError)
            and not isinstance(exc, StateDbCorruptError)
            and not cls._is_fts_write_corruption_error(exc)
            and classify_persistence_error(exc) == "corrupt"
        )

    def _corrupt_error(self, prefix: str = "") -> "StateDbCorruptError":
        """Build the quarantine error for this handle (message assembled once)."""
        return StateDbCorruptError(
            f"{prefix}{_STATE_DB_CORRUPT_MSG} (cause: {self._db_corrupt_reason})"
        )

    def _halt_db_corrupt(self, exc: BaseException) -> None:
        """Quarantine this handle and raise; never run in-file repair here."""
        self._db_corrupt = True
        self._db_corrupt_reason = str(exc)
        self._disable_close_time_checkpoint()
        logger.error(
            "state.db %s reported structural corruption outside the FTS "
            "indexes (%s); quarantining this handle: no further writes, no "
            "automatic reopen, no explicit WAL checkpoint at close. Stop the "
            "gateway and run `hermes sessions recover --source %s --inspect-only`.",
            self.db_path,
            exc,
            self.db_path,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            if getattr(exc, attr, None) is not None:
                setattr(err, attr, getattr(exc, attr))
        raise err from exc

    def _disable_close_time_checkpoint(self) -> None:
        """Best-effort SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python 3.12+): skipping
        our explicit checkpoint isn't enough, sqlite3's close() still runs the
        internal last-connection checkpoint that wrote the incident's 15 pages
        under wrong page numbers. See StateDbCorruptError."""
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if flag is None or conn is None or setconfig is None:
            return
        try:
            setconfig(flag, True)
        except Exception:
            logger.debug(
                "Could not disable SQLite's close-time checkpoint on the quarantined handle for %s",
                self.db_path, exc_info=True,
            )

    def _raise_if_db_corrupt(self) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()

    def _sleep_before_write_retry(self, deadline: float, patience_s: float) -> bool:
        """Sleep one jitter interval if the budget allows; True = retry, False =
        deadline passed. Small jitter for the first _WRITE_RETRY_SLOW_AFTER_S,
        then backs off; never overshoots the deadline by a full slow-jitter."""
        now = time.monotonic()
        if now >= deadline:
            return False
        slow = now - (deadline - patience_s) >= self._WRITE_RETRY_SLOW_AFTER_S
        jitter = (
            random.uniform(self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S) if slow
            else random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
        )
        time.sleep(min(jitter, max(deadline - now, 0.001)))
        return True

    @staticmethod
    def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        """Corruption SQLite identifies as FTS-scoped: SQLITE_CORRUPT_VTAB, or
        (older builds) an ``fts5:`` message. A bare malformed-image error is
        structural and must not trigger live FTS maintenance."""
        error_code = getattr(exc, "sqlite_errorcode", None)
        if error_code is not None:
            return error_code == getattr(sqlite3, "SQLITE_CORRUPT_VTAB", 267)
        msg = str(exc).lower()
        return msg.startswith("fts5:") and "corrupt structure" in msg

    def _foreign_state_db_holders(self) -> List[Tuple[int, str]]:
        """Foreign processes holding this DB or its WAL sidecars. Automatic FTS
        repair is structural maintenance and must not run while another process
        is attached (a sidecar reset under it splits the WAL inodes). A scan
        failure is reported as an unknown holder: skipping optional maintenance
        beats assuming quiescence.
        """
        # Split-brain needs POSIX unlink semantics (Windows refuses to replace
        # open sidecars); psutil.open_files() there can block for minutes.
        if _IS_WINDOWS:
            return []
        if psutil is None:
            return [(-1, "open-file scan unavailable")]
        db_path = os.path.abspath(os.fspath(self.db_path))
        watched = {
            _canonical_sqlite_path(db_path), _canonical_sqlite_path(db_path + "-wal"),
            _canonical_sqlite_path(db_path + "-shm"),
        }
        holders: List[Tuple[int, str]] = []
        # Linux: read /proc/<pid>/fd directly. psutil.open_files() stats the
        # literal path, so an unlinked "state.db-wal (deleted)" entry is silently
        # dropped and the split-brain holder never seen; readlink keeps the suffix.
        if sys.platform.startswith("linux"):
            try:
                own_pid = os.getpid()
                for pid_str in os.listdir("/proc"):
                    if not pid_str.isdigit():
                        continue
                    pid = int(pid_str)
                    if pid == own_pid:
                        continue
                    try:
                        targets = list(_proc_fd_targets(pid))
                    except OSError:
                        # Unreadable fd table (other user: root gateway vs user
                        # desktop). cmdline is world-readable: flag only
                        # uninspectable holders that look like Hermes.
                        cmdline = _read_proc_cmdline(pid)
                        if cmdline is not None and _looks_like_hermes(cmdline):
                            holders.append((pid, f"uninspectable holder: {cmdline[:80]}"))
                        continue
                    holders.extend((pid, t) for t in targets if _canonical_sqlite_path(t) in watched)
            except Exception as exc:
                return self._foreign_holder_scan_failed(holders, exc)
            return holders
        # macOS / BSD: psutil.open_files(). macOS does not use the "(deleted)"
        # suffix convention, so psutil's filtering is safe here. psutil's
        # as_dict() converts AccessDenied to None -> empty iteration; acceptable
        # on macOS (the root-gateway/user-desktop topology is Linux-specific).
        try:
            for process in psutil.process_iter(["pid", "open_files"]):
                pid = int(process.info["pid"])
                if pid == os.getpid():
                    continue
                for opened in process.info.get("open_files") or ():
                    path = getattr(opened, "path", "")
                    if path and _canonical_sqlite_path(path) in watched:
                        holders.append((pid, path))
        except Exception as exc:
            return self._foreign_holder_scan_failed(holders, exc)
        return holders

    @staticmethod
    def _foreign_holder_scan_failed(holders: List[Tuple[int, str]], exc: Exception) -> List[Tuple[int, str]]:
        logger.warning(
            "Could not prove state.db has no foreign holders; "
            "deferring automatic FTS maintenance: %s",
            exc,
        )
        return holders or [(-1, f"open-file scan failed: {exc}")]

    def _enter_fts_fail_open(self, exc: sqlite3.DatabaseError) -> bool:
        """Detach corrupt FTS indexes so canonical writes can continue. Stale
        breadcrumb + trigger drop commit atomically: once triggers are absent
        the index has a gap of unknown extent, so no process may reinstall them
        without rebuilding every row."""
        if not self._fts_enabled or not self._is_fts_write_corruption_error(exc):
            return False
        self._raise_if_db_corrupt()
        self._halt_if_db_generation_changed()
        try:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (FTS_STALE_KEY,),
                    )
                    cjk_triggers_present = self._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                        f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)}) "
                        "LIMIT 1",
                        _FTS_CJK_TRIGGERS,
                    ).fetchone()
                    if cjk_triggers_present:
                        self._conn.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (FTS_CJK_STALE_KEY,),
                        )
                    self._drop_all_fts_triggers(self._conn.cursor())
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
        except sqlite3.Error as detach_exc:
            logger.error(
                "Could not detach corrupt FTS indexes; canonical write still cannot proceed: %s",
                detach_exc,
            )
            return False
        self._fts_stale = True
        self._fts_enabled = False
        self._trigram_available = False
        self._fts_cjk_available = False
        logger.error(
            "state.db FTS indexes remain corrupt (%s); disabled FTS sync and "
            "retrying the canonical write. Search temporarily uses LIKE until "
            "a later SessionDB open rebuilds the indexes.",
            exc,
        )
        return True

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint; never raises. PASSIVE never blocks
        writers and leaves the WAL at its high-water mark (bounded by
        journal_size_limit); the old TRUNCATE strategy corrupted B-trees on
        65K+ page databases under exclusive-lock I/O pressure."""
        if self._db_corrupt:
            return  # quarantined: never checkpoint over a damaged image
        try:
            with self._lock:
                result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if result and result[1] > 0:
                    logger.debug("WAL checkpoint: %d/%d pages checkpointed", result[2], result[1])
        except Exception as exc:
            logger.warning("WAL checkpoint (PASSIVE) failed: %s", exc)

    def __enter__(self) -> "SessionDB":
        """``with SessionDB(path) as db:`` closes the handle on exit. Owners must
        release deterministically: a started token writer used to pin the
        instance (bound-method target + strong atexit hook) so __del__ never ran
        for exactly the handles that leaked descriptors; the writer now retires
        when idle and the hook is weak, but "eventually after a GC cycle" is not
        a release policy. close() stays idempotent."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the handle; never suppress the caller's exception."""
        self.close()
        return False

    def close(self):
        """Close the connection: drain queued token deltas (the writer needs the
        connection), then a PASSIVE checkpoint on writable handles (NOT
        TRUNCATE: per-cron-run connections close many times an hour and a full
        WAL reset races the gateway's live writer, tearing B-tree pages).
        A registry-shared instance RELEASES one refcount instead, so one
        caller's close cannot tear down a connection others still use."""
        if getattr(self, "_shared_registry_owned", False):
            from hermes_state_registry import release
            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Closed flag first (under the lock): an in-flight reader then closes its
        # own connection instead of re-populating the drained pool.
        with self._read_conns_lock:
            self._read_conns_closed = True
        while self._evict_one_idle_read_conn():
            pass
        with self._lock:
            if self._conn:
                if self._db_corrupt:  # quarantined: no checkpoint over a damaged image
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed structural corruption (%s). Take a "
                        "snapshot of state.db, -wal and -shm before restarting, "
                        "then run `hermes sessions recover --source %s --inspect-only`.",
                        self.db_path,
                        self._db_corrupt_reason,
                        self.db_path,
                    )
                elif not self.read_only:  # PASSIVE, not TRUNCATE (see docstring)
                    try:
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception as exc:
                        logger.debug("WAL checkpoint (PASSIVE) at close failed: %s", exc)
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)
                # A clean last close lets SQLite unlink the sidecars — a
                # legitimate end of this generation, not a split. Drop it so a
                # teardown-race reopen re-adopts what exists instead of halting.
                self._db_sidecar_identity = {}

    def __del__(self) -> None:
        """Safety net: close() if the caller forgot (read pool, token writer and
        atexit hook too). Attribute access stays guarded: module teardown order
        is undefined."""
        if self.__dict__.get("_conn") is None:
            return
        try:
            self.close()
        except Exception:
            pass

    # ── Chunked FTS rebuild engine (v23 opt-in optimize) ──
    # `optimize_fts_storage()` drops the legacy inline FTS indexes and backfills
    # the external-content ones. One blocking rebuild held the write lock ~16
    # minutes on a 25 GB DB, so the backfill runs in small chunks, each its own
    # short write transaction: readers/writers are never starved, an interrupted
    # run resumes from fts_rebuild_progress, and concurrent runners just
    # interleave chunks (each chunk claims work by compare-and-swap).
    #
    # THROTTLING: a greedy chunk loop re-acquires BEGIN IMMEDIATE back-to-back
    # and starves other processes' writers (an early 5000-row version owned the
    # lock ~85% of the time). Small chunks (500 rows: a foreground write queues
    # for tens of ms at most) plus an inter-chunk pause of max(MIN_PAUSE, chunk
    # cost x DUTY_FACTOR) cap this process's share of DB bandwidth — works
    # cross-process because it bounds our own duty cycle unconditionally.
    _FTS_REBUILD_CHUNK_ROWS = 500
    _FTS_REBUILD_DUTY_FACTOR = 4.0      # sleep >= 4x chunk cost (≤20% duty)
    _FTS_REBUILD_MIN_PAUSE = 0.2        # seconds — floor between chunks

    # Demoted v22 FTS shadow tables awaiting teardown: DROP of a multi-GB vtable
    # blocks for minutes, so the v23 migration demotes the vtable definitions
    # out of sqlite_master and renames the orphaned shadow tables (now plain
    # tables) to fts_v22_trash_*; the worker empties them in chunks, then drops.
    _FTS_TRASH_PREFIX = "fts_v22_trash_"

    def _has_fts_trash(self, conn) -> bool:
        """True when demoted v22 shadow tables are still awaiting teardown.
        Caller must hold ``self._lock`` (or pass a migration-time cursor)."""
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\' LIMIT 1",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        ).fetchone())

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    _PROFILE_DIR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

    def _own_profile_name(self) -> Optional[str]:
        """The profile owning THIS store, from ``db_path`` alone (``<root>/state.db``
        → default, ``<root>/profiles/<name>/state.db`` → name). Path-based, not
        get_active_profile_name(): a gateway serving a NON-launch profile opens
        that profile's store and rows must carry the store's owner. None outside
        the profile tree — keep NULL rather than a fabricated owner."""
        try:
            from hermes_constants import get_default_hermes_root
            root = get_default_hermes_root().resolve()
            parent = Path(self.db_path).resolve().parent
            if parent == root:
                return "default"
            if parent.parent == root / "profiles" and self._PROFILE_DIR_RE.match(parent.name):
                return parent.name
        except Exception:
            logger.debug("own-profile derivation failed", exc_info=True)
        return None

    @staticmethod
    def _inherit_parent_session_metadata(conn, session_id: str) -> None:
        """NULL-fill a new child row's cwd/git/profile from its parent; compression forks also inherit routing.

        Child creators historically did not propagate cwd/git_repo_root/
        git_branch/profile_name, so lineages dropped out of the project sidebar
        or aggregated as "default" on every fork. profile_name is inherited only
        across the same ``agent:<ns>:`` key namespace (``_SAME_KEY_NAMESPACE_SQL``).

        The second UPDATE is belt-and-suspenders for gateway routing metadata:
        the gateway re-records the peer on the child after rotation, but a hard
        crash between child creation and that write leaves the child without
        origin columns, so ``find_latest_gateway_session_for_peer`` can't
        recover the mapping on restart. Inherit them at creation — but ONLY for
        compression forks (parent ``end_reason='compression'``). Delegate/
        subagent children are spawned while the parent is still live and must
        NOT inherit routing keys, or peer recovery could repoint gateway traffic
        into a subagent's session.
        """
        conn.execute(
            f"""UPDATE sessions
                       SET cwd = COALESCE(sessions.cwd,
                                 (SELECT p.cwd FROM sessions p
                                   WHERE p.id = sessions.parent_session_id)),
                           git_repo_root = COALESCE(sessions.git_repo_root,
                                           (SELECT p.git_repo_root FROM sessions p
                                             WHERE p.id = sessions.parent_session_id)),
                           git_branch = COALESCE(sessions.git_branch,
                                        (SELECT p.git_branch FROM sessions p
                                          WHERE p.id = sessions.parent_session_id)),
                           profile_name = COALESCE(sessions.profile_name,
                                          (SELECT p.profile_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id
                                              AND ({_SAME_KEY_NAMESPACE_SQL})))
                     WHERE id = ? AND parent_session_id IS NOT NULL""",
            (session_id,),
        )
        conn.execute(
            """UPDATE sessions
                       SET user_id = COALESCE(sessions.user_id,
                                     (SELECT p.user_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           session_key = COALESCE(sessions.session_key,
                                         (SELECT p.session_key FROM sessions p
                                           WHERE p.id = sessions.parent_session_id)),
                           chat_id = COALESCE(sessions.chat_id,
                                     (SELECT p.chat_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           chat_type = COALESCE(sessions.chat_type,
                                       (SELECT p.chat_type FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           thread_id = COALESCE(sessions.thread_id,
                                       (SELECT p.thread_id FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           display_name = COALESCE(sessions.display_name,
                                          (SELECT p.display_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id)),
                           origin_json = COALESCE(sessions.origin_json,
                                         (SELECT p.origin_json FROM sessions p
                                           WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL
                       AND EXISTS (
                           SELECT 1 FROM sessions p
                           WHERE p.id = sessions.parent_session_id
                             AND p.end_reason = 'compression'
                       )""",
            (session_id,),
        )

    def _insert_session_row(
        self, session_id: str, source: str, model: str = None, model_config: Dict[str, Any] = None,
        system_prompt: str = None, user_id: str = None, session_key: Optional[str] = None,
        chat_id: str = None, chat_type: str = None, thread_id: str = None,
        parent_session_id: str = None, cwd: str = None, profile_name: Optional[str] = None,
        git_repo_root: str = None, origin_json: str = None, display_name: str = None,
    ) -> None:
        """Insert a session row, enriching NULL metadata on conflict.

        The gateway creates a bare row (source + user_id) before the agent
        exists and the agent's later ``create_session`` carries the real
        model/model_config/system_prompt; the upsert ``COALESCE``-fills columns
        that are still NULL and never overwrites what an earlier writer set (a
        later bare source="unknown" call cannot clobber a real source/model).
        ``chat_id``/``thread_id`` record the messaging origin so gateway
        ``/resume`` can prove an inactive row belongs to the caller (IDOR scoping).
        With ``parent_session_id`` NULL metadata is backfilled from the parent
        (:meth:`_inherit_parent_session_metadata`). With no ``profile_name`` the
        row is stamped with THIS store's own profile (:meth:`_own_profile_name`);
        profile-keyed consumers treat NULL as unowned. Stores outside the
        profile tree keep NULL — never guess.
        """
        if not (profile_name or "").strip():
            profile_name = self._own_profile_name()
        def _do(conn):
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                """INSERT INTO sessions (
                   id, source, user_id, session_key, chat_id, chat_type, thread_id,
                   model, model_config, system_prompt, system_prompt_hash,
                   parent_session_id, cwd, profile_name, git_repo_root,
                   origin_json, display_name, started_at
                )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = CASE
                           WHEN excluded.model_config IS NOT NULL
                                AND json_type(
                                    sessions.model_config, '$._reset_from'
                                ) IS NOT NULL
                                AND json_remove(
                                    sessions.model_config, '$._reset_from'
                                ) = '{}'
                           THEN json_set(
                               excluded.model_config,
                               '$._reset_from',
                               json_extract(
                                   sessions.model_config, '$._reset_from'
                               )
                           )
                           ELSE COALESCE(
                               sessions.model_config, excluded.model_config
                           )
                       END,
                       system_prompt_hash = COALESCE(
                           sessions.system_prompt_hash,
                           excluded.system_prompt_hash
                       ),
                       system_prompt = CASE
                           WHEN sessions.system_prompt_hash IS NULL
                                AND excluded.system_prompt_hash IS NOT NULL
                           THEN NULL
                           ELSE sessions.system_prompt
                       END,
                       session_key = COALESCE(sessions.session_key, excluded.session_key),
                       chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                       chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                       thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                       parent_session_id = COALESCE(sessions.parent_session_id, excluded.parent_session_id),
                       cwd = COALESCE(sessions.cwd, excluded.cwd),
                       profile_name = COALESCE(sessions.profile_name, excluded.profile_name),
                       git_repo_root = COALESCE(sessions.git_repo_root, excluded.git_repo_root),
                       origin_json = COALESCE(sessions.origin_json, excluded.origin_json),
                       display_name = COALESCE(sessions.display_name, excluded.display_name)""",
                (
                    session_id, source, user_id, session_key, chat_id, chat_type, thread_id, model,
                    json.dumps(model_config) if model_config else None, system_prompt_hash,
                    parent_session_id, cwd, profile_name, git_repo_root, origin_json, display_name,
                    time.time(),
                ),
            )
            if system_prompt_hash is not None:
                self._delete_unreferenced_system_prompts(conn)
            if parent_session_id:
                self._inherit_parent_session_metadata(conn, session_id)
        # Transcript-critical: a failed row creation aborts the turn. Ride out long holds.
        self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create a new session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def set_expiry_finalized(self, session_id: str, finalized: bool = True) -> None:
        """Mirror ``SessionEntry.expiry_finalized`` so it survives a lost sessions.json."""
        if not session_id:
            return
        self._write_sql(
            "UPDATE sessions SET expiry_finalized = ? WHERE id = ?",
            (1 if finalized else 0, session_id),
        )

    # ── Gateway routing index (replaces sessions.json) ────

    def find_session_by_origin(
        self, *, platform: str, chat_id: str, thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Most recent live session_id for source + chat_id (+ thread_id). With
        ``user_id``, exact sender matches win; if several distinct users share
        the chat and none matches, None rather than contaminating another
        participant's session."""
        if not platform or chat_id in (None, ""):
            return None
        query = """
            SELECT id, user_id, started_at FROM sessions
            WHERE LOWER(source) = LOWER(?)
              AND session_key IS NOT NULL
              AND chat_id = ?
              AND ended_at IS NULL
        """
        params: list = [platform, str(chat_id)]
        if thread_id is not None:
            query += " AND COALESCE(thread_id, '') = ?"
            params.append(str(thread_id))
        query += " ORDER BY started_at DESC"
        rows = [dict(r) for r in self._read_all(query, params)]
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len(rows) > 1:
            distinct_users = {u for u in (str(r.get("user_id") or "").strip() for r in rows) if u}
            if len(distinct_users) > 1:
                return None
        return str(rows[0]["id"])

    # ── Orphaned gateway-session repair (``hermes sessions repair-routing``) ──
    # A write-path failure between routing publication and row creation leaves
    # the live transcript in a row without identity columns, invisible to
    # recovery (the chat resolves to a days-older keyed row). Widest plausible
    # gap between a keyed predecessor going quiet and its unkeyed successor:
    # the reported incident was ~60s; 15 minutes stays generous without
    # spanning unrelated conversations.
    _ORPHAN_ADOPTION_MAX_GAP_S = 900.0

    # Children with a ``parent_session_id`` that are NOT compression
    # continuations (branches, delegate runs, tool sessions). Markers are bound
    # to the queried parent id: compression continuations inherit the rotated
    # agent's model_config verbatim, so a delegate's continuation carries
    # ``_delegate_from=<the delegate's own parent>`` and presence-matching
    # misclassified real continuations as delegate children.
    _NON_CONTINUATION_CHILD_FILTER_SQL = (
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._branched_from'), '') != ?\n"
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._delegate_from'), '') != ?\n  AND COALESCE({alias}source, '') != 'tool'\n"
    )

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session ended. The first end_reason wins (no-op when already
        ended): a compression split must keep ``'compression'`` even if a stale
        desynced-CLI end_session() targets it later. reopen_session() first to
        deliberately re-end with a new reason."""
        def _do(conn):
            changed = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            ).rowcount
            # Only a boundary this call wrote advances the generation (a no-op must not rotate the peer).
            if changed:
                self._bump_conversation_generation(conn, session_id, end_reason)
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed. First stamp
        markerless legacy reset children that depend on the parent's mutable
        end_reason (WHERE shared with the listing predicate via
        _legacy_reset_child_sql so the two cannot drift)."""
        def _do(conn):
            placeholders = ",".join("?" for _ in _RESET_END_REASONS)
            conn.execute(
                "UPDATE sessions AS child SET model_config = json_set("
                "COALESCE(child.model_config, '{}'), '$._reset_from', child.parent_session_id) "
                "WHERE child.parent_session_id = ? "
                "AND json_extract(COALESCE(child.model_config, '{}'), "
                "                 '$._reset_from') IS NULL "
                f"AND {_legacy_reset_child_sql('child', placeholders)}",
                (session_id, *_RESET_END_REASONS),
            )
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def promote_to_session_reset(self, session_id: str, reason: str = "session_reset") -> bool:
        """Durably mark a session ended by an intentional reset boundary.

        Promotes only live rows or rows carrying a *recoverable* accidental
        end_reason (``agent_close``, ``ws_orphan_reap``); explicit boundaries
        (compression, session_reset, ...) are preserved — first writer wins.
        Plain end_session() is not enough: it no-ops on an already-ended row,
        so an ``agent_close`` row would stay recoverable and stale-route
        recovery would resurrect the reset session with its full history. Keep
        the promotion set in sync with find_latest_gateway_session_for_peer.
        ``reason`` keeps reset paths auditable (idle, daily, suspended, ...).
        True when promoted, False when skipped or missing.
        """
        if not session_id:
            return False
        now = time.time()
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND (ended_at IS NULL "
                f"OR end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))",
                (now, reason, session_id),
            )
            # /new and policy auto-resets promote rather than end_session, so the
            # generation advances here too — same transaction, only when written.
            if cursor.rowcount:
                self._bump_conversation_generation(conn, session_id, reason)
            return cursor.rowcount
        try:
            return bool(self._execute_write(_do))
        except Exception:
            return False

    def update_session_cwd(
        self, session_id: str, cwd: str, git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None, replace_git_meta: bool = False,
    ) -> Optional[int]:
        """Persist the authoritative cwd and claim a Git metadata generation.

        ``git_branch`` is the branch checked out at start/resume (the sidebar
        groups by it; the checkout's *current* branch is transient and would
        misattribute past sessions). ``git_repo_root`` is the authoritative
        project key, resolved here so every surface reads the same membership.
        Each field is written only when non-empty so a probe failure never
        clobbers a captured value — except ``replace_git_meta`` (a deliberate
        workspace MOVE must overwrite the old repo identity even when the new
        cwd resolves to none). Every call bumps ``git_metadata_generation`` in
        the same transaction; async probes publish via
        :meth:`publish_session_git_metadata` with that generation, so an older
        worker cannot overwrite a newer claim (A -> B -> A, or another process).
        """
        if not session_id or not cwd:
            return None
        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()
        def _do(conn):
            current = conn.execute(
                "SELECT cwd FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if current is None:
                return None
            current_cwd = current[0]
            sets = ["cwd = ?", "git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1"]
            params: List[Any] = [cwd]
            if current_cwd != cwd or replace_git_meta:
                sets.extend(("git_branch = ?", "git_repo_root = ?"))
                params.extend((branch or None, repo_root or None))
            elif branch:
                sets.append("git_branch = ?")
                params.append(branch)
            if repo_root and current_cwd == cwd and not replace_git_meta:
                sets.append("git_repo_root = ?")
                params.append(repo_root)
            params.append(session_id)
            conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", params)
            row = conn.execute(
                "SELECT git_metadata_generation FROM sessions WHERE id = ?", (session_id,),
            ).fetchone()
            return None if row is None else int(row[0])
        return self._execute_write(_do)

    def publish_session_git_metadata(
        self, session_id: str, cwd: str, generation: int, git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
    ) -> bool:
        """Publish async Git enrichment only while its cwd claim is current."""
        if (
            not session_id
            or not cwd
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            return False
        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()
        if not branch and not repo_root:
            return False
        sets: List[str] = []
        params: List[Any] = []
        if branch:
            sets.append("git_branch = ?")
            params.append(branch)
        if repo_root:
            sets.append("git_repo_root = ?")
            params.append(repo_root)
        params.extend((session_id, cwd, generation))
        return self._write_rowcount(
            f"UPDATE sessions SET {', '.join(sets)} "
            "WHERE id = ? AND cwd = ? AND git_metadata_generation = ?",
            params,
        ) == 1

    def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        """Backfill git repo roots for cwds without one (pre-column sessions);
        never clobbers a recorded root; empty roots are skipped."""
        pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
        if pairs:
            self._write_sql(
                "UPDATE sessions SET git_repo_root = ? "
                "WHERE cwd = ? AND COALESCE(git_repo_root, '') = ''",
                pairs, many=True,
            )

    # Compression locks (atomic per-session, keyed by session_id, recovered via
    # expires_at) live in SessionCompressionMixin; they stop two AIAgents that
    # share a session_id from both rotating it into two orphan children.

    def touch_session_activity(
        self, session_id: str, ts: Optional[float] = None, *, description: Optional[str] = None,
        provenance: Optional[ActivityProvenance] = None,
    ) -> None:
        """Stamp durable mid-turn activity (observation-only; rate-limited by
        AIAgent._touch_activity) so surfaces see API/tool/compaction activity
        before any message row lands. Never moves ``last_activity_at`` backwards."""
        if not session_id:
            return
        from agent.session_activity import (
            bound_activity_description, normalize_activity_provenance,
        )
        when = float(ts if ts is not None else time.time())
        desc = bound_activity_description(description)
        prov = normalize_activity_provenance(provenance).value
        self._write_sql(
            "UPDATE sessions SET last_activity_at = ?, "
            "last_activity_description = ?, last_activity_provenance = ? "
            "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
            (when, desc, prov, session_id, when),
            patience_s=self._ACTIVITY_WRITE_PATIENCE_S,
        )

    def clear_session_activity_labels(self, session_id: str) -> None:
        """Clear activity labels after a turn (keep ``last_activity_at`` so idle /
        watchdog clocks stay continuous; an idle turn must not keep advertising
        "compressing"). Runs in the turn's finally: a no-op clear skips the
        write transaction, a real one uses the short activity budget."""
        if not session_id:
            return
        from agent.session_activity import ActivityProvenance
        try:
            row = self._read_one(
                "SELECT last_activity_description, last_activity_provenance "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )
        except sqlite3.Error:
            row = None
        if row is not None and not row[0] and (not row[1] or row[1] == ActivityProvenance.UNKNOWN.value):
            return
        self._write_sql(
            "UPDATE sessions SET last_activity_description = ?, "
            "last_activity_provenance = ? WHERE id = ?",
            ("", ActivityProvenance.UNKNOWN.value, session_id),
            patience_s=self._ACTIVITY_WRITE_PATIENCE_S,
        )

    def get_session_activity(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the durable activity snapshot for *session_id*, or None."""
        row = self.get_session(session_id) if session_id else None
        if not row:
            return None
        from agent.session_activity import build_activity_snapshot
        return build_activity_snapshot(
            last_activity_at=row.get("last_activity_at"),
            last_activity_description=row.get("last_activity_description"),
            last_activity_provenance=row.get("last_activity_provenance"),
        )

    def update_session_meta(
        self, session_id: str, model_config_json: str, model: Optional[str] = None,
    ) -> None:
        """Update model_config and (COALESCE) optionally model."""
        self.flush_token_counts()  # barrier against queued token deltas — see update_session_model
        self._write_sql(
            "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
            (model_config_json, model, session_id),
        )

    def update_system_prompt(self, session_id: str, system_prompt: Optional[str]) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                "UPDATE sessions SET system_prompt_hash = ?, system_prompt = NULL WHERE id = ?",
                (system_prompt_hash, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def update_session_tool_names(self, session_id: str, tool_names: Optional[List[str]]) -> None:
        """Persist the resolved ``tools[]`` name order so a rebuilt AIAgent
        (agent-cache eviction) can't fork the cached tool prefix on a flipped
        check_fn verdict. ``None`` clears the pin."""
        payload = json.dumps(list(tool_names)) if tool_names is not None else None
        self._write_sql("UPDATE sessions SET tool_names = ? WHERE id = ?", (payload, session_id))

    def update_session_model(
        self, session_id: str, model: str, provider: Optional[str] = None
    ) -> None:
        """Set the model after a mid-session /model switch (unconditionally,
        unlike update_token_counts' COALESCE), null system_prompt so stale
        Model:/Provider: footers rebuild, and replace any confirmed Browser
        runtime lock while keeping lineage markers. *provider* is merged into
        model_config so resume recombines the model with the provider that
        actually serves it, not the config.yaml primary."""
        # This write bypasses the token queue: a still-queued first delta carries
        # the pre-switch route and, applied after this UPDATE, would trip the
        # first_accounted_route overwrite and resurrect the old model/provider.
        self.flush_token_counts()
        # browser_model_lock is deleted via a None patch value (same semantics
        # as the old json_remove); lineage markers survive the merge.
        patch: Dict[str, Any] = {"browser_model_lock": None}
        if model:
            patch["model"] = model
        if provider:
            patch["provider"] = provider
        self._write_model_config_patch(
            session_id, patch,
            "UPDATE sessions SET model = ?, model_config = ?, "
            "system_prompt = NULL, system_prompt_hash = NULL WHERE id = ?",
            lambda merged: (model, merged, session_id),
            clear_prompts=True,
        )

    def _write_model_config_patch(
        self, session_id: str, patch: Dict[str, Any], sql: str,
        params: Callable[[Optional[str]], tuple], *, clear_prompts: bool = False,
    ) -> None:
        """Merge ``patch`` into model_config then run ``sql`` with ``params(merged)``.

        One write transaction; no-op when the row doesn't exist. ``clear_prompts``
        additionally garbage-collects unreferenced system_prompts (for writers
        that NULL the row's system_prompt_hash).
        """
        def _do(conn):
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(sql, params(merged))
            if clear_prompts:
                self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def _merge_model_config_json(
        self, conn, session_id: str, patch: Dict[str, Any], *, on_missing: str = "skip",
    ):
        """SELECT + tolerant-parse + merge ``patch`` into model_config — the one
        place the merge discipline keeping ``_branched_from``/``_delegate_from``
        alive lives. ``None`` deletes a key. Runs inside the caller's write
        transaction. Returns serialized JSON (``None`` when empty, matching
        create_session's NULL) or ``_MODEL_CONFIG_ROW_MISSING`` when the row
        doesn't exist (``on_missing="raise"`` raises ValueError instead)."""
        row = conn.execute(
            "SELECT model_config FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if row is None:
            if on_missing == "raise":
                raise ValueError(f"Session not found: {session_id}")
            return _MODEL_CONFIG_ROW_MISSING
        config = _parse_model_config(row[0])
        for key, value in patch.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = value
        return json.dumps(config) if config else None

    def patch_session_model_config(self, session_id: str, patch: Dict[str, Any]) -> None:
        """Merge ``patch`` into model_config atomically (``None`` removes a key);
        no-op when the row or patch is empty. The transcript-coupled path is
        archive_and_compact's ``model_config_patch``."""
        if not session_id or not patch:
            return
        self._write_model_config_patch(
            session_id, patch, "UPDATE sessions SET model_config = ? WHERE id = ?",
            lambda merged: (merged, session_id),
        )

    def get_session_model_config_value(self, session_id: str, key: str, default: Any = None) -> Any:
        """Read one key out of a session's model_config JSON (tolerant parse)."""
        session = self.get_session(session_id) or {}
        return _parse_model_config(session.get("model_config")).get(key, default)

    def update_session_runtime_lock(
        self, session_id: str, *, model: Optional[str] = None, provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None, route_source: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """Persist a Browser / API-client runtime lock into model_config (lineage
        markers survive); null system_prompt so cached footers cannot lie."""
        lock = {
            "provider": provider or "", "model": model or "", "model_options": model_options or {},
            "route_source": route_source or "", "confirmed": bool(confirmed),
            "updated_at": time.time(),
        }
        self._write_model_config_patch(
            session_id, {"browser_model_lock": lock},
            """UPDATE sessions SET
                   model_config = ?,
                   model = COALESCE(?, model),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
            lambda merged: (merged, model, session_id),
            clear_prompts=True,
        )

    def set_session_yolo(self, session_id: str, enabled: bool) -> None:
        """Persist the per-session YOLO flag into model_config so ``/yolo`` or
        ``--yolo`` survives ``hermes --resume``. No-op when the row doesn't exist
        yet (creation-time model_config carries the flag for --yolo launches)."""
        if not session_id:
            return
        self._write_model_config_patch(
            session_id, {"yolo_mode": bool(enabled)},
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            lambda merged: (merged, session_id),
        )

    @staticmethod
    def session_yolo_enabled(session_meta: Optional[Dict[str, Any]]) -> bool:
        """Persisted YOLO flag from a session row (JSON string or parsed dict);
        False on any parse failure — resume must never enable the bypass by accident."""
        return bool(_parse_model_config((session_meta or {}).get("model_config")).get("yolo_mode"))

    # ── Async token accounting (SessionUsageMixin) ──
    # update_token_counts() stalls the turn thread for tens-hundreds of ms on a
    # cold multi-GB DB after EVERY API call; queue_token_counts() reduces the
    # critical path to a deque append, a single-writer thread applies deltas in
    # order, coalescing consecutive same-route deltas. Exact readers call
    # flush_token_counts() first. Route fields must be equal for two deltas to
    # merge (model/billing_* feed COALESCE backfill and the per-model
    # attribution key; cost_status/source are last-non-None-wins) so the merged
    # UPDATE is byte-for-byte equivalent to applying the deltas sequentially.
    _TOKEN_DELTA_SUM_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_write_tokens", "reasoning_tokens", "api_call_count",
    )
    _TOKEN_DELTA_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")
    _TOKEN_DELTA_ROUTE_FIELDS = (
        "model", "cost_status", "cost_source", "pricing_version",
        "billing_provider", "billing_base_url", "billing_mode",
    )

    def ensure_session(
        self, session_id: str, source: str = "unknown", model: str = None, **kwargs,
    ) -> str:
        """Ensure a session row exists (upsert). Accepts optional kwargs."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID (drains queued token deltas first so cost readers see exact totals)."""
        self.flush_token_counts()
        row = self._read_one(
            "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
            "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            "WHERE s.id = ?",
            (session_id,),
        )
        return self._session_row_dict(row) if row else None

    def get_dominant_session_model_route(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Main-loop model route that served most API calls. ``sessions`` is a
        legacy aggregate mixing route changes; ``session_model_usage`` keeps the
        coherent per-call tuple, so status/billing reads prefer it."""
        self.flush_token_counts()
        row = self._read_one(
            """SELECT model, billing_provider, billing_base_url, billing_mode,
                      api_call_count
                 FROM session_model_usage
                WHERE session_id = ?
                  AND task = ''
                  AND model <> 'unknown'
                  AND billing_provider <> ''
                ORDER BY api_call_count DESC,
                         (input_tokens + output_tokens + cache_read_tokens +
                          cache_write_tokens + reasoning_tokens) DESC,
                         last_seen DESC
                LIMIT 1""",
            (session_id,),
        )
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Exact id, else the single unambiguous prefix match, else None."""
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]
        escaped = _escape_like(session_id_or_prefix)
        matches = [row["id"] for row in self._read_all(
            "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
            (f"{escaped}%",),
        )]
        return matches[0] if len(matches) == 1 else None

    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority: auto-titling may only
    # replace a strictly lower-authority title, so ``derived`` upgrades to
    # ``llm`` exactly once and nothing generated clobbers a user-typed name.
    TITLE_SOURCE_DERIVED = "derived"
    TITLE_SOURCE_LLM = "llm"
    TITLE_SOURCE_USER = "user"
    _TITLE_SOURCE_RANK = {TITLE_SOURCE_DERIVED: 0, TITLE_SOURCE_LLM: 1, TITLE_SOURCE_USER: 2}

    # Bot Mode's canonical chat is resolved by exact-title lookup (no session-id
    # pointer exists); the title IS the identity, so _set_session_title refuses
    # renames of a hidden row holding it.
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

    def backfill_null_session_profiles(self, profile_name: str) -> int:
        """Stamp this store's own profile onto legacy ``profile_name IS NULL``
        rows, which the fail-closed owner ladder cannot route once a Desktop
        registers a second connection (pre-ownership sessions became
        unresumable). Single-match, not a guess: a store belongs to exactly one
        profile. Never overwrites a non-NULL owner; idempotent. Returns rows stamped."""
        stamp = (profile_name or "").strip()
        if not stamp:
            return 0
        return int(self._write_rowcount(
            """UPDATE sessions
               SET profile_name = ?
             WHERE profile_name IS NULL OR TRIM(profile_name) = ''""",
            (stamp,),
        ) or 0)

    def _set_lineage_column(self, column: str, session_id: str, value: Any) -> bool:
        """Set one ``sessions`` column across a whole compression lineage
        (ancestors + descendants joined by end_reason='compression'): Desktop
        projects roots forward to their tip, and updating only the displayed tip
        would let the untouched root resurrect it on refresh. True if any row changed."""
        return self._write_rowcount(
            f"""
            WITH RECURSIVE
              ancestors(id) AS (
                SELECT ?
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE parent.end_reason = 'compression'
              ),
              descendants(id) AS (
                SELECT ?
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions parent ON parent.id = d.id
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.end_reason = 'compression'
              ),
              lineage(id) AS (
                SELECT id FROM ancestors
                UNION
                SELECT id FROM descendants
              )
            UPDATE sessions
            SET {column} = ?
            WHERE id IN (SELECT id FROM lineage)
            """,
            (session_id, session_id, value),
        ) > 0

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        """Soft-hide (or unhide) a session and its whole compression lineage;
        messages are kept. True when at least one row changed."""
        return self._set_lineage_column('archived', session_id, 1 if archived else 0)

    # Accidental end reasons recovery treats as resumable; the same constant is
    # interpolated into the recovery/promotion SQL so literals cannot drift.
    RECOVERABLE_END_REASONS = _RECOVERABLE_END_REASONS

    def unarchive_recoverable_session(self, session_id: str) -> bool:
        """Un-archive a session archived by a recoverable accident (ws_orphan_reap,
        agent_close) — used by registry lookups like Bot Mode's canonical chat.
        Deliberate archives (no end_reason, or an explicit boundary) are left
        alone. True only when a recoverable row was un-archived (whole lineage)."""
        if not session_id:
            return False
        try:
            row = self.get_session(session_id)
        except Exception:
            return False
        if not row or not row.get("archived"):
            return False
        # The accidental stamp lives on the live TIP (the registry row keeps
        # end_reason='compression'); judge recoverability there.
        tip = row
        try:
            tip_id = self.get_compression_tip(session_id) or session_id
            if tip_id != session_id:
                tip = self.get_session(tip_id) or row
        except Exception:
            tip_id = session_id
        if (tip.get("end_reason") or "") not in self.RECOVERABLE_END_REASONS:
            return False
        if not self.set_session_archived(session_id, False):
            return False
        # Clear the accidental end stamp, or a LATER deliberate archive (which
        # never writes end_reason) would auto-resurrect on the next lookup.
        self._write_sql(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (tip["id"],),
        )
        return True

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        """Pin/unpin a session and its compression lineage. Pinned sessions are
        exempt from the ``sessions.auto_archive`` sweep; Desktop mirrors its
        sidebar pins here so backend sweeps honour them."""
        return self._set_lineage_column('pinned', session_id, 1 if pinned else 0)

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        """Hide/unhide a session and its compression lineage from the default
        list_sessions_rich listing; it stays resumable by the owning surface
        (plugins such as kanban manage their own sessions)."""
        return self._set_lineage_column('hidden', session_id, 1 if hidden else 0)

    def set_session_read(self, session_id: str, read: bool = True) -> bool:
        """Mark read/unread across the compression lineage. ``last_read_at`` is a
        watermark, not a flag: unread when activity postdates it, so new
        messages flip it back without any write on the message path. NULL =
        never tracked = read (shipping the column doesn't badge all history);
        0 = explicitly unread; timestamp = read up to then."""
        return self._set_lineage_column('last_read_at', session_id, time.time() if read else 0.0)

    @staticmethod
    def session_unread(session_row: Dict[str, Any]) -> bool:
        """Unread = activity postdates the ``last_read_at`` watermark (NULL = read)."""
        last_read = session_row.get("last_read_at")
        if last_read is None:
            return False
        last_active = session_row.get("last_active") or session_row.get("started_at")
        return float(last_active or 0) > float(last_read)

    # compact_rows excludes only payload-heavy blobs no list consumer renders;
    # the projection derives from SCHEMA_SQL so new columns join automatically.
    _SESSION_COMPACT_EXCLUDED = frozenset(
        {"system_prompt", "system_prompt_hash", "git_metadata_generation"}
    )
    _session_compact_cols_sql: Optional[str] = None

    @staticmethod
    def _chain_search_where(where_sql: str, id_needle: str, search_needle: str) -> Tuple[str, List[Any]]:
        """Extend ``where_sql`` with the id_query / search_query chain filters.

        A surfaced row is admitted when its own id, or any id in its forward
        compression chain (the ``chain`` CTE), matches ``id_needle``; the
        search variant also matches titles, plus a punctuation-stripped form so
        ``an94`` finds ``AN-94``. LIKE with a leading wildcard can't use an
        index, but chain membership and the small result set keep it bounded —
        far cheaper than fetching every session and scanning in Python.
        """
        params: List[Any] = []
        clauses: List[str] = []
        def _like_pattern(needle: str) -> str:
            return f"%{_escape_like(needle)}%"
        if id_needle:
            clauses.append(
                "EXISTS (SELECT 1 FROM chain cq        WHERE cq.root_id = s.id"
                "          AND LOWER(cq.cur_id) LIKE ? ESCAPE '\\')"
            )
            params.append(_like_pattern(id_needle))
        if search_needle:
            compact_needle = re.sub(r"[\W_]+", "", search_needle)
            compact_sql = (
                "REPLACE(REPLACE(REPLACE(REPLACE(LOWER(COALESCE({0}, '')),"
                " '-', ''), '_', ''), '.', ''), ' ', '')"
            )
            search_clause = (
                "EXISTS (SELECT 1 FROM chain cq JOIN sessions cs ON cs.id = cq.cur_id"
                " WHERE cq.root_id = s.id AND (LOWER(COALESCE(cs.title, '')) LIKE ? ESCAPE '\\'"
                " OR LOWER(cq.cur_id) LIKE ? ESCAPE '\\'"
            )
            params.extend([_like_pattern(search_needle)] * 2)
            if compact_needle:
                search_clause += f" OR {compact_sql.format('cs.title')} LIKE ? ESCAPE '\\'"
                params.append(_like_pattern(compact_needle))
            clauses.append(search_clause + "))")
        if not clauses:
            return where_sql, params
        combined = " AND ".join(clauses)
        return (f"{where_sql} AND {combined}" if where_sql else f"WHERE {combined}"), params

    def _project_compression_tips(self, sessions: List[Dict[str, Any]], compact_rows: bool) -> List[Dict[str, Any]]:
        """Replace each compression root's surfaced fields with its live tip's.

        The entry then acts as the live conversation while keeping the root's
        ``started_at`` for stable chronological ordering. ``get_compression_chain``
        is a per-root graph walk (not batchable), but the tip ROW fetch is one
        batched query instead of one ``_get_session_rich_row`` per root.
        ``_lineage_ids`` carries every id on the chain, intermediates included: a
        persisted tile/route can hold a MIDDLE segment's id, and with only the
        root/tip pair a surface cannot prove it names this conversation — which
        is how one chat ends up open twice after compaction.
        """
        tip_ids_by_root: Dict[str, str] = {}
        chain_by_root: Dict[str, List[str]] = {}
        for s in sessions:
            if s.get("end_reason") != "compression":
                continue
            chain = self.get_compression_chain(s["id"])
            tip_id = chain[-1] if chain else s["id"]
            if tip_id != s["id"]:
                tip_ids_by_root[s["id"]] = tip_id
                chain_by_root[s["id"]] = chain
        tip_rows = (
            self._get_session_rich_rows_batch(set(tip_ids_by_root.values()), compact_rows=compact_rows)
            if tip_ids_by_root else {}
        )
        projected = []
        for s in sessions:
            tip_id = tip_ids_by_root.get(s["id"])
            tip_row = tip_rows.get(tip_id) if tip_id else None
            if not tip_row:
                projected.append(s)
                continue
            merged = dict(s)
            for key in (
                "id", "ended_at", "end_reason", "message_count",
                "tool_call_count", "title", "last_active", "preview",
                "model", "system_prompt", "cwd", "git_branch", "git_repo_root",
            ):
                if key in tip_row:
                    merged[key] = tip_row[key]
            merged["_lineage_root_id"] = s["id"]
            merged["_lineage_ids"] = chain_by_root.get(s["id"]) or None
            projected.append(merged)
        return projected

    @classmethod
    def _list_row(cls, row: sqlite3.Row) -> Dict[str, Any]:
        """Project a list_sessions_rich row: shape the preview, drop internal ordering columns."""
        s = cls._session_row_dict(row)
        s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
        s.pop("_effective_last_active", None)
        return s

    def list_sessions_rich(
        self, source: str = None, sources: List[str] = None, exclude_sources: List[str] = None,
        cwd_prefix: str = None, limit: int = 20, offset: int = 0, include_children: bool = False,
        min_message_count: int = 0, project_compression_tips: bool = True,
        order_by_last_active: bool = False, include_archived: bool = False,
        archived_only: bool = False, id_query: str = None, search_query: str = None,
        compact_rows: bool = False, include_pinned: bool = False, session_key: str = None,
        include_hidden: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and ``last_active``
        (freshest of heartbeat / latest message / started_at) in one query.

        Subagent runs and compression continuations are excluded unless
        ``include_children``; branch/reset children stay listable.
        ``project_compression_tips`` surfaces each chain as ONE entry with the
        live tip's fields. ``order_by_last_active`` sorts by the chain TIP's
        activity via a recursive CTE so LIMIT/OFFSET stay cheap; ``id_query`` /
        ``search_query`` (that path only) match across the forward chain, the
        latter also a punctuation-stripped variant. ``compact_rows`` omits the
        system_prompt blob so SQLite never copies tens of KB per row.
        ``include_pinned`` back-fills pins the page window missed (a pin means
        "always reachable"), still obeying the other filters.
        """
        self.flush_token_counts()  # rows carry token/cost totals
        where_clauses, params = _session_filter_where(
            exclude_children=not include_children, source=source, sources=sources,
            session_key=session_key, exclude_sources=exclude_sources, cwd_prefix=cwd_prefix,
            min_message_count=min_message_count, archived_only=archived_only,
            include_archived=include_archived,
        )
        if not include_hidden:
            where_clauses.append("s.hidden = 0")
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        base_where_params = list(params)  # pinned back-fill reuses the WHERE before LIMIT/OFFSET
        prompt_select = (
            "" if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            "" if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )
        _sel = self._compact_session_cols() if compact_rows else "s.*"
        id_needle = (id_query or "").strip().lower()
        search_needle = (search_query or "").strip().lower()
        if order_by_last_active:
            # The CTE seeds from rows the outer WHERE admits and walks
            # compression-continuation edges forward; MAX over the chain gives
            # effective_last_active so ORDER BY + LIMIT happen in SQL. Do NOT
            # require child.started_at >= parent.ended_at: races insert the
            # continuation before the parent's ended_at is written, while stale
            # websocket siblings could pass the timestamp test and hijack projection.
            outer_where, id_params = self._chain_search_where(where_sql, id_needle, search_needle)
            query = f"""
                WITH RECURSIVE chain(root_id, cur_id) AS (
                    SELECT s.id, s.id FROM sessions s {where_sql}
                    UNION ALL
                    SELECT c.root_id, child.id
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX({_sql_session_last_active_by_id("cur_id")}) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                )
                SELECT {_sel}{prompt_select},
                    {_PREVIEW_COL_SQL},
                    {_sql_session_last_active("s")} AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {prompt_join}
                {outer_where}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
                LIMIT ? OFFSET ?
            """
            params = params + params + id_params + [limit, offset]  # WHERE binds twice (seed + outer)
        else:
            query = f"""
                SELECT {_sel}{prompt_select},
                    {_PREVIEW_COL_SQL},
                    {_sql_session_last_active("s")} AS last_active
                FROM sessions s
                {prompt_join}
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
        sessions = [self._list_row(row) for row in self._read_all(query, params)]
        # Pinned back-fill runs BEFORE compression projection so a back-filled
        # root projects to its tip like any other row. One query, never N+1.
        if include_pinned:
            seen_ids = {s["id"] for s in sessions}
            pinned_where = (f"{where_sql} AND s.pinned = 1" if where_sql else "WHERE s.pinned = 1")
            pinned_query = f"""
                SELECT {_sel}{prompt_select},
                    {_PREVIEW_COL_SQL},
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                FROM sessions s
                {prompt_join}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
            for row in self._read_all(pinned_query, base_where_params):
                s = self._list_row(row)
                if s["id"] not in seen_ids:
                    seen_ids.add(s["id"])
                    sessions.append(s)
        if project_compression_tips and not include_children:
            sessions = self._project_compression_tips(sessions, compact_rows)
        # last_read_at is lineage-stamped, so root and tip watermarks agree.
        for s in sessions:
            s["unread"] = self.session_unread(s)
        return sessions

    def session_lifecycle_statuses(self, session_ids: List[str]) -> Dict[str, str]:
        """``{session_id: status}`` from each session's LAST message row (see
        :func:`classify_session_status`; ``'empty'`` when no messages). One query:
        MAX(id) per session (index seek) joined back for that row — never scans transcripts."""
        ids = [sid for sid in (session_ids or []) if sid]
        if not ids:
            return {}
        statuses: Dict[str, str] = {sid: "empty" for sid in ids}
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT m.session_id, m.role,
                   m.tool_calls IS NOT NULL AS has_tool_calls,
                   m.finish_reason
            FROM messages m
            JOIN (
                SELECT session_id, MAX(id) AS max_id
                FROM messages
                WHERE session_id IN ({placeholders})
                GROUP BY session_id
            ) latest ON m.id = latest.max_id
        """
        rows = self._read_all(query, ids)
        for row in rows:
            statuses[row["session_id"]] = classify_session_status(
                role=row["role"], has_tool_calls=bool(row["has_tool_calls"]),
                finish_reason=row["finish_reason"],
            )
        return statuses

    # ── Message storage constants (SessionMessagesMixin) ──
    # Prefix distinguishing JSON-encoded structured content (multimodal parts)
    # from plain strings; NUL is not legal in normal text, so it cannot collide.
    _CONTENT_JSON_PREFIX = "\x00json:"
    #: Reactions live inside ``display_metadata`` (not a side table) so they
    #: survive rewind/compaction row rewrites with the row itself.
    REACTIONS_METADATA_KEY = "reactions"
    # Columns every conversation projection decodes (model-fed and display
    # views share one SELECT); ``active`` rides along so a display read can
    # split compaction-archived rows from the live set without a second query.
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, active, api_content, display_kind, display_metadata"
    )

    def assert_export_safe(self, session_id: str, max_messages: Optional[int] = None) -> int:
        """Active row count of this segment (compression ancestors excluded), or
        raise SessionExportTooLargeError. The LIMITed subquery stops once it
        proves the bound is exceeded. ``None`` resolves ``sessions.max_export_messages``;
        0 disables the guard (returns 0 without counting)."""
        if max_messages is None:
            max_messages = resolved_max_export_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        row = self._read_one(
            "SELECT COUNT(*) FROM ("
            "SELECT 1 FROM messages WHERE session_id = ? AND active = 1 LIMIT ?)",
            (session_id, max_messages + 1),
        )
        message_count = int(row[0] if row else 0)
        if message_count > max_messages:
            raise SessionExportTooLargeError(session_id, message_count, max_messages)
        return message_count

    def _is_explicit_branch_session(self, session_id: str) -> bool:
        """Copied user-facing branch (``_branched_from`` marker)? Branches own a
        copied transcript; compression continuations need the parent's archived rows."""
        if not session_id:
            return False
        row = self._read_one("SELECT model_config FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            return False
        return bool(_parse_model_config(row[0]).get("_branched_from"))

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]
        chain = []
        current = session_id
        seen = set()
        with self._read_ctx() as conn:
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?", (current,),
                ).fetchone()
                if row is None:
                    break
                current = row[0]
        return list(reversed(chain)) or [session_id]

    def search_sessions(
        self, source: str = None, limit: int = 20, offset: int = 0, workspace_key: str = None,
    ) -> List[Dict[str, Any]]:
        """Sessions MRU-first with a computed ``last_active``; ``workspace_key``
        scopes to one workspace (:func:`workspace_key` semantics) so
        ``hermes -c``/``--resume`` picks the current workspace's last session."""
        select_with_last_active = (
            "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
            f"{_sql_session_last_active('s')} AS last_active "
            "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        )
        where_clauses = []
        params: list = []
        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if workspace_key:
            ws_clause, ws_params = _workspace_key_clause(workspace_key)
            where_clauses.append(ws_clause)
            params.extend(ws_params)
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.extend([limit, offset])
        return [self._session_row_dict(row) for row in self._read_all(
            f"{select_with_last_active}{where_sql} "
            "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
            params,
        )]

    def session_count(
        self, source: str = None, sources: List[str] = None, cwd_prefix: str = None,
        min_message_count: int = 0, include_archived: bool = False, archived_only: bool = False,
        exclude_children: bool = False, exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions with the same filters as list_sessions_rich, so a
        paired "load more" total matches the listable rows (children or a
        cron-excluded page would otherwise inflate it and never settle)."""
        where_clauses, params = _session_filter_where(
            exclude_children=exclude_children, source=source, sources=sources,
            exclude_sources=exclude_sources, cwd_prefix=cwd_prefix,
            min_message_count=min_message_count,
            archived_only=archived_only, include_archived=include_archived,
        )
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return self._read_one(f"SELECT COUNT(*) FROM sessions s{where_sql}", params)[0]

    def session_count_ge(self, n: int = 1) -> bool:
        """At least N sessions exist (archived included — "has this install ever
        had sessions"). LIMIT short-circuits: 4us vs session_count()'s 543us
        index scan on a 20k-session DB."""
        rows = self._read_all("SELECT 1 FROM sessions LIMIT ?", (n,))
        return len(rows) >= n

    def session_count_by_source(
        self, *, include_archived: bool = False, archived_only: bool = False,
        exclude_children: bool = False,
    ) -> Dict[str, int]:
        """``{source: count}`` via one GROUP BY (uses idx_sessions_source unless
        ``exclude_children``, whose predicates need a table scan like
        list_sessions_rich). ``exclude_children`` mirrors listing visibility."""
        where_clauses, params = _session_filter_where(
            exclude_children=exclude_children,
            archived_only=archived_only, include_archived=include_archived,
        )
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._read_ctx() as conn:
            if self._conn is None:
                raise RuntimeError("SessionDB connection is closed")
            rows = conn.execute(
                "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, COUNT(*) AS count "
                f"FROM sessions s{where_sql} "
                "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') ORDER BY count DESC",
                params,
            ).fetchall()
        return {str(row["source"]): int(row["count"] or 0) for row in rows}

    def declared_scope_identity(self, session_id: str) -> Tuple[bool, str]:
        """(is_fork_child, source) for *session_id* in ONE read — prompt_cache_scope
        needs both from the same row. A missing row is (False, ""); DB errors
        propagate so the caller fails closed."""
        session = self.get_session(session_id)
        if not session:
            return False, ""
        return (self._is_explicit_fork_child_row(session), str(session.get("source") or "").strip())

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove ``<id>.json``/``.jsonl`` and gateway ``request_dump_<id>_*.json``;
        OSError is swallowed so a filesystem hiccup never blocks a DB operation."""
        if sessions_dir is None:
            return
        targets = [sessions_dir / f"{session_id}{suffix}" for suffix in (".json", ".jsonl")]
        try:
            # request_dump files use session_id as a prefix component
            targets.extend(sessions_dir.glob(f"request_dump_{session_id}_*.json"))
        except OSError:
            pass
        for p in targets:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def get_session_delete_targets(self, session_id: str) -> List[str]:
        """Rows :meth:`delete_session` would remove: the session, then its
        recursive delegate children (branch/compression children are orphaned, not deleted)."""
        with self._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone():
                return []
            # The borrowed read connection, never self._conn (unlocked writer use).
            delegate_ids = _collect_delegate_child_ids(conn, [session_id])
        return [session_id, *sorted(delegate_ids)]

    def delete_session(
        self, session_id: str, sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        """Delete a session and its messages. Delegate children cascade (they'd
        resurface as orphans in pickers); branch/compression children are
        orphaned (parent -> NULL). *sessions_dir*: also remove transcript files.
        *expected_delete_ids*: proceed only if parent + delegate cascade still
        equals that set (export-before-delete fails closed if a new delegate
        appeared); the tree is re-walked inside the transaction on purpose (TOCTOU)."""
        removed_delegate_ids: List[str] = []
        expected_ids = set(expected_delete_ids) if expected_delete_ids is not None else None
        def _do(conn):
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is None:
                return False
            if expected_ids is not None and expected_ids != {
                session_id, *_collect_delegate_child_ids(conn, [session_id])
            }:
                return False
            removed_delegate_ids.extend(_delete_delegate_children(conn, [session_id]))
            conn.execute(  # orphan remaining children (branches) so FK is satisfied
                "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._delete_unreferenced_system_prompts(conn)
            return True
        deleted = self._execute_write(_do)
        if deleted:
            for sid in removed_delegate_ids + [session_id]:
                self._remove_session_files(sessions_dir, sid)
        return bool(deleted)

    def delete_session_if_empty(self, session_id: str, sessions_dir: Optional[Path] = None) -> bool:
        """Delete *session_id* only if it has no messages, no title and no
        children (a parent that spawned work is not "empty"), so start-and-quit
        sessions don't pile up in /resume. Check and delete share one
        transaction so a concurrently flushed message can't be lost."""
        def _do(conn):
            cursor = conn.execute(
                """
                DELETE FROM sessions
                WHERE id = ?
                  AND title IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions child
                      WHERE child.parent_session_id = sessions.id
                  )
                """,
                (session_id,),
            )
            if cursor.rowcount > 0:
                self._delete_unreferenced_system_prompts(conn)
            return cursor.rowcount > 0
        deleted = self._execute_write(_do)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return bool(deleted)

    def delete_sessions(self, session_ids: List[str], sessions_dir: Optional[Path] = None) -> int:
        """Bulk delete (dashboard multi-select) with :meth:`delete_session`
        semantics per row, in ONE transaction so a partial failure can't leave
        "messages gone, row still there". Unknown ids are skipped (UI selection
        can race another tab's delete: succeed-on-the-rest). Returns the number
        that actually existed and were deleted."""
        if not session_ids:
            return 0
        unique_ids = list({sid for sid in session_ids if isinstance(sid, str) and sid})
        if not unique_ids:
            return 0
        removed_ids: list[str] = []
        removed_delegate_ids: list[str] = []
        def _do(conn):
            # Filter to IDs that actually exist: return the real deleted count.
            existing = [row["id"] for row in conn.execute(
                f"SELECT id FROM sessions WHERE id IN ({','.join('?' * len(unique_ids))})",
                unique_ids,
            ).fetchall()]
            if not existing:
                return 0
            existing_placeholders = ",".join("?" * len(existing))
            removed_delegate_ids.extend(_delete_delegate_children(conn, existing))
            conn.execute(  # orphan children whose parent is in the kill list (FK)
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({existing_placeholders})",
                existing,
            )
            conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({existing_placeholders})", existing,
            )
            conn.execute(f"DELETE FROM sessions WHERE id IN ({existing_placeholders})", existing)
            self._delete_unreferenced_system_prompts(conn)
            removed_ids.extend(existing)
            return len(existing)
        count = self._execute_write(_do)
        for sid in removed_delegate_ids + removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    #: Shared by count_empty_sessions / delete_empty_sessions so badge and sweep
    #: agree. ``message_count`` counts live rows only — rewind and compaction
    #: reset it to 0 while keeping dropped turns as ``active = 0`` (the only
    #: recoverable copy) — so NOT EXISTS is the authority; message_count = 0 is
    #: a cheap prefilter.
    _EMPTY_SESSION_WHERE = (
        "message_count = 0 AND ended_at IS NOT NULL AND archived = 0 AND NOT EXISTS ("
        "SELECT 1 FROM messages WHERE messages.session_id = sessions.id)"
    )

    def count_empty_sessions(self) -> int:
        """Count of empty, ended, non-archived sessions (:data:`_EMPTY_SESSION_WHERE`).
        The ended_at guard matches prune_sessions: a fresh session whose first
        message hasn't landed is never sniped out from under the runtime."""
        return self._read_one(f"SELECT COUNT(*) FROM sessions WHERE {self._EMPTY_SESSION_WHERE}")[0]

    def delete_empty_sessions(self, sessions_dir: Optional[Path] = None) -> int:
        """Delete every empty, ended, non-archived session (:data:`_EMPTY_SESSION_WHERE`)
        in one transaction, orphaning (not cascading) children so branch/subagent
        transcripts survive. Transcript files are swept too: the gateway can
        leave a stub request_dump_* if it crashed before the first reply."""
        removed_ids: list[str] = []
        def _do(conn):
            session_ids = {row["id"] for row in conn.execute(
                f"SELECT id FROM sessions WHERE {self._EMPTY_SESSION_WHERE}"
            ).fetchall()}
            if not session_ids:
                return 0
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({','.join('?' * len(session_ids))})",
                list(session_ids),
            )
            for sid in session_ids:
                # DELETE FROM messages is paranoia — the selector's NOT EXISTS
                # probe proved these own no rows — but a row inserted between
                # the SELECT and here would otherwise dangle (clean FK state).
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)
        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    def archive_sessions(
        self, older_than_days: Optional[float] = None, source: str = None, **filters,
    ) -> int:
        """Bulk soft-hide with prune_sessions' filter surface, via
        set_session_archived so each lineage flips as a unit. ``archived``
        defaults to False so repeat runs are idempotent. Returns matches."""
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(older_than_days=older_than_days, source=source, **filters)
        for row in rows:
            self.set_session_archived(row["id"], True)
        return len(rows)

    # ── Meta key/value (scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read state_meta[key]. On self._lock, not _read_ctx: fts_rebuild_step
        reads progress before its write transaction, and a read-only WAL
        connection would not see uncommitted meta writes."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None) -> None:
        """Upsert state_meta[key]. With ``cursor`` the write is inline (_init_schema
        already holds a transaction; _execute_write would nest BEGIN IMMEDIATE and deadlock)."""
        sql = (
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        if cursor is not None:
            cursor.execute(sql, (key, value))
        else:
            self._write_sql(sql, (key, value))

    def retag_kanban_worker_sessions(self, workspaces_root: str) -> int:
        """Retag legacy kanban worker rows (spawned without HERMES_SESSION_SOURCE)
        from ``cli`` to ``kanban``, identified by cwd under the board's workspaces
        root — a path only the dispatcher runs sessions in. Gated once per root
        via state_meta. Returns rows retagged."""
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0
        gate = f"kanban_worker_source_retagged:{prefix}"
        if self.get_meta(gate) == "1":
            return 0
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET source = 'kanban' "
                "WHERE source = 'cli' AND (cwd = ? OR cwd LIKE ? ESCAPE '\\')",
                (prefix, _escape_like(prefix) + "/%"),
            )
            # rowcount BEFORE set_meta reuses this cursor for its INSERT.
            retagged = cursor.rowcount or 0
            self.set_meta(gate, "1", cursor=cursor)
            return retagged
        return self._execute_write(_do)

    def list_meta_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """``[(key, value), ...]`` for state_meta keys starting with the literal
        ``prefix`` (LIKE wildcards escaped) — e.g. ``loop:<session_id>`` rows."""
        if not prefix:
            return []
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'", (escaped + "%",),
        )
        return [(row[0], row[1]) for row in rows]

    # FTS5 tables merged on optimize; trigram may be disabled and cjk exists only
    # with the loadable tokenizer, so each is probed before touching (optimize_fts).
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")

    def maybe_auto_archive(
        self, idle_days: float = 3, min_interval_hours: int = 24, exclude_pinned: bool = True,
    ) -> Dict[str, Any]:
        """Idempotent auto-archive of sessions idle for ``idle_days`` (ages on last
        activity, non-destructive). ``state_meta['last_auto_archive']`` gates
        runs within ``min_interval_hours``; safe to call opportunistically.
        Never raises: {"skipped", "archived", "error"?}."""
        result: Dict[str, Any] = {"skipped": False, "archived": 0}
        try:
            last_raw = self.get_meta("last_auto_archive")
            now = time.time()
            if last_raw:
                try:
                    if now - float(last_raw) < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta; treat as no prior run
            archived = result["archived"] = self.archive_stale_sessions(idle_days, exclude_pinned=exclude_pinned)
            # Record even a zero-archive run so we don't re-sweep every call.
            self.set_meta("last_auto_archive", str(now))
            if archived > 0:
                logger.info(
                    "state.db auto-archive: archived %d session(s) idle >= %s days", archived,
                    idle_days,
                )
        except Exception as exc:
            logger.warning("state.db auto-archive failed: %s", exc)
            result["error"] = str(exc)
        return result

class AsyncSessionDB:
    """Async door onto SessionDB: each call is offloaded via asyncio.to_thread so a
    blocking SQLite call never freezes the event loop (no method returns a live cursor)."""

    def __init__(self, db: "SessionDB") -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr
        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)
        return _offloaded
