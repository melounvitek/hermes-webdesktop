#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
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
# Intrinsic persistence marker stamped on message dicts that are known-durable
# (#92231). One shared constant with agent.context_compressor (this module
# already imports agent.* at module level, and context_compressor is a
# transitive dependency via hermes_state_common). run_agent keeps its own
# predating copy — hermes_state cannot import run_agent (circular) — guarded
# by test_marker_constant_in_sync.
from agent.context_compressor import (  # noqa: F401  (re-exported; tests import it from here)
    _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY,
)
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, cast

from hermes_state_common import (  # noqa: F401  (re-exported for back-compat)
    AUTO_VACUUM_MIN_FREELIST_RATIO,
    _BRANCH_CHILD_SQL,
    _COMPRESSION_CHILD_SQL,
    _FTS_CJK_TRIGGERS,
    _FTS_TRIGGERS,
    _LISTABLE_CHILD_SQL,
    _PREVIEW_ELIGIBLE_SQL,
    _PREVIEW_RAW_SELECT,
    _RECOVERABLE_END_REASONS,
    _RECOVERABLE_END_REASONS_SQL,
    is_automatic_end_reason,
    _RESET_END_REASONS,
    _RESET_END_REASONS_SQL,
    _ephemeral_child_sql,
    _legacy_reset_child_sql,
    _shape_preview,
    _sql_session_last_active,
    _sql_session_last_active_by_id,
    escape_like as _escape_like,
    DEFERRED_INDEX_SQL,
    FTS_CJK_STALE_KEY,
    FTS_REBUILD_DEFERRAL_KEY,
    FTS_SQL,
    FTS_STALE_KEY,
    FTS_STORAGE_VERSION,
    FTS_TRIGRAM_SQL,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    MAX_FTS5_QUERY_CHARS,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    _PREVIEW_CONTENT_SQL,
    _PREVIEW_HEAD_CHARS,
    _PREVIEW_MAX_CHARS,
    _PREVIEW_SCAFFOLD_WINDOW,
    _PREVIEW_SCAFFOLDED_SQL,
    _acquire_db_flock,
    _clear_lock_holder_record,
    _describe_lock_holder,
    _read_lock_holder_record,
    is_advisory_lock_contention,
    stat_db_file_identity as _stat_db_file_identity,
)
from hermes_state_portability import SessionPortabilityMixin
from hermes_state_telegram import SessionTelegramTopicsMixin, _normalize_telegram_topic_profile_name  # noqa: F401  (re-exported for back-compat)
from hermes_state_schema import SessionSchemaMixin
from hermes_state_dbfile import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _HEADER_PROBE_FDS,
    _HEADER_PROBE_LOCK,
    _HERMES_CMDLINE_MARKERS,
    _RETIRED_HEADER_PROBE_FDS,
    _canonical_sqlite_path,
    _concrete_state_db_holder_pids,
    _connect_tracked_db,
    _is_inactive_orphan_desktop_holder,
    _looks_like_hermes,
    _pread_db_header,
    _read_proc_cmdline,
    _read_sqlite_application_id,
    _stat_sqlite_sidecar_identity,
    _watched_sqlite_sidecar_paths,
    collect_state_db_stats,
    count_db_holders,
    is_zeroed_state_db,
    iter_deleted_sqlite_sidecar_holders,
    quarantine_cross_process_lock,
    quarantine_zeroed_state_db,
    refuse_deleted_wal_generation,
)
from hermes_state_messages import SessionMessagesMixin
from hermes_state_wal import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    WalUnsupportedError,
    _SYNCHRONOUS_FULL,
    _SYNCHRONOUS_LEVELS,
    _SYNCHRONOUS_NAMES,
    _WAL_INCOMPAT_MARKERS,
    _WAL_SIZE_LIMIT_BYTES,
    _apply_delete_for_wal_reset_bug,
    _apply_macos_checkpoint_barrier,
    _apply_synchronous_pragma,
    _apply_wal_size_limit,
    _database_has_content,
    _delete_overridden_warned_lock,
    _delete_overridden_warned_paths,
    _enforce_macos_synchronous_full,
    _journal_upgrade_warned_lock,
    _journal_upgrade_warned_paths,
    _log_configured_delete_overridden_once,
    _log_journal_mode_upgrade_once,
    _log_wal_fallback_once,
    _log_wal_reset_bug_once,
    _on_disk_journal_mode,
    _set_journal_mode_no_wait,
    _wal_fallback_warned_lock,
    _wal_fallback_warned_paths,
    _wal_reset_bug_warned_lock,
    _wal_reset_bug_warned_paths,
    _wal_reset_repair_hint,
    apply_database_pragmas,
    apply_wal_with_fallback,
    is_sqlite_wal_reset_vulnerable,
    resolve_journal_mode,
    resolve_synchronous_level,
    sqlite_source_id,
)
from hermes_state_repair import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _DB_SIDECAR_SUFFIXES,
    _FINGERPRINT_SAMPLE_BYTES,
    _FINGERPRINT_VOLATILE_HEADER_RANGES,
    _MAX_MALFORMED_BACKUPS,
    _MAX_PERSISTENT_REPAIR_ATTEMPTS,
    _REPAIR_BACKUP_FREE_FRACTION,
    _REPAIR_BACKUP_MIN_FREE_BYTES,
    _REPAIR_LOCK_POLL_SECONDS,
    _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND,
    _backup_content_identity,
    _backup_db_file,
    _bump_schema_cookie,
    _claim_repair_attempt,
    _connect_repair_durable,
    _copy_database_snapshot,
    _cross_process_repair_lock,
    _db_fingerprint,
    _db_opens_cleanly,
    _exclusive_repair_db_guard,
    _existing_malformed_backups,
    _live_writer_holds_db,
    _mask_volatile_header,
    _persistent_repair_attempts_exhausted,
    _persistent_repair_exhausted_error,
    _probe_journal_mode_for_repair,
    _prune_malformed_backups,
    _read_repair_ledger,
    _reapply_durability_barriers,
    _record_repair_outcome,
    _release_auto_maintenance_lock,
    _repair_backup_headroom_bytes,
    _repair_failure_consumes_attempt,
    _repair_ledger_path,
    _repair_scratch_space_error,
    _repair_snapshot_timeout_seconds,
    _repair_state_db_schema_locked,
    _restore_journal_mode_after_repair,
    _run_repair_strategies,
    _try_acquire_auto_maintenance_lock,
    _unlink_db_triple,
    apply_durability_barriers,
    preflight_db_writability,
    repair_state_db_schema,
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
    """Resolve a transcript safety limit from config at call time.

    Reads ``sessions.<key>`` from config.yaml lazily (avoiding a circular
    import at module load) and falls back to the module constant when the
    config subsystem is unavailable (scaffold installs, stripped test
    environments). A value of 0 disables the guard entirely. No caching:
    ``load_config_readonly`` is already mtime-cached, and resolving fresh
    keeps tests that monkeypatch config or the module constants working.
    """
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
    return _configured_transcript_limit(
        "max_resume_messages", MAX_SAFE_RESUME_MESSAGES
    )


def resolved_max_export_messages() -> int:
    """Config-resolved in-memory export guard limit (0 disables the guard)."""
    return _configured_transcript_limit(
        "max_export_messages", MAX_SAFE_EXPORT_MESSAGES
    )


class SessionResumeTooLargeError(ValueError):
    def __init__(
        self,
        message_count: int,
        limit: int = MAX_SAFE_RESUME_MESSAGES,
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
    def __init__(
        self,
        session_id: str,
        message_count: int,
        limit: int = MAX_SAFE_EXPORT_MESSAGES,
    ):
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
    """Return True only when a structured lock holder's local PID is gone.

    Compression locks are stored in a host-local SQLite database and holder
    IDs created by ``conversation_compression`` start with ``pid=<n>``. A
    process killed during gateway shutdown cannot release its lease, so waiting
    for the full TTL makes every new turn repeatedly attempt compaction. Reclaim
    only when the kernel proves that PID no longer exists; legacy/unstructured
    holders, same-process holders, permission errors, and any probe doubt
    remain protected until normal TTL expiry (conservative: PID reuse must
    never steal a live lease, and a wrongly-kept lease self-heals via TTL).
    """
    match = _COMPRESSION_LOCK_HOLDER_PID_RE.search(holder or "")
    if match is None:
        return False
    try:
        pid = int(match.group(1))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        # Same-process holder (e.g. another thread's live lease): never
        # self-reclaim — the lease refresher and release path own it.
        return False
    if psutil is not None:
        try:
            # psutil is the canonical cross-platform liveness answer
            # (CONTRIBUTING.md "Critical rules" #1). pid_exists() reports
            # recycled PIDs as alive — conservative, the TTL still applies.
            return not psutil.pid_exists(pid)
        except Exception:
            return False  # any doubt → keep the lease until TTL expiry
    # Scaffold-phase fallback only (psutil missing), and POSIX-only: stdlib
    # os.kill(pid, 0) is NOT a no-op probe on Windows (bpo-14484 — sig=0 maps
    # to CTRL_C_EVENT and can kill the target's console group). Without psutil
    # a Windows host stays TTL-only; the lease TTL remains the recovery path.
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
    """Replace lone surrogates when *value* is text; pass anything else through.

    sqlite3 encodes bound ``str`` parameters as UTF-8 and raises
    ``UnicodeEncodeError`` on lone surrogates (U+D800..U+DFFF), so a single
    such code point anywhere in a message aborts the whole write. No-op for
    well-formed text.
    """
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def workspace_key(row: Dict[str, Any]) -> Optional[str]:
    """A session's workspace grouping key: its git repo root when known, else
    its cwd.

    Branch is deliberately excluded so checking out a new branch doesn't
    fragment a workspace's session history. Returns None for cwd-less (unbound)
    sessions. Both fields are already recorded on ``sessions`` — this just picks
    the coarser identity for grouping/filtering.
    """
    root = (row.get("git_repo_root") or "").strip()
    if root:
        return root

    cwd = (row.get("cwd") or "").strip()
    return cwd or None


def _delegate_from_json(col: str = "model_config") -> str:
    return f"json_extract(COALESCE({col}, '{{}}'), '$._delegate_from')"


# Sentinel returned by SessionDB._merge_model_config_json when the session row
# doesn't exist and on_missing="skip" — distinguishes "no row" from the legal
# None result ("merged config is empty → store NULL").
_MODEL_CONFIG_ROW_MISSING = object()

# Billing-bucket classes that aren't a routable provider identity on their
# own — used by session_gateway_runtime's billing_provider fallback and by
# tui_gateway.server._stored_session_runtime_overrides. A session that
# persisted only one of these (never ran /model) must fall back to the
# ambient config default rather than restore a bare bucket. Shared here so
# both consumers stay in sync (previously duplicated as a set in
# tui_gateway/server.py).
_BARE_BILLING_PROVIDERS = frozenset({"auto", "custom"})


def _cwd_prefix_clause(cwd_prefix: str) -> Tuple[str, List[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    # ``_`` and ``%`` are LIKE wildcards but ordinary characters in a path
    # (``my_project``), so an unescaped prefix also matches sibling directories.
    # Escape the needle and pair it with ESCAPE; the literal separator
    # backslash in the Windows pattern needs escaping for the same reason. The
    # ``=`` arm is an exact compare and keeps the raw prefix.
    esc = _escape_like(prefix)
    return (
        "(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\' OR s.cwd LIKE ? ESCAPE '\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )


def _workspace_key_clause(key: str) -> Tuple[str, List[str]]:
    """Match sessions whose ``workspace_key(row)`` equals ``key``.

    Mirrors :func:`workspace_key`: a session belongs to workspace ``key``
    when its recorded ``git_repo_root`` equals ``key``, or — for rows that
    predate per-session git metadata — when its ``cwd`` is at or under
    ``key`` (so a session started in ``repo/src`` still groups with ``repo``).
    Used by ``hermes -c``/``--resume`` to continue the most recent session in
    the *current* workspace rather than the global MRU.
    """
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


def _collect_delegate_child_ids(conn, parent_ids: List[str]) -> List[str]:
    """Delegate-subagent ids to cascade-delete with *parent_ids*.

    Only rows carrying the ``_delegate_from`` marker (set at creation, and
    backfilled by the v16 migration) — generic untagged children keep the
    orphan-don't-delete contract. Walks marker chains recursively so an
    orchestrator subagent's own delegate children go too (FK safety).
    """
    df = _delegate_from_json()
    seeds = {sid for sid in parent_ids if sid}
    # Seed the visited set with the parents themselves. A delegation marker
    # chain can loop back onto a parent — a cycle, or a parent that is also
    # another parent's delegate child when several ids are deleted at once —
    # and without this guard that parent would be collected as one of its own
    # descendants and cascade-deleted along with all of its messages. Callers
    # delete the parents separately, so parents must never appear in the
    # returned child set. (#49148)
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
    # Return only the discovered children — never the parents themselves.
    return [sid for sid in found if sid not in seeds]


def _delete_delegate_children(conn, parent_ids: List[str]) -> List[str]:
    ids = _collect_delegate_child_ids(conn, parent_ids)
    if ids:
        ph = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", ids)
        # FK safety: orphan any untagged stragglers pointing at a doomed row.
        conn.execute(
            f"UPDATE sessions SET parent_session_id = NULL "
            f"WHERE parent_session_id IN ({ph})",
            ids,
        )
        conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", ids)
    return ids

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

# How long SessionDB stops attempting read-only opens after one fails, before
# probing again. Long enough that a genuinely unreadable file isn't retried per
# query; short enough that transient fd pressure doesn't strand the read pool.
_READ_OPEN_RETRY_SECONDS = 60.0

# Transient SQLITE_IOERR retry budget for READ-ONLY opens (#100436). A WAL
# database being actively written (checkpoint, WAL reset/truncate, frame
# flush) can surface "disk I/O error" to a concurrent ``mode=ro`` reader in
# a millisecond-wide transition window: the read-only connection cannot
# perform the WAL recovery a read through a stale or mid-update -shm file
# needs, because recovery requires writing the -shm index, which mode=ro
# refuses. The window closes on its own (the writer finishes the transition),
# so a bounded number of short retries makes the open succeed instead of
# 500-ing the whole /api/sessions poll (or any other read-only opener).
# Deliberately NOT attempted on writable opens: a writer owns the
# transition, so an IOERR there means a real storage/fd problem.
_READ_ONLY_IOERR_RETRY_ATTEMPTS = 3
_READ_ONLY_IOERR_RETRY_BACKOFF_S = 0.05

# Hard ceiling on read-only connections ALIVE at once against one database
# FILE — pooled idle ones and checked-out ones together, summed over every
# SessionDB in this process that points at that file. See _PathReadBudget.
#
# Deliberately one constant for both the pool's maxsize and the permit count,
# because bounding only the pool bounds the wrong thing. A LifoQueue caps how
# many connections are *returned*; it says nothing about how many are *open*.
# With an open-on-miss checkout, N readers arriving on an empty pool all miss,
# all open, and peak at N — the surplus is closed on release, so nothing
# accumulates forever, but EMFILE is a peak-instant condition and the burst
# that empties the pool is exactly the burst that exhausts the fd table.
#
# So a connection holds a permit for its whole lifetime: acquired in
# _get_read_conn() before the open, released in _close_read_conn() after the
# close. Once permits are gone the read path degrades to the locked writer
# connection instead of opening more descriptors — slower under load, which is
# the correct trade against a process-wide wedge the supervisor cannot see.
_READ_POOL_MAX = 8

# Hard ceiling on read-only connections ALIVE at once in this PROCESS, across
# every state.db it has open.
#
# _READ_POOL_MAX bounds one file. A multiplexed gateway serves N profiles from
# one process and each profile has its OWN state.db, so a per-file ceiling
# still lets the descriptor cost grow with the profile count — the same shape
# as the per-instance bug, one level out (#98573).
#
# Three profiles' worth. Past it, readers on the (N+1)th file degrade to their
# writer connection instead of opening descriptors, which is the same trade
# _READ_POOL_MAX makes and for the same reason: a slow read path is
# recoverable, a process-wide EMFILE is not.
_READ_POOL_PROCESS_MAX = 24

# Warn when one process accumulates more than this many SessionDB handles on a
# single file. Not a limit — writer connections cannot be rationed the way read
# connections can — a diagnostic for the duplicate-handle class of bug.
_HANDLES_PER_PATH_WARN = 4

# Descriptors kept in reserve for everything that is NOT this module: httpx
# sockets, terminal subprocess pipes, log files.
#
# The ceilings above bound Hermes's SQLite descriptors, which is only ever part
# of the fd table. The #98573 report is exactly that case: ~20 state.db
# descriptors were not the whole 256, they were the share that pushed httpx and
# terminal pipes over, and the EMFILE surfaced in tools/terminal_tool.py rather
# than here. So the read pool also yields when the PROCESS is close to its
# limit, whatever is consuming it.
_FD_HEADROOM_RESERVE = 64

# The fd count is a directory listing; cache it briefly so a burst of reads
# does not turn one syscall per query. Stale by at most this long, which can
# let through at most the ceiling's worth of opens — already bounded above.
_FD_USAGE_CACHE_SECONDS = 0.25

_process_read_permits = threading.BoundedSemaphore(_READ_POOL_PROCESS_MAX)

# Count of read opens refused because the process was low on descriptors. The
# only externally visible signal that the guard is firing; guarded by
# _read_budgets_lock.
_read_open_denied_fd_headroom = 0

_fd_usage_lock = threading.Lock()
_fd_usage_cache: "tuple[float, Optional[int]]" = (0.0, None)


def _open_fd_count() -> Optional[int]:
    """Descriptors open in THIS process, or None when it cannot be measured.

    ``/proc/self/fd`` on Linux, ``/dev/fd`` on macOS and the BSDs. Windows has
    neither, and no RLIMIT_NOFILE to compare against, so the guard is inert
    there — which is correct: the CRT limit is thousands of handles, not 256.
    """
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError as exc:
            if exc.errno in (errno.EMFILE, errno.ENFILE):
                # The probe itself could not get a descriptor. That IS the
                # answer: there is no headroom.
                return -1
            continue
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
    """Whether the process can spare a descriptor for a new read connection.

    Fails OPEN when the platform cannot be measured (Windows, no fd directory,
    unlimited RLIMIT_NOFILE): an unmeasurable platform is not a tight one, and
    refusing every read there would be a self-inflicted convoy. Fails CLOSED
    only on evidence — a measured shortfall, or a probe that could not get a
    descriptor of its own.
    """
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
    if cached < 0:
        return False
    return (soft - cached) > _FD_HEADROOM_RESERVE


def _reclaim_idle_read_conn_anywhere() -> bool:
    """Close one idle read connection on ANY path in this process.

    The process ceiling is shared across files, so the connection that has to
    go to make room may belong to a different database entirely — a profile
    that has been quiet for an hour should not hold descriptors the profile
    being served right now needs.
    """
    with _read_budgets_lock:
        budgets = list(_read_budgets.values())
    for budget in budgets:
        if budget.reclaim_idle():
            return True
    return False


class _PathReadBudget:
    """The read-connection permits for ONE database file, shared process-wide.

    ``_READ_POOL_MAX`` used to be enforced by a ``BoundedSemaphore`` owned by
    each SessionDB, which bounded the wrong noun: the descriptors are spent on
    a *file*, so N SessionDB objects on one state.db each got their own
    allowance and peak scaled as ``N x (1 + _READ_POOL_MAX)``.  A long-lived
    gateway holds at least two (``SessionStore`` and ``GatewayRunner`` open
    independent handles per profile path) and the count grows with the profile
    count, which is how a healthy process walked into EMFILE — #98573.

    Holding the permits here instead makes the ceiling mean what its docstring
    always claimed: read connections ALIVE at once against this path.

    One consequence has to be handled rather than documented away.  A pooled
    idle connection keeps its permit, so the first instance to warm up would
    otherwise pin all eight and every later instance — a cron job's transient
    handle, a second profile's store — would be permanently demoted to the
    locked writer connection.  That is why a permit miss first reclaims an
    IDLE connection from a peer instance on the same path: idle descriptors
    are transferable, in-use ones are not.
    """

    def __init__(self) -> None:
        self.permits = threading.BoundedSemaphore(_READ_POOL_MAX)
        self._lock = threading.Lock()
        # Weak so a SessionDB that is dropped without close() cannot pin its
        # peers' budget object; __del__ still runs close() and returns the
        # permits.
        self._members: "weakref.WeakSet[SessionDB]" = weakref.WeakSet()
        self._duplicate_handles_warned = False

    def register(self, db: "SessionDB") -> None:
        with self._lock:
            self._members.add(db)
            handles = len(self._members)
            warn = (
                handles > _HANDLES_PER_PATH_WARN
                and not self._duplicate_handles_warned
            )
            if warn:
                self._duplicate_handles_warned = True
        if warn:
            # The read connections are capped; the WRITER connection each
            # handle holds is not, and cannot be — a SessionDB without one
            # cannot write. The only real bound on writers is not opening
            # redundant handles in the first place (which is what
            # GatewayRunner borrowing SessionStore's handle does, #98573), so
            # the next duplicate should be visible before it becomes an
            # incident rather than inferred from an lsof after one.
            logger.warning(
                "%d live SessionDB handles on %s in this process; each holds "
                "its own writer connection (read connections are capped at %d "
                "for the file). A long-lived process should share one handle "
                "per path.",
                handles,
                db.db_path,
                _READ_POOL_MAX,
            )

    def acquire(self, requester: "SessionDB") -> bool:
        """Take a permit for a new read connection, or refuse.

        Three gates, broadest first: the process's descriptor headroom, the
        process-wide read ceiling, then this file's ceiling. Refusing means
        the caller serves the read from the locked writer connection — slower,
        never an error.
        """
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
        if _process_read_permits.acquire(blocking=False):
            return True
        if not _reclaim_idle_read_conn_anywhere():
            return False
        # Another thread may take the freed permit first; that is a legitimate
        # loss, and the caller degrades to the writer lock rather than looping.
        return _process_read_permits.acquire(blocking=False)

    def _acquire_path_permit(self, requester: "SessionDB") -> bool:
        if self.permits.acquire(blocking=False):
            return True
        if not self.reclaim_idle(exclude=requester):
            return False
        return self.permits.acquire(blocking=False)

    def reclaim_idle(self, exclude: "Optional[SessionDB]" = None) -> bool:
        """Close one idle pooled connection held by a member. True if one went.

        Closing it runs release(), which returns both the path permit and the
        process permit, so this is the single reclaim primitive both ceilings
        use.
        """
        with self._lock:
            members = [db for db in self._members if db is not exclude]
        for member in members:
            if member._evict_one_idle_read_conn():
                return True
        return False


# canonical db path -> the permits for that file. Weak values: the budget
# lives exactly as long as some SessionDB on that path holds a strong
# reference to it, so a test that churns tmp_path databases does not grow
# this map for the life of the process.
_read_budgets: "weakref.WeakValueDictionary[str, _PathReadBudget]" = (
    weakref.WeakValueDictionary()
)
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


# Import-time snapshot used by _default_db_path() to detect a deliberately
# re-pointed DEFAULT_DB_PATH (tests monkeypatch the constant directly).
_IMPORT_DEFAULT_DB_PATH = DEFAULT_DB_PATH


def _default_db_path() -> Path:
    """Resolve the default state DB path at call time.

    ``DEFAULT_DB_PATH`` is computed when this module is first imported, which
    freezes the developer's real ``~/.hermes`` even when a test fixture later
    redirects ``HERMES_HOME`` — importing this module during collection was
    enough to point every default ``SessionDB()`` at the real state.db.

    Precedence:

    1. A deliberately re-pointed ``DEFAULT_DB_PATH`` (differs from the
       import-time snapshot — the established test escape hatch) wins.
    2. Otherwise resolve ``get_hermes_home()`` fresh so a runtime
       ``HERMES_HOME`` redirect takes effect regardless of import order.
    """
    if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH
    return get_hermes_home() / "state.db"


# ---------------------------------------------------------------------------
# Live-DB test-isolation guard
# ---------------------------------------------------------------------------
# Forensic evidence (Aug 2026, live developer machine): the production
# ~/.hermes/state.db accumulated pytest fixture rows — sessions with
# chat_id='chat-1'/'123'/'wx-chat' and gateway_routing scopes literally under
# /tmp/pytest-of-*/ — and a pytest-spawned process flipped the journal mode
# out from under the WAL-mode gateway writer, destroying committed
# transcripts ("Persisted transcript lagged live cached history ... possible
# FTS write corruption").  The hermetic conftest redirects HERMES_HOME per
# test, but any escape (a session-scoped fixture running before the autouse
# fixture, a subprocess child launched without HERMES_HOME, a stale worktree
# without the re-pin, or a developer shell that exports HERMES_HOME to the
# real home so the conftest session sandbox is skipped) silently fell
# through to the real database.
#
# This guard is the single choke point: EVERY ``SessionDB`` construction
# resolves its path here, so under pytest a resolution that lands on a
# production state.db fails hard instead of corrupting live data.  It is
# env-based (``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` are set by pytest
# and inherited by subprocess children), so it also protects children that
# never import the test conftest.

#: Escape hatch for the rare legitimate case (a test that genuinely needs
#: the real DB).  The in-tree conftest sets this for tests marked
#: ``@pytest.mark.live_system_guard_bypass``; scripts may set it explicitly.
_STATE_DB_GUARD_BYPASS = False

#: Env-carried twin of ``_STATE_DB_GUARD_BYPASS``.  A module global cannot
#: cross a process boundary, so a test that deliberately points a *child* at
#: the live DB has no way to opt out once ancestry arms the guard there.
#: Export this in the child's env instead.
_STATE_DB_GUARD_BYPASS_ENV = "HERMES_STATE_DB_GUARD_BYPASS"

#: Additional production roots to refuse (beyond the platform default
#: ``~/.hermes``).  The test conftest injects the pre-sandbox production
#: root here so custom-``HERMES_HOME`` deployments are covered too.
_STATE_DB_GUARD_EXTRA_DENY_ROOTS: Tuple[Path, ...] = ()


def _real_platform_state_root() -> Optional[Path]:
    """Resolve the REAL platform-default Hermes root for the guard.

    Deliberately avoids ``Path.home()`` / ``hermes_constants``: tests
    routinely monkeypatch ``Path.home`` to a tempdir, and ``hermes_state``
    is often imported lazily *while* such a patch is active — resolving
    through the patched callable would misidentify the test's own hermetic
    home as "production" (false positive) or, worse, miss the real one
    (false negative).  ``os.path.expanduser`` reads the HOME environment
    variable / passwd entry, which the hermetic conftest never rewrites.
    """
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


#: Env marker exported by the hermetic test conftest at the same moment it
#: redirects ``HERMES_HOME`` to the per-session tmp isolation root.  Its
#: value is that isolation root.  Unlike ``PYTEST_*`` (owned by pytest, and
#: routinely scrubbed by tests that rebuild a child environment), this marker
#: is OURS: it declares "this process tree is running under Hermes test
#: isolation", and it inherits into subprocess children by default — so a
#: child that received the patched ``HERMES_HOME`` also received the marker,
#: and a child that resolves a production DB while carrying it is, by
#: definition, an isolation escape (#82770).
_TEST_ISOLATION_MARKER_ENV = "HERMES_TEST_ISOLATION"


def _running_under_pytest() -> bool:
    """True when this process (or a parent test process) is a pytest run."""
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("PYTEST_VERSION")
        or os.environ.get(_TEST_ISOLATION_MARKER_ENV)
    )


#: Names that identify a pytest launcher in a process command line.  Matched
#: against the *basename* of each argv token so ``/tmp/pytest-of-dev/...``
#: paths — which do show up in real argv — cannot false-positive.
_PYTEST_LAUNCHER_NAMES = frozenset(
    {"pytest", "py.test", "pytest.exe", "py.test.exe"}
)

#: Memoised ancestry answer.  The process tree above us does not change in a
#: way that matters here, and the walk must not cost anything on the hot path.
_PYTEST_ANCESTOR: Optional[bool] = None


def _process_looks_like_pytest(proc: Any) -> bool:
    """True when *proc*'s command line is a pytest invocation.

    Covers both ``pytest ...`` (launcher on argv[0]) and ``python -m pytest``
    (launcher as a bare ``pytest`` token).  A process whose command line we
    cannot read is treated as "not pytest": guessing the other way would
    refuse production opens for unrelated reasons.
    """
    try:
        cmdline = proc.cmdline() or []
    except Exception:
        return False
    for arg in cmdline:
        try:
            token = str(arg).strip('"').strip("'")
            # Split on both separators on every host: os.path.basename is
            # POSIX-only under Linux and would leave a Windows-style path
            # intact, making the matcher's answer depend on the platform.
            name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
        except Exception:
            continue
        if name in _PYTEST_LAUNCHER_NAMES:
            return True
    return False


def _has_pytest_ancestor() -> bool:
    """True when some ancestor process of this one is a pytest run.

    ``_running_under_pytest`` reads ``PYTEST_*`` env vars, which a child
    spawned with a rebuilt environment loses at the same moment it loses the
    ``HERMES_HOME`` redirect: that child aims at the production DB *and*
    disarms the guard in one step (#82770).  Ancestry is the one test-context
    signal that survives an env rebuild, so it backs the env check up.

    Fails open (``False``) when ``psutil`` is unavailable or the walk errors —
    that restores the previous env-only behaviour rather than blocking real
    user runs on a psutil hiccup.
    """
    global _PYTEST_ANCESTOR
    if _PYTEST_ANCESTOR is not None:
        return _PYTEST_ANCESTOR
    found = False
    if psutil is not None:
        try:
            for parent in psutil.Process().parents():
                if _process_looks_like_pytest(parent):
                    found = True
                    break
        except Exception:
            found = False
    _PYTEST_ANCESTOR = found
    return found


def _in_test_context() -> bool:
    """True when this process is a test run, by environment or by ancestry.

    Order matters for cost: the env probe is two dict lookups and covers the
    common in-process case, so the ancestry walk only runs for processes the
    environment claims are ordinary user runs — and its answer is memoised,
    so a real ``hermes`` invocation pays for at most one walk.
    """
    if _running_under_pytest():
        return True
    return _has_pytest_ancestor()


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
    """True when *resolved* is a DB file of the real Hermes home *root*.

    Matches files directly in the root (``<root>/state.db``) and profile
    homes (``<root>/profiles/<name>/state.db``).  Deliberately does NOT
    match deeper scratch paths (e.g. repo worktrees that happen to live
    under ``~/.hermes/hermes-agent/...``) so hermetic tests using unusual
    tempdirs cannot false-positive.
    """
    if resolved.parent == root:
        return True
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) == 3 and parts[0] == "profiles"


def _ensure_test_isolation(db_path: Path) -> None:
    """Fail hard when a pytest-context process resolves a production DB.

    Raises ``RuntimeError`` before any connection, mkdir, journal-mode
    pragma, or byte probe can touch the live database.  No-op outside
    pytest and for hermetic (tmp ``HERMES_HOME``) paths.

    "pytest context" means environment *or* process ancestry — see
    :func:`_in_test_context`.  Env alone is not enough: a child spawned with
    a rebuilt environment loses ``PYTEST_*`` and ``HERMES_HOME`` together,
    which is precisely the state in which it writes to production (#82770).
    """
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
                "HERMES_HOME. If this test genuinely needs the live "
                "database, mark it with "
                "@pytest.mark.live_system_guard_bypass — or, for a spawned "
                f"child process, export {_STATE_DB_GUARD_BYPASS_ENV}=1 in "
                "its environment."
            )


# Last SessionDB() init error, per-process.  Surfaced in /resume and
# related slash-command error strings so users know WHY the DB is
# unavailable instead of getting a bare "Session database not available."
# Only SessionDB.__init__ writes to this; kanban_db.connect() failures
# do not update it (by design — kanban failures are reported via their
# own caller's error handling, not via /resume-style slash commands).
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear) the most recent state.db init failure.

    Thread-safe via _last_init_error_lock.  Callers pass a message to
    record a failure or None to clear.  SessionDB.__init__ only calls
    this to SET on failure — it deliberately does NOT clear on success,
    because in a multi-threaded caller (e.g. gateway / web_server per-
    request SessionDB() instantiation), a concurrent successful open
    racing past a different thread's failure would erase the cause
    string that thread's /resume handler is about to format.  Explicit
    clears (e.g. test fixtures) are still supported by passing None.
    """
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Return the most recent state.db init failure, if any.

    Slash-command handlers (``/resume``, ``/title``, ``/history``, ``/branch``)
    call this to surface the underlying cause in their error messages when
    ``_session_db is None``.  Returns ``None`` if SessionDB initialized
    successfully (or hasn't been attempted).
    """
    return _last_init_error


# Distinctive opening shared by both background-review harness prompts
# (_SKILL_REVIEW_PROMPT and _MEMORY_REVIEW_PROMPT in agent/background_review.py).
# Matched case-sensitively against the leading content of a user/system message.
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: Dict[str, Any]) -> bool:
    """True when ``msg`` is a persisted background-review harness prompt.

    These are user/system turns the forked skill/memory review agent wrote into
    a real session in older builds (before the ``_persist_disabled`` isolation
    fix). They instruct the agent to act as the curator under a hard tool
    restriction, so replaying them as live history hijacks the session.
    """
    if not isinstance(msg, dict):
        return False
    if msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    head = content.lstrip()
    return any(head.startswith(p) for p in _REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop background-review harness messages and the curator-mode assistant
    reply that immediately followed each one.

    Walk the list once; when a harness user/system message is found, skip it and
    also skip the next message if it is the assistant turn that answered it.
    Everything else passes through untouched and in order.
    """
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
                # The curator-mode reply to the harness prompt — drop it.
                continue
        out.append(msg)
    return out


# Matches a bare protocol/tool-name marker such as "[memory]" or "[skill_manage]".
_STALE_TOOL_CALL_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")


def _is_stale_tool_call_marker_message(msg: Dict[str, Any]) -> bool:
    """True when ``msg`` is a persisted assistant turn whose content is a bare
    bracketed marker (e.g. ``[memory]``) left over from a tool-call turn.

    Before the #78148 fix in ``agent.conversation_loop``, a local tool-call
    template could emit a bare marker as assistant content alongside a real
    tool call. The loop cached that marker as a fallback and later replayed
    it as the "final response", persisting it into the session. Sessions
    written before the fix can still carry these rows.
    """
    if not isinstance(msg, dict):
        return False
    if msg.get("role") != "assistant":
        return False
    if not msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    return bool(_STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()))


def _strip_stale_tool_call_markers(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Clear bare protocol-marker content persisted before the #78148 fix.

    Replaying "[memory]" as if the model had actually answered teaches the
    model, by example, to keep emitting the same marker in later turns — the
    exact symptom the issue reported. Only the stray ``content`` field is
    blanked; the tool call and its result are left untouched so provider
    tool_call/tool_result pairing stays intact. Sessions with no affected
    rows pass through unchanged.
    """
    repaired = 0
    for msg in messages:
        if _is_stale_tool_call_marker_message(msg):
            msg["content"] = ""
            repaired += 1
    if repaired:
        logger.info(
            "Cleared %d stale tool-call marker message(s) while restoring session (#78148)",
            repaired,
        )
    return messages


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    """Format a user-facing 'session DB unavailable' message with cause.

    When ``SessionDB()`` init fails, callers set ``_session_db = None`` and
    several slash commands (/resume, /title, /history, /branch) previously
    responded with a bare ``"Session database not available."`` — no
    indication of WHY.  This helper includes the captured cause (typically
    ``"locking protocol"`` from NFS/SMB) and points users at the known
    culprit so they can fix it themselves.

    Example output:
        Session database not available: locking protocol (state.db may be
        on NFS/SMB — see https://www.sqlite.org/wal.html).
    """
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
# A distinct, nastier failure class than a malformed FTS *inverted index*:
# the ``sqlite_master`` schema table itself becomes inconsistent — most
# commonly a DUPLICATE object definition, e.g. two ``CREATE VIRTUAL TABLE
# messages_fts`` rows.  SQLite parses the entire schema while preparing the
# FIRST statement on a connection, so on this class *every* statement raises
# before it runs — including ``PRAGMA journal_mode`` (which is why this trips
# in ``apply_wal_with_fallback`` during ``SessionDB.__init__``, long before
# ``_init_schema`` is reached) and even ``PRAGMA integrity_check`` and a plain
# ``DROP TABLE``.  The only operations that still work are
# ``PRAGMA writable_schema=ON`` plus direct ``sqlite_master`` surgery.
#
# Symptom users hit (Desktop/Dashboard show "no sessions" while 200+ JSON
# files sit on disk):
#   sqlite3.DatabaseError: malformed database schema (messages_fts) -
#   table messages_fts already exists
#
# The canonical ``sessions`` / ``messages`` data is intact in these cases —
# only the derived schema is broken — so recovery preserves all transcripts
# and merely rebuilds the FTS layer.
_MALFORMED_SCHEMA_MARKERS = ("malformed database schema",)
_MALFORMED_DB_MARKERS = (
    *_MALFORMED_SCHEMA_MARKERS,
    "database disk image is malformed",
)

# Process-global guard so auto-repair is attempted at most once per DB path
# per process (prevents repair loops and serialises concurrent web_server /
# gateway opens against the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()


def is_malformed_db_error(exc: BaseException) -> bool:
    """True for explicit malformed-schema or generic corrupt-image errors.

    This broad classifier is for diagnostics and explicit offline recovery
    dispatch. Runtime repair must use :func:`is_malformed_schema_error`, since
    a generic corrupt-image error does not identify the damaged object.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    return any(marker in str(exc).lower() for marker in _MALFORMED_DB_MARKERS)


# SQLITE_IOERR, matched as a plain substring so wrapped error strings still
# classify. Shared by the read-only open retry and the write-path BEGIN retry.
_DISK_IO_ERROR_MARKER = "disk i/o error"

# Broader set for HTTP classification: a read that failed for one of these
# reasons found the store BUSY, not gone. Callers map it to 503 (retry, the
# list was not cleared) instead of 500. Corruption is deliberately absent —
# a malformed store must surface, not be retried into a timeout.
_TRANSIENT_SQLITE_MARKERS = (
    _DISK_IO_ERROR_MARKER,
    "database is locked",
    "database table is locked",
    "busy",
)


def is_transient_sqlite_error(exc: BaseException) -> bool:
    """True when a SQLite failure means "busy right now", not "damaged".

    One predicate so the read paths cannot drift apart on what counts as
    recoverable: the read-only open retry, and the HTTP 503-vs-500 split on
    the session-list endpoints, classify the same way.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_SQLITE_MARKERS)


def _is_transient_read_only_ioerr(exc: sqlite3.OperationalError, *, attempt: int) -> bool:
    """True when a read-only open should be retried rather than raised.

    A ``mode=ro`` connection cannot perform WAL recovery (recovery needs to
    write the -shm index, which read-only mode refuses), so a concurrent WAL
    checkpoint / reset / frame-flush can surface ``SQLITE_IOERR`` ("disk I/O
    error") to a reader on an otherwise healthy database (#100436). The
    transition is millisecond-scale, so a bounded number of short retries
    clears it without changing classification for genuine storage failures —
    a persistent IOERR still exhausts the budget and propagates.
    """
    return (
        attempt < _READ_ONLY_IOERR_RETRY_ATTEMPTS
        and _DISK_IO_ERROR_MARKER in str(exc).lower()
    )


def is_malformed_schema_error(exc: BaseException) -> bool:
    """True only when SQLite explicitly reports malformed schema text.

    A generic ``database disk image is malformed`` error is SQLITE_CORRUPT
    and may come from any B-tree or freelist page.  It does not prove that
    canonical rows are intact, so runtime schema/FTS repair must fail closed.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    return any(marker in str(exc).lower() for marker in _MALFORMED_SCHEMA_MARKERS)


# Markers that mean the host filesystem cannot accept another write. Kept as
# plain substrings so OSError, sqlite3.OperationalError, and wrapped RPC
# error strings all match the same helper.
_DISK_FULL_MARKERS = (
    "no space left on device",
    "not enough space",
    "database or disk is full",  # SQLITE_FULL
    "disk full",
    "full disk",
    "enospc",
)


def is_disk_full_error(exc: BaseException | str | None) -> bool:
    """True when *exc* (or a stringified error) is a disk-full / ENOSPC failure.

    Covers:
      * ``OSError`` with ``errno.ENOSPC``
      * SQLite ``OperationalError: database or disk is full`` (SQLITE_FULL)
      * Plain English / errno strings that survive RPC wrapping
    """
    if exc is None:
        return False
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    text = exc if isinstance(exc, str) else str(exc)
    lowered = text.lower()
    return any(marker in lowered for marker in _DISK_FULL_MARKERS)


# Every cause bucket classify_persistence_error can return. Consumers that
# enumerate causes (e.g. the cron scheduler's explainer-variant suppression)
# must iterate this tuple instead of hardcoding the list, so adding a bucket
# can never silently desynchronize them.
PERSISTENCE_ERROR_CAUSES = (
    "locked",
    "compression",
    "compression_closed",
    "turn_lease",
    "corrupt",
    "replaced",
    "disk",
    "unknown",
)


# Markers that mean the database FILE itself is structurally damaged.  Kept
# as plain substrings so sqlite3.DatabaseError, wrapped RPC strings, and
# logged message text all match the same helper.  NOTE: "database disk image
# is malformed" contains the word "disk", so this check MUST run before the
# disk-full/readonly bucket in classify_persistence_error — otherwise real
# B-tree corruption gets reported to the user as "free some disk space"
# (the misdiagnosis documented on #77386).
_DB_CORRUPTION_MARKERS = (
    "malformed",              # "database disk image is malformed" (SQLITE_CORRUPT)
    "file is not a database", # SQLITE_NOTADB (also connection-level poisoning)
    "not a database",
    "database corruption",
)


def classify_persistence_error(exc_or_str) -> str:
    """Classify a session-persistence failure into a coarse cause bucket.

    Fast-failing a turn on a SessionDB write error is deliberate (the
    transcript would otherwise be lost on restart), but the *guidance* the
    user gets must match the cause: sustained SQLite write-lock contention
    ("database is locked" on a shared state.db) needs "storage was busy,
    send it again", while a full disk or read-only database needs the
    disk-space/permissions advice. Returns one of PERSISTENCE_ERROR_CAUSES:

    * ``"locked"``  — SQLite lock/busy contention (another process holds the
      database write lock); transient, retry-later guidance applies.
    * ``"compression"`` — a live compression lease refused the transcript
      write; the database itself is healthy and unlocked.
    * ``"compression_closed"`` — the write targeted a session already
      rotated (closed) by compression and no live continuation was adopted;
      the store is healthy — the client must refresh/adopt the new session
      id, so disk-space advice would be a misdiagnosis.
    * ``"turn_lease"`` — a presented session-turn-lease holder no longer
      owns the conversation (expired, released, or reclaimed); fail-fast
      fencing, not a storage fault.
    * ``"corrupt"`` — the database file itself is structurally damaged
      (``database disk image is malformed`` / SQLITE_NOTADB).  Distinct from
      ``"disk"``: freeing space cannot help, the user needs the repair path
      (``hermes doctor`` / automatic schema surgery).
    * ``"replaced"`` — the ``state.db`` path no longer names the file this
      process opened (out-of-band ``cp``/``mv``/restore). In-file FTS repair
      cannot help; writes to the live handle must stop.
    * ``"disk"``    — disk full / read-only / permission-shaped failures
      (delegates the disk-full patterns to :func:`is_disk_full_error` so the
      two classifiers can never drift apart — e.g. ENOSPC).
    * ``"unknown"`` — anything else (or no visible exception at all).
    """
    if exc_or_str is None:
        return "unknown"
    # A refused write during a live compression lease is contention, not
    # storage damage — but its message ("is being compressed by another
    # writer" / "Compression lease lost") contains neither "locked" nor
    # "busy", so it must be matched by type and by phrase (for strings that
    # survived RPC wrapping).
    if isinstance(exc_or_str, SessionTurnLeaseLostError):
        return "turn_lease"
    if isinstance(exc_or_str, CompressionSessionClosedError):
        return "compression_closed"
    if isinstance(exc_or_str, CompressionSessionBusyError):
        return "compression"
    if isinstance(exc_or_str, StateDbReplacedError):
        # Includes DeletedWalGenerationError (subclass).
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
    # Structural corruption BEFORE the lock and disk buckets: "database disk
    # image is malformed" contains "disk" (and some wrapped corruption
    # strings mention "locked" recovery attempts), so later buckets would
    # steal it and misdiagnose damage as space/contention.
    if any(marker in text for marker in _DB_CORRUPTION_MARKERS):
        return "corrupt"
    if (
        "locked" in text
        or "busy" in text
    ):
        return "locked"
    if (
        is_disk_full_error(exc_or_str)
        or "disk" in text
        or "readonly" in text
        or "read-only" in text
    ):
        return "disk"
    return "unknown"


# Cross-process serialisation for the schema-surgery paths below.  The
# ``_repair_attempt_lock`` above is a ``threading.Lock`` — it only covers
# threads inside ONE interpreter, yet a normal Hermes host runs several
# independent processes against the same ``state.db``: the gateway service,
# the Desktop app's own ``hermes serve`` backend, interactive CLI sessions,
# and the TUI slash worker.  Two of those hitting a malformed DB at once each
# ran the full ``writable_schema`` surgery + ``VACUUM`` on their own private
# connection, with nothing serialising them.
#
# The timeout is sized for the slowest legitimate holder — a ``VACUUM`` over a
# multi-GB DB in strategy 2.  Waiting that long is not a new stall: before this
# lock the losing caller spent the same minutes running its own surgery, it
# just did so on top of the winner's.
_REPAIR_LOCK_TIMEOUT_SECONDS = 120.0
_IS_WINDOWS = sys.platform == "win32"


# ── Repair-loop bounding + dead-backup hygiene (#86747) ─────────────────────
#
# ``_claim_repair_attempt`` above is an in-memory set: it bounds the loop
# only WITHIN one process. A corruption class the strategies cannot heal
# (b-tree page damage) failed repair on EVERY process start, and each pass
# took a fresh ~900MB forensic backup — 105 attempts / 89GB of identical
# dead copies in the reporting install. Two persistent bounds fix the class:
#
# * a sidecar attempt ledger (``<db>.repair-attempts.json``) that refuses
#   further surgery after ``_MAX_PERSISTENT_REPAIR_ATTEMPTS`` failures on
#   the SAME damaged file (fingerprint = size + a bounded content sample; any
#   successful repair or replacement changes it and resets the count);
# * backup dedupe + a retention cap in ``_backup_db_file`` — an identical
#   damaged file is never copied twice, and only the newest
#   ``_MAX_MALFORMED_BACKUPS`` forensic copies are kept.


# ── CJK-bigram FTS index (replaces the trigram index when available) ────
#
# The trigram tokenizer needs >=3 chars per query term, so 1-2 char CJK
# terms (ubiquitous in Korean/Chinese: 일본, 구글, 项目, ...) fall through
# to a LIKE full-table scan — measured 3-6s CPU per query on multi-GB
# installs and the dominant base cost of session_search on CJK workloads.
#
# ``cjk_unicode61`` (native/fts5_cjk/, a ~250-line loadable FTS5 tokenizer
# with no dependencies) wraps unicode61: maximal CJK runs are re-emitted as
# overlapping character bigrams (Lucene CJKAnalyzer semantics), everything
# else passes through unchanged. FTS5 phrase semantics turn a query term's
# consecutive bigrams into exact substring matching down to 2 chars at
# index speed. Contributed by Soju06 (PR #65544).
#
# Same v23 storage discipline as the trigram table it replaces:
# external-content over a tool-row-excluding view (zero inline text
# copies; tool rows stay searchable via ``messages_fts``), triggers gated
# on a DEDICATED marker pair (``fts_cjk_rebuild_high_water`` /
# ``fts_cjk_rebuild_progress``) so a cjk-only backfill — e.g. the
# trigram→cjk upgrade on an already-optimized DB — never gates the
# complete ``messages_fts`` index's triggers.
#
# The table exists ONLY when the loadable tokenizer is available
# (``~/.hermes/lib/libfts5_cjk.so``, built by ``native/fts5_cjk/build.sh``).
# A process that cannot load it self-heals by dropping the cjk triggers
# (message writes keep working; the index goes stale and is rebuilt by the
# next ``hermes sessions optimize-storage`` on a capable host).
#
# Split DDL: the table/view part is safe to ensure any time; the triggers
# are created ONLY while the index is complete-or-marker-gated. A stale
# index (trigger gap of unknown extent) must keep its triggers DROPPED —
# an external-content 'delete' op for a rowid the index never held is the
# canonical FTS5 index-corruption hazard the v23 marker gating exists to
# prevent.
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
    return os.getenv("HERMES_CJK_FTS", "1").strip().lower() not in (
        "0", "false", "off", "no",
    )


def load_fts5_cjk_extension(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer into ``conn``.

    Returns False (never raises) when the .so is absent, the feature is
    disabled via ``sessions.cjk_fts``, or this Python build has extension
    loading compiled out — every caller treats False as "behave exactly as
    before the cjk index existed".
    """
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
    """A concurrent writer collided with a *live* compression lock.

    Split out from :class:`CompressionSessionBusyError` because the two
    conditions that class covers need opposite handling. This one is
    transient: a healthy compressor holds the session for a few seconds and
    the lock row carries its own ``expires_at``, so the write can simply wait
    (see ``_execute_write``'s patience loop). The other case, a compressor
    discovering its own lease is gone, is permanent and must fail fast rather
    than spin out the whole patience budget.

    Subclassing keeps every existing ``except CompressionSessionBusyError``
    handler working unchanged.
    """


class SessionTurnLeaseLostError(RuntimeError):
    """A transcript write presented a turn-lease holder that no longer owns it.

    Fail-fast fencing: do not retry inside ``_execute_write``. The caller
    either still thinks it owns the conversation after expiry/reclaim, or
    the lease row is gone. A later writer may already be persisting a
    newer turn; landing this write would interleave a stale reply.
    """


class StateDbReplacedError(RuntimeError):
    """The state.db path no longer names the file this SessionDB opened.

    Raised when an out-of-band ``cp``/``mv``/restore replaces the database
    under a live gateway. In-place FTS repair and fail-open trigger
    dropping cannot fix a generation mismatch; they amplify it.
    """


class DeletedWalGenerationError(StateDbReplacedError):
    """A live process holds a deleted state.db-wal / -shm generation.

    Opening or writing through this handle would mint a second WAL inode
    (or keep committing on the orphan) — the split-brain that produces
    intermittent SQLITE_CORRUPT / SQLITE_IOERR. Stop the writers; do not
    unlink the WAL yourself. ``database.journal_mode: delete`` is operator
    containment, not a default change.

    Subclasses :class:`StateDbReplacedError` so every downstream consumer
    that already stops SQLite writes and diverts pending transcripts on a
    replaced store (gateway retry queue, run_agent flush) handles the split
    WAL generation identically — the correct response is the same: stop
    writing, preserve the transcript tail on disk.
    """


# SQLite header: 4-byte big-endian application_id at offset 68. Distinct from
# inode: ``cp`` onto the same path keeps st_ino and truncates+rewrites.
_STATE_DB_APPLICATION_ID_OFFSET = 68
_STATE_DB_GENERATION_KEY = "db_file_generation"
_STATE_DB_REPLACED_MSG = (
    "FATAL: state.db was replaced underneath the gateway; refusing further "
    "writes to this file. Divert transcripts to sessions/<id>.jsonl (and the "
    "gateway pending_messages spool) and restore or reopen after operator "
    "intervention."
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
    """A live SessionDB observed structural (non-FTS) corruption and is quarantined.

    Raised once a write on this handle reports bare ``SQLITE_CORRUPT`` /
    ``SQLITE_NOTADB`` that is neither FTS-scoped (``_is_fts_write_corruption_error``)
    nor a replaced-file case (``StateDbReplacedError``). Subclasses
    ``sqlite3.DatabaseError`` so every existing ``except sqlite3.Error``
    degrade path keeps working; ``sqlite_errorcode``/``sqlite_errorname``
    are copied from the originating error.

    The quarantine is sticky for the life of the handle: later writes fail
    fast, the handle never reopens after ``close()``, and ``close()`` skips
    its own WAL checkpoint. Field evidence (the #90837 lost/reordered-page
    signature, the #90950 page-1 clobber): a handle that kept writing for ~50
    minutes after the first structural error checkpointed 15 pages under the
    wrong page numbers on shutdown, turning a still-readable file into
    ``file is not a database``. Stopping the writes is what prevents that;
    skipping the explicit checkpoint is the second line of defence. SQLite
    still runs its own last-connection checkpoint inside ``close()`` (and
    deletes the ``-wal`` sidecar) unless ``SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE``
    is set — Python exposes it via ``Connection.setconfig()`` on 3.12+, so
    quarantine disables the close-time checkpoint there and the WAL survives
    on disk for forensics; on 3.11 the internal checkpoint is unavoidable
    (post-quarantine it can only carry pre-corruption committed frames, since
    no further writes are accepted). The
    recovery boundary is a process restart on a repaired or restored file.
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
    """Append pending messages as JSON lines under HERMES_HOME/sessions.

    Used when state.db is replaced under a live process so the current
    turn is not only in RAM. Returns the jsonl path, or None when there
    is nothing to write.
    """
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


# ── Process-wide shared SessionDB registry (#90837) ──
#
# The registry itself lives in hermes_state_registry.py — a bounded
# module owning acquisition, generation identity, refcounting,
# retirement, and teardown.  These re-exports keep the historical
# import path (``from hermes_state import get_shared_session_db``)
# working for every call site and test that imports from here.
#
# Routing rules (see hermes_state_registry for the full lifecycle):
#   - Long-lived in-process callers (gateway, tui_gateway, cron,
#     in-process tools) share ONE writer connection per resolved path
#     via get_shared_session_db().
#   - CLI one-shots, recovery flows, and read-only cross-profile opens
#     keep using SessionDB() directly with their own close().

from hermes_state_registry import (  # noqa: F401  (re-export)
    close_shared_session_dbs,
    get_shared_session_db,
    release_or_close,
    release_shared_session_db,
)


# ── Read-only health/stats probes (hermes doctor, dashboards) ──────────


# Lifecycle statuses surfaced by session pickers. Classification looks ONLY at
# a session's final message row — role, whether it carries tool_calls, and its
# finish_reason — so it stays O(1) per session (see
# SessionDB.session_lifecycle_statuses).
SESSION_STATUS_COMPLETE = "complete"
SESSION_STATUS_INTERRUPTED = "interrupted"
SESSION_STATUS_ERROR = "error"
SESSION_STATUS_EMPTY = "empty"

# finish_reason values that mark the turn as having ended in a provider or
# agent error (vs. a normal 'stop'/'length'/'tool_calls' completion).
_ERROR_FINISH_REASONS = frozenset({"error", "agent_error", "content_filter"})


def classify_session_status(
    role: Optional[str],
    has_tool_calls: bool,
    finish_reason: Optional[str],
) -> str:
    """Classify a session's lifecycle from the shape of its final message.

    - assistant with a normal finish → ``complete``
    - assistant that still has pending tool_calls (no tool result row ever
      followed, or it would be the last row instead) → ``interrupted``
    - user or tool as the last row → ``interrupted`` (the agent never got to
      answer / never consumed the tool result)
    - an error finish_reason on the last row → ``error``
    - anything unrecognized → ``complete`` (benign default; pickers must not
      alarm on unknown shapes)
    """
    if (finish_reason or "").strip().lower() in _ERROR_FINISH_REASONS:
        return SESSION_STATUS_ERROR
    r = (role or "").strip().lower()
    if r == "assistant":
        # The last row being an assistant message WITH tool_calls means the
        # matching tool result never landed — an interrupted tool turn.
        return SESSION_STATUS_INTERRUPTED if has_tool_calls else SESSION_STATUS_COMPLETE
    if r in {"user", "tool"}:
        return SESSION_STATUS_INTERRUPTED
    return SESSION_STATUS_COMPLETE


# Parent→child ``profile_name`` inheritance fence (#88381). ``agent:<ns>:...``
# gateway keys encode the profile namespace; a keyless row (CLI / subagent
# lineage) carries none and inherits freely. Two keyed rows must agree on
# ``agent:<ns>:`` — a default child (``agent:main:``) forked from a sibling
# profile's row must not be durably mislabelled as that profile's.
_SAME_KEY_NAMESPACE_SQL = (
    "p.session_key IS NULL OR sessions.session_key IS NULL"
    " OR substr(p.session_key, 1, instr(substr(p.session_key, 7), ':') + 6)"
    "  = substr(sessions.session_key, 1, instr(substr(sessions.session_key, 7), ':') + 6)"
)


class SessionDB(
    SessionSearchMixin,
    SessionSchemaMixin,
    SessionPortabilityMixin,
    SessionTelegramTopicsMixin,
    SessionCompressionMixin,
    SessionGatewayMixin,
    SessionMaintenanceMixin,
    SessionUsageMixin,
    SessionTitlesMixin,
    SessionMessagesMixin,
):
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    # Only these state-owned producers participate in automatic stale-open
    # reconciliation. Messaging-platform and UI/desktop sources have separate
    # lifecycle owners; unknown/future sources fail closed (#60609).
    _AUTO_PRUNE_STALE_OPEN_SOURCES: Tuple[str, ...] = (
        "cli",
        "cron",
        "kanban",
        "acp",
        "api_server",
        "subagent",
        "tool",
    )

    # ── Write-contention tuning ──
    # With multiple hermes processes (gateway + CLI sessions + worktree agents)
    # all sharing one state.db, WAL write-lock contention causes visible TUI
    # freezes.  SQLite's built-in busy handler uses a deterministic sleep
    # schedule that causes convoy effects under high concurrency.
    #
    # Instead, we keep the SQLite timeout short (1s) and handle retries at the
    # application level with random jitter, which naturally staggers competing
    # writers and avoids the convoy.
    #
    # Patience is TIME-based, not attempt-based.  A shared state.db is
    # legitimately held for multi-second stretches by sibling Hermes
    # processes: a TRUNCATE checkpoint at close on a large WAL, VACUUM after
    # an auto-prune, offline recovery, or an older still-running process
    # whose FTS maintenance predates the bounded-merge protocol (every
    # `hermes update` leaves mixed-version processes sharing the DB until
    # the old ones exit).  An attempt-counted budget (~15s incidental worst
    # case) silently loses that race and surfaces as
    # session_persistence_failed — a destroyed turn — even though the store
    # is healthy and merely busy (#74478).
    #
    # Two budgets: routine writes give up after _WRITE_PATIENCE_S so
    # background/UI callers don't stall excessively, while transcript
    # writes (append_message / session-row creation — the ones whose
    # failure aborts the user's turn) ride out anything shorter than
    # _TRANSCRIPT_WRITE_PATIENCE_S.  Jitter stays small for the first
    # _WRITE_RETRY_SLOW_AFTER_S (fast reclaim on millisecond contention),
    # then backs off so a long hold isn't hammered with BEGIN IMMEDIATE
    # attempts.
    _WRITE_PATIENCE_S = 20.0
    _TRANSCRIPT_WRITE_PATIENCE_S = 60.0
    # Observation-only activity heartbeat/label writes (#76354 review S1):
    # these run on (or adjacent to) the response-critical path and must never
    # wait out the full routine patience under contention. Sub-second budget;
    # a skipped write is retried naturally at the next heartbeat window.
    _ACTIVITY_WRITE_PATIENCE_S = 0.5
    # A live compression lock gets its own, much shorter budget than the write
    # lock. Compression publishes in a couple of seconds, so a brief wait saves
    # the overwhelming majority of concurrent turns (#75083). It deliberately
    # stays short: the lease is a correctness boundary, not just a busy signal
    # (see test_compression_lease_blocks_non_owner_but_allows_owner_flush), so
    # a writer that is still locked out after this budget must still be
    # refused rather than allowed to land a stale turn in a session whose
    # compression is genuinely long-running or wedged.
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    _WRITE_RETRY_SLOW_AFTER_S = 2.0
    _WRITE_RETRY_SLOW_MIN_S = 0.250  # 250ms
    _WRITE_RETRY_SLOW_MAX_S = 1.000  # 1s
    # Attempt a WAL checkpoint every N successful writes (PASSIVE mode).
    _CHECKPOINT_EVERY_N_WRITES = 50
    # Retain the existing coarse 1000-write maintenance cadence, but replace
    # the unbounded FTS5 ``'optimize'`` (measured holding the write lock for
    # 9-18 s per index on a 10 GB production DB — longer than a competing
    # writer's full retry patience, surfacing as "database is locked" /
    # session_persistence_failed) with bounded ``'merge'`` commands. A
    # positive merge rank is an approximate output-page budget, so each
    # command holds the write lock for milliseconds; up to
    # ``_FTS_MERGE_COMMANDS_PER_PASS`` commands run per index per cadence,
    # stopping early on the documented no-progress signal. ``usermerge`` is
    # lowered to 2 so positive merges act on any level with >= 2 segments —
    # without that, levels below the default threshold of 4 are skipped and
    # a fragmented index never converges (SQLite FTS5 §6.8-6.9).
    _FTS_MERGE_EVERY_N_WRITES = 1000
    _FTS_MERGE_MAX_PAGES_PER_INDEX = 500
    _FTS_MERGE_COMMANDS_PER_PASS = 4
    # Session imports intentionally use a lower cap than exports: import holds
    # one BEGIN IMMEDIATE transaction, so bounded batches avoid starving live
    # gateway/CLI writers. The dashboard accepts one exported JSON/JSONL file
    # at a time, so these still cover normal history restores.
    _IMPORT_MAX_SESSIONS = 500
    _IMPORT_MAX_MESSAGES_PER_SESSION = 10_000
    _IMPORT_MAX_TOTAL_MESSAGES = 50_000
    _IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
    _IMPORT_MAX_TOTAL_BYTES = 25 * 1024 * 1024
    # Demand-started accounting workers retire after an idle window so their
    # bound targets do not keep abandoned SessionDB instances (and SQLite
    # descriptors) alive forever. A later enqueue starts a fresh worker.
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
            "DELETE FROM system_prompts "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions "
            "WHERE sessions.system_prompt_hash = system_prompts.hash"
            ")"
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
        # Fail hard (before any connection/pragma/mkdir) if a pytest-context
        # process resolved the developer's production state.db — see the
        # live-DB test-isolation guard block near _default_db_path().
        _ensure_test_isolation(self.db_path)
        self.read_only = read_only

        self._lock = threading.Lock()
        # Read-path split (WAL only): recall/browse queries borrow a
        # read-only connection from a bounded pool so they never queue
        # behind writer flushes on self._lock. See _read_ctx().
        #
        # The pool is BOUNDED because the previous per-thread
        # (threading.local + strong set) scheme pinned one connection per
        # (SessionDB x thread) for the life of the process. Starlette
        # dispatches sync routes on anyio worker threads, so a SessionDB
        # that is never closed accumulated a connection — and two fds, the
        # database and its -wal — for every worker thread that ever read,
        # until the process hit the 256 soft RLIMIT_NOFILE a service manager
        # hands it and every request failed with EMFILE while the process
        # stayed alive, so the supervisor's restart-on-exit never fired.
        # Same bug class as the closing(...) fix in gateway/readiness.py
        # (#69678 / #69567).
        self._read_pool: "queue.LifoQueue[sqlite3.Connection]" = queue.LifoQueue(
            maxsize=_READ_POOL_MAX
        )
        # One permit per live read connection, held from before the open in
        # _get_read_conn() until after the close in _close_read_conn().  This
        # is what bounds PEAK descriptors; _read_pool alone bounds only the
        # idle set.  See _READ_POOL_MAX.  Acquired non-blocking on purpose: a
        # reader that cannot get a permit must degrade to the writer lock, not
        # queue here — blocking would convert fd exhaustion into a stall, which
        # is the same outage with a different stack trace.
        #
        # Permits are shared per DATABASE PATH, not per instance: the
        # descriptors they ration belong to the file, and one process holds
        # several SessionDB objects on the same state.db (#98573). See
        # _PathReadBudget.
        self._read_budget = _read_budget_for(self.db_path)
        self._read_budget.register(self)
        # Bound to the semaphore itself so every release site
        # (_close_read_conn and the _get_read_conn failure paths) is unchanged.
        self._read_permits = self._read_budget.permits
        # Count of reads that found no permit and fell back to the locked
        # writer connection. Not load-bearing; it is the only externally
        # visible signal that the ceiling is actually being reached, so a
        # too-small _READ_POOL_MAX is diagnosable from a running process
        # instead of inferred from latency.
        self._read_permit_exhausted = 0
        self._read_conns_lock = threading.Lock()
        # Set when close() begins.  _read_ctx checks this under the lock
        # before returning a connection to the pool, so a reader still in
        # flight during the drain closes its own connection instead of
        # re-populating a pool nobody will drain again.
        self._read_conns_closed = False
        # "read-only opens are failing against this file" backoff stamp.
        # Instance-wide rather than per-thread: with a shared pool the open
        # is no longer a per-thread event, and retrying a known-bad open on
        # every query is a syscall storm for no benefit. The locked writer
        # connection still serves reads while the backoff holds.
        # Deliberately a TIMESTAMP, not a sticky bool: the likeliest trigger
        # is transient fd pressure (EMFILE) — the very condition this pool
        # exists to prevent — and a permanent flag would demote every reader
        # on this instance to the writer lock for the life of the process.
        # The gateway shares one SessionDB across every agent, so that turns
        # a momentary blip into a permanent global convoy. Expires after
        # _READ_OPEN_RETRY_SECONDS so the read path self-heals.
        self._read_open_failed_at = 0.0
        self._wal_active = False
        self._write_count = 0
        # File identity of the state.db this instance opened. Compared on
        # every write (and before FTS fail-open / reopen-after-close) so an
        # out-of-band replace cannot limp through in-place surgery.
        # Inode catches mv/new-file; application_id catches cp onto the
        # same path (same inode, truncate+rewrite).
        self._db_file_identity: Optional[tuple] = None
        self._db_file_application_id: int = 0
        self._db_file_generation_token: str = ""
        self._db_replaced = False
        # Sticky: set once a write on THIS handle reports bare SQLITE_CORRUPT /
        # NOTADB that is not FTS-scoped and not a replaced-file case. Never
        # cleared; the recovery boundary is a process restart on a repaired or
        # restored file (see StateDbCorruptError).
        self._db_corrupt = False
        self._db_corrupt_reason = ""
        self._db_sidecar_identity: Dict[str, tuple] = {}
        self._db_wal_generation_lost = False
        # One-shot guard for the usermerge-floor config write on the
        # incremental FTS merge cadence (see _merge_fts_incrementally).
        self._fts_usermerge_floor_applied = False
        self._fts_enabled = False
        self._fts_stale = False
        self._trigram_available = False
        # CJK-bigram index (cjk_unicode61 loadable tokenizer). _fts_cjk_loaded:
        # extension present on the writer connection; _fts_cjk_available: the
        # messages_fts_cjk table is queryable AND not marked stale. Set during
        # _init_schema / _probe_fts_cjk.
        self._fts_cjk_loaded = False
        self._fts_cjk_available = False
        self._fts_unavailable_warned = False
        self._conn = None
        # Async token accounting (see queue_token_counts). The condition
        # guards queue + writer state; it is distinct from self._lock so
        # enqueue/flush bookkeeping never contends with SQLite writes.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = False
        self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None
        # Set True when this instance is opened via get_shared_session_db().
        # Makes close() a no-op so the registry (not individual callers)
        # controls the connection lifecycle (#90837).
        self._shared_registry_owned = False
        initialization_complete = False
        try:
            if read_only:
                self._open_read_only()
                self._record_db_file_identity()
                initialization_complete = True
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Read-only file/sidecar preflight (port of kilocode#12508):
            # repair-or-refuse BEFORE the first connection so users get an
            # actionable message instead of an opaque "attempt to write a
            # readonly database" from deep inside _init_schema.
            if not read_only:
                preflight_db_writability(self.db_path, db_label="state.db")

            # #68474 / #97568: Serialize startup across zero-byte check, quarantine,
            # connect, and schema commit so concurrent openers don't race on an
            # absent-path -> connect -> schema-commit window.
            needs_startup_guard = not read_only and (
                not self.db_path.exists() or is_zeroed_state_db(self.db_path)
            )

            try:
                self._open_with_optional_startup_guard(needs_startup_guard)
            except sqlite3.DatabaseError as exc:
                # The malformed-schema class (e.g. a duplicate sqlite_master
                # row for messages_fts) fails on the very first statement —
                # before _init_schema can run — so it can't be caught at the
                # FTS-rebuild layer. Recover by repairing sqlite_master in
                # place (backup first; canonical sessions/messages preserved),
                # then reopen once. This is what lets Desktop/Dashboard
                # self-heal instead of silently showing "no sessions".
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

            # NOTE: the v23 FTS optimization is OPT-IN (`hermes db optimize`),
            # never auto-started on open. Legacy installs keep their working
            # v22 inline FTS untouched here; only the explicit foreground
            # command demotes + rebuilds. This avoids a background worker
            # racing session lifecycle and the surprise disk/latency cost on
            # an unattended open. (An interrupted optimize resumes when the
            # user re-runs the command.)
            self._ensure_db_file_generation()
            self._record_db_file_identity()
            initialization_complete = True
        except Exception as exc:
            # Capture the cause so /resume and friends can surface WHY the
            # session DB is unavailable instead of a bare "Session database
            # not available."  Callers that catch this exception keep their
            # existing ``self._session_db = None`` degradation path.
            #
            # Note: we deliberately do NOT clear _last_init_error on the
            # success path (no else branch).  In multi-threaded callers
            # (gateway, web_server per-request SessionDB()), a concurrent
            # successful open racing past this failure would erase the
            # cause that another thread's /resume is about to format.
            # Tests that need to reset the state can call
            # ``hermes_state._set_last_init_error(None)`` explicitly.
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
                    f"file:{self.db_path}?mode=ro",
                    tracking_path=self.db_path,
                    uri=True,
                    check_same_thread=False,
                    timeout=1.0,
                    isolation_level=None,
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
        snaps = self.db_path.parent / "state-snapshots"
        msg = (
            f"state.db looks ZEROED ({zsize} bytes, no SQLite header). "
            f"Preserved at {qpath or '(quarantine failed — file left in place)'}. "
            f"Restore from {snaps} via `hermes snapshot list` / "
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
        """Open a fresh read-only connection, or None when unavailable.

        Callers must return the connection to self._read_pool (see
        _read_ctx); this opens, it does not track.

        Only used under WAL: WAL readers see a consistent snapshot and never
        block on (or get blocked by) the writer, so recall/browse queries can
        skip self._lock entirely. Under DELETE journal mode (NFS fallback) a
        reader can hit SQLITE_BUSY storms during writes, so we keep the
        legacy locked single-connection path there.

        Fresh read transactions begin per statement (autocommit), so each
        query observes everything committed so far — read-your-writes holds
        for the flush-then-search patterns in a turn.
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
        # Take the descriptor permit BEFORE the open, so concurrent openers
        # race for permits rather than for file descriptors. Non-blocking:
        # losing the race means "use the writer connection", not "wait".
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
        # Bound before the try: the except handlers close it if the open
        # half-succeeded, and an unbound name there would raise NameError over
        # the top of the real failure.
        conn = None
        try:
            conn = _connect_tracked_db(
                f"file:{self.db_path}?mode=ro",
                tracking_path=self.db_path,
                uri=True,
                # Pooled connections are borrowed by whichever thread runs
                # the next read, and sqlite3 otherwise refuses cross-thread
                # use ("SQLite objects created in a thread can only be used
                # in that same thread") — including on close(), which is how
                # the old per-thread connections became unclosable and leaked
                # their fds. Exclusive ownership is enforced by the pool
                # checkout/return, not by sqlite3. Matches the writer opens.
                check_same_thread=False,
                timeout=5.0,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            apply_database_pragmas(conn, db_label="state.db")
            # Load the CJK tokenizer extension on this connection so
            # messages_fts_cjk queries work on the read path. The .so
            # registers the tokenizer in the connection's in-memory
            # registry, not the database file, so mode=ro is fine.
            if self._fts_cjk_loaded:
                load_fts5_cjk_extension(conn)
        except sqlite3.Error:
            # A partially-constructed connection — _connect_tracked_db
            # succeeded, the CJK extension load did not — must be closed here.
            # Dropping it on the floor still open leaves a live descriptor the
            # tracking registry still counts: the same leak shape this pool
            # exists to fix, one level further down.
            self._discard_partial_read_conn(conn)
            # Back off from retrying the open on every query; the locked
            # writer connection still serves reads until the stamp expires.
            with self._read_conns_lock:
                self._read_open_failed_at = time.monotonic()
            logger.debug("read-only connection open failed for %s", self.db_path, exc_info=True)
            self._read_budget.release()
            return None
        except BaseException:
            # Anything else (a non-sqlite3 extension-load failure, MemoryError,
            # KeyboardInterrupt landing between open and return) must not
            # strand the permit: a stranded permit is not a transient error, it
            # permanently shrinks the read path by one slot for the life of the
            # process.
            self._discard_partial_read_conn(conn)
            self._read_budget.release()
            raise
        return conn

    def _evict_one_idle_read_conn(self) -> bool:
        """Close one connection sitting idle in this instance's pool.

        Called by _PathReadBudget when a peer SessionDB on the same file wants
        a permit this instance is holding but not using. Only the idle set is
        reachable from here — a checked-out connection is not in the queue —
        so this can never pull a connection out from under a live reader.

        Returns whether a connection (and therefore a permit) was released.
        """
        try:
            conn = self._read_pool.get_nowait()
        except queue.Empty:
            return False
        self._close_read_conn(conn)
        return True

    def _discard_partial_read_conn(self, conn) -> None:
        """Close a connection that failed between open and hand-off.

        Separate from _close_read_conn because that one releases a permit and
        this runs on paths that release their own.
        """
        if conn is None:
            return
        try:
            conn.close()
        except Exception as exc:
            logger.warning(
                "partially-opened read conn close failed for %s: %s", self.db_path, exc
            )

    def _close_read_conn(self, conn) -> None:
        """Close a pooled read connection and release its descriptor permit.

        This was a bare ``except Exception: pass``, which silently swallowed
        the sqlite3.ProgrammingError raised when close() ran on a thread
        other than the one that opened the connection — the exact signature
        of the fd leak this pool fixes. A close that fails leaks a tracked
        fd, so it must not be invisible.

        The permit is released even when close() raises: the descriptor is
        already lost at that point, and withholding the permit too would turn
        one leaked fd into a permanently narrower read path — failing twice for
        one fault. The warning is the signal that matters.

        Pairs with _get_read_conn(). Calling this on a connection that did not
        come from there over-releases the BoundedSemaphore, which raises
        ValueError rather than silently widening the ceiling.
        """
        try:
            conn.close()
        except Exception as exc:
            logger.warning("read-conn close failed for %s: %s", self.db_path, exc)
        finally:
            self._read_budget.release()

    def _checkout_read_conn(self) -> Optional[sqlite3.Connection]:
        """Borrow a read connection from the pool, opening one on a miss.

        The single acquisition seam for the read path: the WAL/read_only gate,
        the pool checkout and the open-on-miss all live here, so there is
        exactly one place to exercise (and one place for a caller to bypass by
        accident). Returns None when the read path is unavailable and the
        caller must fall back to the locked writer connection.

        A pool hit costs no permit — the connection it hands back is already
        holding one. Only the miss path can open, and only _get_read_conn() can
        take a permit, so peak live connections is bounded by _READ_POOL_MAX no
        matter how many threads miss simultaneously.
        """
        if not self._wal_active or self.read_only:
            return None
        try:
            return self._read_pool.get_nowait()
        except queue.Empty:
            return self._get_read_conn()

    @contextmanager
    def _read_ctx(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for read-only statements.

        WAL: a read-only connection borrowed from a bounded pool with NO
        lock — recall queries never convoy behind writer flushes (the
        gateway shares one SessionDB across every agent, so this lock was a
        global choke point). The connection is checked out for the duration
        of the block, so no two threads ever touch it concurrently.
        Non-WAL, read-conn failure, or _READ_POOL_MAX already reached: the
        shared writer connection under self._lock, byte-for-byte the legacy
        behavior.

        That last case is the deliberate degradation. Past the ceiling readers
        convoy on the writer lock instead of opening descriptors — measurably
        slower under a burst, and the alternative is EMFILE, which takes the
        whole process down in a way a restart-on-exit supervisor cannot see.
        """
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
                    # close() has already drained the pool, so this connection
                    # is surplus. Close it here — dropping it on the floor is
                    # what leaked the fd.
                    #
                    # queue.Full is now unreachable in practice (permits and
                    # maxsize are both _READ_POOL_MAX, so there can never be a
                    # ninth connection to return), but the branch stays: it is
                    # load-bearing if those two ever drift apart, and a leak is
                    # the failure mode it prevents.
                    self._close_read_conn(conn)
            return
        with self._lock:
            if self._conn is None:
                # close() ran while a reader was still unwinding (#94736
                # class) — reopen instead of yielding None to a .execute.
                self._reopen_after_close_locked(context="read")
            yield cast(sqlite3.Connection, self._conn)

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        """Reopen the writer connection after ``close()`` raced a live caller.

        The #94736 failure shape: a teardown owner (cron ``run_job``'s
        ``finally``, a delegate timeout owner abandoning its worker, agent
        ``close()``) calls ``SessionDB.close()`` — which sets ``_conn = None``
        — while a still-unwinding worker thread has one more transcript flush
        to land. The next ``_execute_write`` then died on
        ``'NoneType' object has no attribute 'execute'``, the conversation
        loop force-ended the turn as ``session_persistence_failed``, and the
        in-flight tail of the subagent/cron session was silently dropped
        (delivered as ``last_status: ok``).

        The transcript append is THE critical write — losing it destroys the
        turn — so at this shared persistence boundary we self-heal: reopen a
        connection to the same database file and let the write land. The
        reopen is loud (WARNING names the race) and bounded (only fires when
        ``_conn`` is ``None``, i.e. after an explicit ``close()`` — the
        constructor never leaves a live instance with a ``None`` handle).
        ``__del__`` delegates to ``close()``, so the reopened connection is
        still released at GC/exit time.

        Caller must hold ``self._lock``. Raises ``sqlite3.OperationalError``
        with an explicit cause when the reopen itself fails, so the surfaced
        persistence error names the teardown race instead of the opaque
        ``NoneType`` attribute error.
        """
        if self.read_only:
            raise sqlite3.ProgrammingError(
                f"SessionDB for {self.db_path} was closed (read-only handle); "
                f"cannot serve a {context} after close()"
            )
        # A reopen resolves the PATH again — if the file at that path is no
        # longer the one this instance originally opened (out-of-band
        # restore/cp/mv), reconnecting would write into the new generation
        # through stale WAL/shm assumptions (#89332). Refuse instead.
        if self._db_replaced or self._db_file_was_replaced():
            self._halt_db_replaced()
        # A quarantined handle must never come back: reopening would hand a
        # fresh connection (and its own close-time checkpoint) to a file we
        # already know is structurally damaged.
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
                str(self.db_path),
                check_same_thread=False,
                timeout=1.0,
                isolation_level=None,
            )
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen "
                f"failed: {exc}"
            ) from exc
        try:
            conn.row_factory = sqlite3.Row
            self._wal_active = (
                apply_wal_with_fallback(conn, db_label="state.db") == "wal"
            )
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn)
        except Exception as exc:
            self._close_connection_quietly(conn)
            raise sqlite3.OperationalError(
                f"state.db reopen after close() succeeded but connection "
                f"setup failed: {exc}"
            ) from exc
        # Schema was initialised by this instance's original open; the file
        # cannot have lost it, so no _init_schema here (no DDL races with
        # sibling processes during teardown).
        self._conn = conn

    # ── Core write helper ──

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        err = str(exc).lower()
        if "no such module" in err and "fts5" in err:
            return True
        # SQLite builds that have FTS5 but lack the optional trigram tokenizer
        # raise "no such tokenizer: trigram" instead of "no such module".
        # Scope to trigram specifically to avoid masking unrelated tokenizer errors.
        if "no such tokenizer: trigram" in err:
            return True
        # The cjk_unicode61 tokenizer is a loadable extension — a process
        # that couldn't load it sees the same capability-error shape.
        if "no such tokenizer: cjk_unicode61" in err:
            return True
        return False

    @staticmethod
    def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """True when only an optional tokenizer is missing (FTS5 itself works).

        Covers the built-in trigram tokenizer (needs SQLite >= 3.34) and the
        loadable cjk_unicode61 tokenizer — both mean "this one index can't be
        served here", never "disable FTS".
        """
        err = str(exc).lower()
        return (
            "no such tokenizer: trigram" in err
            or "no such tokenizer: cjk_unicode61" in err
        )

    @staticmethod
    def _db_has_legacy_inline_fts(cursor: sqlite3.Cursor) -> bool:
        """True when messages_fts exists in ANY pre-v23 shape.

        v23's messages_fts is external-content over THREE real columns
        (content, tool_name, tool_calls). Every pre-v23 shape lacks the
        tool_name/tool_calls columns — whether the old inline single-column
        form (v11..v22) or the even older external-content single-column form
        (v10-era, pre-#16751). We therefore detect "needs optimize" as "the
        stored CREATE lacks the tool_name column", which is the precise v23
        marker and correctly catches BOTH legacy variants.

        Returns False when messages_fts doesn't exist yet (fresh DB mid-init):
        the post-migration FTS setup block will create it in the v23 shape.
        """
        row = cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        if row is None:
            return False
        sql = row[0] or ""
        # The v23 table declares tool_name/tool_calls columns. Their absence
        # means a legacy shape that doesn't index tool metadata → optimize.
        return "tool_name" not in sql

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
            "current Python (managed uv guarantees FTS5). "
            "(underlying error: %s)",
            self.db_path,
            exc,
        )

    def _ensure_fts_cjk_schema(self, cursor) -> None:
        """Create / repair / self-heal the CJK-bigram index surface.

        ``cursor`` may be a Cursor or a Connection (both expose execute /
        executescript). Called only for v23-shape DBs with the base FTS
        surface healthy. Sets ``self._fts_cjk_available``. Never raises;
        every failure mode degrades to "no cjk index" (trigram/LIKE routing
        keeps working).

        Cases:
          tokenizer loaded, table absent  → create. Empty DB: index is
              complete by construction (triggers cover everything). Populated
              DB: set the cjk backfill markers so the id-gated triggers stay
              correct and `optimize-storage` can backfill; the index is NOT
              served until the backfill completes.
          tokenizer loaded, table present → ensure triggers (recreates any
              dropped by a tokenizer-less process), honour the stale
              breadcrumb (serve only when absent and no backfill pending).
          tokenizer NOT loaded, table present with live triggers → drop the
              cjk triggers so message INSERTs don't fail at trigger time,
              and leave the stale breadcrumb (#self-heal). The table itself
              stays for a later capable open to rebuild.
        """
        try:
            cjk_present = bool(cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'messages_fts_cjk'"
            ).fetchone())

            if not self._fts_cjk_loaded:
                if cjk_present:
                    live = [
                        r[0] for r in cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                            f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)})",
                            _FTS_CJK_TRIGGERS,
                        ).fetchall()
                    ]
                    if live:
                        # Self-heal: this process cannot tokenize, so every
                        # message INSERT would die inside the cjk trigger.
                        # Breadcrumb FIRST (crash between the two statements is
                        # merely conservative), then drop.
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
            # Mirror the tokenizer-loaded except below: the presence check
            # and self-heal above run before the loaded/not-loaded branch is
            # even known to be safe, so they need the same never-raises
            # guarantee the docstring promises for the rest of the method.
            logger.warning(
                "messages_fts_cjk presence check failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
            self._fts_cjk_available = False
            return

        try:
            cursor.executescript(FTS_CJK_TABLE_SQL)
            if not cjk_present:
                # Freshly created. An empty DB's index is complete by
                # construction (triggers will cover every future row); a
                # populated DB (e.g. a v23 install predating the cjk index)
                # gets the dedicated marker pair so the id-gated triggers
                # keep NEW rows indexed while old rows await the
                # `optimize-storage` backfill. Either way any old stale
                # breadcrumb refers to a table that no longer exists.
                cursor.execute(
                    "DELETE FROM state_meta WHERE key = ?",
                    (FTS_CJK_STALE_KEY,),
                )
                n_msgs = cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
                ).fetchone()[0]
                if n_msgs > 0:
                    hw = cursor.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM messages"
                    ).fetchone()[0]
                    for k, v in (
                        ("fts_cjk_rebuild_high_water", str(hw)),
                        ("fts_cjk_rebuild_progress", "0"),
                    ):
                        cursor.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (k, v),
                        )
            stale = cursor.execute(
                "SELECT 1 FROM state_meta WHERE key = ?",
                (FTS_CJK_STALE_KEY,),
            ).fetchone()
            if stale:
                # A tokenizer-less process dropped the triggers at some
                # unknown point — the index has a gap of unknown extent.
                # Do NOT reinstall triggers (an external-content 'delete'
                # for an unindexed rowid corrupts the index); the next
                # `optimize-storage` run rebuilds from scratch.
                self._fts_cjk_available = False
                return
            cursor.executescript(FTS_CJK_TRIGGER_SQL)
            backfill_pending = cursor.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
            ).fetchone()
            self._fts_cjk_available = not backfill_pending
        except sqlite3.OperationalError:
            # Includes "no such tokenizer: cjk_unicode61" if the extension
            # loaded but registration failed — degrade to trigram/LIKE.
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

    def _ensure_fts_schema(
        self,
        cursor: sqlite3.Cursor,
        table_name: str,
        ddl: str,
    ) -> bool:
        status = self._fts_table_probe(cursor, table_name)
        if status is None:
            return False
        try:
            # Run even when the virtual table exists so any dropped or missing
            # triggers are recreated after a previous no-FTS5 runtime disabled
            # them to keep message writes working.
            cursor.executescript(ddl)
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            # Only disable FTS entirely when the whole FTS5 module is missing.
            # A missing specific tokenizer (e.g. trigram) means only that
            # particular table cannot be created — the base FTS5 table is fine.
            if self._is_trigram_unavailable_error(exc):
                self._warn_trigram_unavailable(exc)
            else:
                self._warn_fts5_unavailable(exc)
            return False

    def _execute_write(
        self,
        fn: Callable[[sqlite3.Connection], T],
        patience_s: Optional[float] = None,
    ) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random jitter, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        *patience_s* is the total time budget for lock retries (default
        ``_WRITE_PATIENCE_S``).  Transcript-critical writes pass
        ``_TRANSCRIPT_WRITE_PATIENCE_S`` so a sibling process holding the
        lock for a legitimate long operation (VACUUM, TRUNCATE checkpoint,
        pre-bounded-merge FTS optimize from an older still-running
        install) exhausts routine writers' patience without destroying a
        user turn.  Jitter starts small (20-150ms) for fast reclaim on
        millisecond contention and backs off to 250ms-1s once the lock has
        been held longer than ``_WRITE_RETRY_SLOW_AFTER_S``.

        Returns whatever *fn* returns.
        """
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        # Set on the first compression-busy collision so the short wait is
        # measured from then, not from the start of the write.
        compression_deadline: Optional[float] = None
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself. The
        # callback has not run at that point, so there is no durable effect
        # to replay and the retry is exactly-once safe (#99502's contract).
        # Once the callback starts, an IOERR leaves the write's settlement
        # unknown and must propagate — this helper owns non-idempotent
        # transcript/counter mutations, not just idempotent UPSERTs.
        ioerr_begin_retried = False

        # Transient engine-level error observed on contended WAL appends
        # (dual gateway/agent writers; FTS5 trigram sync holds the write
        # lock). The identical write succeeds standalone, so it is
        # retryable like locked/busy. The exception CLASS varies with the
        # SQLite build — some surface it as InterfaceError, which lives
        # OUTSIDE DatabaseError and escaped the retry net entirely on
        # attempt 0 — so the check is message-scoped, not class-scoped.
        def _is_no_more_rows(exc: sqlite3.Error) -> bool:
            return "no more rows available" in str(exc).lower()

        while True:
            self._raise_if_db_corrupt()
            self._raise_if_db_replaced()
            fn_started = False
            try:
                with self._lock:
                    if self._conn is None:
                        # close() ran while this writer was still unwinding
                        # (#94736) — reopen instead of dying on None.execute.
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
                # A live foreign compression lock is transient: the compressor
                # publishes in a couple of seconds. Without any wait, a steer
                # that lands mid-compression aborts the user's turn as
                # session_persistence_failed and sends the operator hunting
                # disk space that was never the problem (#75083).
                #
                # The budget is _COMPRESSION_BUSY_WAIT_S, not the write-lock
                # patience: the lease is a correctness boundary, so a writer
                # still locked out after a short wait must be refused rather
                # than left to land a stale turn once a long-running or wedged
                # compression finally lets go.
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
                    # Patience exhausted — say what actually happened so the
                    # surfaced error doesn't read as disk/permission damage.
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
                    # BEGIN IMMEDIATE itself hit a transient WAL-transition
                    # IOERR. Nothing has been mutated, so retrying on the SAME
                    # connection replays nothing. Never close()+reopen to
                    # "heal" it: close() cancels this process's POSIX locks on
                    # the file for every sibling connection (howtocorrupt §2.2).
                    ioerr_begin_retried = True
                    continue
                # Non-lock error, the callback already ran (settlement is
                # unknown — do not replay), or patience exhausted.
                raise
            except sqlite3.DatabaseError as exc:
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                # An out-of-band replace of state.db (restore/cp/mv under a
                # live process) surfaces as this same corruption error class.
                # In-file repair on a NEW file generation amplifies the
                # damage (#89332) — halt writes on this handle instead.
                if (
                    "not a database" in str(exc).lower()
                    or is_malformed_db_error(exc)
                    or self._is_fts_write_corruption_error(exc)
                ):
                    self._raise_if_db_replaced()
                # Corrupt FTS shadow tables make every write raise the
                # malformed/corrupt error class through the FTS sync triggers
                # while the canonical messages table is intact. Never run a
                # full-message FTS5 rebuild from this live persistence path:
                # on a multi-gigabyte state.db that can hold the writer lock
                # for minutes. Atomically detach the derived indexes instead,
                # then retry the canonical write. The existing stale-open and
                # explicit repair paths retain rebuild ownership.
                if self._enter_fts_fail_open(exc):
                    continue
                # Bare SQLITE_CORRUPT / NOTADB that survived the replaced-file
                # check and the FTS-scoped fail-open is structural damage:
                # quarantine the handle (see StateDbCorruptError).
                if self._is_structural_corruption_error(exc):
                    self._halt_db_corrupt(exc)
                raise
            except sqlite3.Error as exc:
                # Catch-all for builds that surface 'no more rows available'
                # as InterfaceError (a sibling of DatabaseError, not a
                # subclass) or another sqlite3.Error class outside the two
                # handlers above. Message-scoped: anything else propagates
                # untouched.
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                raise

    def _write_sql(
        self,
        sql: str,
        params: Any = (),
        *,
        many: bool = False,
        patience_s: Optional[float] = None,
    ) -> None:
        """Run one INSERT/UPDATE/DELETE through ``_execute_write``."""
        def _do(conn):
            (conn.executemany if many else conn.execute)(sql, params)

        self._execute_write(_do, patience_s=patience_s)

    def _write_rowcount(
        self, sql: str, params: Any = (), *, patience_s: Optional[float] = None
    ) -> int:
        """Run one UPDATE/DELETE through ``_execute_write``; return rows changed.

        Falls back to ``SELECT changes()`` when the driver reports an unknown
        rowcount (None / negative).
        """
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

        First opener wins via INSERT OR IGNORE. application_id is written
        only when still 0 so racers converge on the same header value.
        PASSIVE checkpoint only — never TRUNCATE (#45383).
        """
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
                    "SELECT value FROM state_meta WHERE key = ?",
                    (_STATE_DB_GENERATION_KEY,),
                ).fetchone()
                if row and row[0]:
                    token = str(row[0])
                self._db_file_generation_token = token
                current = 0
                pragma_row = self._conn.execute("PRAGMA application_id").fetchone()
                if pragma_row:
                    current = int(pragma_row[0] or 0)
                if current == 0:
                    app_id = int(token[:8], 16) & 0x7FFFFFFF
                    if app_id == 0:
                        app_id = 1
                    self._conn.execute(f"PRAGMA application_id={app_id}")
                    current = app_id
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
                if pragma_row and pragma_row[0]:
                    self._db_file_application_id = int(pragma_row[0])
            except sqlite3.Error:
                pass

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
            # Header 0 means the WAL has not been checkpointed yet — not a
            # replace. A copied Hermes DB that minted its own id is nonzero.
            if disk_app and disk_app != recorded_app:
                return True
        return False

    def _halt_db_replaced(self) -> None:
        """Stop writes and raise; do not run in-file repair on a new generation."""
        self._db_replaced = True
        logger.error(_STATE_DB_REPLACED_MSG)
        raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this instance opened is gone.

        Steady state (a sidecar generation is recorded): pure stat — a
        recorded inode that is missing or replaced by a new file at the same
        path means the generation split. No /proc walk on healthy writes.

        Empty-identity state (fresh DB whose WAL appears only after open, or
        identity cleared by a clean ``close()``): fall back to a
        ``/proc/self/fd`` deleted-fd probe, and adopt the current sidecars as
        this handle's generation once the probe comes back clean. The full
        ``/proc/*/fd`` walk is reserved for
        :func:`refuse_deleted_wal_generation` on open, where we must see
        *foreign* deleted holders before ``sqlite3.connect`` mints a new WAL.
        """
        recorded = self._db_sidecar_identity or {}
        base = os.fspath(self.db_path)
        if recorded:
            for suffix, recorded_ident in recorded.items():
                current = _stat_db_file_identity(Path(base + suffix))
                if current is None or current != recorded_ident:
                    return True
            return False
        if not self._wal_active:
            # No WAL on this handle (journal_mode=delete/truncate fallback):
            # there is no sidecar generation to lose, and probing every write
            # would put a /proc walk on the hot path of exactly the
            # delete-mode deployments the field report used as containment.
            return False
        if sys.platform.startswith("linux"):
            watched = _watched_sqlite_sidecar_paths(self.db_path)
            fd_dir = f"/proc/{os.getpid()}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        target = os.readlink(f"{fd_dir}/{fd}")
                    except OSError:
                        continue
                    if " (deleted)" in target and _canonical_sqlite_path(target) in watched:
                        return True
            except OSError:
                return False
        # Probe clean (or unavailable on this platform): adopt whatever
        # sidecar generation exists now so subsequent writes use the cheap
        # stat check.
        current_identity = _stat_sqlite_sidecar_identity(self.db_path)
        if current_identity:
            self._db_sidecar_identity = current_identity
        return False

    def _halt_deleted_wal_generation(self) -> None:
        """Stop writes; do not mint or keep committing on a split WAL."""
        self._db_wal_generation_lost = True
        logger.error(_DELETED_WAL_GENERATION_MSG)
        raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)

    def _raise_if_db_replaced(self) -> None:
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost:
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)
        if self._db_file_was_replaced():
            self._halt_db_replaced()
        if self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()

    @classmethod
    def _is_structural_corruption_error(cls, exc: BaseException) -> bool:
        """Bare SQLITE_CORRUPT/NOTADB with no FTS provenance.

        ``_is_fts_write_corruption_error`` is the positive FTS classifier;
        everything else in the ``corrupt`` bucket of
        ``classify_persistence_error`` is damage to a canonical B-tree, the
        schema, or the freelist — never repairable from the live write path.
        """
        if not isinstance(exc, sqlite3.DatabaseError):
            return False
        if isinstance(exc, StateDbCorruptError):
            return False
        if cls._is_fts_write_corruption_error(exc):
            return False
        return classify_persistence_error(exc) == "corrupt"

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
            "gateway and run `hermes sessions recover --source %s "
            "--inspect-only`.",
            self.db_path,
            exc,
            self.db_path,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            value = getattr(exc, attr, None)
            if value is not None:
                setattr(err, attr, value)
        raise err from exc

    def _disable_close_time_checkpoint(self) -> None:
        """Best-effort: stop SQLite's own last-connection checkpoint on close.

        Skipping our explicit ``PRAGMA wal_checkpoint(PASSIVE)`` in
        ``close()`` is not enough on its own: ``sqlite3.Connection.close()``
        still runs SQLite's internal last-connection PASSIVE checkpoint and
        unlinks the ``-wal``/``-shm`` sidecars. On the field incident's file
        that close-time checkpoint is exactly what wrote 15 pages under the
        wrong page numbers. Python 3.12+ exposes the switch as
        ``Connection.setconfig(SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE)``; on 3.11
        neither the constant nor ``setconfig`` exists, so the internal
        checkpoint remains (it can only carry pre-quarantine committed
        frames — no further writes are accepted on this handle).
        """
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        if flag is None:
            return
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if conn is None or setconfig is None:
            return
        try:
            setconfig(flag, True)
        except Exception:
            logger.debug(
                "Could not disable SQLite's close-time checkpoint on the "
                "quarantined handle for %s",
                self.db_path,
                exc_info=True,
            )

    def _raise_if_db_corrupt(self) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()

    def _sleep_before_write_retry(
        self, deadline: float, patience_s: float
    ) -> bool:
        """Sleep one jitter interval if the patience budget still allows it.

        Returns True when the caller should retry, False when *deadline* has
        passed and the error should propagate. Jitter stays small for the
        first ``_WRITE_RETRY_SLOW_AFTER_S`` (fast reclaim on millisecond
        contention) and backs off after that, and never overshoots the
        deadline by a full slow-jitter.
        """
        now = time.monotonic()
        if now >= deadline:
            return False
        elapsed = now - (deadline - patience_s)
        if elapsed >= self._WRITE_RETRY_SLOW_AFTER_S:
            jitter = random.uniform(
                self._WRITE_RETRY_SLOW_MIN_S,
                self._WRITE_RETRY_SLOW_MAX_S,
            )
        else:
            jitter = random.uniform(
                self._WRITE_RETRY_MIN_S,
                self._WRITE_RETRY_MAX_S,
            )
        time.sleep(min(jitter, max(deadline - now, 0.001)))
        return True

    @staticmethod
    def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        """Return true only when SQLite identifies corruption as FTS-scoped.

        Newer SQLite builds include ``fts5`` in the error text.  Older builds
        may emit only ``database disk image is malformed`` while exposing the
        extended ``SQLITE_CORRUPT_VTAB`` result code.  A bare
        ``SQLITE_CORRUPT``/malformed-image error is structural and must not
        trigger live FTS maintenance: it does not prove that canonical B-trees
        are intact.
        """
        corrupt_vtab = getattr(sqlite3, "SQLITE_CORRUPT_VTAB", 267)
        error_code = getattr(exc, "sqlite_errorcode", None)
        if error_code is not None:
            return error_code == corrupt_vtab
        msg = str(exc).lower()
        return msg.startswith("fts5:") and "corrupt structure" in msg

    def _foreign_state_db_holders(self) -> List[Tuple[int, str]]:
        """Return foreign processes holding this DB or its WAL sidecars.

        Automatic FTS repair is structural maintenance, not an ordinary WAL
        write.  It must not run while another process remains attached: a
        sidecar reset under that holder can leave the two processes writing
        through different WAL inodes.

        A scan failure is represented as an unknown holder.  Skipping optional
        automatic maintenance is safer than assuming quiescence; canonical
        writes continue through the stale-FTS fail-open path.
        """
        # The split-brain mechanism requires POSIX unlink semantics: Windows
        # refuses to replace SQLite sidecars while another process has them
        # open.  Avoid psutil.open_files() there; querying arbitrary Windows
        # processes can block for minutes on device-backed handles.
        if _IS_WINDOWS:
            return []
        if psutil is None:
            return [(-1, "open-file scan unavailable")]

        db_path = os.path.abspath(os.fspath(self.db_path))
        watched = {
            _canonical_sqlite_path(db_path),
            _canonical_sqlite_path(db_path + "-wal"),
            _canonical_sqlite_path(db_path + "-shm"),
        }
        holders: List[Tuple[int, str]] = []

        # On Linux, read /proc/<pid>/fd symlinks directly.  psutil's
        # open_files() filters through isfile_strict(), which stats the
        # literal path — for an unlinked WAL sidecar the kernel returns
        # "/path/state.db-wal (deleted)" and stat fails, so the entry is
        # silently dropped and the split-brain holder is never seen.
        # /proc readlinks preserve the "(deleted)" suffix so _canonical can
        # strip it and match.
        if sys.platform.startswith("linux"):
            try:
                own_pid = os.getpid()
                for pid_str in os.listdir("/proc"):
                    if not pid_str.isdigit():
                        continue
                    pid = int(pid_str)
                    if pid == own_pid:
                        continue
                    fd_dir = f"/proc/{pid}/fd"
                    try:
                        fds = os.listdir(fd_dir)
                    except OSError:
                        # Cannot read this process's fd table (different
                        # user, e.g. root gateway vs user desktop).
                        # /proc/<pid>/cmdline is world-readable by default,
                        # so check whether this is a Hermes process —
                        # only flag uninspectable holders that look like
                        # another Hermes instance, not every system daemon.
                        cmdline = _read_proc_cmdline(pid)
                        if cmdline is not None and _looks_like_hermes(cmdline):
                            holders.append((pid, f"uninspectable holder: {cmdline[:80]}"))
                        continue
                    for fd in fds:
                        try:
                            target = os.readlink(f"{fd_dir}/{fd}")
                        except OSError:
                            continue
                        if _canonical_sqlite_path(target) in watched:
                            holders.append((pid, target))
            except Exception as exc:
                logger.warning(
                    "Could not prove state.db has no foreign holders; "
                    "deferring automatic FTS maintenance: %s",
                    exc,
                )
                return holders or [(-1, f"open-file scan failed: {exc}")]
            return holders

        # macOS / BSD: use psutil.open_files().  macOS does not use the
        # "(deleted)" suffix convention, so psutil's filtering is safe here.
        try:
            for process in psutil.process_iter(["pid", "open_files"]):
                info = process.info
                pid = int(info["pid"])
                if pid == os.getpid():
                    continue
                # psutil's as_dict() converts AccessDenied to None, which
                # or-() turns into an empty iteration.  On macOS this is
                # acceptable: the gateway/desktop topology from the issue is
                # Linux-specific (systemd units running as root).
                for opened in info.get("open_files") or ():
                    path = getattr(opened, "path", "")
                    if path and _canonical_sqlite_path(path) in watched:
                        holders.append((pid, path))
        except Exception as exc:
            logger.warning(
                "Could not prove state.db has no foreign holders; "
                "deferring automatic FTS maintenance: %s",
                exc,
            )
            return holders or [(-1, f"open-file scan failed: {exc}")]
        return holders


    def _enter_fts_fail_open(self, exc: sqlite3.DatabaseError) -> bool:
        """Detach corrupt FTS indexes so canonical writes can continue.

        The stale breadcrumb and trigger removal commit atomically. Its
        ordering is load-bearing: after triggers are absent, new canonical
        rows create an index gap of unknown extent, so another process must
        never reinstall the triggers without first rebuilding every row.
        """
        if not self._fts_enabled or not self._is_fts_write_corruption_error(exc):
            return False
        self._raise_if_db_corrupt()
        if self._db_replaced or self._db_file_was_replaced():
            self._halt_db_replaced()
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()

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
                "Could not detach corrupt FTS indexes; canonical write still "
                "cannot proceed: %s",
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
        """Best-effort PASSIVE WAL checkpoint.  Never raises.

        Flushes committed WAL frames back into the main DB file without
        requiring an exclusive lock.  PASSIVE is safe for frequent
        periodic use because it does not block concurrent writers and
        cannot corrupt B-tree pages under I/O pressure.

        PASSIVE does not truncate the WAL file — it stays at its
        high-water mark. Explicit checkpoints on the shared ``state.db`` no
        longer truncate the WAL; it is bounded by ``journal_size_limit`` and
        the writer's natural post-checkpoint reset rather than by a TRUNCATE
        at every close or maintenance command.

        Previous TRUNCATE strategy caused B-tree corruption on large
        databases (65K+ pages) due to the exclusive-lock I/O pressure
        from checkpointing thousands of frames at once (issue #45383).
        """
        if self._db_corrupt:
            return  # quarantined: never checkpoint over a damaged image
        try:
            with self._lock:
                result = self._conn.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2], result[1],
                    )
        except Exception as exc:
            logger.warning("WAL checkpoint (PASSIVE) failed: %s", exc)

    def __enter__(self) -> "SessionDB":
        """Enter a scope that closes this handle on the way out.

        Ownership of a SessionDB should be released explicitly.
        Historically an instance with a started token writer pinned ITSELF
        (bound-method writer target plus a strong ``atexit`` drain hook), so
        ``__del__`` never ran for exactly the instances that leaked
        descriptors (#88033).  The writer now retires after an idle window
        and the atexit hook holds only a weak reference, so abandoned
        handles are eventually collectible — but "eventually, after the
        idle window and a GC cycle" is not a release policy.  Call sites
        owning a handle are still expected to close it deterministically
        (see the ownership comments in ``run_agent.py`` and
        ``tui_gateway/methods_session.py``).

        This makes the correct usage the easy one, so an owning scope can be
        exception-safe by construction rather than by remembering a
        ``try/finally``:

            with SessionDB(path) as db:
                db.append_message(...)

        Purely additive: it changes nothing for callers that already call
        ``close()`` directly, and ``close()`` stays idempotent, so a scope
        that closes early still exits cleanly.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the handle, then let any exception propagate.

        Returns False (never suppressing), so ``with`` here only manages the
        descriptor lifetime and never swallows a caller's error.
        """
        self.close()
        return False

    def close(self):
        """Close the database connection.

        Drains queued token deltas first (the background writer needs the
        connection). Writable connections then attempt a PASSIVE WAL
        checkpoint (NOT TRUNCATE: transient per-cron-run connections close
        many times an hour, and a TRUNCATE fires a full WAL reset that
        races the gateway's live writer and tears B-tree pages — issue
        #45383). Read-only connections never request a checkpoint.

        When this instance is shared (opened via ``get_shared_session_db``),
        ``close()`` RELEASES one refcount instead of tearing down the
        connection: the registry owns the lifecycle and only closes on the
        final release (#90837).  This prevents one caller's close from
        tearing down the writer connection that other callers in the same
        process are still using — while still letting legacy ``close()``
        call sites return their reference instead of leaking it.
        """
        if getattr(self, "_shared_registry_owned", False):
            from hermes_state_registry import release

            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Drain the read-only connection pool.  Setting the closed flag
        # under the lock first means a reader still in flight closes its own
        # connection on release instead of re-populating a pool that has
        # already been drained.
        with self._read_conns_lock:
            self._read_conns_closed = True
        while True:
            try:
                conn = self._read_pool.get_nowait()
            except queue.Empty:
                break
            self._close_read_conn(conn)
        with self._lock:
            if self._conn:
                if self._db_corrupt:
                    # Quarantined handle (see StateDbCorruptError): no explicit
                    # checkpoint over a damaged page image.
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed structural corruption (%s). Take a "
                        "snapshot of state.db, -wal and -shm before restarting, "
                        "then run `hermes sessions recover --source %s "
                        "--inspect-only`.",
                        self.db_path,
                        self._db_corrupt_reason,
                        self.db_path,
                    )
                elif not self.read_only:
                    # PASSIVE, not TRUNCATE. Every cron run_agent opens+closes a
                    # transient SessionDB, so a TRUNCATE here fires a full WAL
                    # reset many times/hour, racing the gateway's long-lived
                    # writer on large WAL databases and tearing hot B-tree
                    # pages -- the #45383 corruption this class's own periodic
                    # checkpoint was already made PASSIVE to avoid. TRUNCATE
                    # belongs only on a sole-opener/quiescent connection.
                    try:
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception as exc:
                        logger.debug(
                            "WAL checkpoint (PASSIVE) at close failed: %s",
                            exc,
                        )
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)
                # A clean close of the last connection lets SQLite unlink the
                # WAL/SHM sidecars — a legitimate end of this handle's sidecar
                # generation, not a split (#94736 late writes must still
                # self-heal). Drop the recorded generation so a teardown-race
                # reopen re-adopts whatever exists then instead of halting.
                self._db_sidecar_identity = {}

    def __del__(self) -> None:
        """Safety net: close the connection if the caller forgot.

        The async accounting worker retires when idle and its atexit hook
        holds only a weak reference, so neither can pin an otherwise orphaned
        instance. During interpreter teardown the order of module cleanup is
        undefined, so every attribute access remains guarded.

        Delegates to ``close()`` so the read pool, token writer, and atexit
        hook are all cleaned up — not just the writer connection.
        """
        if self.__dict__.get("_conn") is None:
            return
        try:
            self.close()
        except Exception:
            pass

    # ── Chunked FTS rebuild engine (v23 opt-in optimize) ──
    #
    # `optimize_fts_storage()` (the `hermes sessions optimize-storage`
    # command) drops the legacy inline FTS indexes and backfills the new
    # external-content ones. A single blocking rebuild measured ~16 minutes
    # of held write lock on a real 25 GB DB, so the backfill runs in small
    # chunks, each in its own short write transaction:
    #   - concurrent readers/writers are never starved (WAL stays small,
    #     each chunk checkpoints via the normal _execute_write cadence);
    #   - an interrupted run (Ctrl-C, crash) resumes from
    #     fts_rebuild_progress when the command is re-run;
    #   - multiple processes sharing the DB don't double-run it — each chunk
    #     claims work by compare-and-swap on fts_rebuild_progress, so even a
    #     concurrent second runner just interleaves chunks safely.
    #
    # THROTTLING (the part that keeps a live gateway sharing the DB
    # responsive): a greedy chunk loop re-acquires BEGIN IMMEDIATE nearly
    # back-to-back and can starve another process's writer into exhausting
    # its lock retries (an early 5000-row/50ms version owned the write lock
    # ~85% of the time and visibly froze concurrent CLI sessions on a large
    # install). Two layers prevent that:
    #   1. Small chunks (500 rows) — a foreground write queues behind a
    #      chunk for at most ~tens of ms.
    #   2. Inter-chunk pause — the loop sleeps max(_FTS_REBUILD_MIN_PAUSE,
    #      chunk cost x _FTS_REBUILD_DUTY_FACTOR) between chunks, capping
    #      this process's share of DB bandwidth so concurrent writers always
    #      find open windows. This works cross-process (unlike any
    #      same-process activity stamp) because it bounds our own duty
    #      cycle unconditionally.

    _FTS_REBUILD_CHUNK_ROWS = 500
    _FTS_REBUILD_DUTY_FACTOR = 4.0      # sleep >= 4x chunk cost (≤20% duty)
    _FTS_REBUILD_MIN_PAUSE = 0.2        # seconds — floor between chunks

    # Demoted v22 FTS shadow tables awaiting teardown (see the v23 migration:
    # DROP of a multi-GB FTS vtable blocks for minutes, so the migration
    # demotes the vtable definitions out of sqlite_master and renames the
    # orphaned shadow tables — now plain tables — to fts_v22_trash_*; the
    # worker empties them in bounded chunks, then drops them cheaply).
    _FTS_TRASH_PREFIX = "fts_v22_trash_"

    # ── CJK-bigram index backfill (dedicated marker pair) ──
    #
    # Same chunk engine as the main deferred rebuild, but on the
    # ``fts_cjk_rebuild_*`` markers so a cjk-only backfill (the common case:
    # an already-optimized v23 DB gaining the cjk index) never gates the
    # complete ``messages_fts`` / trigram triggers.

    # ── Opt-in v23 FTS storage optimization (`hermes sessions optimize-storage`) ──
    #
    # This is the ONLY path that migrates an existing legacy (v22 inline) DB
    # to the v23 external-content schema. It is deliberately foreground and
    # user-invoked, never automatic, because it is disk-heavy and long. It
    # runs the throttled/resumable chunk engine above to completion
    # synchronously — demote → new schema → chunked backfill → chunked
    # teardown — with progress callbacks, a disk preflight in the CLI
    # wrapper, a VACUUM at the end, and a defensive schema_version bump.

    def _has_fts_trash(self, conn) -> bool:
        """True when demoted v22 shadow tables are still awaiting teardown.
        Caller must hold ``self._lock`` (or pass a migration-time cursor)."""
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE ? ESCAPE '\\' LIMIT 1",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        ).fetchone())

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    _PROFILE_DIR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

    def _own_profile_name(self) -> Optional[str]:
        """The profile that owns THIS store, derived from ``db_path`` alone.

        Every profile-tree ``state.db`` belongs to exactly one profile
        (``<root>/state.db`` → ``default``,
        ``<root>/profiles/<name>/state.db`` → ``<name>``), so the derivation
        is a single match, never a guess — the same contract
        :meth:`backfill_null_session_profiles` and the web listing's
        ``row_profile`` stamp rely on. Path-based (not
        ``get_active_profile_name()``) on purpose: a gateway serving a
        NON-launch profile opens that profile's store directly, and the row
        must be stamped with the store's owner, not the serving process's
        launch profile. Returns ``None`` for stores outside the profile tree
        (explicit ``db_path`` in tests, ad-hoc copies) — those rows keep the
        legacy NULL rather than a fabricated owner.
        """
        try:
            from hermes_constants import get_default_hermes_root

            root = get_default_hermes_root().resolve()
            parent = Path(self.db_path).resolve().parent
            if parent == root:
                return "default"
            if parent.parent == root / "profiles" and self._PROFILE_DIR_RE.match(
                parent.name
            ):
                return parent.name
        except Exception:
            logger.debug("own-profile derivation failed", exc_info=True)
        return None

    def _insert_session_row(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        session_key: Optional[str] = None,
        chat_id: str = None,
        chat_type: str = None,
        thread_id: str = None,
        parent_session_id: str = None,
        cwd: str = None,
        profile_name: Optional[str] = None,
        git_repo_root: str = None,
        origin_json: str = None,
        display_name: str = None,
    ) -> None:
        """Insert a session row, enriching NULL metadata on conflict.

        The gateway creates a bare row (source + user_id) before the agent
        exists and the agent's later ``create_session`` carries the real
        model/model_config/system_prompt; ``INSERT OR IGNORE`` dropped that
        enrichment, so the upsert ``COALESCE``-fills columns that are still
        NULL and never overwrites what an earlier writer set (a later bare
        source="unknown" call cannot clobber a real source/model).

        ``chat_id``/``thread_id`` record the messaging origin so gateway
        ``/resume`` can prove an inactive row belongs to the caller's
        chat/thread (IDOR scoping).

        With ``parent_session_id`` (compression fork, delegate spawn, branch),
        NULL ``cwd``/``git_repo_root``/``git_branch``/``profile_name`` are
        backfilled from the parent — child creators historically did not
        propagate them, so lineages dropped out of the project sidebar or were
        aggregated as "default" on every fork. NULL-fill only. Compression
        forks (parent ``end_reason='compression'``) also inherit the gateway
        origin columns (user_id/session_key/chat_id/chat_type/thread_id/
        display_name/origin_json) so a crash before the gateway re-records the
        peer cannot strand the child without a routing mapping.

        With no ``profile_name`` the row is stamped with THIS store's own
        profile (:meth:`_own_profile_name`): every state.db belongs to exactly
        one profile (the contract :meth:`backfill_null_session_profiles`
        relies on), and profile-keyed consumers treat NULL as unowned, hiding
        the session from the sidebar. Stores outside the profile tree derive
        nothing and keep NULL — never guess.
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
                    session_id,
                    source,
                    user_id,
                    session_key,
                    chat_id,
                    chat_type,
                    thread_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    cwd,
                    profile_name,
                    git_repo_root,
                    origin_json,
                    display_name,
                    time.time(),
                ),
            )
            if system_prompt_hash is not None:
                self._delete_unreferenced_system_prompts(conn)
            if parent_session_id:
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
                # Belt-and-suspenders for gateway routing metadata (#59527):
                # the gateway re-records the peer on the child after rotation
                # (d5b4879d4), but a hard crash between child creation and that
                # write leaves the child row without origin columns, so
                # ``find_latest_gateway_session_for_peer`` can't recover the
                # mapping on restart. Inherit them from the parent at creation
                # time — but ONLY for compression forks (parent already ended
                # with end_reason='compression'). Delegate/subagent children
                # are spawned while the parent is still live and must NOT
                # inherit routing keys, or peer recovery could repoint gateway
                # traffic into a subagent's session.
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
        # Session-row creation is transcript-critical: if it fails, the
        # first flush of a new session fails and the turn is aborted as
        # session_persistence_failed. Ride out long sibling holds.
        self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create a new session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id


    def set_expiry_finalized(self, session_id: str, finalized: bool = True) -> None:
        """Mark a gateway session's expiry-finalization flag in state.db.

        Mirrors ``SessionEntry.expiry_finalized`` (sessions.json) so the flag
        survives even if the JSON index is pruned or lost (#9006).
        """
        if not session_id:
            return

        self._write_sql(
            "UPDATE sessions SET expiry_finalized = ? WHERE id = ?",
            (1 if finalized else 0, session_id),
        )

    # ── Gateway routing index (replaces sessions.json, #9006 follow-up) ────


    def find_session_by_origin(
        self,
        *,
        platform: str,
        chat_id: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Find the most recent live session_id for a platform + chat origin.

        Equivalent of gateway/mirror's sessions.json scan: matches on
        source + chat_id (+ thread_id when provided).  When ``user_id`` is
        provided, exact sender matches are preferred; if multiple distinct
        users share the chat and none matches, returns None rather than
        contaminating another participant's session.
        """
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
        with self._read_ctx() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len(rows) > 1:
            distinct_users = {
                str(r.get("user_id") or "").strip()
                for r in rows
                if str(r.get("user_id") or "").strip()
            }
            if len(distinct_users) > 1:
                return None
        return str(rows[0]["id"])


    # ── Orphaned gateway-session repair (#82616) ──────────────────────────
    # A write-path failure (corrupt FTS, crash between routing publication
    # and row creation) can leave the live conversation in a session row
    # that never received its identity columns. Both queries above require
    # those columns, so the row holding the real transcript is invisible to
    # recovery: the chat resolves to the last keyed row instead — days older
    # — and the conversation time-travels. Hardening the write side cannot
    # reach a row that is *already* damaged; these two methods are the
    # offline repair path behind ``hermes sessions repair-routing``.

    # Widest plausible gap between a keyed predecessor going quiet and its
    # unkeyed successor being minted. The reported incident gap was ~60s;
    # 15 minutes stays generous without spanning unrelated conversations.
    _ORPHAN_ADOPTION_MAX_GAP_S = 900.0


    # Children that carry a ``parent_session_id`` but are NOT compression
    # continuations: branches, delegate/subagent runs, and tool sessions.
    # A marker only disqualifies a child when it points at the parent being
    # queried — compression continuations inherit the rotated agent's
    # ``model_config`` verbatim (``publish_compression_child`` callers pass
    # ``agent._session_init_model_config``), so a delegate subagent's
    # continuation carries ``_delegate_from=<the delegate's own parent>``.
    # Matching markers by mere presence misclassified those real
    # continuations as delegate children (fail-open for orphan reopen,
    # fail-closed for adoption). Bind the parent id for both markers.
    _NON_CONTINUATION_CHILD_FILTER_SQL = (
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._branched_from'), '') != ?\n"
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._delegate_from'), '') != ?\n"
        "  AND COALESCE({alias}source, '') != 'tool'\n"
    )


    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended.

        No-ops when the session is already ended. The first end_reason wins:
        compression-split sessions must keep their ``end_reason = 'compression'``
        record even if a later stale ``end_session()`` call (e.g. from a
        desynced CLI session_id after ``/resume`` or ``/branch``) targets them
        with a different reason. Use ``reopen_session()`` first if you
        intentionally need to re-end a closed session with a new reason.
        """
        def _do(conn):
            changed = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            ).rowcount
            # Only a boundary this call actually wrote advances the generation:
            # the first end_reason wins, so a no-op must not rotate the peer.
            if changed:
                self._bump_conversation_generation(conn, session_id, end_reason)
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed.

        Before clearing a reset boundary, stabilize markerless legacy reset
        children that still depend on the parent's mutable end_reason.
        """
        def _do(conn):
            placeholders = ",".join("?" for _ in _RESET_END_REASONS)
            # WHERE shape shared with _RESET_CHILD_SQL's fallback arm via
            # _legacy_reset_child_sql so the stamping and the listing
            # predicate cannot drift.
            conn.execute(
                "UPDATE sessions AS child SET model_config = json_set("
                "COALESCE(child.model_config, '{}'), '$._reset_from', "
                "child.parent_session_id) "
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

    def promote_to_session_reset(
        self, session_id: str, reason: str = "session_reset"
    ) -> bool:
        """Durably mark a session as ended by an intentional reset boundary.

        Promotes *only* live rows (``ended_at IS NULL``) or rows carrying an
        accidental end_reason that the recovery query
        (``find_latest_gateway_session_for_peer``) treats as recoverable:
        ``agent_close`` (older gateway cleanup bug) and ``ws_orphan_reap``
        (mistaken TUI reaper).  Explicit conversation boundaries such as
        ``compression``, ``session_reset``, ``session_switch``, etc. are
        preserved — the first writer wins for those, and a later expiry
        finalization must not silently overwrite them.

        Plain ``end_session()`` is NOT sufficient for reset boundaries: it
        no-ops on an already-ended row, so a row that agent cleanup already
        closed as ``agent_close`` would stay recoverable and stale-route
        recovery would resurrect the reset session with its full history
        (#61220, #61993, #63539).

        Keep this promotion set in sync with the recoverable set in
        ``find_latest_gateway_session_for_peer`` — any reason recovery would
        reopen must be promotable here.

        ``reason`` lets reset paths keep their auditable specific reasons
        (``idle``, ``daily``, ``suspended``, ``resume_pending_expired``).

        Returns ``True`` when the row was promoted, ``False`` when skipped
        (already has a different explicit end_reason, or row not found).
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
            # /new and the policy auto-resets promote rather than end_session,
            # so the generation has to advance here too — in the same
            # transaction as the boundary, and only when one was written.
            if cursor.rowcount:
                self._bump_conversation_generation(conn, session_id, reason)
            return cursor.rowcount

        try:
            rows = self._execute_write(_do)
            return bool(rows)
        except Exception:
            return False

    def update_session_cwd(
        self,
        session_id: str,
        cwd: str,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
        replace_git_meta: bool = False,
    ) -> Optional[int]:
        """Persist the authoritative cwd and claim a Git metadata generation.

        ``git_branch`` records the git branch checked out in ``cwd`` at the time
        the session started/resumed. The sidebar groups main-checkout sessions
        by this so feature-branch work doesn't pile under a single "main" row
        (the main checkout's *current* branch is transient and would
        misattribute past sessions).

        ``git_repo_root`` records the git repo this cwd belongs to — the
        authoritative project key. Resolving it here, at the lowest level, means
        every surface reads the same membership instead of re-probing git in the
        GUI over a partial page. Each field is only written when non-empty so a
        probe failure never clobbers a previously-captured value.

        ``replace_git_meta`` inverts that non-empty rule: a deliberate workspace
        MOVE (re-homing a session into another project) must overwrite the old
        repo identity even when the new cwd resolves to none — keeping the stale
        root would leave the session grouped under the project it just left.

        Every call increments ``git_metadata_generation`` in the same write
        transaction. Async Git probes must publish through
        :meth:`publish_session_git_metadata` with the returned generation, so
        an older worker cannot overwrite a newer cwd claim even after an
        A -> B -> A transition or from another process sharing this database.
        Metadata from a different cwd is cleared atomically with the move.
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
            sets = [
                "cwd = ?",
                "git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1",
            ]
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
            conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", params
            )
            row = conn.execute(
                "SELECT git_metadata_generation FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            value = row[0]
            return int(value)

        return self._execute_write(_do)

    def publish_session_git_metadata(
        self,
        session_id: str,
        cwd: str,
        generation: int,
        git_branch: Optional[str] = None,
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
            "WHERE id = ? AND cwd = ? "
            "AND git_metadata_generation = ?",
            params,
        ) == 1

    def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        """Persist resolved git repo roots for cwds that don't have one yet.

        Backfills history so projects light up for sessions created before the
        column existed, without clobbering an already-recorded root. Only
        non-empty roots are written (a non-git cwd stays NULL).
        """
        pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
        if not pairs:
            return

        def _do(conn):
            for root, cwd in pairs:
                conn.execute(
                    "UPDATE sessions SET git_repo_root = ? "
                    "WHERE cwd = ? AND COALESCE(git_repo_root, '') = ''",
                    (root, cwd),
                )

        self._execute_write(_do)


    # ──────────────────────────────────────────────────────────────────────
    # Compression locks
    # ──────────────────────────────────────────────────────────────────────
    # Atomic per-session locks that prevent two compression paths from
    # racing on the same session_id and producing orphan child sessions.
    #
    # The race: ``conversation_compression.py`` rotates ``agent.session_id``
    # as a side effect of a successful compression (end old session, create
    # new). That mutation is local to the AIAgent instance — but ``state.db``
    # is shared across all instances. Two AIAgents that share the same
    # ``session_id`` at the moment they both decide to compress (most
    # commonly the parent turn's agent + a background-review fork started
    # right after the turn ended) each end the parent and create their own
    # NEW session, parented to the same old id. The gateway SessionEntry
    # only catches one rotation; the other child silently accumulates
    # writes — Damien's "parent → two orphan children" repro shape.
    #
    # The lock is keyed by ``session_id`` and is held for the duration of
    # the compress() call plus the rotation. ``holder`` identifies the
    # current owner (pid:tid:nonce) for diagnostics; the lock is recovered
    # via ``expires_at`` if the holder process crashed without releasing.


    def touch_session_activity(
        self,
        session_id: str,
        ts: Optional[float] = None,
        *,
        description: Optional[str] = None,
        provenance: Optional[ActivityProvenance] = None,
    ) -> None:
        """Stamp durable mid-turn session activity (observation-only).

        Called (rate-limited) from ``AIAgent._touch_activity`` so gateway/CLI
        surfaces and stall consumers observe API/tool/compaction activity
        even when no new message row has been written yet (#72016 / #72039).

        Never moves ``last_activity_at`` backwards. When the timestamp
        advances, bounded ``last_activity_description`` /
        ``last_activity_provenance`` are written with it. No-ops when
        ``session_id`` is empty or the row does not exist.
        """
        if not session_id:
            return
        from agent.session_activity import (
            bound_activity_description,
            normalize_activity_provenance,
        )

        when = float(ts if ts is not None else time.time())
        desc = bound_activity_description(description)
        prov = normalize_activity_provenance(provenance).value

        # Observation-only write: never let it ride the full routine
        # write-patience budget (#76354 review S1). Under contention a
        # heartbeat that waits ~20s would delay the response-critical path
        # it is merely observing; give up after a sub-second budget instead
        # (the next due window retries naturally).
        self._write_sql(
            "UPDATE sessions SET "
            "last_activity_at = ?, "
            "last_activity_description = ?, "
            "last_activity_provenance = ? "
            "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
            (when, desc, prov, session_id, when),
            patience_s=self._ACTIVITY_WRITE_PATIENCE_S,
        )

    def clear_session_activity_labels(self, session_id: str) -> None:
        """Clear mid-turn activity labels after a turn ends.

        Keeps ``last_activity_at`` intact so idle / watchdog clocks stay
        continuous. Description and provenance are observation labels for
        *what was happening at* that timestamp during an active turn; once
        the turn is idle they must not keep advertising "compressing" /
        "executing tool" (#72039).

        Response-critical-path contract (#76354 review S1): runs in the
        turn's ``finally``; a no-op clear (labels already empty) skips the
        write transaction entirely, and a real clear uses the same short
        sub-second busy budget as :meth:`touch_session_activity` instead of
        the full routine write patience.
        """
        if not session_id:
            return
        from agent.session_activity import ActivityProvenance

        # No-op fast path: skip the transaction when there is nothing to
        # clear. Read-only, no write lock.
        try:
            row = self._read_one(
                "SELECT last_activity_description, last_activity_provenance "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )
        except sqlite3.Error:
            row = None
        if row is not None:
            desc = row[0]
            prov = row[1]
            if not desc and (
                not prov or prov == ActivityProvenance.UNKNOWN.value
            ):
                return

        self._write_sql(
            "UPDATE sessions SET "
            "last_activity_description = ?, "
            "last_activity_provenance = ? "
            "WHERE id = ?",
            ("", ActivityProvenance.UNKNOWN.value, session_id),
            patience_s=self._ACTIVITY_WRITE_PATIENCE_S,
        )

    def get_session_activity(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the durable activity snapshot for *session_id*, or None."""
        if not session_id:
            return None
        row = self.get_session(session_id)
        if not row:
            return None
        from agent.session_activity import build_activity_snapshot

        return build_activity_snapshot(
            last_activity_at=row.get("last_activity_at"),
            last_activity_description=row.get("last_activity_description"),
            last_activity_provenance=row.get("last_activity_provenance"),
        )

    def update_session_meta(
        self,
        session_id: str,
        model_config_json: str,
        model: Optional[str] = None,
    ) -> None:
        """Update model_config and optionally model for an existing session.

        Uses COALESCE so that passing model=None leaves the stored model
        column unchanged.  Routes through _execute_write for the standard
        BEGIN IMMEDIATE + jitter-retry + lock guarantee.
        """
        # Barrier against queued token deltas — see update_session_model.
        self.flush_token_counts()

        self._write_sql(
            "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
            (model_config_json, model, session_id),
        )

    def update_system_prompt(
        self, session_id: str, system_prompt: Optional[str]
    ) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                "UPDATE sessions "
                "SET system_prompt_hash = ?, system_prompt = NULL WHERE id = ?",
                (system_prompt_hash, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def update_session_tool_names(
        self, session_id: str, tool_names: Optional[List[str]]
    ) -> None:
        """Persist the session's resolved ``tools[]`` name order (JSON array).

        Read back by ``tools.mcp_tool.restore_agent_tool_prefix`` when a fresh
        ``AIAgent`` is rebuilt for an existing session (gateway agent-cache
        eviction) so a flipped ``check_fn`` verdict can't fork the cached tool
        prefix. ``None`` clears the pin.
        """
        payload = json.dumps(list(tool_names)) if tool_names is not None else None

        self._write_sql(
            "UPDATE sessions SET tool_names = ? WHERE id = ?",
            (payload, session_id),
        )

    def update_session_model(
        self, session_id: str, model: str, provider: Optional[str] = None
    ) -> None:
        """Update the model for a session after a mid-session switch.

        Unlike ``update_token_counts`` which uses ``COALESCE(model, ?)``
        (only filling in NULL), this unconditionally sets the model column
        so that the dashboard reflects the user's latest /model choice.
        Also nulls ``system_prompt`` so stale ``Model:`` / ``Provider:``
        footer metadata is rebuilt on the next turn. A successful /model
        switch explicitly replaces any confirmed Browser runtime lock while
        preserving unrelated lineage markers in ``model_config``.

        When *provider* is given, it is merged into ``model_config``
        alongside the model (``$.model`` / ``$.provider``) so a later
        resume recombines the persisted model with the provider that
        actually serves it instead of the config.yaml primary provider
        (#79536). Callers without provider knowledge leave any stored
        provider untouched.
        """
        # This write bypasses the token queue, so deltas enqueued before the
        # switch must land first: a still-queued first delta carries the
        # pre-switch route, and applying it after this UPDATE would trip the
        # first_accounted_route overwrite in update_token_counts (row sees
        # api_call_count == 0 + a route mismatch) and resurrect the old
        # model/provider. Flushing here restores the pre-queue ordering.
        self.flush_token_counts()

        def _do(conn):
            # Use the shared merge discipline so lineage markers like
            # _branched_from / _delegate_from survive. browser_model_lock
            # is deleted via a None patch value (same semantics as the
            # old json_remove).
            patch: Dict[str, Any] = {"browser_model_lock": None}
            if model:
                patch["model"] = model
            if provider:
                patch["provider"] = provider
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                "UPDATE sessions SET "
                "model = ?, model_config = ?, "
                "system_prompt = NULL, system_prompt_hash = NULL "
                "WHERE id = ?",
                (model, merged, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def _merge_model_config_json(
        self,
        conn,
        session_id: str,
        patch: Dict[str, Any],
        *,
        on_missing: str = "skip",
    ):
        """SELECT + tolerant-parse + merge ``patch`` into a session's model_config.

        Shared by every model_config writer (``update_session_runtime_lock``,
        ``set_session_yolo``, ``archive_and_compact``,
        ``patch_session_model_config``) so the merge discipline that keeps
        lineage markers like ``_branched_from`` / ``_delegate_from`` alive
        lives in exactly one place. A ``None`` patch value deletes that key.
        Must run inside an open write transaction (callers own the UPDATE).

        Returns the serialized merged JSON — ``None`` when the merged dict is
        empty (matching ``create_session``'s NULL convention) — or the
        ``_MODEL_CONFIG_ROW_MISSING`` sentinel when the row doesn't exist and
        ``on_missing == "skip"``; ``on_missing == "raise"`` raises ValueError.
        """
        row = conn.execute(
            "SELECT model_config FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            if on_missing == "raise":
                raise ValueError(f"Session not found: {session_id}")
            return _MODEL_CONFIG_ROW_MISSING
        raw = row[0]
        config: Dict[str, Any] = {}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    config = parsed
            except (json.JSONDecodeError, TypeError):
                config = {}
        elif isinstance(raw, dict):
            config = dict(raw)
        for key, value in patch.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = value
        return json.dumps(config) if config else None

    def patch_session_model_config(
        self, session_id: str, patch: Dict[str, Any]
    ) -> None:
        """Merge ``patch`` into a session's model_config JSON atomically.

        A ``None`` patch value removes that key. No-op when the session row
        doesn't exist or the patch is empty. This is the standalone setter for
        callers that need to update model_config *without* rewriting the
        transcript (the transcript-coupled path is ``archive_and_compact``'s
        ``model_config_patch``, which shares the same merge helper).
        """
        if not session_id or not patch:
            return

        def _do(conn):
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ?",
                (merged, session_id),
            )

        self._execute_write(_do)

    def get_session_model_config_value(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        """Read one key out of a session's model_config JSON (tolerant parse)."""
        session = self.get_session(session_id) or {}
        raw = session.get("model_config")
        config: Dict[str, Any] = {}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    config = parsed
            except (json.JSONDecodeError, TypeError):
                config = {}
        elif isinstance(raw, dict):
            config = raw
        return config.get(key, default)

    def update_session_runtime_lock(
        self,
        session_id: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        route_source: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """Persist a Browser / API client runtime lock without clobbering lineage markers.

        Merges ``browser_model_lock`` into the existing ``model_config`` JSON so
        ``_branched_from`` / ``_delegate_from`` survive. Nulls ``system_prompt``
        so cached ``Model:`` / ``Provider:`` footers cannot lie after a switch.
        """
        lock = {
            "provider": provider or "",
            "model": model or "",
            "model_options": model_options or {},
            "route_source": route_source or "",
            "confirmed": bool(confirmed),
            "updated_at": time.time(),
        }

        def _do(conn):
            merged = self._merge_model_config_json(
                conn, session_id, {"browser_model_lock": lock}
            )
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                """UPDATE sessions SET
                   model_config = ?,
                   model = COALESCE(?, model),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (merged, model, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def set_session_yolo(self, session_id: str, enabled: bool) -> None:
        """Persist the per-session YOLO bypass flag into ``model_config``.

        Merges ``yolo_mode`` into the existing ``model_config`` JSON (same
        merge discipline as ``update_session_runtime_lock`` so lineage
        markers like ``_branched_from`` / ``_delegate_from`` survive). The
        CLI resume paths read this flag back so a ``/yolo ON`` toggle — or a
        ``--yolo`` launch — survives ``hermes --resume`` into a fresh
        process. No-op when the session row doesn't exist yet; the
        creation-time ``model_config`` carries the flag for ``--yolo``
        launches.
        """
        if not session_id:
            return

        def _do(conn):
            merged = self._merge_model_config_json(
                conn, session_id, {"yolo_mode": bool(enabled)}
            )
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ?",
                (merged, session_id),
            )
        self._execute_write(_do)

    @staticmethod
    def session_yolo_enabled(session_meta: Optional[Dict[str, Any]]) -> bool:
        """Read the persisted YOLO flag off a session row dict.

        Accepts the dict returned by ``get_session`` (``model_config`` is a
        JSON string) or an already-parsed dict. Returns False on any parse
        failure — resume must never enable the bypass by accident.
        """
        raw = (session_meta or {}).get("model_config")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return False
        if not isinstance(raw, dict):
            return False
        return bool(raw.get("yolo_mode"))


    # ── Async token accounting ──
    # update_token_counts() runs a sessions UPDATE (plus a per-model usage
    # upsert) inside BEGIN IMMEDIATE; against a cold multi-GB state.db one
    # call can stall the turn thread for tens to hundreds of ms, and the
    # tool loop pays it after EVERY API call (measured p50 3.3ms / p95 70ms
    # per call in production). queue_token_counts() reduces the critical
    # path to a deque append: a dedicated single-writer thread applies
    # deltas in enqueue order, coalescing consecutive same-route deltas
    # into one UPDATE when a backlog forms. Readers that need exact
    # mid-turn totals (get_session and friends) call flush_token_counts()
    # first — a plain attribute check when nothing is queued.

    # Delta fields summed when coalescing. Route fields must be equal for
    # two deltas to merge: model/billing_* feed COALESCE backfill and the
    # per-model usage attribution key, and cost_status/cost_source are
    # last-non-None-wins — equality makes the merged UPDATE byte-for-byte
    # equivalent to applying the deltas sequentially.
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
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
        **kwargs,
    ) -> str:
        """Ensure a session row exists (INSERT OR IGNORE). Accepts optional kwargs."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id


    # ── Cross-backend heartbeat API (#94895) ───────────────────────────
    # Each serve / tui_gateway process registers a heartbeat row at startup
    # and refreshes ``last_heartbeat`` periodically. The startup orphan
    # sweep reads these rows to avoid reaping sessions owned by another
    # still-live backend that just happens to be idle. Backends remove
    # their own row on graceful shutdown; a row that survives a crash is
    # reclaimed by the staleness sweep once ``last_heartbeat`` ages out.


    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        # Cost/usage readers (/status, /usage, gateway endpoints) reach the
        # row through here; drain queued token deltas so they see exact
        # totals. No-op attribute check when nothing is queued.
        self.flush_token_counts()
        row = self._read_one(
            "SELECT s.*, "
            "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
            "FROM sessions s "
            "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            "WHERE s.id = ?",
            (session_id,),
        )
        return self._session_row_dict(row) if row else None

    def get_dominant_session_model_route(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the main-loop model route that served most API calls.

        ``sessions`` is a legacy aggregate row and can hold model/provider fields
        written by different route changes. ``session_model_usage`` keeps the
        coherent per-call tuple, so persisted status and billing reads should use
        its dominant main-loop route when one is available.
        """
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
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]

        escaped = _escape_like(session_id_or_prefix)
        matches = [row["id"] for row in self._read_all(
            "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
            (f"{escaped}%",),
        )]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority. An auto-titling write may
    # only replace a title of strictly lower authority, so the instant
    # ``derived`` title upgrades to the model's ``llm`` title exactly once and
    # nothing the agent generates can ever clobber a name the user typed.
    TITLE_SOURCE_DERIVED = "derived"
    TITLE_SOURCE_LLM = "llm"
    TITLE_SOURCE_USER = "user"
    _TITLE_SOURCE_RANK = {
        TITLE_SOURCE_DERIVED: 0,
        TITLE_SOURCE_LLM: 1,
        TITLE_SOURCE_USER: 2,
    }

    # Bot Mode's forever-chat registry: the session titled exactly this, on a
    # bot's profile, IS the bot's canonical chat — resolved by exact-title
    # lookup on every open (no session-id pointer exists). The title is the
    # identity, which is why _set_session_title refuses renames of a hidden
    # row holding it (#92473).
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"


    def backfill_null_session_profiles(self, profile_name: str) -> int:
        """One-shot owner backfill for legacy pre-ownership session rows.

        Sessions created before the durable-ownership work (#95407 lineage)
        carry ``profile_name = NULL``. On single-backend installs that was
        harmless, but once a Desktop registers a second connection the
        fail-closed owner ladder (which is correct for new sessions) can no
        longer route those rows anywhere — every pre-campaign session becomes
        unresumable after upgrade (#94724, field report).

        This store belongs to exactly one profile — the profile whose
        ``state.db`` this is — so stamping its own name onto rows that never
        recorded one is a single-match backfill, not a guess. Rules mirror the
        ``create_session`` COALESCE contract:

        * only ``NULL``/empty ``profile_name`` rows are touched — a non-NULL
          owner is NEVER overwritten;
        * idempotent and one-shot-per-row: a second run matches zero rows.

        Returns the number of rows stamped (0 when nothing was legacy).
        """
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
        """Set one ``sessions`` column across a whole compression lineage.

        Walks ancestors and descendants joined by ``end_reason='compression'``
        so the root and every continuation flip as a unit — Desktop projects
        compression roots forward to their latest tip, and updating only the
        displayed tip would let the untouched root resurrect it on refresh.
        Returns True when at least one row changed.
        """
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
        """Archive or unarchive a session.

        Archived sessions are hidden from the default session list but keep all
        their messages — this is a soft hide, not a delete. For compression
        chains, archive the whole logical conversation. Desktop lists compression
        roots projected forward to their latest continuation; updating only the
        displayed tip lets the still-unarchived root resurrect it on refresh.
        Returns True when at least one row was updated.
        """
        return self._set_lineage_column('archived', session_id, 1 if archived else 0)

    # Accidental end reasons that recovery treats as resumable. Single source
    # of truth: hermes_state_common._RECOVERABLE_END_REASONS, interpolated
    # into find_latest_gateway_session_for_peer / promote_to_session_reset
    # SQL — literals cannot drift (docs/session-lifecycle.md "recoverable
    # accidental reasons").
    RECOVERABLE_END_REASONS = _RECOVERABLE_END_REASONS

    def unarchive_recoverable_session(self, session_id: str) -> bool:
        """Un-archive a session that was archived by a recoverable accident.

        Registry-style lookups (Bot Mode's canonical "Bot Chat") use this to
        resurrect a row the ws-orphan reaper (``ws_orphan_reap``) or older
        agent cleanup (``agent_close``) archived: those ends are accidents,
        not user intent, so the identity-scoped canonical chat must survive
        them (#92687). Sessions archived with no end_reason or an explicit
        boundary reason (user archived deliberately, ``session_reset``, …)
        are left untouched — returns ``False`` for those, ``True`` only when
        the row was archived for a recoverable reason and is now un-archived
        (whole compression lineage, via :meth:`set_session_archived`).
        """
        if not session_id:
            return False
        try:
            row = self.get_session(session_id)
        except Exception:
            return False
        if not row or not row.get("archived"):
            return False
        # A compressed lineage's registry row keeps end_reason='compression';
        # the accidental stamp lives on the live TIP. Judge recoverability at
        # the tip (== the row itself when uncompressed).
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

        # Clear the accidental end stamp: the session is live again, and a
        # surviving ws_orphan_reap/agent_close reason would make a LATER
        # deliberate archive (which never writes end_reason) auto-resurrect
        # on the next lookup — permanently overriding user intent.
        def _clear_end(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (tip["id"],),
            )
            return 1

        self._execute_write(_clear_end)
        return True

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        """Pin or unpin a session (and its whole compression lineage).

        ``pinned`` is a durable "keep" flag: pinned sessions are exempt from
        the ``sessions.auto_archive`` stale sweep (see
        :meth:`archive_stale_sessions`). Desktop is the current writer — its
        sidebar pins mirror here so a backend/other-surface sweep honours
        them. Like :meth:`set_session_archived` the whole compression chain is
        flipped as a unit, so pinning the surfaced tip protects the root (and
        vice-versa) no matter which id the caller holds. Returns True when at
        least one row changed.
        """
        return self._set_lineage_column('pinned', session_id, 1 if pinned else 0)

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        """Hide or unhide a session (and its whole compression lineage).

        ``hidden`` is a generic "don't show in the global Sessions sidebar"
        flag: a hidden session is dropped from the default
        :meth:`list_sessions_rich` listing (which omits ``include_hidden``) but
        stays fully resumable by the surface that owns it — useful for plugins
        that manage their own sessions (e.g. kanban) and don't want them
        cluttering the shared recents list. Like :meth:`set_session_archived`
        / :meth:`set_session_pinned` the whole compression chain is flipped as
        a unit, so hiding the surfaced tip hides the root (and vice-versa) no
        matter which id the caller holds. Returns True when at least one row
        changed.
        """
        return self._set_lineage_column('hidden', session_id, 1 if hidden else 0)

    def set_session_read(self, session_id: str, read: bool = True) -> bool:
        """Mark a session read or unread (and its whole compression lineage).

        Read state is a watermark, not a flag: ``last_read_at`` records when
        the conversation was last read, and it counts as unread when activity
        postdates that watermark (the derived ``unread`` key on
        :meth:`list_sessions_rich` rows). New messages therefore flip a read
        conversation back to unread without any write on the message path.
        Three states:

        * NULL — never tracked (every pre-feature row): treated as read, so
          shipping the column doesn't badge a user's entire history at once.
        * 0 — explicitly marked unread: any activity postdates it.
        * timestamp — read up to that moment.

        Like :meth:`set_session_archived` / :meth:`set_session_pinned`, the
        whole compression chain is stamped as a unit, so reading the surfaced
        tip clears the root (and vice-versa) no matter which id the caller
        holds. Returns True when at least one row changed.
        """
        return self._set_lineage_column('last_read_at', session_id, time.time() if read else 0.0)

    @staticmethod
    def session_unread(session_row: Dict[str, Any]) -> bool:
        """Derive unread from a session row's watermark and activity.

        Shared by ``list_sessions_rich`` and any future surface that holds a
        row (or projected row) with ``last_read_at`` and ``last_active``.
        NULL watermark = never tracked = read.
        """
        last_read = session_row.get("last_read_at")
        if last_read is None:
            return False
        last_active = session_row.get("last_active") or session_row.get("started_at")
        return float(last_active or 0) > float(last_read)


    # Columns excluded from compact_rows projections: only the payload-heavy
    # blob no list consumer renders. Everything else — including gateway
    # routing fields and desktop sidebar fields like git_branch — stays, and
    # the projection is derived from SCHEMA_SQL so columns added later via
    # declarative reconciliation are included automatically instead of
    # silently dropping out of list rows.
    _SESSION_COMPACT_EXCLUDED = frozenset(
        {"system_prompt", "system_prompt_hash", "git_metadata_generation"}
    )
    _session_compact_cols_sql: Optional[str] = None


    def list_sessions_rich(
        self,
        source: str = None,
        sources: List[str] = None,
        exclude_sources: List[str] = None,
        cwd_prefix: str = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        min_message_count: int = 0,
        project_compression_tips: bool = True,
        order_by_last_active: bool = False,
        include_archived: bool = False,
        archived_only: bool = False,
        id_query: str = None,
        search_query: str = None,
        compact_rows: bool = False,
        include_pinned: bool = False,
        session_key: str = None,
        include_hidden: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last_active, in one query.

        ``last_active`` = freshest of the activity heartbeat and the latest
        message timestamp, else ``started_at``. Implementation-detail children
        (subagent runs, compression continuations) are excluded unless
        ``include_children``; branch/reset children stay listable.

        ``project_compression_tips`` (default) surfaces each compression chain
        as ONE entry showing the live tip's id/message_count/title/last_active;
        ``False`` returns raw root rows (admin/debug UIs).
        ``order_by_last_active`` sorts by the chain TIP's activity via a
        recursive CTE, so LIMIT/OFFSET stay cheap and a recently continued old
        conversation lands in the right slot. ``search_query`` (that path
        only) substring-matches title/id across the forward chain, plus a
        punctuation-stripped variant (``an94`` finds ``AN-94``).
        ``compact_rows`` omits the ``system_prompt`` blob from the SELECT so
        SQLite never copies tens of KB per row out of the B-tree page.
        ``include_pinned`` back-fills pinned conversations the page window left
        out (a pin means "always reachable"; the desktop sidebar would
        otherwise render an empty Pinned section) — they still obey the
        source/archived/min_message_count filters. ``session_key`` restricts to
        one gateway conversation scope.
        """
        # Rows carry token/cost totals — drain queued deltas first so
        # listings (sidebar, /resume, dashboards) show exact counters.
        self.flush_token_counts()
        where_clauses = []
        params = []

        if not include_children:
            # Show roots and user-visible branch/reset sessions, while still
            # hiding sub-agent runs and compression continuations. All four
            # carry parent_session_id, so the shared predicate classifies the
            # edge from stable markers plus legacy-compatible parent metadata.
            #
            # Branch sessions are identified two ways, OR'd for robustness:
            #   1. A stable ``_branched_from`` marker in model_config, written
            #      by /branch at creation time. This survives the parent being
            #      reopened and re-ended with a different end_reason (e.g.
            #      tui_shutdown overwriting 'branched'), which otherwise hides
            #      the branch — see issue #20856.
            #   2. The legacy heuristic (parent ended with 'branched' before the
            #      child started), covering branch sessions created before the
            #      marker existed.
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")

        include_sources = [source] if source else list(sources or [])
        if include_sources:
            placeholders = ",".join("?" for _ in include_sources)
            where_clauses.append(f"s.source IN ({placeholders})")
            params.extend(include_sources)
        if session_key:
            where_clauses.append("s.session_key = ?")
            params.append(session_key)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where_clauses.append(clause)
            params.extend(clause_params)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")
        if not include_hidden:
            where_clauses.append("s.hidden = 0")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        # Snapshot the filter params before the query builders below extend
        # them with LIMIT/OFFSET — the pinned back-fill reuses the same WHERE.
        base_where_params = list(params)
        prompt_select = (
            "" if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            "" if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )

        # Optional session-id filter, pushed into SQL so callers (Desktop
        # session-id search) don't have to fetch every row and filter in
        # Python. ``id_query`` is matched as a case-insensitive substring
        # against each surfaced row's id AND every id in its forward
        # compression chain — so searching a compression *root* id or a *tip*
        # id both resolve to the same projected conversation. Only used in the
        # order_by_last_active path (which builds the chain CTE); other callers
        # pass id_query=None.
        id_needle = (id_query or "").strip().lower()
        search_needle = (search_query or "").strip().lower()
        if order_by_last_active:
            # Compute effective_last_active by walking each surfaced session's
            # compression-continuation chain forward in SQL and taking the MAX
            # timestamp across the chain. This lets us ORDER BY + LIMIT at SQL
            # level instead of fetching every row and sorting in Python, while
            # still surfacing old compression roots whose live tip is fresh.
            #
            # The CTE seeds from rows the outer WHERE admits (roots +
            # user-visible branch/reset children), then recursively joins through
            # compression-continuation edges. Do NOT require
            # child.started_at >= parent.ended_at here: real desktop/gateway
            # races can insert the continuation row before the parent's
            # ended_at is written, while stale websocket siblings may satisfy
            # the timestamp test and hijack resume/list projection.
            outer_where = where_sql
            id_params: List[Any] = []
            filter_clauses: List[str] = []

            def _like_pattern(needle: str) -> str:
                return f"%{_escape_like(needle)}%"

            if id_needle:
                # Admit a surfaced row if its own id or any id in its forward
                # compression chain matches the needle. LIKE with a leading
                # wildcard can't use an index, but the chain membership and
                # the small result set keep this bounded — far cheaper than
                # fetching every session and scanning in Python.
                filter_clauses.append(
                    "EXISTS (SELECT 1 FROM chain cq"
                    "        WHERE cq.root_id = s.id"
                    "          AND LOWER(cq.cur_id) LIKE ? ESCAPE '\\')"
                )
                id_params.append(_like_pattern(id_needle))
            if search_needle:
                # Same chain-membership trick as id_query, but matching either
                # the title or the id of any session in the chain. The compact
                # (punctuation-stripped) variant lets `an94` match `AN-94`.
                compact_needle = re.sub(r"[\W_]+", "", search_needle)
                compact_sql = (
                    "REPLACE(REPLACE(REPLACE(REPLACE(LOWER(COALESCE({0}, '')),"
                    " '-', ''), '_', ''), '.', ''), ' ', '')"
                )
                search_clause = (
                    "EXISTS (SELECT 1 FROM chain cq"
                    " JOIN sessions cs ON cs.id = cq.cur_id"
                    " WHERE cq.root_id = s.id"
                    " AND (LOWER(COALESCE(cs.title, '')) LIKE ? ESCAPE '\\'"
                    " OR LOWER(cq.cur_id) LIKE ? ESCAPE '\\'"
                )
                id_params.extend([_like_pattern(search_needle)] * 2)
                if compact_needle:
                    search_clause += (
                        f" OR {compact_sql.format('cs.title')} LIKE ? ESCAPE '\\'"
                    )
                    id_params.append(_like_pattern(compact_needle))
                filter_clauses.append(search_clause + "))")
            if filter_clauses:
                combined = " AND ".join(filter_clauses)
                outer_where = (
                    f"{where_sql} AND {combined}" if where_sql else f"WHERE {combined}"
                )
            _sel = self._compact_session_cols() if compact_rows else "s.*"
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
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    {_sql_session_last_active("s")} AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {prompt_join}
                {outer_where}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
                LIMIT ? OFFSET ?
            """
            # WHERE params apply twice (CTE seed + outer select); the id filter
            # only applies to the outer select.
            params = params + params + id_params + [limit, offset]
        else:
            _sel = self._compact_session_cols() if compact_rows else "s.*"
            query = f"""
                SELECT {_sel}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    {_sql_session_last_active("s")} AS last_active
                FROM sessions s
                {prompt_join}
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
        rows = self._read_all(query, params)
        sessions = []
        for row in rows:
            s = self._session_row_dict(row)
            s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
            # Drop the internal ordering column so callers see a clean dict.
            s.pop("_effective_last_active", None)
            sessions.append(s)

        # Back-fill pinned conversations the page missed. A pin outlives
        # recency, so this runs BEFORE compression projection below — a
        # back-filled root then projects to its live tip exactly like a row
        # that had made the page on its own. One extra query, bounded by the
        # number of pins (a handful), never N+1 per pin.
        if include_pinned:
            seen_ids = {s["id"] for s in sessions}
            pinned_where = (
                f"{where_sql} AND s.pinned = 1" if where_sql else "WHERE s.pinned = 1"
            )
            _sel = self._compact_session_cols() if compact_rows else "s.*"
            pinned_query = f"""
                SELECT {_sel}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                FROM sessions s
                {prompt_join}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
            pinned_rows = self._read_all(pinned_query, base_where_params)
            for row in pinned_rows:
                s = self._session_row_dict(row)
                if s["id"] in seen_ids:
                    continue
                s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
                seen_ids.add(s["id"])
                sessions.append(s)

        # Project compression roots forward to their tips. Each row whose
        # end_reason is 'compression' has a continuation child; replace the
        # surfaced fields (id, message_count, title, last_active, ended_at,
        # end_reason, preview) with the tip's values so the list entry acts
        # as the live conversation. Keep the root's started_at to preserve
        # chronological ordering by original conversation start.
        if project_compression_tips and not include_children:
            # get_compression_tip() walks each root's chain individually (it's
            # a per-session graph walk, not batchable in one query), but the
            # tip *row* fetch afterward was previously one _get_session_rich_row()
            # call per compression root. Batch that half instead: resolve
            # every tip id first, then fetch all tip rows in a single query.
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
                self._get_session_rich_rows_batch(
                    set(tip_ids_by_root.values()), compact_rows=compact_rows
                )
                if tip_ids_by_root
                else {}
            )

            projected = []
            for s in sessions:
                tip_id = tip_ids_by_root.get(s["id"])
                tip_row = tip_rows.get(tip_id) if tip_id else None
                if not tip_row:
                    projected.append(s)
                    continue
                # Preserve the root's started_at for stable sort order, but
                # surface the tip's identity and activity data.
                merged = dict(s)
                for key in (
                    "id", "ended_at", "end_reason", "message_count",
                    "tool_call_count", "title", "last_active", "preview",
                    "model", "system_prompt", "cwd", "git_branch", "git_repo_root",
                ):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged["_lineage_root_id"] = s["id"]
                # Every id on the chain, intermediates included. Root and tip
                # alone are not enough client-side: a persisted tile or route
                # can hold a MIDDLE segment's id (it was the tip when opened,
                # then rotated again), and with only the root/tip pair such a
                # surface can no longer prove it names this conversation —
                # which is how one chat ends up open twice after compaction.
                merged["_lineage_ids"] = chain_by_root.get(s["id"]) or None
                projected.append(merged)
            sessions = projected

        # Derive read state per surfaced conversation. ``last_read_at`` is
        # lineage-stamped by set_session_read, so a projected row's root
        # watermark and its tip's are the same value — comparing it against
        # the tip's last_active is correct either way.
        for s in sessions:
            s["unread"] = self.session_unread(s)

        return sessions

    def session_lifecycle_statuses(
        self, session_ids: List[str]
    ) -> Dict[str, str]:
        """Classify each session's lifecycle state from its LAST message row.

        Returns ``{session_id: status}`` where status is one of:

        - ``'complete'``    — last message is a normal assistant reply
        - ``'interrupted'`` — last message is a user turn, a pending assistant
          tool call (no tool result followed), or a tool result the assistant
          never responded to
        - ``'error'``       — last message carries an error finish_reason
        - ``'empty'``       — session has no messages

        Cost-bounded by design: one query that resolves each listed session's
        newest message id via ``MAX(id)`` (an index seek on
        ``idx_messages_session_id``) and joins back for that single row's
        role/tool_calls/finish_reason. Never scans transcripts, so it stays
        cheap on large databases regardless of total message volume.
        """
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
                role=row["role"],
                has_tool_calls=bool(row["has_tool_calls"]),
                finish_reason=row["finish_reason"],
            )
        return statuses

    # =========================================================================
    # Message storage
    # =========================================================================

    # Sentinel prefix used to distinguish JSON-encoded structured content
    # (multimodal messages: lists of parts like text + image_url) from plain
    # string content. The NUL byte is not legal in normal text, so this
    # cannot collide with real user content.
    _CONTENT_JSON_PREFIX = "\x00json:"


    #: Key under which message reactions live inside ``display_metadata``.
    #: Reactions share the existing per-message JSON column rather than a side
    #: table so they survive rewind/compaction row rewrites with the row itself.
    REACTIONS_METADATA_KEY = "reactions"


    # Columns every conversation projection decodes. Shared by
    # get_messages_as_conversation and get_resume_conversations so a single
    # SELECT can feed both the model-fed and display views. ``active`` rides
    # along so a display read can split the compaction-archived rows from the
    # live set (and feed _dedupe_display_generations) without a second query.
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, active, "
        "api_content, display_kind, display_metadata"
    )


    def assert_export_safe(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> int:
        """Return active row count or reject an unsafe in-memory export.

        Exporting one session does not include compression ancestors, so this
        guard deliberately counts only the requested segment. The limited
        subquery stops as soon as it proves the transcript exceeds the bound.

        ``max_messages=None`` resolves the limit from config
        (``sessions.max_export_messages``); 0 disables the guard and returns
        the active row count without raising.
        """
        if max_messages is None:
            max_messages = resolved_max_export_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            # Guard disabled by config — skip the COUNT; live callers use
            # this for its raise side effect only (and skip calling it
            # entirely when the limit is 0).
            return 0
        row = self._read_one(
            "SELECT COUNT(*) FROM ("
            "SELECT 1 FROM messages WHERE session_id = ? AND active = 1 LIMIT ?"
            ")",
            (session_id, max_messages + 1),
        )
        message_count = int(row[0] if row else 0)
        if message_count > max_messages:
            raise SessionExportTooLargeError(session_id, message_count, max_messages)
        return message_count


    def _is_explicit_branch_session(self, session_id: str) -> bool:
        """Return whether *session_id* is a copied user-facing branch.

        Branches and compression continuations both use ``parent_session_id``,
        but they have different history semantics: a branch owns a copied
        transcript, while a compression continuation needs its ended parent's
        archived rows for display. The durable ``_branched_from`` marker is the
        existing discriminator written by all branch creation paths.
        """
        if not session_id:
            return False
        row = self._read_one(
            "SELECT model_config FROM sessions WHERE id = ?",
            (session_id,),
        )
        if row is None:
            return False
        raw_config = row["model_config"] if hasattr(row, "keys") else row[0]
        if not raw_config:
            return False
        try:
            config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(config, dict) and bool(config.get("_branched_from"))


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
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                current = row["parent_session_id"] if hasattr(row, "keys") else row[0]
        return list(reversed(chain)) or [session_id]


    # =========================================================================
    # Rewind (soft-delete) — see /rewind slash command + issue #21910
    # =========================================================================


    # =========================================================================
    # Search
    # =========================================================================

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
        workspace_key: str = None,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column
        (freshest of ``last_activity_at`` and latest message timestamp,
        else ``started_at``), ordered by most-recently-used first.

        Pass ``workspace_key`` to scope rows to one workspace - matching
        :func:`workspace_key` semantics (git repo root, else cwd). Used by
        ``hermes -c``/``--resume`` so the "last" session is the last one in
        the *current* workspace, not the global MRU.
        """
        select_with_last_active = (
            "SELECT s.*, "
            "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
            f"{_sql_session_last_active('s')} AS last_active "
            "FROM sessions s "
            "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
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
            f"{select_with_last_active}"
            f"{where_sql} "
            "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
            params,
        )]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(
        self,
        source: str = None,
        sources: List[str] = None,
        cwd_prefix: str = None,
        min_message_count: int = 0,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
        exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions, optionally filtered by source.

        Pass ``exclude_children=True`` to count only the conversations that
        ``list_sessions_rich`` surfaces (root + branch/reset sessions), hiding
        sub-agent runs and compression continuations. Use it whenever the count
        is paired with a ``list_sessions_rich`` page (e.g. sidebar "load more"
        totals) so the total matches the number of listable rows — otherwise the
        raw row count is inflated by children and "load more" never settles.

        Pass ``exclude_sources`` to drop whole source classes from the count
        (e.g. ``["cron"]`` so the recents "load more" total matches a
        cron-excluded ``list_sessions_rich`` page and doesn't keep "load more"
        stuck on for buried scheduler sessions).
        """
        where_clauses = []
        params = []

        if exclude_children:
            # Mirror list_sessions_rich's child-exclusion clause exactly so the
            # count lines up with the rows: roots plus user-visible branch/reset
            # children.
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")
        include_sources = [source] if source else list(sources or [])
        if include_sources:
            placeholders = ",".join("?" for _ in include_sources)
            where_clauses.append(f"s.source IN ({placeholders})")
            params.extend(include_sources)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where_clauses.append(clause)
            params.extend(clause_params)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        return self._read_one(f"SELECT COUNT(*) FROM sessions s{where_sql}", params)[0]

    def session_count_ge(self, n: int = 1) -> bool:
        """Check if at least N sessions exist (archived included).

        Short-circuits via LIMIT — much cheaper than ``session_count()``,
        which pays a full index scan for its default ``archived = 0``
        filter (measured 543us vs 4us on a 20k-session DB). Archived
        sessions count: every caller so far asks "has this install ever
        had sessions", and an archived session is still a created one.
        Use this instead of ``session_count() >= n`` when the exact count
        is irrelevant.
        """
        rows = self._read_all("SELECT 1 FROM sessions LIMIT ?", (n,))
        return len(rows) >= n

    def session_count_by_source(
        self,
        *,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
    ) -> Dict[str, int]:
        """Return a ``{source: count}`` dict via a single ``GROUP BY`` query.

        Replaces the O(N) ``list_sessions_rich`` histogram loop with an
        aggregate query. When ``exclude_children`` is False the query uses
        ``idx_sessions_source``; when True, the child-exclusion predicates
        require a full table scan (same as ``session_count`` and
        ``list_sessions_rich``).

        ``exclude_children=True`` mirrors ``list_sessions_rich`` visibility
        (roots + branch/reset sessions, excluding sub-agent runs, delegates,
        and compression continuations) so the source counts match what the
        Sessions page actually lists.
        """
        where_clauses = []
        params: list = []

        if exclude_children:
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._read_ctx() as conn:
            if self._conn is None:
                raise RuntimeError("SessionDB connection is closed")
            rows = conn.execute(
                "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, COUNT(*) AS count "
                f"FROM sessions s{where_sql} "
                "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') "
                "ORDER BY count DESC",
                params,
            ).fetchall()
        return {str(row["source"]): int(row["count"] or 0) for row in rows}


    # =========================================================================
    # Export and cleanup
    # =========================================================================


    def declared_scope_identity(self, session_id: str) -> Tuple[bool, str]:
        """Fork verdict and recorded ``source`` for *session_id*, in ONE read.

        ``agent/prompt_cache_scope.py`` needs both to resolve a host-declared
        conversation scope, and both live on the same ``sessions`` row; asking
        for them separately read that row twice per resolution (@teknium1 on
        #98811).  The marker rules stay here, beside
        :meth:`is_explicit_fork_child`, instead of being re-implemented by the
        caller.

        A missing row is not a fork and has no source, which is the right
        answer before ``_ensure_db_session`` persists it.  This raises whatever
        :meth:`get_session` raises: the caller fails closed on a DB error, and
        merging the two reads cannot weaken that, because the fork check was
        already the first of the two.
        """
        session = self.get_session(session_id)
        if not session:
            return False, ""
        return (
            self._is_explicit_fork_child_row(session),
            str(session.get("source") or "").strip(),
        )


    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove on-disk transcript files for a session.

        Cleans up ``{session_id}.json``, ``{session_id}.jsonl``, and any
        ``request_dump_{session_id}_*.json`` files left by the gateway.
        Silently skips files that don't exist and swallows OSError so a
        filesystem hiccup never blocks a DB operation.
        """
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            p = sessions_dir / f"{session_id}{suffix}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        # request_dump files use session_id as a prefix component
        try:
            for p in sessions_dir.glob(f"request_dump_{session_id}_*.json"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def get_session_delete_targets(self, session_id: str) -> List[str]:
        """Return every session row that :meth:`delete_session` would remove.

        The requested session is first, followed by its recursively discovered
        delegate/subagent children. Branch and compression children are not
        included because deletion preserves them by orphaning their parent
        reference.
        """
        with self._read_ctx() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            ).fetchone()
            if not exists:
                return []
            # Use the borrowed read connection, never self._conn: handing the
            # shared writer connection to a helper here executes on it without
            # self._lock — the same unsynchronized-read class as #99349/#90734.
            delegate_ids = _collect_delegate_child_ids(conn, [session_id])
        return [session_id, *sorted(delegate_ids)]

    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        """Delete a session and all its messages.

        Delegate subagent children (``model_config._delegate_from``) are
        cascade-deleted with the parent so they never resurface in session
        pickers as orphaned rows. Branch / compression children are orphaned
        (``parent_session_id → NULL``) so they remain accessible independently.
        When *sessions_dir* is provided, also removes on-disk transcript
        files (``.json`` / ``.jsonl`` / ``request_dump_*``) for every deleted
        session. When *expected_delete_ids* is provided, deletion proceeds only
        if the parent plus delegate cascade still matches that exact set. This
        lets export-before-delete callers fail closed if a new delegate appears
        after they materialize their archive. The delegate tree is re-walked
        inside the write transaction on purpose (TOCTOU guard); the cost is
        accepted for correctness. Returns True if the session was found and
        deleted.
        """
        removed_delegate_ids: List[str] = []
        expected_ids = (
            set(expected_delete_ids) if expected_delete_ids is not None else None
        )

        def _do(conn):
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            )
            if cursor.fetchone() is None:
                return False
            if expected_ids is not None:
                actual_ids = {
                    session_id,
                    *_collect_delegate_child_ids(conn, [session_id]),
                }
                if actual_ids != expected_ids:
                    return False
            removed_delegate_ids.extend(_delete_delegate_children(conn, [session_id]))
            # Orphan remaining child sessions (branches, etc.) so FK is satisfied.
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._delete_unreferenced_system_prompts(conn)
            return True

        deleted = self._execute_write(_do)
        if deleted:
            for delegate_id in removed_delegate_ids:
                self._remove_session_files(sessions_dir, delegate_id)
            self._remove_session_files(sessions_dir, session_id)
        return bool(deleted)

    def delete_session_if_empty(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        """Delete *session_id* only when it never gained resumable content.

        A session is considered empty when it has no messages and no
        user-assigned title. Used by CLI exit / session-rotation paths so
        immediately-started-and-quit sessions don't pile up in ``/resume``
        and ``hermes sessions list`` output. (Pattern ported from
        google-gemini/gemini-cli#27770.)

        The emptiness check and delete run in one transaction, so a message
        flushed concurrently by another writer can't be lost. Sessions with
        children (delegate subagent runs) are preserved — a parent that
        spawned work is not "empty" even if its own transcript never
        flushed. Returns True if the session was deleted.
        """
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

    def delete_sessions(
        self,
        session_ids: List[str],
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete every session in *session_ids* in a single transaction.

        Backs the dashboard's bulk-select-then-delete flow on the
        sessions page (``POST /api/sessions/bulk-delete``). Mirrors the
        single-session :meth:`delete_session` contract per row:

        * Unknown IDs are silently skipped (no 404) — selection state
          in the UI can race against another tab's delete, and we'd
          rather succeed-on-the-rest than fail-the-whole-batch.
        * Delegate subagent children (``model_config._delegate_from``) are
          cascade-deleted with their parent; branch children are orphaned
          (``parent_session_id → NULL``) so they stay accessible.
        * Messages and the session row both go in one
          ``_execute_write`` call so a partial failure can't leave the
          DB in a "messages gone but session row still there" state.
        * On-disk transcript / ``request_dump_*`` files are cleaned up
          outside the DB transaction when *sessions_dir* is provided,
          matching :meth:`prune_sessions` and
          :meth:`delete_empty_sessions`.

        Returns the count of sessions that actually existed and were
        deleted (may be less than ``len(session_ids)`` if some IDs were
        already gone).
        """
        if not session_ids:
            return 0
        # Dedup + drop any non-string entries up-front. Avoids
        # double-counting in the WHERE-IN list and protects against
        # callers that pass a list with stray ``None`` values.
        unique_ids = list({sid for sid in session_ids if isinstance(sid, str) and sid})
        if not unique_ids:
            return 0

        removed_ids: list[str] = []
        removed_delegate_ids: list[str] = []

        def _do(conn):
            placeholders = ",".join("?" * len(unique_ids))
            # First, filter to IDs that actually exist — we want to
            # return the real deleted count, not the input length.
            cursor = conn.execute(
                f"SELECT id FROM sessions WHERE id IN ({placeholders})",
                unique_ids,
            )
            existing = [row["id"] for row in cursor.fetchall()]
            if not existing:
                return 0

            existing_placeholders = ",".join("?" * len(existing))
            removed_delegate_ids.extend(_delete_delegate_children(conn, existing))
            # Orphan remaining children whose parent is in the kill list so the
            # FK constraint stays satisfied. Pin children whose parent
            # is itself in the kill list rather than NULL-ing parents
            # of survivors — the IN list on ``parent_session_id`` does
            # exactly this.
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({existing_placeholders})",
                existing,
            )
            conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({existing_placeholders})",
                existing,
            )
            conn.execute(
                f"DELETE FROM sessions WHERE id IN ({existing_placeholders})",
                existing,
            )
            self._delete_unreferenced_system_prompts(conn)
            removed_ids.extend(existing)
            return len(existing)

        count = self._execute_write(_do)
        for sid in removed_delegate_ids:
            self._remove_session_files(sessions_dir, sid)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    #: Shared selector for :meth:`count_empty_sessions` and
    #: :meth:`delete_empty_sessions` so the badge and the sweep agree.
    #:
    #: ``message_count`` tracks live (``active = 1``) rows only; rewind
    #: (:meth:`replace_messages` w/ ``archive_dropped``) and in-place
    #: compaction (:meth:`archive_and_compact`) reset it to 0 while keeping
    #: dropped turns on disk as ``active = 0`` — the only recoverable copy
    #: (#70516 / #80763 / #82756). The ``NOT EXISTS`` probe is the authority;
    #: ``message_count = 0`` stays as a cheap prefilter. Same shape as every
    #: other emptiness guard in this module. (#95868)
    _EMPTY_SESSION_WHERE = (
        "message_count = 0 "
        "AND ended_at IS NOT NULL "
        "AND archived = 0 "
        "AND NOT EXISTS ("
        "SELECT 1 FROM messages WHERE messages.session_id = sessions.id"
        ")"
    )

    def count_empty_sessions(self) -> int:
        """Return the count of empty, non-active, non-archived sessions.

        "Empty" = the session holds no message rows at all AND has ended
        (``ended_at IS NOT NULL``) AND is not archived. The ``ended_at``
        guard matches the safety contract used by :meth:`prune_sessions`:
        only ended sessions are candidates for bulk deletion, so a freshly
        spawned session whose first message hasn't landed yet — or one
        held open by the live agent — is never sniped out from under
        the runtime.

        Emptiness is decided by :data:`_EMPTY_SESSION_WHERE` — see that
        constant for why the ``NOT EXISTS`` probe is needed instead of
        trusting ``message_count`` alone.

        Backs the ``GET /api/sessions/empty/count`` endpoint that lets the
        web dashboard hide its "Delete empty" button when there's nothing
        to clean up, and pre-populate the confirm dialog with the actual
        count.
        """
        return self._read_one(f"SELECT COUNT(*) FROM sessions WHERE {self._EMPTY_SESSION_WHERE}")[0]

    def delete_empty_sessions(
        self,
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete every empty, ended, non-archived session.

        Mirrors :meth:`prune_sessions`' transactional shape:

        * Selects candidate IDs first (:data:`_EMPTY_SESSION_WHERE`) so we
          never touch a live session, one the user deliberately archived,
          or one whose transcript survives as soft-archived rows.
        * Orphans any child whose parent is in the kill list — children
          of an empty parent are kept and re-parented to ``NULL`` rather
          than cascade-deleted, matching ``delete_session`` /
          ``prune_sessions`` semantics so branch/subagent transcripts
          survive an inadvertent parent cleanup.
        * Deletes the rows in a single ``_execute_write`` callback so
          the operation is atomic — a partial failure (e.g. SIGKILL
          mid-loop) doesn't leave the DB in a "messages-deleted but
          session-row-still-there" half-state.
        * Cleans up on-disk transcript files (``.json`` / ``.jsonl`` /
          ``request_dump_*``) outside the DB transaction when
          ``sessions_dir`` is provided. Empty sessions don't typically
          have transcript files, but the gateway can leave a stub
          ``request_dump_*`` if it crashed before the first reply —
          so we still sweep, matching ``prune_sessions``.

        Returns the number of sessions deleted.
        """
        removed_ids: list[str] = []

        def _do(conn):
            cursor = conn.execute(
                f"SELECT id FROM sessions WHERE {self._EMPTY_SESSION_WHERE}"
            )
            session_ids = {row["id"] for row in cursor.fetchall()}

            if not session_ids:
                return 0

            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )

            for sid in session_ids:
                # DELETE FROM messages is paranoia — the selector's
                # ``NOT EXISTS`` probe already proved these sessions own no
                # message rows — but a row inserted between the SELECT and
                # this statement would otherwise be left dangling, so we
                # still leave a clean FK state.
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (sid,)
                )
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)

        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count


    def archive_sessions(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> int:
        """Bulk-archive (soft-hide) every session matching the filters.

        Same filter surface as :meth:`prune_sessions`, but instead of deleting
        rows it flips ``archived = 1`` via :meth:`set_session_archived` so
        each match's compression lineage is archived as a unit (an unarchived
        compression root would otherwise resurrect the conversation in
        Desktop's projected list). Nothing is deleted; messages and transcript
        files are untouched. Returns the number of sessions matched.

        ``archived`` defaults to ``False`` here (only select rows not yet
        archived) so repeat runs are idempotent no-ops.
        """
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(
            older_than_days=older_than_days, source=source, **filters
        )
        for row in rows:
            self.set_session_archived(row["id"], True)
        return len(rows)


    # ── Meta key/value (for scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the state_meta key/value store."""
        # Kept on self._lock (not _read_ctx) because callers like
        # fts_rebuild_step read progress before entering a write
        # transaction, and the read-only WAL connection sees only
        # committed data — a pending write transaction's uncommitted
        # meta writes would be invisible.  This is a cheap point lookup,
        # not the convoy bottleneck the read-path split targets.
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def set_meta(
        self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None
    ) -> None:
        """Write a value to the state_meta key/value store.

        When ``cursor`` is provided the write is issued on that cursor
        inline (used during ``_init_schema``, which already holds an open
        transaction — routing through ``_execute_write`` there would nest
        BEGIN IMMEDIATE and deadlock). Otherwise a normal write transaction
        is used.
        """
        if cursor is not None:
            cursor.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            return

        self._write_sql(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def retag_kanban_worker_sessions(self, workspaces_root: str) -> int:
        """Retag legacy kanban worker rows from ``cli`` to ``kanban``.

        Workers used to spawn without ``HERMES_SESSION_SOURCE``, so their runs
        landed as untitled ``cli`` rows and the sidebar rendered one per attempt
        labeled with the worker's own prompt. New workers tag themselves; this
        reclaims the rows already on disk so they drop out of the session lists
        too. Identified by cwd under the board's workspaces root — a path only
        the dispatcher ever runs a session in.

        Gated per workspaces root (``state_meta``) so each board reclaims its
        own rows exactly once. Returns the number of rows retagged.
        """
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
            # Read rowcount before set_meta reuses this cursor for its INSERT,
            # which would otherwise overwrite it with the meta write's count.
            retagged = cursor.rowcount or 0
            self.set_meta(gate, "1", cursor=cursor)
            return retagged

        return self._execute_write(_do)

    def list_meta_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """Return ``[(key, value), ...]`` for state_meta keys with ``prefix``.

        Used by feature stores that persist one row per session under a
        namespaced key (e.g. ``loop:<session_id>``) and need to enumerate
        them across sessions (the gateway's idle /loop wakeup watcher).
        ``prefix`` is matched literally — LIKE wildcards in it are escaped.
        """
        if not prefix:
            return []
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'",
            (escaped + "%",),
        )
        return [(row[0], row[1]) for row in rows]


    # ── Space reclamation ──

    # FTS5 virtual tables whose b-tree segments we merge on optimize. The
    # trigram table is created lazily / may be disabled, and the cjk-bigram
    # table only exists (and is only queryable) when the loadable tokenizer
    # is present — so we probe each before touching it (see optimize_fts).
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")


    def maybe_auto_archive(
        self,
        idle_days: float = 3,
        min_interval_hours: int = 24,
        exclude_pinned: bool = True,
    ) -> Dict[str, Any]:
        """Idempotent auto-archive: soft-hide sessions idle for ``idle_days``.

        Sibling of :meth:`maybe_auto_prune_and_vacuum` but non-destructive —
        it archives (hides) rather than deletes, and ages on last activity
        (see :meth:`archive_stale_sessions`) rather than creation. Records the
        last run in ``state_meta['last_auto_archive']`` so calls within
        ``min_interval_hours`` no-op; safe to call opportunistically (startup
        hooks, or when the Desktop backend lists sessions).

        Never raises. Returns a dict with:
          - ``"skipped"`` (bool) — within min_interval_hours of last run
          - ``"archived"`` (int) — sessions archived this run
          - ``"error"`` (str, optional) — present only on failure
        """
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

            archived = self.archive_stale_sessions(
                idle_days, exclude_pinned=exclude_pinned
            )
            result["archived"] = archived

            # Record even a zero-archive run so we don't re-sweep every call
            # within the interval window.
            self.set_meta("last_auto_archive", str(now))

            if archived > 0:
                logger.info(
                    "state.db auto-archive: archived %d session(s) idle >= %s days",
                    archived,
                    idle_days,
                )
        except Exception as exc:
            logger.warning("state.db auto-archive failed: %s", exc)
            result["error"] = str(exc)

        return result

    # ── Handoff (cross-platform session transfer) ──────────────────────────
    #
    # State machine:
    #   None       — no handoff in flight
    #   "pending"  — CLI requested handoff, gateway hasn't picked it up yet
    #   "running"  — gateway is processing (session switch + synthetic turn)
    #   "completed"— gateway successfully delivered the synthetic turn
    #   "failed"   — gateway hit an error; reason in handoff_error
    #
    # The CLI writes "pending" then poll-waits for terminal state. The gateway
    # watcher transitions pending→running→{completed,failed}.


class AsyncSessionDB:
    """Async door onto SessionDB: offloads each call via asyncio.to_thread so a blocking SQLite call never freezes the event loop. Generic forwarder — the audit confirms no method returns a live cursor/generator."""

    def __init__(self, db: "SessionDB") -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr

        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _offloaded
