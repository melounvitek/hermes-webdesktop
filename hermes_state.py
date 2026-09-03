#!/usr/bin/env python3
"""SQLite state store for Hermes Agent: session metadata, message history, model
config, FTS5 search. WAL mode (concurrent readers + one writer); compression
splits sessions via parent_session_id chains; sessions are source-tagged
('cli', 'telegram', ...). Batch-runner / RL trajectories live elsewhere.
"""

import asyncio
import atexit
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
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from agent.message_sanitization import _sanitize_surrogates
# Known-durable message marker (run_agent keeps a copy: circular import; a test pins them in sync).
from agent.context_compressor import (  # noqa: F401  (re-exported; tests import it from here)
    _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY,
)
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, cast

from hermes_state_common import (  # noqa: F401  (re-exported; tests import from hermes_state)
    AUTO_VACUUM_MIN_FREELIST_RATIO, _FTS_CJK_TRIGGERS, _FTS_TRIGGERS, _LISTABLE_CHILD_SQL,
    _PREVIEW_ELIGIBLE_SQL, _PREVIEW_RAW_SELECT, _RECOVERABLE_END_REASONS,
    _RECOVERABLE_END_REASONS_SQL, _RESET_END_REASONS, _legacy_reset_child_sql, _shape_preview,
    _sql_session_last_active, _sql_session_last_active_by_id, escape_like as _escape_like,
    FTS_CJK_STALE_KEY, FTS_REBUILD_DEFERRAL_KEY, FTS_SQL, FTS_STALE_KEY, FTS_STORAGE_VERSION,
    FTS_TRIGRAM_SQL, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, SCHEMA_SQL, SCHEMA_VERSION,
    stat_db_file_identity as _stat_db_file_identity,
)
from hermes_state_errors import (  # noqa: F401  (re-exported; the historical import path)
    _DELETED_WAL_GENERATION_MSG, _DISK_IO_ERROR_MARKER, _STATE_DB_APPLICATION_ID_OFFSET,
    _STATE_DB_CORRUPT_MSG, _STATE_DB_GENERATION_KEY, _STATE_DB_REPLACED_MSG, PERSISTENCE_ERROR_CAUSES,
    CompressionSessionBusyError, CompressionSessionClosedError, DeletedWalGenerationError,
    SessionCompressionInProgressError, SessionTurnLeaseLostError, StateDbCorruptError,
    StateDbReplacedError, _is_no_more_rows, classify_persistence_error, is_disk_full_error,
    is_malformed_db_error, is_malformed_schema_error, is_transient_sqlite_error,
)
from hermes_state_guard import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _STATE_DB_GUARD_BYPASS_ENV, _in_test_context, _is_production_state_db,
    _process_looks_like_pytest, _real_platform_state_root, _running_under_pytest,
    _set_last_init_error, get_last_init_error,
)
from hermes_state_readpool import (  # noqa: F401  (re-exported; tests import from hermes_state)
    _READ_POOL_MAX, _proc_fd_targets, _process_read_permits, _read_budget_for,
)
from hermes_state_sessions import (  # noqa: F401  (re-exported; the historical import path)
    SessionSessionsMixin, _cwd_prefix_clause, workspace_key,
)
from hermes_state_fts import SessionFtsSetupMixin, load_fts5_cjk_extension  # noqa: F401  (re-exported)
from hermes_state_portability import SessionPortabilityMixin
from hermes_state_telegram import SessionTelegramTopicsMixin
from hermes_state_schema import SessionSchemaMixin
from hermes_state_dbfile import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _canonical_sqlite_path, _concrete_state_db_holder_pids, _connect_tracked_db,
    _is_inactive_orphan_desktop_holder, _looks_like_hermes, _read_proc_cmdline,
    _read_sqlite_application_id, _stat_sqlite_sidecar_identity, _watched_sqlite_sidecar_paths,
    collect_state_db_stats, count_db_holders, is_zeroed_state_db,
    iter_deleted_sqlite_sidecar_holders, quarantine_cross_process_lock, quarantine_zeroed_state_db,
    refuse_deleted_wal_generation,
)
from hermes_state_messages import SessionMessagesMixin
from hermes_state_wal import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    WalUnsupportedError, _WAL_INCOMPAT_MARKERS, _apply_macos_checkpoint_barrier,
    _apply_synchronous_pragma, _database_has_content, _delete_overridden_warned_paths,
    _enforce_macos_synchronous_full, _journal_upgrade_warned_paths, _on_disk_journal_mode,
    _wal_fallback_warned_paths, _wal_reset_bug_warned_paths, _wal_reset_repair_hint,
    apply_database_pragmas, apply_wal_with_fallback, is_sqlite_wal_reset_vulnerable,
    resolve_journal_mode, resolve_synchronous_level, sqlite_source_id,
)
from hermes_state_repair import (  # noqa: F401  (re-exported; tests patch hermes_state.<name>)
    _MAX_MALFORMED_BACKUPS, _MAX_PERSISTENT_REPAIR_ATTEMPTS, _REPAIR_BACKUP_MIN_FREE_BYTES,
    _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND, _backup_content_identity, _backup_db_file,
    _claim_repair_attempt, _connect_repair_durable, _copy_database_snapshot,
    _cross_process_repair_lock, _db_fingerprint, _db_opens_cleanly, _existing_malformed_backups,
    _live_writer_holds_db, _persistent_repair_attempts_exhausted, _probe_journal_mode_for_repair,
    _prune_malformed_backups, _read_repair_ledger, _record_repair_outcome,
    _release_auto_maintenance_lock, _repair_backup_headroom_bytes, _repair_ledger_path,
    _repair_scratch_space_error, _repair_snapshot_timeout_seconds, _repair_state_db_schema_locked,
    _run_repair_strategies, _try_acquire_auto_maintenance_lock, _unlink_db_triple,
    apply_durability_barriers, preflight_db_writability, repair_state_db_schema,
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
    *fallback*. 0 disables the guard. Not cached (load_config_readonly is)."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("sessions") or {}).get(key)
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
    Reclaim on kernel proof only: unstructured/same-process holders (another
    thread's live lease) and any probe doubt keep the lease until TTL expiry
    (PID reuse must never steal a live lease; a wrongly-kept one self-heals)."""
    match = _COMPRESSION_LOCK_HOLDER_PID_RE.search(holder or "")
    pid = int(match.group(1)) if match else 0
    if pid <= 0 or pid == os.getpid():
        return False
    if psutil is not None:
        try:
            return not psutil.pid_exists(pid)  # recycled PIDs read as alive (conservative)
        except Exception:
            return False
    # psutil-less fallback is POSIX-only: on Windows os.kill(pid, 0) maps sig=0 to
    # CTRL_C_EVENT and can kill the target's console group.
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


# Billing buckets that aren't a routable provider identity: a session that
# persisted only one of these (never ran /model) falls back to the config default.
# Shared by session_gateway_runtime and tui_gateway.server so they cannot drift.
_BARE_BILLING_PROVIDERS = frozenset({"auto", "custom"})


T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

# Back off from read-only opens after one fails: not retried per query, but short
# enough that transient fd pressure doesn't strand the read pool.
_READ_OPEN_RETRY_SECONDS = 60.0

# Transient SQLITE_IOERR retry budget for READ-ONLY opens: a WAL writer's
# checkpoint/reset/frame flush surfaces "disk I/O error" to a concurrent mode=ro
# reader for a millisecond-wide window (ro cannot do the -shm recovery). Never
# for writable opens: a writer owns the transition, so an IOERR there is real.
_READ_ONLY_IOERR_RETRY_ATTEMPTS = 3
_READ_ONLY_IOERR_RETRY_BACKOFF_S = 0.05


# Import-time snapshot so _default_db_path() can detect a re-pointed
# DEFAULT_DB_PATH (tests monkeypatch the constant directly).
_IMPORT_DEFAULT_DB_PATH = DEFAULT_DB_PATH


def _default_db_path() -> Path:
    """Default state DB path at CALL time: a re-pointed ``DEFAULT_DB_PATH`` wins,
    else ``get_hermes_home()`` is resolved fresh so a runtime HERMES_HOME redirect
    works regardless of import order."""
    if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH
    return get_hermes_home() / "state.db"


# Live-DB guard knobs live HERE (not in hermes_state_guard): the hermetic conftest
# monkeypatches ``hermes_state._STATE_DB_GUARD_BYPASS`` (escape hatch for
# ``@pytest.mark.live_system_guard_bypass``) and ``_EXTRA_DENY_ROOTS`` (the
# pre-sandbox root, so custom-HERMES_HOME deployments are covered too).
_STATE_DB_GUARD_BYPASS = False
_STATE_DB_GUARD_EXTRA_DENY_ROOTS: Tuple[Path, ...] = ()


def _production_state_roots() -> List[Path]:
    roots = [r for r in (_real_platform_state_root(),) if r is not None]
    for extra in _STATE_DB_GUARD_EXTRA_DENY_ROOTS:
        try:
            roots.append(Path(extra).expanduser().resolve())
        except Exception:
            continue
    return roots


def _ensure_test_isolation(db_path: Path) -> None:
    """Raise RuntimeError before any connection/mkdir/pragma/byte probe when a
    pytest-context process (env OR ancestry) resolves a production DB."""
    if _STATE_DB_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV) or not _in_test_context():
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


# Openings of the background-review harness prompts (agent/background_review.py).
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: Dict[str, Any]) -> bool:
    """Persisted harness prompt (older builds wrote the forked curator's turns
    into real sessions; replaying them hijacks the session)."""
    if not isinstance(msg, dict) or msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(_REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop harness messages and the curator-mode assistant reply that immediately followed each."""
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
    """Assistant tool-call turn whose content is a bare ``[marker]`` (an older
    conversation_loop persisted a local template's marker as the final response)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return False
    content = msg.get("content")
    return isinstance(content, str) and bool(_STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()))


def _strip_stale_tool_call_markers(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Blank stale ``[marker]`` assistant content (replaying it teaches the model
    to keep emitting it); tool_call/result pairing stays intact."""
    repaired = 0
    for msg in filter(_is_stale_tool_call_marker_message, messages):
        msg["content"] = ""
        repaired += 1
    if repaired:
        logger.info(
            "Cleared %d stale tool-call marker message(s) while restoring session (#78148)", repaired,
        )
    return messages


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    """User-facing "session DB unavailable" message with the captured init cause
    (plus a WAL-docs hint for NFS/SMB-style locking failures)."""
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    hint = ""
    if any(marker in cause.lower() for marker in _WAL_INCOMPAT_MARKERS):
        hint = " (state.db may be on NFS/SMB/FUSE/ZFS — see https://www.sqlite.org/wal.html)"
    return f"{prefix}: {cause}{hint}."


# Auto-repair at most once per DB path per process (no repair loops; serialises
# concurrent web_server / gateway opens on the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()

# Cross-process schema-surgery lock timeout (``_repair_attempt_lock`` covers one
# interpreter only); sized for the slowest legitimate holder (VACUUM, multi-GB DB).
_REPAIR_LOCK_TIMEOUT_SECONDS = 120.0
_IS_WINDOWS = sys.platform == "win32"


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


# Process-wide shared SessionDB registry (hermes_state_registry): long-lived
# in-process callers share ONE writer connection per resolved path via
# get_shared_session_db(); one-shots use SessionDB() with their own close().
from hermes_state_registry import (  # noqa: F401  (re-export)
    close_shared_session_dbs, get_shared_session_db, release_or_close,
)


class SessionDB(
    SessionSessionsMixin, SessionFtsSetupMixin, SessionSearchMixin, SessionSchemaMixin,
    SessionPortabilityMixin, SessionTelegramTopicsMixin, SessionCompressionMixin,
    SessionGatewayMixin, SessionMaintenanceMixin, SessionUsageMixin, SessionTitlesMixin,
    SessionMessagesMixin,
):
    """SQLite-backed session storage with FTS5 search. Thread-safe for the gateway
    pattern (many reader threads, one writer via WAL)."""

    # Only these state-owned producers join automatic stale-open reconciliation;
    # messaging/UI sources have their own lifecycle owners; unknown sources fail closed.
    _AUTO_PRUNE_STALE_OPEN_SOURCES: Tuple[str, ...] = (
        "cli", "cron", "kanban", "acp", "api_server", "subagent", "tool",
    )

    # ── Write-contention tuning ──
    # SQLite's deterministic busy handler convoys under many hermes processes, so
    # the SQLite timeout stays short (1s) and retries use random jitter. Patience
    # is TIME-based (a sibling legitimately holds the lock for seconds: checkpoint
    # at close, VACUUM, recovery, an old process's FTS optimize); attempt-counted
    # budgets destroyed turns on a healthy store. Transcript writes (failure
    # aborts the user's turn) get the longer budget; observation-only activity
    # writes sit on the response-critical path and get a sub-second one.
    _WRITE_PATIENCE_S = 20.0
    _TRANSCRIPT_WRITE_PATIENCE_S = 60.0
    _ACTIVITY_WRITE_PATIENCE_S = 0.5
    # A live compression lock gets a short wait (compression publishes in seconds),
    # but the lease is a correctness boundary: a writer still locked out afterwards
    # is refused rather than landing a stale turn in a wedged compression.
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    _WRITE_RETRY_SLOW_AFTER_S = 2.0
    _WRITE_RETRY_SLOW_MIN_S = 0.250  # 250ms
    _WRITE_RETRY_SLOW_MAX_S = 1.000  # 1s
    # PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50
    # Bounded FTS ``'merge'`` (ms of lock each) instead of ``'optimize'`` (9-18s per
    # index on a 10GB DB — longer than a writer's patience); up to
    # _FTS_MERGE_COMMANDS_PER_PASS per index, stopping on no-progress.
    _FTS_MERGE_EVERY_N_WRITES = 1000
    _FTS_MERGE_MAX_PAGES_PER_INDEX = 500
    _FTS_MERGE_COMMANDS_PER_PASS = 4
    # Imports cap lower than exports: an import holds one BEGIN IMMEDIATE.
    _IMPORT_MAX_SESSIONS = 500
    _IMPORT_MAX_MESSAGES_PER_SESSION = 10_000
    _IMPORT_MAX_TOTAL_MESSAGES = 50_000
    _IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
    _IMPORT_MAX_TOTAL_BYTES = 25 * 1024 * 1024
    # Accounting workers retire when idle so a bound-method target can't keep an
    # abandoned SessionDB (and its descriptors) alive.
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
        # BOUNDED pool so they never queue behind writer flushes on self._lock (see
        # _read_ctx); the old per-thread connections pinned fds for the process
        # lifetime and hit EMFILE while staying alive (supervisor never restarted).
        self._read_pool: "queue.LifoQueue[sqlite3.Connection]" = queue.LifoQueue(maxsize=_READ_POOL_MAX)
        # Permits bound PEAK descriptors (the pool bounds only the idle set), shared
        # per DATABASE PATH (_PathReadBudget); acquired non-blocking so a reader
        # without a permit degrades to the writer lock instead of stalling.
        self._read_budget = _read_budget_for(self.db_path)
        self._read_budget.register(self)
        self._read_permits = self._read_budget.permits
        self._read_conns_lock = threading.Lock()
        # Set when close() begins; an in-flight reader then closes its own
        # connection instead of re-populating a pool nobody will drain again.
        self._read_conns_closed = False
        # "read-only opens are failing" backoff: a TIMESTAMP, not a sticky bool —
        # the likeliest trigger is transient EMFILE, and a permanent flag would
        # demote every reader to the writer lock forever.
        self._read_open_failed_at = 0.0
        self._wal_active = False
        self._write_count = 0
        # File identity of the opened state.db, compared on every write so an
        # out-of-band replace cannot limp through in-place surgery. Inode catches
        # mv/new-file; application_id catches cp onto the same path.
        self._db_file_identity: Optional[tuple] = None
        self._db_file_application_id: int = 0
        self._db_sidecar_identity: Dict[str, tuple] = {}
        self._db_replaced = self._db_wal_generation_lost = False
        # Sticky quarantine (see StateDbCorruptError); never cleared.
        self._db_corrupt = False
        self._db_corrupt_reason = ""
        self._fts_usermerge_floor_applied = False  # one-shot usermerge-floor write guard
        self._fts_enabled = self._fts_stale = self._trigram_available = False
        # _fts_cjk_loaded: tokenizer present on the writer connection;
        # _fts_cjk_available: messages_fts_cjk is queryable AND not marked stale.
        self._fts_cjk_loaded = self._fts_cjk_available = self._fts_unavailable_warned = False
        self._conn = None
        # Async token accounting; distinct from self._lock so enqueue/flush
        # bookkeeping never contends with SQLite writes.
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
            # Read-only file/sidecar preflight BEFORE the first connection: an
            # actionable message instead of an opaque "attempt to write a readonly
            # database" from deep inside _init_schema.
            preflight_db_writability(self.db_path, db_label="state.db")
            # Serialize zero-byte check, quarantine, connect and schema commit so
            # concurrent openers don't race the absent-path -> schema-commit window.
            needs_startup_guard = not self.db_path.exists() or is_zeroed_state_db(self.db_path)
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
                self._close_connection_quietly(self._conn)
                if not repair_state_db_schema(self.db_path).get("repaired"):
                    raise
                self._connect_and_init_with_lock_patience()
            # The v23 FTS optimization is OPT-IN (`hermes db optimize`), never
            # auto-started on open (no background worker racing session lifecycle).
            self._ensure_db_file_generation()
            self._record_db_file_identity()
            initialization_complete = True
        except Exception as exc:
            # Surface WHY via /resume and friends (never cleared on success, see
            # _set_last_init_error); callers keep their ``_session_db = None`` path.
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not initialization_complete:
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)

    def _open_read_only(self) -> None:
        """Read-only attach for cross-profile aggregation: no schema init, NO write
        lock (sidebar polling never contends with that profile's backend); the DB
        must exist. FTS flags are probed with SELECTs only, and the connection is
        closed on ANY probe failure (malformed schema raises DatabaseError) so a
        leaked tracked connection cannot block the forensic backup the writable heal takes next."""
        for attempt in range(_READ_ONLY_IOERR_RETRY_ATTEMPTS + 1):
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
                    self._close_connection_quietly(conn)
                    raise
                return
            except sqlite3.OperationalError as ioerr:
                # Transient SQLITE_IOERR window (see _READ_ONLY_IOERR_RETRY_ATTEMPTS);
                # a persistent one exhausts the budget and propagates.
                if attempt >= _READ_ONLY_IOERR_RETRY_ATTEMPTS or _DISK_IO_ERROR_MARKER not in str(ioerr).lower():
                    raise
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)

    def _handle_quarantine_if_zeroed(self, already_locked: bool = False) -> None:
        """Quarantine a zero-byte/headerless state.db so a fresh one can open; if
        quarantine failed, raise the clear message instead of opening the zeroed file."""
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

    def _open_writer_conn(self) -> sqlite3.Connection:
        """Connect + WAL/pragma/tokenizer setup for a writer connection (no schema init).
        Short timeout: application-level jittered retry handles contention, not
        SQLite's busy handler; isolation_level=None: explicit BEGIN IMMEDIATE."""
        conn = _connect_tracked_db(
            str(self.db_path), check_same_thread=False, timeout=1.0, isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            self._wal_active = apply_wal_with_fallback(conn, db_label="state.db") == "wal"
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn)
        except BaseException:
            self._close_connection_quietly(conn)
            raise
        return conn

    def _connect_and_init(self) -> None:
        # Refuse before sqlite3.connect (under the startup lock) so we cannot mint
        # a replacement WAL while a live writer still holds a deleted sidecar inode.
        refuse_deleted_wal_generation(self.db_path)
        self._conn = self._open_writer_conn()
        self._init_schema()

    def _connect_and_init_with_lock_patience(self) -> None:
        """Open + init, waiting out a sibling's write lock with jittered patience:
        _init_schema's DDL runs on a 1s-timeout connection, so a sibling's VACUUM
        or checkpoint used to fail the ENTIRE open and callers disabled
        persistence for the whole run. Non-lock errors propagate immediately."""
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        while True:
            try:
                self._connect_and_init()
                return
            except sqlite3.OperationalError as exc:
                err = str(exc).lower()
                if "locked" not in err and "busy" not in err:
                    raise
                self._close_connection_quietly(self._conn)
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
        return it to self._read_pool). WAL only: WAL readers never block on the
        writer, so reads skip self._lock; under DELETE journal mode (NFS fallback)
        readers hit SQLITE_BUSY storms, so the legacy locked path stays. Autocommit
        reads see everything committed so far (read-your-writes for flush-then-search)."""
        if not self._wal_active or self.read_only:
            return None
        with self._read_conns_lock:
            if self._read_conns_closed:
                return None
            if (
                self._read_open_failed_at
                and time.monotonic() - self._read_open_failed_at < _READ_OPEN_RETRY_SECONDS
            ):
                return None
        # Permit BEFORE the open: openers race for permits, not descriptors.
        if not self._read_budget.acquire(self):
            logger.debug(
                "read pool at capacity (%d) for %s; serving this read from the "
                "locked writer connection", _READ_POOL_MAX, self.db_path,
            )
            return None
        conn = None  # bound before the try so the handlers can close a half-open one
        try:
            # check_same_thread=False: pooled connections are borrowed by whichever
            # thread reads next; exclusive ownership is enforced by pool checkout.
            conn = _connect_tracked_db(
                f"file:{self.db_path}?mode=ro", tracking_path=self.db_path, uri=True,
                check_same_thread=False, timeout=5.0, isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            apply_database_pragmas(conn, db_label="state.db")
            if self._fts_cjk_loaded:  # registers in the connection, not the file: ro is fine
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
        permit); never pulls a connection out from under a live reader."""
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
        """Close a pooled read connection and release its permit. A failing close
        leaks a tracked fd (logged, never swallowed) but still releases the permit:
        withholding it would permanently narrow the read path. Pairs with
        _get_read_conn(); over-releasing the BoundedSemaphore raises ValueError."""
        try:
            conn.close()
        except Exception as exc:
            logger.warning("read-conn close failed for %s: %s", self.db_path, exc)
        finally:
            self._read_budget.release()

    def _checkout_read_conn(self) -> Optional[sqlite3.Connection]:
        """Borrow a read connection, opening on a miss; None when the read path is
        unavailable. A pool hit costs no permit (the connection already holds one)."""
        if not self._wal_active or self.read_only:
            return None
        try:
            return self._read_pool.get_nowait()
        except queue.Empty:
            return self._get_read_conn()

    @contextmanager
    def _read_ctx(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for read-only statements: a pooled read-only
        connection with NO lock under WAL; otherwise (non-WAL, open failure,
        ceiling reached) the writer connection under self._lock — deliberate
        degradation: slower beats EMFILE, which the supervisor cannot see."""
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
                    # close() drained the pool (or queue.Full: unreachable while
                    # permits == maxsize, load-bearing if they drift): surplus.
                    self._close_read_conn(conn)
            return
        with self._lock:
            if self._conn is None:  # close() raced a still-unwinding reader
                self._reopen_after_close_locked(context="read")
            yield cast(sqlite3.Connection, self._conn)

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        """Reopen the writer after ``close()`` raced a live caller (a teardown owner
        set ``_conn = None`` while a worker still had a transcript flush to land).
        Loud (WARNING) and bounded (only after an explicit close()). Caller holds
        ``self._lock``. No _init_schema: no DDL races with siblings during teardown."""
        if self.read_only:
            raise sqlite3.ProgrammingError(
                f"SessionDB for {self.db_path} was closed (read-only handle); "
                f"cannot serve a {context} after close()"
            )
        # A reopen resolves the PATH again: a replaced file would be written through
        # stale WAL/shm assumptions; a quarantined handle must never hand a fresh
        # connection (and its close-time checkpoint) to a damaged file.
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
            "flight — reopening (teardown/worker race, #94736)", self.db_path, context,
        )
        try:
            self._conn = self._open_writer_conn()
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen failed: {exc}"
            ) from exc

    def _execute_write(
        self, fn: Callable[[sqlite3.Connection], T], patience_s: Optional[float] = None,
    ) -> T:
        """Run *fn(conn)* inside BEGIN IMMEDIATE with jittered lock retry; commit
        is handled here (callers must not commit). Returns *fn*'s result.
        BEGIN IMMEDIATE takes the WAL write lock up front so contention surfaces
        immediately; on locked/busy the Python lock is released, a jitter slept,
        and the WHOLE callback retried — *fn* must stay idempotent under retry."""
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        # Set on the first compression-busy collision: the short wait is measured from then.
        compression_deadline: Optional[float] = None
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself (callback not
        # run: nothing replayed). Once it has started, an IOERR leaves settlement
        # unknown and must propagate — this helper owns non-idempotent mutations.
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
                # Transient (see _COMPRESSION_BUSY_WAIT_S): without a wait, a steer
                # landing mid-compression aborts the turn.
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
                    # Retry on the SAME connection: close()+reopen cancels this
                    # process's POSIX locks for every sibling (howtocorrupt §2.2).
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
                # Corrupt FTS shadow tables fail every write via the sync triggers
                # while canonical rows are intact: detach the derived indexes
                # atomically and retry (never rebuild from the live write path).
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
            # Header 0 = WAL not yet checkpointed, not a replace; any real
            # replacement (a copied Hermes DB minted its own id) is nonzero.
            if disk_app and disk_app != recorded_app:
                return True
        return False

    def _halt_db_replaced(self) -> None:
        """Stop writes and raise; do not run in-file repair on a new generation."""
        self._db_replaced = True
        logger.error(_STATE_DB_REPLACED_MSG)
        raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this handle opened is gone. Recorded
        generation: pure stat (no /proc walk on healthy writes). Empty identity
        (WAL appeared after open, or cleared by a clean close()): probe
        /proc/self/fd for deleted sidecars and adopt the current ones once clean."""
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
        """Best-effort SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python 3.12+): sqlite3's
        close() otherwise runs the internal last-connection checkpoint that wrote
        the incident's pages under wrong page numbers (see StateDbCorruptError).
        <3.12 has no setconfig; the residual checkpoint only carries
        pre-quarantine committed frames, which is tolerable."""
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if flag is None or setconfig is None:
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

    def _foreign_state_db_holders(self) -> List[Tuple[int, str]]:
        """Foreign processes holding this DB or its WAL sidecars: automatic FTS
        repair must not run while another process is attached (a sidecar reset
        under it splits the WAL inodes). A scan failure is reported as an unknown
        holder — skipping optional maintenance beats assuming quiescence."""
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
        # Linux: readlink /proc/<pid>/fd directly; psutil.open_files() stats the
        # literal path and silently drops "state.db-wal (deleted)" entries.
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
                        # Unreadable fd table (other user); cmdline is world-readable:
                        # flag only uninspectable holders that look like Hermes.
                        cmdline = _read_proc_cmdline(pid)
                        if cmdline is not None and _looks_like_hermes(cmdline):
                            holders.append((pid, f"uninspectable holder: {cmdline[:80]}"))
                        continue
                    holders.extend((pid, t) for t in targets if _canonical_sqlite_path(t) in watched)
            except Exception as exc:
                return self._foreign_holder_scan_failed(holders, exc)
            return holders
        # macOS / BSD: psutil.open_files() (no "(deleted)" suffix convention there;
        # AccessDenied -> None -> empty iteration is acceptable on macOS).
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
            "deferring automatic FTS maintenance: %s", exc,
        )
        return holders or [(-1, f"open-file scan failed: {exc}")]

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint; never raises. PASSIVE never blocks
        writers; the old TRUNCATE strategy corrupted B-trees on 65K+ page
        databases under exclusive-lock I/O pressure."""
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
        """``with SessionDB(path) as db:`` closes on exit; owners must release
        deterministically ("eventually after a GC cycle" is not a release policy)."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the handle; never suppress the caller's exception."""
        self.close()
        return False

    def close(self):
        """Drain queued token deltas, then a PASSIVE checkpoint on writable handles
        (NOT TRUNCATE: a full WAL reset races the gateway's live writer, tearing
        B-tree pages). A registry-shared instance RELEASES one refcount instead."""
        if self._shared_registry_owned:
            from hermes_state_registry import release
            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Closed flag first: an in-flight reader then closes its own connection.
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
                # A clean close lets SQLite unlink the sidecars (a legitimate end of
                # the generation, not a split): a teardown-race reopen must re-adopt.
                self._db_sidecar_identity = {}

    def __del__(self) -> None:
        """Safety net: close() if the caller forgot. Attribute access stays
        guarded: module teardown order is undefined."""
        if self.__dict__.get("_conn") is None:
            return
        try:
            self.close()
        except Exception:
            pass

    # ── Async token accounting (SessionUsageMixin) ──
    # queue_token_counts() reduces the critical path to a deque append; a
    # single-writer thread applies deltas in order, coalescing consecutive
    # same-route deltas (route fields must be EQUAL to merge so the merged UPDATE
    # equals applying them sequentially). Exact readers call flush_token_counts().
    _TOKEN_DELTA_SUM_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_write_tokens", "reasoning_tokens", "api_call_count",
    )
    _TOKEN_DELTA_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")
    _TOKEN_DELTA_ROUTE_FIELDS = (
        "model", "cost_status", "cost_source", "pricing_version",
        "billing_provider", "billing_base_url", "billing_mode",
    )

    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority: auto-titling may only
    # replace a strictly lower-authority title, so ``derived`` upgrades to
    # ``llm`` exactly once and nothing generated clobbers a user-typed name.
    TITLE_SOURCE_DERIVED = "derived"
    TITLE_SOURCE_LLM = "llm"
    TITLE_SOURCE_USER = "user"
    _TITLE_SOURCE_RANK = {TITLE_SOURCE_DERIVED: 0, TITLE_SOURCE_LLM: 1, TITLE_SOURCE_USER: 2}

    # Bot Mode's canonical chat is resolved by exact-title lookup: the title IS the
    # identity, so _set_session_title refuses renames of a hidden row holding it.
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

    # ── Message storage constants (SessionMessagesMixin) ──
    # Prefix marking JSON-encoded structured content; NUL cannot collide with text.
    _CONTENT_JSON_PREFIX = "\x00json:"
    #: Reactions live inside ``display_metadata`` so they survive row rewrites.
    REACTIONS_METADATA_KEY = "reactions"
    # Columns every conversation projection decodes; ``active`` rides along so a
    # display read can split compaction-archived rows without a second query.
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, active, api_content, display_kind, display_metadata"
    )

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
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'",
            (_escape_like(prefix) + "%",),
        )
        return [(row[0], row[1]) for row in rows]


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
