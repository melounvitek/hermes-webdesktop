"""SQLite connection lifecycle for the Kanban DB: open/configure, cross-process init and dispatch-tick locks, WAL checkpoints, corruption detection + quarantine + repair, additive migrations and the busy-retrying ``write_txn`` boundary.

Split out of ``hermes_cli.kanban_db``; every name is re-exported there, and
origin-resident helpers are reached late-bound via ``_kb`` so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import random
import re
import secrets
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing
from pathlib import Path
from typing import Any
from typing import Optional

# Log-record parity with the origin module.
_log = logging.getLogger("hermes_cli.kanban_db")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()


_INIT_LOCK = threading.RLock()


_SQLITE_HEADER = b"SQLite format 3\x00"


DEFAULT_BUSY_TIMEOUT_MS = 120_000


# Maximum number of ``<db>.corrupt.<hash>.bak`` quarantine files retained per
# board DB. Content-addressing already dedupes identical corrupt bytes, but
# repeatedly-mutating corruption (partial repairs, further damage between
# dispatcher retries) mints a new fingerprint each time; without a cap a user
# accumulated 124 backups. Oldest-by-mtime files beyond the cap are pruned
# right after each new backup is created.
_CORRUPT_BACKUP_RETENTION = 10


# Bounded acquire for the cross-process init lock (#36644). The original bare
# blocking flock had no timeout, so a wedged holder blocked the dispatcher's
# next-tick connect forever. We retry a non-blocking acquire up to this
# deadline, polling at this interval, then proceed without the cross-process
# lock (the in-process _INIT_LOCK + idempotent init remain the backstop).
_INIT_LOCK_TIMEOUT_SECONDS = 10.0


_INIT_LOCK_POLL_SECONDS = 0.05


def _resolve_busy_timeout_ms() -> int:
    """Return the SQLite busy timeout for Kanban connections.

    Kanban is the shared cross-profile dispatch bus, so worker stampedes are
    expected.  A long busy timeout lets SQLite serialize writers via WAL rather
    than surfacing transient ``database is locked`` failures during bursts.
    """
    return _kb._env_int("HERMES_KANBAN_BUSY_TIMEOUT_MS", DEFAULT_BUSY_TIMEOUT_MS, minimum=1)


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a Kanban SQLite connection with consistent lock waiting.

    Uses ``connect_tracked`` so the live-connection registry knows this file
    is open: while it is, byte-level probes of the same file are refused,
    because an ``open()``/``close()`` would cancel this process's POSIX
    advisory locks on the database (see ``hermes_cli.sqlite_safe_read``).
    The registration is released automatically when the connection closes.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    busy_timeout_ms = _resolve_busy_timeout_ms()
    conn = connect_tracked(
        path,
        connect_fn=sqlite3.connect,
        isolation_level=None,
        timeout=busy_timeout_ms / 1000.0,
    )
    try:
        # ``sqlite3.connect(timeout=...)`` normally maps to busy_timeout, but set
        # the PRAGMA explicitly so it is observable and survives future wrapper
        # changes. Parameter binding is not supported for PRAGMA assignments.
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    except BaseException:
        # A half-open connection abandoned here would leak its fd AND leave a
        # stale entry in the connect_tracked live-connection registry (which
        # only clears on close), permanently blocking byte-level probes of
        # this database file. Close before re-raising.
        try:
            conn.close()
        except Exception:
            pass
        raise
    return conn


def _try_lock_nb(handle) -> bool:
    """One non-blocking exclusive lock attempt on ``handle``; False when held elsewhere.

    Windows uses a 1-byte ``msvcrt.locking`` range at offset 0, POSIX ``flock``.
    """
    if _kb._IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def _unlock(handle) -> None:
    """Release a lock taken by :func:`_try_lock_nb` (same byte range / flock)."""
    if _kb._IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _cross_process_init_lock(path: Path):
    """Serialize first-connect WAL/schema/integrity setup across processes.

    ``_INIT_LOCK`` only protects threads inside one Python process. During a
    dispatcher burst, many worker processes can all hit a fresh/legacy board at
    once and each process has an empty ``_INITIALIZED_PATHS`` cache. This file
    lock keeps header validation, integrity probing, WAL activation, and
    additive migrations single-file/single-writer across the whole host while
    leaving normal post-init DB usage concurrent under SQLite WAL.

    The acquire is **bounded**: a blocking ``flock`` let one stalled holder (or
    a stale lock from a wedged worker) hang every other ``connect()`` — the
    gateway dispatcher's next tick included — with no traceback. We retry a
    non-blocking acquire until a deadline, then WARN and proceed WITHOUT the
    lock. Safe because ``_INIT_LOCK`` still serializes same-process threads
    and init is idempotent (``CREATE TABLE IF NOT EXISTS`` + additive
    migrations): two racing first-inits mean redundant work, not corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".init.lock")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + _kb._INIT_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                acquired = _try_lock_nb(handle)
            except OSError:
                acquired = False
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(_INIT_LOCK_POLL_SECONDS)
        if not acquired:
            _kb._log.warning(
                "kanban init lock for %s not acquired within %.0fs — proceeding "
                "without the cross-process lock (in-process lock + idempotent "
                "init are the correctness backstop). A stuck holder is no longer "
                "able to block this connect indefinitely (#36644).",
                lock_path, _kb._INIT_LOCK_TIMEOUT_SECONDS,
            )
        yield
    finally:
        try:
            if acquired:
                _unlock(handle)
        finally:
            handle.close()


@contextlib.contextmanager
def _dispatch_tick_lock(db_path: Path):
    """Non-blocking single-writer guard around one dispatcher tick.

    Yields ``True`` when this process holds the board's dispatch lock and
    may proceed with the tick, or ``False`` when another process already
    holds it (the caller should skip the tick this round).

    An orphan gateway (``gateway run --replace`` / restart on a systemd or
    launchd host escaping the service cgroup) becomes a second long-lived
    writer on the same ``kanban.db``; two dispatchers both pass SQLite
    ``busy_timeout`` and race on WAL frames — the root cause of multi-writer
    corruption. ``_guard_supervised_gateway_conflict`` blocks the common way
    an orphan is born; this lock is the defense-in-depth regardless of how
    the second dispatcher got there.

    **Non-blocking** on purpose: the gateway's async watcher must never stall
    on a held lock. A loser skips its tick (the winner is making progress on
    the same board) and tries again next interval.

    Board-scoped: the lock file is a ``.dispatch.lock`` sibling of the
    board's ``kanban.db``, so unrelated boards tick independently. On
    platforms without ``fcntl``/``msvcrt`` the guard degrades to a no-op
    (yields ``True``) — single-writer enforcement is best-effort and the
    orphan-dispatcher scenario is specific to POSIX service managers.
    """
    lock_path = db_path.with_name(db_path.name + ".dispatch.lock")
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            acquired = _try_lock_nb(handle)
        except (OSError, AttributeError):
            acquired = False
    except OSError:
        # Could not even open the lock file (permissions, read-only FS).
        # Degrade to a no-op so a probe failure never blocks dispatch.
        acquired = True
        handle = None
    try:
        yield acquired
    finally:
        if handle is not None:
            try:
                if acquired:
                    _unlock(handle)
            except (OSError, AttributeError):
                pass
            finally:
                handle.close()


# Periodic WAL checkpoint state for the dispatcher tick path. The kanban
# connections run with ``wal_autocheckpoint=100``, but a passive
# autocheckpoint can be starved on a busy multi-process board (any reader
# with an open snapshot blocks the WAL reset), letting the -wal file grow
# between gateway restarts. Once per coarse interval the dispatcher issues
# an explicit ``wal_checkpoint(PASSIVE)``.
#
# PASSIVE, not TRUNCATE (same class fix as the state.db checkpoints,
# #45383/#80255/#44795): the dispatch flock only makes the dispatcher the
# sole *dispatcher* — CLI kanban commands in other processes write to the
# same board without taking that flock, so a TRUNCATE here races live
# writers exactly like the state.db close() path did. PASSIVE never takes
# the exclusive checkpoint lock; the WAL file size is instead bounded by
# ``journal_size_limit`` (set at connection init) which truncates the file
# on the writer's natural post-checkpoint reset.
# Best-effort: a busy/locked checkpoint is logged at DEBUG and retried next
# interval. Keyed per resolved DB path so multi-board dispatchers checkpoint
# each board on its own clock.
_WAL_CHECKPOINT_INTERVAL_SECONDS = 300.0


_LAST_WAL_CHECKPOINT: dict[str, float] = {}


_WAL_CHECKPOINT_LOCK = threading.Lock()


def _maybe_checkpoint_wal(conn: sqlite3.Connection, db_path: Path) -> None:
    """Run ``PRAGMA wal_checkpoint(PASSIVE)`` at a coarse interval.

    Called from the dispatcher tick while the board's dispatch lock is
    held. No-ops (cheaply) until ``_WAL_CHECKPOINT_INTERVAL_SECONDS`` has
    elapsed since this process last checkpointed this board. Never raises:
    the checkpoint is pure hygiene and must not fail a dispatch tick.
    """
    try:
        key = str(db_path.resolve())
    except OSError:
        key = str(db_path)
    now = time.monotonic()
    with _WAL_CHECKPOINT_LOCK:
        last = _kb._LAST_WAL_CHECKPOINT.get(key)
        if last is not None and (now - last) < _WAL_CHECKPOINT_INTERVAL_SECONDS:
            return
        # Claim the slot before doing the work so concurrent ticks (other
        # threads in this process) don't double-checkpoint on the boundary.
        _kb._LAST_WAL_CHECKPOINT[key] = now
    try:
        row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        _kb._log.debug(
            "kanban WAL checkpoint (PASSIVE) on %s -> %s "
            "(busy, wal_frames, checkpointed_frames)",
            key, tuple(row) if row is not None else None,
        )
    except sqlite3.Error as exc:
        _kb._log.debug("kanban WAL checkpoint on %s skipped: %s", key, exc)


def _looks_like_tls_record_at(data: bytes, offset: int) -> bool:
    """Return True for a TLS record header at ``data[offset:]``."""
    if len(data) < offset + 5:
        return False
    content_type = data[offset]
    major = data[offset + 1]
    minor = data[offset + 2]
    length = int.from_bytes(data[offset + 3:offset + 5], "big")
    return (
        content_type in {0x14, 0x15, 0x16, 0x17}
        and major == 0x03
        and minor in {0x00, 0x01, 0x02, 0x03, 0x04}
        and 0 < length <= 18432
    )


def _validate_sqlite_header(path: Path) -> None:
    """Fail early with an actionable error for non-SQLite Kanban DB files.

    ``sqlite3.connect()`` creates missing and zero-byte files, so those are
    allowed. Existing non-empty files must have the SQLite header before we
    hand them to SQLite/WAL setup. This keeps corrupted page-0 failures from
    being collapsed into a generic PRAGMA error and lets the gateway's corrupt
    board handling identify the board by fingerprint.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.st_size == 0:
        return
    # Byte-level probe, so it must run BEFORE any connection to this path
    # exists (connect() calls it under the init lock, ahead of _sqlite_connect).
    # read_header_bytes_preopen refuses once a connection is live, because the
    # close() would cancel this process's POSIX locks on the file.
    from hermes_cli.sqlite_safe_read import read_header_bytes_preopen

    head = read_header_bytes_preopen(path, length=64)
    if head is None:
        return
    if head.startswith(_SQLITE_HEADER):
        return
    signature = ""
    if head.startswith(b"SQLit") and _looks_like_tls_record_at(head, 5):
        signature = " (TLS record header detected at byte offset 5)"
    elif _looks_like_tls_record_at(head, 0):
        signature = " (TLS record header detected at byte offset 0)"
    raise sqlite3.DatabaseError(
        "file is not a database: invalid SQLite header for "
        f"{path}{signature}; first_32={head[:32].hex(' ')}"
    )


class KanbanDbCorruptError(RuntimeError):
    """Raised when an existing kanban DB file fails integrity checks.

    Fail-closed guard against silent recreation of a corrupt board file,
    which would otherwise destroy the user's tasks. Carries both the
    original path and the timestamped backup we made before refusing.
    """

    def __init__(self, db_path: Path, backup_path: Optional[Path], reason: str):
        self.db_path = db_path
        self.backup_path = backup_path
        self.reason = reason
        backup_str = str(backup_path) if backup_path is not None else "<backup failed>"
        super().__init__(
            f"Refusing to open corrupt kanban DB at {db_path}: {reason}. "
            f"Original preserved; backup at {backup_str}."
        )


def _prune_corrupt_backups(
    parent: Path, base_name: str, keep: Optional[Path] = None,
) -> None:
    """Cap the number of retained ``<db>.corrupt.<hash>.bak`` files.

    Content-addressed backups dedupe identical corrupt bytes, but a board
    whose file keeps changing between corruption events (partial repairs,
    ongoing damage, fleets of retrying dispatchers) can still accumulate
    backups without bound — a user reported 124 of them. After creating a
    new backup we keep only the ``_CORRUPT_BACKUP_RETENTION`` most recent
    (by mtime) and delete the rest, including their copied ``-wal``/``-shm``
    sidecars. ``keep`` (the just-created backup) is never pruned regardless
    of its mtime — ``shutil.copy2`` preserves the source file's timestamp,
    which may be older than existing backups. Best-effort: prune failures
    never mask the corruption error the caller is about to raise.
    """
    try:
        backups = [
            candidate
            for candidate in parent.glob(f"{base_name}.corrupt.*.bak")
            if candidate.is_file() and candidate != keep
        ]
    except OSError:
        return
    budget = _kb._CORRUPT_BACKUP_RETENTION - (1 if keep is not None else 0)
    budget = max(budget, 0)
    if len(backups) <= budget:
        return

    def _mtime(item: Path) -> float:
        try:
            return item.stat().st_mtime
        except OSError:
            return 0.0

    backups.sort(key=_mtime, reverse=True)
    for stale in backups[budget:]:
        for victim in (
            stale,
            stale.with_name(stale.name + "-wal"),
            stale.with_name(stale.name + "-shm"),
        ):
            try:
                victim.unlink(missing_ok=True)
            except OSError:
                pass


def _backup_corrupt_db(path: Path) -> Optional[Path]:
    """Copy a corrupt DB (and its WAL/SHM sidecars) to a content-addressed backup.

    The backup filename is deterministic in the main DB's sha256, so repeated
    quarantines of the same corrupt bytes (gateway restarts, dispatcher retries,
    multi-profile fleets all hitting the same shared DB) reuse one backup
    instead of amplifying disk usage by N. If the corrupt bytes actually
    change between attempts — e.g. a partial repair or further damage — the
    fingerprint changes and a separate backup is preserved.

    Returns the backup path of the main DB file, or ``None`` if the copy
    itself failed (the caller still raises loudly in that case).

    Writes are confined to the original DB's parent directory. The backup
    basename is derived purely from ``path.name`` and a content hash, never
    from caller-supplied directory segments — no traversal is possible.
    """
    # Resolve once and pin the parent so subsequent path operations cannot
    # escape it. ``Path.resolve()`` collapses any ``..`` segments and
    # symlinks, and we only ever write inside ``parent``.
    resolved = path.resolve()
    parent = resolved.parent
    base_name = resolved.name  # basename only
    # This reads the whole DB file to fingerprint it. That is a close()-on-a-
    # database-file hazard (it cancels this process's POSIX advisory locks --
    # see hermes_cli.sqlite_safe_read), so it must only run once the board has
    # been taken out of service. Every caller reaches here on the corrupt/
    # quarantine path after closing its probe connection, but another
    # SessionDB/kanban connection elsewhere in the process would still be at
    # risk -- so REFUSE rather than warn-and-proceed. Losing a forensic copy
    # is strictly better than corrupting the live database we are trying to
    # rescue.
    from hermes_cli.sqlite_safe_read import has_live_connection

    if has_live_connection(resolved):
        _kb._log.error(
            "refusing to quarantine %s: a connection to it is still open in "
            "this process, and fingerprinting the file would cancel that "
            "connection's POSIX locks. Close all connections first.",
            resolved,
        )
        return None
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    token = digest.hexdigest()[:16]
    candidate = parent / f"{base_name}.corrupt.{token}.bak"
    # Defensive: candidate must still be inside parent after construction.
    if candidate.parent != parent:
        return None
    if not candidate.exists():
        try:
            shutil.copy2(resolved, candidate)
        except OSError:
            return None
        # A NEW backup landed on disk — enforce the retention cap so
        # mutating-corruption loops can't accumulate quarantines forever.
        _prune_corrupt_backups(parent, base_name, keep=candidate)
    for suffix in ("-wal", "-shm"):
        sidecar = parent / (base_name + suffix)
        if sidecar.parent != parent or not sidecar.exists():
            continue
        sidecar_backup = parent / (candidate.name + suffix)
        if sidecar_backup.parent != parent or sidecar_backup.exists():
            continue
        try:
            shutil.copy2(sidecar, sidecar_backup)
        except OSError:
            pass
    return candidate


# Repairable integrity_check error classes. Both shapes are *index-scoped*:
# the table b-tree is intact and only a secondary index disagrees with it,
# which REINDEX rebuilds losslessly from the table data. The index name is
# parsed generically from the message — no hardcoded index list. Any other
# integrity_check message (page corruption, "database disk image is
# malformed", freelist damage, …) is NOT repairable this way and keeps the
# fail-closed behavior.
_REPAIRABLE_INDEX_ERROR_PATTERNS = (
    re.compile(r"^wrong # of entries in index (?P<index>.+)$"),
    re.compile(r"^row \d+ missing from index (?P<index>.+)$"),
)


def _integrity_messages_ok(messages: list[str]) -> bool:
    """True iff ``PRAGMA integrity_check`` output is the single ``ok`` row."""
    return len(messages) == 1 and messages[0].strip().lower() == "ok"


def _run_integrity_check(conn: sqlite3.Connection) -> list[str]:
    """Return all ``PRAGMA integrity_check`` message rows as strings."""
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    return [str(row[0]) for row in rows if row is not None and row[0] is not None]


def _probe_integrity(path: Path) -> list[str]:
    """Open ``path`` read/write and return its ``integrity_check`` messages.

    Opening read/write lets SQLite recover or checkpoint a healthy WAL /
    hot-journal DB before we judge it. ``sqlite3.OperationalError`` (locked,
    busy, transient IO) propagates raw — it is not corruption.
    """
    probe = _sqlite_connect(path)
    try:
        return _run_integrity_check(probe)
    finally:
        probe.close()


def _repairable_index_names(messages: list[str]) -> Optional[list[str]]:
    """Return the distinct index names iff EVERY message is index-repairable.

    ``None`` when any line falls outside the repairable index-class errors
    (or when there are no messages at all) — the caller must then fail
    closed exactly as before. Order of first appearance is preserved so the
    REINDEX pass is deterministic.
    """
    names: list[str] = []
    saw_any = False
    for raw in messages:
        message = (raw or "").strip()
        if not message:
            continue
        for pattern in _REPAIRABLE_INDEX_ERROR_PATTERNS:
            match = pattern.match(message)
            if match:
                break
        else:
            return None
        saw_any = True
        name = match.group("index").strip()
        if name and name not in names:
            names.append(name)
    if not saw_any or not names:
        return None
    return names


def _attempt_index_reindex_repair(
    path: Path, index_names: list[str],
) -> tuple[bool, list[str]]:
    """REINDEX the named indexes, then re-run ``PRAGMA integrity_check``.

    Tries a per-index ``REINDEX "<name>"`` first (cheapest, most targeted);
    if any per-index statement fails — e.g. the parsed name does not resolve
    because integrity_check reported an internal/auto index — falls back to
    a bare ``REINDEX`` of the whole database. Returns
    ``(clean, post_repair_messages)``; never raises. Callers must hold the
    board's cross-process init flock so no other process connects mid-repair.
    """
    try:
        conn = _sqlite_connect(path)
    except sqlite3.Error as exc:
        return False, [f"could not reopen for REINDEX: {exc}"]
    try:
        try:
            for name in index_names:
                escaped = name.replace('"', '""')
                conn.execute(f'REINDEX "{escaped}"')
        except sqlite3.Error:
            # Per-index rebuild failed (unresolvable parsed name, auto
            # index, …) — bare REINDEX rebuilds every index in the DB.
            conn.execute("REINDEX")
        messages = _run_integrity_check(conn)
    except sqlite3.Error as exc:
        return False, [f"REINDEX failed: {exc}"]
    finally:
        conn.close()
    return _integrity_messages_ok(messages), messages


def _guard_existing_db_is_healthy(path: Path) -> None:
    """Run ``PRAGMA integrity_check`` on an existing non-empty DB file.

    Opens the probe in read/write mode so SQLite can recover or
    checkpoint a healthy WAL/hot-journal DB before we declare it
    corrupt.

    **Narrow auto-repair:** when the integrity failure consists *only* of
    index-scoped errors (``wrong # of entries in index <name>`` / ``row N
    missing from index <name>``), the table b-trees are intact and REINDEX
    rebuilds the damaged indexes losslessly. In that case we take the
    corrupt backup FIRST (same content-addressed quarantine as the
    fail-closed path), run REINDEX under the caller-held init flock,
    re-run ``integrity_check``, and proceed only if it comes back clean.
    Anything else — page corruption, ``malformed`` images, a REINDEX that
    does not produce a clean re-check — fails closed exactly as before:
    copy the file (and any WAL/SHM sidecars) to a backup and raise
    :class:`KanbanDbCorruptError` so callers cannot silently recreate the
    schema on top of a damaged DB.

    Transient lock/busy errors (``sqlite3.OperationalError``) are NOT
    treated as corruption; they propagate raw so the caller sees a
    normal lock failure and no spurious ``.corrupt`` backup is made.

    No-op for missing files, zero-byte files (treated as fresh), and
    paths already proven healthy this process (cache hit).

    Path-trust note: ``path`` arrives via :func:`connect`, which itself
    resolves it from an explicit ``db_path`` argument, the
    :func:`kanban_db_path` env-var chain, or the kanban-home default —
    all sources Hermes treats as user-controlled-but-trusted on the
    user's own machine. We additionally resolve the path here and
    confine all filesystem writes to its parent directory so any
    accidental ``..`` segments are collapsed before any I/O happens.
    """
    # Resolve before any I/O. ``Path.resolve()`` normalizes ``..`` and
    # symlinks, giving us a canonical path whose parent dir we can pin.
    try:
        resolved = path.resolve()
    except OSError:
        return
    try:
        if not resolved.exists() or resolved.stat().st_size == 0:
            return
    except OSError:
        return
    if str(resolved) in _INITIALIZED_PATHS:
        return
    reason: Optional[str] = None
    messages: list[str] = []
    try:
        messages = _probe_integrity(resolved)
        if not _integrity_messages_ok(messages):
            reason = (
                f"integrity_check returned "
                f"{messages[0] if messages else '<no row>'!r}"
            )
    except sqlite3.OperationalError:
        # Lock contention, busy, transient IO — not corruption. Let it propagate.
        raise
    except sqlite3.DatabaseError as exc:
        reason = f"sqlite refused to open file: {exc}"
    if reason is None:
        return
    # Quarantine FIRST — both the repair path and the fail-closed path
    # preserve the pre-touch bytes before anything mutates the file.
    backup = _backup_corrupt_db(resolved)
    index_names = _repairable_index_names(messages)
    if index_names:
        _kb._log.warning(
            "kanban DB %s failed integrity_check with index-only errors "
            "(%s); pre-repair backup at %s — attempting REINDEX auto-repair.",
            resolved, ", ".join(index_names),
            backup if backup is not None else "<backup failed>",
        )
        repaired, post = _attempt_index_reindex_repair(resolved, index_names)
        if repaired:
            _kb._log.warning(
                "kanban DB %s auto-repaired via REINDEX (%s); "
                "integrity_check now clean. Pre-repair copy kept at %s.",
                resolved, ", ".join(index_names),
                backup if backup is not None else "<backup failed>",
            )
            return
        reason = (
            f"{reason}; REINDEX auto-repair attempted but integrity_check "
            f"still returned {post[0] if post else '<no row>'!r}"
        )
    raise KanbanDbCorruptError(resolved, backup, reason)


@dataclass
class RepairResult:
    """Outcome of :func:`repair_db` for CLI/status reporting.

    ``status`` is one of:

    * ``"ok"``        — integrity_check was already clean; nothing done.
    * ``"repaired"``  — index-only errors found, REINDEX applied, re-check
      clean. ``backup_path`` holds the pre-repair quarantine copy.
    * ``"corrupt"``   — still corrupt: either a non-index error class
      (fail-closed, no repair attempted) or a REINDEX whose re-check did
      not come back clean.
    * ``"missing"``   — no DB file (or zero-byte placeholder); nothing to do.
    """

    status: str
    db_path: Path
    messages: list[str] = field(default_factory=list)
    post_repair_messages: list[str] = field(default_factory=list)
    backup_path: Optional[Path] = None
    reindexed: list[str] = field(default_factory=list)


def repair_db(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> RepairResult:
    """Probe a kanban DB and apply the narrow index-REINDEX repair if needed.

    Shares the exact policy of :func:`_guard_existing_db_is_healthy`: only
    integrity failures composed *entirely* of index-scoped errors are
    repairable; the corrupt bytes are quarantined via
    :func:`_backup_corrupt_db` BEFORE any mutation; the REINDEX runs under
    the board's cross-process init flock; and anything else stays corrupt
    (fail-closed) for the caller to surface. Unlike the guard this never
    raises :class:`KanbanDbCorruptError` — it returns a structured
    :class:`RepairResult` so ``hermes kanban repair`` can report and choose
    its own exit code.

    Transient ``sqlite3.OperationalError`` (locked/busy) still propagates
    raw, exactly like the guard: a locked healthy DB is not corruption and
    must not be quarantined.
    """
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        if not resolved.exists() or resolved.stat().st_size == 0:
            return RepairResult(status="missing", db_path=resolved)
    except OSError:
        return RepairResult(status="missing", db_path=resolved)

    with _kb._cross_process_init_lock(resolved):
        messages: list[str] = []
        try:
            messages = _probe_integrity(resolved)
        except sqlite3.OperationalError:
            # Locked/busy — not corruption; let the caller report it raw.
            raise
        except sqlite3.DatabaseError as exc:
            # Same quarantine the connect-time guard takes for a file
            # sqlite refuses to open at all (e.g. malformed page 1).
            return RepairResult(
                status="corrupt",
                db_path=resolved,
                messages=[f"sqlite refused to open file: {exc}"],
                backup_path=_backup_corrupt_db(resolved),
            )
        if _integrity_messages_ok(messages):
            return RepairResult(status="ok", db_path=resolved, messages=messages)

        # Quarantine FIRST — identical policy to the connect-time guard.
        backup = _backup_corrupt_db(resolved)
        index_names = _repairable_index_names(messages)
        if not index_names:
            return RepairResult(
                status="corrupt",
                db_path=resolved,
                messages=messages,
                backup_path=backup,
            )
        repaired, post = _attempt_index_reindex_repair(resolved, index_names)
        # The file changed on disk; force the next connect() in this process
        # to re-probe instead of trusting the stale healthy-path cache.
        with _INIT_LOCK:
            _INITIALIZED_PATHS.discard(str(resolved))
        return RepairResult(
            status="repaired" if repaired else "corrupt",
            db_path=resolved,
            messages=messages,
            post_repair_messages=post,
            backup_path=backup,
            reindexed=index_names,
        )


def _schema_is_present(conn: sqlite3.Connection) -> bool:
    """Whether an open connection actually sees the kanban schema.

    ``tasks`` is the sentinel: :data:`SCHEMA_SQL` always creates it, and
    SQLite loses tables all-or-nothing (a file is either the one we
    initialized or a fresh one created by this very open), so one
    ``sqlite_master`` lookup on the already-resident page 1 is enough. Cheap
    by design — it runs on every steady-state :func:`connect`.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks' LIMIT 1"
        ).fetchone()
    except sqlite3.DatabaseError:
        # Unreadable schema table is not this guard's call — let the full init
        # path's header/integrity probes classify and quarantine it.
        return False
    return row is not None


def _open_configured(path: Path, under_lock) -> tuple[sqlite3.Connection, Any]:
    """Open ``path`` with the kanban PRAGMA set, then run ``under_lock(conn)``.

    WAL activation and ``under_lock`` share the process-local ``_INIT_LOCK``
    critical section: WAL setup can take an exclusive lock while SQLite
    creates sidecar files for a fresh DB, and concurrent gateway startup
    threads must not race before ``_INITIALIZED_PATHS`` is populated. The
    connection is closed if anything raises.
    """
    conn = _sqlite_connect(path)
    try:
        conn.row_factory = sqlite3.Row
        with _INIT_LOCK:
            # WAL doesn't work on network filesystems (NFS/SMB/FUSE); the shared
            # helper falls back to DELETE with one ERROR log (see
            # hermes_state._WAL_INCOMPAT_MARKERS).
            from hermes_state import apply_wal_with_fallback
            apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
            # FULL (was NORMAL): fsync before each checkpoint to narrow the
            # crash window that can leave a b-tree page header torn.
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA wal_autocheckpoint=100")
            # Bound the WAL file: the periodic explicit checkpoint is PASSIVE
            # (never truncates), so SQLite trims the -wal file to this limit on
            # the writer's natural post-checkpoint reset. 8 MiB is generous.
            conn.execute("PRAGMA journal_size_limit=8388608")
            conn.execute("PRAGMA foreign_keys=ON")
            # Zero freed pages so a later torn write cannot expose stale cell
            # content; persisted in the DB header for new DBs.
            conn.execute("PRAGMA secure_delete=ON")
            # Surface corrupt cells as read errors instead of silent wrong data.
            conn.execute("PRAGMA cell_size_check=ON")
            out = under_lock(conn)
    except Exception:
        conn.close()
        raise
    return conn, out


def connect(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> sqlite3.Connection:
    """Open (and initialize if needed) the kanban DB.

    WAL mode is enabled on every connection; it's a no-op after the first
    time but keeps the code robust if the DB file is ever re-created.

    The first connection to a given path auto-runs :func:`init_db` so
    fresh installs and test harnesses that construct `connect()`
    directly don't have to remember a separate init step. Subsequent
    connections skip the schema check via a module-level path cache.

    Path resolution:

    * ``db_path`` explicit → used as-is (legacy callers, tests).
    * ``board`` explicit → resolves to that board's DB.
    * Neither → :func:`kanban_db_path` resolves via
      ``HERMES_KANBAN_DB`` env → ``HERMES_KANBAN_BOARD`` env →
      ``<root>/kanban/current`` → ``default``.
    """
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: once THIS process has initialized this path, the expensive
    # first-open work (header validation, integrity probe, schema + additive
    # migrations) is already done and cached in _INITIALIZED_PATHS. Acquiring
    # the cross-process init lock on every connect is what let a single stalled
    # holder (e.g. an external `hermes kanban list` mid-integrity-probe) block
    # the long-lived gateway dispatcher's next-tick connect() forever — an
    # unbounded flock with no timeout, no LOCK_NB, no recovery (#36644). On the
    # steady-state path there is nothing for the cross-process lock to protect
    # (no schema/migration writes run), so skip it entirely and just open the
    # connection with WAL/pragmas under the cheap in-process _INIT_LOCK.
    resolved = str(path.resolve())
    if resolved in _INITIALIZED_PATHS:
        conn, schema_present = _open_configured(path, _schema_is_present)
        if schema_present:
            return conn
        # The cache says "initialized", the file says otherwise: it was deleted
        # or replaced under a live process, and the open above silently
        # recreated an empty DB. Left alone, every query on this path fails
        # with "no such table: tasks" for the rest of the process's life and
        # the board just renders empty (#83445). Drop the stale cache entry and
        # fall through to the full init path, which re-runs the header and
        # integrity probes and the schema script under the cross-process lock.
        conn.close()
        with _INIT_LOCK:
            _INITIALIZED_PATHS.discard(resolved)
        _kb._log.warning(
            "kanban DB %s lost its schema after this process initialized it "
            "(deleted or replaced externally); re-initializing.",
            path,
        )

    with _kb._cross_process_init_lock(path):
        # Read-only file/sidecar preflight (port of kilocode#12508) —
        # repair-or-refuse before the header/integrity probes so a stray
        # read-only kanban.db fails with an actionable message instead of
        # "attempt to write a readonly database" mid-init.
        from hermes_state import preflight_db_writability
        preflight_db_writability(path, db_label=f"kanban.db ({path.name})")
        # Cheap byte-level check first — catches the #29507 TLS-overwrite shape
        # and other invalid-header cases without opening a sqlite connection.
        _validate_sqlite_header(path)
        # Full integrity probe — catches corruption past the header (malformed
        # pages, broken internal metadata). Cached per-path after first success
        # via _INITIALIZED_PATHS so it only runs once per process per path.
        _guard_existing_db_is_healthy(path)
        resolved = str(path.resolve())

        def _init_if_needed(conn: sqlite3.Connection) -> None:
            # Idempotent: CREATE TABLE IF NOT EXISTS + the additive migrations,
            # cached so later connect() calls in this process are cheap. Runs
            # under _INIT_LOCK so same-process dispatcher threads can't race
            # through the ALTER TABLE pass with stale PRAGMA snapshots.
            if resolved not in _INITIALIZED_PATHS:
                conn.executescript(_kb.SCHEMA_SQL)
                _migrate_add_optional_columns(conn)
                _INITIALIZED_PATHS.add(resolved)

        conn, _ = _open_configured(path, _init_if_needed)
    return conn


@contextlib.contextmanager
def connect_closing(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
):
    """Open a kanban DB connection and guarantee it is closed on exit.

    Use this instead of ``with kb.connect() as conn:`` — sqlite3's connection
    context manager only commits/rolls back; it does NOT close the file
    descriptor. Long-lived processes (gateway, dashboard) that route every
    kanban operation through ``connect()`` otherwise leak FDs to ``kanban.db``
    / ``kanban.db-wal`` until ``[Errno 24] Too many open files`` kills them.
    ``connect()`` itself is unchanged for callers that manage the lifetime.
    """
    conn = _kb.connect(db_path=db_path, board=board)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> Path:
    """Create the schema if it doesn't exist; return the path used.

    Kept as a public entry point so CLI ``hermes kanban init`` and the
    daemon have something explicit to call. Unlike :func:`connect`'s
    first-time auto-init (which caches by path), ``init_db`` always
    re-runs the migration pass. Callers that know the on-disk schema
    may have drifted — tests that write legacy event kinds directly,
    external tools that upgrade an old DB file — can call this to
    force re-migration.
    """
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    # Clear the cache entry so the underlying connect() re-runs the
    # schema + migration pass unconditionally.
    with _INIT_LOCK:
        _INITIALIZED_PATHS.discard(resolved)
    with contextlib.closing(_kb.connect(path)):
        pass
    return path


# Additive ``tasks`` columns, in the order legacy DBs receive them (order is
# the physical column order for ``SELECT *`` on migrated boards).
_EARLY_TASK_COLUMNS = (
    ("tenant", "tenant TEXT"),
    ("result", "result TEXT"),
    ("branch_name", "branch_name TEXT"),
    ("project_id", "project_id TEXT"),
    ("idempotency_key", "idempotency_key TEXT"),
)


# (new column, ddl, legacy source column, copy statement) — see the
# RENAME-avoidance note in ``_migrate_add_optional_columns``.
_RENAMED_TASK_COLUMNS = (
    (
        "consecutive_failures", "consecutive_failures INTEGER NOT NULL DEFAULT 0",
        "spawn_failures", "UPDATE tasks SET consecutive_failures = COALESCE(spawn_failures, 0)",
    ),
    ("worker_pid", "worker_pid INTEGER", None, None),
    (
        "last_failure_error", "last_failure_error TEXT",
        "last_spawn_error", "UPDATE tasks SET last_failure_error = last_spawn_error",
    ),
)


# NULL / 0 defaults on every column below reproduce the behaviour existing
# rows had before the column existed (worker-profile model/provider/reasoning,
# global failure limit, single-shot worker, untyped human block, ...).
_LATER_TASK_COLUMNS = (
    ("max_runtime_seconds", "max_runtime_seconds INTEGER"),
    ("last_heartbeat_at", "last_heartbeat_at INTEGER"),
    ("current_run_id", "current_run_id INTEGER"),
    ("workflow_template_id", "workflow_template_id TEXT"),
    ("current_step_key", "current_step_key TEXT"),
    # JSON array of skill names the dispatcher force-loads via --skills.
    ("skills", "skills TEXT"),
    # Per-task override for the consecutive-failure circuit breaker; NULL =
    # ``kanban.failure_limit`` config, then ``DEFAULT_FAILURE_LIMIT``.
    ("max_retries", "max_retries INTEGER"),
    ("model_override", "model_override TEXT"),
    ("provider_override", "provider_override TEXT"),
    ("reasoning_effort", "reasoning_effort TEXT"),
    # Ralph-style goal loop toggle; 0 = classic single-shot worker.
    ("goal_mode", "goal_mode INTEGER NOT NULL DEFAULT 0"),
    ("goal_max_turns", "goal_max_turns INTEGER"),
    # Originating agent/chat session id (``HERMES_SESSION_ID``, e.g. ACP).
    ("session_id", "session_id TEXT"),
    # Typed block reason (VALID_BLOCK_KINDS); NULL = generic human blocker.
    ("block_kind", "block_kind TEXT"),
    # Unblock-loop counter; existing rows start at 0.
    ("block_recurrences", "block_recurrences INTEGER NOT NULL DEFAULT 0"),
)


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after v1 release to legacy DBs.

    Called by ``init_db`` so opening an old DB is always safe.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    for name, ddl in _EARLY_TASK_COLUMNS:
        if name not in cols:
            _add_column_if_missing(conn, "tasks", name, ddl)
    # ``idx_tasks_idempotency`` is created unconditionally below alongside
    # the other additive-column indexes.

    # Re-snapshot: DBs partially migrated by older releases may already carry
    # later columns (e.g. ``consecutive_failures``) even when the first
    # snapshot did not, so the legacy-column migration stays idempotent.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}

    # Legacy column migration: ``spawn_failures`` → ``consecutive_failures``
    # and ``last_spawn_error`` → ``last_failure_error``. ADD-then-copy rather
    # than ``RENAME COLUMN``: very old DBs may lack the legacy column entirely
    # (RENAME raises "no such column"), and RENAME reparses the whole schema,
    # failing if views/triggers reference the old name. Preserves historical
    # counter values when the legacy columns do exist.
    for name, ddl, legacy, copy_sql in _RENAMED_TASK_COLUMNS:
        if name not in cols:
            added = _add_column_if_missing(conn, "tasks", name, ddl)
            if added and legacy is not None and legacy in cols:
                conn.execute(copy_sql)
    for name, ddl in _LATER_TASK_COLUMNS:
        if name not in cols:
            if name == "model_override":
                conn.execute("ALTER TABLE tasks ADD COLUMN model_override TEXT")
            else:
                _add_column_if_missing(conn, "tasks", name, ddl)

    # Indexes over additive ``tasks`` columns must be created after the
    # columns exist. Keeping them in SCHEMA_SQL breaks legacy boards: SQLite
    # parses each statement in ``executescript`` against the live schema, so a
    # ``CREATE INDEX`` over a missing column aborts initialization before the
    # additive ``ALTER TABLE`` migrations below can run. Re-running them here
    # is cheap thanks to ``IF NOT EXISTS`` and stays correct on fresh DBs
    # (where the columns already exist from SCHEMA_SQL).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)"
    )

    # task_events gained a run_id column; back-fill it as NULL for
    # historical events (they predate runs and can't be attributed).
    ev_cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_events)")}
    if "run_id" not in ev_cols:
        _add_column_if_missing(conn, "task_events", "run_id", "run_id INTEGER")

    # Same ordering rule as the additive ``tasks`` indexes above: create the
    # index after the additive column migration so legacy ``task_events``
    # tables don't fail during SCHEMA_SQL execution before ``run_id`` exists.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_run "
        "ON task_events(run_id, id)"
    )

    notify_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kanban_notify_subs'"
    ).fetchone() is not None
    if notify_table_exists:
        notify_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(kanban_notify_subs)")
        }
        for name, ddl in (
            ("notifier_profile", "notifier_profile TEXT"),
            ("delivery_mode", "delivery_mode TEXT NOT NULL DEFAULT 'notify'"),
            ("chat_type", "chat_type TEXT"),
            # Platform-specific stable alt ID (Signal UUID, Feishu union_id, ...)
            # so an active-wake replay reconstructs the SAME ``build_session_key``
            # (which prefers ``user_id_alt`` over ``user_id``). NULL is inert.
            ("user_id_alt", "user_id_alt TEXT"),
            ("delivery_metadata", "delivery_metadata TEXT"),
        ):
            if name in notify_cols:
                continue
            _add_column_if_missing(conn, "kanban_notify_subs", name, ddl)
            if name == "delivery_mode":
                # Backfill ONLY on first-add: pre-column gateway subscriptions
                # had de facto active wake (the notifier woke the originating
                # session whenever the task carried a session_id); defaulting
                # them to 'notify' would silently disable that on upgrade.
                # TUI/CLI rows keep 'notify' (matches _maybe_auto_subscribe).
                # A user's later explicit downgrade is never overwritten.
                conn.execute(
                    "UPDATE kanban_notify_subs SET delivery_mode = 'notify+wake' "
                    "WHERE platform != 'tui'"
                )

    # One-shot backfill: any task that is 'running' before runs existed
    # had its claim_lock / claim_expires / worker_pid on the task row.
    # Synthesize a matching task_runs row so subsequent end-run / heartbeat
    # calls have something to write to. Wrapped in write_txn to serialize
    # against any concurrent dispatcher, and the per-row UPDATE uses
    # ``current_run_id IS NULL`` as a CAS guard so a racing claim can't
    # produce an orphaned row if it interleaves with the backfill pass.
    runs_exist = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs'"
    ).fetchone() is not None
    if runs_exist:
        with write_txn(conn):
            inflight = conn.execute(
                "SELECT id, assignee, claim_lock, claim_expires, worker_pid, "
                "       max_runtime_seconds, last_heartbeat_at, started_at "
                "FROM tasks "
                "WHERE status = 'running' AND current_run_id IS NULL"
            ).fetchall()
            for row in inflight:
                started = row["started_at"] or int(time.time())
                cur = conn.execute(
                    """
                    INSERT INTO task_runs (
                        task_id, profile, status,
                        claim_lock, claim_expires, worker_pid,
                        max_runtime_seconds, last_heartbeat_at,
                        started_at
                    ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"], row["assignee"], row["claim_lock"],
                        row["claim_expires"], row["worker_pid"],
                        row["max_runtime_seconds"], row["last_heartbeat_at"],
                        started,
                    ),
                )
                # CAS: only install the pointer if nothing else claimed
                # the task between our SELECT and here (shouldn't happen
                # under the write_txn, but belt-and-suspenders). If the
                # CAS fails we've got an orphan run_row — mark it
                # reclaimed so it doesn't look in-flight.
                upd = conn.execute(
                    "UPDATE tasks SET current_run_id = ? "
                    "WHERE id = ? AND current_run_id IS NULL",
                    (cur.lastrowid, row["id"]),
                )
                if upd.rowcount != 1:
                    conn.execute(
                        "UPDATE task_runs SET status = 'reclaimed', "
                        "    outcome = 'reclaimed', ended_at = ? "
                        "WHERE id = ?",
                        (int(time.time()), cur.lastrowid),
                    )

    # One-shot event-kind rename pass. The old names ("ready", "priority",
    # "spawn_auto_blocked") still worked but were awkward on the wire;
    # rename them in-place so existing DBs migrate cleanly. Fires once
    # per DB because after the UPDATE no rows match the old kinds.
    for old, new in (
        ("ready", "promoted"),
        ("priority", "reprioritized"),
        ("spawn_auto_blocked", "gave_up"),
    ):
        conn.execute("UPDATE task_events SET kind = ? WHERE kind = ?", (new, old))

    _rebuild_drifted_tables(conn)


# Legacy DBs defined these tables with a ``TEXT PRIMARY KEY`` id (or, for
# ``kanban_notify_subs``, a nullable ``TEXT last_event_id``). The current
# schema uses ``INTEGER PRIMARY KEY AUTOINCREMENT`` / ``INTEGER NOT NULL
# DEFAULT 0``. ``CREATE TABLE IF NOT EXISTS`` skips existing tables
# regardless of schema and ``_add_column_if_missing`` only adds columns, so
# neither can fix a drifted column type — the table must be rebuilt. See
# #35096.
#
# Each entry pairs the canonical CREATE TABLE with the CREATE INDEX
# statements that DROP TABLE would otherwise take down with it (including
# ``idx_events_run``, added by the additive pass above). To guard against
# this list drifting from SCHEMA_SQL, ``test_rebuilt_schema_matches_fresh``
# asserts a rebuilt legacy DB is byte-identical to a fresh one.
_REBUILD_SPECS = {
    "task_events": (
        "CREATE TABLE task_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, run_id INTEGER, kind TEXT NOT NULL,"
        " payload TEXT, created_at INTEGER NOT NULL)",
        (
            "CREATE INDEX idx_events_task ON task_events(task_id, created_at)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ),
    ),
    "task_comments": (
        "CREATE TABLE task_comments ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL,"
        " created_at INTEGER NOT NULL)",
        ("CREATE INDEX idx_comments_task ON task_comments(task_id, created_at)",),
    ),
    "task_runs": (
        "CREATE TABLE task_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, profile TEXT, step_key TEXT,"
        " status TEXT NOT NULL, claim_lock TEXT, claim_expires INTEGER,"
        " worker_pid INTEGER, max_runtime_seconds INTEGER,"
        " last_heartbeat_at INTEGER, started_at INTEGER NOT NULL,"
        " ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT,"
        " error TEXT)",
        (
            "CREATE INDEX idx_runs_task ON task_runs(task_id, started_at)",
            "CREATE INDEX idx_runs_status ON task_runs(status)",
        ),
    ),
    "kanban_notify_subs": (
        "CREATE TABLE kanban_notify_subs ("
        " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, user_id_alt TEXT,"
        " chat_type TEXT,"
        " notifier_profile TEXT, delivery_mode TEXT NOT NULL DEFAULT 'notify',"
        " delivery_metadata TEXT, created_at INTEGER NOT NULL,"
        " last_event_id INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (task_id, platform, chat_id, thread_id))",
        ("CREATE INDEX idx_notify_task ON kanban_notify_subs(task_id)",),
    ),
}


def _table_has_drifted(conn: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` still carries the legacy (pre-AUTOINCREMENT) shape."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not info:
        return False  # table absent — nothing to rebuild
    if table == "kanban_notify_subs":
        lei = next((c for c in info if c["name"] == "last_event_id"), None)
        return lei is not None and (lei["type"] or "").upper() != "INTEGER"
    # task_events / task_comments / task_runs: id must be INTEGER and a PK.
    id_col = next((c for c in info if c["name"] == "id"), None)
    if id_col is None:
        return False
    return not ((id_col["type"] or "").upper() == "INTEGER" and id_col["pk"])


def _rebuild_drifted_tables(conn: sqlite3.Connection) -> None:
    """Rebuild any kanban table whose column types drifted from SCHEMA_SQL.

    Old boards crash the gateway notifier (``int(None)`` on a NULL id in
    ``unseen_events_for_sub``) and never match the ``id > cursor`` filter, so
    every kanban notification is silently lost (#35096). Each affected table is
    rebuilt with the standard SQLite pattern — CREATE new → INSERT shared
    columns → DROP old → RENAME — recreating its indexes too (DROP TABLE takes
    them down). The legacy TEXT ids are dropped (they aren't valid integers);
    AUTOINCREMENT assigns fresh ones and ``last_event_id`` cursors reset to 0,
    so the first post-migration tick replays a task's event history once —
    the safe failure mode for a feature that was already fully broken.

    The whole pass runs in one transaction so an interruption can't leave a
    table half-renamed, and under ``connect()``'s init locks so nothing races
    it. Idempotent: a correctly-typed DB skips every table and returns without
    opening a transaction.
    """
    drifted = [t for t in _REBUILD_SPECS if _table_has_drifted(conn, t)]
    if not drifted:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in drifted:
            create_sql, index_sqls = _REBUILD_SPECS[table]
            old_cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})")]
            _kb._log.info("kanban migration: rebuilding %s to match current schema", table)
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            conn.execute(create_sql)
            new_cols = {c["name"] for c in conn.execute(f"PRAGMA table_info({table})")}
            if table == "kanban_notify_subs":
                # Cast the legacy TEXT cursor to INTEGER; NULL / non-numeric → 0.
                shared = [c for c in old_cols if c in new_cols and c != "last_event_id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}, last_event_id) "
                    f"SELECT {cols_csv}, COALESCE(CAST(last_event_id AS INTEGER), 0) "
                    f"FROM {table}_legacy"
                )
            else:
                # Drop the legacy TEXT id; AUTOINCREMENT reassigns it.
                shared = [c for c in old_cols if c in new_cols and c != "id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}) "
                    f"SELECT {cols_csv} FROM {table}_legacy"
                )
            conn.execute(f"DROP TABLE {table}_legacy")
            for index_sql in index_sqls:
                conn.execute(index_sql)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise


def _check_file_length_invariant(conn: sqlite3.Connection) -> None:
    """Compare SQLite's own page accounting against the file size on disk.

    Raises sqlite3.DatabaseError if the file is shorter than the header claims
    (torn-extend corruption).

    Both sides are read WITHOUT opening the database file. The header side
    comes from ``PRAGMA page_count`` over the existing connection; the on-disk
    side from ``stat()``. An earlier version read the header field with a bare
    ``open(path,"rb")`` -- but ``close()`` cancels every POSIX advisory lock
    this process holds on the file, so that probe silently dropped the locks
    of concurrent writers (and of a running VACUUM) and let other processes
    write into a database a writer still believed it owned. That is the
    documented corruption route in sqlite.org/howtocorrupt.html section 2.2.
    """
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    # In WAL mode a just-committed page can still live in the -wal file, so
    # the main file legitimately lags its page count. Only enforce the
    # invariant under a rollback journal, where every committed page must
    # already be in the main file.
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(row[0]).lower() if row and row[0] is not None else ""
    except sqlite3.Error:
        return
    if journal_mode == "wal":
        return

    ok = file_length_matches_header(conn)
    if ok is False:
        raise sqlite3.DatabaseError(
            "torn-extend detected: the database file is shorter than its "
            "header page count claims"
        )


# SQLite's own busy_timeout uses a near-deterministic backoff, so concurrent
# writers re-collide in lockstep under a stampede. A jittered retry on the
# transaction boundary breaks that convoy. Mirrors state.db's _execute_write:
# a fixed 20-150ms jitter band (a 20ms floor prevents a near-zero retry from
# busy-spinning back into the collision). Only BEGIN IMMEDIATE and COMMIT are
# retried -- both are idempotent re-issues that touch no transaction body, so a
# CAS inside write_txn is never replayed. kanban keeps fewer retries than
# state.db (5 vs 15) because its 120s busy_timeout already absorbs most waits;
# the retry is the backstop for the tail SQLite returns BUSY on immediately.
_BUSY_MAX_RETRIES = 5


_BUSY_RETRY_MIN_S = 0.020  # 20ms


_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection, *, allow_nested: bool = False):
    """Context manager for an IMMEDIATE write transaction.

    Use for any multi-statement write (creating a task + link, claiming a
    task + recording an event, etc.). A claim CAS inside this context is
    atomic -- at most one concurrent writer can succeed.

    Nesting is an explicit opt-in: a caller already inside a transaction
    gets a loud ``RuntimeError`` unless it passes ``allow_nested=True``,
    in which case a SQLite savepoint is used instead of a second
    ``BEGIN IMMEDIATE``. Only composition primitives that graph builders
    deliberately run under one outer commit (``create_task``,
    ``add_comment``) opt in — helpers with post-commit side effects
    (``complete_task`` & co.) must never run under an open outer
    transaction, because their side effects (workspace cleanup, ready
    recomputation, failure-counter clears) would fire while the outer
    transaction can still roll back.

    The explicit ROLLBACK on exception is wrapped in try/except so that
    a SQLite auto-rollback (which leaves no active transaction) does not
    shadow the original exception with a spurious rollback error.
    """
    _kb._assert_not_delegated_child_mutation()
    if getattr(conn, "in_transaction", False):
        if not allow_nested:
            raise RuntimeError(
                "write_txn: already inside a transaction. Nested composition "
                "must opt in explicitly with write_txn(conn, allow_nested=True) "
                "(savepoint semantics; the inner RELEASE is not durable until "
                "the outer transaction commits)."
            )
        savepoint = f"hermes_nested_{secrets.token_hex(8)}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield conn
        except Exception:
            try:
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
            except sqlite3.OperationalError:
                pass
            raise
        else:
            conn.execute(f"RELEASE {savepoint}")
        return

    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            # SQLite has already auto-rolled-back the transaction (typical
            # under EIO, lock contention, or corruption). Nothing to undo;
            # do not let this secondary failure shadow the real one.
            pass
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            # COMMIT exhausted retries with the txn still open; roll back so the
            # connection isn't poisoned for the next BEGIN IMMEDIATE.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        # Post-commit file-length check: header page_count must match actual file pages.
        # A discrepancy means a torn-extend — raise now rather than silently corrupt.
        _kb._check_file_length_invariant(conn)


# Late-bound origin namespace (see module docstring). Imported LAST so this
# module is fully populated before ``kanban_db`` re-exports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
