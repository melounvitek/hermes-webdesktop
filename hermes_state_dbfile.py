"""state.db file-level health helpers.

Split out of ``hermes_state.py``: header probes (application_id / zeroed-file
detection), deleted-WAL-sidecar holder scans, quarantine of zeroed or
lock-poisoned databases, ``collect_state_db_stats`` and holder-process
classification. Every name is re-imported into ``hermes_state`` so
``hermes_state.<name>`` keeps resolving — and tests that monkeypatch it keep
intercepting, because intra-module calls to patched helpers go through a
lazy ``from hermes_state import ...`` at call time.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_state_common import (
    FTS_REBUILD_DEFERRAL_KEY,
    stat_db_file_identity as _stat_db_file_identity,
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


# _read_sqlite_application_id runs on EVERY write via _raise_if_db_replaced,
# against the LIVE state.db.  A bare open()/read()/close() there is the
# howtocorrupt §2.2 bug: close() cancels every POSIX advisory lock this
# process holds on the file — measured on Linux/SQLite 3.53.1, one probe call
# drops the WAL-mode DMS shared lock the writer connection holds on state.db
# (see hermes_cli/sqlite_safe_read.py for the module built around this rule).
# With the DMS lock gone, a fresh opener in another process can treat this
# writer as dead and rerun WAL-index recovery underneath it.
#
# The probe therefore reads through a per-path fd cached for the life of the
# process: opening an fd never cancels locks (only close() does), and
# os.pread takes no shared file position.  When the path is re-pointed at a
# new inode (the very replacement this probe exists to detect), the stale fd
# is RETIRED, never closed — closing it would cancel the live connection's
# locks on the old file, the exact bug being avoided.  Replacement events are
# rare and halt writes anyway, so the leak is bounded.
_HEADER_PROBE_LOCK = threading.Lock()


_HEADER_PROBE_FDS: "dict[str, tuple[int, int, int]]" = {}  # key -> (fd, dev, ino)


_RETIRED_HEADER_PROBE_FDS: "list[int]" = []  # intentionally never closed


def _pread_db_header(db_path: Path, length: int) -> "Optional[bytes]":
    """Lock-safe raw header read of a possibly-live SQLite database.

    POSIX: pread from a cached, never-closed fd (rebound when the path names
    a new inode).  Windows: plain read — advisory-lock cancellation is a
    POSIX-only hazard and msvcrt locks do not share the failure mode.
    """
    from hermes_state import _IS_WINDOWS
    if _IS_WINDOWS:
        try:
            with db_path.open("rb") as handle:
                return handle.read(length)
        except OSError:
            return None
    key = str(db_path)
    try:
        st = os.stat(db_path)
    except OSError:
        return None
    with _HEADER_PROBE_LOCK:
        cached = _HEADER_PROBE_FDS.get(key)
        if cached is not None and (cached[1], cached[2]) != (st.st_dev, st.st_ino):
            # Path re-pointed at a new file. Retire (never close) the old fd.
            _RETIRED_HEADER_PROBE_FDS.append(cached[0])
            cached = None
            del _HEADER_PROBE_FDS[key]
        if cached is None:
            try:
                fd = os.open(db_path, os.O_RDONLY)
            except OSError:
                return None
            try:
                fst = os.fstat(fd)
            except OSError:
                _RETIRED_HEADER_PROBE_FDS.append(fd)
                return None
            cached = (fd, fst.st_dev, fst.st_ino)
            _HEADER_PROBE_FDS[key] = cached
        try:
            return os.pread(cached[0], length, 0)
        except OSError:
            return None


def _read_sqlite_application_id(db_path: Path) -> "Optional[int]":
    """Read application_id from the SQLite header without opening a connection.

    Safe against live databases: routed through :func:`_pread_db_header`,
    which never issues a ``close()`` that would cancel this process's POSIX
    locks on the file (howtocorrupt §2.2).
    """
    from hermes_state import _STATE_DB_APPLICATION_ID_OFFSET
    header = _pread_db_header(db_path, _STATE_DB_APPLICATION_ID_OFFSET + 4)
    if header is None:
        return None
    if len(header) < _STATE_DB_APPLICATION_ID_OFFSET + 4:
        return None
    if header[:16] != b"SQLite format 3\x00":
        return None
    return int(
        struct.unpack(
            ">I",
            header[_STATE_DB_APPLICATION_ID_OFFSET:_STATE_DB_APPLICATION_ID_OFFSET + 4],
        )[0]
    )


def _stat_sqlite_sidecar_identity(db_path: Path) -> Dict[str, tuple]:
    """Snapshot ``(st_dev, st_ino)`` for existing WAL/SHM sidecars."""
    identities: Dict[str, tuple] = {}
    base = os.fspath(db_path)
    for suffix in ("-wal", "-shm"):
        ident = _stat_db_file_identity(Path(base + suffix))
        if ident is not None:
            identities[suffix] = ident
    return identities


def _canonical_sqlite_path(path: str) -> str:
    """Normalize a /proc fd target, stripping the Linux `` (deleted)`` suffix."""
    return os.path.normcase(os.path.abspath(path.removesuffix(" (deleted)")))


def _watched_sqlite_sidecar_paths(db_path) -> Set[str]:
    base = os.path.abspath(os.fspath(db_path))
    return {
        _canonical_sqlite_path(base + "-wal"),
        _canonical_sqlite_path(base + "-shm"),
    }


def iter_deleted_sqlite_sidecar_holders(db_path) -> List[Tuple[int, str]]:
    """Return processes holding an unlinked ``state.db-wal`` / ``-shm``.

    Linux-only (``/proc/<pid>/fd`` readlink). Windows and other hosts
    return ``[]`` — Windows cannot unlink a sidecar another process still
    holds, and macOS does not use the `` (deleted)`` suffix.

    The scan includes this process: on the SessionDB open/write refuse
    path, the in-process writer that still holds the orphan inode is the
    one that must not mint a replacement WAL (and must stop committing).
    ``_foreign_state_db_holders`` keeps skipping this PID for FTS
    maintenance so a process does not block its own optional repair.
    """
    if not sys.platform.startswith("linux"):
        return []

    holders: List[Tuple[int, str]] = []
    watched = _watched_sqlite_sidecar_paths(db_path)
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            fd_dir = f"/proc/{pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if " (deleted)" not in target:
                    continue
                if _canonical_sqlite_path(target) in watched:
                    holders.append((pid, target))
    except Exception as exc:
        logger.debug("deleted-WAL holder scan failed for %s: %s", db_path, exc)
        return holders
    return holders


def refuse_deleted_wal_generation(db_path) -> None:
    """Raise if any process holds a deleted WAL/SHM generation for *db_path*.

    Called *before* ``sqlite3.connect`` so a second opener cannot mint a
    replacement WAL inode while a live writer still holds the orphan.
    """
    from hermes_state import DeletedWalGenerationError, _DELETED_WAL_GENERATION_MSG
    holders = iter_deleted_sqlite_sidecar_holders(db_path)
    if not holders:
        return
    logger.error(_DELETED_WAL_GENERATION_MSG)
    raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)


def _connect_tracked_db(path, tracking_path=None, **kwargs):
    """``sqlite3.connect`` that registers the open fd for lock-safety.

    While a connection is live, byte-level probes of the same file are
    refused: an ``open()``/``close()`` cancels every POSIX advisory lock this
    process holds on it -- including a running VACUUM's EXCLUSIVE lock.
    Released automatically on ``close()``.

    The ONLY tolerated fallback is the helper being absent entirely
    (scaffold/embed installs that ship hermes_state without hermes_cli). A
    real connection failure must propagate: silently retrying an *untracked*
    connect would disable the guard for the lifetime of that connection,
    which is precisely the failure mode this module exists to prevent.
    """
    try:
        from hermes_cli.sqlite_safe_read import connect_tracked
    except ImportError:
        logger.debug(
            "hermes_cli.sqlite_safe_read unavailable; opening %s untracked "
            "(byte-probe guard inactive in this install)",
            path,
        )
        return sqlite3.connect(str(path), **kwargs)

    # Open through THIS module's sqlite3.connect so callers (and tests) that
    # patch hermes_state.sqlite3.connect keep control of connection creation;
    # the helper still owns tracking.
    return connect_tracked(
        path,
        tracking_path=tracking_path,
        connect_fn=sqlite3.connect,
        **kwargs,
    )


def is_zeroed_state_db(
    path: Path, *, probe_bytes: int = 100, force: bool = False
) -> bool:
    """Detect the #68474/#97568 zeroed state.db signature (0-byte or NUL header).

    Byte-level probe, so it is only safe BEFORE any connection to *path*
    exists in this process: ``close()`` cancels every POSIX advisory lock the
    process holds on the file, which can pull the EXCLUSIVE lock out from
    under a running VACUUM and corrupt the database. The read is routed
    through ``read_header_bytes_preopen``, which refuses (returning False
    here) once a connection is live. Pass ``force=True`` only for offline
    files -- quarantined copies, snapshots, archives.

    Prefer ``hermes_cli.backup.is_zeroed_sqlite_file`` when available; this
    local copy keeps SessionDB openable without importing the CLI package
    in constrained embed paths.
    """
    try:
        from hermes_cli.backup import is_zeroed_sqlite_file

        return is_zeroed_sqlite_file(path, probe_bytes=probe_bytes, force=force)
    except Exception:
        pass
    try:
        if not path.is_file():
            # Special files (FIFO, device, socket) are never "zeroed", and
            # probing a FIFO would block until a writer appears.
            return False
        size = path.stat().st_size
    except OSError:
        return False
    if size < 0:
        return False
    from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen

    if not force and has_live_connection(path):
        return False

    head = read_header_bytes_preopen(
        path, length=max(16, probe_bytes), force=force
    )
    if head is None:
        return False
    if len(head) == 0:
        return True
    if head.startswith(b"SQLite format 3"):
        return False
    return all(byte == 0 for byte in head)


@contextlib.contextmanager
def quarantine_cross_process_lock(path: Path, timeout: float = 5.0):
    """Acquire the cross-process lock for path.quarantine.lock."""
    import platform

    lock_path = path.with_name(path.name + ".quarantine.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        if platform.system() == "Windows":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.020)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.020)
        yield acquired
    finally:
        try:
            if acquired:
                if platform.system() == "Windows":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, AttributeError):
            pass
        finally:
            handle.close()


def quarantine_zeroed_state_db(
    path: Path, *, already_locked: bool = False
) -> Optional[Path]:
    """Move a zeroed state.db aside (preserve bytes) and return quarantine path.

    Uses a cross-process lock (``#68805``) so two concurrent startups cannot
    race: the first process moves the zeroed file and the second re-checks
    under the lock, finding the file already gone (or a fresh DB in its place)
    instead of clobbering the quarantine.
    """
    def _do_quarantine():
        if not path.exists():
            logger.info(
                "quarantine_zeroed_state_db: %s already moved by another process",
                path,
            )
            return None
        if not is_zeroed_state_db(path):
            logger.info(
                "quarantine_zeroed_state_db: %s is no longer zeroed (another "
                "process quarantined it and a fresh DB was created)",
                path,
            )
            return None

        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
        except Exception:
            ts = "unknown"
        dest = path.with_name(
            f"{path.name}.zeroed-{ts}-{os.getpid()}.bak"
        )
        n = 0
        while dest.exists():
            n += 1
            dest = path.with_name(
                f"{path.name}.zeroed-{ts}-{os.getpid()}-{n}.bak"
            )
        try:
            path.rename(dest)
        except OSError as exc:
            logger.error("Failed to quarantine zeroed %s: %s", path, exc)
            return None
        for suffix in ("-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists():
                try:
                    side.rename(Path(str(dest) + suffix))
                except OSError:
                    pass
        return dest

    if already_locked:
        return _do_quarantine()

    with quarantine_cross_process_lock(path) as acquired:
        if not acquired:
            logger.error(
                "quarantine lock for %s not acquired within 5s — refusing to "
                "quarantine without the cross-process lock. The zeroed file "
                "is left in place. If sessions fail to load, restore from "
                "state-snapshots via `hermes snapshot list` / "
                "`hermes snapshot restore <id>`.",
                path,
            )
            return None
        return _do_quarantine()


def collect_state_db_stats(db_path: Path) -> Dict[str, Any]:
    """Best-effort, strictly read-only stats snapshot of a state.db file.

    Opens the database with ``mode=ro`` (URI) and a short timeout so it can
    run against a *live* database held by a gateway without ever taking a
    write lock or mutating the file. Every field is collected independently:
    a failed pragma/SELECT yields ``None`` for that field, and the helper
    itself never raises.

    Deliberately does NOT instantiate :class:`SessionDB` — its constructor
    runs schema DDL (migrations, FTS table creation), which is exactly the
    kind of write a diagnostics probe must never perform.

    Returned keys (all present, any may be None on failure):

    - ``page_count``, ``page_size``, ``freelist_count`` — PRAGMA values
    - ``logical_size_bytes`` — page_count * page_size (post-checkpoint size)
    - ``wal_size_bytes`` — stat() of ``<db>-wal`` (0 when absent)
    - ``journal_mode`` — PRAGMA journal_mode string
    - ``messages`` / ``sessions`` — row counts
    - ``fts_tables`` — dict of {table_name: bool} presence for
      messages_fts / messages_fts_trigram / messages_fts_cjk
    - ``fts_storage_version`` — int from state_meta, None when the marker is
      absent (legacy pre-v23 inline layout)
    - ``fts_rebuild_pending`` — True when the deferred v23 backfill has not
      finished (high_water present and progress < high_water)
    - ``fts_rebuild_high_water`` / ``fts_rebuild_progress`` — raw ints
    - ``fts_rebuild_deferral`` — durable blocked-repair diagnostic, when present
    """
    from hermes_state import _connect_tracked_db
    stats: Dict[str, Any] = {
        "page_count": None,
        "page_size": None,
        "freelist_count": None,
        "logical_size_bytes": None,
        "wal_size_bytes": None,
        "journal_mode": None,
        "messages": None,
        "sessions": None,
        "fts_tables": None,
        "fts_storage_version": None,
        "fts_rebuild_pending": None,
        "fts_rebuild_high_water": None,
        "fts_rebuild_progress": None,
        "fts_rebuild_deferral": None,
    }

    # WAL sidecar size needs no connection at all.
    try:
        wal_path = Path(str(db_path) + "-wal")
        stats["wal_size_bytes"] = wal_path.stat().st_size if wal_path.exists() else 0
    except OSError:
        pass

    conn = None
    try:
        # mode=ro refuses to create the file and refuses every write; a
        # short timeout keeps doctor snappy when a writer holds the lock.
        # Route through the tracked connect so byte-probe helpers
        # (read_header_bytes_preopen) see this connection and refuse raw
        # opens that could cancel our POSIX locks mid-read.
        conn = _connect_tracked_db(
            f"file:{Path(db_path)}?mode=ro",
            tracking_path=Path(db_path),
            uri=True,
            timeout=2.0,
        )
    except Exception as exc:
        logger.debug("collect_state_db_stats: cannot open %s read-only: %s",
                     db_path, exc)
        return stats

    def _scalar(sql: str) -> Any:
        try:
            row = conn.execute(sql).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    try:
        pc = _scalar("PRAGMA page_count")
        ps = _scalar("PRAGMA page_size")
        stats["page_count"] = int(pc) if pc is not None else None
        stats["page_size"] = int(ps) if ps is not None else None
        if stats["page_count"] is not None and stats["page_size"] is not None:
            stats["logical_size_bytes"] = stats["page_count"] * stats["page_size"]

        fl = _scalar("PRAGMA freelist_count")
        stats["freelist_count"] = int(fl) if fl is not None else None

        jm = _scalar("PRAGMA journal_mode")
        stats["journal_mode"] = str(jm) if jm is not None else None

        msgs = _scalar("SELECT COUNT(*) FROM messages")
        stats["messages"] = int(msgs) if msgs is not None else None
        sess = _scalar("SELECT COUNT(*) FROM sessions")
        stats["sessions"] = int(sess) if sess is not None else None

        # FTS table presence via sqlite_master (never SELECTs from the
        # virtual tables themselves — a corrupt index must not fail stats).
        try:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN (?, ?, ?)",
                    ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"),
                ).fetchall()
            }
            stats["fts_tables"] = {
                t: (t in names)
                for t in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")
            }
        except Exception:
            pass

        # Raw state_meta reads — cheap, and independent of SessionDB.
        def _meta_int(key: str) -> Optional[int]:
            try:
                row = conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?", (key,)
                ).fetchone()
                return int(row[0]) if row and row[0] is not None else None
            except Exception:
                return None

        stats["fts_storage_version"] = _meta_int("fts_storage_version")
        high_water = _meta_int("fts_rebuild_high_water")
        progress = _meta_int("fts_rebuild_progress")
        stats["fts_rebuild_high_water"] = high_water
        stats["fts_rebuild_progress"] = progress
        if high_water is None:
            stats["fts_rebuild_pending"] = False
        else:
            stats["fts_rebuild_pending"] = (progress or 0) < high_water
        try:
            row = conn.execute(
                "SELECT value FROM state_meta WHERE key = ? LIMIT 1",
                (FTS_REBUILD_DEFERRAL_KEY,),
            ).fetchone()
            if row:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    stats["fts_rebuild_deferral"] = parsed
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return stats


def count_db_holders(db_path: Path) -> Optional[int]:
    """Best-effort count of processes holding ``db_path`` open (Linux only).

    Scans ``/proc/*/fd`` symlinks for the resolved database path. Returns
    the number of distinct PIDs with the file open, or ``None`` on any
    error or on non-Linux platforms. Never raises; no lsof dependency.
    Unreadable per-process fd dirs (other users' processes without root)
    are silently skipped, so the count is a lower bound.
    """
    try:
        if not sys.platform.startswith("linux"):
            return None
        target = os.path.realpath(str(db_path))
        holders = 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue  # process gone or not ours
            for fd in fds:
                try:
                    if os.readlink(f"{fd_dir}/{fd}") == target:
                        holders += 1
                        break  # one hit per PID
                except OSError:
                    continue
        return holders
    except Exception:
        return None


def _is_inactive_orphan_desktop_holder(
    *,
    ppid: int,
    age_seconds: float,
    min_age_seconds: float,
    ephemeral_backend: bool,
    connection_statuses: List[str],
) -> bool:
    """Pure safety predicate for the narrow Desktop holder reap."""
    return (
        ppid in (0, 1)
        and age_seconds >= min_age_seconds
        and ephemeral_backend
        and "ESTABLISHED" not in connection_statuses
    )


def _concrete_state_db_holder_pids(
    db_path: Path, holders: List[Tuple[int, str]]
) -> List[int]:
    """Return unique PIDs proven to hold this DB or one of its sidecars."""
    canonical_db = os.path.normcase(os.path.abspath(os.fspath(db_path)))
    watched = {
        canonical_db,
        canonical_db + "-wal",
        canonical_db + "-shm",
    }
    pids: List[int] = []
    seen = set()
    for pid, path in holders:
        canonical_path = os.path.normcase(
            os.path.abspath(path.removesuffix(" (deleted)"))
        )
        if pid <= 0 or pid in seen or canonical_path not in watched:
            continue
        seen.add(pid)
        pids.append(pid)
    return pids


def _read_proc_cmdline(pid: int) -> Optional[str]:
    """Read /proc/<pid>/cmdline, world-readable even when fd table is not.

    Returns the cmdline as a space-joined string, or None when unreadable
    (process exited, or hidepid mount).
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        if not raw:
            return None
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return None


_HERMES_CMDLINE_MARKERS = ("hermes_cli.main", "hermes_cli/main", "hermes serve",
                           "hermes-agent", "hermes gateway", "hermes chat")


def _looks_like_hermes(cmdline: str) -> bool:
    """Heuristic: does this cmdline look like a Hermes process?

    Used to decide whether an uninspectable process (fd table unreadable
    due to different user) should be treated as a potential state.db holder.
    We only flag processes that look like Hermes, not every system daemon.
    """
    lower = cmdline.lower()
    return any(marker in lower for marker in _HERMES_CMDLINE_MARKERS)
