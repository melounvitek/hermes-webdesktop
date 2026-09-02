"""state.db repair, backup and writability preflight (split from hermes_state).

Every name is re-imported into ``hermes_state``; intra-module calls to
patchable helpers go through a lazy ``from hermes_state import ...`` at call
time so monkeypatches there still intercept.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from hermes_startup_watchdog import report_startup_progress
from hermes_state_common import (
    _acquire_db_flock,
    _clear_lock_holder_record,
    _describe_lock_holder,
    _read_lock_holder_record,
    is_advisory_lock_contention,
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


def _claim_repair_attempt(db_path: Path) -> bool:
    """Claim the one-shot per-process repair attempt for *db_path*.

    True for the first caller, False afterwards: bounds the repair/reopen loop
    and stops concurrent callers racing surgery on one file.
    """
    from hermes_state import _repair_attempt_lock, _repair_attempted_paths
    key = str(db_path)
    with _repair_attempt_lock:
        if key in _repair_attempted_paths:
            return False
        _repair_attempted_paths.add(key)
        return True


_REPAIR_LOCK_POLL_SECONDS = 0.1


# Snapshot copies are data transfer, not inter-process locking: bound them
# separately at 10 MiB/s, with the historical two-minute floor.
_REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND = 10 * 1024 * 1024


@contextlib.contextmanager
def _cross_process_repair_lock(db_path: Path):
    """Serialize state.db schema surgery across processes.

    Yields True when this process holds the repair lock for *db_path*, False
    when the bounded acquire timed out or the lock file could not be opened.
    Unlike the kanban init lock (idempotent critical section), running surgery
    unlocked IS the unsafe interleaving this prevents: a caller that gets
    False must NOT do surgery.

    ``flock`` because the kernel drops it when the holder dies (a pidfile
    would wedge every future repair); a forked child that inherited the fd is
    the exception, so the acquire records the holder's pid + start time and
    breaks the lock when that holder is provably dead (``_acquire_db_flock``).
    The acquire is bounded because a *live* repairer can sit in ``VACUUM``
    for minutes, and an unbounded wait would hang the caller's open silently.
    """
    from hermes_state import _IS_WINDOWS, _REPAIR_LOCK_TIMEOUT_SECONDS
    lock_path = db_path.with_name(db_path.name + ".repair.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    except OSError as exc:
        # Fail closed, like a timed-out acquire. An unopenable lock file means
        # out of space/inodes/descriptors — and a sibling that opened ITS
        # handle before the disk filled may still be inside surgery; yielding
        # True here once let two processes run surgery on the same live
        # state.db. Callers handle False by re-probing.
        logger.warning(
            "Could not open state.db repair lock %s (%s) — skipping schema "
            "surgery rather than running it without cross-process authority.",
            lock_path, exc,
        )
        yield False
        return

    acquired = False
    try:
        if _IS_WINDOWS:
            deadline = time.monotonic() + _REPAIR_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except (BlockingIOError, OSError) as exc:
                    if not is_advisory_lock_contention(exc):
                        logger.warning(
                            "Could not acquire state.db repair lock %s (%s) — "
                            "skipping schema surgery on a non-contention error.",
                            lock_path, exc,
                        )
                        acquired = None
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_REPAIR_LOCK_POLL_SECONDS)
        else:
            acquired, handle = _acquire_db_flock(
                str(lock_path),
                handle,
                _REPAIR_LOCK_TIMEOUT_SECONDS,
                _REPAIR_LOCK_POLL_SECONDS,
                "state.db repair lock",
            )
        if acquired is None:
            # Non-contention failure already logged with its errno.
            acquired = False
        elif not acquired:
            record = None if _IS_WINDOWS else _read_lock_holder_record(handle)
            logger.warning(
                "state.db repair lock %s held by another process for more "
                "than %.0fs — skipping schema surgery in this process to "
                "avoid racing the repairer. Recorded holder: %s.",
                lock_path, _REPAIR_LOCK_TIMEOUT_SECONDS,
                _describe_lock_holder(record),
            )
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


def _try_acquire_auto_maintenance_lock(db_path: Path) -> Optional[Any]:
    """Non-blocking cross-process lock for one auto-maintenance pass.

    Advisory lock the kernel releases if the holder exits. A caller that cannot
    acquire it must skip the pass: otherwise two startups both pass the interval
    check and the second prunes a row the first has only just closed recoverably.
    """
    from hermes_state import _IS_WINDOWS
    lock_path = db_path.with_name(db_path.name + ".auto-maintenance.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    except OSError as exc:
        logger.warning(
            "Could not open state.db auto-maintenance lock %s (%s) — skipping "
            "automatic maintenance.",
            lock_path,
            exc,
        )
        return None

    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(), msvcrt.LK_NBLCK, 1  # type: ignore[attr-defined]
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        handle.close()
        return None
    return handle


def _release_auto_maintenance_lock(handle: Any) -> None:
    """Release a handle returned by :func:`_try_acquire_auto_maintenance_lock`."""
    from hermes_state import _IS_WINDOWS
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(), msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:  # pragma: no cover - best effort release
        pass
    finally:
        handle.close()


def _bump_schema_cookie(conn: sqlite3.Connection) -> None:
    """Increment the schema cookie after direct ``sqlite_master`` surgery.

    Ordinary DDL bumps this counter and other connections compare it before
    running a prepared statement — that is how they discard a cached schema.
    Editing ``sqlite_master`` under ``writable_schema=ON`` does NOT bump it,
    so live connections elsewhere keep compiling against the old schema (e.g.
    firing triggers into ``messages_fts*`` shadow tables that no longer
    exist). Best-effort, never raises: a failed bump leaves the status quo.
    """
    try:
        current = conn.execute("PRAGMA schema_version").fetchone()[0]
        # Wrap within SQLite's 32-bit signed range; peers compare for equality.
        conn.execute(f"PRAGMA schema_version={(int(current) + 1) & 0x7FFFFFFF}")
    except (sqlite3.DatabaseError, TypeError, IndexError) as exc:
        logger.warning("Could not bump state.db schema cookie: %s", exc)


_MAX_PERSISTENT_REPAIR_ATTEMPTS = 3


_MAX_MALFORMED_BACKUPS = 3


# Sidecars copied alongside a damaged DB and pruned with it. ``-journal``
# matters because rollback-journal (DELETE) mode — Hermes's fallback on
# NFS/SMB/FUSE/ZFS and WAL-reset-vulnerable SQLite builds — leaves a hot
# journal whenever a transaction was open; without it the forensic copy
# cannot be rolled back to a consistent state by hand.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


# Head/tail bytes sampled by ``_db_fingerprint``: changes on any genuine
# repair/truncation/restore while staying O(1) on a multi-GB file.
_FINGERPRINT_SAMPLE_BYTES = 65536


# Header ranges that move on ordinary commits rather than on repair, masked
# out of the content sample: file change counter (24-27) and version-valid-for
# (92-95). In DELETE mode a commit writes the main file directly and a
# malformed-SCHEMA DB still accepts writes, so without the mask any live write
# re-keys the ledger and the repair budget resets to 1 forever. (WAL mode
# routes commits to the -wal sidecar; masking is harmless there.) The page-1
# sqlite_master b-tree — what repair identity depends on — sits after byte 100
# and stays in the sample.
_FINGERPRINT_VOLATILE_HEADER_RANGES = ((24, 28), (92, 96))


def _mask_volatile_header(head: bytes) -> bytes:
    """Zero the commit-counter fields so ordinary writes don't re-key the ledger."""
    if len(head) < 96:
        return head
    buf = bytearray(head)
    for start, end in _FINGERPRINT_VOLATILE_HEADER_RANGES:
        buf[start:end] = b"\x00" * (end - start)
    return bytes(buf)


# Free-space headroom for the pre-repair forensic backup: a full raw copy of
# the damaged DB plus sidecars, so a repair loop on a large state.db is a disk
# amplifier (one incident wrote ~98MB every ~10s until the volume was nearly
# full). Proportional, not a flat floor: an absolute multi-GB reserve would
# refuse backups that fit on small container/VM volumes, and since a refused
# backup is a HARD STOP that would turn "repair loops" into "repair never
# runs" there. Require the copy plus a small slice of the volume, with a
# modest floor.
_REPAIR_BACKUP_MIN_FREE_BYTES = 256 * 1024 * 1024  # 256 MiB absolute floor


_REPAIR_BACKUP_FREE_FRACTION = 0.02  # plus 2% of the volume


def _repair_backup_headroom_bytes(total_bytes: int) -> int:
    """Free space required *beyond* the copy itself, for a volume of *total_bytes*."""
    return max(
        _REPAIR_BACKUP_MIN_FREE_BYTES,
        int(total_bytes * _REPAIR_BACKUP_FREE_FRACTION),
    )


def _repair_scratch_space_error(db_path: Path) -> Optional[str]:
    """Return an error unless snapshot, VACUUM and promotion can fit safely."""
    import shutil

    try:
        main_bytes = db_path.stat().st_size
        snapshot_bytes = main_bytes
        for suffix in _DB_SIDECAR_SUFFIXES:
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                snapshot_bytes += sidecar.stat().st_size
        usage = shutil.disk_usage(db_path.parent)
        headroom = _repair_backup_headroom_bytes(usage.total)
        # Strategy 2 runs VACUUM on the staged DB, which SQLite documents may
        # need up to 2x the database size in extra space; the same reserve then
        # covers transactional promotion into the live DB.
        required = snapshot_bytes + (2 * snapshot_bytes) + headroom
        if usage.free >= required:
            return None
        return (
            f"only {usage.free / 1e9:.2f}GB free on {db_path.parent}; the "
            f"repair snapshot needs up to {snapshot_bytes / 1e9:.2f}GB, "
            f"VACUUM may need another {(2 * snapshot_bytes) / 1e9:.2f}GB, and "
            f"{headroom / 1e9:.2f}GB must remain as headroom. Free disk space, "
            "then retry."
        )
    except OSError as exc:
        return (
            f"could not determine free space on {db_path.parent} ({exc}); "
            "refusing the repair snapshot rather than risk filling the volume"
        )


def _repair_snapshot_timeout_seconds(source_path: Path) -> float:
    """Bound one SQLite snapshot by source size, including live sidecars.

    A WAL can hold committed rows not yet in the main file; count it so a
    healthy large-database copy is not cut off by the repair-lock timeout.
    """
    from hermes_state import _REPAIR_LOCK_TIMEOUT_SECONDS, _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND
    source_bytes = 0
    for suffix in ("", *_DB_SIDECAR_SUFFIXES):
        candidate = (
            source_path
            if not suffix
            else source_path.with_name(source_path.name + suffix)
        )
        try:
            source_bytes += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return max(
        _REPAIR_LOCK_TIMEOUT_SECONDS,
        source_bytes / _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND,
    )


def _repair_failure_consumes_attempt(exc: BaseException) -> bool:
    """Whether a pre-strategy SQLite failure proves deterministic corruption.

    Lock contention, timeouts, disk-full, I/O and filesystem failures are
    environmental — a retry may succeed, so they must not burn the repair
    ledger. Only SQLite's corruption/image result codes prove deterministic
    damage, even when SQLite cannot stage a snapshot far enough to run a
    named strategy.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        # Extended result codes keep the primary code in the low byte.
        primary_code = error_code & 0xFF
        return primary_code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)

    # Older sqlite3 without result-code attributes: narrow message match only,
    # never turning generic "disk is full"/"readonly" into permanent failures.
    message = str(exc).lower()
    return (
        "file is not a database" in message
        or "database disk image is malformed" in message
    )


def _repair_ledger_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".repair-attempts.json")


def _db_fingerprint(db_path: Path) -> "Optional[str]":
    """Cheap identity for a damaged DB file: size + a bounded content sample.

    Deliberately EXCLUDES mtime: the malformed-schema class still accepts
    writes, so live writers, WAL checkpoints and the strategies themselves move
    mtime between passes; keyed on mtime, every pass looked like a NEW file,
    the attempt counter reset to 1 forever and each pass wrote another
    full-size forensic copy. Hashing a multi-GB file on every open is the cost
    this ledger exists to avoid, so sample the head/tail slices any real
    repair, truncation or restore necessarily changes.

    Runs under ``offline_file_access``: ``close()`` on ANY raw descriptor
    cancels every POSIX advisory lock this process holds on the file, including
    a peer connection's RESERVED lock (``hermes_cli.sqlite_safe_read`` rule 1),
    and a live peer is the expected case here (this runs BEFORE
    ``_backup_db_file``'s ``has_live_connection`` guard). Returns ``None``
    ("identity unavailable") when that makes the read unsafe. Callers MUST NOT
    substitute a differently-shaped key: the ledger compares keys for equality,
    so alternating shapes never matches and the unbounded loop returns; the
    ledger helpers keep the recorded key instead.
    """
    try:
        st = db_path.stat()
        try:
            from hermes_cli.sqlite_safe_read import (
                LiveConnectionError,
                offline_file_access,
            )
        except ImportError:
            # Scaffold/embed installs ship hermes_state without hermes_cli; no
            # tracked connections exist there, so the raw read is safe.
            @contextmanager
            def offline_file_access(_path, **_kw):
                yield

            class LiveConnectionError(Exception):
                pass

        try:
            with offline_file_access(db_path, what="fingerprint"):
                with open(db_path, "rb") as fh:
                    head = fh.read(_FINGERPRINT_SAMPLE_BYTES)
                    if st.st_size > _FINGERPRINT_SAMPLE_BYTES:
                        fh.seek(max(0, st.st_size - _FINGERPRINT_SAMPLE_BYTES))
                        tail = fh.read(_FINGERPRINT_SAMPLE_BYTES)
                    else:
                        tail = b""
        except LiveConnectionError:
            return None
        digest = hashlib.sha256(_mask_volatile_header(head) + tail).hexdigest()[:32]
        return f"{st.st_size}:{digest}"
    except OSError:
        return None


def _backup_content_identity(db_path: Path) -> "Optional[str]":
    """Recovery-image identity for forensic-backup dedupe: whole file + sidecars.

    A DIFFERENT equivalence relation from :func:`_db_fingerprint`; never
    conflate them. The fingerprint answers "same repair epoch?" and masks
    commit counters / samples only head+tail so an ordinary write does not mint
    a fresh repair budget. A live writer can commit rows into an *interior*
    page while preserving size and the first/last 64 KiB, so two materially
    different recovery images share one fingerprint; reusing a backup on that
    basis hands the operator a snapshot predating real user data. A forensic
    copy must claim byte identity, so this digests the ENTIRE main file plus
    every present sidecar (the WAL can hold uncheckpointed committed
    frames). The O(n) read is cheaper than the O(n) write it avoids. Runs under
    ``offline_file_access`` (same POSIX-lock reason as ``_db_fingerprint``);
    ``None`` when a live connection makes the read unsafe — the caller then
    takes a fresh backup, never a false reuse.
    """
    try:
        from hermes_cli.sqlite_safe_read import (
            LiveConnectionError,
            offline_file_access,
        )
    except ImportError:
        @contextmanager
        def offline_file_access(_path, **_kw):
            yield

        class LiveConnectionError(Exception):
            pass

    def _hash_whole(path: Path, hasher: "Any") -> None:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)

    try:
        hasher = hashlib.sha256()
        with offline_file_access(db_path, what="backup-identity"):
            # Length-delimit every member (main file included) so the
            # concatenation is prefix-free; otherwise a main-file tail could
            # coincide with a main+sidecar split and dedupe two images together.
            hasher.update(f"\0main:{db_path.stat().st_size}\0".encode())
            _hash_whole(db_path, hasher)
            for suffix in _DB_SIDECAR_SUFFIXES:
                sidecar = db_path.with_name(db_path.name + suffix)
                if sidecar.exists():
                    hasher.update(f"\0{suffix}:{sidecar.stat().st_size}\0".encode())
                    _hash_whole(sidecar, hasher)
        return hasher.hexdigest()
    except LiveConnectionError:
        return None
    except OSError:
        return None


def _read_repair_ledger(db_path: Path) -> "Dict[str, Any]":
    try:
        raw = json.loads(_repair_ledger_path(db_path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except (OSError, ValueError):
        pass
    return {}


def _persistent_repair_attempts_exhausted(db_path: Path) -> bool:
    """Whether *db_path* has already burned its cross-restart repair budget.

    True only when the ledger records ``_MAX_PERSISTENT_REPAIR_ATTEMPTS``
    failures against the CURRENT fingerprint. Never raises; a missing/corrupt
    ledger or unstatable DB reads as "not exhausted" (the in-process claim and
    cross-process lock still bound one run). When a live connection makes the
    fingerprint unavailable, fall back to the SIZE the ledger recorded —
    otherwise a peer connection hides an exhausted budget on every pass.
    """
    ledger = _read_repair_ledger(db_path)
    recorded = ledger.get("fingerprint")
    fp = _db_fingerprint(db_path)
    if fp is None:
        # Size is the one key component that needs no raw read.
        try:
            size_prefix = f"{db_path.stat().st_size}:"
        except OSError:
            return False
        if not isinstance(recorded, str) or not recorded.startswith(size_prefix):
            return False
    elif recorded != fp:
        return False
    return int(ledger.get("failed_attempts", 0)) >= _MAX_PERSISTENT_REPAIR_ATTEMPTS


def _persistent_repair_exhausted_error(db_path: Path) -> str:
    """The stable operator-facing diagnostic for an exhausted repair budget."""
    return (
        f"automatic repair has already failed "
        f"{_MAX_PERSISTENT_REPAIR_ATTEMPTS} times on this exact file — "
        "the corruption is beyond the schema/FTS repair strategies "
        "(likely b-tree page damage). Manual recovery required: restore "
        f"a backup, or salvage with `sqlite3 {db_path} \".recover\"`. "
        f"Delete {_repair_ledger_path(db_path).name} to force another "
        "automatic attempt."
    )


def _record_repair_outcome(
    db_path: Path, *, repaired: bool, fingerprint: "Optional[str]" = None
) -> None:
    """Update the persistent attempt ledger after a repair pass. Never raises.

    Defaults to the post-attempt fingerprint (what the NEXT exhaustion probe
    observes). When a live connection makes it unavailable, keep the recorded
    key and still increment — dropping the pass would let a peer connection
    reset the budget every time. Never write a differently shaped key.
    """
    ledger_path = _repair_ledger_path(db_path)
    try:
        if repaired:
            ledger_path.unlink(missing_ok=True)
            return
        ledger = _read_repair_ledger(db_path)
        recorded = ledger.get("fingerprint")
        fp = fingerprint if fingerprint is not None else _db_fingerprint(db_path)
        if fp is None:
            if not isinstance(recorded, str):
                # No prior key to extend and no safe way to mint one; the
                # in-process claim and cross-process lock still bound this run.
                return
            fp = recorded
        attempts = (
            int(ledger.get("failed_attempts", 0)) + 1 if recorded == fp else 1
        )
        import datetime

        ledger_path.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "failed_attempts": attempts,
                    "last_attempt": datetime.datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not update state.db repair ledger: %s", exc)


def _existing_malformed_backups(db_path: Path) -> "List[Path]":
    """Timestamped forensic backups of *db_path*, newest first."""
    prefix = f"{db_path.name}.malformed-backup-"
    try:
        found = [
            p
            for p in db_path.parent.iterdir()
            if p.name.startswith(prefix)
            and not p.name.endswith(_DB_SIDECAR_SUFFIXES)
        ]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.name, reverse=True)


def _prune_malformed_backups(db_path: Path, keep: int = _MAX_MALFORMED_BACKUPS) -> None:
    """Delete all but the *keep* newest forensic backups (and sidecars)."""
    for stale in _existing_malformed_backups(db_path)[keep:]:
        for victim in (
            stale,
            *(stale.with_name(stale.name + suffix) for suffix in _DB_SIDECAR_SUFFIXES),
        ):
            try:
                victim.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("Could not prune stale DB backup %s: %s", victim, exc)


def _backup_db_file(db_path: Path) -> "Tuple[Optional[Path], Optional[str]]":
    """Raw-copy a (possibly malformed) DB plus sidecars to a timestamped backup.

    Raw bytes on purpose: the DB won't open cleanly, so preserve them exactly
    for forensics / manual restore. Returns ``(backup_path, None)`` or
    ``(None, reason)``; the repair path treats a refused backup as a HARD STOP
    because the forensic bundle is the recovery path when every strategy fails.
    Refuses while a connection to this DB is live in the process: reading the
    file would ``close()`` a descriptor and cancel that connection's POSIX
    advisory locks (see ``hermes_cli.sqlite_safe_read``) — a real case, since
    one SessionDB can enter repair while the gateway holds others.
    """
    import datetime
    import shutil

    try:
        from hermes_cli.sqlite_safe_read import has_live_connection
    except ImportError:
        has_live_connection = None  # type: ignore[assignment]

    if has_live_connection is not None and has_live_connection(db_path):
        reason = (
            f"a connection to {db_path} is still open in this process; "
            "raw-copying it would cancel that connection's POSIX advisory "
            "locks. Close all SessionDB handles first."
        )
        logger.error("Refusing to raw-copy %s for backup: %s", db_path, reason)
        return None, reason

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.malformed-backup-{stamp}")
    # Same-second collision (two damaged states within one second) must not
    # overwrite the earlier forensic copy.
    seq = 1
    while backup_path.exists():
        backup_path = db_path.with_name(
            f"{db_path.name}.malformed-backup-{stamp}_{seq}"
        )
        seq += 1
    try:
        # Sweep staging debris from an earlier interrupted pass BEFORE the
        # dedupe: leftover staging is byte-identical to the damaged DB, so
        # dedupe would hand it back as a legitimate backup. Also sweeps the
        # pre-merge ``.incomplete`` spelling, which prefix-matches as a backup,
        # sorts NEWEST and would otherwise survive prune forever.
        for pattern in (
            f"{db_path.name}.backup-staging-*",
            f"{db_path.name}.malformed-backup-*.incomplete*",
        ):
            for old in db_path.parent.glob(pattern):
                try:
                    old.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best effort
                    pass
        # Dedupe: a repair loop used to copy the SAME damaged bytes on every
        # restart (~900MB a pass, 89GB over 11 days). If the newest existing
        # backup is byte-identical to the current recovery image, reuse it.
        # Match on ``_backup_content_identity`` — NOT mtime (the malformed-
        # SCHEMA class still accepts writes, so mtime missed every pass) and
        # NOT ``_db_fingerprint`` (see its docstring: an interior-page write
        # changes the recovery image without changing the fingerprint).
        try:
            # Only hash the source when there is a candidate to compare
            # against; hashing a multi-GB source right before copying it is
            # pure waste on the common first-corruption pass.
            existing_backups = _existing_malformed_backups(db_path)[:1]
            if existing_backups:
                src_id = _backup_content_identity(db_path)
                for existing in existing_backups:
                    if src_id is not None and _backup_content_identity(existing) == src_id:
                        logger.info(
                            "Reusing existing forensic backup %s (identical to the "
                            "damaged DB).", existing,
                        )
                        return existing, None
        except OSError:
            pass
        # Disk guard: a full raw copy on a nearly-full volume (which a
        # preceding repair loop may itself have caused) can finish off the
        # disk and every process on the machine. Refuse while there is room to.
        try:
            need = db_path.stat().st_size
            for suffix in _DB_SIDECAR_SUFFIXES:
                sidecar = db_path.with_name(db_path.name + suffix)
                if sidecar.exists():
                    need += sidecar.stat().st_size
            usage = shutil.disk_usage(db_path.parent)
            headroom = _repair_backup_headroom_bytes(usage.total)
            if usage.free - need < headroom:
                reason = (
                    f"only {usage.free / 1e9:.2f}GB free on {db_path.parent}; "
                    f"copying the damaged DB needs {need / 1e9:.2f}GB and must "
                    f"leave {headroom / 1e9:.2f}GB headroom. Free disk space, "
                    f"then retry (or recover manually with `sqlite3 {db_path} "
                    '".recover"`).'
                )
                logger.error("Refusing forensic backup of %s: %s", db_path, reason)
                return None, reason
        except OSError as exc:
            # Fail CLOSED: the nearly-full volume this guard exists for is
            # exactly where stat()/disk_usage() is most likely to fail, and
            # proceeding would take the copy that finishes off the disk. Repair
            # then waits (HARD STOP) for a human to free space — the safe side.
            reason = (
                f"could not determine free space on {db_path.parent} ({exc}); "
                "refusing the forensic copy rather than risk filling the "
                f"volume. Free disk space, then retry (or recover manually "
                f'with `sqlite3 {db_path} ".recover"`).'
            )
            logger.error("Refusing forensic backup of %s: %s", db_path, reason)
            return None, reason
        # Copy to a staging name OUTSIDE the ``.malformed-backup-`` prefix and
        # rename into place only once every copy succeeded. A staging name
        # inside the prefix (e.g. ``…-<stamp>.incomplete``) counts as a backup,
        # sorts NEWEST (so prune kept partials and deleted intact copies), and
        # dedupe could return it as ``backup_path`` — passing the hard stop with
        # no real forensic copy on disk.
        staging = db_path.with_name(f"{db_path.name}.backup-staging-{stamp}")
        # (staging_src, final_dst) pairs. PUBLICATION ORDER MATTERS: the main
        # DB name is the bundle's commit marker (what
        # ``_existing_malformed_backups`` counts), so sidecars go FIRST and the
        # main DB LAST; a failure partway then never leaves a countable main
        # backup over a missing sidecar that would pass the hard stop and dedupe.
        staged_sidecars: "List[Tuple[Path, Path, Path]]" = []
        for suffix in _DB_SIDECAR_SUFFIXES:
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                side_staging = staging.with_name(staging.name + suffix)
                side_dst = backup_path.with_name(backup_path.name + suffix)
                staged_sidecars.append((sidecar, side_staging, side_dst))
        main_pair = (staging, backup_path)
        published: "List[Path]" = []
        all_staging_srcs = [staging] + [s for _src, s, _d in staged_sidecars]
        try:
            shutil.copy2(db_path, staging)
            for sidecar, side_staging, _side_dst in staged_sidecars:
                shutil.copy2(sidecar, side_staging)
            publish_order = [
                (s, d) for _src, s, d in staged_sidecars
            ] + [main_pair]
            for src, dst in publish_order:
                os.replace(src, dst)
                published.append(dst)
        except Exception:
            # Roll back unpublished staging files AND anything already promoted,
            # so a failure after the main os.replace leaves no official backup_path.
            for src in all_staging_srcs:
                try:
                    src.unlink(missing_ok=True)
                except OSError:
                    pass
            for dst in published:
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        _prune_malformed_backups(db_path)
        return backup_path, None
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not back up malformed DB %s: %s", db_path, exc)
        return None, f"backup copy failed: {exc}"


def preflight_db_writability(
    db_path: Path,
    *,
    db_label: str = "state.db",
) -> None:
    """Refuse-or-repair read-only DB files BEFORE the first connection opens.

    A stray read-only ``state.db`` / ``-wal`` / ``-shm`` (sudo run, restored
    backup, copied dotfiles) otherwise surfaces as an opaque "attempt to write
    a readonly database" deep inside ``_init_schema``, and the obvious wrong
    "fix" (deleting the ``-wal``) silently loses committed transactions.
    Repairs with ``chmod u+rw`` only inside the Hermes home tree (Hermes owns
    those files, and ``chmod`` fails on files the user doesn't own, which
    bounds the repair exactly); otherwise fails fast naming the exact file and
    ``chmod`` command. Never deletes or truncates a WAL sidecar — once
    writable, the normal open path checkpoints its committed frames.
    ``:memory:`` and ``file:`` URIs are skipped. Shared with ``kanban_db``.
    """
    raw = str(db_path)
    if raw == ":memory:" or raw.startswith("file:"):
        return

    try:
        home: Optional[Path] = Path(get_hermes_home()).resolve()
    except Exception:  # pragma: no cover - defensive
        home = None

    def _in_repair_scope(p: Path) -> bool:
        if home is None:
            return False
        try:
            return p.resolve().is_relative_to(home)
        except (OSError, ValueError):
            return False

    def _ensure_writable(p: Path, *, is_dir: bool = False) -> None:
        import stat as _stat

        if os.access(p, os.R_OK | os.W_OK):
            return
        if _in_repair_scope(p):
            try:
                add = _stat.S_IRUSR | _stat.S_IWUSR | (_stat.S_IXUSR if is_dir else 0)
                os.chmod(p, p.stat().st_mode | add)
            except OSError:
                pass
            if os.access(p, os.R_OK | os.W_OK):
                logger.info(
                    "%s preflight: repaired read-only %s (chmod u+rw%s)",
                    db_label,
                    p,
                    "x" if is_dir else "",
                )
                return
        kind = "directory" if is_dir else "file"
        wal_note = (
            " Do NOT delete the -wal file — it contains committed data that "
            "will be merged into the database once it is writable."
            if p.name.endswith("-wal")
            else ""
        )
        raise sqlite3.OperationalError(
            f"{db_label} is not writable: {kind} {p} is read-only for this "
            f"user. Hermes needs read-write access to open the database. "
            f"Fix with: chmod u+rw{'x' if is_dir else ''} '{p}'"
            f" (files owned by another user may need sudo/chown).{wal_note}"
        )

    parent = db_path.parent
    if parent.is_dir():
        # SQLite needs a writable directory in every journal mode (WAL/SHM
        # sidecars, or the rollback journal in DELETE mode).
        _ensure_writable(parent, is_dir=True)

    for suffix in ("", "-wal", "-shm"):
        p = db_path.with_name(db_path.name + suffix) if suffix else db_path
        if p.is_file():
            _ensure_writable(p)


def _connect_repair_durable(
    db_path: Path, *, timeout: float = 5.0
) -> sqlite3.Connection:
    """``sqlite3.connect`` for the repair/probe paths, with macOS write barriers.

    These paths open ``state.db`` directly (not via ``SessionDB`` /
    :func:`apply_wal_with_fallback`), so they inherited ``synchronous=NORMAL``
    and no ``checkpoint_fullfsync`` — on Darwin, where ``fsync()`` guarantees
    neither data-on-platter nor ordering, an interrupted rewrite leaves
    half-written b-tree pages, and ``REINDEX``/``VACUUM``/``writable_schema``
    surgery rewrite nearly every page. Autocommit (``isolation_level=None``)
    is preserved: DDL and ``VACUUM`` are illegal inside an implicit
    transaction. Barriers are best-effort by necessity: SQLite loads the schema
    before any statement, so on a malformed schema even ``PRAGMA
    synchronous=FULL`` raises — and a malformed DB is this helper's input.
    Whole-file rewrites call :func:`_reapply_durability_barriers` once the
    schema parses again.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout, isolation_level=None)
    _reapply_durability_barriers(conn)
    return conn


def _reapply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Best-effort (re)application of the macOS write barriers. Never raises.

    True when the pragmas were accepted. Call before ``VACUUM``/``REINDEX``
    once the schema parses: a connection opened on a malformed schema could
    not take them at open time.
    """
    from hermes_state import _apply_macos_checkpoint_barrier, _enforce_macos_synchronous_full
    try:
        _apply_macos_checkpoint_barrier(conn)
        _enforce_macos_synchronous_full(conn)
        return True
    except sqlite3.DatabaseError:
        # Schema still unparseable — pragmas cannot be set yet.
        return False
    except Exception:
        return False


def apply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Apply state-store durability barriers without changing journal mode.

    Public entry point for secondary users of ``state.db`` that must inherit
    its owner's journal mode. Also applies the configured
    ``database.synchronous`` level, a per-connection pragma that otherwise
    only rides on the journal-mode setup path guests must not run.
    """
    from hermes_state import _apply_synchronous_pragma
    ok = _reapply_durability_barriers(conn)
    try:
        # Local import: avoids a circular import with hermes_cli.config.
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly()
        raw_synchronous = cfg_get(cfg, "database", "synchronous", default=None)
        if raw_synchronous is not None:
            _apply_synchronous_pragma(
                conn, raw_synchronous, db_label="state.db (guest)"
            )
    except Exception:
        pass
    return ok


@contextmanager
def _exclusive_repair_db_guard(db_path: Path):
    """Yield one live connection that excludes writers for repair surgery.

    ``locking_mode=EXCLUSIVE`` retains file-level exclusion after the short
    ``BEGIN EXCLUSIVE`` is rolled back. The rollback is essential:
    ``Connection.backup`` uses this connection as *source* and later as the
    promotion *destination*, both of which require it transaction-free. It
    stays open across the whole snapshot -> strategies -> promotion window, so
    no other writer can commit a change promotion would overwrite. Existing
    readers make acquisition fail rather than being disturbed: repair fails
    closed unless this process owns the whole window.
    """
    guard: Optional[sqlite3.Connection] = None
    try:
        # The cross-process repair lock already serializes repairers. Do not
        # wait behind an ordinary application connection: a partial repair is
        # less safe than an explicit "stop the gateway and retry".
        guard = _connect_repair_durable(db_path, timeout=0.0)
        guard.execute("PRAGMA locking_mode=EXCLUSIVE")
        guard.execute("BEGIN EXCLUSIVE")
        guard.execute("ROLLBACK")
    except (sqlite3.Error, OSError) as exc:
        if guard is not None:
            try:
                guard.execute("PRAGMA locking_mode=NORMAL")
            except Exception:
                pass
            guard.close()
        yield None, exc
        return

    try:
        yield guard, None
    finally:
        try:
            # Release the exclusive locks before close; also keeps a close-time
            # checkpoint from being mistaken for a repair write by callers that
            # immediately reopen state.db.
            guard.execute("PRAGMA locking_mode=NORMAL")
        except Exception:
            pass
        guard.close()


def _copy_database_snapshot(
    source_path: Path,
    destination_path: Path,
    *,
    source_connection: Optional[sqlite3.Connection] = None,
    destination_connection: Optional[sqlite3.Connection] = None,
) -> None:
    """Copy one complete SQLite snapshot without replacing either file inode.

    The online backup API folds committed WAL frames into the source snapshot
    and writes the destination in one transaction (rolled back if interrupted),
    so ``state.db`` is never swapped out from under handles that refer to it.
    """
    # Compute the deadline before opening an owned source connection: a
    # sidecar vanishing mid-stat must not leak a just-opened descriptor.
    deadline_seconds = _repair_snapshot_timeout_seconds(source_path)
    deadline = time.monotonic() + deadline_seconds
    source = source_connection or _connect_repair_durable(source_path)
    destination = destination_connection
    own_source = source_connection is None
    own_destination = destination_connection is None

    def _check_deadline(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out copying SQLite repair snapshot after "
                f"{deadline_seconds:.0f}s"
            )

    try:
        if destination is None:
            destination = _connect_repair_durable(destination_path)
        elif destination.in_transaction:
            # sqlite3_backup needs a transaction-free destination; the exclusive
            # guard retains exclusion via locking_mode, not a transaction.
            raise sqlite3.ProgrammingError(
                "SQLite repair backup destination has an active transaction"
            )
        source.backup(
            destination,
            pages=256,
            progress=_check_deadline,
            sleep=_REPAIR_LOCK_POLL_SECONDS,
        )
    finally:
        if own_destination and destination is not None:
            destination.close()
        if own_source:
            source.close()


def _db_opens_cleanly(db_path: Path) -> Optional[str]:
    """Probe a DB on a fresh connection. Returns None if healthy, else a reason.

    Runs the first statement that trips the malformed-schema parse (``PRAGMA
    journal_mode``), ``integrity_check``, a ``sessions`` read, FTS5 MATCH
    probes, and a rolled-back ``messages`` write — so FTS5 index corruption,
    which leaves reads and ``integrity_check`` passing while every ``INSERT
    INTO messages`` fails through the FTS triggers, is reported as unhealthy.
    """
    from hermes_state import SessionDB, load_fts5_cjk_extension
    conn = _connect_repair_durable(db_path)
    try:
        # Best-effort tokenizer load: messages_fts_cjk needs cjk_unicode61
        # before any statement (incl. the trigger-driven write probe) can touch
        # it. Without it this probe sees the DB as a tokenizer-less SessionDB
        # would (which drops the cjk triggers), so tokenizer absence must never
        # classify as corruption.
        load_fts5_cjk_extension(conn)
        conn.execute("PRAGMA journal_mode").fetchone()
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        problems = [str(r[0]) for r in rows if r and str(r[0]).lower() != "ok"]
        if problems:
            return "; ".join(problems[:3])
        conn.execute("SELECT COUNT(*) FROM sessions").fetchone()

        # FTS5 read probe. The write probe below misses partial shadow-table
        # corruption where MATCH / snippet / rank raise DatabaseError("database
        # disk image is malformed"), silently breaking session_search and
        # /resume title resolution while check-only reports healthy.
        for fts_table in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
            try:
                # Trigram backs title resolution, so probe it too. MATCH '""'
                # (empty phrase) parses, scans zero rows and exercises the
                # shadow-table read path; FTS5 rejects MATCH '' outright.
                conn.execute(
                    f"SELECT 1 FROM {fts_table} WHERE {fts_table} MATCH '\"\"' LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError as exc:
                # Canonical capability classifier: on builds without fts5 a
                # legacy messages_fts table may exist and MATCH raises "no such
                # module: fts5"; treating that as corruption would send the DB
                # into repair, whose final fallback deletes the messages_fts%
                # schema. Covers "no such tokenizer: trigram" too.
                if SessionDB._is_fts5_unavailable_error(exc):
                    continue
                msg = str(exc).lower()
                if "no such table" in msg or "no such column" in msg:
                    # FTS5 not built yet (brand new file mid-init).
                    continue
                return f"fts5 read probe failed on {fts_table}: {exc}"
            except sqlite3.DatabaseError as exc:
                # Partial shadow-table damage: MATCH raises though the table parses.
                return f"fts5 read probe failed on {fts_table}: {exc}"

        # FTS write probe: drive a row through the messages_fts* triggers in a
        # transaction that is always rolled back. Missing messages/sessions
        # tables (brand new file mid-init) mean "not yet populated", not corruption.
        probe_session_id = f"_hermes_fts_health_probe_{time.time_ns()}"
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (probe_session_id, "_health_probe", time.time()),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (probe_session_id, "user", "_fts_health_probe", time.time()),
            )
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError as exc:
            # Missing tables / FTS disabled — not the corruption class we probe.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return None
            if "no such tokenizer: cjk_unicode61" in msg:
                # This process couldn't load the cjk extension while the DB
                # carries the cjk index — capability gap, not corruption. A
                # tokenizer-less SessionDB self-heals by dropping the triggers.
                return None
            return str(exc)
        return None
    except sqlite3.DatabaseError as exc:
        return str(exc)
    finally:
        conn.close()


def _live_writer_holds_db(db_path: Path) -> bool:
    """True when a connection outside this call still holds ``db_path`` open.

    Asks SQLite for what a repair needs and a live holder cannot grant:
    ``PRAGMA locking_mode=EXCLUSIVE`` then ``BEGIN IMMEDIATE``. In WAL mode
    that needs exclusive locks on the WAL index, so any other open connection
    fails it with SQLITE_BUSY; neither statement parses the schema, so it
    works on malformed DBs. Fails **open** (False) on anything but a positive
    busy/locked signal — refusing to repair a DB nobody holds would strand the
    self-heal path.

    Scope: WAL mode only. In ``journal_mode=DELETE`` (Hermes's fallback on
    WAL-reset-vulnerable builds and NFS/SMB) a held reader takes only SHARED
    and this returns False; repair is then serialised only by the cross-process
    repairer lock. Broadening to DELETE mode is a follow-up.
    """
    probe = None
    try:
        probe = _connect_repair_durable(db_path, timeout=0.0)
        probe.execute("PRAGMA locking_mode=EXCLUSIVE")
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError as exc:
        lowered = str(exc).lower()
        return "locked" in lowered or "busy" in lowered
    except sqlite3.DatabaseError:
        # Malformed/unreadable: no evidence of a live holder either way.
        return False
    except Exception:
        return False
    finally:
        if probe is not None:
            try:
                # Drop exclusive mode before close so the probe never leaves
                # the file pinned.
                probe.execute("PRAGMA locking_mode=NORMAL")
            except Exception:
                pass
            try:
                probe.close()
            except Exception:
                pass


def repair_state_db_schema(db_path: Path, *, backup: bool = True) -> Dict[str, Any]:
    """Repair a state.db whose ``sqlite_master`` is malformed or whose FTS
    indexes reject writes.

    Two corruption classes: malformed schema / "duplicate object definition"
    (even ``PRAGMA`` fails), and FTS write-corruption (base tables read fine,
    ``integrity_check`` passes, writes fail through ``messages_fts*``
    triggers). Least-destructive first: (1) rebuild FTS in place via FTS5
    ``'rebuild'``; (2) de-duplicate ``sqlite_master`` (lowest rowid per
    ``type``/``name``), FTS preserved; (3) drop the FTS schema + ``VACUUM``,
    rebuilt on the next ``SessionDB()`` open. Canonical rows are never
    modified by a failed attempt: strategies run on a complete SQLite snapshot
    and a successful result is copied back transactionally. A raw backup is
    taken first unless ``backup=False``. Surgery is serialised across
    processes (:func:`_cross_process_repair_lock`): the gateway, Desktop
    backend and CLI all open the same file, and concurrent ``writable_schema``
    surgery is itself a corruption source.

    Returns ``{repaired: bool, strategy: str|None, backup_path: str|None,
    error: str|None}``.
    """
    from hermes_state import _cross_process_repair_lock, _db_opens_cleanly, _live_writer_holds_db, _persistent_repair_attempts_exhausted, _probe_journal_mode_for_repair, _record_repair_outcome, _repair_state_db_schema_locked
    report: Dict[str, Any] = {
        "repaired": False,
        "strategy": None,
        "backup_path": None,
        "error": None,
    }

    # Startup-watchdog progress lease: repair is I/O-bound (near-zero CPU),
    # which the watchdog's CPU fallback would misread as a parked deadlock. A
    # single lease (clamped to _MAX_LEASE_S=900) is deliberate: up to that much
    # zombie time on a wedged repair beats per-chunk renewal complexity.
    report_startup_progress(900.0, phase="state_db_repair")

    db_path = Path(db_path)
    if not db_path.exists():
        report["error"] = f"{db_path} does not exist"
        return report

    # Cross-restart attempt cap: the in-memory claim bounds one process, but a
    # class the strategies cannot heal (b-tree page damage) used to re-run the
    # whole surgery, with a fresh forensic backup, on EVERY restart. After
    # _MAX_PERSISTENT_REPAIR_ATTEMPTS failures on the same file, stop.
    if _persistent_repair_attempts_exhausted(db_path):
        report["error"] = _persistent_repair_exhausted_error(db_path)
        logger.error("state.db repair skipped: %s", report["error"])
        return report

    result = report
    with _cross_process_repair_lock(db_path) as holding_lock:
        if not holding_lock:
            # Another process is inside its critical section, or the lock file
            # could not be opened. It may have healed the file already (long
            # VACUUM after a successful strategy), so re-probe before failing.
            if _db_opens_cleanly(db_path) is None:
                report["repaired"] = True
                report["strategy"] = "repaired_by_other_process"
            else:
                report["error"] = (
                    "could not obtain the state.db repair lock (held by "
                    "another process, or the lock file was unopenable); "
                    "skipped schema surgery to avoid racing a concurrent "
                    "repairer"
                )
        else:
            # Recheck exhaustion after acquisition: a queued repairer can have
            # recorded the final failure while this process waited, and this
            # process must not start a fourth attempt.
            if _persistent_repair_attempts_exhausted(db_path):
                report["error"] = _persistent_repair_exhausted_error(db_path)
                logger.error("state.db repair skipped: %s", report["error"])
            # WAL-holder preflight: fail-closed for active readers before a
            # forensic backup is taken. Not the race defence — the exclusive
            # guard in the locked routine excludes writers through promotion and
            # rejects DELETE-mode readers this probe cannot see.
            elif _live_writer_holds_db(db_path):
                report["error"] = (
                    "a live writer still holds state.db; skipped schema surgery "
                    "to avoid tearing b-tree pages under a concurrent writer. "
                    "Stop the gateway (hermes gateway stop) and retry."
                )
                logger.error("state.db repair skipped: %s", report["error"])
            else:
                # Probe the journal mode BEFORE surgery: a rebuilt file comes
                # back in the default (delete) mode and nothing else records
                # the flip (see _restore_journal_mode_after_repair). The probe
                # may fail on a damaged file; then database.journal_mode is
                # the restore target.
                before_mode = _probe_journal_mode_for_repair(db_path)
                result = _repair_state_db_schema_locked(
                    db_path, backup=backup, report=report
                )
                if result.get("repaired"):
                    result["journal_mode_before"] = before_mode
                    _restore_journal_mode_after_repair(db_path, before_mode)
            # Environmental aborts happen before a strategy mutates the
            # snapshot; they are retriable, not proof a strategy was exhausted.
            # Keep that private marker out of the public report. The ledger
            # update stays under the same cross-process lock as surgery so two
            # repairers cannot lose each other's updates; a queued loser must
            # not record at all.
            attempted = bool(result.pop("_repair_attempted", False))
            if attempted or result.get("repaired"):
                _record_repair_outcome(
                    db_path, repaired=bool(result.get("repaired"))
                )
    return result


def _probe_journal_mode_for_repair(db_path: Path) -> Optional[str]:
    """Best-effort journal-mode probe for a (possibly malformed) DB file.

    Returns ``wal``/``delete``, or ``None`` when the file cannot be opened or
    probed (malformed header, concurrent opener's locks — both expected on
    the repair path); callers then fall back to ``database.journal_mode``.
    """
    from hermes_state import _on_disk_journal_mode
    try:
        conn = _connect_repair_durable(db_path)
        try:
            return _on_disk_journal_mode(conn)
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None


def _restore_journal_mode_after_repair(db_path: Path, before_mode: Optional[str]) -> None:
    """Re-apply the journal mode after schema surgery.

    A rebuilt SQLite file comes back in the default (delete) mode; without
    this, a corruption event silently moves a WAL store out of WAL (the
    open-time WAL-reset gate never sees a flip made inside repair). Routed
    through :func:`apply_wal_with_fallback`, not a direct pragma, so it
    inherits the vulnerable-SQLite WAL-reset gate (a rebuilt file IS a new
    database; on a vulnerable runtime the gate deliberately keeps DELETE, and
    "could not reach WAL" is expected there), the macOS-NFS silent-refusal
    handling, and the WAL companions (size limit, checkpoint barrier,
    synchronous=FULL). ``before_mode`` (None if unprobeable) is only for the
    log comparison; the target comes from ``database.journal_mode``.
    Best-effort: the repair already succeeded, so failures log at WARNING.
    """
    from hermes_state import apply_wal_with_fallback
    try:
        conn = _connect_repair_durable(db_path)
        try:
            after = apply_wal_with_fallback(conn, db_label=db_path.name)
        finally:
            conn.close()
        if before_mode and after != before_mode:
            logger.warning(
                "state.db repair changed journal_mode %r -> %r "
                "(pre-surgery probe %r; restore resolved through "
                "apply_wal_with_fallback per database.journal_mode and the "
                "WAL-reset gate)",
                before_mode, after, before_mode,
            )
    except (sqlite3.Error, OSError) as exc:
        logger.warning(
            "state.db repair at %s: post-surgery journal-mode restore "
            "failed (%s); verify with PRAGMA journal_mode on the next open",
            db_path, exc,
        )


def _repair_state_db_schema_locked(
    db_path: Path, *, backup: bool, report: Dict[str, Any]
) -> Dict[str, Any]:
    """Repair strategies for :func:`repair_state_db_schema`.

    Caller must hold the cross-process repair lock for *db_path*. Strategies
    run on a SCRATCH COPY; the result is copied back through SQLite's
    transactional backup API only once proven to open cleanly, so a failed
    repair cannot modify or lose committed canonical data. (A WAL checkpoint
    of already-committed frames on guard release is not a repair mutation.)

    WHY not in place: Strategy 2 ends in ``VACUUM``, which rebuilds the file
    from the schema SQLite can still parse. When the damage IS in the schema
    b-tree — the ``malformed database schema ()`` class handled here — every
    table hanging off the unreadable part is silently dropped, the probe then
    correctly reports STILL malformed, and repair returned ``repaired=False``
    having destroyed what it was asked to save. The forensic backup does not
    close this (nothing reads it back). Not mutating the original is the
    property that holds without a human in the loop.
    """
    from hermes_state import _backup_db_file, _copy_database_snapshot, _db_opens_cleanly, _repair_scratch_space_error, _run_repair_strategies, _unlink_db_triple
    scratch = db_path.with_name(f"{db_path.name}.repair-scratch")
    cleanup_error = _unlink_db_triple(scratch)
    if cleanup_error is not None:
        report["error"] = (
            "could not remove a stale repair snapshot before probing state.db: "
            f"{cleanup_error}"
        )
        logger.error("state.db repair aborted: %s", report["error"])
        return report

    # Re-probe under the lock: a process we queued behind may have just
    # repaired the file; redoing surgery would undo its work (the
    # repair/re-corrupt cascade this lock exists to break).
    if _db_opens_cleanly(db_path) is None:
        report["repaired"] = True
        report["strategy"] = "already_healthy"
        return report

    if backup:
        bpath, backup_error = _backup_db_file(db_path)
        report["backup_path"] = str(bpath) if bpath else None
        if bpath is None:
            # HARD STOP: the forensic image is still required when corruption
            # defeats every strategy, even though strategies run on a snapshot.
            report["error"] = (
                "pre-repair backup refused; aborting schema repair to avoid "
                f"mutating the only copy of the damaged DB: {backup_error}"
            )
            logger.error("state.db repair aborted: %s", report["error"])
            return report

    # The forensic copy deliberately precedes this guard: its raw-copy safety
    # checks inspect real live holders and would be poisoned by our exclusive
    # connection. Everything affecting the repair image or live promotion
    # happens only after writer exclusion is held.
    with _exclusive_repair_db_guard(db_path) as (live_guard, guard_error):
        if live_guard is None:
            report["error"] = (
                "could not acquire exclusive state.db repair ownership; "
                "skipped schema surgery to avoid overwriting a concurrent "
                f"writer. Stop the gateway and retry: {guard_error}"
            )
            if guard_error is not None and _repair_failure_consumes_attempt(
                guard_error
            ):
                report["_repair_attempted"] = True
            logger.error("state.db repair skipped: %s", report["error"])
            return report

        space_error = _repair_scratch_space_error(db_path)
        if space_error is not None:
            report["error"] = space_error
            logger.error("state.db repair aborted: %s", report["error"])
            return report

        try:
            # Reuse live_guard rather than a second source connection: the
            # guard owns the exclusion, and a second connection could be
            # blocked by our own EXCLUSIVE lock on some SQLite builds.
            _copy_database_snapshot(
                db_path, scratch, source_connection=live_guard
            )
        except (OSError, sqlite3.Error, TimeoutError) as exc:
            report["error"] = (
                f"could not stage a complete SQLite repair snapshot of {db_path}: {exc}"
            )
            if _repair_failure_consumes_attempt(exc):
                report["_repair_attempted"] = True
            logger.error("state.db repair aborted: %s", report["error"])
            _unlink_db_triple(scratch)
            return report

        try:
            # Private marker consumed by the outer wrapper: a strategy failure
            # consumes the persistent budget, but a later promotion failure is
            # classified separately (disk/I/O/permission/lock = environmental).
            report["_repair_attempted"] = True
            _run_repair_strategies(scratch, report)
            if report.get("repaired"):
                try:
                    # Do not os.replace the live DB: Windows rejects replacement
                    # under open handles and POSIX would leave those handles on
                    # the old inode. The guard that staged the live image
                    # receives the promotion, keeping writer exclusion throughout.
                    _copy_database_snapshot(
                        scratch,
                        db_path,
                        destination_connection=live_guard,
                    )
                except (OSError, sqlite3.Error, TimeoutError) as exc:
                    report["repaired"] = False
                    report["strategy"] = None
                    report["_repair_attempted"] = _repair_failure_consumes_attempt(
                        exc
                    )
                    report["error"] = (
                        "repaired snapshot could not be promoted transactionally: "
                        f"{exc}"
                    )
                    logger.error("state.db repair promotion failed: %s", exc)
                else:
                    logger.warning(
                        "state.db repaired via '%s' and promoted transactionally: %s",
                        report.get("strategy"),
                        db_path,
                    )
            if not report.get("repaired"):
                # Logged HERE, not in the strategies: they see the scratch copy,
                # and the one message a human acts on must not name a path
                # that no longer exists by the time they read it.
                logger.error(
                    "state.db schema repair could not recover %s automatically "
                    "(no committed canonical data was modified or lost; backup: %s); "
                    "manual restore from backup may be required.",
                    db_path,
                    report["backup_path"],
                )
            return report
        finally:
            # Never leave a half-repaired file beside the DB for a later probe
            # or human to mistake for the real thing.
            cleanup_error = _unlink_db_triple(scratch)
            if cleanup_error is not None:
                logger.warning(
                    "Could not remove state.db repair snapshot after repair: %s",
                    cleanup_error,
                )


def _unlink_db_triple(path: Path) -> Optional[str]:
    """Remove *path* and every SQLite sidecar; return any cleanup failure."""
    from hermes_state import _IS_WINDOWS
    failures: List[str] = []
    for suffix in ("", *_DB_SIDECAR_SUFFIXES):
        victim = path if not suffix else path.with_name(path.name + suffix)
        for attempt in range(10):
            try:
                victim.unlink()
                break
            except FileNotFoundError:
                break
            except PermissionError as exc:
                # Windows may retain a just-closed SQLite handle for a few
                # scheduler ticks; bounded retry. A later open still fails
                # safely if the handle truly remains live.
                if _IS_WINDOWS and attempt < 9:
                    time.sleep(0.05)
                    continue
                failures.append(f"{victim}: {exc}")
                break
            except OSError as exc:
                failures.append(f"{victim}: {exc}")
                break
    return "; ".join(failures) or None


def _run_repair_strategies(
    db_path: Path, report: Dict[str, Any]
) -> Dict[str, Any]:
    """Escalating repair attempts, applied to *db_path* IN PLACE.

    Every strategy mutates its argument, so this is only ever called by
    :func:`_repair_state_db_schema_locked` on a scratch copy nothing else
    holds open — never on the user's database.
    """
    from hermes_state import _db_opens_cleanly, load_fts5_cjk_extension
    # ── Strategy 0: rebuild FTS indexes in place (FTS write-corruption) ──
    # FTS5 'rebuild' rewrites the index from the content table: the
    # least-destructive fix for an index that rejects writes while reads work.
    try:
        conn = _connect_repair_durable(db_path)
        try:
            # The cjk index can only be rebuilt with its tokenizer loaded
            # (best-effort; a tokenizer-less host skips it below).
            load_fts5_cjk_extension(conn)
            for table_name in (
                "messages_fts", "messages_fts_trigram", "messages_fts_cjk"
            ):
                try:
                    conn.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                    )
                except sqlite3.OperationalError:
                    # Table absent (FTS disabled / trigram off / cjk not present
                    # or tokenizer unavailable).
                    continue
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "rebuild_fts"
            logger.warning(
                "state.db FTS indexes rebuilt in place (schema preserved): %s",
                db_path,
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db FTS in-place rebuild pass failed: %s", exc)

    # ── Strategy 0.5: rebuild stale B-tree indexes ──
    # integrity_check reports "wrong # of entries in index" when a B-tree index
    # drifts from its base table; REINDEX rewrites it from the canonical rows.
    try:
        conn = _connect_repair_durable(db_path)
        try:
            # REINDEX rewrites every index b-tree; take the barriers now that
            # the schema parses, in case the open-time attempt was refused.
            _reapply_durability_barriers(conn)
            conn.execute("REINDEX")
            conn.commit()
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "reindex_btree"
            logger.warning(
                "state.db B-tree indexes rebuilt via REINDEX: %s", db_path
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db REINDEX pass failed: %s", exc)

    # ── Strategy 1: de-duplicate sqlite_master (keeps FTS index) ──
    try:
        conn = _connect_repair_durable(db_path)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            dupes = conn.execute(
                "SELECT type, name, COUNT(*) AS c, MIN(rowid) AS keep "
                "FROM sqlite_master GROUP BY type, name HAVING c > 1"
            ).fetchall()
            for type_, name, _count, keep in dupes:
                conn.execute(
                    "DELETE FROM sqlite_master "
                    "WHERE type IS ? AND name IS ? AND rowid <> ?",
                    (type_, name, keep),
                )
            if dupes:
                _bump_schema_cookie(conn)
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "dedup_schema"
            logger.warning(
                "state.db schema repaired by de-duplicating sqlite_master "
                "(FTS index preserved): %s", db_path
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db dedup repair pass failed: %s", exc)

    # ── Strategy 2: drop all FTS schema, VACUUM, rebuild on next open ──
    # The destructive one, and why this path runs on a scratch copy: on a
    # damaged schema b-tree VACUUM silently drops every table hanging off the
    # unreadable part (see _repair_state_db_schema_locked).
    try:
        conn = _connect_repair_durable(db_path)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute("DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'")
            _bump_schema_cookie(conn)
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
            # The schema parses now, so the barriers can finally stick — and
            # VACUUM rewrites the entire file, the worst operation to lose halfway.
            _reapply_durability_barriers(conn)
            conn.execute("VACUUM")
        finally:
            conn.close()
        reason = _db_opens_cleanly(db_path)
        if reason is None:
            report["repaired"] = True
            report["strategy"] = "drop_fts_rebuild"
            logger.warning(
                "state.db schema repaired by dropping FTS schema; indexes "
                "will rebuild from messages on next open: %s", db_path
            )
            return report
        report["error"] = reason
    except sqlite3.DatabaseError as exc:
        report["error"] = str(exc)

    # The "could not recover" log lives in the caller: it must name the user's
    # database, not the scratch copy.
    return report
