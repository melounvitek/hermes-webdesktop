"""SQLite journal-mode and PRAGMA policy for state.db.

Split out of ``hermes_state.py``. Every name is re-imported there so
``hermes_state.<name>`` keeps resolving, and tests that monkeypatch it keep
intercepting because intra-module calls to patched helpers go through a lazy
``from hermes_state import ...`` at call time.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, Optional

from hermes_cli.sqlite_runtime import (
    is_sqlite_wal_reset_vulnerable as _is_sqlite_wal_reset_vulnerable,
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


# ---------------------------------------------------------------------------
# WAL-compatibility fallback
# ---------------------------------------------------------------------------
# WAL needs mmap shared memory and fcntl byte-range locks, which network
# filesystems (NFS, SMB/CIFS, some FUSE, WSL1) don't provide reliably — there
# ``PRAGMA journal_mode=WAL`` raises ``locking protocol`` (SQLITE_PROTOCOL).
# ZFS instead corrupts the -shm file under concurrent connection bursts (COW +
# mmap), presenting as ``disk I/O error``. Either would silently break
# everything backed by state.db/kanban.db, so we fall back to
# ``journal_mode=DELETE`` (works on NFS/ZFS; readers block during a write).
_WAL_INCOMPAT_MARKERS = (
    "locking protocol",       # SQLITE_PROTOCOL on NFS/SMB
    "not authorized",         # Some FUSE mounts block WAL pragma outright
    "disk i/o error",         # ZFS SHM corruption under concurrent connections
)

# SQLite's default is -1 (unlimited), so state.db-wal would keep the high-water
# mark of the largest-ever transaction forever. See _apply_wal_size_limit().
_WAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB

# Once-per-process-per-db_label dedup sets: kanban_db.connect() runs on every
# kanban operation, so an undeduped log line would repeat per connection.
# Tests clear these through ``hermes_state.<name>``; ``_warn_once`` resolves
# the set through hermes_state at call time for the same reason.
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()
_wal_reset_bug_warned_paths: set[str] = set()
_wal_reset_bug_warned_lock = threading.Lock()
# "configured delete overridden by on-disk WAL" ERROR.
_delete_overridden_warned_paths: set[str] = set()
_delete_overridden_warned_lock = threading.Lock()
# Dedup state for _log_journal_mode_upgrade_once.
_journal_upgrade_warned_paths: set = set()
_journal_upgrade_warned_lock = threading.Lock()

_CANNOT_VERIFY_DELETE_MSG = (
    "could not verify journal mode before applying configured "
    "journal_mode=delete (database is locked — possible "
    "concurrent openers); refusing to downgrade a database "
    "this process does not exclusively own"
)


def _warn_once(lock: threading.Lock, set_name: str, key: str) -> bool:
    """True the first time *key* is seen in ``hermes_state.<set_name>``."""
    import hermes_state
    seen = getattr(hermes_state, set_name)
    with lock:
        if key in seen:
            return False
        seen.add(key)
        return True


def _mode_from_row(row) -> str:
    """Lower-cased mode from a ``PRAGMA journal_mode`` row, ``""`` if no row."""
    return str(row[0]).strip().lower() if row and row[0] is not None else ""


def _on_disk_journal_mode(conn: sqlite3.Connection) -> Optional[str]:
    """Read the journal mode from the DB header; ``None`` if undeterminable.

    ``None`` (new DB, or PRAGMA failed) sends callers down their fail-closed
    "unknown → refuse to downgrade" branch. ``disk i/o error`` can be transient
    on virtualized block devices (XFS on cloud hosts), so it is retried a few
    times first: transient EIO clears, deterministic filesystem errors do not.
    """
    last_exc: Optional[Exception] = None
    for _ in range(4):
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "disk i/o error" not in str(exc).lower():
                return None
            time.sleep(0.05)
            continue
        if row is None:
            return None
        mode = row[0]
        if isinstance(mode, bytes):  # defensive: sqlite3 occasionally returns bytes
            try:
                mode = mode.decode("ascii")
            except UnicodeDecodeError:
                return None
        return str(mode).strip().lower() if mode is not None else None
    if last_exc is not None:
        logger.debug("_on_disk_journal_mode: retries exhausted on disk read (%s)", last_exc)
    return None


def _apply_wal_size_limit(conn: sqlite3.Connection) -> None:
    """Bound the WAL so it returns space to the OS after big transactions.

    SQLite's default ``journal_size_limit`` is -1: a checkpointed WAL is reused
    in place, never truncated, so ``state.db-wal`` keeps the high-water mark
    of the largest transaction ever run (a 3 GB optimize left a 3 GB WAL).
    With a limit, each checkpoint truncates the WAL back to it; 64 MiB is
    above normal transaction sizes while capping slack predictably.
    Best-effort: failure only costs disk slack and must not prevent opening.
    """
    try:
        conn.execute(f"PRAGMA journal_size_limit={_WAL_SIZE_LIMIT_BYTES}")
    except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
        logger.debug("journal_size_limit not applied: %s", exc)


def _apply_macos_checkpoint_barrier(conn: sqlite3.Connection) -> None:
    """Enable ``PRAGMA checkpoint_fullfsync`` on macOS (no-op elsewhere).

    Apple's ``fsync(2)`` guarantees neither data-on-platter nor write ordering,
    so WAL's corruption-safety assumption fails on Darwin without ``F_FULLFSYNC``:
    a launchd shutdown drops the page cache and a checkpoint that "reported"
    durable can leave a malformed ``state.db``. The barrier applies only at
    checkpoint boundaries (~+0.1 ms/commit vs ~+4 ms for ``fullfsync=1``).
    Best-effort: never raises.
    """
    if sys.platform != "darwin":
        return
    try:
        conn.execute("PRAGMA checkpoint_fullfsync=1")
    except sqlite3.OperationalError:
        pass


def _enforce_macos_synchronous_full(conn: sqlite3.Connection) -> None:
    """Enforce ``PRAGMA synchronous=FULL`` on macOS to prevent btree corruption.

    With NORMAL, a WAL checkpoint racing process termination can leave
    half-written btree pages (``btreeInitPage error 11``). Called after every
    successful WAL activation so a prior connection's NORMAL never sticks.
    Best-effort: never raises.
    """
    if sys.platform != "darwin":
        return
    try:
        conn.execute("PRAGMA synchronous=FULL")
    except sqlite3.OperationalError:
        pass


def _apply_wal_companions(conn: sqlite3.Connection) -> None:
    """The settings every WAL activation carries: size limit + macOS barriers."""
    _apply_wal_size_limit(conn)
    _apply_macos_checkpoint_barrier(conn)
    _enforce_macos_synchronous_full(conn)


def is_sqlite_wal_reset_vulnerable(version_info: Optional[tuple] = None) -> bool:
    """True when the linked SQLite has the WAL-reset bug (3.7.0–3.51.2;
    fixed 3.51.3+, backports 3.50.7 / 3.44.6). Pre-WAL libraries are safe.
    https://sqlite.org/wal.html#walresetbug
    """
    info = version_info if version_info is not None else sqlite3.sqlite_version_info
    return _is_sqlite_wal_reset_vulnerable(info)


def sqlite_source_id() -> str:
    """Return ``sqlite_source_id()``, or an empty string when unavailable."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            row = conn.execute("SELECT sqlite_source_id()").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return ""
    if not row or row[0] is None:
        return ""
    return str(row[0])


def _database_has_content(conn: sqlite3.Connection) -> bool:
    """Whether the file already holds pages (existing vs brand-new DB).

    ``PRAGMA page_count`` is a lock-free header read. Fail-quiet: any error
    answers False, because the only caller gates a warning on this and an
    unknown-answer warning would fire on every fresh database.
    """
    try:
        row = conn.execute("PRAGMA page_count").fetchone()
    except sqlite3.Error:
        return False
    if not row or row[0] is None:
        return False
    try:
        return int(row[0]) > 0
    except (TypeError, ValueError):
        return False


def resolve_journal_mode() -> str:
    """Return the configured journal mode (``wal`` or ``delete``).

    ``database.journal_mode`` in config.yaml is the canonical operator setting;
    ``wal`` is the default, ``delete`` is for filesystems without WAL-safe
    durability (macOS virtiofs, NFS, SMB). Invalid values fail safe to ``wal``.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        database = config.get("database", {})
        if not isinstance(database, dict):
            return "wal"
        raw = database.get("journal_mode", "wal")
    except Exception:
        return "wal"
    if not isinstance(raw, str):
        return "wal"
    mode = raw.strip().lower()
    return mode if mode in ("wal", "delete") else "wal"


class WalUnsupportedError(sqlite3.OperationalError):
    """Raised by :func:`apply_wal_with_fallback` when ``require_wal=True`` and
    the filesystem cannot provide WAL — whether SQLite *raised*
    ``SQLITE_PROTOCOL`` or (macOS NFS) silently returned the still-effective
    mode. Subclasses ``OperationalError`` so existing DB-init handlers still
    catch it while WAL-mandating callers can catch the narrower type.
    """


def apply_wal_with_fallback(
    conn: sqlite3.Connection, *, db_label: str = "state.db", require_wal: bool = False
) -> str:
    """Set ``journal_mode=WAL`` on ``conn``, falling back to DELETE on failure.

    Returns the mode actually set (``"wal"`` or ``"delete"``). Shared by
    :class:`SessionDB` and ``hermes_cli.kanban_db.connect``.  On
    WAL-incompatible filesystems SQLite either raises ``OperationalError``
    ("locking protocol" / "disk I/O error") or — macOS NFS / SMB / AgentFS —
    silently refuses and leaves the DB in DELETE; either way we log at ERROR
    (once per process per ``db_label``) and fall back to DELETE.
    ``require_wal=True`` raises :class:`WalUnsupportedError` instead.

    WAL-reset-bug builds (https://sqlite.org/wal.html#walresetbug, fixed
    3.51.3+, backports 3.50.7 / 3.44.6) never enable WAL on fresh / non-WAL
    databases; an already-WAL DB keeps WAL with a warning.  This gate is
    deliberately RETAINED: the attempt to revert it was confounded by a newer
    SQLite, and re-measured on the bundled 3.50.4 there is no evidence WAL is
    safer.

    Invariant on every path: never downgrade to DELETE if the on-disk header
    reports WAL or the mode cannot be read — other gateway/cron/worker
    connections may hold the DB open, and a live downgrade destroys their
    committed-but-uncheckpointed transactions.
    """
    from hermes_state import is_sqlite_wal_reset_vulnerable, resolve_journal_mode
    configured = resolve_journal_mode()

    # Vulnerable SQLite: never enable WAL on new/non-WAL files. Resolve the
    # operator setting first so an explicit DELETE request still verifies SQLite
    # accepted DELETE rather than silently returning MEMORY or another mode.
    if is_sqlite_wal_reset_vulnerable():
        return _apply_delete_for_wal_reset_bug(
            conn, db_label=db_label, require_delete=configured == "delete"
        )

    # Read-only probe — no flock, no checkpoint, no WAL/SHM unlink — so
    # WAL-init cannot unlink files other connections hold open.
    current_mode = _on_disk_journal_mode(conn)
    if current_mode == "wal":
        if configured == "delete":
            # Never-live-downgrade keeps WAL; tell the operator their delete did not apply.
            _log_configured_delete_overridden_once(db_label)
        _apply_wal_companions(conn)
        return "wal"

    # Honor the canonical database.journal_mode setting (on-disk WAL DBs were
    # returned above and are never live-downgraded).
    if configured == "delete":
        if current_mode is None:
            # Probe failed (locked/busy): another process may hold this DB open
            # in WAL, so ownership is not provably exclusive. Fail loudly — the
            # operator asked for DELETE and we cannot verify it.
            raise sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
        actual = _set_journal_mode_no_wait(conn, "DELETE")
        if actual != "delete":
            raise sqlite3.OperationalError(
                f"could not set configured journal_mode=delete (got {actual or 'no result'})"
            )
        return actual

    # Decide BEFORE the flip whether it would overwrite a mode somebody chose:
    # the probe and page_count are only readable while the file is untouched.
    # A 0-page DB has no prior choice, so brand-new databases stay quiet.
    _upgrading_existing_db = (
        current_mode is not None and current_mode != "wal" and _database_has_content(conn)
    )

    def _wal_activated() -> str:
        if _upgrading_existing_db:
            _log_journal_mode_upgrade_once(db_label, current_mode)
        _apply_wal_companions(conn)
        return "wal"

    try:
        # ``PRAGMA journal_mode=WAL`` RETURNS the resulting mode. Filesystems
        # that refuse by *raising* SQLITE_PROTOCOL hit the except branch, but
        # macOS NFS, SMB/CIFS and the AgentFS NFS overlay refuse WITHOUT raising
        # and just return the still-effective mode. Trust the row, not the
        # absence of an exception.
        mode = _mode_from_row(conn.execute("PRAGMA journal_mode=WAL").fetchone())
        if mode == "wal":
            return _wal_activated()
        # Silent refusal: WAL was not honored, but nothing raised.
        silent_exc = WalUnsupportedError(f"journal_mode=WAL refused without raising (still {mode!r})")
        if require_wal:
            raise silent_exc
        _log_wal_fallback_once(db_label, silent_exc)
        return mode or "delete"
    except sqlite3.OperationalError as exc:
        # The require_wal silent-refusal raise above lands here (subclass of
        # OperationalError) — propagate unchanged, skip the marker logic.
        if isinstance(exc, WalUnsupportedError):
            raise
        msg = str(exc).lower()
        if not any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            raise  # unrelated OperationalError — don't silently swallow
        # ``disk i/o error`` is ambiguous: deterministic WAL-incompatibility on
        # ZFS / APFS-CoW, or a one-shot transient EIO. Treating a transient EIO
        # as a permanent downgrade signal produced mixed-mode corruption, so
        # retry the pragma: transient EIO clears and we return "wal";
        # deterministic cases keep failing into the guarded DELETE fallback.
        if "disk i/o error" in msg:
            for _ in range(2):
                time.sleep(0.05)
                try:
                    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                except sqlite3.OperationalError as retry_exc:
                    if "disk i/o error" not in str(retry_exc).lower():
                        raise
                    exc = retry_exc
                    continue
                if _mode_from_row(row) == "wal":
                    return _wal_activated()
                break
        # Don't downgrade if another process already set WAL on disk, or if the
        # mode cannot be read (probe blocked by a concurrent opener's locks) —
        # ownership is not provably exclusive either way.
        existing = _on_disk_journal_mode(conn)
        if existing == "wal" or existing is None:
            raise
        if require_wal:
            raise WalUnsupportedError(str(exc)) from exc
        _log_wal_fallback_once(db_label, exc)
        _set_journal_mode_no_wait(conn, "DELETE")
        return "delete"


def _set_journal_mode_no_wait(conn: sqlite3.Connection, mode: str) -> str:
    """Execute ``PRAGMA journal_mode=<mode>`` without waiting on other openers.

    The ONLY place a journal-mode switch may be issued for a non-WAL target.
    Forces ``busy_timeout=0`` so SQLite's exclusivity requirement becomes a
    concurrent-opener detector: leaving WAL needs exclusive access, so if ANY
    other connection holds the DB the pragma fails immediately with ``database
    is locked`` instead of sneaking the flip between a concurrent writer's
    transactions (how committed-but-uncheckpointed WAL transactions die).

    Callers must treat a raised ``OperationalError`` as "not exclusively
    owned: leave the journal mode alone", never as retryable. Returns SQLite's
    reported mode (lowercase), or ``""`` if no row.
    """
    previous_timeout = 0
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        if row and row[0] is not None:
            previous_timeout = int(row[0])
    except (sqlite3.OperationalError, TypeError, ValueError):
        previous_timeout = 0
    conn.execute("PRAGMA busy_timeout=0")
    try:
        return _mode_from_row(conn.execute(f"PRAGMA journal_mode={mode}").fetchone())
    finally:
        try:
            conn.execute(f"PRAGMA busy_timeout={previous_timeout}")
        except sqlite3.OperationalError:
            pass


def _apply_delete_for_wal_reset_bug(
    conn: sqlite3.Connection, *, db_label: str, require_delete: bool = False
) -> str:
    """Avoid enabling WAL when the linked SQLite has the WAL-reset bug.

    - Already-WAL on disk: leave WAL alone (no live downgrade) and warn.
    - Mode unreadable (probe blocked by a concurrent opener's locks): not
      provably exclusive — leave the mode alone and warn. Never treat "could
      not read the mode" as "not WAL": that once flipped a live WAL state.db to
      DELETE under a concurrent writer, destroying its uncheckpointed commits.
    - Otherwise: set DELETE (refusing to wait out concurrent openers) and warn.
    - For an explicit operator request, verify SQLite accepted DELETE.
    """
    current = _on_disk_journal_mode(conn)
    if current == "wal":
        _log_wal_reset_bug_once(db_label, kept_wal=True)
        if require_delete:
            # Upgrading SQLite (the warning above) doesn't help on a
            # WAL-incompatible filesystem; emit the actionable message last.
            _log_configured_delete_overridden_once(db_label)
        # No TRUNCATE / journal_mode=DELETE while other processes may still
        # hold this WAL DB open; same safety rule as the NFS path.
        _apply_wal_companions(conn)
        return "wal"
    if current is None:
        # Probe failed — likely another opener's locks, and the DB may be in
        # WAL under a live writer. Never flip a mode we cannot even read.
        if require_delete:
            raise sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
        _log_wal_reset_bug_once(db_label, kept_wal=True, indeterminate=True)
        return "wal"
    actual = ""
    try:
        actual = _set_journal_mode_no_wait(conn, "DELETE")
    except sqlite3.OperationalError as exc:
        if require_delete:
            raise
        lowered = str(exc).lower()
        if "locked" in lowered or "busy" in lowered:
            # A concurrent opener appeared between probe and flip (or already
            # held the DB): SQLite refused the exclusive lock. Leave the mode as is.
            _log_wal_reset_bug_once(db_label, kept_wal=True, indeterminate=True)
            return current or "delete"
        # Best-effort for the automatic fallback: DELETE is normally already
        # the default for new file-backed databases.
    if require_delete and actual != "delete":
        raise sqlite3.OperationalError(
            "could not set configured journal_mode=delete "
            f"(got {actual or 'no result'})"
        )
    _log_wal_reset_bug_once(db_label, kept_wal=False)
    return "delete"


def _wal_reset_repair_hint() -> str:
    """Repair hint matching what ``hermes update`` can actually do for this
    install type (uv-managed venv vs git/pip/docker/nix)."""
    try:
        from hermes_cli.config import (
            detect_install_method,
            recommended_update_command_for_method,
            get_project_root,
        )
        method = detect_install_method(get_project_root())
        cmd = recommended_update_command_for_method(method)
        if method in {"git", "unknown"}:
            return f"Hermes-managed installs can repair the embedded runtime with `{cmd}`"
        if method == "docker":
            return f"update the container image with `{cmd}`"
        return cmd  # nix/nixos
    except Exception:
        pass
    return (
        "install a Python build bundled with SQLite 3.51.3+ "
        "(or backports 3.50.7 / 3.44.6) and restart Hermes"
    )


def _log_wal_reset_bug_once(db_label: str, *, kept_wal: bool, indeterminate: bool = False) -> None:
    """Log once per (process, db_label) about the WAL-reset vulnerability path."""
    if not _warn_once(_wal_reset_bug_warned_lock, "_wal_reset_bug_warned_paths", db_label):
        return
    if indeterminate:
        action = (
            "journal mode could not be verified or exclusively switched "
            "(database is locked — possible concurrent openers); leaving the "
            "journal mode untouched (no live downgrade under concurrent "
            "openers)"
        )
    elif kept_wal:
        action = (
            "is already in WAL mode — leaving WAL in place (no live "
            "downgrade under concurrent openers)"
        )
    else:
        action = "using journal_mode=DELETE instead of enabling WAL"
    # Install-type-aware so the warning never promises a repair path that
    # doesn't exist for git/pip/system Python installs.
    logger.warning(
        "%s: linked SQLite %s (interpreter %s) is vulnerable to the WAL-reset "
        "corruption bug (https://sqlite.org/wal.html#walresetbug) — %s. "
        "Upgrade to SQLite 3.51.3+ (or backports 3.50.7 / 3.44.6); "
        "%s. See `hermes doctor`. This warning fires once per "
        "process per database.",
        db_label, sqlite3.sqlite_version, sys.executable, action, _wal_reset_repair_hint(),
    )


def _log_journal_mode_upgrade_once(db_label: str, previous_mode: str) -> None:
    """Log a single WARNING per (process, db_label) about a non-WAL -> WAL flip.

    ``PRAGMA journal_mode`` is a property of the FILE: switching an existing DB
    to WAL rewrites its header and outlives the process. Operators do set
    DELETE on the file directly (the documented WAL-reset-bug mitigation), and
    nothing told them the next open would silently put WAL back. WARNING, not
    ERROR: this direction is normally desirable; only its invisibility was the
    problem, so this names the durable setting without claiming a degradation.
    """
    if not _warn_once(_journal_upgrade_warned_lock, "_journal_upgrade_warned_paths", db_label):
        return
    logger.warning(
        "%s: on-disk journal_mode was %s and has been switched to WAL. This "
        "rewrites the database header and persists after this process exits. "
        "If %s was a deliberate choice (for example the mitigation for the "
        "SQLite WAL-reset bug, or a WAL-unsafe filesystem), setting it with "
        "PRAGMA on the file will not survive -- every open re-applies the "
        "configured mode. Set `database.journal_mode: delete` in config.yaml "
        "to make it stick. This message fires once per process per database.",
        db_label, previous_mode, previous_mode,
    )


def _log_wal_fallback_once(db_label: str, exc: Exception) -> None:
    """Log a single ERROR per (process, db_label) about WAL fallback.

    ERROR, not WARNING: silently dropping to DELETE is a real concurrency loss
    (under kanban dispatcher + workers a write blocks readers as SQLITE_BUSY).
    """
    if not _warn_once(_wal_fallback_warned_lock, "_wal_fallback_warned_paths", db_label):
        return
    logger.error(
        "%s: WAL journal_mode unsupported on this filesystem (%s) — "
        "falling back to journal_mode=DELETE (slower rollback-journal "
        "mode; reduces concurrency but works on NFS/SMB/FUSE/ZFS). See "
        "https://www.sqlite.org/wal.html for details. This message "
        "fires once per process per database.",
        db_label, exc,
    )


def _log_configured_delete_overridden_once(db_label: str) -> None:
    """Log a single ERROR per (process, db_label) when the operator configured
    ``journal_mode=delete`` but the on-disk DB is already WAL.

    Never-live-downgrade keeps WAL; without this the operator would never learn
    that ``database.journal_mode: delete`` had no effect and that a one-time
    offline ``PRAGMA journal_mode=DELETE`` (no open connections) is required.
    """
    if not _warn_once(_delete_overridden_warned_lock, "_delete_overridden_warned_paths", db_label):
        return
    logger.error(
        "%s: database.journal_mode=delete is configured but the on-disk "
        "database is already WAL; keeping WAL (a live downgrade under open "
        "connections can corrupt the DB). To apply journal_mode=DELETE, stop "
        "all connections to this DB and run a one-time offline "
        "'PRAGMA journal_mode=DELETE' on the file. This message fires once "
        "per process per database.",
        db_label,
    )


# ---------------------------------------------------------------------------
# Config-driven database pragmas
# ---------------------------------------------------------------------------
# Operators write synchronous as a name; mapped here rather than passed through
# so a typo becomes a warning instead of a silently different durability level.
_SYNCHRONOUS_LEVELS: Dict[str, int] = {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}
_SYNCHRONOUS_NAMES: Dict[int, str] = {v: k for k, v in _SYNCHRONOUS_LEVELS.items()}
_SYNCHRONOUS_FULL = 2


def resolve_synchronous_level(raw_value: Any) -> Optional[int]:
    """Map a configured ``database.synchronous`` value to its PRAGMA integer.

    Accepts SQLite's names (``OFF``/``NORMAL``/``FULL``/``EXTRA``, any case) or
    ``0``-``3``. Anything else returns None so the caller warns and leaves the
    level untouched — guessing at a malformed durability setting is worse.
    """
    if isinstance(raw_value, bool):
        # bool is an int subclass and YAML turns bare `on`/`off` into one.
        # "off" is a real durability choice; True is meaningless.
        return 0 if raw_value is False else None
    if isinstance(raw_value, int):
        return raw_value if raw_value in _SYNCHRONOUS_NAMES else None
    text = str(raw_value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in _SYNCHRONOUS_LEVELS:
        return _SYNCHRONOUS_LEVELS[upper]
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value in _SYNCHRONOUS_NAMES else None


def _apply_synchronous_pragma(conn: sqlite3.Connection, raw_value: Any, *, db_label: str) -> None:
    """Set ``PRAGMA synchronous`` from config, never below FULL on macOS.

    Kept out of the integer loop in :func:`apply_database_pragmas`: this PRAGMA
    decides whether a commit is on the platter, so an unrecognised value must
    not fall through to "SQLite default" the way a bad ``cache_size`` can.
    Darwin floor: :func:`_enforce_macos_synchronous_full` runs during WAL
    activation and this runs after it, so a configured ``NORMAL`` would
    otherwise silently undo the macOS btree protection. Raising the level on
    macOS is allowed; lowering it is refused out loud.
    """
    level = resolve_synchronous_level(raw_value)
    if level is None:
        logger.warning(
            "%s: ignoring unrecognized database.synchronous=%r "
            "(expected OFF, NORMAL, FULL, EXTRA, or 0-3)",
            db_label, raw_value,
        )
        return
    if sys.platform == "darwin" and level < _SYNCHRONOUS_FULL:
        logger.warning(
            "%s: refusing database.synchronous=%s on macOS; keeping FULL. "
            "Darwin's fsync() does not guarantee write ordering, so a lower "
            "level readmits the half-written btree pages FULL exists to "
            "prevent.",
            db_label, _SYNCHRONOUS_NAMES[level],
        )
        return
    try:
        conn.execute(f"PRAGMA synchronous={level}")
    except sqlite3.OperationalError:
        pass


def apply_database_pragmas(conn: sqlite3.Connection, *, db_label: str = "state.db") -> None:
    """Apply optional performance and WAL-sizing PRAGMAs from ``config.yaml``.

    Journal mode is NOT handled here — ``database.journal_mode`` is owned by
    :func:`resolve_journal_mode` inside :func:`apply_wal_with_fallback`.

    Keys under ``database:``: ``cache_size`` (negative = KiB, positive =
    pages), ``mmap_size`` (bytes, 0 = disabled), ``temp_store`` (0-3),
    ``wal_autocheckpoint`` (pages), ``journal_size_limit`` (bytes), and
    ``synchronous`` (``OFF``/``NORMAL``/``FULL``/``EXTRA`` or ``0``-``3``).
    Unset ``synchronous`` leaves SQLite's compile-time default, which differs
    between bundled, distro and Homebrew builds.

    Best-effort: config load or pragma failures are ignored so DB init never
    breaks on a malformed ``database:`` section. Applied to ALL connection
    types: writer, read_only, WAL per-thread readers.
    """
    try:
        # Local import avoids a circular import with hermes_cli.config.
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return
    for pragma_name in ("cache_size", "mmap_size", "temp_store", "wal_autocheckpoint", "journal_size_limit"):
        raw_value = cfg_get(cfg, "database", pragma_name, default=None)
        if raw_value is None:
            continue
        try:
            value = int(str(raw_value).strip())
        except (TypeError, ValueError):
            logger.warning("%s: ignoring non-integer database.%s=%r", db_label, pragma_name, raw_value)
            continue
        try:
            conn.execute(f"PRAGMA {pragma_name}={value}")
        except sqlite3.OperationalError:
            pass
    # Last: the sizing pragmas above cannot change durability, and the macOS
    # enforcement ran earlier during WAL activation (see _apply_synchronous_pragma).
    raw_synchronous = cfg_get(cfg, "database", "synchronous", default=None)
    if raw_synchronous is not None:
        _apply_synchronous_pragma(conn, raw_synchronous, db_label=db_label)
