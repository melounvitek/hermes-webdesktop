"""Lock-safe inspection of SQLite database files.

POSIX advisory locks are cancelled **process-wide** by ``close()`` on *any* file descriptor for
that file, so a bare ``open(db_path, "rb") ... close()`` on a **live** database silently drops
every lock SQLite holds on it from this process -- including the EXCLUSIVE lock a ``VACUUM`` is
holding while it rewrites the whole file, and the RESERVED lock of an in-flight ``BEGIN IMMEDIATE``.
This module tracks live connections so raw reads only happen when none exist.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Guards BOTH the registry and the lifecycle syscalls it describes. Reentrant
# because connect_tracked -> _canonical_db_path -> ... stays on one thread.
_live_lock = threading.RLock()
# canonical path -> number of live connections opened by this process
_live_connections: dict[str, int] = {}


class UntrackableConnectionError(RuntimeError):
    """A connection to a probe-able database could not be tracked.

    Raised rather than silently returning an untracked connection: on these paths tracking is part
    of the correctness contract, not an optimisation.
    """


class LiveConnectionError(RuntimeError):
    """A raw file operation was attempted on a database with live connections."""


def _key(path: Path | str) -> str:
    """Canonicalise a *filesystem* path for use as a registry key."""
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _canonical_db_path(conn: sqlite3.Connection) -> Optional[str]:
    """The on-disk path of ``main`` as SQLite reports it (immune to ``file:`` URIs, relative paths,
    symlinks). ``None`` for in-memory/unnamed databases, which cannot be byte-probed."""
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except sqlite3.Error:
        return None
    if not row or len(row) < 3 or not row[2]:
        return None
    return _key(row[2])


def track_connection(path: Path | str) -> None:
    """Record that this process now holds a connection to *path*.

    Prefer :func:`connect_tracked`; this exists for callers that manage their own connection
    objects, and for tests.
    """
    with _live_lock:
        _track_key(_key(path))


def _track_key(key: str, delta: int = 1) -> None:
    """Adjust the live count for an already-canonical key (caller holds ``_live_lock``)."""
    remaining = _live_connections.get(key, 0) + delta
    if remaining > 0:
        _live_connections[key] = remaining
    else:
        _live_connections.pop(key, None)


def untrack_connection(path: Path | str) -> None:
    """Record that one connection to *path* has been closed."""
    with _live_lock:
        _track_key(_key(path), -1)


def has_live_connection(path: Path | str) -> bool:
    """Whether this process currently holds any connection to *path*."""
    with _live_lock:
        return _key(path) in _live_connections


class _TrackingMixin:
    """Untrack-on-close behaviour, mixable into any Connection subclass.

    The real ``close()`` and the unregister happen together under ``_live_lock`` so a concurrent
    probe can never observe "no live connection" while this descriptor is still open. Unregister
    runs only after ``close()`` succeeds: untracking before a failing close (e.g. cross-thread
    ProgrammingError) would leave the FD open while the byte-probe guard thinks nothing is live.
    """

    _hermes_tracked_path: str | None = None

    def close(self) -> None:  # type: ignore[misc]
        with _live_lock:
            path = getattr(self, "_hermes_tracked_path", None)
            super().close()  # type: ignore[misc]
            if path is not None:
                self._hermes_tracked_path = None
                untrack_connection(path)


class TrackedConnection(_TrackingMixin, sqlite3.Connection):
    """A ``sqlite3.Connection`` that untracks its path exactly once on close (callers close in many
    places, some via ``contextlib.closing``, so counting closes by hand is unreliable)."""


_tracked_factory_cache: dict[type, type] = {}


def _tracking_factory(factory: type) -> type:
    """Return *factory* augmented with untrack-on-close.

    Callers legitimately pass their own ``Connection`` subclasses (tests simulate FTS5-less or
    pragma-failing runtimes); refusing them or leaving them untracked would quietly unguard the
    database, so the tracking ``close()`` is mixed into the caller's class instead.
    """
    if factory is sqlite3.Connection:
        return TrackedConnection
    if issubclass(factory, _TrackingMixin):
        return factory
    cached = _tracked_factory_cache.get(factory)
    if cached is None:
        cached = type(f"Tracked{factory.__name__}", (_TrackingMixin, factory), {})
        _tracked_factory_cache[factory] = cached
    return cached


def connect_tracked(
    path: Path | str, *, tracking_path: Path | str | None = None, connect_fn=None, **kwargs,
) -> sqlite3.Connection:
    """``sqlite3.connect`` that registers the connection for the lifetime of the fd.

    Use for any connection to a database whose file might otherwise be byte-probed (``state.db``,
    ``kanban.db``). The registration is released automatically on ``close()``.

    The open and the registration happen together under ``_live_lock``, so a concurrent
    :func:`read_header_bytes_preopen` cannot slip between them and cancel this connection's locks.
    """
    opener = connect_fn if connect_fn is not None else sqlite3.connect
    kwargs["factory"] = _tracking_factory(kwargs.get("factory", sqlite3.Connection))

    with _live_lock:
        conn = opener(str(path), **kwargs)
        try:
            resolved = _key(tracking_path) if tracking_path is not None else _canonical_db_path(conn)
            if resolved is None:
                return conn  # in-memory / unnamed: nothing on disk to byte-probe
            if not isinstance(conn, _TrackingMixin):
                # The opener substituted its own factory and discarded ours (test doubles
                # simulating FTS5-less runtimes do this). Retag the instance's class so close()
                # still releases the registry entry rather than silently losing probe safety.
                conn = _retrofit_tracking(conn, resolved)
            conn._hermes_tracked_path = resolved
            _track_key(resolved)
            return conn
        except Exception:
            try:
                # Close via sqlite3 directly: the tracking entry was either never made or is
                # being unwound here.
                sqlite3.Connection.close(conn)
            except Exception:
                pass
            raise


def _retrofit_tracking(conn: sqlite3.Connection, resolved: str) -> sqlite3.Connection:
    """Give an already-open connection untrack-on-close semantics by swapping ``__class__`` for
    one mixing in the tracking ``close()`` (used when an opener ignored the factory we asked for)."""
    cls = type(conn)
    try:
        conn.__class__ = _tracking_factory(cls)  # type: ignore[assignment]
        return conn
    except TypeError as exc:
        raise UntrackableConnectionError(
            f"connection to {resolved} uses factory {cls.__name__}, which "
            "cannot release its tracking entry on close; byte-probe safety "
            "for this database would be silently lost"
        ) from exc


def page_count_bytes(conn: sqlite3.Connection) -> Optional[int]:
    """Logical database size in bytes (``page_count * page_size``, the header field at offset 28)
    read via PRAGMA over *conn* so no new fd is opened. ``None`` when the pragmas cannot be read."""
    try:
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        return int(page_count) * int(page_size)
    except (sqlite3.Error, TypeError, IndexError, ValueError) as exc:
        logger.debug("page_count/page_size unavailable: %s", exc)
        return None


def file_length_matches_header(conn: sqlite3.Connection) -> Optional[bool]:
    """Whether the file on disk is at least as long as the header claims ("torn extend" check).

    Never opens the database file: the header side comes from ``PRAGMA page_count`` over *conn*,
    the on-disk side from ``stat()``. In WAL mode a freshly committed page may still live in the
    ``-wal`` file so the main file legitimately lags; callers must treat this as advisory unless
    the database is in a rollback journal mode.
    """
    path_str = _canonical_db_path(conn)
    if path_str is None:
        return None
    logical = page_count_bytes(conn)
    if not logical:
        return None
    try:
        return os.path.getsize(path_str) >= logical
    except OSError:
        return None


def read_header_bytes_preopen(path: Path | str, *, length: int = 100, force: bool = False) -> Optional[bytes]:
    """Read the first *length* bytes of *path* -- only when no connection is live.

    This is the ONLY sanctioned byte-level read of a database file, restricted to first-open
    validation (real SQLite database? zeroed? overwritten?). The registry check and the
    ``open``/``read``/``close`` run together under ``_live_lock`` so a connection cannot be opened
    between deciding "nothing is live" and closing this descriptor.
    """
    with _live_lock:
        if not force and _key(path) in _live_connections:
            logger.debug(
                "refusing byte-level read of %s: a live connection exists in "
                "this process and close() would cancel its POSIX locks",
                path)
            return None
        try:
            with open(path, "rb") as handle:
                return handle.read(length)
        except OSError:
            return None


@contextlib.contextmanager
def offline_file_access(path: Path | str, *, what: str = "read"):
    """Hold the connection-lifecycle lock across a raw read of a database file.

    Checking :func:`has_live_connection` and *then* doing raw I/O is a check/use race: a connection
    opened in between would have its POSIX locks cancelled by the raw ``close()``. The lock is
    held only for the raw I/O, never across caller work on an open connection.
    """
    with _live_lock:
        if _key(path) in _live_connections:
            raise LiveConnectionError(
                f"Refusing to {what} {path}: a connection to it is still open "
                "in this process, and raw file access would cancel that "
                "connection's POSIX advisory locks. Close all database "
                "handles (stop the gateway/dashboard) and retry.")
        yield
