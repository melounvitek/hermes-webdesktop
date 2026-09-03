"""Session-DB access for the dashboard: per-profile SessionDB opening with schema heal, latest-descendant lookup and the auto-archive ticker.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# ---------------------------------------------------------------------------
# Session detail endpoints
# ---------------------------------------------------------------------------


def _session_latest_descendant(session_id: str, db):
    """Resolve a session id to the newest child leaf session.

    /model may create child sessions. Dashboard refresh should continue the
    newest child instead of reopening the old parent.
    """
    def row_get(row, key, index):
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            try:
                return row[index]
            except Exception:
                return None

    sid = db.resolve_session_id(session_id)
    if not sid or not db.get_session(sid):
        return None, []

    conn = (
        getattr(db, "conn", None)
        or getattr(db, "_conn", None)
        or getattr(db, "connection", None)
        or getattr(db, "_connection", None)
    )

    rows = []
    if conn is not None:
        raw_rows = conn.execute(
            """
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """,
            (sid,),
        ).fetchall()
        for row in raw_rows:
            rows.append({
                "id": row_get(row, "id", 0),
                "parent_session_id": row_get(row, "parent_session_id", 1),
                "started_at": row_get(row, "started_at", 2),
            })
    else:
        rows = db.list_sessions_rich(limit=10000, offset=0, compact_rows=True)

    children = {}
    for row in rows:
        rid = row.get("id")
        parent = row.get("parent_session_id")
        if rid and parent:
            children.setdefault(parent, []).append(row)

    def started(row):
        try:
            return float(row.get("started_at") or 0)
        except Exception:
            return 0.0

    current = sid
    path = [sid]
    seen = {sid}

    while children.get(current):
        candidates = [r for r in children[current] if r.get("id") not in seen]
        if not candidates:
            break
        candidates.sort(key=started, reverse=True)
        current = candidates[0]["id"]
        path.append(current)
        seen.add(current)

    return current, path


# Serialises the one-time writable schema bootstrap for read-only opens.
# Concurrent first-load polls otherwise race sqlite file creation: the losers
# open mode=ro against a store whose schema is still being written and every
# query raises "no such table: sessions".
_session_db_bootstrap_lock = threading.Lock()


def _session_db_read_probe_statements() -> tuple:
    """Stale-schema probes for read-only opens, derived from SCHEMA_SQL.

    Read-only opens skip _reconcile_columns(), so an older store would
    otherwise 500 on every poll until something opened it writable. Derived
    from the same schema the writable reconciler applies, so any column
    added there is probed here automatically — the previous hand-written
    probe listed four columns and went stale the first time a new column
    (sessions.last_activity_at) shipped, leaving the desktop sidebar empty
    after `hermes update` until the first message forced a writable open.
    """
    from hermes_state_schema import schema_read_probe_statements

    return schema_read_probe_statements()


# Stores where a heal WRITABLE OPEN SUCCEEDED and the read probe still
# failed afterwards: the schema problem is one reconciliation cannot fix
# (e.g. a NOT-NULL-without-default column SQLite refuses to ADD). Retrying
# the full writable init on every poll would hammer a live DB for nothing,
# so such stores fall back to the raw read-only open until restart. A
# FAILED writable open (transient lock) is deliberately NOT recorded —
# the next poll retries the heal.
_session_db_heal_exhausted: set = set()

# Deduplicates the heal-failure warning per store per process, so a
# persistent problem is loud once instead of once per sidebar poll.
_session_db_heal_warned: set = set()


def _open_session_db_at_path(db_path: Path, *, read_only: bool):
    """Open a SessionDB at an explicit path with an explicit access mode.

    Writable opens keep the full init and repair path. Read-only opens
    bootstrap a missing or zero-byte store once, and heal an older or
    malformed schema through one writable open before reopening read-only.
    The healthy read path never takes a write lock or requests a checkpoint.

    Scope of the heal: the probe checks every table/column declared in
    SCHEMA_SQL (see ``schema_read_probe_statements``), so ANY schema
    addition escalates a stale store to a one-time writable open — the same
    reconcile the store's own backend runs at startup. Tables created
    outside SCHEMA_SQL (telemetry ``tel_*``, FTS shadow tables) are
    deliberately outside both the probe and the heal.
    """
    from hermes_cli.web_server import (
        _session_db_heal_exhausted,
        _session_db_heal_warned,
        _session_db_read_probe_statements,
    )
    import sqlite3

    from hermes_state import SessionDB, is_malformed_schema_error

    if not read_only:
        return SessionDB(db_path=db_path, read_only=False)

    def _needs_bootstrap() -> bool:
        try:
            return db_path.stat().st_size == 0
        except FileNotFoundError:
            return True
        except OSError:
            return False

    if _needs_bootstrap():
        with _session_db_bootstrap_lock:
            if _needs_bootstrap():
                SessionDB(db_path=db_path, read_only=False).close()

    def _open_probed():
        db = SessionDB(db_path=db_path, read_only=True)
        # Unit-test fakes may replace SessionDB without exposing a raw
        # connection. Probe only real connections.
        conn = getattr(db, "_conn", None)
        if conn is not None and str(db_path) not in _session_db_heal_exhausted:
            try:
                for statement in _session_db_read_probe_statements():
                    conn.execute(statement).fetchone()
            except BaseException:
                db.close()
                raise
        return db

    try:
        return _open_probed()
    except (sqlite3.DatabaseError, UnicodeDecodeError) as exc:
        message = str(exc).lower()
        stale_schema = "no such table" in message or "no such column" in message
        if not stale_schema and not (
            # UnicodeDecodeError = pysqlite could not decode SQLite's own
            # error message because corrupt file bytes were embedded in it
            # (#98924). The one-writable-open heal is the only repair path,
            # so route it through the same dispatch as malformed schema.
            is_malformed_schema_error(exc) or isinstance(exc, UnicodeDecodeError)
        ):
            raise
        SessionDB(db_path=db_path, read_only=False).close()
        try:
            return _open_probed()
        except (sqlite3.DatabaseError, UnicodeDecodeError) as still_stale:
            message = str(still_stale).lower()
            if "no such table" not in message and "no such column" not in message:
                raise
            # The writable open succeeded but the store is STILL behind the
            # probe: reconciliation cannot fix this one. Serve reads without
            # the probe (queries touching the broken part will still fail,
            # everything else works) and stop paying the writable init per
            # poll.
            _session_db_heal_exhausted.add(str(db_path))
            if str(db_path) not in _session_db_heal_warned:
                _session_db_heal_warned.add(str(db_path))
                _log.warning(
                    "state.db at %s is missing schema that a writable "
                    "reconcile could not add (%s); read paths may partially "
                    "fail until the store is repaired",
                    db_path,
                    still_stale,
                )
            return _open_probed()


def _open_session_db_for_profile(profile: Optional[str], *, read_only: bool):
    """Open a SessionDB with an explicit access mode for a profile.

    ``profile`` None/empty selects this process's own ``state.db``. A named
    profile opens that profile's on-disk store directly. Access-mode
    semantics are documented on :func:`_open_session_db_at_path`.
    """
    from hermes_cli.web_server import _cron_profile_home
    from hermes_state import _default_db_path

    if profile:
        _name, home = _cron_profile_home(profile)
        db_path = Path(home) / "state.db"
    else:
        db_path = Path(_default_db_path())
    return _open_session_db_at_path(db_path, read_only=read_only)


# In-process throttle for the opportunistic auto-archive trigger, keyed by
# profile. Bounds the config.yaml read to at most once per this window per
# profile; the actual sweep is throttled far more coarsely by state_meta
# (sessions.min_interval_hours) inside maybe_auto_archive.
_AUTO_ARCHIVE_CHECK_INTERVAL_S = 300.0
_last_auto_archive_check: Dict[str, float] = {}


def _maybe_auto_archive_for_profile(profile: Optional[str]) -> None:
    """Run the config-gated stale-session auto-archive for ``profile``.

    The Desktop backend is spawned as ``hermes serve`` — it runs neither the
    interactive CLI nor the messaging gateway, so neither of those startup
    hooks fire for Desktop users. Triggering the (double-throttled, config-off
    by default) sweep from the session-list path is what makes
    ``sessions.auto_archive`` take effect there. Never raises.
    """
    from hermes_cli.web_server import _open_session_db_for_profile
    try:
        key = profile or ""
        now = time.monotonic()
        last = _last_auto_archive_check.get(key)
        if last is not None and now - last < _AUTO_ARCHIVE_CHECK_INTERVAL_S:
            return
        _last_auto_archive_check[key] = now

        from hermes_cli.config import load_config as _load_full_config
        cfg = (_load_full_config().get("sessions") or {})
        if not cfg.get("auto_archive", False):
            return
        db = _open_session_db_for_profile(profile, read_only=False)
        try:
            db.maybe_auto_archive(
                idle_days=float(cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            )
        finally:
            db.close()
    except Exception as exc:
        _log.debug("opportunistic auto-archive skipped: %s", exc)


async def _auto_archive_ticker_loop(
    interval_s: float = 3600.0, initial_delay_s: float = 90.0
) -> None:
    """Live timer for the stale-session auto-archive (primary profile).

    A long-running Desktop/serve backend must keep sweeping on schedule even
    when no ``/api/sessions`` request arrives to fire the opportunistic
    trigger — e.g. the app sits open for days on an idle chat. The real
    cadence is still owned by state_meta (``sessions.min_interval_hours``)
    inside ``maybe_auto_archive``; this loop is only the poll rate.
    """

    def _sweep() -> None:
        _maybe_auto_archive_for_profile(None)

    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            await asyncio.to_thread(_sweep)
        except Exception as exc:
            _log.debug("auto-archive tick skipped: %s", exc)
        await asyncio.sleep(interval_s)
