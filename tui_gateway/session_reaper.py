"""Session flush / reaping / orphan sweep / cross-backend heartbeat: exit-flush signal handlers, idle + LRU eviction, orphaned session-row sweep, backend heartbeat refresher.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib
import threading

from tui_gateway._env import env_float

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# Knobs (_SESSION_TTL_S, _REAPER_SCAN_S, _EXIT_FLUSH_BUDGET_S, _INCREMENTAL_FLUSH_INTERVAL_S) live in server.py.

# ── Flush-on-kill + periodic incremental flush ───────────────────────────
# (a) SIGTERM/SIGINT run a bounded flush to state.db BEFORE normal shutdown, chained to the prior
# handler; (b) the idle-reaper scan piggybacks an incremental flush so a SIGKILL loses at most one interval.


def _flush_session_messages(session: dict | None) -> bool:
    """Best-effort durable flush of one session's transcript via ``agent._persist_session``
    (same marker-deduped contract as ``_finalize_session``): repeated calls only write
    genuinely-unflushed messages, never duplicate rows."""
    agent = session.get("agent") if session else None
    if agent is None or not hasattr(agent, "_persist_session"):
        return False
    snapshot = getattr(agent, "_session_messages", None)
    if not snapshot:
        return False
    try:
        agent._persist_session(snapshot)
        return True
    except Exception:
        logger.debug("incremental session flush failed", exc_info=True)
        return False


def _reaper_session_snapshot() -> list:
    with _sessions_lock:
        return list(_sessions.values())


def _flush_dirty_sessions(now: float | None = None) -> int:
    """Periodic incremental flush, driven by the idle-reaper scan. Skips ``running``
    sessions: the turn thread owns mid-turn persistence and mutates the live message list,
    so racing it from the reaper thread is never safe. Idle sessions flush at most once per
    ``_INCREMENTAL_FLUSH_INTERVAL_S``; ``now`` (monotonic) is injectable for tests."""
    if _INCREMENTAL_FLUSH_INTERVAL_S <= 0:
        return 0
    if now is None:
        now = time.monotonic()
    flushed = 0
    for session in _reaper_session_snapshot():
        if not isinstance(session, dict) or session.get("running"):
            continue
        last = float(session.get("_last_incremental_flush") or 0.0)
        if last and (now - last) < _INCREMENTAL_FLUSH_INTERVAL_S:
            continue
        if _flush_session_messages(session):
            flushed += 1
        session["_last_incremental_flush"] = now
    return flushed


def _flush_sessions_before_exit(budget_s: float | None = None) -> int:
    """Bounded flush of ALL in-memory sessions on the way out. Runs on a daemon worker
    joined with the budget so a hung SQLite write can't block exit past
    ``HERMES_TUI_EXIT_FLUSH_BUDGET_S`` (default 5s). Running sessions are included — the
    process is dying, so a partial transcript beats guaranteed loss."""
    budget = _EXIT_FLUSH_BUDGET_S if budget_s is None else max(0.0, budget_s)
    if budget <= 0:
        return 0
    result = {"flushed": 0}

    def _run() -> None:
        deadline = time.monotonic() + budget
        for session in _reaper_session_snapshot():
            if time.monotonic() >= deadline:
                break
            if _flush_session_messages(session):
                result["flushed"] += 1

    worker = threading.Thread(target=_run, daemon=True, name="hermes-exit-flush")
    worker.start()
    worker.join(budget)
    return result["flushed"]


_exit_flush_prev_handlers: dict[int, Any] = {}
_exit_flush_handlers_installed = False


def _handle_exit_flush_signal(signum, frame) -> None:
    """Flush in-memory sessions, then hand off to the prior handler (uvicorn's graceful
    shutdown, a supervisor's handler, or the default disposition) — this only *prepends*
    a bounded durable flush."""
    with contextlib.suppress(Exception):
        _flush_sessions_before_exit()
    import signal as _signal

    prev = _exit_flush_prev_handlers.get(signum)
    if callable(prev):
        prev(signum, frame)
        return
    if prev is _signal.SIG_IGN:
        return
    # Default disposition: restore it and re-raise so the process dies with the correct
    # signal (exit status visible to supervisors).
    try:
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except Exception:
        raise SystemExit(128 + int(signum)) from None


def install_exit_flush_signal_handlers() -> bool:
    """Install chaining SIGTERM/SIGINT flush handlers (main thread only). Called before
    uvicorn takes over signals: its ``capture_signals()`` saves these as the "original"
    handlers and re-raises into them after graceful shutdown, so the flush also covers
    terminations outside uvicorn's serve window. Idempotent; False off-main-thread/on failure."""
    global _exit_flush_handlers_installed
    if _exit_flush_handlers_installed:
        return True
    if threading.current_thread() is not threading.main_thread():
        return False
    import signal as _signal

    installed = False
    for signum in (_signal.SIGTERM, _signal.SIGINT):
        try:
            prev = _signal.getsignal(signum)
            _signal.signal(signum, _handle_exit_flush_signal)
            _exit_flush_prev_handlers[signum] = prev
            installed = True
        except (ValueError, OSError, RuntimeError):
            continue
    _exit_flush_handlers_installed = installed
    return installed


def _transport_is_dead(transport) -> bool:
    # _detached_ws_transport is the post-disconnect drop sentinel. _stdio_transport is the
    # REAL transport for standalone `hermes --tui` and must NOT count as dead.
    return transport is _detached_ws_transport or getattr(transport, "_closed", None) is True


def _reaper_session_is_detached_idle(sid: str, session: dict) -> bool:
    """Shared hard exemptions for both reapers: never evict a session mid-turn, awaiting
    input, still building, owning live delegated work, or on a live transport. Lazy watch
    sessions never start a build, so their unset agent_ready must not make them immortal."""
    if session.get("running") or _session_pending_kind(sid) or _session_has_active_delegations(sid, session):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    return _transport_is_dead(session.get("transport"))


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    if not _reaper_session_is_detached_idle(sid, session):
        return False
    last_active = float(session.get("last_active") or 0.0)
    created_at = float(session.get("created_at") or 0.0)
    return (now - last_active) > _SESSION_TTL_S and (now - created_at) > _SESSION_TTL_S


def _reap_idle_sessions() -> None:
    now = time.time()
    try:  # piggyback the incremental flush on the reaper tick — no new timer subsystem
        _flush_dirty_sessions()
    except Exception:
        logger.debug("periodic incremental session flush failed", exc_info=True)
    with _sessions_lock:
        victims = [sid for sid, s in _sessions.items() if _session_is_evictable(sid, s, now)]
    for sid in victims:
        _close_session_by_id(
            sid,
            end_reason="idle_timeout",
            predicate=lambda session, vs=sid: _session_is_evictable(vs, session, time.time()),
        )
    _enforce_session_cap()
    _reclaim_orphaned_leases()
    # Long-lived processes: gen2 GC rarely runs at steady state and glibc retains freed
    # pages as RSS, so trim every scan to prevent unbounded RSS growth over days/weeks.
    try:
        from hermes_cli.mem_trim import trim_memory

        trim_memory(reason="idle reaper periodic trim")
    except Exception as exc:
        # debug, not warning — a persistent failure would repeat every scan.
        logger.debug("idle reaper memory trim failed: %s: %s", type(exc).__name__, exc)


def _reclaim_orphaned_leases() -> None:
    """Hand the registry the lease ids we still own so it can drop the rest."""
    try:
        from hermes_cli.active_sessions import release_orphaned_leases

        with _sessions_lock:
            live = {
                lease.lease_id
                for session in _sessions.values()
                if (lease := session.get("active_session_lease")) is not None
            }
        if dropped := release_orphaned_leases(live):
            logger.info("Reclaimed %d orphaned active-session lease(s)", dropped)
    except Exception:
        logger.debug("orphaned lease reclaim failed", exc_info=True)


# Soft LRU cap on in-memory sessions: the TTL reaper only frees sessions idle for hours,
# so a heavy reconnecting user accumulates resident detached agents. The cap evicts the
# least-recently-active DETACHED sessions sooner — never a running / pending / mid-build /
# live-transport one (reopening re-resumes from the DB). 0/null disables.
def _max_live_sessions() -> int:
    try:
        from hermes_cli.active_sessions import coerce_max_concurrent_sessions

        cfg = _load_cfg() or {}
        raw = cfg.get("max_live_sessions")
        if raw is None:
            gateway_cfg = cfg.get("gateway")
            if isinstance(gateway_cfg, dict):
                raw = gateway_cfg.get("max_live_sessions")
        coerced = coerce_max_concurrent_sessions(raw, key="max_live_sessions")
        return int(coerced) if coerced else 0
    except Exception:
        return 0


def _session_is_lru_evictable(sid: str, session: dict) -> bool:
    # TTL-reaper exemptions WITHOUT the age gate: eligible the moment it loses its client.
    return _reaper_session_is_detached_idle(sid, session)


def _enforce_session_cap() -> None:
    cap = _max_live_sessions()
    if cap <= 0:
        return
    with _sessions_lock:
        if len(_sessions) <= cap:
            return
        evictable = [(sid, s) for sid, s in _sessions.items() if _session_is_lru_evictable(sid, s)]
    # Oldest-touched first; evict only down to the cap (may stop short: live sessions are never eligible).
    evictable.sort(key=lambda kv: float(kv[1].get("last_active") or 0.0))
    for sid, _s in evictable:
        with _sessions_lock:
            if len(_sessions) <= cap:
                break
        _close_session_by_id(
            sid,
            end_reason="lru_evict",
            predicate=lambda session, vs=sid: _session_is_lru_evictable(vs, session),
        )


def _reaper_daemon_timer(delay: float, fn) -> None:
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    timer.start()


def _schedule_session_cap_enforcement() -> None:
    """Run the LRU sweep off the response path (eviction can call agent.close)."""

    def _run():
        try:
            _enforce_session_cap()
        except Exception:
            logger.debug("session cap enforcement failed", exc_info=True)

    _reaper_daemon_timer(0.1, _run)


# ── Startup sweep for orphaned session rows ──────────────────────────────
# The WS-orphan reaper is an in-process Timer: a gateway restart kills it before it fires, leaving
# the row `ended_at IS NULL` forever. Scheduled once per process from both gateway entry points
# (stdio `entry.main`, WS sidecar `handle_ws`). state.db is shared by sibling processes on the same
# profile, so eligibility is conservative. Disable via `dashboard.startup_orphan_sweep: false`.
_ORPHAN_SWEEP_SOURCES = ("tui", "desktop", "subagent")
_startup_orphan_sweep_ran = False
_startup_orphan_sweep_lock = threading.Lock()


def _session_orphan_reaper_enabled() -> bool:
    """``dashboard.startup_orphan_sweep`` (default on). Fail-open on errors and
    on a missing key (raw yaml, no DEFAULT_CONFIG merge on this loader)."""
    try:
        dashboard_cfg = (_load_cfg() or {}).get("dashboard") or {}
        if isinstance(dashboard_cfg, dict) and "startup_orphan_sweep" in dashboard_cfg:
            return is_truthy_value(dashboard_cfg.get("startup_orphan_sweep"), default=True)
        return True
    except Exception:
        return True


def _live_session_ids() -> list[str]:
    """Session ids this process currently holds in memory."""
    ids: set[str] = set()
    with _sessions_lock:
        for sid, session in _sessions.items():
            if sid:
                ids.add(str(sid))
            if not isinstance(session, dict):
                continue
            for candidate in (getattr(session.get("agent"), "session_id", None), session.get("session_key")):
                if candidate:
                    ids.add(str(candidate))
    return sorted(ids)


def _sweep_orphaned_session_rows() -> list[str]:
    """End orphaned tui/desktop/subagent rows left by a dead process.

    "Provably orphaned" is inferred conservatively: the row must have been created AND last
    messaged at least the session TTL ago (a fresh row that copied an old transcript is
    protected by its own ``started_at``). Rows held in memory (e.g. a ``session.resume`` in
    the startup grace window) are excluded. Cross-backend: the sweep refuses to close a row
    any live backend (heartbeat within ``2 * TTL``) could own — see
    ``SessionDB.sweep_orphaned_sessions``.
    """
    db = _get_db()
    ttl = _SESSION_TTL_S
    if db is None or ttl <= 0:
        return []
    swept = db.sweep_orphaned_sessions(
        max_idle_seconds=ttl, sources=_ORPHAN_SWEEP_SOURCES, exclude_ids=tuple(_live_session_ids())
    )
    if swept:
        logger.info(
            "Closed %d orphaned session row(s) from a previous gateway process (startup_orphan_reap): %s",
            len(swept), ", ".join(swept),
        )
    return swept


# ── Cross-backend heartbeat ──────────────────────────────────────────────
# Each serve / gateway process registers a heartbeat row in ``gateway_heartbeats`` so the startup
# sweep can tell "owned by a live but idle backend" from "truly orphaned" (else the first process to
# restart reaped every inactive row of the other N−1). Refresh 60s default — far shorter than the
# 6h TTL so a refresh always lands inside the staleness window. Removed at exit; a crashed row ages out.

_HEARTBEAT_REFRESH_S = max(0.0, env_float("HERMES_GATEWAY_HEARTBEAT_REFRESH_S", 60.0))

_heartbeat_refresher_started = False
_heartbeat_refresher_lock = threading.Lock()


def _reaper_hostname() -> str:
    return os.uname().nodename if hasattr(os, "uname") else "host"


def _backend_id_for_this_process() -> str:
    """Stable identity for this process's heartbeat row: pid (readability) AND a startup
    nonce so a PID-reuse respawn cannot inherit the dead predecessor's heartbeat."""
    nonce = getattr(_backend_id_for_this_process, "_nonce", None)
    if nonce is None:
        import secrets as _secrets

        nonce = _secrets.token_hex(4)
        try:
            setattr(_backend_id_for_this_process, "_nonce", nonce)
        except AttributeError:  # pragma: no cover - defensive
            pass
    return f"{_current_profile_name()}@{_reaper_hostname()}:{os.getpid()}:{nonce}"


def _refresh_backend_heartbeat() -> None:
    """Refresh this backend's heartbeat row. No-op when DB unavailable."""
    db = _get_db()
    if db is None:
        return
    try:
        db.register_backend_heartbeat(
            backend_id=_backend_id_for_this_process(),
            pid=os.getpid(),
            started_at=_gateway_started_at(),
            profile=_current_profile_name(),
            host=_reaper_hostname(),
        )
    except Exception:
        logger.debug("backend heartbeat refresh failed", exc_info=True)


def _gateway_started_at() -> float:
    """Wall-clock time this process started (first-call time is a good-enough
    proxy: the heartbeat refresher runs after the gateway is fully wired up)."""
    started = getattr(_gateway_started_at, "_t", None)
    if started is None:
        started = time.time()
        try:
            setattr(_gateway_started_at, "_t", started)
        except AttributeError:  # pragma: no cover
            pass
    return started


def _heartbeat_refresher_loop(stop_event: threading.Event) -> None:
    """Background loop that refreshes the heartbeat on a fixed cadence."""
    while not stop_event.is_set():
        try:
            _refresh_backend_heartbeat()
        except Exception:
            logger.debug("heartbeat refresh loop iteration failed", exc_info=True)
        stop_event.wait(_HEARTBEAT_REFRESH_S)


def _start_backend_heartbeat_refresher() -> None:
    """Register this backend and start the refresher thread (once per process). The first
    refresh writes the row synchronously so this process's own sweep sees itself in the
    heartbeat table. ``_HEARTBEAT_REFRESH_S <= 0`` means "register once, never refresh"."""
    global _heartbeat_refresher_started
    with _heartbeat_refresher_lock:
        if _heartbeat_refresher_started:
            return
        _heartbeat_refresher_started = True
    try:
        _refresh_backend_heartbeat()
    except Exception:
        logger.debug("initial backend heartbeat write failed", exc_info=True)
    if _HEARTBEAT_REFRESH_S <= 0:
        return
    stop_event = threading.Event()

    def _atexit_clear():
        stop_event.set()
        with contextlib.suppress(Exception):
            db = _get_db()
            if db is not None:
                db.clear_backend_heartbeat(_backend_id_for_this_process())

    atexit.register(_atexit_clear)
    threading.Thread(
        target=_heartbeat_refresher_loop, args=(stop_event,), name="hermes-gateway-heartbeat", daemon=True
    ).start()


def _schedule_startup_orphan_sweep() -> None:
    """Schedule the once-per-process startup orphan sweep, delayed by the WS-orphan grace
    window so a client reconnecting right after a restart can ``session.resume`` its row
    first. Grace 0 (park forever), TTL 0 and ``dashboard.startup_orphan_sweep: false``
    all suppress the sweep."""
    global _startup_orphan_sweep_ran
    if _WS_ORPHAN_REAP_GRACE_S <= 0 or _SESSION_TTL_S <= 0 or not _session_orphan_reaper_enabled():
        return
    if _startup_orphan_sweep_ran:
        return
    with _startup_orphan_sweep_lock:
        if _startup_orphan_sweep_ran:
            return
        _startup_orphan_sweep_ran = True

    def _run() -> None:
        try:
            _sweep_orphaned_session_rows()
        except Exception:
            logger.warning("startup orphan session sweep failed", exc_info=True)

    _reaper_daemon_timer(_WS_ORPHAN_REAP_GRACE_S, _run)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
