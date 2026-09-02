"""Session flush / reaping / orphan sweep / cross-backend heartbeat: exit-flush signal handlers, idle + LRU eviction, orphaned session-row sweep, backend heartbeat refresher.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import threading

from tui_gateway._env import env_float

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# Knobs (_SESSION_TTL_S, _REAPER_SCAN_S, _EXIT_FLUSH_BUDGET_S,
# _INCREMENTAL_FLUSH_INTERVAL_S) live in server.py next to _start_idle_reaper.

# ── Flush-on-kill + periodic incremental flush (#94724 item 2) ───────────
# A `hermes serve` killed mid-update used to lose every un-flushed in-memory
# session: the next RPC failed with "session-scoped RPC rejected: not in
# memory (detached/reaped runtime)" and NO store held the transcript. #95576
# made serves survive *future* updates; this closes the kill path itself:
#   (a) SIGTERM/SIGINT run a bounded, best-effort flush of in-memory session
#       transcripts to state.db BEFORE the normal shutdown path, chained to
#       whatever handler was installed before (uvicorn's included);
#   (b) the idle-reaper scan piggybacks a periodic incremental flush so even
#       a SIGKILL loses at most one flush interval.


def _flush_session_messages(session: dict | None) -> bool:
    """Best-effort durable flush of one session's in-memory transcript.

    Rides ``agent._persist_session`` — the same marker-deduped persist
    contract ``_finalize_session`` uses (#13121) — so repeated calls only
    write genuinely-unflushed messages and never duplicate durable rows.
    """
    if not session:
        return False
    agent = session.get("agent")
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


def _flush_dirty_sessions(now: float | None = None) -> int:
    """Periodic incremental flush, driven by the idle-reaper scan.

    Skips ``running`` sessions: the turn thread owns mid-turn persistence
    (it already flushes at every persist point) and
    ``_drop_trailing_empty_response_scaffolding`` mutates the live message
    list, so racing an in-flight turn from the reaper thread is never safe.
    Idle/detached sessions — precisely the ones a kill strands — are flushed
    at most once per ``_INCREMENTAL_FLUSH_INTERVAL_S``. ``now`` is injectable
    for tests (monotonic clock).
    """
    if _INCREMENTAL_FLUSH_INTERVAL_S <= 0:
        return 0
    if now is None:
        now = time.monotonic()
    with _sessions_lock:
        sessions = list(_sessions.values())
    flushed = 0
    for session in sessions:
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
    """Bounded flush of ALL in-memory sessions on the way out.

    Runs on a daemon worker joined with the budget so a hung SQLite write
    can never block exit longer than ``HERMES_TUI_EXIT_FLUSH_BUDGET_S``
    (default 5s). Running sessions are included — the process is dying, so
    a best-effort partial transcript beats guaranteed loss.
    """
    budget = _EXIT_FLUSH_BUDGET_S if budget_s is None else max(0.0, budget_s)
    if budget <= 0:
        return 0
    result = {"flushed": 0}

    def _run() -> None:
        deadline = time.monotonic() + budget
        with _sessions_lock:
            sessions = list(_sessions.values())
        for session in sessions:
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
    """Flush in-memory sessions, then hand off to the prior handler.

    Chaining preserves the pre-existing signal story (uvicorn's graceful
    shutdown, a supervisor's handler, or the default terminate disposition)
    — this handler only *prepends* a bounded durable flush.
    """
    try:
        _flush_sessions_before_exit()
    except Exception:
        pass
    import signal as _signal

    prev = _exit_flush_prev_handlers.get(signum)
    if callable(prev):
        prev(signum, frame)
        return
    if prev is _signal.SIG_IGN:
        return
    # Default disposition: restore it and re-raise so the process still dies
    # with the correct signal (exit status visible to supervisors).
    try:
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except Exception:
        raise SystemExit(128 + int(signum)) from None


def install_exit_flush_signal_handlers() -> bool:
    """Install chaining SIGTERM/SIGINT flush handlers (main thread only).

    Called by ``hermes serve`` / dashboard startup before uvicorn takes over
    signals: uvicorn's ``capture_signals()`` saves these as the "original"
    handlers and restores + re-raises into them after its graceful shutdown,
    so the flush also covers terminations outside uvicorn's serve window.
    Idempotent; returns False off the main thread or when installation fails.
    """
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
    # _detached_ws_transport is the post-WS-disconnect drop sentinel; a session
    # parked on it has no live client. _stdio_transport is the REAL transport
    # for a standalone `hermes --tui`, so it must NOT count as dead here (doing
    # so let the idle reaper evict healthy standalone TUI sessions).
    if transport is _detached_ws_transport:
        return True
    return getattr(transport, "_closed", None) is True


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    if session.get("running") or _session_pending_kind(sid):
        return False
    if _session_has_active_delegations(sid, session):
        return False
    ready = session.get("agent_ready")
    # Lazy watch sessions (subagent spectator windows) never start a build,
    # so their forever-unset agent_ready must not make them immortal.
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    if not _transport_is_dead(session.get("transport")):
        return False
    last_active = float(session.get("last_active") or 0.0)
    created_at = float(session.get("created_at") or 0.0)
    return (now - last_active) > _SESSION_TTL_S and (now - created_at) > _SESSION_TTL_S


def _reap_idle_sessions() -> None:
    now = time.time()
    # Piggyback the periodic incremental flush on the existing reaper tick
    # (#94724 item 2) — no new timer subsystem. Even a SIGKILL then loses at
    # most one flush interval of un-persisted messages.
    try:
        _flush_dirty_sessions()
    except Exception:
        logger.debug("periodic incremental session flush failed", exc_info=True)
    with _sessions_lock:
        victims = [sid for sid, s in _sessions.items() if _session_is_evictable(sid, s, now)]
    for sid in victims:
        _close_session_by_id(
            sid,
            end_reason="idle_timeout",
            predicate=lambda session, victim_sid=sid: _session_is_evictable(
                victim_sid, session, time.time()
            ),
        )
    _enforce_session_cap()
    _reclaim_orphaned_leases()
    # Periodic heap release for long-lived gateway processes.  Even when no
    # session is reaped, Python's generational GC rarely runs gen2 collection
    # under steady-state allocation, and glibc retains freed pages as RSS.
    # Calling trim_memory here ensures every reaper scan (default every 5 min)
    # returns releasable pages, preventing unbounded RSS growth over days/weeks.
    try:
        from hermes_cli.mem_trim import trim_memory

        trim_memory(reason="idle reaper periodic trim")
    except Exception as exc:
        # debug, not warning — persistent failure would repeat every reaper
        # scan (300s) forever; sibling failure branches log at debug.
        logger.debug(
            "idle reaper memory trim failed: %s: %s", type(exc).__name__, exc
        )


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


# Soft LRU cap on in-memory sessions. The 6h TTL reaper above only frees
# sessions that have been idle for hours; a heavy user who reconnects often
# accumulates detached sessions (the report's ``detached_sessions=5``) whose
# agents sit resident for the full TTL. The cap evicts the least-recently-active
# DETACHED sessions sooner so live agents don't pile up under memory pressure.
# Default-on but provably safe: it only touches sessions with no live client
# (reopening re-resumes them from the DB) and never a running / pending /
# mid-build / live-transport one. 0/null disables.
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
    # Same hard exemptions as the TTL reaper (never evict a session mid-turn,
    # awaiting input, still building, or owning active delegated work), but
    # WITHOUT the hours-scale age gate: a detached session is eligible the
    # moment it loses its client.
    if session.get("running") or _session_pending_kind(sid):
        return False
    if _session_has_active_delegations(sid, session):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    return _transport_is_dead(session.get("transport"))


def _enforce_session_cap() -> None:
    cap = _max_live_sessions()
    if cap <= 0:
        return
    with _sessions_lock:
        total = len(_sessions)
        if total <= cap:
            return
        evictable = [
            (sid, s) for sid, s in _sessions.items() if _session_is_lru_evictable(sid, s)
        ]
    # Oldest-touched first; only evict down to the cap (live/focused sessions on
    # a live transport are never eligible, so we may stop short of the cap).
    evictable.sort(key=lambda kv: float(kv[1].get("last_active") or 0.0))
    for sid, _s in evictable:
        with _sessions_lock:
            if len(_sessions) <= cap:
                break
        _close_session_by_id(
            sid,
            end_reason="lru_evict",
            predicate=lambda session, victim_sid=sid: _session_is_lru_evictable(
                victim_sid, session
            ),
        )


def _schedule_session_cap_enforcement() -> None:
    """Run the LRU sweep off the response path (eviction can call agent.close)."""

    def _run():
        try:
            _enforce_session_cap()
        except Exception:
            logger.debug("session cap enforcement failed", exc_info=True)

    timer = threading.Timer(0.1, _run)
    timer.daemon = True
    timer.start()




# ── Startup sweep for orphaned session rows (#65194) ─────────────────────
# The WS-orphan reaper above is an in-process threading.Timer: a gateway
# restart (update, crash, systemd) kills it before it fires, leaving the
# session row `ended_at IS NULL` forever. This is the startup complement
# every other resource type already has (docker_orphan_reaper, compression
# orphans). Scheduled once per process from both gateway entry points
# (stdio `entry.main` and the WS sidecar's `handle_ws`) — desktop/dashboard
# never run `entry.main()`. state.db is shared by sibling processes on the
# same profile, so eligibility is conservative. Disable via
# `dashboard.startup_orphan_sweep: false` (default on).
_ORPHAN_SWEEP_SOURCES = ("tui", "desktop", "subagent")
_startup_orphan_sweep_ran = False
_startup_orphan_sweep_lock = threading.Lock()


def _session_orphan_reaper_enabled() -> bool:
    """``dashboard.startup_orphan_sweep`` (default on). Fail-open on errors."""
    try:
        dashboard_cfg = (_load_cfg() or {}).get("dashboard") or {}
        if isinstance(dashboard_cfg, dict) and "startup_orphan_sweep" in dashboard_cfg:
            return is_truthy_value(
                dashboard_cfg.get("startup_orphan_sweep"), default=True
            )
        # Fail-open: a missing key (raw yaml, no DEFAULT_CONFIG merge on
        # this loader) must keep the sweep on.
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
            agent = session.get("agent") if isinstance(session, dict) else None
            for candidate in (
                getattr(agent, "session_id", None),
                session.get("session_key") if isinstance(session, dict) else None,
            ):
                if candidate:
                    ids.add(str(candidate))
    return sorted(ids)


def _sweep_orphaned_session_rows() -> list[str]:
    """End orphaned tui/desktop/subagent rows left by a dead process.

    "Provably orphaned" is inferred conservatively from inactivity — the
    row must have been created AND last messaged at least the session TTL
    ago (``HERMES_TUI_SESSION_TTL_S``). A freshly created row that copied
    an old transcript is protected by its own ``started_at``. Rows this
    process still holds in memory (e.g. a ``session.resume`` during the
    startup grace window) are excluded so the sweep never races a
    mid-reconnect client.

    Cross-backend liveness (#94895): when one ``state.db`` is shared by
    N serve / gateway processes, each registered a heartbeat row in
    ``gateway_heartbeats``. The sweep refuses to close a row that any
    live backend (heartbeat refreshed within ``2 * TTL``) could
    plausibly own — see ``SessionDB.sweep_orphaned_sessions`` for the
    exact predicate.
    """
    db = _get_db()
    if db is None:
        return []
    ttl = _SESSION_TTL_S
    if ttl <= 0:
        return []
    swept = db.sweep_orphaned_sessions(
        max_idle_seconds=ttl,
        sources=_ORPHAN_SWEEP_SOURCES,
        exclude_ids=tuple(_live_session_ids()),
    )
    if swept:
        logger.info(
            "Closed %d orphaned session row(s) from a previous gateway "
            "process (startup_orphan_reap): %s",
            len(swept),
            ", ".join(swept),
        )
    return swept


# ── Cross-backend heartbeat (#94895) ───────────────────────────────────
# Each serve / gateway process registers a heartbeat row in
# ``gateway_heartbeats`` so the startup orphan sweep can tell "row owned
# by a live but idle backend" from "row truly orphaned".  Without this,
# the first process to restart in a multi-backend topology reaped every
# inactive row — including those held by the other N−1 still-running
# processes (the #94895 reporter saw 473 sessions disappear in one shot).
#
# Refresh cadence: every HEARTBEAT_REFRESH_S (default 60s — much shorter
# than the default 6h session TTL so a refresh always lands inside the
# staleness window).  The heartbeat is removed at process exit so a
# graceful shutdown doesn't leave a stale row behind.  A crashed process
# leaves its row until the heartbeat ages out of the staleness window,
# at which point the sweep treats it as dead.

_HEARTBEAT_REFRESH_S = max(0.0, env_float("HERMES_GATEWAY_HEARTBEAT_REFRESH_S", 60.0))

_heartbeat_refresher_started = False
_heartbeat_refresher_lock = threading.Lock()


def _backend_id_for_this_process() -> str:
    """Stable identity for this process's heartbeat row (#94895).

    Includes pid AND a startup-time nonce so a PID-reuse respawn cannot
    inherit the dead predecessor's heartbeat and protect truly orphaned
    sessions.  The pid is kept for human readability in diagnostics.
    """
    nonce = getattr(_backend_id_for_this_process, "_nonce", None)
    if nonce is None:
        import secrets as _secrets

        nonce = _secrets.token_hex(4)
        try:
            setattr(_backend_id_for_this_process, "_nonce", nonce)
        except AttributeError:  # pragma: no cover - defensive
            pass
    return f"{_current_profile_name()}@{os.uname().nodename if hasattr(os, 'uname') else 'host'}:{os.getpid()}:{nonce}"


def _refresh_backend_heartbeat() -> None:
    """Refresh this backend's heartbeat row (#94895). No-op when DB unavailable."""
    db = _get_db()
    if db is None:
        return
    try:
        db.register_backend_heartbeat(
            backend_id=_backend_id_for_this_process(),
            pid=os.getpid(),
            started_at=_gateway_started_at(),
            profile=_current_profile_name(),
            host=(os.uname().nodename if hasattr(os, "uname") else "host"),
        )
    except Exception:
        logger.debug("backend heartbeat refresh failed", exc_info=True)


def _gateway_started_at() -> float:
    """Wall-clock time when this process started. Module-import time is
    a good-enough proxy: the heartbeat refresher runs after the gateway
    is fully wired up.
    """
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
    """Register this backend and start the refresher thread (#94895).

    Called once per process from both gateway entry points.  The first
    refresh writes the row immediately so even a very fast crash leaves
    a fresh-enough row that other backends can see.  Repeat calls are
    no-ops.  The refresher thread is only spawned when
    ``_HEARTBEAT_REFRESH_S > 0`` — a refresh interval of zero means
    "register the row once, never refresh" (the row ages out naturally
    after the heartbeat staleness window).
    """
    global _heartbeat_refresher_started
    with _heartbeat_refresher_lock:
        if _heartbeat_refresher_started:
            return
        _heartbeat_refresher_started = True
    # Write a row synchronously so the sweep run later in this same
    # process can see ourselves in the heartbeat table too.  Without
    # this, exclude_ids would have to cover every local session — a
    # regression in the strict-ownership case the heartbeat exists to fix.
    try:
        _refresh_backend_heartbeat()
    except Exception:
        logger.debug("initial backend heartbeat write failed", exc_info=True)
    if _HEARTBEAT_REFRESH_S <= 0:
        return
    stop_event = threading.Event()

    def _atexit_clear():
        stop_event.set()
        try:
            db = _get_db()
            if db is not None:
                db.clear_backend_heartbeat(_backend_id_for_this_process())
        except Exception:
            pass

    atexit.register(_atexit_clear)
    thread = threading.Thread(
        target=_heartbeat_refresher_loop,
        args=(stop_event,),
        name="hermes-gateway-heartbeat",
        daemon=True,
    )
    thread.start()


def _schedule_startup_orphan_sweep() -> None:
    """Schedule the once-per-process startup orphan sweep (#65194).

    Called from both gateway entry points. Repeat calls are no-ops. The
    sweep is delayed by the WS-orphan grace window so a client reconnecting
    right after a restart can ``session.resume`` its row before the sweep
    reads the DB. ``HERMES_TUI_WS_ORPHAN_REAP_GRACE_S=0`` (park forever)
    and ``HERMES_TUI_SESSION_TTL_S=0`` both suppress the sweep; so does
    ``dashboard.startup_orphan_sweep: false``.
    """
    global _startup_orphan_sweep_ran
    if _WS_ORPHAN_REAP_GRACE_S <= 0 or _SESSION_TTL_S <= 0:
        return
    if not _session_orphan_reaper_enabled():
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

    timer = threading.Timer(_WS_ORPHAN_REAP_GRACE_S, _run)
    timer.daemon = True
    timer.start()


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
