"""Browser session lifecycle: inactivity janitor, orphan reaper, per-session teardown, atexit emergency cleanup.

Split out of ``tools/browser_tool.py``; every name is re-imported there so
``tools.browser_tool.<name>`` keeps resolving (and monkeypatching). Origin
symbols and module state are read/written through ``_bt`` (the origin module,
resolved per call by :func:`tools.browser_tool_origin.origin_module`) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

import contextlib
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.browser_tool_origin import origin_module as _origin


def _session_expiry_timestamp(session_info: Dict[str, Any]) -> Optional[float]:
    """Return a provider-authoritative session expiry as epoch seconds.

    Cloud providers may omit ``expires_at``. Unknown or malformed values are
    therefore treated as having no known expiry, preserving the existing
    lifecycle for local browsers and providers without an expiry contract.
    """
    _bt = _origin()
    value = session_info.get("expires_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        _bt.logger.warning("Ignoring invalid cloud browser session expiry timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _session_has_expired(
    session_info: Dict[str, Any], *, now: Optional[float] = None
) -> bool:
    """Return whether a cached browser session crossed its provider deadline."""
    _bt = _origin()
    expires_at = _bt._session_expiry_timestamp(session_info)
    if expires_at is None:
        return False
    return (time.time() if now is None else now) >= expires_at


def _emergency_cleanup_all_sessions():
    """
    Emergency cleanup of all active browser sessions.
    Called on process exit or interrupt to prevent orphaned sessions.

    Also runs the orphan reaper to clean up daemons left behind by previously
    crashed hermes processes — this way every clean hermes exit sweeps
    accumulated orphans, not just ones that actively used the browser tool.
    """
    _bt = _origin()
    if _bt._cleanup_done:
        return
    _bt._cleanup_done = True

    # Clean up this process's own sessions first, so their owner_pid files
    # are removed before the reaper scans.
    # Real-profile Chrome processes are launched directly (not by
    # agent-browser), so the session cleanup below never reaps them.
    try:
        _bt._terminate_real_profile_chrome()
    except Exception as e:
        _bt.logger.debug("Real-profile chrome cleanup on exit failed: %s", e)
    if _bt._active_sessions:
        _bt.logger.info("Emergency cleanup: closing %s active session(s)...",
                    len(_bt._active_sessions))
        try:
            _bt.cleanup_all_browsers()
        except Exception as e:
            _bt.logger.error("Emergency cleanup error: %s", e)
        finally:
            with _bt._cleanup_lock:
                _bt._active_sessions.clear()
                _bt._session_last_activity.clear()
                _bt._session_owner_homes.clear()
                _bt._cleanup_failures.clear()
                _bt._recording_sessions.clear()

    # Lightpanda servers (Browser Use mode) are processes we spawned; the
    # session cleanup above stops the tracked ones, this catches any that
    # fell out of ``_active_sessions``.
    try:
        from tools.browser_lightpanda import stop_all_lightpanda

        stop_all_lightpanda()
    except Exception as e:
        _bt.logger.debug("Lightpanda cleanup on exit failed: %s", e)

    # Sweep orphans from other crashed hermes processes.  Safe even if we
    # never used the browser — uses owner_pid liveness to avoid reaping
    # daemons owned by other live hermes processes.
    try:
        _bt._reap_orphaned_browser_sessions()
    except Exception as e:
        _bt.logger.debug("Orphan reap on exit failed: %s", e)


@contextlib.contextmanager
def _session_owner_scope(task_id: str):
    """Run under the Hermes home + secret scope owning ``task_id``'s session (no-op if unrecorded).

    The janitor thread is process-global, so each teardown must re-enter its
    OWN profile's scope rather than inherit the spawning profile's; never falls
    through to ``os.environ``.
    """
    _bt = _origin()
    owner_home = _bt._session_owner_homes.get(task_id)
    if owner_home is None:
        yield
        return

    from agent.secret_scope import (
        build_profile_secret_scope, reset_secret_scope, set_secret_scope
    )
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(owner_home)
    try:
        hydrate_profile_secret_sources(Path(owner_home))
        secret_token = set_secret_scope(build_profile_secret_scope(Path(owner_home)))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def _cleanup_inactive_browser_sessions():
    """Close sessions inactive longer than the timeout (called by the cleanup thread).

    Each session is torn down under its owner profile's scope. A session whose
    cleanup keeps failing is force-reaped after MAX_INACTIVITY_CLEANUP_FAILURES
    attempts instead of retrying forever; only a successful cleanup clears its
    failure count.
    """
    _bt = _origin()
    current_time = time.time()
    sessions_to_cleanup = []

    with _bt._cleanup_lock:
        for task_id, last_time in list(_bt._session_last_activity.items()):
            if current_time - last_time > _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT:
                sessions_to_cleanup.append(task_id)

    for task_id in sessions_to_cleanup:
        elapsed = int(current_time - _bt._session_last_activity.get(task_id, current_time))
        _bt.logger.info("Cleaning up inactive session for task: %s (inactive for %ss)", task_id, elapsed)
        try:
            with _bt._session_owner_scope(task_id):
                _bt.cleanup_browser(task_id)
            with _bt._cleanup_lock:
                _bt._session_last_activity.pop(task_id, None)
                _bt._session_owner_homes.pop(task_id, None)
                _bt._cleanup_failures.pop(task_id, None)
        except Exception as e:
            with _bt._cleanup_lock:
                failures = _bt._cleanup_failures[task_id] = _bt._cleanup_failures.get(task_id, 0) + 1
            if failures < _bt.MAX_INACTIVITY_CLEANUP_FAILURES:
                _bt.logger.warning("Error cleaning up inactive session %s (attempt %d/%d): %s",
                               task_id, failures, _bt.MAX_INACTIVITY_CLEANUP_FAILURES, e)
                continue
            _bt.logger.error("Browser cleanup failed %d times for inactive session %s; "
                         "force-reaping: %s", failures, task_id, e)
            try:
                with _bt._session_owner_scope(task_id):
                    _bt._force_reap_browser_session(task_id)
            except Exception as reap_exc:
                _bt.logger.error("Force-reap of browser session %s failed: %s", task_id, reap_exc)
            finally:
                with _bt._cleanup_lock:
                    _bt._session_owner_homes.pop(task_id, None)
                    _bt._cleanup_failures.pop(task_id, None)


def _write_owner_pid(socket_dir: str, session_name: str) -> None:
    """Record the current hermes PID as the owner of a browser socket dir.

    Written atomically to ``<socket_dir>/<session_name>.owner_pid`` so the
    orphan reaper can distinguish daemons owned by a live hermes process
    (don't reap) from daemons whose owner crashed (reap).  Best-effort —
    an OSError here just falls back to the legacy ``tracked_names``
    heuristic in the reaper.
    """
    _bt = _origin()
    try:
        path = os.path.join(socket_dir, f"{session_name}.owner_pid")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        _bt.logger.debug("Could not write owner_pid file for %s: %s",
                     session_name, exc)


def _verify_reapable_browser_daemon(daemon_pid: int, socket_dir: str,
                                    session_name: str) -> bool:
    """Confirm a live PID is genuinely *this* session's agent-browser daemon.

    The ``.pid`` file lives in a world-writable, predictably-named temp dir and
    is written by the daemon, not us: a same-user actor can plant one pointing
    at a victim PID, or a recycled PID can land on an unrelated process — and
    reaping is a *tree* kill, i.e. an arbitrary-process DoS. Two psutil checks
    must both pass: (1) identity — ``agent-browser`` in the name or cmdline;
    (2) binding — the socket dir path/basename in the cmdline, or
    ``AGENT_BROWSER_SOCKET_DIR`` in its environ. (2) is the real spoof defense:
    an attacker would need a real daemon embedding this exact path, which they
    could already signal. Fail-closed on any ambiguity (unreadable cmdline, no
    match): refuse to reap and leave process and socket dir alone.
    """
    _bt = _origin()
    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        _bt.logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "psutil unavailable for identity verification",
            daemon_pid, session_name)
        return False

    try:
        proc = psutil.Process(daemon_pid)
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline() or []).lower()
    except psutil.NoSuchProcess:
        # Vanished between the liveness check and now — nothing to reap.
        return False
    except (psutil.AccessDenied, OSError) as exc:
        _bt.logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "could not read process identity (%s)",
            daemon_pid, session_name, exc)
        return False

    looks_like_browser = "agent-browser" in name or "agent-browser" in cmdline
    if not looks_like_browser:
        _bt.logger.warning(
            "Refusing to reap PID %d (session %s): not an agent-browser "
            "process (name=%r)", daemon_pid, session_name, name)
        return False

    # Binding check: the live process must reference *this* socket dir.
    socket_dir_l = socket_dir.lower()
    socket_base_l = os.path.basename(socket_dir).lower()
    bound = socket_dir_l in cmdline or (
        socket_base_l and socket_base_l in cmdline)
    if not bound:
        try:
            env_dir = (proc.environ() or {}).get(
                "AGENT_BROWSER_SOCKET_DIR", "")
            bound = bool(env_dir) and os.path.normpath(env_dir) == \
                os.path.normpath(socket_dir)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            # environ() can be denied even same-user on some platforms.
            # cmdline already failed to bind — fail closed.
            bound = False

    if not bound:
        _bt.logger.warning(
            "Refusing to reap agent-browser PID %d: not bound to session "
            "socket dir %s (possible recycled PID or planted pid file)",
            daemon_pid, socket_dir)
        return False

    return True


def _socket_dir_idle_seconds(socket_dir: str) -> Optional[float]:
    """Seconds since anything in ``socket_dir`` was last written; None if unknown (fail safe).

    Every command writes ``_stdout_<cmd>`` / ``_stderr_<cmd>`` there, so the
    newest mtime is a last-activity marker that survives hermes restarts and
    lost in-memory bookkeeping. The dir's own mtime is not enough — rewriting
    an existing ``_stdout_click`` doesn't touch it — so entries are scanned too.
    """
    try:
        latest = os.path.getmtime(socket_dir)
    except OSError:
        return None

    try:
        with os.scandir(socket_dir) as entries:
            for entry in entries:
                try:
                    latest = max(latest, entry.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass  # dir mtime alone is still a usable lower bound

    return max(0.0, time.time() - latest)


def _owner_pid_alive(socket_dir: str, session_name: str) -> Tuple[Optional[int], Optional[bool]]:
    """Read ``<session>.owner_pid`` and report ``(pid, alive)``; ``(None, None)`` when missing/corrupt."""
    owner_pid_file = os.path.join(socket_dir, f"{session_name}.owner_pid")
    if not os.path.isfile(owner_pid_file):
        return None, None
    try:
        owner_pid = int(Path(owner_pid_file).read_text(encoding="utf-8").strip())
        # ``os.kill(pid, 0)`` is NOT a no-op on Windows; use the cross-platform check.
        from gateway.status import _pid_exists
        return owner_pid, _pid_exists(owner_pid)
    except (ValueError, OSError):
        return None, None  # corrupt file — fall through to legacy handling


def _reap_socket_dir(socket_dir: str, session_name: str, tracked_names: set) -> bool:
    """Reap one ``agent-browser-<session>`` dir if orphaned; return True when a daemon was killed.

    Ownership priority: (1) a live ``owner_pid`` means another hermes process
    owns it — leave it alone UNLESS it is untracked here and idle past
    ``BROWSER_ORPHAN_GRACE_SECONDS`` (owner-alive alone made leaked daemons
    immortal: in-memory tracking is lost on any exception path and the daemon's
    own idle timeout doesn't fire when it is wedged); (2) no owner_pid (legacy)
    falls back to this process's tracking. A pidless dir is only stale after the
    grace period — deleting it immediately races the creator's first stdout open.
    The daemon PID is verified as ours before a tree-kill (world-writable dir,
    recycled PIDs), and refused without a start-time fingerprint.
    """
    _bt = _origin()
    owner_pid, owner_alive = _bt._owner_pid_alive(socket_dir, session_name)
    if owner_alive is True:
        if session_name in tracked_names:
            return False
        idle_s = _bt._socket_dir_idle_seconds(socket_dir)
        if idle_s is None or idle_s < _bt.BROWSER_ORPHAN_GRACE_SECONDS:
            return False  # unknown age or within grace — fail safe
        _bt.logger.warning(
            "Browser session %s has a live owner (PID %s) but is untracked "
            "and idle for %ds (grace %ds) — treating as leaked and reaping",
            session_name, owner_pid, int(idle_s),
            _bt.BROWSER_ORPHAN_GRACE_SECONDS)
    elif owner_alive is None and session_name in tracked_names:
        return False

    pid_file = os.path.join(socket_dir, f"{session_name}.pid")
    if not os.path.isfile(pid_file):
        idle_s = _bt._socket_dir_idle_seconds(socket_dir)
        if idle_s is None or idle_s < _bt.BROWSER_ORPHAN_GRACE_SECONDS:
            return False
        shutil.rmtree(socket_dir, ignore_errors=True)
        return False

    try:
        daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        shutil.rmtree(socket_dir, ignore_errors=True)
        return False

    from gateway.status import _pid_exists
    if not _pid_exists(daemon_pid):
        shutil.rmtree(socket_dir, ignore_errors=True)
        return False

    if not _bt._verify_reapable_browser_daemon(daemon_pid, socket_dir, session_name):
        return False  # leave process and dir for a later sweep once the imposter PID is gone

    # Tree-kill so Chromium children (renderer, GPU, ...) go too, not just the daemon.
    reaped = False
    try:
        from gateway.status import get_process_start_time
        from tools.process_registry import ProcessRegistry
        daemon_start = get_process_start_time(daemon_pid)
        if daemon_start is None:
            _bt.logger.warning(
                "Refusing to reap browser daemon PID %d (session %s): "
                "no start-time fingerprint available", daemon_pid, session_name)
            return False
        ProcessRegistry._terminate_host_pid(daemon_pid, daemon_start)
        _bt.logger.info("Reaped orphaned browser daemon PID %d (session %s)",
                    daemon_pid, session_name)
        reaped = True
    except (ProcessLookupError, PermissionError, OSError):
        pass
    shutil.rmtree(socket_dir, ignore_errors=True)
    return reaped


def _reap_orphaned_browser_sessions():
    """Kill agent-browser daemons whose owning hermes process is gone.

    When the process that created a session exits uncleanly (SIGKILL, crash,
    gateway restart) the in-memory ``_active_sessions`` tracking is lost but the
    node + Chromium processes keep running. Scans the tmp dir for
    ``agent-browser-*`` socket dirs and applies ``_reap_socket_dir``'s ownership
    rules (owner_pid file first — cross-process safe, two hermes instances never
    reap each other — then in-process tracking for legacy daemons).
    Safe to call from any context — atexit, cleanup thread, or on demand.
    """
    _bt = _origin()
    import glob

    # Lightpanda servers (Browser Use mode) keep their own records (no
    # agent-browser socket dir); sweep them with the same owner-liveness rule
    # BEFORE the daemon scan, which may return early.
    try:
        from tools.browser_lightpanda import reap_orphaned_lightpanda

        reap_orphaned_lightpanda()
    except Exception as e:
        _bt.logger.debug("Lightpanda orphan reap failed: %s", e)

    tmpdir = _bt._socket_safe_tmpdir()
    socket_dirs = []
    for prefix in ("agent-browser-h_*", "agent-browser-cdp_*", "agent-browser-hermes_*"):
        socket_dirs += glob.glob(os.path.join(tmpdir, prefix))
    if not socket_dirs:
        return

    with _bt._cleanup_lock:
        tracked_names = {
            info.get("session_name")
            for info in _bt._active_sessions.values()
            if info.get("session_name")
        }

    reaped = 0
    for socket_dir in socket_dirs:
        session_name = os.path.basename(socket_dir).removeprefix("agent-browser-")
        if session_name and _bt._reap_socket_dir(socket_dir, session_name, tracked_names):
            reaped += 1

    if reaped:
        _bt.logger.info("Reaped %d orphaned browser session(s) from previous run(s)", reaped)


def _browser_cleanup_thread_worker():
    """Every 30s: close sessions idle past BROWSER_SESSION_INACTIVITY_TIMEOUT.

    Also reaps orphaned daemons on startup AND every BROWSER_ORPHAN_REAP_INTERVAL
    seconds — a daemon can fall out of in-memory tracking at any point in a
    long-lived process, and a startup-only reap could never recover from that.
    """
    _bt = _origin()
    reap_every_cycles = max(1, round(_bt.BROWSER_ORPHAN_REAP_INTERVAL / 30))
    cycle = 0

    while _bt._cleanup_running:
        # cycle 0 is the startup reap; then every reap_every_cycles.
        if cycle % reap_every_cycles == 0:
            try:
                _bt._reap_orphaned_browser_sessions()
            except Exception as e:
                _bt.logger.warning("Orphan reap error: %s", e)
        cycle += 1

        try:
            _bt._cleanup_inactive_browser_sessions()
        except Exception as e:
            _bt.logger.warning("Cleanup thread error: %s", e)

        # Sleep in 1-second intervals so we can stop quickly if needed
        for _ in range(30):
            if not _bt._cleanup_running:
                break
            time.sleep(1)


def _start_browser_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    _bt = _origin()

    with _bt._cleanup_lock:
        if _bt._cleanup_thread is None or not _bt._cleanup_thread.is_alive():
            _bt._cleanup_running = True
            _bt._cleanup_thread = threading.Thread(
                target=_bt._browser_cleanup_thread_worker, daemon=True, name="browser-cleanup"
            )
            _bt._cleanup_thread.start()
            _bt.logger.info("Started inactivity cleanup thread (timeout: %ss)", _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT)


def _stop_browser_cleanup_thread():
    """Stop the background cleanup thread."""
    _bt = _origin()
    _bt._cleanup_running = False
    if _bt._cleanup_thread is not None:
        _bt._cleanup_thread.join(timeout=5)


def _update_session_activity(task_id: str):
    """Update the last activity timestamp for a session.

    Also records the owning Hermes home on first sight so the process-global
    janitor can tear the session down under its owner's scope.  An
    activity touch deliberately does NOT reset ``_cleanup_failures`` — only a
    successful cleanup does.
    """
    _bt = _origin()
    with _bt._cleanup_lock:
        _bt._session_last_activity[task_id] = time.time()
        _bt._session_owner_homes.setdefault(task_id, str(get_hermes_home()))


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort kill of *proc* and every descendant it spawned; never raises.

    ``Popen.kill()`` only signals the direct child. npm/npx fork helpers and
    agent-browser's detached daemon grandchild, which survive a plain kill and
    keep a capture pipe open so ``communicate()`` never sees EOF — on Windows
    there is no non-blocking read to poll around that, so the whole tree must
    go. No grace period: the caller already burned its full timeout waiting.
    Delegates to :func:`agent.deadline.kill_process_tree` (taskkill /T /F,
    killpg, plus a psutil sweep that reaches ``setsid``'d descendants) and
    falls back to :func:`_legacy_kill_process_tree` on any failure.
    """
    _bt = _origin()
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _bt._legacy_kill_process_tree(proc)


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Local tree-kill — fallback when agent.deadline is unavailable."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            pass
        return
    # os.killpg/signal.SIGKILL don't exist on Windows; this branch is
    # POSIX-only (the `os.name == "nt"` check above already returns first
    # on Windows), but resolve them defensively via getattr anyway so an
    # accidental future refactor that drops that guard degrades to a plain
    # kill() instead of AttributeError — same discipline as
    # tools/mcp_stdio_watchdog.py's _terminate_process_group.
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok - non-POSIX fallback
        try:
            proc.kill()
        except Exception:
            pass
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return


def _pid_exists(pid: int) -> bool:
    """Best-effort 'is this PID alive' check (signal 0 / psutil on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — psutil.pid_exists above handles Windows
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cleanup_old_screenshots(screenshots_dir, max_age_hours=24):
    """Remove browser screenshots older than max_age_hours to prevent disk bloat.

    Throttled to run at most once per hour per directory to avoid repeated
    scans on screenshot-heavy workflows.
    """
    _bt = _origin()
    key = str(screenshots_dir)
    now = time.time()
    if now - _bt._last_screenshot_cleanup_by_dir.get(key, 0.0) < 3600:
        return
    _bt._last_screenshot_cleanup_by_dir[key] = now

    try:
        cutoff = time.time() - (max_age_hours * 3600)
        for f in screenshots_dir.glob("browser_screenshot_*.png"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                _bt.logger.debug("Failed to clean old screenshot %s: %s", f, e)
    except Exception as e:
        _bt.logger.debug("Screenshot cleanup error (non-critical): %s", e)


def _cleanup_old_recordings(max_age_hours=72):
    """Remove browser recordings older than max_age_hours to prevent disk bloat."""
    _bt = _origin()
    try:
        hermes_home = get_hermes_home()
        recordings_dir = hermes_home / "browser_recordings"
        if not recordings_dir.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for f in recordings_dir.glob("session_*.webm"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                _bt.logger.debug("Failed to clean old recording %s: %s", f, e)
    except Exception as e:
        _bt.logger.debug("Recording cleanup error (non-critical): %s", e)


def _drop_last_active_binding(task_id: str) -> None:
    """Drop stale last-active ownership after cleaning ``task_id``.

    Cleaning a bare task drops its binding; cleaning a sidecar drops the binding
    only if that sidecar was still the recorded owner — so a later
    click/snapshot can't resurrect a cleaned sidecar on about:blank while a
    primary-session binding is preserved.
    """
    _bt = _origin()
    if _bt._is_local_sidecar_key(task_id):
        bare_task_id = _bt._bare_task_id_for_session_key(task_id)
        if _bt._last_active_session_key.get(bare_task_id) == task_id:
            _bt._last_active_session_key.pop(bare_task_id, None)
    else:
        _bt._last_active_session_key.pop(task_id, None)


def cleanup_browser(task_id: Optional[str] = None) -> None:
    """Clean up browser session(s) for a task (task completion / inactivity timeout).

    A bare task id reaps BOTH the primary session and any hybrid local sidecar
    spawned for it; a key already carrying ``::local`` (inactivity loop) reaps
    only that one.
    """
    _bt = _origin()
    if task_id is None:
        task_id = "default"

    session_keys = [task_id]
    if not _bt._is_local_sidecar_key(task_id):
        sidecar_key = f"{task_id}{_bt._LOCAL_SUFFIX}"
        with _bt._cleanup_lock:
            if sidecar_key in _bt._active_sessions:
                session_keys.append(sidecar_key)

    for session_key in session_keys:
        _bt._cleanup_single_browser_session(session_key)
    _bt._drop_last_active_binding(task_id)


def _kill_verified_daemon(socket_dir: str, session_name: str) -> bool:
    """Tree-kill the daemon recorded in ``<socket_dir>/<session>.pid`` if it is verifiably ours.

    The .pid file lives in a world-writable temp dir and PIDs recycle: the
    process must pass ``_verify_reapable_browser_daemon`` and have a start-time
    fingerprint (so the kill refuses if the PID is swapped between check and
    kill). Returns True when a kill was issued. Never raises.
    """
    _bt = _origin()
    pid_file = os.path.join(socket_dir, f"{session_name}.pid")
    if not os.path.isfile(pid_file):
        return False
    try:
        from tools.process_registry import ProcessRegistry
        daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
        if not _bt._verify_reapable_browser_daemon(daemon_pid, socket_dir, session_name):
            _bt.logger.debug(
                "Skipped daemon kill for %s: pid %s failed identity "
                "verification", session_name, daemon_pid)
            return False
        from gateway.status import get_process_start_time
        daemon_start = get_process_start_time(daemon_pid)
        if daemon_start is None:
            _bt.logger.debug(
                "Skipped daemon kill for %s: no start-time "
                "fingerprint for pid %s", session_name, daemon_pid)
            return False
        ProcessRegistry._terminate_host_pid(daemon_pid, daemon_start)
        _bt.logger.debug("Killed daemon pid %s for %s", daemon_pid, session_name)
        return True
    except (ProcessLookupError, ValueError, PermissionError, OSError):
        _bt.logger.debug("Could not kill daemon pid for %s (already dead or inaccessible)", session_name)
        return False


def _release_session_resources(task_id: str, session_info: Dict[str, Any]) -> None:
    """Untrack ``task_id``, close its cloud provider session, kill its daemon.

    The unconditional tail of ``_cleanup_single_browser_session``; also the
    whole of the janitor's force-reap path, which skips the polite
    agent-browser/Camofox ``close`` that kept failing but must still release
    the cloud session and the local Chromium.
    """
    _bt = _origin()
    bb_session_id = session_info.get("bb_session_id", "unknown")
    with _bt._cleanup_lock:
        _bt._active_sessions.pop(task_id, None)
        _bt._session_last_activity.pop(task_id, None)
        _bt._session_owner_homes.pop(task_id, None)
        _bt._cleanup_failures.pop(task_id, None)

    # Cloud mode only — local sidecars have bb_session_id=None.
    if bb_session_id:
        provider = _bt._get_cloud_provider()
        if provider is not None:
            try:
                provider.close_session(bb_session_id)
            except Exception as e:
                _bt.logger.warning("Could not close cloud browser session: %s", e)

    session_name = session_info.get("session_name", "")
    if session_name:
        socket_dir = os.path.join(_bt._socket_safe_tmpdir(), f"agent-browser-{session_name}")
        if os.path.exists(socket_dir):
            _bt._kill_verified_daemon(socket_dir, session_name)
            shutil.rmtree(socket_dir, ignore_errors=True)


def _force_reap_browser_session(task_id: str) -> None:
    """Janitor last resort after repeated cleanup failures.

    Skips the ``close`` round-trips that keep failing and goes straight to
    ``_release_session_resources`` (cloud close + daemon kill + untrack).
    """
    _bt = _origin()
    _bt._stop_cdp_supervisor(task_id)
    with _bt._cleanup_lock:
        session_info = _bt._active_sessions.get(task_id)
        _bt._session_last_activity.pop(task_id, None)
        _bt._recording_sessions.discard(task_id)
    if session_info:
        _bt._release_session_resources(task_id, session_info)
    _bt._drop_last_active_binding(task_id)


def _cleanup_single_browser_session(task_id: str) -> None:
    """Internal: reap a single browser session by its exact session key."""
    # Stop the CDP supervisor for this task FIRST so we close our WebSocket
    # before the backend tears down the underlying CDP endpoint.
    _bt = _origin()
    _bt._stop_cdp_supervisor(task_id)

    # Also clean up Camofox session if running in Camofox mode.
    # Skip full close when managed persistence is enabled — the browser
    # profile (and its session cookies) must survive across agent tasks.
    # The inactivity reaper still frees idle resources.
    if _bt._is_camofox_mode():
        try:
            from tools.browser_camofox import camofox_close, camofox_soft_cleanup
            if not camofox_soft_cleanup(task_id):
                camofox_close(task_id)
        except Exception as e:
            _bt.logger.debug("Camofox cleanup for task %s: %s", task_id, e)

    _bt.logger.debug("cleanup_browser called for task_id: %s", task_id)
    _bt.logger.debug("Active sessions: %s", list(_bt._active_sessions.keys()))

    # Check if session exists (under lock), but don't remove yet -
    # _run_browser_command needs it to build the close command.
    with _bt._cleanup_lock:
        session_info = _bt._active_sessions.get(task_id)

    if session_info:
        bb_session_id = session_info.get("bb_session_id", "unknown")
        _bt.logger.debug("Found session for task %s: bb_session_id=%s", task_id, bb_session_id)

        # Stop auto-recording before closing (saves the file)
        _bt._maybe_stop_recording(task_id)

        # A Lightpanda session is a process Hermes spawned itself (Browser
        # Use mode); there is no agent-browser daemon to send ``close`` to.
        # An expired cloud CDP URL cannot accept an agent-browser close command.
        # Avoid feeding it back through _get_session_info(), which would try to
        # renew the session recursively while cleanup is still in progress.
        if (session_info.get("features") or {}).get("lightpanda"):
            try:
                from tools.browser_lightpanda import stop_lightpanda

                stop_lightpanda(session_info.get("session_name", ""))
            except Exception as e:
                _bt.logger.warning("lightpanda stop failed for task %s: %s", task_id, e)
        elif _bt._session_has_expired(session_info):
            _bt.logger.debug(
                "Skipping agent-browser close for expired session %s", task_id
            )
        else:
            try:
                _bt._run_browser_command(task_id, "close", [], timeout=10)
                _bt.logger.debug(
                    "agent-browser close command completed for task %s", task_id
                )
            except Exception as e:
                _bt.logger.warning("agent-browser close failed for task %s: %s", task_id, e)

        _bt._release_session_resources(task_id, session_info)

        _bt.logger.debug("Removed task %s from active sessions", task_id)
    else:
        _bt.logger.debug("No active session found for task_id: %s", task_id)


def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.

    Useful for cleanup on shutdown.
    """
    _bt = _origin()
    with _bt._cleanup_lock:
        task_ids = list(_bt._active_sessions.keys())
    for task_id in task_ids:
        _bt.cleanup_browser(task_id)

    # Tear down CDP supervisors for all tasks so background threads exit.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        SUPERVISOR_REGISTRY.stop_all()
    except Exception:
        pass

    # Reset cached lookups so they are re-evaluated on next use.
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False
    _bt._discover_homebrew_node_dirs.cache_clear()
    # Flip the resolved flag BEFORE nulling the cache so a concurrent
    # reader never sees ``resolved=True`` with ``cache=None``.
    _bt._command_timeout_resolved = False
    _bt._cached_command_timeout = None
    _bt._snapshot_threshold_resolved = False
    _bt._cached_snapshot_threshold = None
    _bt._cached_chromium_installed = None
    _bt._chromium_autoinstall_attempted = False
    _bt._cached_browser_engine = None
    _bt._browser_engine_resolved = False
