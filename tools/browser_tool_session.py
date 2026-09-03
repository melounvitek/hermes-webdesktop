"""agent-browser session management: daemon spawn, per-backend session creation (local/lightpanda/cdp/cloud), cached session lookup, command execution with timeout handling and output interpretation.

Split out of ``tools/browser_tool.py``; every name is re-imported there so
``tools.browser_tool.<name>`` keeps resolving (and monkeypatching). Origin
symbols and module state are read/written through ``_bt`` (the origin module,
resolved per call by :func:`tools.browser_tool_origin.origin_module`) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.browser_tool_origin import origin_module as _origin


def _needs_chromium_sandbox_bypass() -> bool:
    """Return True when Chromium needs --no-sandbox to start reliably."""
    _bt = _origin()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if _bt._running_in_docker():
        return True
    userns_restrict = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        with open(userns_restrict, encoding="utf-8") as f:
            if f.read().strip() == "1":
                return True
    except OSError:
        pass
    return False


def _apply_chromium_sandbox_args(browser_env: Dict[str, str]) -> None:
    """Add required Chromium sandbox flags without overriding user settings."""
    _bt = _origin()
    if (
        "AGENT_BROWSER_ARGS" not in browser_env
        and "AGENT_BROWSER_CHROME_FLAGS" not in browser_env
        and _bt._needs_chromium_sandbox_bypass()
    ):
        _bt.logger.debug(
            "browser: sandbox bypass needed (root/docker/AppArmor userns) — "
            "injecting --no-sandbox"
        )
        browser_env["AGENT_BROWSER_ARGS"] = "--no-sandbox,--disable-dev-shm-usage"


def _read_command_output_files(stdout_path: str, stderr_path: str) -> tuple[str, str]:
    """Best-effort read of agent-browser stdout/stderr temp files."""
    stdout = stderr = ""
    for path, slot in ((stdout_path, "stdout"), (stderr_path, "stderr")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            continue
        if slot == "stdout":
            stdout = text
        else:
            stderr = text
    return stdout, stderr


def _unlink_command_output_files(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _format_browser_timeout_error(
    command: str, timeout: int, stdout: str, stderr: str
) -> str:
    """Build an actionable timeout message from captured daemon output."""
    _bt = _origin()
    parts = [f"Command timed out after {timeout} seconds"]
    detail = (stderr or stdout or "").strip()
    if detail:
        parts.append(detail[:1500])

    combined = f"{stderr}\n{stdout}".lower()
    hints: list[str] = []
    if "sandbox" in combined:
        hints.append(
            "Chromium sandbox launch failed. Set AGENT_BROWSER_ARGS="
            "'--no-sandbox,--disable-dev-shm-usage' in your environment, "
            "or run: npx agent-browser install --with-deps"
        )
    elif command == "open" and _bt._is_local_mode():
        if _bt._running_in_docker():
            hints.append(
                "The browser daemon may still be starting or Chromium may be "
                "missing. Pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hints.append(
                "The browser daemon may still be starting, or Chromium may be "
                "missing system libraries. Install/repair with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
    if hints:
        parts.extend(hints)
    return "\n".join(parts)


def _agent_browser_argv(browser_cmd: str) -> list:
    """Command prefix to invoke agent-browser (concrete binary or npx sentinel).

    Concrete executable paths stay a single argv item (spaces intact); only the
    synthetic npx sentinel expands. npx is resolved through the same
    PATH + extended-PATH cascade ``_find_agent_browser`` uses — a bare
    ``shutil.which("npx")`` would let a broken system npx shadow a healthy
    Hermes-managed one. If npx isn't found at all (Termux, bare container) the
    bare name is used so Popen raises a readable ``FileNotFoundError: 'npx'``.
    ``--ignore-scripts``: AGENT_BROWSER_NPX_SPEC is a floating range, not an
    exact pin — a compromised future patch must not run install-time scripts.
    """
    _bt = _origin()
    if _bt._is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _bt._resolve_npx_bin() or "npx"
        return [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", _bt.AGENT_BROWSER_NPX_SPEC]
    return [browser_cmd]


def _prepare_session_socket_dir(session_name: str) -> str:
    """Create the per-session agent-browser socket dir and claim it with our PID.

    Each session gets its own dir so parallel workers don't fight over the
    default socket path ("Failed to create socket directory: Permission
    denied"). The owner_pid file is written BEFORE first use: another hermes
    process's orphan reaper rmtree's any agent-browser-* dir in the shared
    tmpdir that carries no live owner, which would delete this one mid-command.
    """
    _bt = _origin()
    socket_dir = os.path.join(_bt._socket_safe_tmpdir(), f"agent-browser-{session_name}")
    os.makedirs(socket_dir, mode=0o700, exist_ok=True)
    _bt._write_owner_pid(socket_dir, session_name)
    return socket_dir


def _agent_browser_command_env(socket_dir: str) -> Dict[str, str]:
    """Credential-scrubbed env for one agent-browser command.

    Adds the discovery-time PATH fallbacks, the session socket dir, and the
    daemon-side idle self-termination (``AGENT_BROWSER_IDLE_TIMEOUT_MS``,
    agent-browser 0.24+) mirroring the Python-side inactivity janitor —
    unless the user set the idle timeout explicitly.
    """
    _bt = _origin()
    env = _bt._build_browser_env()
    env["PATH"] = _bt._merge_browser_path(env.get("PATH", ""))
    env["AGENT_BROWSER_SOCKET_DIR"] = socket_dir
    if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in env:
        env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = str(_bt.BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)
    return env


def _popen_agent_browser(argv: List[str], env: Dict[str, str], socket_dir: str, tag: str) -> "subprocess.Popen":
    """Spawn agent-browser with stdout/stderr redirected to ``socket_dir/_stdout_<tag>``.

    Temp files instead of pipes: the CLI forks a background daemon that inherits
    its fds, so with pipes ``communicate()`` never sees EOF until the timeout.
    Windows: CREATE_NO_WINDOW only (NOT CREATE_NEW_PROCESS_GROUP, which on
    Python 3.11 cancels asyncio's running loop task and surfaces as
    KeyboardInterrupt in the CLI), STARTF_USESTDHANDLES so CreateProcess hands
    the child ONLY our three handles (leaked parent console handles make the
    Rust binary's daemon grandchild die silently), close_fds=True for the rest.
    Returns the Popen; the caller reads/unlinks the two files.
    """
    _bt = _origin()
    stdout_path = os.path.join(socket_dir, f"_stdout_{tag}")
    stderr_path = os.path.join(socket_dir, f"_stderr_{tag}")
    stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        _popen_extra: dict = {}
        if os.name == "nt":
            _popen_extra["creationflags"] = _bt.windows_hide_flags()
            _popen_extra["close_fds"] = True
            _si = subprocess.STARTUPINFO()
            _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
            _popen_extra["startupinfo"] = _si
        return subprocess.Popen(
            argv, stdout=stdout_fd, stderr=stderr_fd,
            stdin=subprocess.DEVNULL, env=env, **_popen_extra,
        )
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)


def _create_local_session(task_id: str, allow_real_profile: bool = True) -> Dict[str, str]:
    _bt = _origin()
    import uuid

    # Real-profile consent: attach this local session (via CDP) to the user's
    # browser running on a hermes-owned SNAPSHOT of their real profile, logins
    # included. Fail closed on resolver/launch errors — a consented user must
    # never be silently downgraded to a throwaway. The hybrid private-URL
    # sidecar passes allow_real_profile=False: handing the user's cookie jar to
    # an arbitrary internal host the model chose is a larger, unconsented
    # exposure than the routing rule protects against (and a real-profile
    # failure must not break private-URL routing).
    if allow_real_profile:
        cdp_url, err = _bt._real_profile_cdp()
        if err:
            raise RuntimeError(err)
        if cdp_url:
            session_name = f"rp_{uuid.uuid4().hex[:10]}"
            _bt.logger.info(
                "Created real-profile local session %s for task %s", session_name, task_id
            )
            return {
                "session_name": session_name,
                "bb_session_id": None,
                "cdp_url": _bt._resolve_cdp_override(cdp_url),
                "features": {"local": True, "real_profile": True},
            }

    # Browser Use mode drives whatever CDP endpoint it is handed; with
    # ``browser.engine: lightpanda`` that endpoint is a Hermes-spawned
    # ``lightpanda serve``. The built-in tools never reach this branch —
    # they are hidden in Browser Use mode — and keep driving Lightpanda via
    # ``agent-browser --engine lightpanda`` on the plain local session below.
    if _bt._is_browser_use_cli_mode() and _bt._using_lightpanda_engine():
        return _bt._create_lightpanda_session(task_id)

    session_name = f"h_{uuid.uuid4().hex[:10]}"
    _bt.logger.info("Created local browser session %s for task %s",
                session_name, task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": None,
        "features": {"local": True},
    }


def _create_lightpanda_session(task_id: str) -> Dict[str, Any]:
    """Spawn ``lightpanda serve`` for this session key (Browser Use mode)."""
    _bt = _origin()
    import uuid
    from tools.browser_lightpanda import launch_lightpanda

    session_name = f"lp_{uuid.uuid4().hex[:10]}"
    server, err = launch_lightpanda(
        session_name, block_private_networks=not _bt._is_local_backend()
    )
    if err:
        raise RuntimeError(err)
    _bt.logger.info(
        "Created Lightpanda session %s (port %s) for task %s", session_name, server.port, task_id
    )
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": server.cdp_url,
        "features": {"local": True, "lightpanda": True},
    }


def _local_backend_process_dead(session_info: Dict[str, Any]) -> bool:
    """True for a Lightpanda session whose ``lightpanda serve`` is gone."""
    if not (session_info.get("features") or {}).get("lightpanda"):
        return False
    from tools.browser_lightpanda import get_server

    server = get_server(session_info.get("session_name", ""))
    return server is None or not server.is_alive()


def _create_cdp_session(task_id: str, cdp_url: str) -> Dict[str, str]:
    """Create a session that connects to a user-supplied CDP endpoint."""
    _bt = _origin()
    import uuid
    session_name = f"cdp_{uuid.uuid4().hex[:10]}"
    _bt.logger.info("Created CDP browser session %s → %s for task %s",
                session_name, _bt._sanitize_url_for_logs(cdp_url), task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": {"cdp_override": True},
    }


def _create_cloud_session_or_fallback(task_id: str, provider) -> Dict[str, Any]:
    """Create a cloud session; fall back to local Chromium (marked degraded) on failure.

    Some cloud providers (Browser-Use v3) return an HTTP CDP discovery URL
    instead of a raw websocket endpoint, so ``cdp_url`` is resolved here.
    """
    _bt = _origin()
    try:
        session_info = provider.create_session(task_id)
        if not session_info or not isinstance(session_info, dict):
            raise ValueError(f"Cloud provider returned invalid session: {session_info!r}")
        if session_info.get("cdp_url"):
            session_info = dict(session_info)
            session_info["cdp_url"] = _bt._resolve_cdp_override(str(session_info["cdp_url"]))
        return session_info
    except Exception as e:
        provider_name = type(provider).__name__
        _bt.logger.warning(
            "Cloud provider %s failed (%s); attempting fallback to local "
            "Chromium for task %s",
            provider_name, e, task_id,
            exc_info=True,
        )
        try:
            session_info = _bt._create_local_session(task_id)
        except Exception as local_error:
            raise RuntimeError(
                f"Cloud provider {provider_name} failed ({e}) and local "
                f"fallback also failed ({local_error})"
            ) from e
        # Mark session as degraded for observability
        if isinstance(session_info, dict):
            session_info = dict(session_info)
            session_info["fallback_from_cloud"] = True
            session_info["fallback_reason"] = str(e)
            session_info["fallback_provider"] = provider_name
        return session_info


def _create_session_for_key(task_id: str, force_local: bool) -> Dict[str, Any]:
    """Create a fresh session for ``task_id`` (runs OUTSIDE the lock: cloud mode makes a network call).

    Precedence: CDP override > hybrid local sidecar > cloud provider > local.
    The hybrid private-URL sidecar NEVER gets the real profile — presenting real
    cookies to an arbitrary LAN host the model routed there is unconsented
    exposure (see ``_create_local_session``).
    """
    _bt = _origin()
    cdp_override = _bt._get_cdp_override()
    if cdp_override and not force_local:
        return _bt._create_cdp_session(task_id, cdp_override)
    if force_local:
        return _bt._create_local_session(task_id, allow_real_profile=False)
    provider = _bt._get_cloud_provider()
    if provider is None:
        return _bt._create_local_session(task_id)
    return _bt._create_cloud_session_or_fallback(task_id, provider)


def _get_session_info(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Get or create session info for a session key (thread-safe).

    ``task_id`` may carry the ``::local`` suffix (hybrid local sidecar), which
    forces a local Chromium even when a cloud provider is configured. Also
    starts the inactivity cleanup thread and touches activity tracking.
    Returns a dict with ``session_name`` (always) plus ``bb_session_id`` /
    ``cdp_url`` for cloud sessions.
    """
    _bt = _origin()
    if task_id is None:
        task_id = "default"

    # Start the cleanup thread if not running (handles inactivity timeouts)
    _bt._start_browser_cleanup_thread()

    # Update activity timestamp for this session
    _bt._update_session_activity(task_id)

    with _bt._cleanup_lock:
        # Check if we already have a session for this task
        existing_session = _bt._active_sessions.get(task_id)

    # Suspect-session recycle: a previous command
    # timeout marked this cached session suspect via the SuspectableBackend
    # adapter.  ensure_healthy() tears it down here, at next use, and we fall
    # through to create a fresh session — the expensive recycle lives on this
    # path, not on the timeout path (mark must stay cheap).
    if existing_session is not None and not _bt._browser_session_backend(task_id).ensure_healthy():
        # Teardown removes the activity entry; the replacement must be
        # tracked by the inactivity reaper like an initial session.
        _bt._update_session_activity(task_id)
        with _bt._cleanup_lock:
            replacement = _bt._active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            # Another thread already recycled and re-created it.
            return replacement
        existing_session = None

    if existing_session is not None:
        if (
            not _bt._session_has_expired(existing_session)
            and not _bt._local_backend_process_dead(existing_session)
        ):
            return existing_session

        _bt.logger.info(
            "Replacing expired or dead browser session for task %s", task_id
        )
        _bt._cleanup_single_browser_session(task_id)
        # Cleanup removes the activity entry. The replacement session must be
        # tracked by the inactivity reaper just like an initial session.
        _bt._update_session_activity(task_id)

        # Guard against a concurrent replacement: another thread may have
        # already cleaned up the expired session and created a fresh one
        # while we were waiting.  If so, return the live replacement instead
        # of falling through to create yet another session.
        with _bt._cleanup_lock:
            replacement = _bt._active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            return replacement

    # Hybrid routing: session keys ending with ``::local`` force a local
    # Chromium regardless of the globally-configured cloud provider.  Public
    # URLs in the same conversation continue to use the cloud session under
    # the bare task_id key.
    force_local = _bt._is_local_sidecar_key(task_id)
    session_info = _bt._create_session_for_key(task_id, force_local)

    with _bt._cleanup_lock:
        # Double-check: another thread may have created a session while we
        # were doing the network call. Use the existing one to avoid leaking
        # orphan cloud sessions.
        if task_id in _bt._active_sessions:
            return _bt._active_sessions[task_id]
        session_info = dict(session_info)
        session_info.setdefault("session_key", task_id)
        session_info.setdefault("owner_task_id", _bt._bare_task_id_for_session_key(task_id))
        _bt._active_sessions[task_id] = session_info
        # A brand-new session is healthy by definition — drop any stale
        # suspect flag left by a wedged-path eviction of its predecessor.
        _bt._suspect_browser_sessions.pop(task_id, None)

    # Lazy-start the CDP supervisor now that the session exists (if the
    # backend surfaces a CDP URL via override or session_info["cdp_url"]).
    # Idempotent; swallows errors. See _ensure_cdp_supervisor for details.
    # Skip for local sidecars — they have no CDP URL — and for Lightpanda
    # sessions: those only exist in Browser Use mode, where the browser_*
    # tools that consume supervisor state are hidden, so the supervisor
    # would just hold an idle second CDP connection to the process.
    if not force_local and not (session_info.get("features") or {}).get("lightpanda"):
        _bt._ensure_cdp_supervisor(task_id)

    return session_info


def _discard_timed_out_browser_session(
    task_id: str, session_info: Dict[str, Any], task_socket_dir: str
) -> None:
    """Drop a stuck client generation without losing cloud cleanup state."""
    _bt = _origin()
    with _bt._cleanup_lock:
        if _bt._active_sessions.get(task_id) is not session_info:
            return
        _bt._stop_cdp_supervisor(task_id)
        if session_info.get("bb_session_id") or session_info.get("cdp_url"):
            import uuid
            replacement = dict(session_info)
            replacement["session_name"] = f"h_{uuid.uuid4().hex[:10]}"
            replacement.pop("_first_nav", None)
            _bt._active_sessions[task_id] = replacement
        else:
            _bt._active_sessions.pop(task_id, None)
            _bt._session_last_activity.pop(task_id, None)

        bare_task_id = _bt._bare_task_id_for_session_key(task_id)
        if _bt._last_active_session_key.get(bare_task_id) == task_id:
            _bt._last_active_session_key.pop(bare_task_id, None)

    session_name = str(session_info.get("session_name") or "")
    if session_name:
        pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
        if os.path.isfile(pid_file):
            try:
                daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                if not _bt._verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name):
                    return
                # Tree-kill: the daemon spawns Chromium
                # children; terminating only the daemon PID leaks the whole
                # Chromium tree.  agent.deadline.kill_process_tree escalates
                # SIGTERM → SIGKILL across the tree.
                from agent import deadline as _deadline

                _deadline.kill_process_tree(daemon_pid)
            except (ProcessLookupError, ValueError, PermissionError, OSError):
                _bt.logger.debug("Could not kill timed-out browser daemon for %s", session_name)
                return
    shutil.rmtree(task_socket_dir, ignore_errors=True)


def _read_browser_daemon_pid(task_socket_dir: str, session_name: str) -> Optional[int]:
    """Read the agent-browser daemon PID for a session (best-effort)."""
    pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
    try:
        return int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _browser_daemon_responsive(task_socket_dir: str, probe_timeout_s: float = 1.0) -> bool:
    """Cheap liveness probe: connect to the daemon's unix control socket.

    A successful connect proves the accept loop is alive (the command wedged on
    the page/CDP side, not the daemon). Windows uses named pipes — no probe is
    possible, so report unresponsive (tree-kill + respawn is the safe recovery).
    """
    if os.name == "nt":
        return False
    import socket as socket_mod

    if not hasattr(socket_mod, "AF_UNIX"):
        return False
    try:
        entries = os.listdir(task_socket_dir)
    except OSError:
        return False
    sock_paths = [
        os.path.join(task_socket_dir, e) for e in entries if e.endswith(".sock")
    ]
    for sock_path in sock_paths:
        try:
            with socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM) as s:
                s.settimeout(probe_timeout_s)
                s.connect(sock_path)
                return True
        except OSError:
            continue
    return False


def _handle_browser_command_timeout(
    task_id: str, session_info: Dict[str, Any], task_socket_dir: str
) -> None:
    """Recover session state after a browser command timeout.

    * Cloud / CDP sessions: no local daemon to probe — replace the stuck client
      generation now (fresh ``session_name``, same ``bb_session_id`` so cloud
      cleanup still works).
    * Local daemon alive (PID live, identity-verified, control socket accepts):
      only the *command* wedged; mark the session suspect and let the next use
      recycle it via ``ensure_healthy`` → clean ``close`` → fresh session.
    * Local daemon wedged/dead: it cannot service a clean close and its Chromium
      children would leak — tree-kill and evict now; the next call respawns.

    Both local branches ``mark_suspect`` first (cheap, lock-free) so the
    poisoned-cache invariant holds even if eviction races another thread's
    replacement (the flag then costs one harmless no-op teardown).
    """
    _bt = _origin()
    if session_info.get("bb_session_id") or session_info.get("cdp_url"):
        _bt._discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
        return

    _bt._browser_session_backend(task_id).mark_suspect(
        "browser command timed out; session may be poisoned"
    )

    session_name = str(session_info.get("session_name") or "")
    daemon_pid = _bt._read_browser_daemon_pid(task_socket_dir, session_name) if session_name else None
    daemon_alive = (
        daemon_pid is not None
        and _bt._pid_exists(daemon_pid)
        and _bt._verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name)
        and _bt._browser_daemon_responsive(task_socket_dir)
    )
    if daemon_alive:
        _bt.logger.warning(
            "browser daemon for %s is alive after command timeout; session "
            "marked suspect and will be recycled at next use", task_id,
        )
        return

    _bt.logger.warning(
        "browser daemon for %s is wedged or dead after command timeout; "
        "tree-killing and evicting the session", task_id,
    )
    _bt._discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
    # The poisoned entry is gone (evicted, or superseded by a concurrent
    # replacement discard refused to touch) — either way the cache no longer
    # holds the timed-out session, so drop the flag: it must not poison a
    # session created later under the same key.
    _bt._suspect_browser_sessions.pop(task_id, None)


def _interpret_browser_command_output(command: str, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
    """Turn a finished agent-browser process's output into a result dict.

    Empty stdout with rc=0 is a broken state (stale daemon) and is reported as
    failure rather than a silent ``{"success": True, "data": {}}`` — except for
    commands in ``_EMPTY_OK_COMMANDS``. Non-JSON output is an error, except
    for ``screenshot`` where the saved path is recovered from the prose.
    """
    _bt = _origin()
    if stderr and stderr.strip():
        level = logging.WARNING if returncode != 0 else logging.DEBUG
        _bt.logger.log(level, "browser '%s' stderr: %s", command, stderr.strip()[:500])

    stdout_text = stdout.strip()
    if not stdout_text and returncode == 0 and command not in _bt._EMPTY_OK_COMMANDS:
        _bt.logger.warning("browser '%s' returned empty output (rc=0)", command)
        return {"success": False, "error": f"Browser command '{command}' returned no output"}
    if not stdout_text:
        if returncode != 0:
            error_msg = stderr.strip() if stderr else f"Command failed with code {returncode}"
            _bt.logger.warning("browser '%s' failed (rc=%s): %s", command, returncode, error_msg[:300])
            return {"success": False, "error": error_msg}
        return {"success": True, "data": {}}

    try:
        parsed = json.loads(stdout_text)
    except json.JSONDecodeError:
        raw = stdout_text[:2000]
        _bt.logger.warning("browser '%s' returned non-JSON output (rc=%s): %s",
                       command, returncode, raw[:500])
        if command == "screenshot":
            stderr_text = (stderr or "").strip()
            combined_text = "\n".join(part for part in [stdout_text, stderr_text] if part)
            recovered_path = _bt._extract_screenshot_path_from_text(combined_text)
            if recovered_path and Path(recovered_path).exists():
                _bt.logger.info(
                    "browser 'screenshot' recovered file from non-JSON output: %s", recovered_path
                )
                return {"success": True, "data": {"path": recovered_path, "raw": raw}}
        return {"success": False, "error": f"Non-JSON output from agent-browser for '{command}': {raw}"}

    # Empty snapshot content is a common sign of daemon/CDP issues.
    if command == "snapshot" and parsed.get("success"):
        snap_data = parsed.get("data", {})
        if not snap_data.get("snapshot") and not snap_data.get("refs"):
            _bt.logger.warning("snapshot returned empty content. "
                           "Possible stale daemon or CDP connection issue. "
                           "returncode=%s", returncode)
    return parsed


def _run_browser_command(
    task_id: str,
    command: str,
    args: List[str] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one agent-browser CLI command against the task's session; returns its parsed JSON.

    ``timeout=None`` reads ``browser.command_timeout`` (default 30s).
    ``_engine_override`` forces an engine for this call only (the Lightpanda
    fallback uses it to retry with Chrome without touching global state).
    """
    _bt = _origin()
    if timeout is None:
        timeout = _bt._safe_command_timeout()
    args = args or []

    # Build the command
    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError as e:
        _bt.logger.warning("agent-browser CLI not found: %s", e)
        return {"success": False, "error": str(e)}

    if _bt._requires_real_termux_browser_install(browser_cmd):
        error = _bt._termux_browser_install_error()
        _bt.logger.warning("browser command blocked on Termux: %s", error)
        return {"success": False, "error": error}

    # Local mode with no Chromium on disk: fail fast with an actionable
    # message instead of hanging for _command_timeout seconds per call.
    # Skip when engine=lightpanda — LP doesn't need Chromium for navigation.
    if (
        _bt._is_local_mode()
        and not _bt._chromium_installed()
        and _bt._get_browser_engine() != "lightpanda"
        and not _bt._maybe_autoinstall_chromium()
    ):
        if _bt._running_in_docker():
            hint = (
                "Chromium browser is missing. You're running in Docker — pull "
                "the latest image to get the bundled Chromium: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chromium browser is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        _bt.logger.warning("browser command blocked: %s", hint)
        return {"success": False, "error": hint}

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = _bt._get_session_info(task_id)
    except Exception as e:
        _bt.logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    # Cleanup stops the supervisor before closing the backend; keep it stopped.
    if command != "close" and session_info.get("cdp_url"):
        _bt._ensure_cdp_supervisor(task_id)

    # Build the command with the appropriate backend flag.
    # Cloud mode: --cdp <websocket_url> connects to Browserbase.
    # Local mode: --session <name> launches a local headless Chromium.
    # The rest of the command (--json, command, args) is identical.
    if session_info.get("cdp_url"):
        # Cloud mode — connect to remote Browserbase browser via CDP
        # IMPORTANT: Do NOT use --session with --cdp. In agent-browser >=0.13,
        # --session creates a local browser instance and silently ignores --cdp.
        backend_args = ["--cdp", session_info["cdp_url"]]
    else:
        # Local mode — launch Chromium (headless by default, headed when configured)
        backend_args = ["--session", session_info["session_name"]]
        if _bt._is_headed_mode():
            backend_args.append("--headed")

    # Lightpanda engine injection (local mode only, agent-browser v0.25.3+).
    # Use the resolved session backend rather than global cloud-provider state:
    # hybrid private-URL routing can create a local sidecar while a cloud
    # provider remains configured for public URLs.
    engine = _engine_override or _bt._get_browser_engine()
    if engine != "auto" and not _bt._is_camofox_mode() and not session_info.get("cdp_url"):
        backend_args += ["--engine", engine]

    cmd_parts = _bt._agent_browser_argv(browser_cmd) + backend_args + ["--json", command] + args

    try:
        task_socket_dir = _bt._prepare_session_socket_dir(session_info["session_name"])
        _bt.logger.debug("browser cmd=%s task=%s socket_dir=%s (%d chars)",
                     command, task_id, task_socket_dir, len(task_socket_dir))
        browser_env = _bt._agent_browser_command_env(task_socket_dir)

        # Chromium-only launch flags are rejected by Lightpanda. Strip both
        # the current and legacy variables for Lightpanda commands; explicit
        # Chrome commands and fallback use the shared Chromium policy.
        if engine == "lightpanda":
            _stripped_args = browser_env.pop("AGENT_BROWSER_ARGS", None)
            _stripped_flags = browser_env.pop("AGENT_BROWSER_CHROME_FLAGS", None)
            if _stripped_args is not None or _stripped_flags is not None:
                _bt.logger.debug(
                    "browser: stripped Chromium-only AGENT_BROWSER_ARGS/"
                    "AGENT_BROWSER_CHROME_FLAGS for Lightpanda command %s "
                    "(agent-browser rejects them with --engine lightpanda)",
                    command,
                )
        else:
            _bt._apply_chromium_sandbox_args(browser_env)

        stdout_path = os.path.join(task_socket_dir, f"_stdout_{command}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{command}")
        proc = _bt._popen_agent_browser(cmd_parts, browser_env, task_socket_dir, command)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout, stderr = _bt._read_command_output_files(stdout_path, stderr_path)
            _bt._unlink_command_output_files(stdout_path, stderr_path)
            _bt._handle_browser_command_timeout(task_id, session_info, task_socket_dir)
            if stderr and stderr.strip():
                _bt.logger.warning(
                    "browser '%s' stderr after timeout: %s", command, stderr.strip()[:500]
                )
            _bt.logger.warning("browser '%s' timed out after %ds (task=%s, socket_dir=%s)",
                           command, timeout, task_id, task_socket_dir)
            result = {
                "success": False,
                "error": _bt._format_browser_timeout_error(command, timeout, stdout, stderr),
            }
            # Fall through to fallback check below
        else:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read()
            with open(stderr_path, "r", encoding="utf-8") as f:
                stderr = f.read()
            _bt._unlink_command_output_files(stdout_path, stderr_path)
            result = _bt._interpret_browser_command_output(command, stdout, stderr, proc.returncode)

    except Exception as e:
        _bt.logger.warning("browser '%s' exception: %s", command, e, exc_info=True)
        result = {"success": False, "error": str(e)}

    # --- Lightpanda automatic Chrome fallback ---
    # If engine is lightpanda and the result looks broken, retry with Chrome.
    # This runs for ALL exit paths (timeout, empty, non-JSON, nonzero rc, parsed).
    fallback_reason = _bt._lightpanda_fallback_reason(engine, command, result)
    if fallback_reason:
        _bt.logger.info(
            "Lightpanda fallback: retrying '%s' with Chrome (task=%s): %s",
            command,
            task_id,
            fallback_reason,
        )
        # For screenshots, use the dedicated Chrome fallback helper
        # (spins up a separate Chrome session to the same URL).
        if command == "screenshot":
            fallback_result = _bt._chrome_fallback_screenshot(task_id, args or [], timeout)
        else:
            fallback_result = _bt._run_chrome_fallback_command(task_id, command, args, timeout)
        return _bt._annotate_lightpanda_fallback(fallback_result, fallback_reason)

    return result
