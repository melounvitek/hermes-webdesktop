"""agent-browser session management: daemon spawn, per-backend session creation
(local/lightpanda/cdp/cloud), cached session lookup, command execution with timeout
handling and output interpretation.

Split out of ``tools/browser_tool.py``; every name is re-imported there. Origin
symbols and module state are read/written through ``_bt`` (the origin module,
the :data:`tools.browser_tool_origin.origin` proxy) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.browser_tool_origin import origin as _bt

_CHROMIUM_MISSING_DOCKER_HINT = (
    "Chromium browser is missing. You're running in Docker — pull "
    "the latest image to get the bundled Chromium: "
    "docker pull ghcr.io/nousresearch/hermes-agent:latest"
)
_CHROMIUM_MISSING_HINT = (
    "Chromium browser is missing. Install it with: "
    "npx agent-browser install --with-deps "
    "(or: npx playwright install --with-deps chromium)"
)


def _needs_chromium_sandbox_bypass() -> bool:
    """True when Chromium needs --no-sandbox to start reliably (root, Docker, AppArmor userns)."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if _bt._running_in_docker():
        return True
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _apply_chromium_sandbox_args(browser_env: Dict[str, str]) -> None:
    """Add required Chromium sandbox flags without overriding user settings."""
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
    out = []
    for path in (stdout_path, stderr_path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                out.append(f.read().strip())
        except OSError:
            out.append("")
    return out[0], out[1]


def _unlink_command_output_files(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _format_browser_timeout_error(
    command: str, timeout: int, stdout: str, stderr: str
) -> str:
    """Actionable timeout message from captured daemon output."""
    parts = [f"Command timed out after {timeout} seconds"]
    detail = (stderr or stdout or "").strip()
    if detail:
        parts.append(detail[:1500])

    combined = f"{stderr}\n{stdout}".lower()
    if "sandbox" in combined:
        parts.append(
            "Chromium sandbox launch failed. Set AGENT_BROWSER_ARGS="
            "'--no-sandbox,--disable-dev-shm-usage' in your environment, "
            "or run: npx agent-browser install --with-deps"
        )
    elif command == "open" and _bt._is_local_mode():
        if _bt._running_in_docker():
            parts.append(
                "The browser daemon may still be starting or Chromium may be "
                "missing. Pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            parts.append(
                "The browser daemon may still be starting, or Chromium may be "
                "missing system libraries. Install/repair with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
    return "\n".join(parts)


def _agent_browser_argv(browser_cmd: str) -> list:
    """Command prefix to invoke agent-browser (concrete binary or npx sentinel).

    Concrete paths stay a single argv item; only the npx sentinel expands. npx is
    resolved through the same PATH cascade as ``_find_agent_browser`` (a bare
    ``shutil.which("npx")`` would let a broken system npx shadow a healthy managed
    one); if absent the bare name gives a readable ``FileNotFoundError: 'npx'``.
    ``--ignore-scripts``: AGENT_BROWSER_NPX_SPEC is a floating range — a compromised
    future patch must not run install-time scripts.
    """
    if _bt._is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _bt._resolve_npx_bin() or "npx"
        return [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", _bt.AGENT_BROWSER_NPX_SPEC]
    return [browser_cmd]


def _prepare_session_socket_dir(session_name: str) -> str:
    """Create the per-session socket dir and claim it with our PID.

    Per-session dirs keep parallel workers from fighting over the default socket
    path. The owner_pid file is written BEFORE first use: another hermes process's
    orphan reaper rmtree's any ownerless agent-browser-* dir in the shared tmpdir.
    """
    socket_dir = os.path.join(_bt._socket_safe_tmpdir(), f"agent-browser-{session_name}")
    os.makedirs(socket_dir, mode=0o700, exist_ok=True)
    _bt._write_owner_pid(socket_dir, session_name)
    return socket_dir


def _agent_browser_command_env(socket_dir: str) -> Dict[str, str]:
    """Credential-scrubbed env for one agent-browser command: discovery-time PATH
    fallbacks, the session socket dir, and daemon-side idle self-termination
    (``AGENT_BROWSER_IDLE_TIMEOUT_MS``, agent-browser 0.24+) mirroring the Python
    janitor — unless the user set it explicitly."""
    env = _bt._build_browser_env()
    env["PATH"] = _bt._merge_browser_path(env.get("PATH", ""))
    env["AGENT_BROWSER_SOCKET_DIR"] = socket_dir
    if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in env:
        env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = str(_bt.BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)
    return env


def _popen_agent_browser(argv: List[str], env: Dict[str, str], socket_dir: str, tag: str) -> "subprocess.Popen":
    """Spawn agent-browser with stdout/stderr redirected to ``socket_dir/_stdout_<tag>``.

    Temp files instead of pipes: the CLI forks a daemon that inherits its fds, so
    with pipes ``communicate()`` never sees EOF until the timeout. Windows:
    CREATE_NO_WINDOW only (NOT CREATE_NEW_PROCESS_GROUP — on 3.11 it cancels
    asyncio's running task and surfaces as KeyboardInterrupt), STARTF_USESTDHANDLES
    so the child gets ONLY our three handles (leaked console handles make the Rust
    daemon grandchild die silently), close_fds=True for the rest.
    """
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


def _session_record(prefix: str, cdp_url: Optional[str], features: Dict[str, Any]) -> Dict[str, Any]:
    """Fresh session dict with a random ``<prefix>_<hex10>`` session name."""
    return {
        "session_name": f"{prefix}_{uuid.uuid4().hex[:10]}",
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": features,
    }


def _create_local_session(task_id: str, allow_real_profile: bool = True) -> Dict[str, str]:
    """Local Chromium session; consented real-profile CDP attach when allowed.

    Real-profile fails closed on resolver/launch errors — a consented user must
    never be silently downgraded to a throwaway. The hybrid private-URL sidecar
    passes ``allow_real_profile=False``: handing the user's cookie jar to an
    arbitrary internal host the model chose is a larger, unconsented exposure.
    """
    if allow_real_profile:
        cdp_url, err = _bt._real_profile_cdp()
        if err:
            raise RuntimeError(err)
        if cdp_url:
            info = _session_record("rp", _bt._resolve_cdp_override(cdp_url), {"local": True, "real_profile": True})
            _bt.logger.info(
                "Created real-profile local session %s for task %s", info["session_name"], task_id
            )
            return info

    # Browser Use mode drives whatever CDP endpoint it is handed; with
    # ``browser.engine: lightpanda`` that is a Hermes-spawned ``lightpanda serve``.
    # The built-in tools never reach this branch (hidden in Browser Use mode).
    if _bt._is_browser_use_cli_mode() and _bt._using_lightpanda_engine():
        return _bt._create_lightpanda_session(task_id)

    info = _session_record("h", None, {"local": True})
    _bt.logger.info("Created local browser session %s for task %s",
                info["session_name"], task_id)
    return info


def _create_lightpanda_session(task_id: str) -> Dict[str, Any]:
    """Spawn ``lightpanda serve`` for this session key (Browser Use mode)."""
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
    """Session connecting to a user-supplied CDP endpoint."""
    info = _session_record("cdp", cdp_url, {"cdp_override": True})
    _bt.logger.info("Created CDP browser session %s → %s for task %s",
                info["session_name"], _bt._sanitize_url_for_logs(cdp_url), task_id)
    return info


def _create_cloud_session_or_fallback(task_id: str, provider) -> Dict[str, Any]:
    """Cloud session; fall back to local Chromium (marked degraded) on failure.

    Some providers (Browser-Use v3) return an HTTP CDP discovery URL instead of a
    raw websocket endpoint, so ``cdp_url`` is resolved here.
    """
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
        if isinstance(session_info, dict):  # mark degraded for observability
            session_info = dict(session_info)
            session_info["fallback_from_cloud"] = True
            session_info["fallback_reason"] = str(e)
            session_info["fallback_provider"] = provider_name
        return session_info


def _create_session_for_key(task_id: str, force_local: bool) -> Dict[str, Any]:
    """Fresh session for ``task_id`` (runs OUTSIDE the lock: cloud mode makes a network call).

    Precedence: CDP override > hybrid local sidecar > cloud provider > local. The
    hybrid sidecar NEVER gets the real profile (see ``_create_local_session``).
    """
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

    A ``::local``-suffixed key (hybrid sidecar) forces local Chromium even with a
    cloud provider configured. Also starts the inactivity thread and touches
    activity tracking. Returns ``session_name`` (always) plus ``bb_session_id`` /
    ``cdp_url`` for cloud sessions.
    """
    if task_id is None:
        task_id = "default"

    _bt._start_browser_cleanup_thread()
    _bt._update_session_activity(task_id)

    with _bt._cleanup_lock:
        existing_session = _bt._active_sessions.get(task_id)

    def _replacement_after_teardown() -> Optional[Dict[str, Any]]:
        # Teardown removes the activity entry; the replacement must be tracked by
        # the reaper like an initial session. Another thread may already have
        # recycled and re-created it — return that live one instead of a third.
        _bt._update_session_activity(task_id)
        with _bt._cleanup_lock:
            replacement = _bt._active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            return replacement
        return None

    if existing_session is not None:
        # Suspect-session recycle: a previous command timeout marked this cached
        # session via the SuspectableBackend adapter; the expensive recycle lives
        # here at next use, not on the timeout path (mark must stay cheap).
        if not _bt._browser_session_backend(task_id).ensure_healthy():
            replacement = _replacement_after_teardown()
            if replacement is not None:
                return replacement
            existing_session = None
        elif (
            not _bt._session_has_expired(existing_session)
            and not _bt._local_backend_process_dead(existing_session)
        ):
            return existing_session
        else:
            _bt.logger.info(
                "Replacing expired or dead browser session for task %s", task_id
            )
            _bt._cleanup_single_browser_session(task_id)
            replacement = _replacement_after_teardown()
            if replacement is not None:
                return replacement

    force_local = _bt._is_local_sidecar_key(task_id)
    session_info = _bt._create_session_for_key(task_id, force_local)

    with _bt._cleanup_lock:
        # Another thread may have created a session during the network call; use
        # it to avoid leaking orphan cloud sessions.
        if task_id in _bt._active_sessions:
            return _bt._active_sessions[task_id]
        session_info = dict(session_info)
        session_info.setdefault("session_key", task_id)
        session_info.setdefault("owner_task_id", _bt._bare_task_id_for_session_key(task_id))
        _bt._active_sessions[task_id] = session_info
        # A brand-new session is healthy by definition — drop any stale suspect flag.
        _bt._suspect_browser_sessions.pop(task_id, None)

    # Lazy-start the CDP supervisor (idempotent; swallows errors). Skip for local
    # sidecars (no CDP URL) and Lightpanda sessions (Browser Use mode hides the
    # browser_* tools that consume supervisor state; it would just idle a second CDP connection).
    if not force_local and not (session_info.get("features") or {}).get("lightpanda"):
        _bt._ensure_cdp_supervisor(task_id)

    return session_info


def _discard_timed_out_browser_session(
    task_id: str, session_info: Dict[str, Any], task_socket_dir: str
) -> None:
    """Drop a stuck client generation without losing cloud cleanup state."""
    with _bt._cleanup_lock:
        if _bt._active_sessions.get(task_id) is not session_info:
            return
        _bt._stop_cdp_supervisor(task_id)
        if session_info.get("bb_session_id") or session_info.get("cdp_url"):
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
    if session_name and os.path.isfile(os.path.join(task_socket_dir, f"{session_name}.pid")):
        daemon_pid = _bt._read_browser_daemon_pid(task_socket_dir, session_name)
        if daemon_pid is None:  # corrupt pid file
            _bt.logger.debug("Could not kill timed-out browser daemon for %s", session_name)
            return
        if not _bt._verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name):
            return
        try:
            # Tree-kill: terminating only the daemon PID leaks the Chromium tree.
            from agent import deadline as _deadline

            _deadline.kill_process_tree(daemon_pid)
        except (ProcessLookupError, PermissionError, OSError):
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

    A successful connect proves the accept loop is alive (the command wedged on the
    page/CDP side). Windows uses named pipes — no probe possible, so report
    unresponsive (tree-kill + respawn is the safe recovery).
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
    for entry in entries:
        if not entry.endswith(".sock"):
            continue
        try:
            with socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM) as s:
                s.settimeout(probe_timeout_s)
                s.connect(os.path.join(task_socket_dir, entry))
                return True
        except OSError:
            continue
    return False


def _handle_browser_command_timeout(
    task_id: str, session_info: Dict[str, Any], task_socket_dir: str
) -> None:
    """Recover session state after a browser command timeout.

    * Cloud / CDP: no local daemon to probe — replace the stuck client generation
      now (fresh ``session_name``, same ``bb_session_id`` so cloud cleanup works).
    * Local daemon alive (PID live, identity-verified, control socket accepts): only
      the *command* wedged; mark suspect and let next use recycle via ``ensure_healthy``.
    * Local daemon wedged/dead: it cannot service a clean close and its Chromium
      children would leak — tree-kill and evict now.

    Both local branches ``mark_suspect`` first (cheap, lock-free) so the
    poisoned-cache invariant holds even if eviction races another thread's replacement.
    """
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
    # The poisoned entry is gone either way; the flag must not poison a session
    # created later under the same key.
    _bt._suspect_browser_sessions.pop(task_id, None)


def _interpret_browser_command_output(command: str, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
    """Turn a finished agent-browser process's output into a result dict.

    Empty stdout with rc=0 is a broken state (stale daemon) and is reported as
    failure rather than a silent success — except for ``_EMPTY_OK_COMMANDS``.
    Non-JSON output is an error, except ``screenshot`` where the saved path is
    recovered from the prose.
    """
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


def _browser_command_preflight() -> Dict[str, Any]:
    """Fail fast before spawning: missing CLI, Termux install gap, interrupt, or no
    Chromium in local mode (else every call hangs for command_timeout). Returns an
    error result, or ``{"browser_cmd": path}`` on success."""
    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError as e:
        _bt.logger.warning("agent-browser CLI not found: %s", e)
        return {"success": False, "error": str(e)}

    if _bt._requires_real_termux_browser_install(browser_cmd):
        error = _bt._termux_browser_install_error()
        _bt.logger.warning("browser command blocked on Termux: %s", error)
        return {"success": False, "error": error}

    # Skip when engine=lightpanda — LP doesn't need Chromium for navigation.
    if (
        _bt._is_local_mode()
        and not _bt._chromium_installed()
        and _bt._get_browser_engine() != "lightpanda"
        and not _bt._maybe_autoinstall_chromium()
    ):
        hint = _CHROMIUM_MISSING_DOCKER_HINT if _bt._running_in_docker() else _CHROMIUM_MISSING_HINT
        _bt.logger.warning("browser command blocked: %s", hint)
        return {"success": False, "error": hint}

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"success": False, "error": "Interrupted"}
    return {"browser_cmd": browser_cmd}


def _spawn_and_collect(
    task_id: str, session_info: Dict[str, Any], cmd_parts: List[str],
    command: str, engine: str, timeout: int,
) -> Dict[str, Any]:
    """Run the prepared agent-browser argv once and interpret its output (handles timeout)."""
    task_socket_dir = _bt._prepare_session_socket_dir(session_info["session_name"])
    _bt.logger.debug("browser cmd=%s task=%s socket_dir=%s (%d chars)",
                 command, task_id, task_socket_dir, len(task_socket_dir))
    browser_env = _bt._agent_browser_command_env(task_socket_dir)

    # Lightpanda rejects Chromium-only launch flags: strip current and legacy vars;
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
        return {
            "success": False,
            "error": _bt._format_browser_timeout_error(command, timeout, stdout, stderr),
        }
    with open(stdout_path, "r", encoding="utf-8") as f:
        stdout = f.read()
    with open(stderr_path, "r", encoding="utf-8") as f:
        stderr = f.read()
    _bt._unlink_command_output_files(stdout_path, stderr_path)
    return _bt._interpret_browser_command_output(command, stdout, stderr, proc.returncode)


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
    if timeout is None:
        timeout = _bt._safe_command_timeout()
    args = args or []

    preflight = _browser_command_preflight()
    if "browser_cmd" not in preflight:
        return preflight
    browser_cmd = preflight["browser_cmd"]

    try:
        session_info = _bt._get_session_info(task_id)
    except Exception as e:
        _bt.logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    # Cleanup stops the supervisor before closing the backend; keep it stopped.
    if command != "close" and session_info.get("cdp_url"):
        _bt._ensure_cdp_supervisor(task_id)

    # Cloud/CDP: ``--cdp <ws_url>`` (NEVER with --session: agent-browser >=0.13
    # would create a local browser and silently ignore --cdp). Local: ``--session <name>``.
    if session_info.get("cdp_url"):
        backend_args = ["--cdp", session_info["cdp_url"]]
    else:
        backend_args = ["--session", session_info["session_name"]]
        if _bt._is_headed_mode():
            backend_args.append("--headed")

    # Engine injection keys off the resolved session backend, not global provider
    # state: hybrid routing can create a local sidecar while a cloud provider stays configured.
    engine = _engine_override or _bt._get_browser_engine()
    if engine != "auto" and not _bt._is_camofox_mode() and not session_info.get("cdp_url"):
        backend_args += ["--engine", engine]

    cmd_parts = _bt._agent_browser_argv(browser_cmd) + backend_args + ["--json", command] + args

    try:
        result = _spawn_and_collect(task_id, session_info, cmd_parts, command, engine, timeout)
    except Exception as e:
        _bt.logger.warning("browser '%s' exception: %s", command, e, exc_info=True)
        result = {"success": False, "error": str(e)}

    # Lightpanda automatic Chrome fallback — runs for ALL exit paths (timeout,
    # empty, non-JSON, nonzero rc, parsed).
    fallback_reason = _bt._lightpanda_fallback_reason(engine, command, result)
    if fallback_reason:
        _bt.logger.info(
            "Lightpanda fallback: retrying '%s' with Chrome (task=%s): %s",
            command,
            task_id,
            fallback_reason,
        )
        if command == "screenshot":  # separate Chrome session to the same URL
            fallback_result = _bt._chrome_fallback_screenshot(task_id, args or [], timeout)
        else:
            fallback_result = _bt._run_chrome_fallback_command(task_id, command, args, timeout)
        return _bt._annotate_lightpanda_fallback(fallback_result, fallback_reason)

    return result
