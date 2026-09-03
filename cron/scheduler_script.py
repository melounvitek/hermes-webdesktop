"""Cron pre-run script execution: timeouts, Windows venv bootstrap, process-tree termination,
and the claim-heartbeat thread that keeps a long script's run claim alive.

Split out of ``cron.scheduler``; every name is re-exported there, and origin-resident
helpers are reached late-bound via ``_sched`` so monkeypatching ``cron.scheduler.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from cron.jobs import _ensure_cron_dir
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cron.scheduler import _CancelEventLike

# Log-record parity with the origin module.
logger = logging.getLogger("cron.scheduler")


def _get_script_timeout() -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if _sched._SCRIPT_TIMEOUT != _sched._DEFAULT_SCRIPT_TIMEOUT:
        try:
            timeout = int(float(_sched._SCRIPT_TIMEOUT))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", _sched._SCRIPT_TIMEOUT)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    try:
        cfg = _sched.load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("script_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron script timeout from config: %s", exc)

    return _sched._DEFAULT_SCRIPT_TIMEOUT


_DEFAULT_MEDIA_SEND_TIMEOUT = 300


def _get_media_send_timeout() -> int:
    """Per-attachment media-send timeout: HERMES_CRON_MEDIA_SEND_TIMEOUT env, then
    ``cron.media_send_timeout_seconds``, then 300s (long TTS audio can exceed a 30s window)."""
    env_value = os.getenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning(
                "Invalid HERMES_CRON_MEDIA_SEND_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = _sched.load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("media_send_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron media-send timeout from config: %s", exc)

    return _DEFAULT_MEDIA_SEND_TIMEOUT


def _get_session_db_timeout() -> float:
    """Bound on run_job's SessionDB init: HERMES_CRON_SESSION_DB_TIMEOUT env, then
    ``cron.session_db_timeout_seconds`` (in DEFAULT_CONFIG), then 10s. Unlike sibling timeouts,
    0 is meaningful (unlimited, debugging opt-in), so values pass through untouched."""
    env_value = os.getenv("HERMES_CRON_SESSION_DB_TIMEOUT", "").strip()
    if env_value:
        try:
            return float(env_value)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid HERMES_CRON_SESSION_DB_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = _sched.load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("session_db_timeout_seconds")
        if configured is not None:
            return float(configured)
    except Exception as exc:
        logger.debug("Failed to load cron.session_db_timeout_seconds from config: %s", exc)

    return 10.0


def _read_windows_pyvenv_cfg(venv_dir: Path) -> dict[str, str]:
    cfg_path = venv_dir / "pyvenv.cfg"
    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    parsed: dict[str, str] = {}
    for raw in lines:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _windows_cron_python_invocation(python_exe: str) -> tuple[str, dict[str, str]]:
    """Hidden, output-capable Python invocation for Windows cron scripts. ``pythonw.exe`` loses
    captured output; uv venv launchers can re-exec the base console python and flash a window even
    with CREATE_NO_WINDOW, so run the base python directly with venv paths overlaid in env."""
    if sys.platform != "win32":
        return python_exe, {}

    interpreter = _sched.Path(python_exe)
    venv_dir = interpreter.parent.parent
    env_overlay: dict[str, str] = {}

    if interpreter.name.lower() == "pythonw.exe":
        sibling = interpreter.with_name("python.exe")
        if sibling.exists():
            interpreter = sibling

    cfg = _read_windows_pyvenv_cfg(venv_dir)
    home = cfg.get("home", "")
    site_packages = venv_dir / "Lib" / "site-packages"
    if "uv" in cfg and home:
        base_python = _sched.Path(home) / "python.exe"
        if base_python.exists() and site_packages.exists():
            interpreter = base_python
            env_overlay["VIRTUAL_ENV"] = str(venv_dir)
            pythonpath_entries = [str(_sched.Path(__file__).resolve().parents[1]), str(site_packages)]
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            env_overlay["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return str(interpreter), env_overlay


def _terminate_cron_script_process(proc: subprocess.Popen) -> None:
    """Best-effort hard stop of a cron script and every child it spawned."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=_sched.windows_hide_flags(),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    else:
        try:
            process_group: Optional[int] = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            process_group = None
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGTERM)  # windows-footgun: ok — POSIX-only branch (win32 handled above)
            except (ProcessLookupError, PermissionError, OSError):
                process_group = None
            if process_group is not None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
                # Escalate if ANY group member survived TERM: a survivor holds the pipe write ends
                # open and the caller's communicate() would block on EOF forever.
                try:
                    os.killpg(process_group, 0)  # windows-footgun: ok — POSIX-only branch
                except (ProcessLookupError, OSError):
                    process_group = None
                if process_group is not None:
                    with contextlib.suppress((ProcessLookupError, PermissionError, OSError)):
                        os.killpg(process_group, getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


def _terminate_cron_script_tree(proc: subprocess.Popen) -> None:
    """Terminate a script tree, then fall back to the local process-group path."""
    if proc.poll() is not None:
        # Already reaped: kill_process_tree would log a spurious "no signal" warning.
        return
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        logger.warning(
            "Cron script tree-kill received invalid pid %r; "
            "falling back to process-group termination",
            pid,
        )
        _sched._terminate_cron_script_process(proc)
        return
    try:
        # Function-local (monkeypatchable); separate try so an import problem is not
        # misreported as a kill failure.
        from agent.deadline import kill_process_tree
    except Exception:
        logger.warning(
            "agent.deadline.kill_process_tree unavailable; "
            "falling back to process-group termination",
            exc_info=True,
        )
        _sched._terminate_cron_script_process(proc)
        return
    try:
        if kill_process_tree(pid):
            return
        logger.warning(
            "Cron script tree-kill reported no signal for pid %s; "
            "falling back to process-group termination",
            pid,
        )
    except Exception:
        logger.warning(
            "Cron script tree-kill failed for pid %s; "
            "falling back to process-group termination",
            pid,
            exc_info=True,
        )
    _sched._terminate_cron_script_process(proc)


def _drain_script_pipes(proc: subprocess.Popen) -> None:
    """Reap a terminated script without blocking forever: a surviving descendant can hold the pipe
    write ends open, so bound the drain and abandon the pipes (output is not needed)."""
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.communicate(timeout=5.0)
        return
    with contextlib.suppress(OSError):
        proc.kill()
    for stream in (proc.stdout, proc.stderr):
        with contextlib.suppress(OSError):
            if stream is not None:
                stream.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5.0)


def _windows_cron_bootstrap_argv(
    python_exe: str,
    env_overlay: dict[str, str],
    script_path: str,
) -> list[str]:
    """Bootstrap a cron script under the base interpreter with ``.pth`` support.

    Overlay mode runs base ``python.exe`` (avoids the launcher flashing a console window) with the
    venv on ``PYTHONPATH`` — but ``.pth`` files are only processed by ``site.addsitedir()``, so
    editable installs would be invisible. Bootstrap via addsitedir + ``runpy.run_path`` (keeps
    ``__file__`` and ``sys.path[0]`` semantics); plain invocation if the venv is unresolvable.
    """
    site_packages = _sched.Path(env_overlay.get("VIRTUAL_ENV", "")) / "Lib" / "site-packages"
    if not site_packages.is_dir():
        # Warn: silent fallback would make "editable installs invisible" undiagnosable.
        logger.warning(
            "Windows cron script: venv site-packages %s not found; running "
            "without .pth processing (editable installs may be unimportable)",
            site_packages,
        )
        return [python_exe, script_path]
    bootstrap = (
        "import os, runpy, site, sys;"
        f"site.addsitedir({str(site_packages)!r});"
        "script = sys.argv[1];"
        "sys.argv = [script] + sys.argv[2:];"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(script)));"
        "runpy.run_path(script, run_name='__main__')"
    )
    return [python_exe, "-c", bootstrap, script_path]


def _run_job_script(
    script_path: str,
    workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> tuple[bool, str]:
    """Execute a cron job's script and return ``(success, output)``; on failure *output* is the
    error message for the LLM to report.

    Scripts MUST resolve inside HERMES_HOME/scripts/ (relative, absolute and ``~`` paths are all
    validated — path traversal / absolute-path injection). Interpreter by extension:
    ``.sh``/``.bash`` → bash, else ``sys.executable``. Env goes through ``build_subprocess_env``
    (SECURITY.md §2.3).
    ``workdir`` sets the subprocess cwd only; the Python process cwd is NEVER mutated (an
    ``os.chdir()`` would leak into concurrent gateway sessions).
    """
    scripts_dir = _sched._get_hermes_home() / "scripts"
    _ensure_cron_dir(scripts_dir)
    scripts_dir_resolved = scripts_dir.resolve()

    # Same contract as cron.lifecycle_guard._expand_candidate_path. Reject NUL eagerly: on Windows
    # Path ops raise ValueError *after* expanduser so the try below would not catch it. str() first
    # so the guard itself cannot raise on a non-str script_path.
    if "\x00" in str(script_path):
        return False, f"Blocked: script path contains a NUL byte: {script_path!r}"

    try:
        raw = _sched.Path(script_path).expanduser()
    except (ValueError, RuntimeError, OSError):
        # RuntimeError: unexpandable ``~`` (no resolvable HOME).
        return False, f"Blocked: script path is not a valid filesystem path: {script_path!r}"
    path = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()

    # Traversal / absolute-path / symlink escape guard — MUST stay inside HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    script_timeout = _sched._get_script_timeout()

    # Interpreter by extension; the shebang is deliberately NOT honoured (small, auditable surface).
    suffix = path.suffix.lower()
    if suffix in {".sh", ".bash"}:
        # which() finds Git Bash on Windows; None there → clear error instead of a "[WinError 2]".
        _bash = shutil.which("bash") or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
        if _bash is None:
            return False, (
                f"Cannot run .sh/.bash script {path.name!r}: bash not found on PATH. "
                "On Windows, install Git for Windows (which ships Git Bash) "
                "or rewrite the script as Python (.py)."
        )
        argv = [_bash, str(path)]
        env_overlay: dict[str, str] = {}
    else:
        python_exe, env_overlay = _sched._windows_cron_python_invocation(sys.executable)
        if env_overlay:
            # Windows uv-venv overlay: needs the .pth bootstrap for editable installs.
            argv = _windows_cron_bootstrap_argv(python_exe, env_overlay, str(path))
        else:
            argv = [python_exe, str(path)]

    try:
        from tools.environments.local import build_subprocess_env

        popen_kwargs: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            popen_kwargs = {
                "creationflags": _sched.windows_hide_flags()
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                "encoding": "utf-8",
                "errors": "replace",
            }
        env = build_subprocess_env()
        env.update(env_overlay)
        # Subprocess cwd only (default: scripts-dir parent). NEVER os.chdir() the process.
        _script_cwd = workdir or str(path.parent)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_script_cwd,
            env=env,
            **popen_kwargs,
        )
        deadline = time.monotonic() + script_timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # Tree-kill here too: a cancelled fire must not orphan own-session grandchildren.
                _sched._terminate_cron_script_tree(proc)
                _drain_script_pipes(proc)
                return False, "Script cancelled because cron fire ownership was lost"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Timeout must leave ZERO descendants: killpg misses setsid grandchildren
                # (watchdogs, backgrounded shell jobs); kill_process_tree snapshots descendants
                # BEFORE signalling.
                _sched._terminate_cron_script_tree(proc)
                _drain_script_pipes(proc)
                return False, f"Script timed out after {script_timeout}s: {path}"
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        stdout = (stdout_raw or "").strip()
        stderr = (stderr_raw or "").strip()

        # Redact secrets before ANY return path.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as e:
            logger.warning("Failed to redact sensitive text from output: %s", e)
            stdout = "[REDACTED - redaction failed]"
            stderr = "[REDACTED - redaction failed]"

        if proc.returncode != 0:
            parts = [f"Script exited with code {proc.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _start_heartbeat_thread(loop_fn, name: str, fail_log) -> Optional[threading.Thread]:
    """Start ``loop_fn`` on a daemon thread inside a copy of the current context (multiplexed
    profile ContextVars). On failure calls ``fail_log()`` inside the except (traceback intact) and
    returns None."""
    thread = threading.Thread(
        target=contextvars.copy_context().run, args=(loop_fn,), name=name, daemon=True,
    )
    try:
        thread.start()
    except Exception:
        fail_log()
        return None
    return thread


def _run_job_script_with_claim_heartbeat(
    job: dict,
    script_path: str,
    workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> tuple[bool, str]:
    """Run a cron script while heartbeating its owned one-shot claim.

    A long script can outlive the stale-claim TTL; without a heartbeat another scheduler would
    re-dispatch the one-shot. Recurring/unclaimed runs have no durable claim → no thread. The owner
    is captured from the dispatched job, never re-read, so a stale runner cannot extend a
    replacement owner's claim.
    """
    schedule = job.get("schedule")
    claim = job.get("run_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not (
        isinstance(schedule, dict)
        and schedule.get("kind") == "once"
        and owner
    ):
        return _sched._run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)

    job_id = str(job.get("id") or "")
    stop = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop.wait(_sched._RUN_CLAIM_HEARTBEAT_SECONDS):
            try:
                _sched.heartbeat_run_claim(job_id, expected_owner=owner)
            except Exception:
                logger.debug("Job '%s': script run_claim heartbeat failed", job_id, exc_info=True)

    heartbeat_thread = _start_heartbeat_thread(
        _heartbeat_loop, "cron-script-claim-heartbeat",
        lambda: logger.debug(
            "Job '%s': could not start script run_claim heartbeat", job_id, exc_info=True,
        ),
    )
    if heartbeat_thread is None:
        return _sched._run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)

    try:
        return _sched._run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)
    finally:
        stop.set()
        # Bounded join: the heartbeat may be blocked on another process's jobs-file lock.
        heartbeat_thread.join(timeout=1.0)


# Late-bound origin namespace (see module docstring). Imported LAST so this module is fully
# populated before ``scheduler`` re-exports from it.
from cron import scheduler as _sched  # noqa: E402
