"""Dashboard/serve support: managed-service restart (systemd/respawn), status/listening probes, SSH session token file, named-profile routing, web-dist resolution, update stdio hangup protection.

Split out of ``hermes_cli/main.py``; every moved name is re-imported there, so
``hermes_cli.main.<name>`` keeps resolving (and monkeypatching) as before.
Names that stay in main are imported lazily inside the functions that use them
(call-time resolution keeps ``hermes_cli.main.<name>`` patches effective and
avoids an import cycle).
"""

import os
import re
import shlex
import subprocess
import sys

from pathlib import Path
from hermes_cli.cli_output import line_input


def _find_stale_dashboard_pids(
    *,
    exclude_pids: set[int] | None = None,
) -> list[int]:
    """Return PIDs of stale ``dashboard``/``serve`` processes for update cleanup."""
    from hermes_cli.main import _self
    return [pid for pid, _cmd in _self()._scan_dashboard_processes(exclude_pids=exclude_pids)]


def _parse_dashboard_runtime(command: str) -> tuple[str, str, int] | None:
    """Best-effort parse of a dashboard/server cmdline into mode, host, and port."""
    mode = None
    if any(
        pattern in command
        for pattern in (
            "hermes dashboard",
            "hermes_cli.main dashboard",
            "hermes_cli/main.py dashboard",
        )
    ):
        mode = "dashboard"
    elif any(
        pattern in command
        for pattern in (
            "hermes serve",
            "hermes_cli.main serve",
            "hermes_cli/main.py serve",
        )
    ):
        mode = "serve"
    if mode is None:
        return None

    port = 9119
    host = "127.0.0.1"

    port_match = re.search(r"(?:^|\s)--port(?:=|\s+)(\d+)", command)
    if port_match:
        try:
            port = int(port_match.group(1))
        except ValueError:
            return None

    host_match = re.search(r"(?:^|\s)--host(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)", command)
    if host_match:
        host = host_match.group(1).strip("\"'") or "127.0.0.1"

    return mode, host, port


def _dashboard_probe_host(host: str | None) -> str:
    """Map wildcard binds to a loopback address suitable for local probing."""
    normalized = (host or "127.0.0.1").strip().strip("[]")
    if normalized in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return normalized


_DASHBOARD_SYSTEMD_UNIT = "hermes-dashboard.service"


def _restart_managed_dashboard_service(
    reason: str,
    unit: str = _DASHBOARD_SYSTEMD_UNIT,
) -> bool:
    """Restart a systemd-managed dashboard instead of raw-killing its PID.

    Returns True when a dashboard unit was found and handled (successfully or
    with a printed actionable failure).  Returning True deliberately prevents
    the caller from falling back to ``os.kill``: systemd treats a direct
    SIGTERM of the service's main PID as a clean stop, so ``Restart=on-failure``
    will not bring the dashboard back.
    """
    if sys.platform == "win32":
        return False

    def _systemctl(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["systemctl", *args],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )

    # Probe the user manager first: Hermes installs Linux services in the
    # user's systemd scope by default.  Only fall back to the system manager
    # when the unit is not present there, preserving root/system deployments.
    # Crucially, keep the selected scope for *all* probes and the restart — a
    # user unit must never be restarted through the system manager (or raw-killed).
    scope: tuple[str, ...] | None = None
    listed: subprocess.CompletedProcess | None = None
    for candidate in (("--user",), ()):
        try:
            result = _systemctl(
                *candidate, "list-unit-files", unit, "--no-legend", "--no-pager"
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        unit_rows = (result.stdout or "").splitlines()
        if any(row.split()[0:1] == [unit] for row in unit_rows if row.split()):
            scope = candidate
            listed = result
            break

    if scope is None or listed is None:
        return False

    try:
        active = _systemctl(*scope, "is-active", unit)
        enabled = _systemctl(*scope, "is-enabled", unit)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    active_state = (active.stdout or "").strip()
    enabled_state = (enabled.stdout or "").strip()
    if active_state != "active" and enabled_state not in {
        "enabled",
        "enabled-runtime",
        "linked",
        "linked-runtime",
        "static",
        "generated",
    }:
        return False

    print()
    print(f"⟲ Restarting managed dashboard service ({reason})")

    scope_label = "systemctl --user" if scope else "sudo systemctl"
    restart = ("systemctl", *scope, "restart", unit)
    commands = [restart]
    if not scope:
        # System units may require privilege escalation; user units must use
        # the user manager directly and never prompt for sudo.
        commands.append(("sudo", "-n", "systemctl", "restart", unit))

    errors: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(
                list(command),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            errors.append(f"{' '.join(command)}: {e}")
            continue
        if result.returncode == 0:
            print(f"    ✓ restarted {unit}")
            return True
        errors.append(
            f"{' '.join(command)}: {(result.stderr or result.stdout or '').strip()}"
        )

    print(f"    ✗ failed to restart {unit}")
    for err in errors:
        if err.strip():
            print(f"      {err}")
    print(
        "  Dashboard is managed by systemd; not raw-killing its PID because "
        "systemd would treat that as a clean stop."
    )
    print(f"  Restart manually: {scope_label} restart {unit}")
    return True


def _get_systemd_service_for_pid(pid: int) -> str | None:
    """If *pid* belongs to a systemd service unit, return the unit name.

    Reads ``/proc/<pid>/cgroup`` and extracts the service name (e.g.
    ``hermes-serve.service``).  Returns ``None`` when the PID is not
    part of a systemd service, when the file is unreadable, or on
    non-Linux platforms.
    """
    try:
        cgroup_path = Path(f"/proc/{pid}/cgroup")
        if not cgroup_path.is_file():
            return None
        text = cgroup_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            # Format: 0::/system.slice/hermes-serve.service
            #         0::/user.slice/user-1000.slice/session-42.scope
            parts = line.split("::", 1)
            if len(parts) != 2:
                continue
            cg_path = parts[1]
            if cg_path.endswith(".service"):
                svc_name = cg_path.rsplit("/", 1)[-1]
                if svc_name:
                    return svc_name
    except (OSError, PermissionError):
        pass
    return None


def _extract_scope_from_cgroup(cgroup_entry: str) -> str | None:
    """Extract the systemd scope (``user`` or ``system``) from a cgroup path.

    The cgroup path format is ``/system.slice/<name>.service`` for system
    services and ``/user.slice/user-<uid>.slice/<name>.service`` for user
    services.  Returns ``None`` when the scope cannot be determined.
    """
    if "/system.slice/" in cgroup_entry:
        return "system"
    if "/user.slice/" in cgroup_entry:
        return "user"
    return None


def _get_pid_cgroup_path(pid: int) -> str | None:
    """Return the cgroup path from ``/proc/<pid>/cgroup``, or ``None``.

    Only the unified (``0::``) hierarchy cgroup entry is examined.
    """
    try:
        cgroup_path = Path(f"/proc/{pid}/cgroup")
        if not cgroup_path.is_file():
            return None
        text = cgroup_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            parts = line.split("::", 1)
            if len(parts) == 2:
                return parts[1]
    except (OSError, PermissionError):
        pass
    return None


def _try_restart_systemd_service(svc_name: str, cgroup_path: str | None = None) -> bool:
    """Attempt to restart *svc_name* via systemctl.

    Uses ``systemctl --user`` for user-scope services and ``systemctl``
    for system-scope services.  Returns ``True`` on success.
    """
    scope = _extract_scope_from_cgroup(cgroup_path) if cgroup_path else None
    if scope == "user":
        cmd = ["systemctl", "--user", "restart", svc_name]
    elif scope == "system":
        cmd = ["systemctl", "restart", svc_name]
    else:
        # Unknown scope — try system first, then user
        cmd = None
        for candidate in (
            ["systemctl", "restart", svc_name],
            ["systemctl", "--user", "restart", svc_name],
        ):
            try:
                r = subprocess.run(
                    candidate,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=15,
                )
                if r.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return False

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=15,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _dashboard_cmdline_for_pid(pid: int) -> list[str] | None:
    """Return the exact argv of a running process, when recoverable.

    Linux: reads ``/proc/<pid>/cmdline`` (NUL-separated, lossless).
    macOS: falls back to ``ps -o command=`` + shlex (best effort — quoting
    is reconstructed, but hermes launch commands don't embed exotic args).
    Windows: returns ``None``; taskkill /F gives no graceful window and the
    desktop app manages its own backend there.
    """
    if sys.platform == "win32":
        return None
    try:
        cmdline_path = f"/proc/{pid}/cmdline"
        if os.path.exists(cmdline_path):
            with open(cmdline_path, "rb") as f:
                raw = f.read()
            argv = [
                part.decode("utf-8", errors="replace")
                for part in raw.split(b"\x00")
                if part
            ]
            return argv or None
        # macOS (no /proc): best-effort via ps.
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            return None
        command = (result.stdout or "").strip()
        if not command:
            return None
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = command.split()
        return argv or None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _respawn_dashboard_processes(commands: list[list[str]]) -> list[list[str]]:
    """Best-effort respawn of manually-started dashboards after ``hermes update``.

    Spawns each recovered argv detached (new session, output to the profile's
    ``logs/dashboard-restart.log``).  Returns the commands that failed to
    spawn; the caller prints the manual hint for those.

    Callers must pre-filter via ``_filter_dashboard_respawn_candidates`` so
    Desktop ``serve|dashboard --port 0`` backends are not replayed and
    duplicates are capped per profile (#78821).
    """
    from hermes_constants import get_hermes_home

    respawned: list[list[str]] = []
    failed: list[tuple[list[str], str]] = []
    log_path = get_hermes_home() / "logs" / "dashboard-restart.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    for command in commands:
        try:
            # Keep restarted dashboards headless; reopening a browser after a
            # background update is noisy and fails in SSH/headless sessions.
            if "dashboard" in command and "--no-open" not in command:
                command = [*command, "--no-open"]
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            respawned.append(command)
        except (OSError, ValueError) as exc:
            failed.append((command, str(exc)))

    for command in respawned:
        print(f"    ✓ restarted: {shlex.join(command)}")
    for command, err_msg in failed:
        print(f"    ✗ failed to restart ({shlex.join(command)}): {err_msg}")
    return [command for command, _ in failed]


class _UpdateOutputStream:
    """Stream wrapper used during ``hermes update`` to survive terminal loss.

    Wraps the process's original stdout/stderr so that:

    * Every write is also mirrored to an append-only log file
      (``~/.hermes/logs/update.log``) that users can inspect after the
      terminal disconnects.
    * Writes to the original stream that fail with ``BrokenPipeError`` /
      ``OSError`` / ``ValueError`` (closed file) no longer cascade into
      process exit — the update keeps going, only the on-screen output
      stops.

    Combined with ``SIGHUP -> SIG_IGN`` installed by
    ``_install_hangup_protection``, this makes ``hermes update`` safe to
    run in a plain SSH session that might disconnect mid-install.
    """

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file
        self._original_broken = False

    def write(self, data):
        # Mirror to the log file first — it's the most reliable destination.
        if self._log is not None:
            try:
                self._log.write(data)
            except Exception:
                # Log errors should never abort the update.
                pass

        if self._original_broken:
            return len(data) if isinstance(data, (str, bytes)) else 0

        try:
            return self._original.write(data)
        except (BrokenPipeError, OSError, ValueError):
            # Terminal vanished (SSH disconnect, shell close).  Stop trying
            # to write to it, but keep the update running.
            self._original_broken = True
            return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        if self._log is not None:
            try:
                self._log.flush()
            except Exception:
                pass
        if self._original_broken:
            return
        try:
            self._original.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._original_broken = True

    def isatty(self):
        if self._original_broken:
            return False
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        # Some tools probe fileno(); defer to the underlying stream and let
        # callers handle failures (same behaviour as the unwrapped stream).
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_hangup_protection(gateway_mode: bool = False):
    """Protect ``cmd_update`` from SIGHUP and broken terminal pipes.

    Users commonly run ``hermes update`` in an SSH session or a terminal
    that may close mid-install.  Without protection, ``SIGHUP`` from the
    terminal kills the Python process during ``pip install`` and leaves
    the venv half-installed; the documented workaround ("use screen /
    tmux") shouldn't be required for something as routine as an update.

    Protections installed:

    1. ``SIGHUP`` is set to ``SIG_IGN``.  POSIX preserves ``SIG_IGN``
       across ``exec()``, so pip and git subprocesses also stop dying on
       hangup.
    2. ``sys.stdout`` / ``sys.stderr`` are wrapped to mirror output to
       ``~/.hermes/logs/update.log`` and to silently absorb
       ``BrokenPipeError`` when the terminal vanishes.

    ``SIGINT`` (Ctrl-C) and ``SIGTERM`` (systemd shutdown) are
    **intentionally left alone** — those are legitimate cancellation
    signals the user or OS sent on purpose.

    In gateway mode (``hermes update --gateway``) the update is already
    spawned detached from a terminal, so this function is a no-op.

    Returns a dict that ``cmd_update`` can pass to
    ``_finalize_update_output`` on exit.  Returning a dict rather than a
    tuple keeps the call site forward-compatible with future additions.
    """
    state = {
        "prev_stdout": sys.stdout,
        "prev_stderr": sys.stderr,
        "log_file": None,
        "installed": False,
    }

    if gateway_mode:
        return state

    import signal as _signal

    # (1) Ignore SIGHUP for the remainder of this process.
    if hasattr(_signal, "SIGHUP"):
        try:
            _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)
        except (ValueError, OSError):
            # Called from a non-main thread — not fatal.  The update still
            # runs, just without hangup protection.
            pass

    # (2) Mirror output to update.log and wrap stdio for broken-pipe
    # tolerance.  Any failure here is non-fatal; we just skip the wrap.
    try:
        # Late-bound import so tests can monkeypatch
        # hermes_cli.config.get_hermes_home to simulate setup failure.
        from hermes_cli.config import get_hermes_home as _get_hermes_home

        logs_dir = _get_hermes_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "update.log"
        log_file = open(log_path, "a", buffering=1, encoding="utf-8")

        import datetime as _dt

        log_file.write(
            f"\n=== hermes update started "
            f"{_dt.datetime.now().isoformat(timespec='seconds')} ===\n"
        )

        state["log_file"] = log_file
        sys.stdout = _UpdateOutputStream(state["prev_stdout"], log_file)
        sys.stderr = _UpdateOutputStream(state["prev_stderr"], log_file)
        state["installed"] = True
    except Exception:
        # Leave stdio untouched on any setup failure.  Update continues
        # without mirroring.
        state["log_file"] = None

    return state


def _finalize_update_output(state):
    """Restore stdio and close the update.log handle opened by ``_install_hangup_protection``."""
    if not state:
        return
    if state.get("installed"):
        try:
            sys.stdout = state.get("prev_stdout", sys.stdout)
        except Exception:
            pass
        try:
            sys.stderr = state.get("prev_stderr", sys.stderr)
        except Exception:
            pass
    log_file = state.get("log_file")
    if log_file is not None:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass


def _report_dashboard_status() -> int:
    """Print live listening dashboard/serve processes and return the count.

    Serve-mode backends are INCLUDED (#81564): `--stop` kills them, so
    `--status` hiding them left Desktop SSH backends invisible to the CLI —
    an operator could kill what they couldn't see. Ledger-registered serves
    (profiled launches the argv scan can't match) surface via the
    spawn-ledger augmentation in _scan_dashboard_processes.
    """
    from hermes_cli.main import _dashboard_listening, _self
    from gateway.status import _pid_exists

    live: list[tuple[int, str, str]] = []
    for pid, command in _self()._scan_dashboard_processes():
        runtime = _parse_dashboard_runtime(command)
        if runtime is None:
            continue
        mode, host, port = runtime
        if port <= 0 or not _pid_exists(pid):
            continue
        if not _dashboard_listening(host, port):
            continue
        live.append((pid, command, mode))

    if not live:
        print("No hermes dashboard or serve processes running.")
        return 0

    print(f"{len(live)} hermes dashboard/serve process(es) running:")
    for pid, command, mode in live:
        print(f"    PID {pid} [{mode}]: {command}")
    return len(live)


def _dashboard_listening(host: str, port: int) -> bool:
    """True when something is accepting TCP connections at host:port.

    Any listener counts — even a 401 response proves a dashboard is up.
    Used by the unified profile-launch routing to decide attach-vs-start.
    """
    import socket

    try:
        with socket.create_connection((_dashboard_probe_host(host), port), timeout=1.5):
            return True
    except OSError:
        return False


def _maybe_setup_dashboard_auth_interactively(args) -> None:
    """Offer to configure dashboard auth when the gate engages and none exists.

    Called from ``cmd_dashboard`` just before ``start_server``. The auth
    gate engages on every non-loopback bind (``--insecure`` is a no-op since
    the June 2026 hardening) and whenever ``dashboard.public_url`` declares a
    non-loopback browser-facing hostname. ``start_server`` fails closed when no
    ``DashboardAuthProvider`` is registered. Rather than greet an interactive
    operator with that hard error, prompt them to set up the bundled password
    provider on the spot — or point them at ``hermes dashboard register`` for
    OAuth.

    No-ops (so the existing fail-closed ``SystemExit`` remains the backstop)
    when:
      * neither the bind nor configured public URL engages the gate, or
      * a provider is already registered, or
      * stdin/stdout isn't a TTY (Docker/s6, CI, piped ``--no-open`` runs).
    """
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"

    try:
        from hermes_cli.web_server import should_require_dashboard_auth
        if not should_require_dashboard_auth(host):
            return  # local-only bind and URL — gate does not engage
    except Exception:
        return  # if we can't tell, defer to start_server's own gate

    try:
        from hermes_cli.dashboard_auth import list_providers
        if list_providers():
            return  # a provider is already configured/registered
    except Exception:
        return

    # Only prompt an interactive operator. Non-TTY callers fall through to
    # start_server's fail-closed SystemExit (with the corrected fix hint).
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    print()
    print(f"⚠ Dashboard authentication is required for this configuration ({host}).")
    print(
        "  Non-loopback binds and configured external dashboard.public_url "
        "values require authentication (--insecure does not bypass this)."
    )
    print()
    print("  How do you want to authenticate the dashboard?")
    print("    [1] Username & password (quickest; for a trusted LAN / VPN)")
    print("    [2] OAuth via Nous Portal (run `hermes dashboard register`)")
    print("    [3] Cancel")
    print()

    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)

    if choice == "2":
        print()
        print(
            "  Run this on the host where the dashboard lives, then start "
            "the dashboard again:\n"
            "    hermes dashboard register\n"
            "  It provisions a Nous Portal OAuth client and writes "
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID into ~/.hermes/.env for you.\n"
            "  Docs: https://hermes-agent.nousresearch.com/docs/"
            "user-guide/features/web-dashboard#authentication-gated-mode"
        )
        sys.exit(0)

    if choice not in ("1",):
        print("  Cancelled.")
        sys.exit(1)

    # ── Username/password setup ──────────────────────────────────────────
    import getpass
    import secrets

    print()
    try:
        username = line_input("  Username [admin]: ").strip() or "admin"
        password = getpass.getpass("  Password: ")
        confirm = getpass.getpass("  Confirm password: ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)

    if not password:
        print("  ✗ Empty password — aborting.")
        sys.exit(1)
    if password != confirm:
        print("  ✗ Passwords don't match — aborting.")
        sys.exit(1)

    try:
        from plugins.dashboard_auth.basic import hash_password
    except Exception as exc:
        print(f"  ✗ Could not load the password provider: {exc}")
        sys.exit(1)

    password_hash = hash_password(password)
    # A stable token-signing secret so sessions survive a dashboard restart.
    secret = secrets.token_urlsafe(32)

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.plugins_cmd import ensure_basic_auth_plugin_enabled_in_config

        cfg = load_config()
        dash = cfg.setdefault("dashboard", {})
        basic = dash.setdefault("basic_auth", {})
        basic["username"] = username
        basic["password_hash"] = password_hash
        # Never persist plaintext: clear any stale plaintext password key.
        basic["password"] = ""
        if not str(basic.get("secret", "") or "").strip():
            basic["secret"] = secret
        # The bundled basic provider is a backend plugin that still honours
        # plugins.disabled. Unblock it when we just wrote basic_auth so the
        # discover_plugins(force=True) call below can register the provider
        # (#54489). Surface the mutation so an operator who deliberately
        # disabled it isn't surprised.
        if ensure_basic_auth_plugin_enabled_in_config(cfg):
            print(
                "  ✓ Re-enabled the bundled 'basic' auth plugin "
                "(was in plugins.disabled)"
            )
        save_config(cfg)
    except Exception as exc:
        print(f"  ✗ Failed to write config.yaml: {exc}")
        sys.exit(1)

    # Re-run plugin discovery so the basic provider registers from the
    # just-written config before start_server's gate check runs.
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins(force=True)
    except Exception as exc:
        print(f"  ⚠ Plugin re-discovery failed ({exc}); the gate may still "
              "fail closed. Set the password again or restart the dashboard.")

    print()
    print(f"  ✓ Username/password auth configured (user: {username}).")
    print("    Saved to config.yaml under dashboard.basic_auth.")
    print("    Sign in at the dashboard with these credentials.")
    print()


def _read_ssh_session_token_file(path: str) -> str:
    """Read and unlink a Desktop SSH token from its private runtime directory."""
    if sys.platform == "win32":
        from hermes_cli.windows_ssh_runtime import read_token
        return read_token(path)

    import stat as _stat
    from pathlib import Path as _Path

    if not os.path.isabs(path):
        raise SystemExit("--ssh-session-token-file must be absolute")

    token_path = _Path(path)
    # The Desktop client writes the token under $HOME/.hermes/desktop-ssh: a
    # literal "~/.hermes/desktop-ssh" in apps/desktop/electron/remote-lifecycle.ts
    # expanded against the account's $HOME, independent of HERMES_HOME and the
    # active profile. Anchor validation to that same OS-home path, NOT to
    # get_hermes_home(): a non-default sticky profile (or any HERMES_HOME pointing
    # elsewhere, e.g. a Docker /opt/data root) re-homes get_hermes_home() and
    # would otherwise reject every token the client legitimately wrote (#69551).
    token_root = _Path.home() / ".hermes" / "desktop-ssh"
    try:
        relative = token_path.relative_to(token_root)
    except ValueError as exc:
        raise SystemExit("--ssh-session-token-file must be under the desktop-ssh directory") from exc
    if len(relative.parts) != 2 or not re.fullmatch(r"[0-9a-f]{32}", relative.parts[0]):
        raise SystemExit("--ssh-session-token-file has an invalid runtime path")
    if not re.fullmatch(r"[0-9a-f]{16}\.token", relative.parts[1]):
        raise SystemExit("--ssh-session-token-file has an invalid filename")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = -1
    directory_fd = -1
    file_fd = -1
    try:
        try:
            root_fd = os.open(token_root, directory_flags)
            root_stat = os.fstat(root_fd)
            if not _stat.S_ISDIR(root_stat.st_mode):
                raise SystemExit("--ssh-session-token-file has an unsafe runtime root")
            if hasattr(os, "getuid") and root_stat.st_uid != os.getuid():
                raise SystemExit("--ssh-session-token-file runtime root has the wrong owner")
            directory_fd = os.open(relative.parts[0], directory_flags, dir_fd=root_fd)
            directory_stat = os.fstat(directory_fd)
            if not _stat.S_ISDIR(directory_stat.st_mode):
                raise SystemExit("--ssh-session-token-file has an unsafe parent directory")
            if hasattr(os, "getuid") and directory_stat.st_uid != os.getuid():
                raise SystemExit("--ssh-session-token-file parent has the wrong owner")
            if (directory_stat.st_mode & 0o777) != 0o700:
                raise SystemExit("--ssh-session-token-file parent has unsafe permissions")
            file_fd = os.open(relative.parts[1], file_flags, dir_fd=directory_fd)
        except SystemExit:
            raise
        except OSError as exc:
            if exc.errno == getattr(__import__("errno"), "ELOOP", -1):
                raise SystemExit("--ssh-session-token-file is a symlink") from exc
            raise SystemExit("--ssh-session-token-file is not accessible") from exc

        file_stat = os.fstat(file_fd)
        if not _stat.S_ISREG(file_stat.st_mode):
            raise SystemExit("--ssh-session-token-file is not a regular file")
        if file_stat.st_size != 64:
            raise SystemExit("--ssh-session-token-file contains an invalid token")
        if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
            raise SystemExit("--ssh-session-token-file has the wrong owner")
        if hasattr(os, "getuid") and (file_stat.st_mode & 0o777) & ~0o600:
            raise SystemExit("--ssh-session-token-file has unsafe permissions")

        with os.fdopen(file_fd, "r", encoding="utf-8") as token_stream:
            file_fd = -1
            token = token_stream.read(65)

        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise SystemExit("--ssh-session-token-file contains an invalid token")
        return token
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            try:
                os.unlink(relative.parts[1], dir_fd=directory_fd)
            except OSError:
                pass
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _is_electron_packaged_web_dist(path: str) -> bool:
    """True when *path* looks like an Electron-packaged renderer dist.

    Packaged Desktop sets ``HERMES_WEB_DIST`` to ``.../app.asar/dist`` or
    ``.../app.asar.unpacked/dist``. A standalone ``hermes dashboard`` that
    inherits that value serves the desktop frontend in the browser
    (issue #52945 — "Desktop IPC bridge is unavailable").
    """
    if not path:
        return False
    # Both app.asar and app.asar.unpacked contain this marker; normalize
    # separators so Windows paths match too.
    return "app.asar" in path.replace("\\", "/")


def _route_named_profile_dashboard(
    args, _headless_backend: bool, _ssh_owner_nonce: str, _token_file: str
) -> None:
    """Named-profile launches route to the single MACHINE dashboard.

    The dashboard manages every profile via per-request ``?profile=`` scoping,
    so one server per profile only fragments it (port collisions, N
    processes). When a named profile launches: if the machine dashboard is
    already listening, open the browser at ``?profile=<name>`` and exit;
    otherwise re-exec as the machine dashboard pinned to ``-p default`` (so
    ``_apply_profile_override`` can't re-route through the sticky
    active_profile file) with this profile preselected. ``--isolated`` opts
    out; Desktop pool backends (HERMES_DESKTOP=1) stay per-profile. Returns
    normally when no routing applies.
    """
    from hermes_cli.main import _dashboard_listening
    try:
        from hermes_cli.profiles import get_active_profile_name
        _launch_profile = get_active_profile_name()
    except Exception:
        _launch_profile = "default"

    if (
        _launch_profile not in ("default", "custom")
        and not getattr(args, "isolated", False)
        and not getattr(args, "open_profile", "")
        # Desktop pool backends are intentionally per-profile.
        and os.environ.get("HERMES_DESKTOP") != "1"
    ):
        url = f"http://{args.host or '127.0.0.1'}:{args.port}/?profile={_launch_profile}"
        if _dashboard_listening(args.host, args.port):
            print(f"Machine dashboard already running on port {args.port}.")
            print(f"  Managing profile '{_launch_profile}': {url}")
            if not args.no_open:
                try:
                    import webbrowser
                    webbrowser.open(url)
                except Exception:
                    pass
            sys.exit(0)

        print(
            f"Routing to the machine dashboard (profile '{_launch_profile}' "
            f"preselected). Use --isolated for a dedicated per-profile server."
        )
        reexec_argv = [
            sys.executable, "-m", "hermes_cli.main",
            "-p", "default",
            # Preserve the lean serve path across the re-exec so a named-profile
            # `serve` doesn't silently rebuild the UI as `dashboard`.
            "serve" if _headless_backend else "dashboard",
            "--port", str(args.port),
            "--host", args.host,
            "--open-profile", _launch_profile,
        ]
        if _ssh_owner_nonce:
            reexec_argv.extend(["--ssh-owner-nonce", _ssh_owner_nonce])
        if _token_file:
            reexec_argv.extend(["--ssh-session-token-file", _token_file])
        if args.no_open:
            reexec_argv.append("--no-open")
        if getattr(args, "insecure", False):
            reexec_argv.append("--insecure")
        if getattr(args, "skip_build", False):
            reexec_argv.append("--skip-build")
        from tools.environments.local import build_subprocess_env
        # Exact env preservation: HERMES_HOME is explicitly pinned to the
        # machine root below — the factory must not re-inject a profile home.
        env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
        # Pin the child to the machine ROOT, not the launching profile's
        # HERMES_HOME.  We must resolve the root explicitly instead of just
        # dropping HERMES_HOME: in the Docker layout the machine root is
        # /opt/data (set via `ENV HERMES_HOME=/opt/data`), so an unset
        # HERMES_HOME falls back to $HOME/.hermes = /opt/data/.hermes — an
        # empty, auto-seeded home where the dashboard sees only the default
        # profile and the install-method stamp is missing (so the Docker
        # update-button guard also misfires).  get_default_hermes_root()
        # returns the root for both layouts: ~/.hermes for a standard install
        # and /opt/data for Docker (it strips a trailing profiles/<name>).
        # See the support report for the double-mount workaround this avoids.
        try:
            from hermes_constants import get_default_hermes_root
            env["HERMES_HOME"] = str(get_default_hermes_root())
        except Exception:
            # Best-effort: if root resolution fails, fall back to the prior
            # behaviour (drop HERMES_HOME) rather than block the reroute.
            env.pop("HERMES_HOME", None)
        # On Windows, os.execvpe() does not truly replace the process — it
        # spawns via CreateProcess then the parent exits.  Under Python 3.14+
        # this can crash with STATUS_ACCESS_VIOLATION (0xC0000005) when
        # re-executing the dashboard for a non-default profile.  Use
        # subprocess.Popen + sys.exit() on Windows to avoid the crash.
        if sys.platform == "win32":
            proc = subprocess.Popen(reexec_argv, env=env)
            sys.exit(proc.wait())
        else:
            os.execvpe(sys.executable, reexec_argv, env)


def _resolve_dashboard_web_dist(args, _headless_backend: bool) -> None:
    """Build or validate the web UI dist before the server imports.

    ``serve`` sets HERMES_SERVE_HEADLESS so mount_spa() stays off even if a
    stray dist exists. Otherwise build unless HERMES_WEB_DIST or --skip-build
    says a dist is pre-built — in which case verify index.html exists (an
    unverified promise means the server starts and serves 404s, #23817).
    --skip-build on the default dist location gets ONE recovery build
    (#59288); a caller-managed HERMES_WEB_DIST cannot be populated and is
    written back expanded because web_server reads it raw at import.
    """
    from hermes_cli.main import PROJECT_ROOT, _build_web_ui
    if _headless_backend:
        # Don't build the SPA, and tell mount_spa() (read at web_server import
        # below) to disable it even if a stray dist exists. Set it first.
        os.environ["HERMES_SERVE_HEADLESS"] = "1"
    elif "HERMES_WEB_DIST" not in os.environ and not getattr(args, "skip_build", False):
        if not _build_web_ui(PROJECT_ROOT / "web", fatal=True):
            sys.exit(1)
    elif getattr(args, "skip_build", False):
        # --build-mode skip trusts the caller to have pre-built the web UI.
        # Verify the dist actually exists; otherwise the server will start
        # and serve 404s with no obvious cause (issue #23817).
        _dist_root = (
            Path(os.environ["HERMES_WEB_DIST"])
            if "HERMES_WEB_DIST" in os.environ
            else PROJECT_ROOT / "hermes_cli" / "web_dist"
        )
        if not (_dist_root / "index.html").exists():
            # The caller promised a pre-built dist but there isn't one.
            # Instead of hard-failing (issue #59288 — desktop launches with
            # --build-mode skip after a wipe of web_dist), warn and attempt
            # ONE recovery build through the normal build path. Only the
            # default dist location is recoverable: a custom HERMES_WEB_DIST
            # points at a caller-managed directory the build cannot populate.
            _recoverable = "HERMES_WEB_DIST" not in os.environ
            if _recoverable:
                print(f"⚠ --skip-build was passed but no web dist found at: {_dist_root}")
                print("  Attempting one recovery build of the web UI...")
                _build_web_ui(PROJECT_ROOT / "web", fatal=True)
            if not (_dist_root / "index.html").exists():
                print(f"✗ --skip-build was passed but no web dist found at: {_dist_root}")
                if _recoverable:
                    print("  The recovery build did not produce a usable dist.")
                print("  Pre-build first:  npm install --workspace web && npm run build -w web")
                print("  Or drop --skip-build to build automatically.")
                sys.exit(1)
            print("  ✓ Recovery build produced a web dist")
        print(f"→ Skipping web UI build (--skip-build); using dist at {_dist_root}")
    else:
        # HERMES_WEB_DIST is set without --skip-build: the build is skipped
        # (the env var points at a caller-managed dist), so validate it the
        # same way the --skip-build branch does — otherwise the server starts
        # and serves 404s with no obvious cause (same failure mode as #23817,
        # via the env-var path).
        _dist_root = Path(os.environ["HERMES_WEB_DIST"]).expanduser()
        if not (_dist_root / "index.html").exists():
            print(f"✗ HERMES_WEB_DIST is set but no web dist found at: {_dist_root}")
            print("  Pre-build first:  npm install --workspace web && npm run build -w web")
            print("  Or unset HERMES_WEB_DIST to build and use the default web UI dist.")
            sys.exit(1)
        # Write the expanded path back: web_server reads HERMES_WEB_DIST raw
        # at import (no expanduser), so a validated "~/dist" would otherwise
        # pass here and still 404 there.
        os.environ["HERMES_WEB_DIST"] = str(_dist_root)
        print(f"→ Using web dist from HERMES_WEB_DIST: {_dist_root}")
