"""Abstract service manager interface."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

ServiceManagerKind = Literal["systemd", "launchd", "windows", "s6", "none"]

# Profile name → service directory mapping. Profile names must be safe
# as filesystem directory names because the s6 backend creates a service
# directory at ``<scandir>/gateway-<profile>/``. We reject anything that
# could traverse paths, span filesystems, or break s6's own naming rules.
_VALID_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_PROFILE_LEN = 251  # s6-svscan default name_max


def validate_profile_name(name: str) -> None:
    """Raise ValueError if ``name`` is not usable as a profile name.

    Profile names are used as s6 service directory names, so they must match a conservative subset
    of filesystem-safe characters. Reject empty strings, uppercase, paths-traversal sequences, and
    anything longer than s6's default ``name_max``.
    """
    if not name:
        raise ValueError("profile name must not be empty")
    if len(name) > _MAX_PROFILE_LEN:
        raise ValueError(
            f"profile name too long ({len(name)} > {_MAX_PROFILE_LEN})"
        )
    if not _VALID_PROFILE_RE.match(name):
        raise ValueError(
            f"profile name must match [a-z0-9][a-z0-9_-]*, got {name!r}"
        )


@runtime_checkable
class ServiceManager(Protocol):
    """Abstract interface for init-system-specific service operations.

    Lifecycle methods (start/stop/restart/is_running) exist on every backend. Runtime registration
    (register/unregister/list_profile_gateways) is s6-only — callers MUST check
    ``supports_runtime_registration()`` before invoking it.
    """

    kind: ServiceManagerKind

    # Lifecycle of a pre-declared service.
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    # Runtime registration (s6 only).
    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(
        self,
        profile: str,
        *,
        extra_env: dict[str, str] | None = None,
        start_now: bool = True,
    ) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...


def detect_service_manager() -> ServiceManagerKind:
    """Detect which service manager is available in this environment.

    Returns "s6" (s6-svscan is PID 1), "windows", "launchd", "systemd" (working bus) or "none"
    (Termux, sandboxes). Does NOT replace ``supports_systemd_services()`` for host call sites; it
    exists for backend-agnostic code (profile hooks, the s6 dispatch in ``hermes gateway``).
    """
    # Imports deferred so importing this module doesn't drag in the
    # whole gateway dependency graph for callers that only need the
    # Protocol type or validate_profile_name().
    from hermes_cli.gateway import (
        is_macos,
        is_windows,
        supports_systemd_services,
    )

    # Gate on _s6_running() alone (PID 1 comm == s6-svscan AND /run/s6/basedir),
    # NOT is_container(): the latter only detects Docker/Podman/lxc, so it is
    # False on Fly's Firecracker microVMs even though s6-overlay is PID 1 there.
    # That false negative made the whole s6 dispatch path inert on Fly, so
    # `hermes gateway start/stop/restart` fell through to host code that spawns
    # a foreground gateway competing with the supervised one. _s6_running() is
    # already an s6-overlay-specific signal, so the container gate was redundant.
    if _s6_running():
        return "s6"
    if is_windows():
        return "windows"
    if is_macos():
        return "launchd"
    if supports_systemd_services():
        return "systemd"
    return "none"


def _s6_running() -> bool:
    """True when s6-svscan is running as PID 1 in this container.

    Must work for root AND the unprivileged hermes user: ``/proc/1/exe`` is unreadable for other
    UIDs and ``resolve()`` silently yields the literal ``exe``, which made runtime registration
    inert in production. Probe the world-readable ``/proc/1/comm`` and ``/run/s6/basedir`` instead;
    both are required since either alone can false-positive.
    """
    try:
        comm = Path("/proc/1/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if comm != "s6-svscan":
        return False
    return Path("/run/s6/basedir").is_dir()


# ---------------------------------------------------------------------------
# Backend wrappers
#
# These adapters are thin facades over the existing module-level functions
# in ``hermes_cli.gateway`` (systemd/launchd) and ``hermes_cli.gateway_windows``
# (Windows Scheduled Tasks). The protocol's ``name`` parameter is currently
# unused for host backends — they operate on whichever profile is currently
# active (set via the ``hermes -p <profile>`` flag before the call). This
# matches existing host-side semantics; the parameter shape is designed
# for s6 where each profile maps to a distinct service directory.
# ---------------------------------------------------------------------------


class _HostServiceManager:
    """Table-driven host backend: ``start``/``stop``/``restart`` resolve to
    ``<_fn_prefix><op>`` on the ``hermes_cli.<_backend>`` submodule at call time
    (lazy import; tests monkeypatch the submodule or its functions). Runtime
    registration is unsupported on every host backend.
    """

    kind: ServiceManagerKind
    _backend: str
    _fn_prefix: str = ""

    def _backend_module(self):
        import importlib

        import hermes_cli

        importlib.import_module(f"hermes_cli.{self._backend}")
        return getattr(hermes_cli, self._backend)

    def _call(self, op: str) -> None:
        getattr(self._backend_module(), f"{self._fn_prefix}{op}")()

    def start(self, name: str) -> None:
        self._call("start")

    def stop(self, name: str) -> None:
        self._call("stop")

    def restart(self, name: str) -> None:
        self._call("restart")

    def supports_runtime_registration(self) -> bool:
        return False

    def register_profile_gateway(
        self,
        profile: str,
        *,
        extra_env: dict[str, str] | None = None,
        start_now: bool = True,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support runtime profile "
            "gateway registration (container-only feature)"
        )

    def unregister_profile_gateway(self, profile: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support runtime profile "
            "gateway unregistration (container-only feature)"
        )

    def list_profile_gateways(self) -> list[str]:
        return []


class SystemdServiceManager(_HostServiceManager):
    """Thin wrapper around the ``systemd_*`` functions in hermes_cli.gateway.

    Existing host call sites keep using those functions directly; this wrapper exists for
    backend-agnostic code such as the profile create/delete hooks.
    """

    kind: ServiceManagerKind = "systemd"
    _backend = "gateway"
    _fn_prefix = "systemd_"

    def is_running(self, name: str) -> bool:
        _, running = self._backend_module()._probe_systemd_service_running()
        return running


class LaunchdServiceManager(_HostServiceManager):
    """Thin wrapper around the ``launchd_*`` functions in hermes_cli.gateway."""

    kind: ServiceManagerKind = "launchd"
    _backend = "gateway"
    _fn_prefix = "launchd_"

    def is_running(self, name: str) -> bool:
        return self._backend_module()._probe_launchd_service_running()


class WindowsServiceManager(_HostServiceManager):
    """Thin wrapper around ``hermes_cli.gateway_windows`` (Scheduled Task / Startup-folder
    fallback).

    A Scheduled Task is not a true init service, but the lifecycle protocol is the same. ``install``
    takes Windows-specific kwargs (start_now, start_on_login, elevated_handoff) passed straight
    through — non-Windows callers must never call ``install`` here.
    """

    kind: ServiceManagerKind = "windows"
    _backend = "gateway_windows"

    def install(
        self,
        *,
        force: bool = False,
        start_now: bool | None = None,
        start_on_login: bool | None = None,
        elevated_handoff: bool = False,
    ) -> None:
        self._backend_module().install(
            force=force,
            start_now=start_now,
            start_on_login=start_on_login,
            elevated_handoff=elevated_handoff,
        )

    def is_running(self, name: str) -> bool:
        from hermes_cli.gateway import find_gateway_pids
        if not self._backend_module().is_installed():
            return False
        return bool(find_gateway_pids())


def get_service_manager() -> ServiceManager:
    """Return the ServiceManager instance for the current environment."""
    cls = _MANAGER_CLASSES.get(detect_service_manager())
    if cls is None:
        raise RuntimeError("no supported service manager detected")
    return cls()


# ---------------------------------------------------------------------------
# S6ServiceManager (container-only)
#
# Per-profile gateways are registered dynamically when `hermes profile create`
# runs inside the container (Phase 4). Static services (main-hermes, dashboard)
# live in /etc/s6-overlay/s6-rc.d/ and are NOT managed by this class — they're
# part of the image, not runtime-created.
# ---------------------------------------------------------------------------


# s6-overlay's dynamic scandir for runtime-registered services. Lives on
# tmpfs and is the directory s6-svscan watches. Writes here trigger
# automatic supervision on the next rescan.
S6_DYNAMIC_SCANDIR = Path("/run/service")
S6_SERVICE_PREFIX = "gateway-"


def _profile_from_service(name: str) -> str:
    """Strip the ``gateway-`` prefix back off (matches what the user typed via ``-p``)."""
    return name[len(S6_SERVICE_PREFIX):] if name.startswith(S6_SERVICE_PREFIX) else name


def _profile_dir_for_gateway_service(name: str) -> Path:
    """Resolve ``gateway-<profile>`` to its persistent profile directory.

    s6 lifecycle commands may be invoked from any active profile, including ``gateway stop --all``.
    Do not write the caller's HERMES_HOME blindly; derive the shared profile root from the current
    HERMES_HOME and map the service suffix to either the root default profile or
    ``<root>/profiles/<profile>``.
    """
    profile = _profile_from_service(name)
    validate_profile_name(profile)
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    root = hermes_home.parent.parent if hermes_home.parent.name == "profiles" else hermes_home
    return root if profile == "default" else root / "profiles" / profile


def _write_gateway_desired_state(name: str, desired_state: str) -> None:
    """Persist durable s6 gateway intent next to runtime status.

    ``gateway_state`` stays the volatile field written by the gateway; ``desired_state`` records the
    operator's start/stop intent so boot reconciliation can restore s6 want-up/down after pod
    recreation even if the last runtime state was transient. Best-effort: a failed write must not
    block immediate s6 lifecycle control.
    """
    profile_dir = _profile_dir_for_gateway_service(name)
    state_file = profile_dir / "gateway_state.json"
    try:
        if not profile_dir.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data["desired_state"] = desired_state
        data["updated_at"] = int(time.time())
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        tmp.replace(state_file)
    except OSError:
        return


# s6-overlay installs its binaries under /command/ and only adds that
# directory to PATH for processes started under the supervision tree
# (services started by s6-svscan, cont-init.d scripts, etc.). Code
# that runs via `docker exec` or any other out-of-tree entry point —
# notably our Phase 4 profile create/delete hooks — inherits the
# container's base PATH which does NOT include /command/.
#
# Rather than asking every caller to fix up its environment, the
# S6ServiceManager calls s6-* binaries by absolute path via this
# constant. We don't use `/usr/bin/s6-…` symlinks because the
# s6-overlay-symlinks-noarch tarball only links a subset, and we
# want every s6 invocation to be guaranteed-findable.
_S6_BIN_DIR = "/command"


def _s6_run(cmd: str, *args: str, timeout: float = 5, check: bool = False):
    """Run an s6 binary by absolute path with the shared capture/decode settings."""
    return subprocess.run(
        [f"{_S6_BIN_DIR}/{cmd}", *args],
        check=check, capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
    )


# UID/GID of the in-image ``hermes`` user. Hardcoded to match what
# ``stage2-hook.sh`` enforces (the runtime invariant — see also
# tests/docker/test_uid_remap.py). The container starts s6-supervise
# under root and immediately drops to this UID via ``s6-setuidgid``.
_HERMES_UID = 10000
_HERMES_GID = 10000


def _chown_hermes(path: Path) -> None:
    try:
        os.chown(path, _HERMES_UID, _HERMES_GID)
    except PermissionError:
        # Running as the hermes user already — directory is hermes-
        # owned by default. The chown is a no-op in that case, so
        # swallowing this keeps both root and unprivileged callers
        # on one code path.
        pass


def _seed_supervise_skeleton(svc_dir: Path) -> None:
    """Pre-create the ``supervise/`` and top-level ``event/`` skeleton inside a service directory,
    owned by the hermes user.

    s6-supervise runs as root and creates ``event/``/``supervise/`` mode 0700 plus the control FIFO
    0600, so the unprivileged hermes user gets EACCES on every ``s6-svc``/``s6-svstat``. s6 treats
    EEXIST as success and skips its chown/chmod fix-up, so laying the skeleton down hermes-owned
    before ``s6-svscanctl -a`` makes s6-supervise inherit it. The same skeleton is seeded under
    ``log/`` when present, else unregister teardown EACCESes on the logger. Idempotent: existing
    entries (possibly live FIFOs) are left untouched.
    """

    def _mkdir_owned(path: Path, mode: int) -> None:
        if path.exists():
            return
        path.mkdir(parents=False, exist_ok=False)
        path.chmod(mode)
        _chown_hermes(path)

    def _seed(root: Path) -> None:
        # Top-level event/ dir (this is the s6-svlisten1 event-subscription
        # dir at the service root, distinct from supervise/event/).
        _mkdir_owned(root / "event", 0o3730)
        # supervise/ dir + its inner event/ dir.
        supervise = root / "supervise"
        _mkdir_owned(supervise, 0o755)
        _mkdir_owned(supervise / "event", 0o3730)
        # supervise/control FIFO. Same EEXIST-safe pattern: if it's already
        # there (s6-supervise has already started against this slot), leave
        # it alone. The explicit chmod after mkfifo is required because
        # mkfifo honors the process umask, which can strip group-write
        # (e.g. the default 0022 on most dev hosts → 0o660 becomes 0o640).
        # The container runs with umask 0 inside s6-overlay's stage2, but
        # being defensive here keeps the helper consistent under any
        # invocation context.
        control = supervise / "control"
        if not control.exists():
            os.mkfifo(control, 0o660)
            control.chmod(0o660)
            _chown_hermes(control)

    _seed(svc_dir)
    # If a log/ subdir is present (the canonical s6 logger pattern —
    # see servicedir(7)), it gets its own s6-supervise instance and
    # needs the same skeleton. Without this, unregister teardown
    # would EACCES on the logger's root-owned supervise/ dir even
    # when the parent slot's supervise/ is hermes-owned.
    log_dir = svc_dir / "log"
    if log_dir.is_dir():
        _seed(log_dir)


class S6Error(RuntimeError):
    """Base error for S6ServiceManager lifecycle failures.

    Subclasses carry the slot name (and subprocess output where useful) so the CLI can render an
    actionable message instead of a raw ``CalledProcessError`` traceback.
    """

    def __init__(self, message: str, *, service: str | None = None) -> None:
        super().__init__(message)
        self.service = service


class GatewayNotRegisteredError(S6Error):
    """Raised when a lifecycle method targets a slot that doesn't exist.

    Carries the unprefixed profile name (not ``gateway-<profile>``) so callers can phrase "no such
    gateway 'typo'".
    """

    def __init__(self, profile: str) -> None:
        self.profile = profile
        super().__init__(
            f"no such gateway {profile!r}: register it with "
            f"`hermes profile create {profile}` first, or pass "
            "an existing profile name via `-p <name>`",
            service=f"gateway-{profile}",
        )


class S6CommandError(S6Error):
    """Raised when an s6 command fails for a reason other than a missing slot — e.g. permission denied
    on the supervise control FIFO, or s6-svc returning a non-zero exit for an unexpected reason.
    Carries the stderr from the failing command so callers can surface it.
    """

    def __init__(
        self, *, service: str, action: str, returncode: int, stderr: str,
    ) -> None:
        self.action = action
        self.returncode = returncode
        self.stderr = stderr
        message = (
            f"s6-svc {action} on {service!r} failed (rc={returncode})"
        )
        if stderr.strip():
            message += f": {stderr.strip()}"
        super().__init__(message, service=service)


class S6ServiceManager:
    """Per-profile gateway supervision via s6-overlay.

    Only handles runtime-registered services under ``S6_DYNAMIC_SCANDIR``. Static services (main-
    hermes, dashboard) are managed by s6-rc at image-build time and are out of scope.
    """

    kind: ServiceManagerKind = "s6"

    def __init__(self, scandir: Path = S6_DYNAMIC_SCANDIR) -> None:
        self.scandir = scandir

    # -- internal helpers --------------------------------------------------

    def _service_dir(self, profile: str) -> Path:
        validate_profile_name(profile)
        return self.scandir / f"{S6_SERVICE_PREFIX}{profile}"

    @staticmethod
    def _render_run_script(
        profile: str,
        extra_env: dict[str, str],
    ) -> str:
        """Generate the run script for a profile-gateway s6 service.

        The script sources HERMES_HOME via with-contenv (honored at run time, not baked in at
        registration), resets ``HOME`` before the privilege drop so root's HOME does not leak,
        activates the venv, and drops to hermes. ``profile == "default"`` emits NO ``-p`` flag: it
        is the sentinel for the root HERMES_HOME profile, and ``-p default`` would look up
        ``profiles/default/`` instead. Port comes from the profile's own env (``API_SERVER_PORT``,
        default 8642); two profiles that both leave it unset will collide, so give each a distinct
        port.
        """
        lines = [
            "#!/command/with-contenv sh",
            "# shellcheck shell=sh",
            "set -e",
            "export HOME=/opt/data",
            "cd /opt/data",
            ". /opt/hermes/.venv/bin/activate",
        ]
        for k, v in sorted(extra_env.items()):
            lines.append(f"export {k}={shlex.quote(v)}")
        # Sentinel for the supervised-child path. Prevents recursive
        # redirect when the supervised gateway re-enters
        # `_gateway_command_inner` with subcmd == "run" — without it the
        # supervisor would dispatch `gateway start` which would re-exec
        # `gateway run --replace` which would re-dispatch `gateway
        # start`, etc. See `_gateway_command_inner` for the matching
        # guard.
        lines.append("export HERMES_S6_SUPERVISED_CHILD=1")
        # Generalized supervisor marker (#74872) — same meaning for the
        # profile-redirect guard in hermes_cli.main._apply_profile_override,
        # kept alongside the s6-specific sentinel for back-compat.
        lines.append("export HERMES_SUPERVISED_CHILD=1")
        # ``--replace`` makes the supervised gateway authoritative for its
        # profile's HERMES_HOME. Without it, a gateway started OUTSIDE s6
        # (a stray ``hermes gateway run`` from a shell, an agent action, or
        # the Open WebUI helper) grabs the per-HERMES_HOME PID lock first;
        # the supervised slot then execs a bare ``gateway run``, hits the
        # "Another gateway instance is already running" guard, exits
        # non-zero, and s6 restarts it — a restart loop that floods the
        # log and never binds (NS-505). ``--replace``
        # instead reaps the stale holder (hardened takeover path: marker +
        # SIGTERM→SIGKILL-with-confirmation + scoped-lock cleanup, see
        # gateway/run.py) so s6 always wins. The HERMES_S6_SUPERVISED_CHILD
        # sentinel above prevents the run→start→run redirect recursion.
        # Each profile is scoped to its own HERMES_HOME and s6 guarantees a
        # single supervised instance per slot, so there is no legitimate
        # supervised sibling for ``--replace`` to clobber.
        if profile == "default":
            gateway_cmd = "hermes gateway run --replace"
        else:
            gateway_cmd = f"hermes -p {shlex.quote(profile)} gateway run --replace"
        # Skip the drop when already non-root (setgroups() lacks CAP_SETGID →
        # s6 boot-loop).
        lines.append(f'[ "$(id -u)" = 0 ] || exec {gateway_cmd}')
        lines.append(f"exec s6-setuidgid hermes {gateway_cmd}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_finish_script() -> str:
        """Generate the finish script for a profile-gateway s6 service.

        Exit 78 (EX_CONFIG, fatal config error) makes the script exit 125 so s6 stops restarting. A
        clean exit 0 is an intentional stop, not a crash — restarting after it turns any normal exit
        into a reconnect storm. Only non-zero, non-78 exits let s6 restart normally.
        """
        from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

        code = GATEWAY_FATAL_CONFIG_EXIT_CODE
        return (
            "#!/command/with-contenv sh\n"
            "# shellcheck shell=sh\n"
            "# $1 = exit code from the run script.\n"
            f"# Exit {code} (EX_CONFIG) = fatal config error — don't restart.\n"
            "# Exit 0 (clean stop) = intentional stop — don't restart.\n"
            f'if [ "$1" = "{code}" ]; then\n'
            "  exit 125\n"
            "fi\n"
            'if [ "$1" = "0" ]; then\n'
            "  exit 125\n"
            "fi\n"
            "exit 0\n"
        )

    @staticmethod
    def _render_log_run(profile: str) -> str:
        """Generate the log/run script for a profile-gateway service.

        Output routing — the script is two action directives, applied per line, in order:

        ``T`` is non-sticky: it only prefixes lines for the next action directive. We deliberately
        put ``T`` between ``1`` and the log dir (not before ``1``) so:
        """
        prof = shlex.quote(profile)
        return (
            f"#!/command/with-contenv sh\n"
            f"# shellcheck shell=sh\n"
            f': "${{HERMES_HOME:=/opt/data}}"\n'
            f'log_dir="$HERMES_HOME/logs/gateways/{prof}"\n'
            # Create the leaf and clear a stale s6-log lock as hermes when
            # this script starts as root. Never chown or unlink hermes-writable
            # volume paths from this restartable root-context script:
            # log/supervise/control is hermes-owned, so an unprivileged user
            # can race a pathname op through a symlink swap (CWE-59 /
            # CWE-367). Parent logs/gateways is seeded hermes-owned at stage2
            # boot (#45258; tests/docker/test_log_dir_seed.py).
            f'if [ "$(id -u)" = 0 ]; then\n'
            f'  s6-setuidgid hermes mkdir -p "$log_dir"\n'
            f'  s6-setuidgid hermes rm -f "$log_dir/lock"\n'
            f'else\n'
            f'  mkdir -p "$log_dir"\n'
            f'  rm -f "$log_dir/lock"\n'
            f'fi\n'
            # Skip the drop when already non-root (CAP_SETGID).
            f'[ "$(id -u)" = 0 ] || exec s6-log 1 n10 s1000000 T "$log_dir"\n'
            f'exec s6-setuidgid hermes s6-log 1 n10 s1000000 T "$log_dir"\n'
        )

    # -- lifecycle ---------------------------------------------------------

    def _run_svc(self, action_flag: str, action_label: str, name: str) -> None:
        """Shared lifecycle dispatch for start / stop / restart.

        Pre-empts a missing service dir with ``GatewayNotRegisteredError`` (clear "no such gateway"
        instead of s6-svc's opaque failure) and wraps anything else (EACCES on the control FIFO,
        timeout) in ``S6CommandError`` carrying return code and stderr. ``action_flag`` is the
        ``s6-svc`` flag, ``action_label`` the human verb used in messages.
        """
        service_dir = self.scandir / name
        if not service_dir.is_dir():
            raise GatewayNotRegisteredError(_profile_from_service(name))

        try:
            _s6_run("s6-svc", action_flag, str(service_dir), check=True)
        except subprocess.CalledProcessError as exc:
            raise S6CommandError(
                service=name,
                action=action_label,
                returncode=exc.returncode,
                stderr=exc.stderr or "",
            ) from exc

    def start(self, name: str) -> None:
        """Bring up a registered service (``s6-svc -u``)."""
        self._run_svc("-u", "start", name)
        _write_gateway_desired_state(name, "running")

    def _supervised_pid(self, name: str) -> int | None:
        """Return the PID of the supervised gateway process, or None.

        Parses ``s6-svstat`` output. Used to write the planned-stop marker before an operator stop
        so the gateway classifies the SIGTERM as intentional. Best-effort: any failure returns None.
        """
        try:
            result = _s6_run("s6-svstat", str(self.scandir / name))
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        m = re.search(r"\(pid (\d+)\)", result.stdout)
        return int(m.group(1)) if m else None

    def stop(self, name: str) -> None:
        """Bring down a registered service (``s6-svc -d``).

        Writes a planned-stop marker naming the supervised gateway PID BEFORE sending the down
        command, so the gateway's shutdown handler recognises this SIGTERM as an operator-initiated
        stop and persists ``gateway_state=stopped`` (respecting the explicit intent).
        """
        pid = self._supervised_pid(name)
        if pid is not None:
            try:
                from gateway.status import write_planned_stop_marker

                write_planned_stop_marker(pid)
            except Exception:
                pass
        self._run_svc("-d", "stop", name)
        _write_gateway_desired_state(name, "stopped")

    def restart(self, name: str) -> None:
        """Restart a registered service (``s6-svc -t`` = SIGTERM)."""
        self._run_svc("-t", "restart", name)
        _write_gateway_desired_state(name, "running")

    def is_running(self, name: str) -> bool:
        """True iff ``s6-svstat`` reports the service as up."""
        result = _s6_run("s6-svstat", str(self.scandir / name))
        return result.returncode == 0 and "up " in result.stdout

    # -- runtime registration ---------------------------------------------

    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(
        self,
        profile: str,
        *,
        extra_env: dict[str, str] | None = None,
        start_now: bool = True,
    ) -> None:
        """Create the s6 service directory for a profile gateway.

        Triggers ``s6-svscanctl -a`` so s6-svscan picks the directory up immediately. With
        ``start_now=False`` a ``down`` marker is written so the service stays stopped until an
        explicit ``gateway start``. Raises ValueError on an invalid name or existing directory,
        RuntimeError if ``s6-svscanctl`` fails.
        """
        svc_dir = self._service_dir(profile)
        if svc_dir.exists():
            raise ValueError(
                f"profile gateway {profile!r} already registered at {svc_dir}"
            )

        # Build the service directory atomically: write to a sibling
        # temp dir, then rename. The staging name is DOT-PREFIXED
        # (``.gateway-<profile>.tmp``) so s6-svscan ignores it while it
        # is half-built: s6-svscan skips any scandir entry whose name
        # begins with ``.``. Without the dot prefix, a concurrent
        # ``s6-svscanctl -a`` rescan (fired by the cont-init reconciler
        # registering ``gateway-default``, or by a sibling register)
        # would supervise the still-being-seeded ``.tmp`` slot: it has a
        # valid ``type``/``run`` by that point, so s6-supervise spawns
        # AS ROOT and mkdir's ``supervise/`` root-owned 0700 — then this
        # process's ``_seed_supervise_skeleton`` early-returns on the now-
        # existing ``supervise/`` and the next ``mkdir supervise/event``
        # hits EACCES. That is the arm64-only CI flake on
        # test_s6_unregister_removes_service_dir_in_live_container
        # (the wider scheduling jitter on the native arm64 runner lets the
        # rescan land inside the ~ms seed window). The atomic rename to
        # the dotless live name below is unaffected.
        tmp_dir = svc_dir.with_name("." + svc_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)

        def _write_script(path: Path, text: str) -> None:
            path.write_text(text, encoding="utf-8")
            path.chmod(0o755)

        try:
            (tmp_dir / "type").write_text("longrun\n", encoding="utf-8")
            _write_script(tmp_dir / "run", self._render_run_script(profile, extra_env or {}))
            _write_script(tmp_dir / "finish", self._render_finish_script())
            # Persistent log rotation (OQ8-C).
            (tmp_dir / "log").mkdir()
            _write_script(tmp_dir / "log" / "run", self._render_log_run(profile))

            # Pre-create the supervise/ skeleton with hermes ownership
            # BEFORE we publish the slot. s6-supervise will EEXIST our
            # dirs/FIFOs and inherit the ownership, so the runtime
            # s6-svc / s6-svstat / s6-svwait calls (all dispatched as
            # the hermes user) won't hit EACCES on root-owned 0700
            # dirs. See ``_seed_supervise_skeleton`` for the full
            # rationale.
            _seed_supervise_skeleton(tmp_dir)

            # When start_now is False, write a `down` marker so
            # s6-supervise does not auto-start the service on rescan.
            # Mirrors the same pattern in container_boot.py
            # _register_gateway_slot when start=False.
            if not start_now:
                (tmp_dir / "down").touch()

            tmp_dir.rename(svc_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        # Trigger rescan so s6-svscan picks up the new service.
        result = _s6_run("s6-svscanctl", "-a", str(self.scandir))
        if result.returncode != 0:
            # Clean up: rescan failed, leave the directory in place would
            # be confusing (no supervisor watching it).
            shutil.rmtree(svc_dir, ignore_errors=True)
            raise RuntimeError(
                f"s6-svscanctl failed: {result.stderr or result.stdout}"
            )

    def unregister_profile_gateway(self, profile: str) -> None:
        """Stop the profile gateway service and remove its directory.

        Idempotent: absent services are a no-op. Best-effort stop + wait-for-down before removal so
        the running gateway process gets a chance to shut down cleanly before its service dir
        disappears.

        Teardown ordering matters: ``s6-svscanctl -an`` is fired **before** ``rmtree`` so s6-svscan
        reaps the supervise child process (releasing its handle on ``supervise/lock`` and the
        regular files inside the supervise dir), giving us a clean directory to remove.
        """
        svc_dir = self._service_dir(profile)
        if not svc_dir.exists():
            return

        # Stop the service (best effort — service may already be down).
        _s6_run("s6-svc", "-d", str(svc_dir))
        # Wait for it to actually go down (up to 10s).
        _s6_run("s6-svwait", "-D", "-t", "10000", str(svc_dir), timeout=15)

        # Reap the supervise child FIRST: -n tells s6-svscan to drop
        # any supervise processes whose service dir is gone (which
        # includes any service dir we're about to remove). This
        # releases the file handles s6-supervise holds against the
        # supervise/lock + supervise/status + supervise/death_tally
        # files inside the slot, so the upcoming rmtree doesn't race.
        _s6_run("s6-svscanctl", "-an", str(self.scandir))
        # Give s6-svscan a moment to reap. There's no synchronous
        # "scan completed" handshake — the -a/-n trigger just sets a
        # flag s6-svscan reads on its next loop iteration. 200ms is
        # comfortably above the loop's resolution but well under any
        # user-perceived latency.
        time.sleep(0.2)

        # Now the supervise dir's files are no longer held open by a
        # live s6-supervise, so rmtree can remove them. Files inside
        # supervise/ are root-owned (death_tally, lock, status, written
        # by s6-supervise itself) — but the parent supervise/ directory
        # is hermes-owned (see ``_seed_supervise_skeleton``), and on
        # POSIX you only need write+execute on the parent to remove
        # contained files regardless of file ownership.
        shutil.rmtree(svc_dir, ignore_errors=True)

    def list_profile_gateways(self) -> list[str]:
        """Return the profile names of all currently-registered gateway services."""
        if not self.scandir.exists():
            return []
        return [
            entry.name[len(S6_SERVICE_PREFIX):]
            for entry in self.scandir.iterdir()
            if not entry.name.startswith(".")
            and entry.is_dir()
            and entry.name.startswith(S6_SERVICE_PREFIX)
        ]


_MANAGER_CLASSES: dict[str, type] = {
    "systemd": SystemdServiceManager,
    "launchd": LaunchdServiceManager,
    "windows": WindowsServiceManager,
    "s6": S6ServiceManager,
}
