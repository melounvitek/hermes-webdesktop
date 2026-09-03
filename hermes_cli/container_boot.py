"""Container-boot reconciliation of per-profile gateway s6 services.

Wired into the image as /etc/cont-init.d/02-reconcile-profiles. Runs as root after
01-hermes-setup (the stage2 hook) has chowned the volume and seeded $HERMES_HOME, but
before s6-rc starts user services.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

log = logging.getLogger(__name__)

# Only this desired state triggers automatic restart. Everything else (startup_failed,
# starting, stopped, missing) registers the slot down and waits for explicit user action —
# avoiding the crash-loop where a broken gateway keeps restarting across `docker restart`.
# Older installs only have gateway_state; newer lifecycle commands persist desired_state
# separately so a transient runtime state does not erase the operator's durable intent.
_AUTOSTART_STATES = frozenset({"running"})

# Transient runtime sub-states of a RUNNING gateway — reached only while up and serving, so
# NOT an operator stop and NOT a failed boot:
#   - `draining`  — drain watcher / scale-to-zero go-dormant path, in-flight quiesce begun.
#   - `degraded`  — came up with some platforms queued for retry, then fell through to the
#                   normal running state; the reconnect watcher takes it from there.
# When a gateway is hard-killed *in one of these states* (container/VM recreate SIGTERMs it
# before `_stop_impl` persists a terminal state), the marker left in gateway_state.json is
# the transient sub-state. With no `desired_state` to fall back to, treating it literally
# would leave the gateway DOWN on every subsequent boot (observed: a relay-opted-in staging
# instance stranded at `draining`; `degraded` is the same wedge class). Map them to `running`
# — mirrors gateway/run.py persisting `running` (not mid-shutdown `draining`) on unexpected
# signal, extended to the case where the gateway died before persisting anything.
# `starting` / `startup_failed` are deliberately NOT included: those mean the gateway died
# mid-boot or failed to come up, and auto-restarting would reintroduce the crash-loop.
_TRANSIENT_RUNNING_STATES = frozenset({"draining", "degraded"})

# Stale runtime files swept before recreating service slots: they hold container-namespaced
# state (PIDs, process tables) that's garbage post-restart — a numerically-equal PID in the
# new container is a different process.
_STALE_RUNTIME_FILES = ("gateway.pid", "processes.json")

ReconcileActionLabel = Literal["started", "registered", "skipped"]


@dataclass(frozen=True)
class ReconcileAction:
    """One profile's outcome from a single reconciliation pass."""
    profile: str
    prior_state: str | None
    action: ReconcileActionLabel
    # How the previous gateway life ended: "clean" (exit path ran), "unclean" (sentinel still
    # says running — SIGKILL/OOM/VM death), or "unknown" (no sentinel / never ran). Container
    # boot is the one place that can stamp "the previous life ended violently" into a
    # durable, volume-persisted log line (gateway.lifecycle_ledger).
    prior_exit: str = "unknown"


def _slot_action(
    profile: str, profile_dir: Path, prior_state: str | None, start: bool,
) -> ReconcileAction:
    return ReconcileAction(profile=profile, prior_state=prior_state,
                           action="started" if start else "registered",
                           prior_exit=_read_prior_exit_label(profile_dir))


def reconcile_profile_gateways(
    *, hermes_home: Path, scandir: Path, dry_run: bool = False,
    container_argv: Sequence[str] | None = None,
) -> list[ReconcileAction]:
    """Recreate s6 service registrations for every persistent profile.

    Always registers a ``gateway-default`` slot for the root profile (the implicit profile at
    the top of ``$HERMES_HOME``, not under ``profiles/``): the dispatcher in
    ``hermes_cli.gateway`` maps an empty profile suffix to it, so it is what
    ``hermes gateway start`` (no ``-p``) targets.
    """
    actions: list[ReconcileAction] = []

    # A multiplexing root/default gateway owns inbound platform connections for every
    # profile. Named slots must still be registered (explicit lifecycle management stays
    # available), but booting them from their persisted run intent would create additional
    # multiplex owners. The runtime resolver gives a recognized environment override
    # precedence over config.yaml and otherwise preserves the configured value.
    from gateway.config import load_gateway_config
    from utils import is_truthy_value
    try:
        multiplex_profiles = load_gateway_config().multiplex_profiles
    except Exception:
        log.warning("Unable to load gateway configuration during container boot; using the "
                    "GATEWAY_MULTIPLEX_PROFILES override if set.", exc_info=True)
        multiplex_profiles = is_truthy_value(os.environ.get("GATEWAY_MULTIPLEX_PROFILES"))

    # Default profile — always register, even if nothing has ever populated the root profile
    # dir; auto-up only when the prior state was "running" (same rule as named profiles). A
    # legacy `gateway run` container with no state yet seeds that intent as `running` so the
    # s6 reconciler preserves the pre-s6 behavior.
    legacy_default_state = _maybe_migrate_legacy_gateway_run_state(
        hermes_home, container_argv=container_argv, dry_run=dry_run)
    default_prior_state = legacy_default_state or _read_desired_state(hermes_home)
    default_should_start = default_prior_state in _AUTOSTART_STATES
    if not dry_run:
        _cleanup_stale_runtime_files(hermes_home)
        _register_service(scandir, "default", start=default_should_start)
    actions.append(_slot_action("default", hermes_home, default_prior_state, default_should_start))

    profiles_root = hermes_home / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            # SOUL.md is always seeded by `hermes profile create` (config.yaml is not — that
            # comes later via `hermes setup`): the "real profile" marker so stray dirs
            # (backups, manual mkdir) aren't picked up.
            if not entry.is_dir() or not (entry / "SOUL.md").exists():
                continue
            # "default" is reserved for the root profile (above); skip a stray
            # ``profiles/default/`` rather than collide on the slot.
            if entry.name == "default":
                log.warning("profiles/default/ exists — skipping to avoid colliding with the "
                            "reserved root-profile s6 slot")
                continue

            prior_state = _read_desired_state(entry)
            should_start = not multiplex_profiles and prior_state in _AUTOSTART_STATES

            if not dry_run:
                _cleanup_stale_runtime_files(entry)
                _register_service(scandir, entry.name, start=should_start)

            actions.append(_slot_action(entry.name, entry, prior_state, should_start))

    if not dry_run:
        _write_reconcile_log(hermes_home, actions)
    return actions


def _maybe_migrate_legacy_gateway_run_state(
    hermes_home: Path, *, container_argv: Sequence[str] | None, dry_run: bool
) -> str | None:
    """Seed root gateway_state for pre-s6 `gateway run` containers.

    The tini image let Docker users run the gateway as the container command. After the s6
    migration profile gateways are restored from persisted gateway_state.json, so a legacy
    container with no state file would register the default service down and never start.
    """
    state_file = hermes_home / "gateway_state.json"
    if state_file.exists():
        return None

    if os.environ.get("HERMES_GATEWAY_NO_SUPERVISE", "").lower() in ("1", "true", "yes"):
        return None

    argv = tuple(container_argv) if container_argv is not None else _read_container_argv()
    if not _is_legacy_gateway_run_request(argv):
        return None

    if not dry_run:
        import time
        state_file.write_text(json.dumps({
            "gateway_state": "running",
            "desired_state": "running",
            "timestamp": int(time.time()),
            "migrated_from": "legacy-container-cmd",
        }) + "\n", encoding="utf-8")
    return "running"


def _cmdline_argv(cmdline: Path) -> tuple[str, ...]:
    raw = cmdline.read_bytes()
    return tuple(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def _read_container_argv() -> tuple[str, ...]:
    """Best-effort read of the container's main program argv.

    s6-overlay v2: PID 1 is ``/init`` and its argv holds ``main-wrapper.sh``. v3: PID 1 is
    ``s6-svscan`` and the real command lives on another PID, so after the PID 1 fast path we
    scan ``/proc/*/cmdline`` for a process whose argv contains ``main-wrapper.sh``.
    """
    try:
        argv = _cmdline_argv(Path("/proc/1/cmdline"))
        if any("main-wrapper.sh" in part for part in argv):
            return argv
    except OSError:
        pass

    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = _cmdline_argv(entry / "cmdline")
            except OSError:
                continue
            if any("main-wrapper.sh" in part for part in argv):
                return argv
    except OSError:
        pass

    return ()


def _strip_container_argv_prefix(argv: Sequence[str]) -> list[str]:
    """Strip the s6/wrapper prefix off the container argv, leaving the hermes args.

    Drop everything through the ``main-wrapper.sh`` token rather than peel leading tokens
    positionally (which broke on the s6 v2→v3 launcher-shape change): the wrapper path is the
    stable boundary the image owns, and the subcommand always follows it.
    """
    args = list(argv)

    wrapper_idx = next((i for i, a in enumerate(args) if a.endswith("main-wrapper.sh")), None)
    if wrapper_idx is not None:
        args = args[wrapper_idx + 1 :]
    elif args and Path(args[0]).name == "init":
        # Defensive: an `init` prefix with no wrapper token in argv.
        args = args[1:]

    # Non-PID-1 entrypoints go through the dispatch shim instead of /init.
    if args and args[0].endswith("entrypoint-dispatch.sh"):
        args = args[1:]

    # The wrapper re-execs `hermes <subcommand>`; peel an explicit hermes.
    if args and Path(args[0]).name == "hermes":
        args = args[1:]
    return args


def _is_legacy_gateway_run_request(argv: Sequence[str]) -> bool:
    """True for Docker commands equivalent to `gateway run`."""
    args = _strip_container_argv_prefix(argv)
    if "--no-supervise" in args:
        return False
    return len(args) >= 2 and args[0] == "gateway" and args[1] == "run"


def _is_dashboard_container(argv: Sequence[str]) -> bool:
    """True when the container's command is the dashboard (which never supervises gateways)."""
    args = _strip_container_argv_prefix(argv)
    return bool(args) and args[0] == "dashboard"


def _read_desired_state(profile_dir: Path) -> str | None:
    """Persisted gateway desired state for reconciliation.

    Newer state files carry ``desired_state`` (operator intent from s6 lifecycle commands);
    older ones only ``gateway_state``, kept as a fallback so existing profiles preserve their
    behavior until the next explicit start/stop. Missing/unparseable files count as "no
    desired state" so a corrupt file can't bork the whole reconciliation.
    """
    state_file = profile_dir / "gateway_state.json"
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read %s; treating as no prior state", state_file)
        return None
    desired_state = data.get("desired_state")
    if desired_state is not None:
        return desired_state
    gateway_state = data.get("gateway_state")
    if gateway_state in _TRANSIENT_RUNNING_STATES:
        return "running"
    return gateway_state


def _cleanup_stale_runtime_files(profile_dir: Path) -> None:
    """Remove PID-namespace-bound runtime files that would confuse the new gateway's checks."""
    for name in _STALE_RUNTIME_FILES:
        (profile_dir / name).unlink(missing_ok=True)


def _read_prior_exit_label(profile_dir: Path) -> str:
    """Exception-free ``lifecycle_ledger.read_prior_exit_label`` — forensics never block boot."""
    try:
        from gateway.lifecycle_ledger import read_prior_exit_label
        return read_prior_exit_label(profile_dir)
    except Exception:
        return "unknown"


def _write_exec(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _register_service(scandir: Path, profile: str, *, start: bool) -> None:
    """Recreate the s6 service slot for one profile.

    Mirrors ``S6ServiceManager.register_profile_gateway`` but sets start state via the
    ``down`` marker directly: cont-init.d runs as root before s6-svscan scans the dynamic
    scandir, so the manager's ``s6-svscanctl -a`` would fail with no control socket. Built in
    a sibling temp dir and ``Path.replace``d into place so an interrupted write never leaves
    a half-populated dir.
    """
    import shutil

    from hermes_cli.service_manager import (
        S6ServiceManager, _seed_supervise_skeleton, validate_profile_name,
    )

    validate_profile_name(profile)
    service_dir = scandir / f"gateway-{profile}"
    # Dot-prefix the staging dir so s6-svscan skips it while half-built. A non-dotted staging
    # name is supervised AS ROOT by any concurrent ``s6-svscanctl -a`` rescan the moment it
    # has a valid ``type``/``run``, creating a root-owned ``supervise/`` that makes
    # ``_seed_supervise_skeleton`` EACCES (see ``S6ServiceManager.register_profile_gateway``).
    tmp_dir = service_dir.with_name("." + service_dir.name + ".tmp")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    try:
        (tmp_dir / "type").write_text("longrun\n", encoding="utf-8")

        # Reuse the manager's script rendering — single source of truth so both registration
        # paths stay consistent. extra_env is empty here; per-profile env comes from the
        # profile's config.yaml (which the gateway itself loads).
        _write_exec(tmp_dir / "run", S6ServiceManager._render_run_script(profile, extra_env={}))
        _write_exec(tmp_dir / "finish", S6ServiceManager._render_finish_script())

        # Persistent log rotation.
        (tmp_dir / "log").mkdir()
        _write_exec(tmp_dir / "log" / "run", S6ServiceManager._render_log_run(profile))

        # A `down` file tells s6-supervise NOT to start the service when s6-svscan picks it
        # up; the user brings it up with `hermes -p <profile> gateway start` (→ `s6-svc -u`).
        if not start:
            (tmp_dir / "down").touch()

        # Pre-create supervise/ with hermes ownership BEFORE publishing: the s6-supervise
        # spawned on pickup will EEXIST our dirs/FIFOs and inherit that ownership, so runtime
        # s6-svc / s6-svstat / s6-svwait calls (dispatched as the hermes user) won't EACCES.
        _seed_supervise_skeleton(tmp_dir)

        # Publish atomically: Path.replace silently replaces an existing target on POSIX, so a
        # previous reconcile pass's slot is overwritten in one operation.
        if service_dir.exists():
            shutil.rmtree(service_dir)
        tmp_dir.replace(service_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# 256 KiB soft cap on container-boot.log, rotated to .1 when crossed (~3000 lines at ~80 B
# each — about a year of daily reboots on a 5-profile container). Tuned for grep-ability
# more than space (the persistent volume has GB).
_LOG_ROTATE_BYTES = 256 * 1024


def _write_reconcile_log(hermes_home: Path, actions: list[ReconcileAction]) -> None:
    """Append one line per profile to $HERMES_HOME/logs/container-boot.log.

    A separate file (vs. agent.log) lets operators debugging "why didn't my profile come back
    up" grep for "profile=foo" without unrelated activity. Rotated to ``.1`` (replacing any
    previous rotation) before appending once it exceeds ``_LOG_ROTATE_BYTES``.
    """
    import time
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "container-boot.log"

    try:
        if log_path.exists() and log_path.stat().st_size >= _LOG_ROTATE_BYTES:
            log_path.replace(log_dir / "container-boot.log.1")
    except OSError as exc:
        # Non-fatal — keep appending rather than lose the entry entirely.
        log.warning("could not rotate %s: %s", log_path, exc)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log_path.open("a", encoding="utf-8") as f:
        for a in actions:
            f.write(
                f"{ts} profile={a.profile} prior_state={a.prior_state} "
                f"action={a.action} prior_exit={a.prior_exit}\n"
            )


def main() -> int:
    """Entry point invoked from /etc/cont-init.d/02-reconcile-profiles."""
    # A dashboard-only container never supervises per-profile gateways, and reconciling here
    # is actively harmful: when gateway and dashboard containers share a bind-mounted
    # HERMES_HOME, both race to flock() the same s6-log lock files under
    # logs/gateways/<profile>/lock → "Resource busy" restart storm. The role is detected from
    # PID 1 argv, not an operator flag — a flag can be forgotten in a hand-written manifest.
    if _is_dashboard_container(_read_container_argv()):
        print("reconcile: skipping (dashboard container — does not need per-profile gateways)")
        return 0

    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    scandir = Path(os.environ.get("S6_PROFILE_GATEWAY_SCANDIR", "/run/service"))
    actions = reconcile_profile_gateways(hermes_home=hermes_home, scandir=scandir)
    for a in actions:
        print(f"reconcile: profile={a.profile} prior_state={a.prior_state} action={a.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
