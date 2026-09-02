"""Container-boot reconciliation of per-profile gateway s6 services.

Wired into the image as /etc/cont-init.d/02-reconcile-profiles by the Dockerfile (Phase 4 Task 4.0).
Runs as root after 01-hermes-setup (the stage2 hook) has chowned the volume and seeded $HERMES_HOME,
but before s6-rc starts user services.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

log = logging.getLogger(__name__)

# Only this desired state triggers automatic restart. Everything else
# (startup_failed, starting, stopped, missing) registers the slot in
# the down state and waits for explicit user action — this avoids the
# crash-loop where a broken gateway keeps being restarted across
# `docker restart` cycles. Older installs only have gateway_state;
# newer lifecycle commands persist desired_state separately so a transient
# runtime state (draining/startup_failed) does not erase the operator's
# durable start/stop intent across pod/container recreation.
_AUTOSTART_STATES = frozenset({"running"})

# Transient runtime sub-states of a RUNNING gateway. A gateway only ever
# reaches these while it is up and serving, so they are NOT an operator stop
# and NOT a failed boot:
#   - `draining`  — written by the drain watcher / scale-to-zero go-dormant
#                   path when an in-flight quiesce begins (gateway/run.py).
#   - `degraded`  — written when the gateway comes up with some platforms
#                   queued for retry, then "falls through to the normal
#                   running state" (gateway/run.py #5196): the process is up,
#                   serving cron + whatever platforms connected, and the
#                   reconnect watcher takes the rest from there.
#
# When a gateway is hard-killed *while in one of these states* (a container/VM
# recreate SIGTERMs it before `_stop_impl` reaches its terminal-state persist),
# the last value left in gateway_state.json is the transient sub-state. With no
# explicit `desired_state` to fall back to, treating that literal value as the
# autostart intent would leave the gateway DOWN on every subsequent boot — the
# gateway never comes back, the dashboard is up but messaging stays dark
# (observed on a relay-opted-in staging instance stranded at `draining`,
# 2026-06; `degraded` is the same wedge class). Map these transient sub-states
# to `running` so a stranded marker reads as the run-intent it actually
# represents. This mirrors gateway/run.py's #42675 handling, which persists
# `running` (not the mid-shutdown `draining`) when an unexpected signal tears
# the gateway down — extended here to the case where the gateway died before it
# could persist anything at all.
#
# `starting` / `startup_failed` are deliberately NOT included: those mean the
# gateway died mid-boot or failed to come up, so auto-restarting them would
# reintroduce the crash-loop the down-marker guard exists to prevent.
_TRANSIENT_RUNNING_STATES = frozenset({"draining", "degraded"})

# Stale runtime files we sweep before recreating service slots. These
# all hold container-namespaced state (PIDs, process tables) that's
# garbage post-restart — a numerically-equal PID in the new container
# is a different process. See the Risk Register in the plan.
_STALE_RUNTIME_FILES = ("gateway.pid", "processes.json")

ReconcileActionLabel = Literal["started", "registered", "skipped"]


@dataclass(frozen=True)
class ReconcileAction:
    """One profile's outcome from a single reconciliation pass."""
    profile: str
    prior_state: str | None
    action: ReconcileActionLabel
    # How the profile's previous gateway life ended: "clean" (exit path ran),
    # "unclean" (sentinel still says running — SIGKILL/OOM/VM death), or
    # "unknown" (no sentinel / never ran). See gateway.lifecycle_ledger
    # (NS-608): at container boot this is the one place that can stamp
    # "the previous container life ended violently" into a durable,
    # volume-persisted log line.
    prior_exit: str = "unknown"


def reconcile_profile_gateways(
    *,
    hermes_home: Path,
    scandir: Path,
    dry_run: bool = False,
    container_argv: Sequence[str] | None = None,
) -> list[ReconcileAction]:
    """Recreate s6 service registrations for every persistent profile.

    Always registers a ``gateway-default`` slot for the root profile (the implicit profile that
    lives at the top of ``$HERMES_HOME``, not under ``profiles/``). The dispatcher in
    ``hermes_cli.gateway`` maps an empty profile suffix to ``gateway-default``, so this slot is what
    ``hermes gateway start`` (no ``-p``) targets.
    """
    actions: list[ReconcileAction] = []

    # A multiplexing root/default gateway owns inbound platform connections
    # for every profile. Named slots must still be registered (so explicit
    # lifecycle management remains available), but booting them from their
    # persisted run intent would create additional multiplex owners.
    # Keep the boot reconciler aligned with the gateway that will own these
    # slots. The runtime resolver gives a recognized environment override
    # precedence over config.yaml and otherwise preserves the configured value.
    from gateway.config import load_gateway_config
    from utils import is_truthy_value

    try:
        multiplex_profiles = load_gateway_config().multiplex_profiles
    except Exception:
        log.warning(
            "Unable to load gateway configuration during container boot; "
            "using the GATEWAY_MULTIPLEX_PROFILES override if set.",
            exc_info=True,
        )
        multiplex_profiles = is_truthy_value(
            os.environ.get("GATEWAY_MULTIPLEX_PROFILES"),
        )

    # Default profile — always register, even if nothing has ever
    # populated the root profile dir. The slot exists so
    # ``hermes gateway start`` (no ``-p``) has somewhere to land;
    # auto-up only when the prior state was "running" (same rule as
    # named profiles). If the container was launched with the legacy
    # `gateway run` command and no state exists yet, seed that intent
    # as `running` so the s6 reconciler preserves the pre-s6 behavior.
    legacy_default_state = _maybe_migrate_legacy_gateway_run_state(
        hermes_home,
        container_argv=container_argv,
        dry_run=dry_run,
    )
    default_prior_state = legacy_default_state or _read_desired_state(hermes_home)
    default_should_start = default_prior_state in _AUTOSTART_STATES
    if not dry_run:
        _cleanup_stale_runtime_files(hermes_home)
        _register_service(scandir, "default", start=default_should_start)
    actions.append(ReconcileAction(
        profile="default",
        prior_state=default_prior_state,
        action="started" if default_should_start else "registered",
        prior_exit=_read_prior_exit_label(hermes_home),
    ))

    profiles_root = hermes_home / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir():
                continue
            # SOUL.md is always seeded by `hermes profile create` (config.yaml
            # is not — that comes later via `hermes setup`). Use it as the
            # "real profile" marker so stray dirs (backups, manual mkdir)
            # aren't picked up.
            if not (entry / "SOUL.md").exists():
                continue
            # The "default" service name is reserved for the root
            # profile (above) — if a user has somehow created a
            # ``profiles/default/`` directory, skip it to avoid the
            # slot collision. Their gateway would still be reachable
            # via ``hermes -p default-named gateway start`` if they
            # rename the directory; we don't try to disambiguate here.
            if entry.name == "default":
                log.warning(
                    "profiles/default/ exists — skipping to avoid colliding "
                    "with the reserved root-profile s6 slot",
                )
                continue

            prior_state = _read_desired_state(entry)
            should_start = (
                not multiplex_profiles and prior_state in _AUTOSTART_STATES
            )

            if not dry_run:
                _cleanup_stale_runtime_files(entry)
                _register_service(scandir, entry.name, start=should_start)

            actions.append(ReconcileAction(
                profile=entry.name,
                prior_state=prior_state,
                action="started" if should_start else "registered",
                prior_exit=_read_prior_exit_label(entry),
            ))

    if not dry_run:
        _write_reconcile_log(hermes_home, actions)
    return actions


def _maybe_migrate_legacy_gateway_run_state(
    hermes_home: Path,
    *,
    container_argv: Sequence[str] | None,
    dry_run: bool,
) -> str | None:
    """Seed root gateway_state for pre-s6 `gateway run` containers.

    The tini image let Docker users run the gateway as the container command (`docker run ...
    gateway run`). After the s6 migration, profile gateways are restored from persisted
    gateway_state.json; a legacy container with no state file would therefore register the default
    service down and never start.
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


def _read_container_argv() -> tuple[str, ...]:
    """Best-effort read of the container's main program argv.

    s6-overlay v2: PID 1 is ``/init`` and its argv holds ``main-wrapper.sh``. v3: PID 1 is
    ``s6-svscan`` and the real command lives on another PID, so after the PID 1 fast path we scan
    ``/proc/*/cmdline`` for a process whose argv contains ``main-wrapper.sh``.
    """
    # Fast path: PID 1 is the command itself (s6-overlay v2 / tini).
    try:
        raw = Path("/proc/1/cmdline").read_bytes()
        argv = tuple(
            part.decode("utf-8", "replace") for part in raw.split(b"\0") if part
        )
        if any("main-wrapper.sh" in part for part in argv):
            return argv
    except OSError:
        pass

    # Slow path: s6-overlay v3 — PID 1 is s6-svscan; find the
    # rc.init-launched process whose argv contains main-wrapper.sh.
    try:
        proc_dir = Path("/proc")
        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            argv = tuple(
                part.decode("utf-8", "replace")
                for part in raw.split(b"\0")
                if part
            )
            if any("main-wrapper.sh" in part for part in argv):
                return argv
    except OSError:
        pass

    return ()


def _strip_container_argv_prefix(argv: Sequence[str]) -> list[str]:
    """Strip the s6/wrapper prefix off the container argv, leaving the hermes args.

    Rather than peel each leading token positionally (which silently breaks the moment s6 changes
    its launcher shape again — exactly what happened in the v2→v3 bump), drop everything up to and
    including the ``main-wrapper.sh`` token: that wrapper path is the stable boundary the image
    owns, and the subcommand always follows it.
    """
    args = list(argv)

    # Preferred boundary: everything through main-wrapper.sh is launcher
    # prefix. Covers s6-overlay v2 (`/init …main-wrapper.sh …`) and v3
    # (`/bin/sh -e …rc.init top …main-wrapper.sh …`) with one rule.
    wrapper_idx = next(
        (i for i, a in enumerate(args) if a.endswith("main-wrapper.sh")),
        None,
    )
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
    """Return True for Docker commands equivalent to `gateway run`."""
    args = _strip_container_argv_prefix(argv)
    if "--no-supervise" in args:
        return False
    return len(args) >= 2 and args[0] == "gateway" and args[1] == "run"


def _is_dashboard_container(argv: Sequence[str]) -> bool:
    """Return True when the container's command is the dashboard.

    A dashboard-only container (``hermes dashboard ...``) never spawns or supervises per-profile
    gateways — that is the gateway container's job.
    """
    args = _strip_container_argv_prefix(argv)
    return bool(args) and args[0] == "dashboard"


def _read_desired_state(profile_dir: Path) -> str | None:
    """Read the persisted gateway desired state for reconciliation.

    Newer state files carry ``desired_state``: operator intent written by s6 lifecycle commands.
    Older files only carry ``gateway_state``; keep that as a compatibility fallback so existing
    running/stopped profiles preserve their behavior until the next explicit start/stop.

    Missing or unparseable files count as "no desired state" so we don't bork the whole
    reconciliation on a corrupt file.
    """
    state_file = profile_dir / "gateway_state.json"
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        desired_state = data.get("desired_state")
        if desired_state is not None:
            return desired_state
        gateway_state = data.get("gateway_state")
        if gateway_state in _TRANSIENT_RUNNING_STATES:
            return "running"
        return gateway_state
    except (OSError, json.JSONDecodeError):
        log.warning(
            "could not read %s; treating as no prior state", state_file,
        )
        return None


def _cleanup_stale_runtime_files(profile_dir: Path) -> None:
    """Remove gateway.pid and processes.json — they reference PIDs in the dead container's process
    namespace and would otherwise confuse the newly-started gateway's process-mismatch checks.
    """
    for name in _STALE_RUNTIME_FILES:
        (profile_dir / name).unlink(missing_ok=True)


def _read_prior_exit_label(profile_dir: Path) -> str:
    """How the profile's previous gateway life ended (clean/unclean/unknown).

    Thin, exception-free wrapper over :func:`gateway.lifecycle_ledger.read_prior_exit_label` — cont-
    init runs in a minimal environment and forensics must never block reconciliation (NS-608).
    """
    try:
        from gateway.lifecycle_ledger import read_prior_exit_label
        return read_prior_exit_label(profile_dir)
    except Exception:
        return "unknown"


def _register_service(scandir: Path, profile: str, *, start: bool) -> None:
    """Recreate the s6 service slot for one profile.

    Mirrors ``S6ServiceManager.register_profile_gateway`` but sets start state via the ``down``
    marker directly: cont-init.d runs as root before s6-svscan scans the dynamic scandir, so the
    manager's ``s6-svscanctl -a`` would fail with no control socket. Built in a sibling temp dir
    and ``Path.replace``d into place so an interrupted write never leaves a half-populated dir.
    """
    import shutil

    from hermes_cli.service_manager import (
        S6ServiceManager,
        _seed_supervise_skeleton,
        validate_profile_name,
    )

    validate_profile_name(profile)
    service_dir = scandir / f"gateway-{profile}"
    # Dot-prefix the staging dir so s6-svscan skips it while half-built
    # (s6-svscan ignores scandir entries whose name starts with ".").
    # A non-dotted ``.tmp`` staging name is supervised AS ROOT by any
    # concurrent ``s6-svscanctl -a`` rescan the moment it has a valid
    # ``type``/``run``, creating a root-owned ``supervise/`` that makes
    # ``_seed_supervise_skeleton`` EACCES — see the matching comment in
    # ``S6ServiceManager.register_profile_gateway``. The atomic
    # ``tmp_dir.replace(service_dir)`` below renames to the dotless live
    # name, so the published slot is unchanged.
    tmp_dir = service_dir.with_name("." + service_dir.name + ".tmp")

    # Wipe any leftover tmp from a previous interrupted run.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    try:
        (tmp_dir / "type").write_text("longrun\n", encoding="utf-8")

        # Reuse the manager's run-script rendering — single source of
        # truth so register_profile_gateway and reconcile_profile_gateways
        # stay consistent. extra_env is empty here; users who need
        # per-profile env can set it via the profile's config.yaml
        # (which the gateway itself loads).
        run = tmp_dir / "run"
        run.write_text(S6ServiceManager._render_run_script(profile, extra_env={}), encoding="utf-8")
        run.chmod(0o755)

        finish = tmp_dir / "finish"
        finish.write_text(S6ServiceManager._render_finish_script(), encoding="utf-8")
        finish.chmod(0o755)

        # Persistent log rotation (OQ8-C).
        log_subdir = tmp_dir / "log"
        log_subdir.mkdir()
        log_run = log_subdir / "run"
        log_run.write_text(S6ServiceManager._render_log_run(profile), encoding="utf-8")
        log_run.chmod(0o755)

        # The presence of a `down` file tells s6-supervise to NOT
        # start the service when s6-svscan picks it up. User brings
        # it up explicitly with `hermes -p <profile> gateway start`
        # (which routes through the Phase 4
        # _dispatch_via_service_manager_if_s6 helper to `s6-svc -u`).
        if not start:
            (tmp_dir / "down").touch()

        # Pre-create the supervise/ skeleton with hermes ownership
        # BEFORE we publish the slot. Mirrors the same pre-creation
        # step in S6ServiceManager.register_profile_gateway — when
        # s6-svscan picks the published slot up, the s6-supervise it
        # spawns will EEXIST our dirs/FIFOs and inherit hermes
        # ownership, so runtime s6-svc / s6-svstat / s6-svwait calls
        # (all dispatched as the hermes user) won't hit EACCES. See
        # ``_seed_supervise_skeleton`` in service_manager.py for the
        # full rationale.
        _seed_supervise_skeleton(tmp_dir)

        # Publish atomically. Path.replace handles the existing-target
        # case the same way os.rename does on POSIX: the target is
        # silently replaced, so a previous reconcile pass's slot is
        # cleanly overwritten in one operation.
        if service_dir.exists():
            shutil.rmtree(service_dir)
        tmp_dir.replace(service_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _write_reconcile_log(
    hermes_home: Path, actions: list[ReconcileAction],
) -> None:
    """Append one line per profile to $HERMES_HOME/logs/container-boot.log.

    Operators inspect this to debug "why didn't my profile come back up". Keeping a separate log
    file (vs. mixing into agent.log) lets troubleshooters grep for "profile=foo" without wading
    through unrelated activity.

    Size-bounded: when the file exceeds ``_LOG_ROTATE_BYTES`` (defaults to 256 KiB ≈ 3000 reconcile
    lines), the current file is renamed to ``container-boot.log.1`` (replacing any previous
    rotation) before the new entries are appended.
    """
    import time
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "container-boot.log"

    # Rotate before opening to append, so the new entries always land
    # in a fresh file when we crossed the threshold last time.
    try:
        if log_path.exists() and log_path.stat().st_size >= _LOG_ROTATE_BYTES:
            log_path.replace(log_dir / "container-boot.log.1")
    except OSError as exc:
        # Rotation failure is non-fatal — keep appending to the
        # existing file rather than losing the entry entirely.
        log.warning("could not rotate %s: %s", log_path, exc)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log_path.open("a", encoding="utf-8") as f:
        for a in actions:
            f.write(
                f"{ts} profile={a.profile} prior_state={a.prior_state} "
                f"action={a.action} prior_exit={a.prior_exit}\n"
            )


# 256 KiB soft cap on container-boot.log; rotated to .1 when crossed.
# At ~80 B per reconcile-action line this is ~3000 lines, or about a
# year of daily reboots on a 5-profile container. Two files = ~512 KiB
# worst case. Tuned for visibility (small enough to grep / cat without
# scrolling forever) more than space (the persistent volume has GB).
_LOG_ROTATE_BYTES = 256 * 1024


def main() -> int:
    """Entry point invoked from /etc/cont-init.d/02-reconcile-profiles."""
    # A dashboard-only container never spawns or supervises per-profile
    # gateways, so reconciling their s6 slots here is pure waste — and
    # actively harmful: when the gateway and dashboard containers share a
    # bind-mounted HERMES_HOME, both race to flock() the same s6-log lock
    # files under logs/gateways/<profile>/lock, producing "Resource busy"
    # failures and a restart storm. Detect the role from PID 1 argv and
    # skip reconciliation in the dashboard container. No operator flag:
    # the role is a fact about the container's command, and a flag can be
    # forgotten in a hand-written manifest, reintroducing the storm.
    if _is_dashboard_container(_read_container_argv()):
        print(
            "reconcile: skipping (dashboard container — does not need "
            "per-profile gateways)"
        )
        return 0

    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    scandir = Path(os.environ.get("S6_PROFILE_GATEWAY_SCANDIR", "/run/service"))
    actions = reconcile_profile_gateways(
        hermes_home=hermes_home, scandir=scandir,
    )
    for a in actions:
        print(
            f"reconcile: profile={a.profile} "
            f"prior_state={a.prior_state} action={a.action}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
