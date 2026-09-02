"""Runtime inventory + update plan for the fleet-update pipeline (#91277 Phase 2).

One read-only pass that answers, BEFORE any mutation: what Hermes runtimes are running on this
machine, how is each one deployed, which of them will this update touch, and how will each be
restarted?

The module is deliberately side-effect free — every collector is a probe over primitives that
already exist (`find_profile_gateway_processes`, `_get_service_pids`, `gateway_state.json` code
stamps from #91283, `detect_install_method`) — so `hermes update --plan` can run on a live fleet
with zero risk, and the update receipt can embed the inventory without changing update behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeRecord:
    """One running (or expected) Hermes runtime on this machine."""

    kind: str                     # gateway | dashboard | serve
    profile: str                  # profile name ("default", ...)
    pid: Optional[int] = None     # live PID when known
    supervisor: str = "manual"    # systemd | launchd | desktop | manual
    code_sha: Optional[str] = None       # stamped running-code sha (#91283)
    code_version: Optional[str] = None
    restart_via: str = ""         # human-readable restart mechanism
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdatePlan:
    """The full pre-update picture: install shape + runtimes + actions."""

    install_method: str = "unknown"       # git | docker | nix | apt | ...
    updatable_in_place: bool = True
    update_mechanism: str = "hermes update"
    expected_sha: Optional[str] = None    # current checkout HEAD (pre-pull)
    expected_version: Optional[str] = None
    profiles: list = field(default_factory=list)
    runtimes: list = field(default_factory=list)  # list[RuntimeRecord]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # recursive: RuntimeRecord entries become dicts, dict entries are copied


def _detect_supervisor_for_pid(pid: int, service_pids: set, windows_service_pids: set | None = None) -> str:
    """Classify how a live gateway PID is supervised."""
    if windows_service_pids and pid in windows_service_pids:
        # SCM-supervised Windows gateway (WinSW/NSSM/sc.exe create): the
        # update pause machinery stops the SERVICE via sc.exe instead of
        # killing the child, so #91277 Phase 2 reconciliation must plan it
        # under its own mechanism id, not "manual".
        return "windows-service"
    if pid in service_pids:
        try:
            from hermes_cli.gateway import is_macos, supports_systemd_services

            if supports_systemd_services():
                return "systemd"
            if is_macos():
                return "launchd"
        except Exception:
            pass
        return "service"
    return "manual"


_RESTART_MECHANISMS = {
    "systemd": "systemd",
    "launchd": "launchd",
    "desktop": "desktop",
    "windows-service": "windows-service",
    "manual-serve": "respawn-argv",
}

_MECHANISM_DESCRIPTIONS = {
    "systemd": "systemctl restart (drain-first SIGUSR1 when supported)",
    "launchd": "launchctl kickstart -k (drain-first, per-label domain)",
    "desktop": "Desktop app respawns its serve backend",
    "windows-service": "sc.exe stop before venv mutation, sc.exe start after update",
    "respawn-argv": "stop before code swap, relaunch with recorded launch args",
}


def _restart_mechanism(supervisor: str, profile: str) -> str:
    """Machine-readable restart mechanism id for a runtime.

    THE policy table (#91277 Phase 2): restart execution consumes these ids via
    :func:`match_runtime_outcomes` / the update's restart phase, and the receipt records per-runtime
    outcomes against them. Display strings are derived by :func:`describe_restart_mechanism` — never
    the other way around.
    """
    return _RESTART_MECHANISMS.get(supervisor, "manual")


def describe_restart_mechanism(mechanism: str, profile: str) -> str:
    """Human-readable description of a restart mechanism id."""
    described = _MECHANISM_DESCRIPTIONS.get(mechanism)
    if described is not None:
        return described
    if profile != "default":
        return f"hermes -p {profile} gateway restart"
    return "hermes gateway restart"


def _runtime(kind: str, profile: str, pid: Optional[int], supervisor: str, **extra: Any) -> RuntimeRecord:
    """A :class:`RuntimeRecord` with ``restart_via`` derived from its supervisor."""
    return RuntimeRecord(
        kind=kind,
        profile=profile,
        pid=pid,
        supervisor=supervisor,
        restart_via=_restart_mechanism(supervisor, profile),
        **extra,
    )


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gateway_record(
    profile: str,
    pid: int,
    supervisor: str,
    code_sha: Any = None,
    code_version: Any = None,
) -> RuntimeRecord:
    return _runtime(
        "gateway",
        profile,
        pid,
        supervisor,
        code_sha=str(code_sha) if code_sha else None,
        code_version=code_version,
    )


@contextmanager
def _probe(label: str):
    """Run one inventory collector; a failure is logged at debug and yields fewer rows, never an exception."""
    try:
        yield
    except Exception as exc:
        logger.debug("%s failed: %s", label, exc)


def collect_runtime_inventory() -> UpdatePlan:
    """Build the pre-update plan. Read-only; never raises.

    Every collector degrades independently — a probe failure yields fewer rows, not an exception.
    The result is embeddable in the update receipt and printable via :func:`print_update_plan`.
    """
    plan = UpdatePlan()

    # --- install shape / deployment kind ---------------------------------
    with _probe("Install-method probe"):
        from hermes_cli.config import (
            detect_install_method,
            get_managed_system,
            recommended_update_command_for_method,
        )

        method = detect_install_method()
        plan.install_method = method
        managed = get_managed_system()
        if managed:
            plan.install_method = managed
        plan.updatable_in_place = method in ("git", "unknown") and not managed
        # Baked image provenance (#91277 Phase 3): when the image marker is
        # present it is authoritative — a bind-mounted checkout inside a
        # container can look like `git` to the heuristics while the running
        # filesystem is actually an immutable image. Fail-closed: an invalid
        # marker still flips the plan to not-updatable.
        with _probe("Image provenance probe"):
            from hermes_cli.image_provenance import read_image_provenance

            provenance = read_image_provenance()
            if provenance is not None:
                plan.updatable_in_place = False
                if provenance.valid and provenance.manager:
                    plan.install_method = provenance.manager
        plan.update_mechanism = recommended_update_command_for_method(method)

    # --- expected code identity (pre-pull) --------------------------------
    with _probe("Code-identity probe"):
        from hermes_cli.build_info import get_code_identity

        identity = get_code_identity(refresh=True)
        plan.expected_sha = identity.get("sha")
        plan.expected_version = identity.get("version")

    # --- profiles ----------------------------------------------------------
    profile_homes: list[tuple[str, Path]] = []
    with _probe("Profile enumeration"):
        from hermes_cli.profiles import (
            _get_default_hermes_home,
            _get_profiles_root,
            _PROFILE_ID_RE,
        )

        default_home = _get_default_hermes_home()
        if default_home.is_dir():
            profile_homes.append(("default", default_home))
        root = _get_profiles_root()
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if entry.is_dir() and entry.name != "default" and _PROFILE_ID_RE.match(entry.name):
                    profile_homes.append((entry.name, entry))
        plan.profiles = [name for name, _ in profile_homes]

    # --- service-managed PIDs (fleet-wide) ---------------------------------
    service_pids: set = set()
    with _probe("Service-PID probe"):
        from hermes_cli.gateway import _get_service_pids

        service_pids = _get_service_pids(all_profiles=True) or set()

    # --- SCM-supervised gateway PIDs (Windows) ------------------------------
    # find_windows_gateway_services() maps validated gateway PIDs through
    # process ancestry to running SCM service PIDs (no-op off Windows). The
    # update's pause phase stops these via `sc.exe stop` / restarts via
    # `sc.exe start`, so the plan must carry the matching mechanism id for
    # the #91277 Phase 2 reconciliation and the fleet check.
    windows_service_pids: set = set()
    with _probe("Windows SCM service-ownership probe"):
        from hermes_cli.gateway import find_windows_gateway_services

        windows_service_pids = {int(service.gateway_pid) for service in find_windows_gateway_services()}

    # --- per-profile gateways (PID files + runtime status stamps) ----------
    seen_pids: set[int] = set()
    with _probe("Gateway-state inventory"):
        from gateway.status import _pid_exists, read_runtime_status

        for profile, home in profile_homes:
            # Prefer the gateway-owned control socket (#92091): identity
            # declared by the process itself, including its own supervisor
            # provenance — no argv/PID inference. Scan fallback below.
            identity = None
            try:
                from gateway.control_socket import identify_gateway

                identity = identify_gateway(home)
            except Exception:
                identity = None
            sock_pid = _int_or_none(identity.get("pid")) if identity else None
            if sock_pid is not None:
                if sock_pid in seen_pids:
                    # One multiplex gateway can answer identify for
                    # several profile homes — one runtime record per
                    # process, not per home.
                    continue
                seen_pids.add(sock_pid)
                declared = identity.get("supervisor")
                supervisor = (
                    str(declared)
                    if declared
                    else _detect_supervisor_for_pid(sock_pid, service_pids, windows_service_pids)
                )
                plan.runtimes.append(
                    _gateway_record(
                        profile, sock_pid, supervisor, identity.get("code_sha"), identity.get("code_version")
                    )
                )
                continue
            record = read_runtime_status(home / "gateway_state.json") or {}
            pid = _int_or_none(record.get("pid"))
            if pid is None or not _pid_exists(pid):
                continue
            seen_pids.add(pid)
            plan.runtimes.append(
                _gateway_record(
                    profile,
                    pid,
                    _detect_supervisor_for_pid(pid, service_pids, windows_service_pids),
                    record.get("code_sha"),
                    record.get("code_version"),
                )
            )

    # PID-file mapped gateways not covered by a runtime-status record
    with _probe("PID-file gateway inventory"):
        from hermes_cli.gateway import find_profile_gateway_processes

        for proc in find_profile_gateway_processes():
            if proc.pid in seen_pids:
                continue
            seen_pids.add(proc.pid)
            plan.runtimes.append(
                _gateway_record(
                    proc.profile, proc.pid, _detect_supervisor_for_pid(proc.pid, service_pids, windows_service_pids)
                )
            )

    # Serve/dashboard backends from the spawn ledger (#63206). These are the
    # runtimes the gateway collectors above can never see: a manually
    # launched `hermes serve --host <ip>` for a remote Desktop, or a
    # long-lived `hermes dashboard`. Every serve/dashboard registers itself
    # (with structured host/port/profile since #63206) at startup, and
    # ledger_entries() live-verifies (pid, create_time) so PID reuse never
    # fabricates a row. Desktop-supervised backends are classified by their
    # recorded spawner still being alive — those restart via the Desktop's
    # own respawn, not ours.
    with _probe("Serve/dashboard ledger inventory"):
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            purpose = entry.get("purpose")
            if purpose not in ("serve", "dashboard"):
                continue
            pid = entry.get("pid")
            if not isinstance(pid, int) or pid in seen_pids:
                continue
            seen_pids.add(pid)
            plan.runtimes.append(
                _runtime(
                    str(purpose),
                    str(entry.get("profile") or "default"),
                    pid,
                    "desktop" if spawner_is_dead(entry) is False else "manual-serve",
                    detail={
                        "argv": entry.get("argv") or "",
                        "host": entry.get("host") or "",
                        "port": entry.get("port"),
                        # Process incarnation, not just the numeric PID: a
                        # post-update survivor probe that compares PIDs alone
                        # calls a NEW serve that reused the number a survivor
                        # (#92145 review).
                        "create_time": entry.get("create_time"),
                    },
                )
            )

    return plan


def print_update_plan(plan: UpdatePlan) -> None:
    """Human-readable plan — what the update will touch and how."""
    print("Update plan:")
    print(f"  Install: {plan.install_method}", end="")
    if plan.expected_version:
        print(f" (v{plan.expected_version}", end="")
        if plan.expected_sha:
            print(f" @ {plan.expected_sha[:8]}", end="")
        print(")", end="")
    print()
    if not plan.updatable_in_place:
        print("  ⚠ This install is NOT updatable in place.")
        print(f"    Update via: {plan.update_mechanism}")
    profiles = ", ".join(plan.profiles) if plan.profiles else "(none found)"
    print(f"  Profiles: {profiles}")
    if not plan.runtimes:
        print("  Running Hermes services: none detected — code swap only.")
        return
    print(f"  Running services to restart ({len(plan.runtimes)}):")
    for runtime in plan.runtimes:
        sha = f" @ {runtime.code_sha[:8]}" if runtime.code_sha else ""
        print(f"    • {runtime.kind} [{runtime.profile}] pid {runtime.pid} — {runtime.supervisor}{sha}")
        print(f"      restart: {describe_restart_mechanism(runtime.restart_via, runtime.profile)}")


_SERVE_KINDS = ("serve", "dashboard")


def _serve_unit_matches_profile(profile: str, unit: object) -> bool:
    """Does *unit* name a ``hermes-serve*``/``hermes-dashboard*`` unit for *profile*?

    Serve/dashboard runtimes have their OWN unit vocabulary; the gateway's ``hermes-gateway*`` names
    never cover them (#100479).
    """
    name = str(unit).removesuffix(".service")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if profile == "default":
        return name in {"hermes-serve", "hermes-dashboard"}
    return name in {f"hermes-serve-{profile}", f"hermes-dashboard-{profile}"}


def _serve_runtime_outcome(
    r: RuntimeRecord,
    *,
    killed: set,
    failed_set: set,
    restarted_set: set,
    stale_serves: "set | None",
) -> str:
    """Outcome for one serve/dashboard runtime — never the gateway's."""
    if r.pid is not None and r.pid in killed:
        return "stopped"
    if any(_serve_unit_matches_profile(r.profile, u) for u in failed_set):
        return "failed"
    if stale_serves is not None:
        # Incarnation-verified: the pre-update process is gone (replaced by
        # its unit / the dashboard cleanup respawn / the Desktop app) or it
        # is still alive on pre-update code.
        return "unaccounted" if r.pid in stale_serves else "restarted"
    if any(_serve_unit_matches_profile(r.profile, s) for s in restarted_set):
        return "restarted"
    return "unaccounted"


def match_runtime_outcomes(
    plan: "UpdatePlan",
    *,
    restarted_services: list,
    relaunched_profiles: list,
    externally_supervised_profiles: list,
    killed_pids: set,
    failed_units: list,
    stale_serve_pids: "set | None" = None,
) -> list[dict[str, Any]]:
    """Reconcile the plan's runtimes against what the restart phase DID.

    The platform restart branches each re-discover their own targets, so a runtime the plan saw can
    be missed with no signal. Returns one ``{kind, profile, pid, mechanism, outcome}`` row per
    planned runtime; outcome is ``restarted``, ``stopped``, ``failed`` or ``unaccounted`` (no
    bookkeeping mentions it — the blind-spot tripwire). Never raises. Serve/dashboard runtimes are
    reconciled in their OWN vocabulary and never borrow the gateway's outcome: with
    ``stale_serve_pids`` a pre-update serve whose incarnation is gone counts as ``restarted``, one
    still alive is ``unaccounted``; without the probe an untouched serve stays ``unaccounted``.
    """
    outcomes: list[dict[str, Any]] = []
    try:
        failed_set = {str(u) for u in (failed_units or [])}
        restarted_set = {str(s) for s in (restarted_services or [])}
        relaunched = set(relaunched_profiles or [])
        external = set(externally_supervised_profiles or [])
        killed = {int(p) for p in (killed_pids or set())}
        stale_serves = {int(p) for p in stale_serve_pids} if stale_serve_pids is not None else None

        def _gateway_names(r: RuntimeRecord, names: set) -> bool:
            # The bare "hermes-gateway" unit name is gateway-specific: a
            # serve/dashboard runtime that merely shares the default
            # profile is a different process the gateway restart never
            # touched, and must not borrow its outcome (#100479).
            return any(
                r.profile in name
                or (
                    r.kind == "gateway"
                    and r.profile == "default"
                    and "hermes-gateway" in name
                )
                for name in names
            )

        for r in plan.runtimes:
            if not isinstance(r, RuntimeRecord):
                continue
            if r.kind in _SERVE_KINDS:
                outcome = _serve_runtime_outcome(
                    r,
                    killed=killed,
                    failed_set=failed_set,
                    restarted_set=restarted_set,
                    stale_serves=stale_serves,
                )
            elif r.profile in relaunched or r.profile in external:
                outcome = "restarted"
            elif r.pid is not None and r.pid in killed:
                outcome = "stopped"
            elif _gateway_names(r, failed_set):
                outcome = "failed"
            elif _gateway_names(r, restarted_set):
                outcome = "restarted"
            else:
                outcome = "unaccounted"
            outcomes.append({
                "kind": r.kind,
                "profile": r.profile,
                "pid": r.pid,
                "mechanism": r.restart_via,
                "outcome": outcome,
            })
    except Exception as exc:
        logger.debug("Runtime-outcome reconciliation failed: %s", exc)
    return outcomes


def report_unaccounted_runtimes(outcomes: list[dict[str, Any]]) -> bool:
    """Print a loud warning for runtimes the restart phase never touched.

    Returns True when at least one planned runtime is unaccounted; the caller escalates like a
    STALE/DOWN fleet row (exit 1) — a promised restart silently missed is the class this phase
    exists to kill.
    """
    missed = [o for o in outcomes if o.get("outcome") == "unaccounted"]
    if not missed:
        return False
    print()
    print("  ⚠ Planned runtimes the restart phase never touched:")
    for o in missed:
        print(f"    ✗ {o['kind']} [{o['profile']}] pid {o['pid']} — planned mechanism: {o['mechanism']}")
    print("    Restart them manually, then verify:")
    if any(o.get("kind") not in _SERVE_KINDS for o in missed):
        print("      hermes gateway restart                # active profile")
        print("      hermes -p <profile> gateway restart   # named profile")
    if any(o.get("kind") in _SERVE_KINDS for o in missed):
        # A serve/dashboard is not reachable by any `gateway restart`
        # command (#100479): name the process, not the wrong verb.
        print("      systemctl --user restart hermes-serve.service   # unit-managed serve")
        print("      relaunch `hermes serve` / `hermes dashboard` / the Desktop app")
    return True


def record_plan_in_receipt(plan: UpdatePlan) -> None:
    """Attach the inventory to the active update receipt. Never raises."""
    try:
        import hermes_cli.update_receipt as ur

        if ur._current is not None:
            ur._current.data["plan"] = plan.to_dict()
    except Exception as exc:
        logger.debug("Could not record plan in receipt: %s", exc)
