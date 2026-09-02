"""Gateway fleet restart + post-update verification for ``hermes update``: systemd/launchd unit restarts, pending-restart marker, fleet probe.

Split out of ``hermes_cli/update_cmd.py``; every moved name is re-imported there, so
``hermes_cli.update_cmd.<name>`` keeps resolving (and monkeypatching) as before.
Origin-internal helpers are imported lazily inside each function (no import cycle;
test patches on ``hermes_cli.update_cmd.<name>`` stay effective).
"""

import logging
import os
import subprocess
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


def _write_gateway_update_exit_code(ok: bool) -> None:
    from hermes_cli.update_cmd import get_hermes_home
    path = get_hermes_home() / ".update_exit_code"
    try:
        path.write_text("0" if ok else "1", encoding="utf-8")
    except OSError:
        pass


# Lives under HERMES_HOME (not next to the venv). Unlike the venv-repair
# markers, this records the fleet-restart obligation after a pull advanced
# HEAD (#95294); cleared only when the restart completes or nothing was running.
_FLEET_RESTART_PENDING_NAME = "fleet_restart_pending"


def _fleet_restart_pending_marker_path() -> Path:
    """HERMES_HOME breadcrumb for a pull that has not yet restarted the fleet."""
    from hermes_cli.update_cmd import get_hermes_home
    return get_hermes_home() / _FLEET_RESTART_PENDING_NAME


def _write_fleet_restart_pending_marker(*, expected_sha: str = "") -> None:
    """Drop the pull→restart obligation breadcrumb. Never raises."""
    from hermes_cli.update_cmd import _m
    path = _fleet_restart_pending_marker_path()
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping fleet-restart-pending marker under pytest (live checkout)")
        return
    try:
        lines = [f"started={_time.time()}", f"pid={os.getpid()}"]
        if expected_sha:
            lines.append(f"expected_sha={expected_sha}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write fleet-restart-pending marker: %s", exc)


def _clear_fleet_restart_pending_marker() -> None:
    """Remove the pull→restart obligation breadcrumb. Never raises."""
    from hermes_cli.update_cmd import _m
    _m()._clear_marker_file(
        _fleet_restart_pending_marker_path(), label="fleet-restart-pending"
    )


def _current_checkout_sha() -> str | None:
    """Current on-disk checkout HEAD, or None if it cannot be resolved."""
    from hermes_cli.update_cmd import _capture_head_sha, _m
    try:
        from hermes_cli.build_info import get_code_identity

        sha = (get_code_identity(refresh=True) or {}).get("sha")
        return str(sha) if sha else None
    except Exception:
        return _capture_head_sha(["git"], _m().PROJECT_ROOT)


def _receipt_looks_unfinished(receipt: dict) -> bool:
    """True when *receipt* is from an update that did not finish cleanly."""
    if receipt.get("stop_reason"):
        return True
    exit_code = receipt.get("exit_code")
    if exit_code not in (0, None):
        return True
    outcome = receipt.get("outcome")
    if outcome in ("failed", "partial", "running"):
        return True
    gateway_restart = receipt.get("gateway_restart")
    if isinstance(gateway_restart, dict) and gateway_restart.get("incomplete"):
        return True
    return False


def _receipt_reports_stale_runtime(expected_sha: str | None = None) -> bool:
    """True when ``update_receipts/latest.json`` records a runtime SHA skew.

    Prefer the post-restart ``fleet`` matrix. ``plan.runtimes[].code_sha`` is
    captured *before* the pull, so a finished update's plan always shows stale
    SHAs and must not retrigger a restart; consult it only for an unfinished
    receipt (#95294).
    """
    from hermes_cli.update_cmd import _current_checkout_sha
    try:
        from hermes_cli.update_receipt import read_latest_receipt

        receipt = read_latest_receipt()
    except Exception:
        receipt = None
    if not isinstance(receipt, dict):
        return False
    if not expected_sha:
        expected_sha = _current_checkout_sha()
    if not expected_sha:
        return False

    def _sha_mismatch(code_sha) -> bool:
        return bool(code_sha) and str(code_sha) != str(expected_sha)

    fleet = receipt.get("fleet")
    if isinstance(fleet, list) and fleet:
        for entry in fleet:
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == "stale":
                return True
            if _sha_mismatch(entry.get("code_sha")):
                return True
        return False

    if not _receipt_looks_unfinished(receipt):
        return False
    plan = receipt.get("plan")
    if not isinstance(plan, dict):
        return False
    for runtime in plan.get("runtimes") or []:
        if isinstance(runtime, dict) and _sha_mismatch(runtime.get("code_sha")):
            return True
    return False


def _pending_fleet_restart_needed() -> bool:
    """True when a prior pull still owes the fleet a restart (#95294)."""
    try:
        if _fleet_restart_pending_marker_path().is_file():
            return True
    except OSError:
        pass
    return _receipt_reports_stale_runtime()


def _warn_pending_fleet_restart(*, startup: bool = False) -> None:
    """Print the specific interrupted-update fleet-restart warning."""
    stream = sys.stderr if startup else sys.stdout
    print(
        "⚠ A previous `hermes update` pulled new code but did not "
        "restart running gateways.",
        file=stream,
    )
    print(
        "  Gateways may still be serving pre-update modules (mixed sys.modules).",
        file=stream,
    )
    if startup:
        print(
            "  Run `hermes update` or `hermes gateway restart`.",
            file=stream,
        )


def _warn_pending_fleet_restart_on_startup() -> None:
    """Cheap CLI-startup hint. Never restarts; never raises."""
    try:
        if not _pending_fleet_restart_needed():
            return
        _warn_pending_fleet_restart(startup=True)
    except Exception:
        pass


def _restart_systemd_gateway_units_best_effort(failed: list) -> None:
    """Best-effort ``systemctl restart`` of every hermes-gateway/serve unit."""
    for scope, scope_cmd in (
        ("user", ["systemctl", "--user"]),
        ("system", ["systemctl"]),
    ):
        try:
            result = _systemctl(
                scope_cmd + ["list-units", "hermes-gateway*", "hermes-serve*",
                             "--plain", "--no-legend", "--no-pager"],
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue

        def process_unit(svc_name: str, _scope=scope, _cmd=scope_cmd) -> None:
            restart_cmd = list(_cmd) + ["--no-ask-password", "restart", svc_name]
            if (
                _scope == "system"
                and hasattr(os, "geteuid")
                and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
            ):
                restart_cmd = ["sudo", "-n"] + restart_cmd
            _systemctl(restart_cmd, timeout=30)

        def on_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
            failed.append(svc_name)

        _for_each_systemd_gateway_unit(
            result.stdout,
            process_unit=process_unit,
            on_unit_timeout=on_timeout,
        )


def _run_pending_fleet_restart() -> bool:
    """Catch-up restart for gateways left on pre-update code (#95294).

    Returns True when restart completed or no services were running.
    Returns False if restart was incomplete. Never raises.
    """
    from hermes_cli.update_cmd import _m
    print("→ Restarting gateways left on pre-update code...")
    try:
        _m()._purge_stale_hermes_modules()
    except Exception:
        pass
    try:
        from hermes_cli.gateway import (
            find_gateway_pids,
            is_macos,
            is_windows,
            kill_gateway_processes,
            supports_systemd_services,
            _wait_for_gateway_exit,
        )
    except Exception as exc:
        _warn_gateway_restart_phase_aborted(exc, None)
        return False

    try:
        pids = list(find_gateway_pids(all_profiles=True))
    except Exception as exc:
        logger.debug("Pending fleet restart: gateway probe failed: %s", exc)
        pids = None

    if pids == []:
        print("  ✓ No running gateways — nothing to restart.")
        return True

    failed: list = []
    try:
        if supports_systemd_services():
            _restart_systemd_gateway_units_best_effort(failed)
        if is_macos():
            restarted: list = []
            try:
                _restart_macos_launchd_gateways(restarted, failed, 45.0)
            except Exception as exc:
                logger.debug("Pending fleet restart: launchd failed: %s", exc)
                failed.append("launchd")
        if is_windows():
            try:
                from hermes_cli import gateway_windows

                if gateway_windows.is_installed():
                    gateway_windows.restart()
            except Exception as exc:
                logger.debug("Pending fleet restart: Windows failed: %s", exc)
                failed.append("windows-gateway")
        leftover: list = []
        try:
            leftover = list(find_gateway_pids(all_profiles=True))
        except Exception:
            leftover = list(pids or [])
        if leftover:
            try:
                kill_gateway_processes(all_profiles=True)
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
            except Exception as exc:
                logger.debug("Pending fleet restart: PID stop failed: %s", exc)
        if failed:
            _warn_incomplete_gateway_fleet_restart(failed)
            return False
        print("  ✓ Pending fleet restart completed.")
        return True
    except Exception as exc:
        surviving = None
        try:
            surviving = list(find_gateway_pids(all_profiles=True))
        except Exception:
            surviving = pids
        _warn_gateway_restart_phase_aborted(exc, surviving)
        return False


def _apply_pending_fleet_restart_catchup() -> None:
    """On an already-up-to-date ``hermes update``, finish a skipped restart.

    No-op when nothing is pending. Exits 1 when the catch-up restart is
    incomplete so automation does not treat the fleet as healthy.
    """
    from hermes_cli.update_cmd import _run_pending_fleet_restart
    if not _pending_fleet_restart_needed():
        return
    print()
    _warn_pending_fleet_restart()
    print("→ Running the pending fleet restart...")
    if _run_pending_fleet_restart():
        _clear_fleet_restart_pending_marker()
        return
    print("  ⚠ Fleet restart incomplete. Recover with: hermes gateway restart")
    sys.exit(1)


def _systemctl(cmd: list, *, timeout: float):
    """Run a systemctl (or sudo systemctl) invocation, capturing utf-8 text with a timeout."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def _systemctl_reset_and_restart(manage_cmd: list, svc_name: str):
    """``reset-failed`` then ``restart`` a unit. Always clear failed state first: if
    systemd's own auto-restart attempts already parked the unit in a failed state,
    a plain ``restart`` can wedge against the RestartSec backoff and leave it dead."""
    _systemctl(manage_cmd + ["reset-failed", svc_name], timeout=10)
    return _systemctl(manage_cmd + ["restart", svc_name], timeout=15)


def _for_each_systemd_gateway_unit(
    list_units_stdout: str,
    *,
    process_unit,
    on_unit_timeout,
) -> None:
    """Process each ``hermes-gateway*.service``/``hermes-serve*.service`` unit
    from ``systemctl list-units``.

    ``subprocess.TimeoutExpired`` raised by ``process_unit`` is isolated to
    that unit via ``on_unit_timeout`` so one wedged systemctl call cannot
    abort the rest of the fleet (#68523).
    """
    for line in (list_units_stdout or "").strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not unit.endswith(".service"):
            continue
        # list-units is already pattern-filtered, but keep the name gate so a
        # stray line cannot enter the restart path. Require the exact base unit
        # or hyphenated profile family: ``startswith("hermes-serve")`` would
        # also accept the unrelated ``hermes-server.service`` (#83595).
        if not (
            unit == "hermes-gateway.service"
            or unit.startswith("hermes-gateway-")
            or unit == "hermes-serve.service"
            or unit.startswith("hermes-serve-")
        ):
            continue
        svc_name = unit.removesuffix(".service")
        try:
            process_unit(svc_name)
        except subprocess.TimeoutExpired as exc:
            on_unit_timeout(svc_name, exc)


def _service_unit_supports_graceful_sigusr1_restart(svc_name: str) -> bool:
    """Whether *svc_name* wires SIGUSR1 to a graceful drain-then-restart.

    Only ``hermes-gateway*`` units run ``gateway/run.py`` (the SIGUSR1
    handler). ``hermes-serve*`` units (#83438) don't: SIGUSR1 would just
    terminate them and burn the full drain budget, so they go straight to the
    blunt ``systemctl restart`` path.

    Same strict exact/hyphenated shape as the unit-name gate in
    ``_for_each_systemd_gateway_unit``, so a near-prefix unit like
    ``hermes-gatewayd`` can't be sent a SIGUSR1 it doesn't handle.
    """
    return svc_name == "hermes-gateway" or svc_name.startswith("hermes-gateway-")


def _warn_incomplete_gateway_fleet_restart(failed_units: list) -> None:
    """Print an explicit incomplete-update warning for unrestarted units."""
    from hermes_cli.gateway import is_macos

    if not failed_units:
        return
    # Preserve discovery order while de-duplicating.
    seen = set()
    ordered = []
    for name in failed_units:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    print()
    print("⚠ Update incomplete — some units were not restarted:")
    for name in ordered:
        print(f"    - {name}")
    if is_macos():
        # A launchd label lands here when launchd was not supervising a live
        # process after the restart (#88848) — very likely deregistered, which
        # `launchctl kickstart` cannot revive.
        print("  Listed services may be deregistered from launchd, or still")
        print("  running pre-update code (mixed sys.modules). Recover with:")
        print("    hermes gateway status")
        print("    launchctl list | grep <label>")
        print("    launchctl bootstrap gui/$(id -u) "
              "~/Library/LaunchAgents/<label>.plist")
        return
    print("  Skipped units may still be running pre-update code (mixed")
    print("  sys.modules). Restart them manually, then verify:")
    print("    hermes gateway status")
    if any(not name.startswith("ai.hermes.") for name in ordered):
        print("    systemctl --user restart <unit>   # user-scope")
        print("    sudo systemctl restart <unit>     # system-scope")
    if any(name.startswith("ai.hermes.") for name in ordered):
        print("    launchctl kickstart -k gui/$UID/<label>   # macOS (or user/$UID)")


def _restart_launchd_gateway_after_update(
    *, supervision_verify: bool = True
) -> tuple[list, list]:
    """Restart the invoking profile's launchd gateway after an update.

    No ``launchctl list``-based classification (#74973): a *booted-out* job
    (plist present, definition deregistered — crashed helper, manual bootout,
    failed prior update) fails that check, and ``launchctl list`` is
    session-scoped and can exit non-zero while the job is alive, so gating on
    it silently skipped the restart while still printing "Update complete!".
    When the plist exists, ``launchd_restart()`` always runs — it drains a
    live PID, kickstarts with ``-k``, and owns the bootout/bootstrap/kickstart
    ladder for the unloaded state. Every failure path is loud and names the
    manual recovery command.

    Returns ``(restarted_labels, failed_labels)``. With ``supervision_verify``
    (the update path), success additionally requires launchd reporting a fresh
    supervised PID (#88848 — "the call returned" is not "supervised").
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        get_launchd_plist_path,
        launchd_restart,
        wait_for_launchd_gateway_supervision,
    )

    current_label = get_launchd_label()
    try:
        if not get_launchd_plist_path().exists():
            return [], []  # not a launchd install — nothing to do or warn
        try:
            launchd_restart()
        except subprocess.CalledProcessError as e:
            stderr = (getattr(e, "stderr", "") or "").strip()
            print(
                f"  ⚠ Gateway restart failed: {stderr}\n"
                "    The gateway may be DOWN on pre-update code. "
                "Recover manually: hermes gateway restart"
            )
            return [], [current_label]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # A plist exists, so a gateway is SUPPOSED to be supervised; a broken/
        # wedged launchctl is not proof nothing needs restarting (#74973's
        # second silent variant). Count it and tell the operator.
        print(
            "  ⚠ Could not restart the gateway "
            f"({e.__class__.__name__}: {e}).\n"
            "    Recover manually: hermes gateway restart"
        )
        return [], [current_label]

    if not supervision_verify:
        return [current_label], []

    # launchd_restart() returning only means "restart REQUESTED" — both the
    # self-restart branch and a plist reload are asynchronous. A helper that
    # dies before its first bootstrap (#88848), or a bootstrap that exits 0
    # without registering (seen on macOS 26.6.1), would otherwise reach "Update
    # complete!" unsupervised. Verified domain-agnostically: a domain locate
    # fails on macOS-26 hosts whose per-user domains reject service management.
    if wait_for_launchd_gateway_supervision(label=current_label):
        return [current_label], []
    print(
        f"  ✗ {current_label} restarted but launchd is not supervising it.\n"
        "    Check logs, then: hermes gateway restart"
    )
    return [], [current_label]


def _restart_macos_launchd_gateways(
    restarted_services: list,
    failed_or_stale_units: list,
    drain_budget: float,
) -> None:
    """Restart every launchd-managed gateway after an update (macOS).

    The git pull is shared across profiles, so every ``ai.hermes.gateway*``
    LaunchAgent must reload it; restarting only the invoking profile leaves
    siblings on pre-update ``sys.modules`` (#41403). Parity with systemd.

    The invoking profile keeps ``launchd_restart()`` (self-restart request →
    drain → kickstart). Siblings get the same drain-first sequence with their
    launchd domain resolved per label (``gui/<uid>`` vs ``user/<uid>``) so
    none is kickstarted in the wrong domain. ``subprocess.TimeoutExpired`` is
    isolated per label so one wedged launchctl call cannot strand the fleet
    on old code (#68523).
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        launchd_gateway_labels_for_install,
        _graceful_restart_via_sigusr1,
        _launchd_kickstart,
        _locate_launchd_gateway_service,
        _wait_for_launchd_service_pid,
    )

    # --- Current profile: unchanged single-service path ---------------------
    _restarted, _failed = _restart_launchd_gateway_after_update(
        supervision_verify=True
    )
    restarted_services.extend(_restarted)
    failed_or_stale_units.extend(_failed)
    current_label = get_launchd_label()

    # --- Sibling profiles ---------------------------------------------------
    for label in launchd_gateway_labels_for_install():
        if label == current_label:
            continue
        try:
            # Locate = liveness + domain in one probe; the kickstart and
            # fresh-PID checks below reuse that domain so a sibling is never
            # probed in one gui/user domain and restarted in another.
            domain, old_pid = _locate_launchd_gateway_service(label)
            if domain is None:
                # Installed but not bootstrapped (stopped/uninstalled
                # mid-way) — nothing is running old code here.
                continue
            graceful_ok = False
            if old_pid is not None and old_pid > 0:
                print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
                graceful_ok = _graceful_restart_via_sigusr1(
                    old_pid, drain_timeout=drain_budget
                )
            if graceful_ok and _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=10.0, domain=domain
            ):
                # Unconditional KeepAlive already respawned it on the new
                # code — a hard kickstart now would kill the fresh process.
                restarted_services.append(label)
                continue
            try:
                _launchd_kickstart(label, domain)
            except subprocess.CalledProcessError as e:
                stderr = (getattr(e, "stderr", "") or "").strip()
                failed_or_stale_units.append(label)
                print(
                    f"  ⚠ Failed to restart {label}: {stderr}\n"
                    f"    Recover manually: launchctl kickstart -k {domain}/{label}"
                )
                continue
            if _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=15.0, domain=domain
            ):
                restarted_services.append(label)
            else:
                failed_or_stale_units.append(label)
                print(
                    f"  ✗ {label} failed to come back after restart.\n"
                    f"    Check logs, then: launchctl kickstart -k {domain}/{label}"
                )
        except subprocess.TimeoutExpired:
            failed_or_stale_units.append(label)
            print(
                f"  ⚠ launchctl timed out restarting {label}; "
                "continuing with remaining gateways"
            )


def _surviving_gateway_pids_after_failed_restart():
    """Best-effort PIDs of gateways still running after the restart phase died.

    ``None`` when undeterminable — notably when ``hermes_cli.gateway`` no
    longer imports, one of the ways the restart phase aborts (the checkout was
    replaced under a process holding old modules). The caller treats ``None``
    and a non-empty list as "assume stale"; only a positive empty result proves
    nothing needs restarting.
    """
    try:
        from hermes_cli.gateway import find_gateway_pids

        return list(find_gateway_pids(all_profiles=True))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not probe for surviving gateways after update: %s", exc)
        return None


_FRESH_RESTART_SUPERVISORS = frozenset({"systemd", "launchd", "service", "s6"})


def _gateway_service_matches_profile(profile: str, service: object) -> bool:
    """Match an exact gateway service/label to a profile.

    Profile names must not be matched as substrings: ``foo`` must not claim
    that ``hermes-gateway-foobar.service`` was already restarted.  These are
    the service/label shapes produced by the existing systemd, launchd, and
    s6 lifecycle implementations.
    """
    name = str(service).removesuffix(".service")
    if profile == "default":
        return name in {
            "hermes-gateway",
            "ai.hermes.gateway",
            "gateway",
            "gateway-default",
        }
    return name in {
        f"hermes-gateway-{profile}",
        f"ai.hermes.gateway-{profile}",
        f"gateway-{profile}",
    }


def _gateway_recovery_partition(
    plan, *, skip_profiles: set[str] | None = None
) -> tuple[dict[str, str], list[dict]]:
    """Partition pre-update runtimes into fresh-restart candidates and skips.

    Uses only the inventory captured before the checkout changed: re-importing
    ``hermes_cli.gateway`` in the failing interpreter is exactly what can raise
    the original ``ImportError``.

    Returns ``(candidates, skipped)``: ``candidates`` maps profile → supervisor
    for supervised gateway runtimes the fresh process may restart; ``skipped``
    lists every other inventoried runtime, each with an explicit reason, so
    nothing from the spawn ledger vanishes silently (manual gateways have no
    relaunch authority; serve/dashboard runtimes have no per-profile command).

    A ``skipped`` serve/dashboard entry does NOT mean unrecoverable: the fresh
    child runs a separate ``hermes-serve*`` systemd pass enumerating units from
    systemd, because the ledger collector cannot classify a systemd-launched
    ``hermes serve`` (no spawner ⇒ ``manual-serve``). Leftovers are caught by
    :func:`_surviving_pre_update_serve_runtimes` (#92145).
    """
    skip_profiles = skip_profiles or set()
    candidates: dict[str, str] = {}
    skipped: list[dict] = []
    try:
        for runtime in getattr(plan, "runtimes", ()) or ():
            kind = getattr(runtime, "kind", None)
            profile = getattr(runtime, "profile", None)
            supervisor = getattr(runtime, "supervisor", None)
            if not isinstance(profile, str) or not profile:
                continue
            if kind == "gateway":
                if profile in skip_profiles:
                    continue
                if supervisor in _FRESH_RESTART_SUPERVISORS:
                    candidates.setdefault(profile, str(supervisor))
                else:
                    skipped.append(
                        {
                            "profile": profile,
                            "kind": "gateway",
                            "supervisor": str(supervisor),
                            "reason": (
                                "manual gateway has no supervisor relaunch"
                                " authority; left running for explicit operator"
                                " restart"
                            ),
                        }
                    )
            elif kind in ("serve", "dashboard"):
                if supervisor == "desktop":
                    reason = (
                        "desktop app owns and respawns this serve backend;"
                        " the recovery pass must not restart it out from under"
                        " its supervisor"
                    )
                else:
                    # NOT a claim that no supervisor exists: a systemd-launched
                    # `hermes serve` sets neither HERMES_SPAWN nor
                    # HERMES_PARENT_PID, so it reads as "manual-serve".
                    # Unit-backed serves are recovered by the fresh child's
                    # systemd pass (enumerated from systemd); survivors are
                    # reported by _surviving_pre_update_serve_runtimes (#92145).
                    reason = (
                        "no per-profile relaunch command reaches a serve/"
                        "dashboard runtime; recovered by the fresh systemd"
                        " unit pass when it owns a hermes-serve* unit, else"
                        " left running for explicit operator restart"
                    )
                skipped.append(
                    {
                        "profile": profile,
                        "kind": str(kind),
                        "supervisor": str(supervisor),
                        "reason": reason,
                    }
                )
    except Exception as exc:
        logger.debug("Could not prepare fresh gateway restart profiles: %s", exc)
    return candidates, skipped


def _warn_gateway_restart_phase_aborted(exc: BaseException, pids) -> None:
    """Print a recovery warning when the whole restart phase raised.

    #78574: the phase was wrapped in a blanket ``except Exception`` logged at
    debug level, so an early failure (e.g. importing ``hermes_cli.gateway``
    from the fresh checkout) erased every drain/restart line; the update
    printed "Update complete!" and exited 0 while the gateway kept serving
    pre-update modules and died on the next turn with an ImportError.
    """
    print()
    print(f"⚠ Update incomplete — gateway auto-restart failed: {exc}")
    if pids:
        listed = ", ".join(str(pid) for pid in pids)
        print(f"  Gateway process(es) still running pre-update code: {listed}")
    else:
        print("  Any gateway still running is serving pre-update code")
        print("  (mixed sys.modules) against the updated checkout.")
    print("  Restart it manually, then verify:")
    print("    hermes gateway restart")
    print("    hermes gateway status")


def _drain_or_signal_gateway_for_update(
    pid: int,
    drain_budget: float,
    label: str,
) -> bool:
    """Decide how ``hermes update`` hands a running gateway over to new code.

    Three-way triage shared by the systemd and bare-process restart paths:

    1. **Gateway is an ancestor of this process** — deadlock break (#100179).
       When ``hermes update`` runs INSIDE the gateway's process tree (the
       hermes-auto-update cron job), waiting for the gateway is circular:
       gateway waits on in-flight work units (#77184) → cron session waits on
       ``hermes update`` → ``hermes update`` waits on the gateway. The
       wedged-loop probe can't break it (the cron session posts activity every
       ~180s, so it is never marked wedged) and the gateway burns the full
       1800s force-drain cap. Fire-and-forget instead: signal the restart and
       return; the gateway's own restart completes once THIS process exits.
    2. **Event loop provably wedged** (#81642) — SIGUSR1 can never drain it;
       bounded escalation (SIGTERM grace → SIGKILL) instead.
    3. **Live, out-of-tree gateway** — normal graceful SIGUSR1 drain, waiting
       up to ``drain_budget`` (including the #86684 cron floor).

    Returns True when the gateway was signalled/stopped successfully.
    """
    from hermes_cli.gateway import (
        GATEWAY_LOOP_WEDGED,
        _escalate_wedged_gateway,
        _graceful_restart_via_sigusr1,
        _is_pid_ancestor_of_current_process,
        _request_gateway_self_restart,
        probe_gateway_loop_liveness,
    )

    if _is_pid_ancestor_of_current_process(pid):
        print(
            f"  → {label}: update is running inside this gateway's "
            "process tree — signalling restart and letting the gateway "
            "drain itself (avoids the cron-update deadlock, #100179)"
        )
        return _request_gateway_self_restart(pid)
    if probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        print(
            f"  ⚠ {label}: gateway event loop is unresponsive — "
            "skipping drain, forcing a bounded stop..."
        )
        _escalate_wedged_gateway(pid)
        return True
    print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
    return _graceful_restart_via_sigusr1(pid, drain_timeout=drain_budget)


def _resolve_manage_cmd(cache: dict, scope_: str, scope_cmd_: list, svc_name_: str):
    """Resolve the command prefix for manage-units operations.

    Read-only systemctl calls work unprivileged, but manage-units verbs
    (``reset-failed``, ``start``, ``restart``) on a *system* service trigger a
    polkit auth prompt for non-root users. That prompt runs inside our captured
    10-15s subprocess — it flashes and dies before the user can answer, and the
    TimeoutExpired used to be swallowed silently.

    Strategy: root → plain systemctl. Otherwise try ``sudo -n`` — a blanket
    probe, then a targeted ``systemctl reset-failed`` probe so a least-privilege
    sudoers entry scoped to ``systemctl ... hermes-gateway*`` also qualifies
    (``reset-failed`` is an idempotent no-op we run before every privileged
    restart anyway). If neither works return None: the caller must SKIP the
    restart (without draining the gateway first!) and print manual steps.
    ``--no-ask-password`` guarantees polkit can never hang this path.
    """
    if scope_ in cache:
        return cache[scope_]
    cmd = scope_cmd_ + ["--no-ask-password"]
    if (
        scope_ == "system"
        and hasattr(os, "geteuid")
        and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
    ):
        sudo_cmd = ["sudo", "-n"] + scope_cmd_ + ["--no-ask-password"]
        sudo_ok = False
        try:
            _probe = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True,
                timeout=5,
            )
            sudo_ok = _probe.returncode == 0
            if not sudo_ok:
                # Blanket sudo refused — a targeted sudoers entry
                # (NOPASSWD for systemctl ... hermes-gateway*)
                # may still allow the exact commands we need.
                _probe = subprocess.run(
                    sudo_cmd + ["reset-failed", svc_name_],
                    capture_output=True,
                    timeout=5,
                )
                sudo_ok = _probe.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            sudo_ok = False
        cmd = sudo_cmd if sudo_ok else None
    cache[scope_] = cmd
    return cmd


def _restart_systemd_gateway_units(
    restarted_services, failed_or_stale_units, restarted_scoped_units, drain_budget
):
    """Restart every active hermes-gateway*/hermes-serve* systemd unit (user + system scope).

    Appends settled units to ``restarted_services`` (bare names) and
    ``restarted_scoped_units`` (``scope/name``), failures to ``failed_or_stale_units``.
    Per-unit timeouts are isolated so one wedged unit never aborts the fleet.
    """
    from hermes_cli.gateway import supports_systemd_services, _ensure_user_systemd_env

    _manage_cmd_cache: dict = {}

    # --- Systemd services (Linux) ---
    # Discover all hermes-gateway* units (default + profiles) plus
    # hermes-serve* units (the Desktop app's backend, #83438).
    if supports_systemd_services():
        try:
            _ensure_user_systemd_env()
        except Exception:
            pass

        for scope, scope_cmd in [
            ("user", ["systemctl", "--user"]),
            ("system", ["systemctl"]),
        ]:
            try:
                result = _systemctl(
                    scope_cmd + ["list-units", "hermes-gateway*", "hermes-serve*",
                                 "--plain", "--no-legend", "--no-pager"],
                    timeout=10,
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired as exc:
                # Discovery timeout — skip this scope, keep the other.
                print(
                    f"  ⚠ systemctl timed out listing {scope}-scope "
                    f"gateway units ({exc.cmd if exc.cmd else 'unknown command'}). "
                    f"Check the gateway with: hermes gateway status"
                )
                continue

            def _restart_one_systemd_gateway_unit(svc_name: str) -> None:
                # Check if active
                check = _systemctl(scope_cmd + ["is-active", svc_name], timeout=5)
                if check.stdout.strip() != "active":
                    return

                # Resolve how we may run manage-units verbs for this scope.
                # None ⇒ no non-interactive privilege path; avoid those verbs
                # entirely or polkit throws an auth prompt inside our captured
                # 10-15s subprocess (it flashes and "exits directly").
                _manage_cmd = _resolve_manage_cmd(_manage_cmd_cache, 
                    scope, scope_cmd, svc_name
                )

                # Prefer a graceful SIGUSR1 restart so in-flight agent runs
                # drain instead of being SIGKILLed: the handler calls
                # request_restart(via_service=True) → drain → exit, and
                # Restart=always respawns the unit. hermes-serve has no such
                # handler, so it skips straight to the blunt restart below.
                _main_pid = 0
                if _service_unit_supports_graceful_sigusr1_restart(svc_name):
                    try:
                        _show = _systemctl(scope_cmd + ["show", svc_name, "--property=MainPID", "--value"], timeout=5)
                        _main_pid = int((_show.stdout or "").strip() or 0)
                    except (
                        ValueError,
                        subprocess.TimeoutExpired,
                        FileNotFoundError,
                    ):
                        _main_pid = 0

                _graceful_ok = False
                if _main_pid > 0:
                    # Three-way triage (#100179 ancestor / #81642 wedged /
                    # graceful drain), shared with the bare-process path.
                    _graceful_ok = _drain_or_signal_gateway_for_update(
                        _main_pid, drain_budget, svc_name
                    )

                if _graceful_ok:
                    # Gateway exited after a planned restart. ``Restart=always``
                    # respawns the unit only after ``RestartSec`` (60s on our
                    # unit file) — a crash-loop guard that is dead time for a
                    # voluntary update restart. ``reset-failed`` + ``start``
                    # skips RestartSec (we initiate the unit manually), taking
                    # ~1-3s on a warm box; if RestartSec already elapsed while
                    # draining, ``start`` is a no-op and we fall through to the
                    # poll below. Either way the 60s+ delay collapses to ~5s.
                    #
                    # The shortcut needs manage-units privileges; without them
                    # skip it — systemd's auto-restart still relaunches the
                    # unit after RestartSec.
                    if _manage_cmd is not None:
                        _systemctl(_manage_cmd + ["reset-failed", svc_name], timeout=10)
                        _systemctl(_manage_cmd + ["start", svc_name], timeout=15)
                        # Short poll: RestartSec was bypassed, so it should be up in seconds.
                        if _wait_for_service_active(
                            scope_cmd,
                            svc_name,
                            timeout=10.0,
                        ):
                            restarted_services.append(svc_name)
                            return
                    # Passive poll: systemd's auto-restart fires after
                    # RestartSec regardless of privileges — the primary path
                    # when _manage_cmd is None, the fallback otherwise.
                    _restart_sec = _service_restart_sec(
                        scope_cmd,
                        svc_name,
                        default=0.0,
                    )
                    _post_drain_timeout = max(
                        10.0,
                        _restart_sec + 10.0,
                    )
                    if _manage_cmd is None and _restart_sec > 5.0:
                        print(
                            f"  → {svc_name}: waiting for systemd "
                            f"auto-restart (~{int(_restart_sec)}s; "
                            "no root for an immediate restart)..."
                        )
                    if _wait_for_service_active(
                        scope_cmd,
                        svc_name,
                        timeout=_post_drain_timeout,
                    ):
                        restarted_services.append(svc_name)
                        return
                    # Exited but not respawned (older unit without
                    # Restart=on-failure / RestartForceExitStatus=75); fall
                    # through to systemctl start/restart.
                    print(
                        f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart"
                    )

                # Forcing a restart needs manage-units privileges. Without a
                # non-interactive path, systemctl would spawn a polkit prompt
                # inside a captured 10-15s subprocess (flashes and dies before
                # the user can answer) — skip with clear instructions.
                if _manage_cmd is None:
                    failed_or_stale_units.append(svc_name)
                    print(
                        f"  ⚠ {svc_name} is a system service and restarting it needs root.\n"
                        f"    Restart it manually to load the new version:\n"
                        f"      sudo systemctl restart {svc_name}\n"
                        f"    To let `hermes update` restart it automatically, allow\n"
                        f"    passwordless sudo for systemctl, or run updates with sudo."
                    )
                    return

                # Fallback: blunt systemctl restart — only reached when the graceful
                # path failed (unit missing SIGUSR1 wiring, drain exceeded the budget,
                # restart-policy mismatch). Mirrors `hermes gateway restart`
                # (`systemd_restart()`, PR #20949).
                restart = _systemctl_reset_and_restart(_manage_cmd, svc_name)
                if restart.returncode == 0:
                    # systemctl restart returns 0 even if the new process
                    # crashes immediately — verify it survived.
                    if _wait_for_service_active(
                        scope_cmd,
                        svc_name,
                        timeout=10.0,
                    ):
                        restarted_services.append(svc_name)
                    else:
                        # Retry once — transient startup failures (stale module
                        # cache, import race) often resolve on the second try.
                        # Clear failed state first so the retry isn't blocked.
                        print(
                            f"  ⚠ {svc_name} died after restart, retrying..."
                        )
                        _systemctl_reset_and_restart(_manage_cmd, svc_name)
                        if _wait_for_service_active(
                            scope_cmd,
                            svc_name,
                            timeout=10.0,
                        ):
                            restarted_services.append(svc_name)
                            print(f"  ✓ {svc_name} recovered on retry")
                        else:
                            failed_or_stale_units.append(svc_name)
                            _scope_flag = "--user " if scope == "user" else ""
                            _sudo_hint = "sudo " if scope == "system" else ""
                            print(
                                f"  ✗ {svc_name} failed to stay running after restart.\n"
                                f"    Check logs: {_sudo_hint}journalctl {_scope_flag}-u {svc_name} --since '2 min ago'\n"
                                f"    Recover manually:\n"
                                f"      {_sudo_hint}systemctl {_scope_flag}reset-failed {svc_name}\n"
                                f"      {_sudo_hint}systemctl {_scope_flag}restart {svc_name}"
                            )
                else:
                    failed_or_stale_units.append(svc_name)
                    print(
                        f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}"
                    )

            def _on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
                # Isolate the timeout to this unit and keep going
                # (#68523). A scope-wide handler used to abort every
                # later gateway and leave the fleet on mixed code.
                failed_or_stale_units.append(svc_name)
                print(
                    f"  ⚠ systemctl timed out restarting {svc_name} "
                    f"({exc.cmd if exc.cmd else 'unknown command'}); "
                    f"continuing with remaining gateways"
                )

            # Qualify everything this scope appended to ``restarted_services``
            # before the next scope can add a same-named unit; ``finally`` so a
            # mid-scope abort still carries the units it settled.
            _scope_mark = len(restarted_services)
            try:
                _for_each_systemd_gateway_unit(
                    result.stdout,
                    process_unit=_restart_one_systemd_gateway_unit,
                    on_unit_timeout=_on_unit_timeout,
                )
            finally:
                restarted_scoped_units.update(
                    f"{scope}/{name}"
                    for name in restarted_services[_scope_mark:]
                )


@dataclass
class _GatewayRestartOutcome:
    """Bookkeeping the post-update gateway restart phase hands back to the update flow.

    ``restarted_services`` keeps bare unit names (the fleet probe, receipt and
    operator summary all read it); ``incomplete`` means at least one gateway
    may still be serving pre-update code.
    """

    incomplete: bool
    phase_errors: list
    pre_restart_gateway_pids: "list | None"
    restarted_services: list
    failed_or_stale_units: list
    relaunched_profiles: list
    externally_supervised_profiles: list
    killed_pids: set


def _restart_manual_gateways(
    _drain_budget,
    *,
    restarted_services,
    killed_pids,
    relaunched_profiles,
    externally_supervised_profiles,
    find_gateway_pids,
    find_profile_gateway_processes,
    _prepare_profile_gateway_update_restart,
    _get_service_pids,
    _wait_for_gateway_exit,
):
    """Drain/stop every manual (non-service) gateway and print the restart summary.

    Mutates ``killed_pids`` / ``relaunched_profiles`` /
    ``externally_supervised_profiles`` in place; raises like the inline
    code did so the caller's phase-abort recovery still fires.
    """
    import signal as _signal

    # --- Manual (non-service) gateways --- excluding PIDs of
    # just-restarted services so we don't kill what systemd/launchd spawned.
    service_pids = _get_service_pids(all_profiles=True)
    manual_pids = find_gateway_pids(
        exclude_pids=service_pids, all_profiles=True
    )
    profile_processes = {
        proc.pid: proc
        for proc in find_profile_gateway_processes(exclude_pids=service_pids)
        if proc.pid in manual_pids
    }
    # Profile gateways we could not arm a relaunch for must NOT keep
    # running on pre-update modules (#88654): hand them to the unmapped
    # sweep below, which stops them and lists them under "Restart manually".
    unrestartable_pids = set()
    for pid, proc in profile_processes.items():
        restart_mode = _prepare_profile_gateway_update_restart(
            proc.profile, pid
        )
        if restart_mode is None:
            # Previously a bare ``continue``: the gateway was neither
            # relaunched nor stopped nor mentioned, so it kept serving
            # from stale modules with no operator signal at all.
            print(
                f"  ⚠ {proc.profile}: could not arm an automatic "
                f"gateway restart for PID {pid} — stopping it instead "
                "so it cannot keep running pre-update code"
            )
            unrestartable_pids.add(pid)
            continue
        # Graceful SIGUSR1 drain first (in-flight runs finish), SIGTERM
        # fallback if unsupported or over budget — the watcher relaunches
        # either way. Three-way triage (ancestor fire-and-forget #100179 /
        # wedged escalation #81642 / normal drain) shared with the systemd
        # path; the helper announces its choice first because a silent
        # full-budget wait reads as a hung update (#44515).
        drained = _drain_or_signal_gateway_for_update(
            pid, _drain_budget, proc.profile
        )
        if not drained:
            try:
                os.kill(pid, _signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Wait up to 5s for the old process to exit before the watcher
        # respawns. Telegram keeps the old getUpdates session alive ~30s;
        # a new gateway connecting inside that window gets a 409 that
        # _handle_polling_conflict() retries through, but a brief wait
        # avoids that path on fast machines (watcher restarts in <1s).
        _wait_for_gateway_exit(timeout=5.0, force_after=None)
        killed_pids.add(pid)
        if restart_mode == "external-supervisor":
            externally_supervised_profiles.append(proc.profile)
        else:
            relaunched_profiles.append(proc.profile)

    for pid in manual_pids:
        if pid in profile_processes and pid not in unrestartable_pids:
            continue
        try:
            os.kill(pid, _signal.SIGTERM)
            killed_pids.add(pid)
        except (ProcessLookupError, PermissionError):
            pass

    if restarted_services or killed_pids:
        print()
        for svc in restarted_services:
            print(f"  ✓ Restarted {svc}")
        if relaunched_profiles:
            names = ", ".join(relaunched_profiles)
            print(f"  ✓ Restarting manual gateway profile(s): {names}")
        if externally_supervised_profiles:
            names = ", ".join(externally_supervised_profiles)
            print(
                "  ✓ Handed gateway profile(s) back to their external "
                f"supervisor: {names}"
            )
        unmapped_count = (
            len(killed_pids)
            - len(relaunched_profiles)
            - len(externally_supervised_profiles)
        )
        if unmapped_count:
            print(f"  → Stopped {unmapped_count} manual gateway process(es)")
            print("    Restart manually: hermes gateway run")
            if unmapped_count > 1:
                print(
                    "    (or: hermes -p <profile> gateway run  for each profile)"
                )


def _force_kill_stuck_gateways(killed_pids, *, find_gateway_pids, _get_service_pids):
    # --- Post-restart survivor sweep (#17648) ---------------------
    # Gateways that ignore SIGTERM (stuck drain, blocked I/O, zombie)
    # never exit, so the 120s profile watcher never respawns and the
    # user keeps hitting ImportError on stale sys.modules. Give graceful
    # paths a moment, then SIGKILL remaining pre-update PIDs.
    try:
        _time.sleep(3.0)
        _service_pids_after = _get_service_pids(all_profiles=True)
        _surviving = find_gateway_pids(
            exclude_pids=_service_pids_after,
            all_profiles=True,
        )
        # Only PIDs we already tried to kill; anything newer started
        # AFTER our restart attempt and is left alone.
        _stuck = [pid for pid in _surviving if pid in killed_pids]
        if _stuck:
            print()
            print(
                f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing"
            )
            from gateway.status import (
                get_process_start_time as _get_process_start_time,
                terminate_pid as _terminate_pid,
            )
            for pid in _stuck:
                try:
                    # taskkill /T /F on Windows, SIGKILL on POSIX —
                    # _signal.SIGKILL doesn't exist on Windows.
                    _terminate_pid(
                        pid,
                        force=True,
                        expected_start_time=_get_process_start_time(pid),
                    )
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            # Give the OS a beat to reap the processes so the
            # watchers see them exit and respawn.
            _time.sleep(1.5)
    except Exception as _sweep_exc:
        logger.debug("Post-restart survivor sweep failed: %s", _sweep_exc)


def _recover_after_restart_phase_abort(
    e,
    _pre_update_plan,
    *,
    gateway_mode,
    gateway_fleet_restart_incomplete,
    gateway_restart_phase_errors,
    _pre_restart_gateway_pids,
    restarted_services,
    restarted_scoped_units,
    failed_or_stale_units,
    relaunched_profiles,
    externally_supervised_profiles,
    killed_pids,
) -> bool:
    """Phase-abort recovery: fresh-child restart + fail-closed verdict.

    Returns the new ``incomplete`` flag; extends ``relaunched_profiles`` and
    ``gateway_restart_phase_errors`` in place.
    """
    from hermes_cli.update_cmd import (
        _abort_recovery_is_complete,
        _recover_gateway_restart_after_abort,
        _surviving_pre_update_serve_runtimes,
        _warn_stale_serve_runtimes,
        _write_gateway_update_exit_code,
    )

    logger.debug("Gateway restart during update failed: %s", e)
    gateway_restart_phase_errors.append(str(e))
    # An escaped exception means the restart output never printed. Treat
    # the fleet as stale unless we can positively prove no gateway runs
    # (#78574). An empty ``_surviving`` proves safety only if nothing was
    # running beforehand; a pre-restart gateway that is gone now was
    # stopped without a verified replacement, so ``[]`` still fails closed.
    _surviving = _surviving_gateway_pids_after_failed_restart()
    _already_restarted_profiles = set(relaunched_profiles)
    _already_restarted_profiles.update(externally_supervised_profiles)
    for runtime in getattr(_pre_update_plan, "runtimes", ()) or ():
        if getattr(runtime, "kind", None) != "gateway":
            continue
        profile = getattr(runtime, "profile", None)
        if not isinstance(profile, str):
            continue
        if any(
            _gateway_service_matches_profile(profile, service)
            for service in restarted_services
        ):
            _already_restarted_profiles.add(profile)
    _recovery_result = _recover_gateway_restart_after_abort(
        _pre_update_plan,
        gateway_mode=gateway_mode,
        skip_profiles=_already_restarted_profiles,
        skip_units=set(restarted_scoped_units),
    )
    _recovery_serve_units = _recovery_result.get("serve_units") or {}
    _serve_units_failed = list(_recovery_serve_units.get("failed") or [])
    # Deliberately NOT merged into ``restarted_services`` (gateway-phase
    # vocabulary feeding the fleet-probe expectation); serve coverage
    # lives in the recovery result and receipt. A serve/dashboard runtime
    # that is still the SAME pre-update process is live on old code
    # (#92145, e.g. tui_gateway under `hermes serve`, unreachable by any
    # `gateway restart`). Recovery may not claim success while one
    # remains, and must never kill one: manual/Desktop-owned serves have
    # no relaunch authority.
    _stale_runtime_rows = _surviving_pre_update_serve_runtimes(
        _pre_update_plan
    )
    _recovery_result["stale_runtimes"] = _stale_runtime_rows
    # Only systemd-VERIFIED outcomes may claim supervisor coverage.
    # A relaunch that merely exited 0 ("relaunch_attempted") was never
    # observed by the code and must not clear the incomplete flag.
    _recovery_verified = set(_recovery_result.get("verified") or [])
    if _recovery_verified:
        relaunched_profiles.extend(
            profile
            for profile in sorted(_recovery_verified)
            if profile not in relaunched_profiles
        )
    _planned_gateway_runtimes = [
        runtime
        for runtime in getattr(_pre_update_plan, "runtimes", ()) or ()
        if getattr(runtime, "kind", None) == "gateway"
        and isinstance(getattr(runtime, "profile", None), str)
    ]
    _planned_gateway_profiles = {
        runtime.profile for runtime in _planned_gateway_runtimes
    }
    _covered_gateway_profiles = (
        _already_restarted_profiles | _recovery_verified
    )
    _recovery_complete = _abort_recovery_is_complete(
        planned_gateway_profiles=_planned_gateway_profiles,
        covered_gateway_profiles=_covered_gateway_profiles,
        recovery_result=_recovery_result,
        stale_runtime_rows=_stale_runtime_rows,
    )
    if _recovery_complete:
        # The fresh child is the recovery terminal result. Leave the
        # final fleet-version matrix below as the authoritative
        # read-back before the update is declared successful.
        gateway_fleet_restart_incomplete = False
    elif (
        _restart_phase_failure_is_incomplete(
            _surviving, _pre_restart_gateway_pids
        )
        or _stale_runtime_rows
        or _serve_units_failed
    ):
        gateway_fleet_restart_incomplete = True
        _warn_gateway_restart_phase_aborted(e, _surviving)
        _warn_stale_serve_runtimes(_stale_runtime_rows)
        if gateway_mode:
            _write_gateway_update_exit_code(False)
    try:
        from hermes_cli.update_receipt import record_gateway_restart

        record_gateway_restart(
            restarted_services=restarted_services,
            relaunched_profiles=relaunched_profiles,
            externally_supervised_profiles=externally_supervised_profiles,
            killed_pids=sorted(killed_pids),
            failed_units=failed_or_stale_units,
            incomplete=gateway_fleet_restart_incomplete,
            phase_error=str(e),
            fresh_recovery=_recovery_result,
        )
    except Exception:
        pass
    return gateway_fleet_restart_incomplete


def _restart_gateway_fleet_after_update(_pre_update_plan, gateway_mode: bool):
    """Restart every running gateway (systemd, launchd, manual) so it picks up the pulled code.

    Never raises: a phase abort runs the fresh-child recovery and fails closed
    (``incomplete=True``) unless every planned gateway is verifiably covered.
    """
    from hermes_cli.update_cmd import _m, _write_gateway_update_exit_code

    gateway_fleet_restart_incomplete = False
    gateway_restart_phase_errors: list[str] = []
    # Gateways running before we touch anything. Stays empty until the probe
    # is imported and we are about to stop/drain, so an early exception has
    # nothing to fail closed on, while a failure after stopping a discovered
    # gateway fails closed on an empty survivor probe (#78574).
    _pre_restart_gateway_pids: list | None = []
    # Declared outside the try/except (never reset to None) so it is safe to
    # read even if the block raises early — already-restarted units are
    # forwarded to ``_finish_dashboard_update_cleanup`` (#83595).
    restarted_services: list = []
    # Scope-qualified twin of ``restarted_services`` (``user/hermes-serve``
    # vs ``system/hermes-serve`` are different processes; abort recovery
    # needs to know WHICH settled). ``restarted_services`` keeps bare names
    # for the fleet probe, receipt and summary (#92145).
    restarted_scoped_units: set = set()
    # Defined up front so abort recovery and fleet reconciliation can read
    # them even when the phase raises before its imports initialize them.
    failed_or_stale_units: list = []
    relaunched_profiles: list = []
    externally_supervised_profiles: list = []
    # Same treatment: the fleet version check uses killed_pids to decide
    # whether to wait for settle, and the except path forwards it to the receipt.
    killed_pids: set = set()

    # The pulled code is shared across profiles, so EVERY running gateway
    # restarts. Purge stale cached Hermes modules FIRST: the import below
    # loads new gateway source into this pre-update interpreter, and a
    # cached sibling (cli_output, status, ...) missing a symbol the new
    # source expects would ImportError and abort the whole phase.
    _m()._purge_stale_hermes_modules()
    try:
        from hermes_cli.gateway import (
            is_macos,
            find_gateway_pids,
            find_profile_gateway_processes,
            _prepare_profile_gateway_update_restart,
            _get_service_pids,
            _wait_for_gateway_exit,
        )

        # Wait budget for graceful SIGUSR1 restarts: covers both the
        # ``restart_after_turn_timeout`` deferral (#77184) and the
        # ``restart_drain_timeout`` inside stop(), so we don't hard-kill a
        # gateway still waiting on a turn. Units without SIGUSR1 wiring
        # just time out and fall back to ``systemctl restart``.
        try:
            from hermes_cli.gateway import _get_restart_exit_wait_budget

            _drain_budget = max(float(_get_restart_exit_wait_budget()), 45.0)
        except Exception:
            _drain_budget = 45.0

        failed_or_stale_units = []
        killed_pids = set()
        relaunched_profiles = []
        externally_supervised_profiles = []

        # Snapshot running gateways before any stop/drain so an empty
        # survivor probe later reads as "stopped and never came back", not
        # "nothing was running" (#78574). If the probe raises, None fails closed.
        try:
            _pre_restart_gateway_pids = list(find_gateway_pids(all_profiles=True))
        except Exception:
            _pre_restart_gateway_pids = None

        _restart_systemd_gateway_units(
            restarted_services, failed_or_stale_units, restarted_scoped_units, _drain_budget
        )

        # --- Launchd services (macOS): EVERY ai.hermes.gateway* LaunchAgent,
        # parity with systemd (#41403). Per-label TimeoutExpired isolation inside.
        if is_macos():
            try:
                _restart_macos_launchd_gateways(
                    restarted_services,
                    failed_or_stale_units,
                    _drain_budget,
                )
            except (FileNotFoundError, ImportError):
                pass

        _restart_manual_gateways(
            _drain_budget,
            restarted_services=restarted_services,
            killed_pids=killed_pids,
            relaunched_profiles=relaunched_profiles,
            externally_supervised_profiles=externally_supervised_profiles,
            find_gateway_pids=find_gateway_pids,
            find_profile_gateway_processes=find_profile_gateway_processes,
            _prepare_profile_gateway_update_restart=_prepare_profile_gateway_update_restart,
            _get_service_pids=_get_service_pids,
            _wait_for_gateway_exit=_wait_for_gateway_exit,
        )

        if failed_or_stale_units:
            gateway_fleet_restart_incomplete = True
            if gateway_mode:
                _write_gateway_update_exit_code(False)
        _warn_incomplete_gateway_fleet_restart(failed_or_stale_units)

        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=restarted_services,
                relaunched_profiles=relaunched_profiles,
                externally_supervised_profiles=externally_supervised_profiles,
                killed_pids=sorted(killed_pids),
                failed_units=failed_or_stale_units,
                incomplete=bool(failed_or_stale_units),
            )
        except Exception:
            pass

        _force_kill_stuck_gateways(
            killed_pids, find_gateway_pids=find_gateway_pids, _get_service_pids=_get_service_pids
        )

    except Exception as e:
        gateway_fleet_restart_incomplete = _recover_after_restart_phase_abort(
            e,
            _pre_update_plan,
            gateway_mode=gateway_mode,
            gateway_fleet_restart_incomplete=gateway_fleet_restart_incomplete,
            gateway_restart_phase_errors=gateway_restart_phase_errors,
            _pre_restart_gateway_pids=_pre_restart_gateway_pids,
            restarted_services=restarted_services,
            restarted_scoped_units=restarted_scoped_units,
            failed_or_stale_units=failed_or_stale_units,
            relaunched_profiles=relaunched_profiles,
            externally_supervised_profiles=externally_supervised_profiles,
            killed_pids=killed_pids,
        )

    return _GatewayRestartOutcome(
        incomplete=gateway_fleet_restart_incomplete,
        phase_errors=gateway_restart_phase_errors,
        pre_restart_gateway_pids=_pre_restart_gateway_pids,
        restarted_services=restarted_services,
        failed_or_stale_units=failed_or_stale_units,
        relaunched_profiles=relaunched_profiles,
        externally_supervised_profiles=externally_supervised_profiles,
        killed_pids=killed_pids,
    )


def _verify_fleet_after_update(
    restart,
    *,
    _pre_update_plan,
    _windows_gateway_resume,
    node_failures,
    update_complete,
):
    """Post-restart verification: legacy-unit warning, dashboard cleanup, stale serve
    probe, fleet version matrix, plan-vs-execution reconciliation, receipt finalize.

    Exits 1 (leaving ``fleet_restart_pending`` in place for the next catch-up)
    when any gateway may still be serving pre-update code; otherwise clears
    the pending marker.
    """
    from hermes_cli.update_cmd import (
        _finish_dashboard_update_cleanup,
        _m,
        _surviving_pre_update_serve_runtimes,
        _warn_stale_serve_runtimes,
    )
    # Legacy hermes.service + hermes-gateway.service SIGTERM-fight over the
    # same bot token (PR #11909); warn on every update until migrated.
    try:
        from hermes_cli.gateway import (
            has_legacy_hermes_units,
            _find_legacy_hermes_units,
            supports_systemd_services,
        )

        if supports_systemd_services() and has_legacy_hermes_units():
            print()
            print("⚠ Legacy Hermes gateway unit(s) detected:")
            for name, path, is_sys in _find_legacy_hermes_units():
                scope = "system" if is_sys else "user"
                print(f"    {path}  ({scope} scope)")
            print()
            print("  These pre-rename units (hermes.service) fight the current")
            print("  hermes-gateway.service for the bot token and cause SIGTERM")
            print("  flap loops. Remove them with:")
            print()
            print("    hermes gateway migrate-legacy")
            print()
            print("  (add `sudo` if any are in system scope)")
    except Exception as e:
        logger.debug("Legacy unit check during update failed: %s", e)

    # Restart a managed dashboard via systemd or stop stale manual ones
    # (raw-killing a systemd-owned PID reads as a clean stop and leaves the
    # Cloudflare origin dead). A failed Node refresh leaves the running
    # dashboard untouched. Already-restarted units (incl. hermes-serve*,
    # #83438) are forwarded so they aren't restarted twice (#83595).
    _finish_dashboard_update_cleanup(
        node_failures, already_restarted_units=set(restart.restarted_services)
    )

    # Success-path twin of the abort-recovery probe (#100479): the restart
    # phase only touches units, so a unit-less `hermes serve` keeps its
    # pre-update sys.modules and its cron ticker ImportErrors. Runs AFTER
    # dashboard cleanup so a respawned manual dashboard isn't a survivor.
    # Rows feed the reconciliation below (survivor → exit 1); ``None``
    # means the probe failed and reconciliation stays fail-closed.
    _stale_serve_rows: "list | None" = None
    try:
        _stale_serve_rows = _surviving_pre_update_serve_runtimes(_pre_update_plan)
        if _stale_serve_rows:
            _warn_stale_serve_runtimes(_stale_serve_rows)
    except Exception as _serve_warn_exc:
        logger.debug("Failed to check for surviving serve runtimes: %s", _serve_warn_exc)

    print()
    print("Tip: You can now select a provider and model:")
    print("  hermes model              # Select provider and model")

    # Phase 1 (#91277): post-update fleet version verification. Compare
    # every live gateway's stamped code_sha against the freshly-updated
    # checkout and surface any gateway still serving pre-update code —
    # instead of assuming the restart phase worked (#88654, #69754).
    _fleet_snapshot: list = []
    try:
        from hermes_cli.update_receipt import (
            collect_fleet_versions,
            print_fleet_version_matrix,
        )

        # Cross-platform "we expected fleet rows" signal (#93406). The
        # old (restart.restarted_services or restart.killed_pids) condition never fires
        # on Windows: the pause/resume phase populates neither list, so
        # a healthy resumed gateway yielded zero rows and exit 0.
        _fleet_rows_expected = _m()._fleet_probe_expected_runtimes(
            _pre_update_plan,
            restart.pre_restart_gateway_pids,
            _windows_gateway_resume,
            restart.restarted_services,
            restart.killed_pids,
        )
        # Settle window (skipped when nothing was running): restarted
        # gateways need time to rewrite gateway_state.json. Windows resumes
        # DETACHED and may take ~10s to boot, so a single 2s sleep reported
        # "no rows" (exit 1) on healthy resumes and the retry re-killed the
        # new gateway. Poll a bounded window instead.
        _fleet_snapshot = []
        if _fleet_rows_expected:
            _fleet_deadline = _time.monotonic() + 30.0
            while True:
                _time.sleep(2.0)
                # Pass the pre-restart PID snapshot so a gateway the
                # restart phase stopped WITHOUT a verified replacement
                # shows as a DOWN row (exit 1) instead of silently
                # producing no row at all.
                _fleet_snapshot = collect_fleet_versions(
                    pre_restart_pids=restart.pre_restart_gateway_pids
                )
                # A "down" row may just be a detached replacement still
                # booting; keep polling until no "down" rows remain or the
                # deadline passes, so a slow gateway isn't misread.
                if _fleet_snapshot and not any(
                    row.get("state") == "down" for row in _fleet_snapshot
                ):
                    break
                if _time.monotonic() >= _fleet_deadline:
                    break
        else:
            _fleet_snapshot = collect_fleet_versions(
                pre_restart_pids=restart.pre_restart_gateway_pids
            )
        if print_fleet_version_matrix(_fleet_snapshot):
            restart.incomplete = True
        elif not _fleet_snapshot and _fleet_rows_expected:
            # Zero rows although a gateway was (or may have been) live
            # pre-update. collect_fleet_versions() swallows every failure,
            # so an empty list is indistinguishable from a healthy fleet —
            # treat it as verification failure (receipt "partial", exit 1) (#93406).
            print(
                "\n⚠ Fleet version check returned no rows even though"
                " gateway runtimes were expected — verification incomplete."
            )
            restart.incomplete = True
    except Exception as _fleet_exc:
        logger.debug("Fleet version verification failed: %s", _fleet_exc)

    # Plan-vs-execution reconciliation (#91277): every runtime the PLAN saw
    # must appear in the restart bookkeeping; an unaccounted one is a
    # silent miss and escalates like a STALE/DOWN row.
    _runtime_outcomes: list = []
    try:
        if _pre_update_plan is not None and _pre_update_plan.runtimes:
            from hermes_cli.update_inventory import (
                match_runtime_outcomes,
                report_unaccounted_runtimes,
            )

            _runtime_outcomes = match_runtime_outcomes(
                _pre_update_plan,
                restarted_services=restart.restarted_services,
                relaunched_profiles=restart.relaunched_profiles,
                externally_supervised_profiles=restart.externally_supervised_profiles,
                killed_pids=restart.killed_pids,
                failed_units=restart.failed_or_stale_units,
                # Serve/dashboard runtimes reconcile by incarnation
                # liveness, not by the gateway's unit names (#100479).
                stale_serve_pids=(
                    {row.get("pid") for row in _stale_serve_rows}
                    if _stale_serve_rows is not None
                    else None
                ),
            )
            if report_unaccounted_runtimes(_runtime_outcomes):
                restart.incomplete = True
            try:
                import hermes_cli.update_receipt as _ur

                if _ur._current is not None:
                    _ur._current.data["runtime_outcomes"] = _runtime_outcomes
            except Exception:
                pass
    except Exception as _outcome_exc:
        logger.debug("Runtime-outcome reconciliation failed: %s", _outcome_exc)

    try:
        from hermes_cli.update_receipt import finalize_update_receipt

        _receipt_path = finalize_update_receipt(
            (
                "partial"
                if restart.incomplete or not update_complete
                else "success"
            ),
            fleet=_fleet_snapshot,
        )
        if _receipt_path is not None:
            logger.info("Update receipt written: %s", _receipt_path)
    except Exception as _receipt_exc:
        logger.debug("Update receipt finalize failed: %s", _receipt_exc)

    if restart.incomplete:
        # Code update itself succeeded, but at least one gateway still
        # runs pre-update modules — surface that as a failed update so
        # automation / operators do not treat the fleet as healthy.
        # Leave ``fleet_restart_pending`` in place so the next
        # ``hermes update`` still runs the catch-up restart.
        sys.exit(1)
    _clear_fleet_restart_pending_marker()


def _restart_phase_failure_is_incomplete(surviving, pre_restart_pids) -> bool:
    """Whether an escaped gateway-restart-phase exception must fail the update.

    Fail closed unless the fleet is provably safe: ``surviving is None``
    (probe couldn't determine state, e.g. new ``hermes_cli.gateway`` no
    longer imports) or non-empty -> stale. ``surviving == []`` is proof of
    safety ONLY if nothing ran beforehand; a pre-restart gateway
    (``pre_restart_pids`` non-empty, or ``None`` = unreadable) that is gone
    now was stopped without a verified replacement (#78574).
    """
    if surviving is None or surviving:
        return True
    # surviving == []: safe only if we know nothing was running beforehand.
    return pre_restart_pids is None or bool(pre_restart_pids)


def _fleet_probe_expected_runtimes(
    pre_update_plan,
    pre_restart_pids,
    windows_resume_token,
    restarted_services,
    killed_pids,
) -> bool:
    """Whether the post-update fleet probe should have produced rows.

    The zero-rows fail-open (#93406): ``collect_fleet_versions()`` swallows
    every probe failure via ``logger.debug()`` and ``print_fleet_version_matrix([])``
    early-returns ``False``, so an empty snapshot reads as \"healthy fleet\" and
    the update exits 0.  An empty snapshot is only proof-of-safety when NOTHING
    says a gateway existed before the update.  Any of these signals means at
    least one runtime was (or may have been) live pre-update, so zero rows is
    verification failure, not health:

    * ``restarted_services`` / ``killed_pids`` — the POSIX restart phase
      touched live gateways.
    * ``pre_restart_pids`` non-empty, or ``None`` (pre-state unreadable —
      cannot prove nothing was running; same contract as
      ``_restart_phase_failure_is_incomplete``, #78574).
    * the pre-update plan inventoried ≥1 runtime.

    ``windows_resume_token`` is deliberately EXCLUDED (#93406 residual). The
    pause/resume token is bookkeeping for ``_pause_windows_gateways_for_update``
    / ``_resume_windows_gateways_after_update`` — it is not a runtime
    inventory, and its entries do not correspond to rows
    ``collect_fleet_versions()`` is capable of returning:

    * ``unmapped`` entries (Scheduled-Task gateways) never publish
      ``gateway_state.json`` rows at all, and
    * a paused profile gateway is resumed as a DETACHED relaunch that may not
      republish its identity within the probe window.

    Counting the token therefore made ``_fleet_rows_expected`` True on every
    Windows update that had paused a gateway, the probe's polling window ran
    out with zero rows on a perfectly healthy update, and verification
    reported "no rows … verification incomplete" and exited 1 after a long
    silent wait. Expected-runtimes must key only on signals that map to rows
    the probe can actually see; a genuinely live pre-update Windows gateway
    is already covered by ``pre_restart_pids`` and the plan inventory. The
    parameter stays in the signature so the call site keeps passing the token
    (cheap, explicit, and the docstring is where the exclusion is explained).

    The same condition gates the 2.0s settle sleep: a freshly restarted
    gateway needs the settle window to rewrite ``gateway_state.json``.

    Note this keys ONLY on zero-rows-despite-expected-runtimes.  A non-empty
    snapshot — including rows in ``unknown`` state — is still judged solely by
    ``print_fleet_version_matrix``.
    """
    del windows_resume_token  # excluded on purpose — see docstring (#93406)
    if restarted_services or killed_pids:
        return True
    if pre_restart_pids is None or pre_restart_pids:
        return True
    try:
        if pre_update_plan is not None and pre_update_plan.runtimes:
            return True
    except Exception:
        pass
    return False


def _wait_for_service_active(
    scope_cmd_: list,
    svc_name_: str,
    timeout: float = 10.0,
) -> bool:
    """Poll ``systemctl is-active`` until the unit reports active.

    systemd's Stopped -> Started transition after a graceful exit
    (or a hard restart) is not instantaneous; a one-shot check
    races that window and falsely reports the unit as down.
    Poll every 0.5s up to ``timeout`` seconds before giving up.
    """
    deadline = _time.monotonic() + max(timeout, 0.5)
    while True:
        try:
            _verify = _systemctl(scope_cmd_ + ["is-active", svc_name_], timeout=5)
            if _verify.stdout.strip() == "active":
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)


def _service_restart_sec(
    scope_cmd_: list,
    svc_name_: str,
    default: float = 0.0,
) -> float:
    """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

    After a graceful exit-75, systemd waits ``RestartSec`` before
    respawning the unit.  Callers that poll for ``is-active``
    must use a timeout >= ``RestartSec`` + transition slack, or
    they'll give up *during* the cooldown window and wrongly
    conclude the unit didn't relaunch.
    """
    try:
        _show = _systemctl(scope_cmd_ + ["show", svc_name_, "--property=RestartUSec", "--value"], timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # systemd emits values like "30s", "100ms", "1min 30s", or
    # "infinity".  Parse conservatively; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in (
            ("ms", 0.001),
            ("us", 0.000001),
            ("min", 60.0),
            ("s", 1.0),
        ):
            if part.endswith(_suf):
                try:
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                except ValueError:
                    pass
                break
    return total if matched else default
