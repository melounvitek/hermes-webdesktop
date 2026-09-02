"""Fresh-process recovery after the update's in-process restart phase aborts.

``hermes update`` performs its fleet restart in the interpreter that started before ``git pull``.

It is deliberately a separate owner from ``update_cmd``: the abort path has its own vocabulary
(``verified`` / ``relaunch_attempted`` / ``failed``, serve units, survivors) and its own fail-closed
contract, and the update monolith must not grow another authority surface (review on #96235).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def _serve_unit_recovery_available() -> bool:
    """Can a fresh process restart ``hermes-serve*`` units on this host?"""
    return sys.platform == "linux" and bool(shutil.which("systemctl"))


def _surviving_pre_update_serve_runtimes(plan) -> list[dict]:
    """Pre-update serve/dashboard runtimes that are STILL the same process.

    Identity is the process incarnation ``(pid, create_time)``, never the PID alone.
    ``ledger_entries()`` re-verifies that pair on every read and prunes anything that is gone, so a
    live entry is a real process — but the numeric PID can be reused, and a serve that was restarted
    correctly can come back on the same number.

    Anything still here after the recovery pass is a live runtime on the pre-update code generation,
    which is precisely the unsafe state #92145 reports. Fail closed on missing evidence: an
    unreadable ledger, or an incarnation neither side can produce, counts the runtime as surviving.
    """
    planned: dict[int, dict] = {}
    try:
        for runtime in getattr(plan, "runtimes", ()) or ():
            if getattr(runtime, "kind", None) not in ("serve", "dashboard"):
                continue
            pid = getattr(runtime, "pid", None)
            if not isinstance(pid, int) or pid <= 0:
                continue
            detail = getattr(runtime, "detail", None)
            planned[pid] = {
                "pid": pid,
                "kind": str(getattr(runtime, "kind", "")),
                "profile": str(getattr(runtime, "profile", "")),
                "supervisor": str(getattr(runtime, "supervisor", "")),
                "_create_time": _numeric(detail.get("create_time") if isinstance(detail, dict) else None),
            }
    except Exception as exc:
        logger.debug("Could not read planned serve runtimes: %s", exc)
        return []
    if not planned:
        return []
    try:
        from hermes_cli.process_identity import ledger_entries

        live: dict[int, float | None] = {
            entry["pid"]: _numeric(entry.get("create_time"))
            for entry in ledger_entries()
            if entry.get("purpose") in ("serve", "dashboard") and isinstance(entry.get("pid"), int)
        }
    except Exception as exc:
        logger.debug("Serve/dashboard survivor probe failed: %s", exc)
        live = None
    survivors = []
    for pid, row in planned.items():
        if live is not None:
            if pid not in live:
                continue
            planned_created, live_created = row["_create_time"], live[pid]
            if (
                planned_created is not None
                and live_created is not None
                and abs(float(live_created) - float(planned_created)) >= 2.0
            ):
                # Same number, different process: the pre-update runtime is gone
                # and something new registered under its PID. Not a survivor.
                continue
        survivors.append(_without_incarnation(row))
    return sorted(survivors, key=lambda row: row["pid"])


def _numeric(value):
    return value if isinstance(value, (int, float)) else None


def _without_incarnation(row: dict) -> dict:
    """The operator-facing survivor row (incarnation is a matching key only)."""
    return {key: value for key, value in row.items() if key != "_create_time"}


def _qualified_serve_skips(skip_units) -> list[dict]:
    """Scope-qualify the units the aborted phase already settled.

    ``restarted_scoped_units`` records ``<scope>/<unit>`` because ``hermes-serve.service`` can exist
    in BOTH the user and the system manager, and those are two different processes.
    """
    rows: list[dict] = []
    for entry in sorted(skip_units or ()):
        scope, sep, unit = str(entry).partition("/")
        if sep and scope in ("user", "system") and unit:
            rows.append({"scope": scope, "unit": unit})
        elif entry:
            rows.append({"unit": str(entry)})
    return rows


def _recover_gateway_restart_after_abort(
    plan,
    *,
    gateway_mode: bool,
    skip_profiles: set[str] | None = None,
    skip_units: set[str] | None = None,
) -> dict[str, list]:
    """Retry supervised gateway restarts from a clean Python process.

    ``hermes update`` normally performs the fleet restart in the interpreter that started before
    ``git pull``.

    Only profiles classified as supervisor-owned by the pre-update inventory are handed off.
    """
    from hermes_cli.update_cmd import _gateway_recovery_partition

    candidates, skipped = _gateway_recovery_partition(plan, skip_profiles=skip_profiles)
    profiles = sorted(candidates)
    recover_serve = _serve_unit_recovery_available()
    _empty_serve: dict[str, list] = {"verified": [], "failed": []}

    def _result(requested, verified, relaunch_attempted, failed, serve_units=None) -> dict[str, list]:
        return {
            "requested": requested,
            "verified": verified,
            "relaunch_attempted": relaunch_attempted,
            "failed": failed,
            "skipped": skipped,
            "serve_units": dict(_empty_serve) if serve_units is None else serve_units,
        }

    if not profiles and not recover_serve:
        return _result([], [], [], [])

    def _all_failed() -> dict[str, list]:
        return _result(profiles, [], [], profiles)

    command = [sys.executable, "-m", "hermes_cli.update_restart_recovery", "--stdin"]
    env = os.environ.copy()
    env["HERMES_UPDATE_RESTART_RECOVERY"] = "1"
    for marker in ("_HERMES_GATEWAY", "HERMES_GATEWAY", "HERMES_GATEWAY_MODE"):
        env.pop(marker, None)

    # A gateway-triggered update may run inside the gateway's systemd cgroup.
    # Put the recovery process in a transient user scope before it asks systemd
    # to restart that gateway, otherwise KillMode can terminate the recovery
    # process together with the old service. If systemd-run is unavailable,
    # fail closed rather than pretending the in-cgroup child is independent.
    if gateway_mode and sys.platform == "linux":
        systemd_run = shutil.which("systemd-run")
        if not systemd_run:
            logger.warning("Cannot isolate fresh gateway recovery from the gateway cgroup")
            return _all_failed()
        command = [systemd_run, "--user", "--scope", "--quiet", "--collect", "--", *command]

    kwargs = {
        "input": json.dumps(
            {
                "profiles": profiles,
                "supervisors": candidates,
                "serve_units": {
                    "recover": recover_serve,
                    "skip": _qualified_serve_skips(skip_units),
                },
            }
        ),
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
        "env": env,
        # Gateway profiles run sequentially at up to 90s each; the serve pass
        # adds its own restart + settle budget on top, so give the child room
        # for both rather than killing a recovery that was working.
        "timeout": max(180, 30 + 90 * len(profiles) + (150 if recover_serve else 0)),
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True

    try:
        result = subprocess.run(command, **kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Fresh gateway restart recovery failed: %s", exc)
        return _all_failed()

    if result.returncode != 0:
        logger.warning("Fresh gateway restart recovery exited %s", result.returncode)
        return _all_failed()

    try:
        recovery_result = json.loads(result.stdout or "")
        verified = recovery_result.get("verified")
        relaunch_attempted = recovery_result.get("relaunch_attempted")
        failed = recovery_result.get("failed")
        raw_serve = recovery_result.get("serve_units") or dict(_empty_serve)
    except (AttributeError, TypeError, ValueError):
        logger.warning("Fresh gateway restart recovery returned invalid JSON")
        return _all_failed()

    serve_units = dict(_empty_serve)
    if (
        isinstance(raw_serve, dict)
        and isinstance(raw_serve.get("verified"), list)
        and isinstance(raw_serve.get("failed"), list)
        and all(
            isinstance(unit, str)
            for unit in (*raw_serve["verified"], *raw_serve["failed"])
        )
    ):
        serve_units = {
            "verified": sorted(raw_serve["verified"]),
            "failed": sorted(raw_serve["failed"]),
        }
    elif recover_serve:
        # An unreadable serve block cannot be read as "nothing to do": the
        # units it describes are exactly the ones that host tui_gateway and
        # may still be serving the pre-update generation.
        logger.warning("Fresh recovery returned an invalid serve-unit result")
        serve_units = {"verified": [], "failed": ["<unreadable>"]}

    buckets = (verified, relaunch_attempted, failed)
    reported: list[str] = []
    if all(isinstance(bucket, list) for bucket in buckets):
        reported = [*verified, *relaunch_attempted, *failed]
    if (
        not all(isinstance(bucket, list) for bucket in buckets)
        or any(not isinstance(profile, str) for profile in reported)
        or set(reported) != set(profiles)
        or len(reported) != len(set(reported))
    ):
        logger.warning("Fresh gateway restart recovery returned incomplete profiles")
        return _all_failed()

    verified, relaunch_attempted, failed = sorted(verified), sorted(relaunch_attempted), sorted(failed)
    for names, text in (
        (verified, "  ✓ Restarted supervised gateway(s) in a fresh process (systemd-verified active): "),
        (relaunch_attempted, "  ⚠ Relaunch attempted in a fresh process but not"
                             " supervisor-verified (check these gateways manually): "),
        (serve_units["verified"], "  ✓ Restarted serve unit(s) in a fresh process (new main PID observed): "),
        (serve_units["failed"], "  ⚠ Could not verify a replacement for serve unit(s): "),
    ):
        if names:
            print(text + ", ".join(names))
    return _result(profiles, verified, relaunch_attempted, failed, serve_units)


def _warn_stale_serve_runtimes(rows) -> None:
    """Name the serve/dashboard processes that survived on pre-update code.

    ``hermes serve`` hosts ``tui_gateway.server``; if its unit was never restarted it keeps the
    pre-pull ``sys.modules`` graph, every chat turn fails with an ``ImportError`` for a symbol
    that imports fine on disk, and no gateway row reveals the culprit. Print the PIDs and the
    exact command that fixes it.
    """
    if not rows:
        return
    print(
        "  ⚠ These serve/dashboard processes still run pre-update code"
        " (they started before the checkout changed):"
    )
    for row in rows:
        supervisor = row.get("supervisor") or "unknown"
        print(
            f"      pid {row.get('pid')} — {row.get('kind')}"
            f" (profile {row.get('profile') or 'default'}, {supervisor})"
        )
    print(
        "    Restart them before using Hermes again, e.g."
        " `systemctl --user restart hermes-serve.service`"
        " or by relaunching `hermes serve` / the Desktop app."
    )


def _abort_recovery_is_complete(
    *,
    planned_gateway_profiles,
    covered_gateway_profiles,
    recovery_result,
    stale_runtime_rows,
) -> bool:
    """May a fresh-process recovery clear the incomplete flag?

    Only when EVERY inventoried runtime family is accounted for.

    ``planned_gateway_profiles`` being empty is deliberately NOT completeness: with no gateway leg
    to prove, the caller's own fail-closed contract (``_restart_phase_failure_is_incomplete``) plus
    the stale-runtime rows decide the outcome.
    """
    if not planned_gateway_profiles:
        return False
    if not set(planned_gateway_profiles) <= set(covered_gateway_profiles):
        return False
    result = recovery_result or {}
    if result.get("failed") or result.get("relaunch_attempted"):
        return False
    if (result.get("serve_units") or {}).get("failed"):
        return False
    return not stale_runtime_rows
