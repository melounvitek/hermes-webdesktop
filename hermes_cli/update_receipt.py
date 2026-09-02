"""Structured update receipts + post-update fleet version verification.

Phase 1 of the fleet-update reliability plan (#91277): the updater must *prove* its outcome instead
of assuming it.

Two additive capabilities, both designed so a failure inside them can never break an update (every
public entry point is exception-swallowing):
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)

_RECEIPT_DIR_NAME = "update_receipts"
_RECEIPT_KEEP = 20  # keep the last N receipts per profile home

# Module-level current receipt. ``hermes update`` is a single-threaded CLI
# command; a module singleton lets the 7k-line updater record steps from
# any depth without threading a handle through every helper.
_current: Optional["UpdateReceipt"] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _str_records(entries: Any, keys: tuple[str, ...], *, pid: bool = False) -> list[dict[str, Any]]:
    """Dict entries reduced to stringified ``keys`` (plus an int ``pid`` first when requested)."""
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record: dict[str, Any] = {"pid": int(entry.get("pid", 0) or 0)} if pid else {}
        record.update({key: str(entry.get(key, "")) for key in keys})
        records.append(record)
    return records


class UpdateReceipt:
    """Collects the observable facts of one ``hermes update`` run."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "schema": 1,
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "argv": list(sys.argv),
            "pid": os.getpid(),
            "outcome": "running",  # running | success | partial | failed
            "pre_update": {},
            "post_update": {},
            "steps": [],
            "skips": [],
            "gateway_restart": {},
            "fleet": [],
        }
        try:
            from hermes_cli.build_info import get_code_identity

            self.data["pre_update"] = get_code_identity()
        except Exception:
            pass

    # -- recording ---------------------------------------------------------
    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.data["steps"].append(
            {"name": name, "ok": bool(ok), "detail": detail, "at": _utc_now_iso()}
        )

    def skip(self, name: str, reason: str) -> None:
        self.data["skips"].append(
            {"name": name, "reason": reason, "at": _utc_now_iso()}
        )

    def gateway_restart_result(
        self,
        *,
        restarted_services: list | None = None,
        relaunched_profiles: list | None = None,
        externally_supervised_profiles: list | None = None,
        killed_pids: list | None = None,
        failed_units: list | None = None,
        incomplete: bool = False,
        phase_error: str = "",
        fresh_recovery: dict[str, Any] | None = None,
    ) -> None:
        result: dict[str, Any] = {
            "restarted_services": list(restarted_services or []),
            "relaunched_profiles": list(relaunched_profiles or []),
            "externally_supervised_profiles": list(
                externally_supervised_profiles or []
            ),
            "killed_pids": [int(p) for p in (killed_pids or [])],
            "failed_units": [str(u) for u in (failed_units or [])],
            "incomplete": bool(incomplete),
            "phase_error": phase_error,
        }
        if fresh_recovery is not None:
            # Conservative outcome vocabulary: "verified" is the only bucket
            # allowed to claim supervisor coverage; "relaunch_attempted" means
            # the relaunch exited 0 without independent supervisor
            # observation. "skipped" preserves runtimes (manual gateways,
            # serve/dashboard entries) the pass deliberately did not touch.
            persisted: dict[str, Any] = {
                key: [str(profile) for profile in fresh_recovery.get(key, [])]
                for key in ("requested", "verified", "relaunch_attempted", "failed")
            }
            persisted["skipped"] = _str_records(
                fresh_recovery.get("skipped", []),
                ("profile", "kind", "supervisor", "reason"),
            )
            # Serve/dashboard coverage (#92145). ``hermes serve`` hosts
            # tui_gateway and is not a gateway profile, so neither the
            # per-profile buckets above nor the fleet-version matrix can
            # describe it. Persist its unit outcomes and any process that
            # survived on the pre-update generation, or the receipt keeps
            # claiming a clean recovery the operator's box contradicts.
            serve_units = fresh_recovery.get("serve_units") or {}
            persisted["serve_units"] = {
                key: [str(unit) for unit in (serve_units.get(key) or [])]
                for key in ("verified", "failed")
            }
            persisted["stale_runtimes"] = _str_records(
                fresh_recovery.get("stale_runtimes", []),
                ("kind", "profile", "supervisor"),
                pid=True,
            )
            result["fresh_recovery"] = persisted
        self.data["gateway_restart"] = result

    def finalize(self, outcome: str) -> None:
        self.data["outcome"] = outcome
        self.data["finished_at"] = _utc_now_iso()
        try:
            from hermes_cli.build_info import get_code_identity

            self.data["post_update"] = get_code_identity(refresh=True)
        except Exception:
            pass


def _receipt_dir() -> Path:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "logs" / _RECEIPT_DIR_NAME


def begin_update_receipt() -> None:
    """Start recording a new update receipt. Never raises."""
    global _current
    try:
        _current = UpdateReceipt()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not start update receipt: %s", exc)
        _current = None


def _record(method: str, what: str, *args: Any, **kwargs: Any) -> None:
    """Invoke ``method`` on the active receipt; no-op when none, never raises."""
    try:
        if _current is not None:
            getattr(_current, method)(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record %s: %s", what, exc)


def record_step(name: str, ok: bool, detail: str = "") -> None:
    """Record one update step outcome. No-op when no receipt is active."""
    _record("step", f"update step {name}", name, ok, detail)


def record_skip(name: str, reason: str) -> None:
    """Record a skipped step WITH the reason it was skipped."""
    _record("skip", f"update skip {name}", name, reason)


def record_gateway_restart(**kwargs: Any) -> None:
    """Record the gateway restart phase outcome (see UpdateReceipt)."""
    _record("gateway_restart_result", "gateway restart result", **kwargs)


def finalize_update_receipt(
    outcome: str, fleet: list | None = None, stop_reason: str = ""
) -> Optional[Path]:
    """Finalize + persist the receipt. Returns the written path or None.

    ``outcome`` is one of ``success`` / ``partial`` / ``failed`` / ``refused``. Exactly-once by
    construction: the module singleton is popped first, so a second call (e.g. the command-boundary
    safety net after an inner path already finalized) is a no-op returning None.
    """
    global _current
    receipt = _current
    _current = None
    if receipt is None:
        return None
    try:
        receipt.finalize(outcome)
        if stop_reason:
            receipt.data["stop_reason"] = stop_reason
        if fleet is not None:
            receipt.data["fleet"] = fleet
        directory = _receipt_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = directory / f"update_{stamp}_{os.getpid()}.json"
        body = json.dumps(receipt.data, indent=2, default=str)
        path.write_text(body, encoding="utf-8")
        # Stable pointer for the dashboard/desktop: latest receipt.
        try:
            (directory / "latest.json").write_text(body, encoding="utf-8")
        except OSError:
            pass
        _prune_old_receipts(directory)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not write update receipt: %s", exc)
        return None


def finalize_pending_update_receipt(
    exit_code: Optional[int] = None, stop_reason: str = ""
) -> Optional[Path]:
    """Command-boundary safety net: persist a still-open receipt, if any.

    ``hermes update`` has many early ``sys.exit`` paths (preflight refusals, venv-holder
    refusal, fetch failure) predating the inner finalize calls; any receipt still open when the
    command unwinds is finalized here so refused/failed runs — where a receipt matters most —
    leave a record. No-op when nothing is open (inner paths finalize exactly-once via the popped
    singleton). Never raises. Exit 0/None → ``success``, exit 2 → ``refused`` (preflight
    convention), else → ``failed``.
    """
    if _current is None:
        return None
    if exit_code in (0, None):
        outcome = "success"
    elif exit_code == 2:
        outcome = "refused"
    else:
        outcome = "failed"
    if exit_code is not None:
        try:
            _current.data["exit_code"] = int(exit_code)
        except Exception:
            pass
    return finalize_update_receipt(outcome, stop_reason=stop_reason)


def _prune_old_receipts(directory: Path) -> None:
    try:
        receipts = sorted(
            (p for p in directory.glob("update_*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in receipts[_RECEIPT_KEEP:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except Exception:
        pass


def read_latest_receipt() -> Optional[dict[str, Any]]:
    """Read the most recent update receipt, or None. Never raises."""
    try:
        path = _receipt_dir() / "latest.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fleet version verification
# ---------------------------------------------------------------------------

def _sha_state(code_sha: Any, expected_sha: Any) -> str:
    if not code_sha or not expected_sha:
        return "unknown"
    return "current" if str(code_sha) == str(expected_sha) else "stale"


def _fleet_row(profile: str, pid: int, code_sha: Any, code_version: Any, expected_sha: Any) -> dict[str, Any]:
    return {
        "profile": profile,
        "pid": pid,
        "code_sha": str(code_sha) if code_sha else None,
        "code_version": code_version,
        "state": _sha_state(code_sha, expected_sha),
    }


def collect_fleet_versions(
    *, pre_restart_pids: Optional[list[int]] = None
) -> list[dict[str, Any]]:
    """Snapshot every profile's gateway code identity vs. the current tree.

    Rollout safety: ``down`` requires membership in ``pre_restart_pids`` — a stale state file from a
    long-dead gateway (machine reboot, manual kill weeks ago) must NOT fail every future update.
    Callers that don't have a pre-restart snapshot (``None``/empty) get the historical behavior:
    dead PIDs are skipped.
    """
    # Runtime-status states that mean "this record does not describe a
    # gateway that should be running now" — no down row for these.
    _NOT_EXPECTED_STATES = {"stopped", "startup_failed"}
    _pre_restart = {int(p) for p in (pre_restart_pids or []) if isinstance(p, int)}
    results: list[dict[str, Any]] = []
    try:
        from hermes_cli.build_info import get_code_identity

        expected_sha = (get_code_identity(refresh=True) or {}).get("sha")
    except Exception:
        expected_sha = None

    try:
        from gateway.status import read_runtime_status, runtime_status_pid_is_live
        from hermes_cli.profiles import (
            _get_default_hermes_home,
            _get_profiles_root,
            _PROFILE_ID_RE,
        )

        homes: list[tuple[str, Path]] = []
        default_home = _get_default_hermes_home()
        if default_home.is_dir():
            homes.append(("default", default_home))
        profiles_root = _get_profiles_root()
        if profiles_root.is_dir():
            for entry in sorted(profiles_root.iterdir()):
                if entry.is_dir() and entry.name != "default" and _PROFILE_ID_RE.match(entry.name):
                    homes.append((entry.name, entry))

        for profile, home in homes:
            # Prefer the gateway-owned control socket (#92091): a live
            # `identify` answer is authoritative — no PID-reuse or stale-file
            # heuristics. Fall back to gateway_state.json for gateways that
            # predate the socket or whose socket didn't bind.
            identity = None
            try:
                from gateway.control_socket import identify_gateway

                identity = identify_gateway(home)
            except Exception:
                identity = None
            if identity:
                try:
                    pid = int(identity.get("pid"))
                except (TypeError, ValueError):
                    pid = None
                if pid is not None:
                    row = _fleet_row(
                        profile, pid, identity.get("code_sha"),
                        identity.get("code_version"), expected_sha,
                    )
                    row["source"] = "socket"
                    results.append(row)
                    continue
            status_path = home / "gateway_state.json"
            record = read_runtime_status(status_path)
            if not record:
                continue
            pid = record.get("pid")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            if not runtime_status_pid_is_live(record):
                # Dead PID (or a live PID recycled by an unrelated process
                # during the update's own churn — #93258): a DOWN row only
                # when this exact pid was alive at update start AND the
                # record still claims a running state — "the restart phase
                # stopped it and nothing came back." Everything else (clean
                # stop, startup failure, stale record from a long-dead
                # gateway) keeps the historical no-row behavior so the
                # feature's rollout can't false-positive.
                #
                # ``_pre_restart`` is a bare set of PIDs, not (pid, start_time)
                # pairs, so a recycled PID from gateway A landing in B's stale
                # record could still mislabel B as down if A's PID happened to
                # be in the pre-restart snapshot — inherent to the snapshot's
                # data model, not something this guard can fix on its own.
                gw_state = record.get("gateway_state")
                if (
                    pid in _pre_restart
                    and isinstance(gw_state, str)
                    and gw_state
                    and gw_state not in _NOT_EXPECTED_STATES
                ):
                    results.append(
                        {
                            "profile": profile,
                            "pid": pid,
                            "code_sha": None,
                            "code_version": record.get("code_version"),
                            "state": "down",
                        }
                    )
                continue
            results.append(
                _fleet_row(
                    profile, pid, record.get("code_sha"),
                    record.get("code_version"), expected_sha,
                )
            )
    except Exception as exc:
        logger.debug("Fleet version probe failed: %s", exc)
    return results


def print_fleet_version_matrix(fleet: list[dict[str, Any]]) -> bool:
    """Print the post-update fleet version matrix.

    Returns True when at least one gateway is provably stale (still serving pre-update code) OR
    provably down (killed by the restart phase, nothing came back), so the caller can escalate.
    ``unknown`` entries are reported but do NOT fail the update: gateways started before the
    code-identity stamp existed have no sha to compare, and failing them would be a false-
    positive storm.
    """
    if not fleet:
        return False
    any_stale = False
    any_down = False
    print()
    print("Fleet version check:")
    for entry in fleet:
        sha = entry.get("code_sha")
        short = sha[:8] if isinstance(sha, str) and sha else "?"
        state = entry.get("state")
        profile = entry.get("profile")
        pid = entry.get("pid")
        if state == "current":
            print(f"  ✓ {profile} (pid {pid}) @ {short} — up to date")
        elif state == "stale":
            any_stale = True
            print(f"  ✗ {profile} (pid {pid}) @ {short} — STALE (pre-update code)")
        elif state == "down":
            any_down = True
            print(
                f"  ✗ {profile} — DOWN (gateway was running before the "
                f"update; pid {pid} is gone and nothing replaced it)"
            )
        else:
            print(
                f"  ? {profile} (pid {pid}) — version unknown "
                "(gateway predates version stamping; restart to enable)"
            )
    if any_stale or any_down:
        print()
        if any_stale:
            print("  ⚠ Stale gateways keep serving pre-update code until restarted:")
        if any_down:
            print("  ⚠ Down gateways stopped serving messaging entirely — restart them:")
        print("      hermes gateway restart                # active profile")
        print("      hermes -p <profile> gateway restart   # named profile")
    return any_stale or any_down
