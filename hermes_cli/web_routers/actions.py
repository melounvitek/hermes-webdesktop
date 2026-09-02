"""Gateway restart/drain, Hermes update and background-action status dashboard routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import logging
import asyncio
import secrets
import subprocess
from fastapi import APIRouter
from hermes_cli.web_deps import late
from fastapi import HTTPException, Request
from hermes_cli import __version__
from hermes_cli.config import format_docker_update_message, recommended_update_command_for_method
from typing import Any, Dict, List, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()
status_router = APIRouter()

# web_server helpers, late-bound so monkeypatch.setattr(web_server, ...) stays authoritative.
_dashboard_local_update_managed_externally = late("_dashboard_local_update_managed_externally")
_durable_completed_update_action_id = late("_durable_completed_update_action_id")
_record_completed_action = late("_record_completed_action")
_spawn_gateway_restart = late("_spawn_gateway_restart")
_spawn_hermes_action = late("_spawn_hermes_action")
_tail_lines = late("_tail_lines")
detect_install_method = late("detect_install_method")
get_hermes_home = late("get_hermes_home")


@router.post("/api/gateway/restart")
async def restart_gateway(profile: Optional[str] = None):
    """Kick off a ``hermes gateway restart`` in the background."""
    try:
        proc, _reused = _spawn_gateway_restart(profile)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway restart")
        raise HTTPException(status_code=500, detail=f"Failed to restart gateway: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "gateway-restart",
    }


@router.post("/api/gateway/drain")
async def gateway_drain(request: Request):
    """Begin or cancel an external (NAS-driven) gateway drain.

    Authenticated by the non-interactive token-auth seam: the
    ``dashboard_auth/drain`` plugin registers this exact path as a token route
    and verifies the ``Authorization`` bearer secret. If that plugin isn't
    active (no ``HERMES_DASHBOARD_DRAIN_SECRET``), the route is NOT a token
    route, so on a gated bind the cookie gate handles it (a browser session can
    still drive it from the dashboard) and on a loopback bind the legacy
    session-token gate applies — either way it is never unauthenticated on a
    network-exposed bind.

    Body: ``{"action": "drain"}`` (begin) or ``{"action": "cancel"}`` (cancel).
    Begin writes the ``.drain_request.json`` marker the gateway's
    ``_drain_control_watcher`` observes (flip to ``draining`` + refuse new
    turns); cancel removes it (revert to ``running`` + re-accept). Idempotent
    on both sides. This endpoint only writes/removes the marker — the gateway
    process owns the actual state transition (there is no HTTP control channel
    into the running gateway; the marker IS the channel, decisions.md Q-B).

    The force-override (D6: "unless a user commands it") is NOT here — an
    immediate, drain-skipping action maps onto the existing
    ``POST /api/gateway/restart`` force path, which supersedes a drain.
    """
    from gateway.drain_control import (
        clear_drain_request,
        drain_requested,
        write_drain_request,
    )

    try:
        body = await request.json()
    except Exception:
        body = {}
    action = str((body or {}).get("action", "drain")).strip().lower()

    # Attribute the request to the verified token principal when present
    # (token-auth seam attaches it); fall back to a generic label otherwise.
    principal_obj = getattr(request.state, "token_principal", None)
    principal = getattr(principal_obj, "principal", None) or "dashboard"

    if action == "cancel":
        existed = clear_drain_request()
        _log.info("Gateway drain CANCEL requested by %s (existed=%s)", principal, existed)
        return {"ok": True, "action": "cancel", "was_draining": existed}

    if action != "drain":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown drain action {action!r}; expected 'drain' or 'cancel'",
        )

    payload = write_drain_request(
        principal=str(principal),
        suppress_notification=bool((body or {}).get("suppress_notification", False)),
    )
    _log.info(
        "Gateway drain BEGIN requested by %s (suppress_notification=%s)",
        principal,
        payload["suppress_notification"],
    )
    return {
        "ok": True,
        "action": "drain",
        "requested_at": payload["requested_at"],
        # Echo so a caller polling /api/status knows the marker is now set;
        # the gateway watcher flips gateway_state -> draining within ~1s.
        "draining": drain_requested(),
        "suppress_notification": payload["suppress_notification"],
    }


@router.post("/api/hermes/update")
async def update_hermes():
    """Kick off ``hermes update`` in the background."""
    from hermes_cli.web_server import PROJECT_ROOT, _ACTION_IDS, _ACTION_PROCS
    if _dashboard_local_update_managed_externally():
        message = (
            "Hermes updates are managed outside this dashboard in "
            "containerized environments. The built-in local updater is "
            "disabled here."
        )
        _record_completed_action("hermes-update", message, exit_code=1)
        return {
            "ok": False,
            "pid": None,
            "name": "hermes-update",
            "error": "dashboard_update_managed_externally",
            "message": message,
            "update_command": "managed outside dashboard",
        }

    # Shared admission gate (#91277 Phase 3): marker-first, then the
    # docker/nix/apt heuristics — one decision with the CLI paths. The
    # response keeps the pre-existing per-kind error codes the dashboard UI
    # already keys on.
    from hermes_cli.update_contract import (
        evaluate_update_admission,
        record_refusal_receipt,
    )

    refusal = evaluate_update_admission(PROJECT_ROOT)
    if refusal is not None:
        _record_completed_action("hermes-update", refusal.message, exit_code=1)
        record_refusal_receipt(refusal)
        error_code = {
            "docker": "docker_update_unsupported",
            "image-marker": "docker_update_unsupported",
            "image-marker-invalid": "docker_update_unsupported",
            "apt": "apt_update_required",
            "nix": "nix_update_unsupported",
        }.get(refusal.code, "update_not_in_place")
        return {
            "ok": False,
            "pid": None,
            "name": "hermes-update",
            "error": error_code,
            "message": refusal.message,
            "update_command": refusal.update_command,
        }

    existing = _ACTION_PROCS.get("hermes-update")
    if existing is not None and existing.poll() is None:
        response = {
            "ok": True,
            "pid": existing.pid,
            "name": "hermes-update",
            "already_running": True,
        }
        action_id = _ACTION_IDS.get("hermes-update")
        if action_id:
            response["action_id"] = action_id
        return response

    action_id = secrets.token_hex(16)
    try:
        proc = _spawn_hermes_action(
            ["update"],
            "hermes-update",
            env_overrides={"HERMES_ACTION_ID": action_id},
        )
    except Exception as exc:
        _log.exception("Failed to spawn hermes update")
        raise HTTPException(status_code=500, detail=f"Failed to start update: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "hermes-update",
        "action_id": action_id,
    }


def _recent_upstream_commits(n: int = 20) -> List[Dict[str, Any]]:
    """Commits the local checkout is behind ``origin/main`` by, newest first.

    Logs the SAME range the behind-count uses (``HEAD..origin/main`` — see
    ``banner._check_via_local_git``), NOT the branch's ``@{upstream}``. On a
    feature-branch checkout ``@{upstream}`` is the branch's own tip (zero
    commits), which would leave the changelog empty even though the count is
    non-zero. Pinning to ``origin/main`` keeps count and changelog consistent.

    Best-effort: returns [] if not a git checkout, origin/main is unreachable,
    or git is unavailable. Never raises into the request path.
    """
    from hermes_cli.web_server import PROJECT_ROOT
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "log",
                "--format=%H%x1f%s%x1f%an%x1f%ct",
                "HEAD..origin/main",
                f"-n{int(n)}",
            ],
            capture_output=True,
            text=True,
            # git log emits UTF-8 (commit subjects can carry emoji/CJK). On
            # Windows text=True defaults to the ANSI code page — a byte like
            # 0x90 (3rd byte of 🐛) is undefined in cp1252 and crashed the
            # stdlib _readerthread, killing the desktop backend (#52649).
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if out.returncode != 0:
            return []
        rows: List[Dict[str, Any]] = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            parts = (line.split("\x1f") + ["", "", "", "0"])[:4]
            sha, summary, author, at = parts
            rows.append(
                {
                    "sha": sha[:7],
                    "summary": summary,
                    "author": author,
                    "at": int(at or 0),
                }
            )
        return rows
    except Exception:
        return []


@router.get("/api/hermes/update/check")
async def check_hermes_update(force: bool = False):
    """Report whether a Hermes update is available, without applying it.

    Powers the dashboard's "check before you update" flow: the System page
    shows the commit-behind count and asks the user to confirm before
    ``POST /api/hermes/update`` actually runs ``hermes update``.

    Returns:
        install_method: 'apt' | 'git' | 'docker' | 'nix' | 'nixos' | 'unknown'
        current_version: installed Hermes version string
        behind: commits behind upstream (>=1), 0 if up to date,
                -1 if behind by an unknown count, or null if the
                check could not run (offline, no remote, etc.)
        update_available: convenience bool (behind is non-zero and not null)
        can_apply: True when the dashboard's update button can apply it
                   in place (git); False for other install methods where the
                   user must update out-of-band
        update_command: the recommended command for this install method
        message: human-readable guidance for non-applyable methods
        commits: for git installs that are behind, a list of the commits
                 the local checkout is behind upstream by — each
                 {sha, summary, author, at}. Absent/empty otherwise. The
                 desktop's remote update overlay renders this as "what's
                 changed". Additive: existing consumers ignore it.
    """
    from hermes_cli.web_server import PROJECT_ROOT
    if _dashboard_local_update_managed_externally():
        return {
            "install_method": "managed-runtime",
            "current_version": __version__,
            "behind": None,
            "update_available": False,
            "can_apply": False,
            "update_command": "managed outside dashboard",
            "message": (
                "Hermes updates are managed outside this dashboard in "
                "containerized environments."
            ),
        }

    install_method = detect_install_method(PROJECT_ROOT)
    update_command = recommended_update_command_for_method(install_method)

    payload: Dict[str, Any] = {
        "install_method": install_method,
        "current_version": __version__,
        "behind": None,
        "update_available": False,
        "can_apply": install_method == "git",
        "update_command": update_command,
        "message": None,
    }

    if install_method == "docker":
        payload["message"] = format_docker_update_message()
        return payload
    if install_method == "apt":
        payload["message"] = (
            "Hermes is managed by Termux APT; run `pkg upgrade hermes-agent`."
        )
        return payload

    # banner.check_for_updates() handles git / nix-revision paths and
    # caches the result for 6h. ``force`` busts the cache so the "Check now"
    # button reflects reality immediately.
    try:
        from hermes_cli.banner import check_for_updates

        if force:
            try:
                (get_hermes_home() / ".update_check").unlink()
            except OSError:
                pass

        behind = await asyncio.to_thread(check_for_updates)
    except Exception:
        _log.exception("Update check failed")
        behind = None

    payload["behind"] = behind
    if behind is None:
        payload["message"] = "Couldn't reach the update source — try again later."
    elif behind == 0:
        payload["message"] = "You're on the latest version."
    else:
        payload["update_available"] = True
        # Enrich with the actual commits we're behind by, so the desktop's
        # remote update overlay can show "what's changed". git only;
        # best-effort (empty list on any failure).
        if install_method == "git":
            payload["commits"] = await asyncio.to_thread(_recent_upstream_commits)

    return payload


@status_router.get("/api/actions/{name}/status")
async def get_action_status(name: str, lines: int = 200):
    """Tail an action log and report whether the process is still running."""
    from hermes_cli.web_server import (
        _ACTION_COMMANDS,
        _ACTION_IDS,
        _ACTION_LOG_DIR,
        _ACTION_LOG_FILES,
        _ACTION_PROCS,
        _ACTION_RESULTS,
    )
    log_file_name = _ACTION_LOG_FILES.get(name)
    if log_file_name is None:
        raise HTTPException(status_code=404, detail=f"Unknown action: {name}")

    log_path = _ACTION_LOG_DIR / log_file_name
    requested_lines = min(max(lines, 1), 2000)
    tail = _tail_lines(log_path, requested_lines)

    durable_update_action_id = None
    update_receipt_summary = None
    if name == "hermes-update":
        durable_lines = _tail_lines(_ACTION_LOG_DIR / "update.log", 2000)
        durable_update_action_id = _durable_completed_update_action_id(durable_lines)
        if durable_update_action_id:
            marker = f"=== hermes-update completed {durable_update_action_id} ==="
            if marker not in tail:
                tail = [*tail, marker][-requested_lines:]
        # Phase-1 bullet 3 (#91277): the update receipt is the durable,
        # structured truth about the last update — written by every run
        # including refused/failed ones, and it survives the dashboard
        # restarting itself mid-action. Surface its summary alongside the
        # log-marker recovery so clients (Desktop, dashboard) READ the
        # outcome instead of inferring it from liveness probes
        # (#81193/#87359 class).
        update_receipt_summary = _latest_update_receipt_summary()

    proc = _ACTION_PROCS.get(name)
    if proc is None:
        result = _ACTION_RESULTS.get(name)
        running = False
        exit_code = result.get("exit_code") if result else None
        pid = result.get("pid") if result else None
        if result is None and durable_update_action_id:
            exit_code = 0
        if (
            result is None
            and exit_code is None
            and update_receipt_summary is not None
            and update_receipt_summary.get("outcome") in ("success", "partial")
        ):
            # No in-memory result and no log marker (e.g. log rotated), but
            # the receipt proves a completed run: report its outcome rather
            # than a null that clients time out on. ``partial`` maps to
            # exit 1 exactly like the CLI run itself did.
            exit_code = 0 if update_receipt_summary["outcome"] == "success" else 1
    else:
        exit_code = proc.poll()
        running = exit_code is None
        pid = proc.pid
        if exit_code is not None:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            _ACTION_RESULTS[name] = {"exit_code": exit_code, "pid": pid}
            _ACTION_PROCS.pop(name, None)
            _ACTION_COMMANDS.pop(name, None)
            _ACTION_IDS.pop(name, None)

    response = {
        "name": name,
        "running": running,
        "exit_code": exit_code,
        "pid": pid,
        "lines": tail,
    }
    if durable_update_action_id:
        response["action_id"] = durable_update_action_id
    if update_receipt_summary is not None:
        response["receipt"] = update_receipt_summary
    return response


def _latest_update_receipt_summary() -> Optional[Dict[str, Any]]:
    """Compact summary of the most recent update receipt, or None.

    Phase-1 bullet 3 (#91277): the receipt (written by EVERY ``hermes
    update`` run since #91283, including refused and failed ones, with a
    ``latest.json`` pointer) is the durable success signal the Desktop and
    dashboard should read instead of inferring outcomes from liveness
    probes across the update's stop/start gap (#81193, #87359). Summary
    only — steps and skips stay in the full receipt endpoint.
    Never raises.
    """
    try:
        from hermes_cli.update_receipt import read_latest_receipt

        receipt = read_latest_receipt()
        if not receipt:
            return None
        fleet = receipt.get("fleet") or []
        return {
            "outcome": receipt.get("outcome"),
            "started_at": receipt.get("started_at"),
            "finished_at": receipt.get("finished_at"),
            "pre_sha": (receipt.get("pre_update") or {}).get("sha"),
            "post_sha": (receipt.get("post_update") or {}).get("sha"),
            "post_version": (receipt.get("post_update") or {}).get("version"),
            "fleet_states": sorted(
                {str(e.get("state")) for e in fleet if isinstance(e, dict)}
            ),
        }
    except Exception:
        return None


@status_router.get("/api/hermes/update/receipt")
async def get_update_receipt():
    """The most recent update receipt — the durable update-outcome record.

    Phase-1 bullet 3 (#91277): dashboards and the Desktop read this instead
    of inferring update success from backend liveness (the inference misread
    the update's own restart gap as 'Backend update failed' / 'boot failed'
    — #81193, #87359). Returns the FULL receipt (steps, skips, gateway
    restart outcome, fleet matrix) plus a compact ``summary``; 404 when no
    update has run since receipts landed.
    """
    try:
        from hermes_cli.update_receipt import read_latest_receipt

        receipt = read_latest_receipt()
    except Exception:
        receipt = None
    if not receipt:
        raise HTTPException(
            status_code=404,
            detail="No update receipt found (no `hermes update` run recorded).",
        )
    return {"receipt": receipt, "summary": _latest_update_receipt_summary()}
