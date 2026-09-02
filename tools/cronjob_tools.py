"""
Cron job management tools for Hermes Agent.

Expose a single compressed action-oriented tool to avoid schema/context bloat.
Compatibility wrappers remain for direct Python callers and legacy tests.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)

# Heartbeat cadence that keeps the calling agent's inactivity watchdog at bay
# while a manual `cronjob(action="run")` executes synchronously in-process
# (mirrors tools/environments/base.py::touch_activity_if_due).
_CRON_RUN_HEARTBEAT_INTERVAL = 10.0

# Hard ceiling on the heartbeat: with HERMES_CRON_TIMEOUT=0 (unlimited) a truly
# hung run would otherwise mask the gateway watchdog forever. Past this, the
# heartbeat stops and the gateway watchdog regains authority over the turn.
_CRON_RUN_HEARTBEAT_CEILING = 6 * 3600.0

sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import (
    AmbiguousJobReference,
    claim_job_for_fire,
    get_job,
    is_job_runnable,
    list_jobs,
    mark_job_run,
    parse_schedule,
    pause_job,
    remove_job,
    resolve_job_ref,
    resume_job,
    update_job,
)
from tools.cronjob_prompt_scan import (  # noqa: F401  (re-exported)
    _CRON_EXFIL_COMMAND_PATTERNS,
    _CRON_INVISIBLE_CHARS,
    _CRON_SKILL_ASSEMBLED_PATTERNS,
    _CRON_THREAT_PATTERNS,
    _scan_cron_prompt,
    _scan_cron_skill_assembled,
)
from tools.cronjob_job_args import (  # noqa: F401  (re-exported)
    _apply_continuity,
    _canonical_skills,
    _format_job,
    _gateway_liveness_notice,
    _local_delivery_notice,
    _mode_guidance_notes,
    _normalize_deliver_param,
    _normalize_optional_job_value,
    _origin_from_env,
    _repeat_display,
    _resolve_cron_context_deliver,
    _split_monitor_arg,
    _validate_bot_chat_deliver,
    _validate_context_from_refs,
    _validate_cron_base_url,
    _validate_cron_script_path,
)


def _notify_provider_jobs_changed_safe() -> None:
    """Tell the active cron scheduler provider the job set changed (no-op for
    the built-in). Best-effort — never lets a provider error break the tool."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Manual run execution (claim -> run_one_job -> report)
# ---------------------------------------------------------------------------

def _relay_fronted_delivery_platforms(job: Dict[str, Any]) -> set:
    """Delivery-platform names for this job that the relay connector fronts."""
    try:
        from gateway.relay import relay_fronted_platforms
    except Exception:
        return set()
    fronted = relay_fronted_platforms()
    if not fronted:
        return set()
    try:
        from cron.scheduler import _resolve_delivery_targets

        targets = _resolve_delivery_targets(job) or []
    except Exception:
        return set()
    return {t.get("platform") for t in targets if t.get("platform")} & fronted


def _forward_relay_fronted_run(
    job: Dict[str, Any], extra_prompt: Optional[str] = None
) -> Optional[str]:
    """Forward a manual run to the gateway when it targets a relay-fronted
    platform and this process has no live relay adapter.

    Relay-fronted delivery has no standalone sender — the gateway's live relay
    adapter is the only path, reached via ``POST /api/jobs/{id}/run`` (which
    marks the job due for the gateway ticker; ``extra_prompt`` rides in the
    body). Returns a JSON result string when forwarding engages, else None to
    fall through to the normal in-process run.
    """
    if not _relay_fronted_delivery_platforms(job):
        return None
    job_id = job["id"]
    import os

    port_raw = os.getenv("API_SERVER_PORT", "").strip()
    try:
        port = int(port_raw) if port_raw else 8642
    except ValueError:
        port = 8642
    # Mirror the api_server's bind resolution (extra.host -> API_SERVER_HOST
    # -> 127.0.0.1); a wildcard bind listens on loopback too.
    host = ""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        host = str(
            cfg_get(
                load_config_readonly(), "platforms", "api_server", "extra", "host",
                default="",
            )
            or ""
        ).strip()
    except Exception:
        host = ""
    if not host:
        host = os.getenv("API_SERVER_HOST", "").strip()
    if not host or host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}/api/jobs/{job_id}/run"

    from agent.secret_scope import get_secret

    key = get_secret("API_SERVER_KEY", "") or ""

    resp = None
    try:
        import httpx

        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=({"prompt": extra_prompt} if extra_prompt else {}),
            timeout=10.0,
        )
    except Exception:
        resp = None

    if resp is not None and resp.status_code < 300:
        return json.dumps(
            {
                "success": True,
                "forwarded_to_gateway": True,
                "note": (
                    "This job targets a relay-fronted platform; it was dispatched "
                    "to the running gateway, whose live relay adapter owns that "
                    "delivery."
                ),
            },
            indent=2,
        )
    return json.dumps(
        {
            "success": False,
            "error": (
                "This job targets a relay-fronted platform, which has no "
                "standalone sender. Start the gateway — its ticker will "
                "deliver the job on schedule via the live relay adapter."
            ),
        },
        indent=2,
    )


def _manual_run_delivery_note(deliver: str, refreshed: Dict[str, Any]) -> str:
    """Parenthetical delivery note for a manual run's completion summary,
    following the refreshed record's ``last_delivery_error`` so the summary
    never claims success over a failed post-run delivery."""
    # Falsy deliver ("", stored JSON null) is normalized to "local" at fire
    # time -> read as saved-locally. Whitespace-only values fall through to
    # the error check so the fire-time "no delivery target" error surfaces.
    if not deliver or deliver == "local":
        return " (output saved locally only)"
    err = str(refreshed.get("last_delivery_error") or "").strip()
    if not err:
        return " (output was delivered there by the job itself)"
    return f" (⚠ delivery FAILED: {err[:200]})"


_ALREADY_RUNNING_ERROR = (
    "Job is already running (a scheduler tick or another "
    "manual run is executing it); not started again."
)


def _claim_for_manual_run(job_id: str, log_label: str):
    """At-most-once claim shared by the sync and background run paths.

    Returns ``(claimed_job, None)`` on success, else ``(None, error_dict)``
    where the dict has the ``_execute_job_now`` result shape. A lost claim is
    labelled precisely: claim_job_for_fire also returns False for paused /
    disabled / missing jobs, which must not read as "already being fired".
    """
    try:
        claimed_job = claim_job_for_fire(job_id, return_job=True)
        if isinstance(claimed_job, dict):
            return claimed_job, None
        refreshed = get_job(job_id)
        if refreshed is None:
            reason = "Job no longer exists; nothing to run."
        elif not is_job_runnable(refreshed):
            reason = "Job is paused/disabled; resume it before running."
        else:
            reason = "Job is already being fired by the scheduler; not run again."
        return None, {"claimed": False, "success": False, "error": reason}
    except Exception as e:
        logger.error("Failed to claim cron job %s for %s: %s", job_id, log_label, e)
        try:
            mark_job_run(job_id, False, str(e))
        except Exception:
            pass
        return None, {"claimed": True, "success": False, "error": str(e)}


def _execute_job_now(
    job: Dict[str, Any], extra_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Execute a cron job immediately, outside the scheduler tick.

    Claims first via ``claim_job_for_fire`` (the same CAS the ticker uses, so a
    concurrent tick cannot double-fire and next_run_at advances), then fires
    through the shared ``run_one_job`` body so delivery / [SILENT] handling
    cannot drift between paths.
    Returns {"claimed": bool, "success": bool, "error": str|None}.
    """
    claimed_job, err = _claim_for_manual_run(job["id"], "immediate run")
    if err is not None:
        return err
    return _run_claimed_job(claimed_job, extra_prompt=extra_prompt)


def _run_claimed_job(
    job: Dict[str, Any], extra_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Fire an already-claimed job through the shared ``run_one_job`` body.

    Split from ``_execute_job_now`` so the background path can take the claim
    synchronously (reporting paused/already-firing immediately) and hand the
    run to a worker. Returns {"claimed": True, "success": bool, "error": ...}.
    """
    job_id = job["id"]
    _registered = False
    fire_owner = None
    try:
        from cron.scheduler import (
            release_running_job,
            run_one_job,
            try_register_running_job,
        )

        # In-flight dedupe: the fire claim's TTL is routinely outlived by real
        # jobs, so register in the scheduler's shared running set (same guard
        # the ticker uses; also visible to the gateway shutdown drain).
        if not try_register_running_job(job_id):
            return {"claimed": True, "success": False, "error": _ALREADY_RUNNING_ERROR}
        _registered = True

        claim = job.get("fire_claim")
        fire_owner = str(claim.get("by") or "") if isinstance(claim, dict) else None

        # A manual run executes synchronously on the caller's thread and can
        # take minutes; without tool activity the gateway inactivity watchdog
        # would kill the parent turn. Heartbeat into the caller's activity
        # tracker while the job runs (best-effort: no callback -> unchanged).
        try:
            from tools.environments.base import get_activity_callback

            # Capture on THIS thread: the callback is thread-local.
            activity_cb = get_activity_callback()
        except Exception:
            activity_cb = None

        _heartbeat_stop = threading.Event()
        _heartbeat_thread = None

        if activity_cb is not None:
            job_name = str(job.get("name") or job_id)

            def _heartbeat_loop() -> None:
                started = time.monotonic()
                while not _heartbeat_stop.wait(_CRON_RUN_HEARTBEAT_INTERVAL):
                    elapsed = time.monotonic() - started
                    if elapsed > _CRON_RUN_HEARTBEAT_CEILING:
                        logger.warning(
                            "cronjob run heartbeat ceiling reached for job "
                            "'%s' (%.0fs) — stopping heartbeat; gateway "
                            "watchdog regains authority",
                            job_name, elapsed,
                        )
                        return
                    try:
                        activity_cb(
                            f"cronjob: running job '{job_name}' ({int(elapsed)}s elapsed)"
                        )
                    except Exception:
                        continue  # one transient callback error must not drop protection

            _heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                daemon=True,
                name="cronjob-run-heartbeat",
            )
            _heartbeat_thread.start()

        # Manual runs from a gateway agent share the process with live platform
        # adapters: pass the gateway adapter map + event loop so delivery runs
        # on the loop that owns clients such as Matrix/aiohttp (a standalone
        # asyncio.run() loop breaks them).
        gateway_module = sys.modules.get("gateway.run")
        runner_ref = getattr(gateway_module, "_gateway_runner_ref", None)
        runner = runner_ref() if callable(runner_ref) else None
        adapters = getattr(runner, "adapters", None) if runner is not None else None
        gateway_loop = getattr(runner, "_gateway_loop", None) if runner is not None else None

        try:
            try:
                # run_one_job records last_run_at/last_status via mark_job_run;
                # `job` is the owner-bearing claimed snapshot, so terminal writes
                # stay fenced by that owner.
                processed = run_one_job(
                    job, adapters=adapters, loop=gateway_loop,
                    extra_prompt=extra_prompt,
                )
            finally:
                _heartbeat_stop.set()
                if _heartbeat_thread is not None:
                    _heartbeat_thread.join(timeout=_CRON_RUN_HEARTBEAT_INTERVAL + 1)
        finally:
            _registered = False
            release_running_job(job_id)
        refreshed = get_job(job_id) or {}
        last_status = refreshed.get("last_status")
        # "delivery_failed": the run succeeded but output never reached the
        # user — not a success for the caller; surface last_delivery_error.
        run_error = refreshed.get("last_error")
        if last_status == "delivery_failed" and not run_error:
            run_error = refreshed.get("last_delivery_error")
        return {
            "claimed": True,
            "success": bool(processed and last_status == "ok"),
            "error": run_error,
        }

    except Exception as e:
        logger.error("Failed to execute cron job %s immediately: %s", job_id, e)
        if _registered:
            # Only release registrations WE took — a bare discard could erase
            # a ticker-owned entry.
            try:
                from cron.scheduler import release_running_job as _release

                _release(job_id)
            except Exception:
                pass
        try:
            mark_job_run(
                job_id,
                False,
                str(e),
                expected_fire_owner=fire_owner,
            )
        except Exception:
            pass
        return {"claimed": True, "success": False, "error": str(e)}


def _latest_job_output_excerpt(job_id: str, max_chars: int = 2000) -> Optional[str]:
    """Best-effort excerpt of the job's most recent saved output file. Never raises."""
    try:
        from cron.jobs import get_cron_output_dir

        out_dir = get_cron_output_dir() / job_id
        files = sorted(out_dir.glob("*.md"))
        if not files:
            return None
        text = files[-1].read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return None
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n… (truncated; full output: {files[-1]})"
        return text
    except Exception:
        return None


def _try_dispatch_background_run(
    job: Dict[str, Any], session_id: Optional[str] = None,
    extra_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Claim ``job`` now, then fire it on the async-delegation daemon executor.

    A cron job is a full agent run (minutes to hours); running it inline made
    the parent turn uninterruptible and serialized batches. This dispatches
    like ``delegate_task``'s background mode: the tool returns a handle, and a
    ``type="async_delegation"`` completion event re-enters the conversation as
    a fresh turn (keeps role alternation legal and the prompt cache intact).
    The claim is taken SYNCHRONOUSLY so unrunnable jobs report immediately.

    Returns None when background delivery is unavailable on this runtime
    (caller falls back to the sync path); ``{"claimed": False, ...}`` on a lost
    claim; ``{"claimed": True, "dispatched": True, "delegation_id": ...}`` when
    running in the background; ``{"claimed": True, "dispatched": False, ...}``
    when the pool was full and the run executed inline (claim already taken).
    """
    # Finite sessions cannot route a detached result back after the turn ends.
    try:
        from gateway.session_context import async_delivery_supported

        if not async_delivery_supported():
            return None
    except Exception:
        pass

    job_id = job["id"]
    job_name = str(job.get("name") or job_id)

    # Reap execution rows left 'claimed'/'running' by a dead owner process
    # (e.g. a prior one-shot `hermes cron run` that exited mid-run). The
    # ticker does this at its own startup; one-shot invocations have no such
    # moment, so a stale claim would block every later manual run. Only
    # provably-dead owners are reaped.
    try:
        from cron.executions import recover_interrupted_executions

        _reclaimed = recover_interrupted_executions()
        if _reclaimed:
            logger.warning(
                "Reclaimed %d stale cron execution(s) from dead owner(s) "
                "before dispatching job '%s'",
                _reclaimed,
                job_name,
            )
    except Exception as _reap_exc:
        logger.debug("Stale execution reclaim failed: %s", _reap_exc)

    # Routing capture on THIS thread (contextvars don't cross the pool), and
    # BEFORE the claim: with no routable session there is no durable consumer
    # for a detached completion, so we must not claim-and-dispatch.
    try:
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""
    if not session_key and session_id:
        # CLI path: the approval contextvar is only bound during gateway/TUI
        # turns; the CLI drain filters completions by the durable session id.
        session_key = str(session_id)
    if not session_key:
        # Direct Python callers (`hermes cron run`, tests): process exits right
        # after the tool returns, so run synchronously.
        return None

    # Best-effort early dedupe so a mid-run job reports in THIS tool response
    # instead of as a delayed error completion. The authoritative (atomic)
    # check is try_register_running_job inside _run_claimed_job.
    try:
        from cron.scheduler import get_running_job_ids

        if job_id in get_running_job_ids():
            return {"claimed": False, "success": False, "error": _ALREADY_RUNNING_ERROR}
    except Exception:
        pass

    claimed_job, err = _claim_for_manual_run(job_id, "background run")
    if err is not None:
        if err["claimed"]:
            err["dispatched"] = False
        return err

    origin_ui_session_id = ""
    try:
        from gateway.session_context import get_session_env

        origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or ""
    except Exception:
        pass

    try:
        from tools.async_delegation import (
            _current_origin_session_id,
            dispatch_async_delegation,
        )

        origin_session_id = _current_origin_session_id()
    except Exception as e:
        logger.warning(
            "cronjob run: async delegation registry unavailable (%s); "
            "running job '%s' inline.", e, job_name,
        )
        result = _run_claimed_job(claimed_job, extra_prompt=extra_prompt)
        result["dispatched"] = False
        return result

    try:
        from tools.delegate_tool import _get_max_async_children

        max_async = _get_max_async_children()
    except Exception:
        max_async = 3

    started_at = time.time()
    # Canonicalize with the scheduler's own normalizer (falsy -> "local", list
    # -> comma string), reading the claimed snapshot the run actually executes.
    from cron.scheduler import _normalize_deliver_value

    deliver = _normalize_deliver_value(claimed_job.get("deliver", "local"))

    def _runner() -> Dict[str, Any]:
        res = _run_claimed_job(claimed_job, extra_prompt=extra_prompt)
        duration = round(time.time() - started_at, 2)
        refreshed = get_job(job_id) or {}
        lines = [
            f"Cron job '{job_name}' ({job_id}) finished its manual run.",
            f"Result: {'ok' if res.get('success') else 'FAILED'}"
            + (f" — {res.get('error')}" if res.get("error") else ""),
            f"Delivery target: {deliver}"
            + _manual_run_delivery_note(deliver, refreshed),
        ]
        if refreshed.get("next_run_at"):
            lines.append(f"Next scheduled run: {refreshed['next_run_at']}")
        excerpt = _latest_job_output_excerpt(job_id)
        if excerpt:
            lines.append("--- JOB OUTPUT ---")
            lines.append(excerpt)
        return {
            "status": "completed" if res.get("success") else "error",
            "summary": "\n".join(lines),
            "error": res.get("error"),
            "api_calls": 0,
            "duration_seconds": duration,
        }

    dispatch = dispatch_async_delegation(
        goal=f"Manual run of cron job '{job_name}' ({job_id})",
        context=(
            "Triggered via cronjob(action='run'). The job executed in its own "
            "fresh cron session; this block reports its outcome."
        ),
        toolsets=None,
        role="cron_run",
        model=job.get("model"),
        session_key=session_key,
        parent_session_id=str(session_id) if session_id else None,
        runner=_runner,
        origin_ui_session_id=origin_ui_session_id,
        origin_session_id=origin_session_id,
        max_async_children=max_async,
    )

    if dispatch.get("status") == "dispatched":
        return {
            "claimed": True,
            "dispatched": True,
            "delegation_id": dispatch.get("delegation_id"),
        }

    # Pool at capacity (or submit failure): the claim is already taken and
    # must not be stranded — run inline exactly as the legacy path did.
    logger.info(
        "cronjob run: background pool unavailable (%s); running job '%s' inline.",
        dispatch.get("error", "rejected"), job_name,
    )
    result = _run_claimed_job(job, extra_prompt=extra_prompt)
    result["dispatched"] = False
    return result


# ---------------------------------------------------------------------------
# Tool actions. Each takes the cronjob() argument dict `a` (and the resolved
# job record for job-bound actions) and returns the JSON result string.
# ---------------------------------------------------------------------------

def _dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _action_create(a: Dict[str, Any]) -> str:
    prompt, script, deliver = a["prompt"], a["script"], a["deliver"]
    if not a["schedule"]:
        return tool_error("schedule is required for create", success=False)
    canonical_skills = _canonical_skills(a["skill"], a["skills"])
    _no_agent = bool(a["no_agent"])
    # no_agent=True -> the script IS the job (prompt/skills optional);
    # otherwise at least one of prompt/skills is required.
    if _no_agent:
        if not script:
            return tool_error(
                "create with no_agent=True requires a script — "
                "the script is the job. In no_agent mode the LLM is "
                "skipped entirely: prompt and skills are ignored, "
                "non-empty stdout is delivered verbatim, empty stdout "
                "sends nothing (watchdog pattern), and a non-zero "
                "exit or timeout sends an error alert.",
                success=False,
            )
    elif not prompt and not canonical_skills:
        return tool_error("create requires either prompt or at least one skill", success=False)
    error = (
        (prompt and _scan_cron_prompt(prompt))
        or (script and _validate_cron_script_path(script))
        or (a["monitor_script"] and _validate_cron_script_path(a["monitor_script"]))
        # A model-supplied base_url must not route a named provider's stored
        # credential to an attacker endpoint.
        or _validate_cron_base_url(a["provider"], a["base_url"])
        # bot-chat targets are machine-local: fail the CREATE, not the run.
        or _validate_bot_chat_deliver(_normalize_deliver_param(deliver))
        # failure_deliver shares deliver's grammar and validators (NS-788).
        or _validate_bot_chat_deliver(_normalize_deliver_param(a["failure_deliver"]))
        or (a["context_from"] and _validate_context_from_refs(
            [a["context_from"]] if isinstance(a["context_from"], str) else a["context_from"]
        ))
    )
    if error:
        return tool_error(error, success=False)

    context_from = a["context_from"]
    if a["continuity"] is not None:
        context_from = _apply_continuity(context_from, a["continuity"])

    from cron.scheduler import (
        CronSchedulerRegistrationError,
        create_job_with_scheduler_registration,
    )

    try:
        job = create_job_with_scheduler_registration(
            prompt=prompt or "",
            schedule=a["schedule"],
            name=a["name"],
            repeat=a["repeat"],
            deliver=_resolve_cron_context_deliver(_normalize_deliver_param(deliver)),
            origin=_origin_from_env(),
            skills=canonical_skills,
            model=_normalize_optional_job_value(a["model"]),
            provider=_normalize_optional_job_value(a["provider"]),
            base_url=_normalize_optional_job_value(a["base_url"], strip_trailing_slash=True),
            script=_normalize_optional_job_value(script),
            context_from=context_from,
            enabled_toolsets=a["enabled_toolsets"] or None,
            workdir=_normalize_optional_job_value(a["workdir"]),
            no_agent=_no_agent,
            attach_to_session=a["attach_to_session"],
            monitor_script=_normalize_optional_job_value(a["monitor_script"]),
            monitor_url=_normalize_optional_job_value(a["monitor_url"]),
            # CLI-only lane: deliberately absent from CRONJOB_SCHEMA and the
            # model dispatch — models do not make model-config decisions.
            reasoning_effort=a["reasoning_effort"],
            failure_deliver=_resolve_cron_context_deliver(
                _normalize_deliver_param(a["failure_deliver"])
            ),
        )
    except CronSchedulerRegistrationError as exc:
        _partial = exc.to_dict()
        return tool_error(_partial.pop("error"), success=False, **_partial)
    _create_message = f"Cron job '{job['name']}' created."
    _local_notice = _local_delivery_notice(job, _normalize_deliver_param(deliver))
    if _local_notice:
        _create_message = f"{_create_message} {_local_notice}"
    # A job created with no gateway running is stored but never fires — tell
    # the model, which otherwise reports a clean success.
    _result = {
        "success": True,
        "job_id": job["id"],
        "name": job["name"],
        "skill": job.get("skill"),
        "skills": job.get("skills", []),
        "schedule": job["schedule_display"],
        "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"),
        "next_run_at": job["next_run_at"],
        "job": _format_job(job),
        "message": _create_message,
        **_gateway_liveness_notice(),
    }
    _notes = _mode_guidance_notes(job, _normalize_deliver_param(deliver))
    if _notes:
        _result["guidance"] = _notes
    return _dumps(_result)


def _action_list(a: Dict[str, Any]) -> str:
    jobs = [_format_job(job) for job in list_jobs(include_disabled=a["include_disabled"])]
    _result = {"success": True, "count": len(jobs), "jobs": jobs}
    # Same inert-job class as create; an empty list has nothing inert.
    if jobs:
        _result.update(_gateway_liveness_notice(plural=True))
    return _dumps(_result)


def _action_remove(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    job_id = job["id"]
    if not remove_job(job_id):
        return tool_error(f"Failed to remove job '{job_id}'", success=False)
    _notify_provider_jobs_changed_safe()
    return _dumps({
        "success": True,
        "message": f"Cron job '{job['name']}' removed.",
        "removed_job": {
            "id": job_id,
            "name": job["name"],
            "schedule": job.get("schedule_display"),
        },
    })


def _action_pause(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    updated = pause_job(job["id"], reason=a["reason"])
    _notify_provider_jobs_changed_safe()
    return _dumps({"success": True, "job": _format_job(updated)})


def _action_resume(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    updated = resume_job(job["id"])
    _notify_provider_jobs_changed_safe()
    return _dumps({"success": True, "job": _format_job(updated)})


def _action_run(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    job_id = job["id"]
    # `prompt` on run is transient per-fire context appended to the stored
    # prompt, never persisted; same strict scan as stored prompts.
    extra_prompt = a["prompt"] or None
    if extra_prompt:
        scan_error = _scan_cron_prompt(extra_prompt)
        if scan_error:
            return tool_error(scan_error, success=False)
    # A manual run must actually run even with no ticker active. Preferred:
    # background dispatch (handle now, outcome as a completion event); falls
    # back to inline execution when the runtime can't receive completions.
    bg = _try_dispatch_background_run(
        job, session_id=a["session_id"], extra_prompt=extra_prompt
    )
    if bg is not None and bg.get("dispatched"):
        _notify_provider_jobs_changed_safe()
        result = _format_job(get_job(job_id) or {"id": job_id})
        result["executed"] = True
        result["execution_mode"] = "background"
        result["delegation_id"] = bg.get("delegation_id")
        return _dumps({
            "success": True,
            "job": result,
            "note": (
                "The job is running in the background. You and the "
                "user can keep working; its outcome re-enters the "
                "conversation as a new message when it finishes. "
                "Do not wait or poll — just continue."
            ),
        })
    if bg is not None:
        exec_result = bg  # terminal result: claim lost or inline fallback
    else:
        # Relay-fronted manual run: no live adapter here — forward to the
        # running gateway, whose adapter owns that delivery.
        forwarded = _forward_relay_fronted_run(job, extra_prompt=extra_prompt)
        if forwarded is not None:
            return forwarded
        exec_result = _execute_job_now(job, extra_prompt=extra_prompt)
    # A claimed direct run advances next_run_at and may race an external
    # provider's one-shot for the same occurrence — reconcile after the run.
    claimed = exec_result.get("claimed", False)
    if claimed:
        _notify_provider_jobs_changed_safe()
    # Re-read so the response reflects the post-run last_run_at/last_status.
    result = _format_job(get_job(job_id) or {"id": job_id})
    result["executed"] = claimed
    result["execution_success"] = exec_result.get("success", False)
    if not claimed:
        result["execution_skipped"] = exec_result.get("error") or (
            "Already being fired by the scheduler; not run again."
        )
    elif exec_result.get("error"):
        result["execution_error"] = exec_result["error"]
    return _dumps({"success": True, "job": result})


def _action_update(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    job_id = job["id"]
    updates: Dict[str, Any] = {}
    prompt, deliver, skill, skills = a["prompt"], a["deliver"], a["skill"], a["skills"]
    script, monitor_script, monitor_url = a["script"], a["monitor_script"], a["monitor_url"]
    context_from, continuity, no_agent = a["context_from"], a["continuity"], a["no_agent"]
    if prompt is not None:
        scan_error = _scan_cron_prompt(prompt)
        if scan_error:
            return tool_error(scan_error, success=False)
        updates["prompt"] = prompt
    if a["name"] is not None and a["name"].strip():
        # Blank name is a no-op, not a clear: a model re-sending the whole
        # schema with type-default empties must not wipe untouched fields.
        updates["name"] = a["name"]
    if deliver is not None:
        bot_chat_error = _validate_bot_chat_deliver(_normalize_deliver_param(deliver))
        if bot_chat_error:
            return tool_error(bot_chat_error, success=False)
        updates["deliver"] = _resolve_cron_context_deliver(_normalize_deliver_param(deliver))
    if a["failure_deliver"] is not None:
        # '' clears the override (job falls back to deliver on failures);
        # non-empty values share deliver's validation AND its cron-context
        # origin resolution (a job created from inside a cron run must never
        # store literal 'origin' — same rule as deliver).
        _norm_fd = _normalize_deliver_param(a["failure_deliver"])
        if _norm_fd:
            bot_chat_error = _validate_bot_chat_deliver(_norm_fd)
            if bot_chat_error:
                return tool_error(bot_chat_error, success=False)
            _norm_fd = _resolve_cron_context_deliver(_norm_fd)
        updates["failure_deliver"] = _norm_fd
    if skills is not None or skill is not None:
        canonical_skills = _canonical_skills(skill, skills)
        updates["skills"] = canonical_skills
        updates["skill"] = canonical_skills[0] if canonical_skills else None
    if a["model"] is not None:
        updates["model"] = _normalize_optional_job_value(a["model"])
    if a["provider"] is not None:
        updates["provider"] = _normalize_optional_job_value(a["provider"])
    if a["base_url"] is not None:
        updates["base_url"] = _normalize_optional_job_value(a["base_url"], strip_trailing_slash=True)
    if a["reasoning_effort"] is not None:
        # CLI-only lane; update_job validates, empty string clears the pin.
        updates["reasoning_effort"] = a["reasoning_effort"]
    # Re-validate the EFFECTIVE provider/base_url on EVERY update: a job
    # persisted before this guard may already hold an unsafe pair, and editing
    # an unrelated field must not leave it schedulable.
    base_url_error = _validate_cron_base_url(
        updates["provider"] if "provider" in updates else job.get("provider"),
        updates["base_url"] if "base_url" in updates else job.get("base_url"),
    )
    if base_url_error:
        return tool_error(base_url_error, success=False)
    # Empty string clears script / monitor fields.
    for field, value in (("script", script), ("monitor_script", monitor_script)):
        if value is not None:
            if value:
                path_error = _validate_cron_script_path(value)
                if path_error:
                    return tool_error(path_error, success=False)
            updates[field] = _normalize_optional_job_value(value) if value else None
    if monitor_url is not None:
        updates["monitor_url"] = _normalize_optional_job_value(monitor_url) if monitor_url else None
    if monitor_script is not None or monitor_url is not None:
        eff_mon_script = updates["monitor_script"] if "monitor_script" in updates else job.get("monitor_script")
        eff_mon_url = updates["monitor_url"] if "monitor_url" in updates else job.get("monitor_url")
        if eff_mon_script and eff_mon_url:
            return tool_error(
                "monitor_script and monitor_url are mutually exclusive — "
                "clear one before setting the other.",
                success=False,
            )
    if context_from is not None or continuity is not None:
        # Empty string / list clears; otherwise every ref must exist. Stored
        # as a list (or None) to match create_job().
        if context_from is None:
            # continuity-only update: start from the job's stored refs.
            existing = job.get("context_from") or []
            refs = [str(j).strip() for j in existing if str(j).strip()]
        elif isinstance(context_from, str):
            refs = [context_from.strip()] if context_from.strip() else []
        else:
            refs = [str(j).strip() for j in context_from if str(j).strip()]
        if continuity is not None:
            refs = _apply_continuity(refs, continuity) or []
        if refs:
            ref_error = _validate_context_from_refs(refs)
            if ref_error:
                return tool_error(ref_error, success=False)
        updates["context_from"] = refs or None
    if a["enabled_toolsets"] is not None:
        updates["enabled_toolsets"] = a["enabled_toolsets"] or None
    if a["attach_to_session"] is not None:
        updates["attach_to_session"] = bool(a["attach_to_session"])
    if a["workdir"] is not None:
        # Empty string clears; otherwise update_job() validates/normalizes.
        updates["workdir"] = _normalize_optional_job_value(a["workdir"]) or None
    if no_agent is not None:
        # Flipping to True needs a script on the job or in this same update,
        # otherwise the next tick would error out.
        target_no_agent = bool(no_agent)
        if target_no_agent:
            effective_script = updates.get("script") if "script" in updates else job.get("script")
            if not effective_script:
                return tool_error(
                    "Cannot set no_agent=True on a job without a script. "
                    "Set `script` in the same update, or on the job first.",
                    success=False,
                )
        updates["no_agent"] = target_no_agent
    if a["repeat"] is not None:
        # Shared chokepoint coerces string forms ('forever'/'once'/'3') and
        # 0/negative values.
        from cron.jobs import normalize_repeat_value
        repeat_state = dict(job.get("repeat") or {})
        repeat_state["times"] = normalize_repeat_value(a["repeat"])
        updates["repeat"] = repeat_state
    if a["schedule"] is not None:
        parsed_schedule = parse_schedule(a["schedule"])
        updates["schedule"] = parsed_schedule
        updates["schedule_display"] = parsed_schedule.get("display", a["schedule"])
        if job.get("state") != "paused":
            updates["state"] = "scheduled"
            updates["enabled"] = True
    if not updates:
        return tool_error("No updates provided.", success=False)
    updated = update_job(job_id, updates)
    _notify_provider_jobs_changed_safe()
    _upd_result: Dict[str, Any] = {"success": True, "job": _format_job(updated)}
    # An update can switch modes or delivery — echo the same guidance as create.
    _upd_notes = _mode_guidance_notes(updated, _normalize_deliver_param(deliver))
    if _upd_notes:
        _upd_result["guidance"] = _upd_notes
    return _dumps(_upd_result)


# Actions that need no job_id, and job-bound actions (job resolved first).
_JOBLESS_ACTIONS = {"create": _action_create, "list": _action_list}
_JOB_ACTIONS = {
    "remove": _action_remove,
    "pause": _action_pause,
    "resume": _action_resume,
    "run": _action_run,
    "run_now": _action_run,
    "trigger": _action_run,
    "update": _action_update,
}


def cronjob(
    action: str,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[str] = None,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    include_disabled: bool = False,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    reason: Optional[str] = None,
    script: Optional[str] = None,
    context_from: Optional[Union[str, List[str]]] = None,
    continuity: Optional[bool] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
    no_agent: Optional[bool] = None,
    attach_to_session: Optional[bool] = None,
    monitor_script: Optional[str] = None,
    monitor_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    failure_deliver: Optional[Union[str, List[str]]] = None,
    task_id: str = None,
    session_id: Optional[str] = None,
) -> str:
    """Unified cron job management tool."""
    a = dict(locals())
    del a["task_id"]  # unused but kept for handler signature compatibility

    try:
        normalized = (action or "").strip().lower()

        handler = _JOBLESS_ACTIONS.get(normalized)
        if handler is not None:
            return handler(a)

        if not job_id:
            return tool_error(f"job_id is required for action '{normalized}'", success=False)

        try:
            job = resolve_job_ref(job_id)
        except AmbiguousJobReference as exc:
            return _dumps({
                "success": False,
                "error": str(exc),
                "matches": [
                    {
                        "id": m["id"],
                        "name": m.get("name"),
                        "schedule": m.get("schedule_display"),
                        "next_run_at": m.get("next_run_at"),
                    }
                    for m in exc.matches
                ],
            })
        if not job:
            return _dumps(
                {"success": False, "error": f"Job with ID or name '{job_id}' not found. Use cronjob(action='list') to inspect jobs."},
            )

        handler = _JOB_ACTIONS.get(normalized)
        if handler is None:
            return tool_error(f"Unknown cron action '{action}'", success=False)
        return handler(job, a)

    except Exception as e:
        return tool_error(str(e), success=False)


CRONJOB_SCHEMA = {
    "name": "cronjob_manage",
    "description": """Manage scheduled cron jobs: action='create' schedules a job from a prompt and/or skills; 'list' inspects jobs; 'update'/'pause'/'resume'/'remove' manage one by job_id (always list first — never guess job IDs); 'run' fires a job immediately in the BACKGROUND (returns a handle at once, outcome re-enters the conversation when done — do not wait or poll; optional 'prompt' adds transient context for that fire only).

Jobs run in a fresh session with no current-chat context, so prompts must be self-contained, and the agent's FINAL RESPONSE is what gets delivered — cron runs are autonomous and cannot ask questions. Prefer updating an existing job over creating near-duplicates.""",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: create, list, update, pause, resume, remove, run. When action=create, the 'schedule' and 'prompt' fields are REQUIRED."
            },
            "job_id": {
                "type": "string",
                "description": "Required for update/pause/resume/remove/run"
            },
            "prompt": {
                "type": "string",
                "description": "For create: the full self-contained prompt (paired with any skills as the task instruction). For run: optional transient context for that single fire (never persisted)."
            },
            "schedule": {
                "type": "string",
                "type": "string",
                "description": "REQUIRED for create. Schedule forms: (1) recurring interval — '30m', 'every 2h', 'every hour' (EVERY 30 minutes / 2 hours / hour, forever by default); (2) explicit one-shot by duration — 'in 30m', 'in 2h' (fires ONCE that far from now; use this for 'remind me in N minutes' — do NOT hand-compute an absolute timestamp); (3) natural day/time — 'every monday 9am', 'weekdays at 9am', 'every day at 9am' (recurring weekly/daily); (4) cron syntax — '0 9 * * *' (daily 9am); (5) absolute one-shot — ISO timestamp '2026-06-01T09:00:00'."
            },
            "name": {
                "type": "string",
                "description": "Optional human-friendly name"
            },
            "repeat": {
                "type": "integer",
                "description": "Optional repeat count. Omit for defaults (once for one-shot, forever for recurring)."
            },
            "deliver": {
                "type": "string",
                "description": "Where the job's output is POSTED as a one-way message (the job itself always runs in a fresh session with no chat context). Omit to address the chat/topic this job was created from. Otherwise: 'local' (save only, no delivery), 'all' (every connected home channel, resolved at fire time), 'bot-chat' or 'bot-chat:<profile>' (inject into a Bot Chat as a real message), or platform:chat_id:thread_id (e.g. 'telegram:-1001234567890:17585'). Comma-combine like 'origin,all'."
            },
            "failure_deliver": {
                "type": "string",
                "description": "Optional override target for FAILURE notices only (same grammar as deliver). When set, engine failure/interruption notices go here instead of the deliver target; 'local' suppresses them entirely (state still recorded in cron list/run history). Use for jobs delivering into shared channels where failure noise is unwanted. Omit = failures follow deliver (default). On update, '' clears."
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ordered skill names loaded before the cron prompt. On update, [] clears."
            },
            "script": {
                "type": "string",
                "description": f"Optional script run each tick; stdout is injected into the agent's prompt as context (with no_agent=True the script IS the job). Relative paths resolve under {display_hermes_home()}/scripts/; .sh/.bash via bash, else Python. On update, '' clears."
            },
            "monitor": {
                "type": "string",
                "description": "Optional change-detector that gates the agent: an http(s) URL (fetched each tick) or a script path (same rules as `script`, run each tick) — cheap, no LLM. Output identical to the previous tick skips the agent run entirely; changed output wakes the agent with a diff injected into the prompt. First tick always runs (baseline). Output must be deterministic (no timestamps) or every tick looks changed. Incompatible with no_agent. On update, '' clears."
            },
            "no_agent": {
                "type": "boolean",
                "default": False,
                "description": "True = no LLM: the scheduler runs `script` (required) on schedule and delivers its stdout verbatim; empty stdout sends nothing (watchdog pattern). Use for script-only pings with fixed output; keep False for anything needing reasoning."
            },
            "context_from": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional job ID(s) whose most recent completed output is injected as context each run — chains jobs (A collects, B processes). For a job's OWN previous output prefer `continuity`. On update, [] clears."
            },
            "continuity": {
                "type": "boolean",
                "description": "True = each run sees the job's own previous output, so it can dedupe and continue where it left off (scouts, monitors, incremental digests). Default false. On update, false turns it off."
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional toolset names to restrict the job's agent to (e.g. [\"web\", \"terminal\"]) — cuts token overhead. Infer from the prompt. Omit for all default tools. On update, [] clears."
            },
            "workdir": {
                "type": "string",
                "description": "Optional absolute existing path to run the job from: injects that directory's AGENTS.md/context files and anchors terminal/file tools there. On update, '' clears."
            },
            "attach_to_session": {
                "type": "boolean",
                "description": "True = the job's delivery is CONTINUABLE — the user can reply and the agent has the brief in context (threads on thread-capable platforms, mirrored into the DM elsewhere). Use for conversational recurring jobs (briefings); leave unset for fire-and-forget alerts. Scope: the job's own conversation only — the origin chat, the home-channel fallback when deliver='origin' captured no origin (script-created jobs), or the job's single explicit platform:chat target (this flag is the only way to attach an explicit target). Broadcast targets are never attached; no effect when deliver='local'."
            },
        },
        "required": ["action"]
    }
}


def check_cronjob_requirements() -> bool:
    """Available in interactive CLI mode and gateway/messaging platforms (the
    scheduler is internal; no crontab needed). Flags must be explicitly truthy
    via the shared ``env_var_enabled`` helper."""
    from utils import env_var_enabled

    return (
        env_var_enabled("HERMES_INTERACTIVE")
        or env_var_enabled("HERMES_GATEWAY_SESSION")
        or env_var_enabled("HERMES_EXEC_ASK")
    )


# --- Registry ---
from tools.registry import registry, tool_error


def _cronjob_handler(args, **kw):
    """Model-tool dispatch for ``cronjob``: resolves the one model-facing
    ``monitor`` field into the stored ``monitor_script``/``monitor_url`` pair
    (legacy field names still accepted for older transcripts)."""
    _mon_script, _mon_url = _split_monitor_arg(
        args.get("monitor"), args.get("monitor_script"), args.get("monitor_url")
    )
    return cronjob(
        action=args.get("action", ""),
        job_id=args.get("job_id"),
        prompt=args.get("prompt"),
        schedule=args.get("schedule"),
        name=args.get("name"),
        repeat=args.get("repeat"),
        deliver=args.get("deliver"),
        failure_deliver=args.get("failure_deliver"),
        include_disabled=args.get("include_disabled", True),
        skill=args.get("skill"),
        skills=args.get("skills"),
        # model / provider / base_url are intentionally NOT read from the
        # agent's arguments: per-job inference pins are user-owned (dashboard,
        # `hermes cron create/edit --model`, or hand-edited jobs). The agent
        # must not be able to point unattended spend at a different model.
        # Programmatic callers of cronjob() itself retain the parameters.
        reason=args.get("reason"),
        script=args.get("script"),
        context_from=args.get("context_from"),
        continuity=args.get("continuity"),
        enabled_toolsets=args.get("enabled_toolsets"),
        workdir=args.get("workdir"),
        no_agent=args.get("no_agent"),
        attach_to_session=args.get("attach_to_session"),
        monitor_script=_mon_script,
        monitor_url=_mon_url,
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
    )


registry.register(
    name="cronjob_manage",
    toolset="cronjob",
    schema=CRONJOB_SCHEMA,
    handler=_cronjob_handler,
    check_fn=check_cronjob_requirements,
    emoji="⏰",
)
