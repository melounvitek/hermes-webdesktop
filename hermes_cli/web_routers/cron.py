"""Cron dashboard routes.

The ``*_sync`` workers, profile resolution and the threadpool wrapper
(``_run_cron_dashboard_io``) live in web_server and are reached through the
late-binding seam so ``monkeypatch.setattr(web_server, ...)`` keeps working.
"""

import asyncio
import functools
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    CronJobCreate,
    CronJobUpdate,
    AutomationBlueprintInstantiate,
)
from hermes_cli.web_routers._common import log as _log

router = APIRouter()

_run_cron_dashboard_io = late("_run_cron_dashboard_io")
_list_cron_jobs_sync = late("_list_cron_jobs_sync")
_get_cron_job_sync = late("_get_cron_job_sync")
_list_cron_job_runs_sync = late("_list_cron_job_runs_sync")
_create_cron_job_sync = late("_create_cron_job_sync")
_update_cron_job_sync = late("_update_cron_job_sync")
_pause_cron_job_sync = late("_pause_cron_job_sync")
_resume_cron_job_sync = late("_resume_cron_job_sync")
_trigger_cron_job_sync = late("_trigger_cron_job_sync")
_delete_cron_job_sync = late("_delete_cron_job_sync")
_find_cron_job_profile = late("_find_cron_job_profile")
_fire_cron_job_for_profile = late("_fire_cron_job_for_profile")
_forward_cron_fire_to_gateway = late("_forward_cron_fire_to_gateway")
_gateway_intentionally_stopped = late("_gateway_intentionally_stopped")
_notify_cron_provider_for_profile = late("_notify_cron_provider_for_profile")
_call_cron_for_profile = late("_call_cron_for_profile")
_raise_if_cron_registration_error = late("_raise_if_cron_registration_error")
load_config = late("load_config")
cfg_get = late("cfg_get")

# Retry-After hint (seconds) on retryable cron-fire 503s: sized to clear a
# scale-to-zero wake or gateway restart so a scheduler that honors it spaces
# its next attempt past the outage instead of burning its retry budget in it.
_CRON_FIRE_RETRY_AFTER_SECONDS = 60


@router.get("/api/cron/jobs")
async def list_cron_jobs(profile: str = "all"):
    return await _run_cron_dashboard_io(_list_cron_jobs_sync, profile)


@router.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_get_cron_job_sync, job_id, profile)


@router.get("/api/cron/jobs/{job_id}/runs")
async def list_cron_job_runs(job_id: str, profile: Optional[str] = None, limit: int = 20):
    return await _run_cron_dashboard_io(_list_cron_job_runs_sync, job_id, profile, limit)


@router.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_create_cron_job_sync, body, profile)


@router.get("/api/cron/delivery-targets")
async def get_cron_delivery_targets():
    """Delivery targets for the cron dropdown: implicit ``local`` plus the
    configured gateway platforms (a platform without a cron home channel is
    still listed with ``home_target_set: false`` so the UI can say so)."""
    targets = [
        {
            "id": "local",
            "name": "Local (save only)",
            "home_target_set": True,
            "home_env_var": None,
        }
    ]
    try:
        from cron.scheduler import cron_delivery_targets

        targets.extend(cron_delivery_targets())
    except Exception:
        _log.exception("GET /api/cron/delivery-targets failed")
    return {"targets": targets}


@router.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_update_cron_job_sync, job_id, body, profile)


@router.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_pause_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_resume_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_trigger_cron_job_sync, job_id, profile)


@router.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_delete_cron_job_sync, job_id, profile)


@router.post("/api/cron/fire")
async def cron_fire_webhook(request: Request):
    """Chronos managed-cron fire webhook (NAS -> agent) — gateway forwarder.

    Gated by the NAS-minted JWT (this path is in ``PUBLIC_API_PATHS``), not the
    dashboard cookie.  The dashboard is only the public door: execution belongs
    to the GATEWAY process (it owns the live platform adapters relay-fronted
    and E2EE targets need), so the fire is forwarded to the gateway
    api_server's own ``/api/cron/fire`` on loopback and its response passed
    through (the gateway re-verifies the JWT).  Gateway unreachable -> 503 so
    NAS retries; deliberately NO local-execution fallback.
    """
    from plugins.cron_providers.chronos.verify import get_fire_verifier

    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else ""

    cfg = await asyncio.to_thread(load_config)
    claims = get_fire_verifier()(
        token=token,
        expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
        jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
        issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None,
    )
    if claims is None:
        return JSONResponse({"error": "invalid fire token"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    job_id = (body or {}).get("job_id") if isinstance(body, dict) else None
    if not job_id:
        return JSONResponse({"error": "missing job_id"}, status_code=400)

    # Walks every profile's job list (file I/O) — off the event loop.
    profile = await _run_cron_dashboard_io(_find_cron_job_profile, job_id)
    if not profile:
        # Job is gone (cancelled / completed): 200 so NAS does not retry.
        return JSONResponse({"status": "gone", "job_id": job_id}, status_code=200)

    forwarded = await _forward_cron_fire_to_gateway(profile, job_id, auth)
    if forwarded is None:
        # Stamp the miss on the job record (last_fire_error) so the dead hop is
        # visible in `cronjob list` / the dashboard, not just gui.log.
        # Best-effort: visibility must never break the retry contract below.
        try:
            await _run_cron_dashboard_io(
                _call_cron_for_profile,
                profile,
                "note_fire_forward_failure",
                job_id,
                "scheduled fire could not be forwarded to the gateway "
                "api_server (127.0.0.1 loopback unreachable); the gateway "
                "process may be down or its api_server adapter not bound "
                "(missing API_SERVER_KEY)",
            )
        except Exception:
            _log.debug("could not stamp last_fire_error for %s", job_id, exc_info=True)
        # Split by operator intent: a deliberately stopped gateway (durable
        # desired_state == "stopped") can never be reached by retrying, so drop
        # with 200 + a structured log line — the Chronos provider re-arms every
        # job on the next gateway start.  A transient window (wake, restart,
        # crash loop) keeps the retryable 503 with a Retry-After hint.
        if await _run_cron_dashboard_io(_gateway_intentionally_stopped, profile):
            _log.info(
                "cron fire dropped: gateway for profile %r is deliberately "
                "stopped (desired_state=stopped); job %s will resume via "
                "Chronos reconcile on next gateway start",
                profile, job_id,
            )
            return JSONResponse(
                {
                    "status": "gateway_stopped",
                    "detail": "gateway deliberately stopped; fire dropped, "
                              "jobs re-arm on next gateway start",
                    "job_id": job_id,
                    "profile": profile,
                },
                status_code=200,
            )
        return JSONResponse(
            {
                "error": "gateway unreachable; retry",
                "job_id": job_id,
                "profile": profile,
            },
            status_code=503,
            headers={"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)},
        )
    status_code, gateway_body = forwarded
    if isinstance(gateway_body, dict):
        gateway_body.setdefault("job_id", job_id)
    # The gateway's own 503s (draining, admission failure) are equally
    # transient — same spacing hint.
    headers = {"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)} if status_code == 503 else None
    return JSONResponse(gateway_body, status_code=status_code, headers=headers)


@router.get("/api/cron/blueprints")
async def list_cron_blueprints():
    """Blueprint catalog as form schemas; the ``deliver`` slot's options are
    rewritten from the actually configured gateway platforms."""
    try:
        from cron.blueprint_catalog import CATALOG, blueprint_catalog_entry

        deliver_options = None
        try:
            from cron.scheduler import cron_delivery_targets

            platforms = [t["id"] for t in cron_delivery_targets() if t.get("id")]
            deliver_options = ["origin", "local", *platforms]
        except Exception:
            _log.debug("cron_delivery_targets unavailable; using static deliver options", exc_info=True)

        entries = []
        for r in CATALOG:
            entry = blueprint_catalog_entry(r)
            if deliver_options:
                for f in entry.get("fields", []):
                    if f.get("name") == "deliver":
                        f["options"] = deliver_options
            entries.append(entry)
        return {"blueprints": entries}
    except Exception as e:
        _log.exception("GET /api/cron/blueprints failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cron/blueprints/instantiate")
async def instantiate_blueprint(body: AutomationBlueprintInstantiate, profile: str = "default"):
    """Fill a blueprint's slots and create the cron job (form-submit path)."""
    try:
        from cron.blueprint_catalog import fill_blueprint, get_blueprint, BlueprintFillError

        blueprint = get_blueprint(body.blueprint)
        if blueprint is None:
            raise HTTPException(status_code=404, detail=f"Unknown blueprint: {body.blueprint}")
        try:
            spec = fill_blueprint(blueprint, body.values)
        except BlueprintFillError as exc:
            # Field-level validation error — 422 so the form can show it inline.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Blueprint jobs deliver to the dashboard's configured target by
        # default; the form's deliver slot overrides via spec["deliver"].
        spec.pop("origin", None)
        # Off the event loop like the sibling endpoints; partial keeps **spec
        # keys from colliding with the wrapper's own parameters.
        _create = functools.partial(_call_cron_for_profile, profile, "create_job", **spec)
        created = await _run_cron_dashboard_io(_create)
        # Reconcile the profile-scoped provider (file I/O + NAS calls) off-loop.
        await _run_cron_dashboard_io(_notify_cron_provider_for_profile, profile)
        return created
    except HTTPException:
        raise
    except Exception as e:
        _raise_if_cron_registration_error(e)
        _log.exception("POST /api/cron/blueprints/instantiate failed")
        raise HTTPException(status_code=400, detail=str(e))
