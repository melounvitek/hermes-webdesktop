"""Dashboard cron helpers: per-profile scheduler I/O, job validation/normalisation, cron fire and gateway forwarding.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import inspect
import re
from fastapi import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli.config import cfg_get
from hermes_cli.web_models import CronJobCreate

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# ---------------------------------------------------------------------------
# Cron job management endpoints
# ---------------------------------------------------------------------------


def _cron_optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _cron_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = re.split(r"[\n,]", value)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        return None
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    return items or None


def _normalize_dashboard_cron_script(value: Any, profile_home: Path) -> Optional[str]:
    """Validate a dashboard-selected cron script against the profile sandbox."""
    text = _cron_optional_text(value)
    if not text:
        return None

    scripts_root = (profile_home / "scripts").resolve()
    raw_path = Path(text).expanduser()
    candidate = raw_path.resolve() if raw_path.is_absolute() else (scripts_root / raw_path).resolve()
    try:
        relative = candidate.relative_to(scripts_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"script must be inside {scripts_root}",
        ) from exc
    if not candidate.exists():
        raise HTTPException(status_code=400, detail=f"script does not exist: {candidate}")
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"script is not a file: {candidate}")
    return str(relative)


def _validate_dashboard_cron_effective_job(job: Dict[str, Any]) -> None:
    prompt = _cron_optional_text(job.get("prompt"))
    script = _cron_optional_text(job.get("script"))
    skills = _cron_string_list(job.get("skills")) or _cron_string_list(job.get("skill"))
    no_agent = bool(job.get("no_agent"))

    if no_agent:
        if not script:
            raise HTTPException(
                status_code=400,
                detail="no_agent=True requires a script",
            )
        return

    if not (prompt or skills or script):
        raise HTTPException(
            status_code=400,
            detail="agent cron jobs require a prompt, skill, or script",
        )


def _validate_dashboard_cron_context_from(
    refs: Optional[List[str]],
    profile_name: str,
) -> None:
    from hermes_cli.web_server import _call_cron_for_profile
    if not refs:
        return
    for ref in refs:
        # "self" (the continuity toggle) resolves to the job's own id at run
        # time — it can't be validated against the store (create precedes the
        # job's existence).
        if isinstance(ref, str) and ref.strip().lower() == "self":
            continue
        if not _call_cron_for_profile(profile_name, "get_job", ref):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"context_from job '{ref}' not found in profile "
                    f"'{profile_name}'"
                ),
            )


def _cron_profile_dicts() -> List[Dict[str, Any]]:
    """Return the minimal profile records needed by cron aggregation.

    The two callers only consume ``name``.  ``list_profiles()`` also parses
    config/distribution metadata, probes gateway processes, and counts skills
    for every profile; polling cron jobs through that path creates avoidable
    GIL pressure on large profile pools.
    """
    from hermes_cli.web_server import _fallback_profile_dicts
    from hermes_cli import profiles as profiles_mod
    try:
        return [
            {
                "name": name,
                "path": str(home),
                "is_default": name == "default",
            }
            for name, home in profiles_mod.profiles_to_serve(multiplex=True)
        ]
    except Exception:
        _log.exception("Failed to list profiles for cron dashboard; falling back to directory scan")
        return _fallback_profile_dicts(profiles_mod)


def _cron_default_profile() -> str:
    """Profile to target when a cron request carries no explicit ``profile``.

    A desktop pool backend runs one process per profile (HERMES_HOME already
    scoped), but these cron endpoints deliberately route storage through the
    profiles tree via ``_cron_profile_home`` — so a hardcoded ``"default"``
    fallback would write a non-default profile's job into ``~/.hermes``.
    Resolve the process's own profile instead. ``custom`` (an unrecognized
    HERMES_HOME outside the profiles tree) has no profile-dir equivalent, so
    it keeps the legacy ``default`` fallback.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name

        name = get_active_profile_name()
    except Exception:
        return "default"
    return "default" if name in ("default", "custom") else name


def _cron_profile_home(profile: Optional[str]) -> Tuple[str, Path]:
    """Resolve a profile query value to (profile_name, HERMES_HOME)."""
    from hermes_cli.web_server import _cron_default_profile
    from hermes_cli import profiles as profiles_mod

    raw = (profile or _cron_default_profile()).strip() or "default"
    try:
        canon = profiles_mod.normalize_profile_name(raw)
        profiles_mod.validate_profile_name(canon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(canon):
        raise HTTPException(status_code=404, detail=f"Profile '{canon}' does not exist.")
    return canon, profiles_mod.get_profile_dir(canon)


def _annotate_cron_job(job: Dict[str, Any], profile: str, home: Path) -> Dict[str, Any]:
    annotated = dict(job)
    annotated["profile"] = profile
    annotated["profile_name"] = profile
    annotated["hermes_home"] = str(home)
    annotated["is_default_profile"] = profile == "default"
    return annotated


def _call_cron_for_profile(target_profile: Optional[str], func_name: str, *args, **kwargs):
    """Run cron.jobs helpers against the selected profile's cron directory.

    The dashboard is a single process that can inspect many profiles. Route
    storage through cron.jobs' execution-context override so dashboard calls
    cannot retarget a concurrent desktop ticker's load/save transaction.
    """
    from hermes_cli.web_server import _cron_profile_home
    profile_name, home = _cron_profile_home(target_profile)
    from cron import jobs as cron_jobs
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(str(home))
    try:
        with cron_jobs.use_cron_store(home):
            if func_name == "create_job":
                from cron.scheduler import create_job_with_scheduler_registration

                result = create_job_with_scheduler_registration(*args, **kwargs)
            else:
                result = getattr(cron_jobs, func_name)(*args, **kwargs)
    finally:
        reset_hermes_home_override(token)

    if isinstance(result, list):
        return [_annotate_cron_job(j, profile_name, home) for j in result]
    if isinstance(result, dict):
        return _annotate_cron_job(result, profile_name, home)
    return result


def _notify_cron_provider_for_profile(target_profile: Optional[str]) -> None:
    """Best-effort provider reconcile against one profile's job store.

    Fail-closed for external providers on a multi-profile dashboard: an
    external provider's ``reconcile`` converges its REMOTE registry toward
    one profile's jobs.json, and its orphan cleanup cancels every remote
    entry absent from that store. The NAS registry is not profile-scoped,
    so reconciling profile B would silently disarm profile A's one-shots.
    Until the provider contract carries a profile identity through
    arm/cancel/list, a multi-profile dashboard must not drive unscoped
    external reconciles at all — the affected profile simply re-arms on
    its next fire/start (idempotent via dedup_key). The built-in provider
    re-reads jobs.json each tick and stays a no-op here.
    """
    from hermes_cli.web_server import _cron_profile_dicts, _cron_profile_home
    try:
        _profile_name, home = _cron_profile_home(target_profile)
        from cron import jobs as cron_jobs
        from cron.scheduler_provider import (
            InProcessCronScheduler,
            resolve_cron_scheduler,
        )
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(str(home))
        try:
            with cron_jobs.use_cron_store(home):
                provider = resolve_cron_scheduler()
                if not isinstance(provider, InProcessCronScheduler):
                    profile_names = [
                        str(p.get("name") or "")
                        for p in _cron_profile_dicts()
                    ]
                    if len([n for n in profile_names if n]) > 1:
                        _log.warning(
                            "Skipping cron provider reconcile for profile %s: "
                            "external provider '%s' reconcile is not "
                            "profile-scoped and would disarm other profiles' "
                            "armed one-shots. The mutated profile re-arms "
                            "idempotently on its next fire/start.",
                            target_profile,
                            provider.name,
                        )
                        return
                provider.on_jobs_changed()
        finally:
            reset_hermes_home_override(token)
    except Exception:
        _log.debug(
            "Cron provider reconciliation failed for profile %s",
            target_profile,
            exc_info=True,
        )


def _mutate_cron_for_profile(
    target_profile: Optional[str], func_name: str, *args, **kwargs
):
    """Apply a cron store mutation and reconcile its scheduler provider."""
    from hermes_cli.web_server import _call_cron_for_profile, _notify_cron_provider_for_profile
    result = _call_cron_for_profile(target_profile, func_name, *args, **kwargs)
    if result:
        _notify_cron_provider_for_profile(target_profile)
    return result


def _find_cron_job_profile(job_id: str) -> Optional[str]:
    from hermes_cli.web_server import _call_cron_for_profile, _cron_profile_dicts
    for profile in _cron_profile_dicts():
        name = str(profile.get("name") or "")
        if not name:
            continue
        jobs = _call_cron_for_profile(name, "list_jobs", True)
        if any(j.get("id") == job_id or j.get("name") == job_id for j in jobs):
            return name
    return None


async def _run_cron_dashboard_io(func, *args, **kwargs):
    """Run cron dashboard profile/job I/O outside the FastAPI event loop."""
    from hermes_cli.web_server import run_in_threadpool
    if inspect.iscoroutinefunction(func):
        raise TypeError("_run_cron_dashboard_io only accepts sync callables")
    result = await run_in_threadpool(func, *args, **kwargs)
    if inspect.isawaitable(result):
        raise TypeError("_run_cron_dashboard_io sync callable returned an awaitable")
    return result


def _raise_if_cron_registration_error(e: Exception) -> None:
    """Re-raise a cron partial-failure (job saved, external scheduler
    registration failed) as HTTP 424 with the structured envelope.

    Shared by every dashboard cron-create surface so the contract can't
    drift between copies. The lazy import keeps cron out of module import.
    """
    from cron.scheduler import CronSchedulerRegistrationError

    if isinstance(e, CronSchedulerRegistrationError):
        raise HTTPException(status_code=424, detail=e.to_dict()) from e


def _create_cron_job_sync(body: CronJobCreate, profile: Optional[str] = None):
    from hermes_cli.web_server import _cron_profile_home
    try:
        profile_name, profile_home = _cron_profile_home(profile)
        script = _normalize_dashboard_cron_script(body.script, profile_home)
        skills = _cron_string_list(body.skills)
        context_from = _cron_string_list(body.context_from)
        _validate_dashboard_cron_context_from(context_from, profile_name)
        no_agent = bool(body.no_agent)
        _validate_dashboard_cron_effective_job({
            "prompt": body.prompt,
            "skills": skills,
            "script": script,
            "no_agent": no_agent,
        })
        return _mutate_cron_for_profile(
            profile_name,
            "create_job",
            prompt=body.prompt or "",
            schedule=body.schedule,
            name=body.name,
            deliver=_cron_optional_text(body.deliver) or "local",
            skills=skills,
            model=_cron_optional_text(body.model),
            provider=_cron_optional_text(body.provider),
            base_url=_cron_optional_text(body.base_url, strip_trailing_slash=True),
            script=script,
            context_from=context_from,
            enabled_toolsets=_cron_string_list(body.enabled_toolsets),
            workdir=_cron_optional_text(body.workdir),
            no_agent=no_agent,
        )
    except HTTPException:
        raise
    except Exception as e:
        _raise_if_cron_registration_error(e)
        _log.exception("POST /api/cron/jobs failed")
        raise HTTPException(status_code=400, detail=str(e))


def _fire_cron_job_for_profile(
    profile: str,
    job_id: str,
    *,
    force: bool = False,
) -> bool:
    """DEPRECATED for NAS webhook fires (superseded by gateway forwarding);
    retained for the dashboard trigger path — do not add new uses.

    Run ONE due cron job end-to-end for ``profile`` via the resolved
    scheduler provider's ``fire_due`` (store CAS claim + ``run_one_job``).

    Superseded by :func:`_forward_cron_fire_to_gateway`: cron fires must
    execute in the GATEWAY process (which owns the live platform adapters),
    not the dashboard. Executing here delivered through the standalone path
    only, which cannot serve relay-fronted logical platforms (their only
    sender is the live relay adapter — no native credential exists on the
    box) or E2EE rooms. Kept temporarily because external callers may still
    resolve it via the web_deps late-binding seam.
    """
    from hermes_cli.web_server import _cron_profile_home
    _profile_name, home = _cron_profile_home(profile)
    from cron import jobs as cron_jobs
    from cron.scheduler_provider import (
        provider_supports_force_fire,
        resolve_cron_scheduler,
    )
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(str(home))
    try:
        with cron_jobs.use_cron_store(home):
            provider = resolve_cron_scheduler()
            if force:
                if not provider_supports_force_fire(provider):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Cron provider '{getattr(provider, 'name', 'custom')}' "
                            "does not support atomic forced firing of paused jobs"
                        ),
                    )
                return bool(
                    provider.fire_due(job_id, adapters=None, loop=None, force=True)
                )
            return bool(provider.fire_due(job_id, adapters=None, loop=None))
    finally:
        reset_hermes_home_override(token)


def _profile_env_value(home: Path, key: str) -> str:
    """Best-effort read of one KEY=VALUE line from a profile's .env file."""
    try:
        env_path = home / ".env"
        if not env_path.is_file():
            return ""
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _gateway_fire_endpoint(profile: str, home: Path) -> str:
    """Resolve the loopback URL of the gateway api_server's cron-fire route.

    Port resolution mirrors gateway/config.py's api_server load order for the
    LISTENER-OWNER profile: ``platforms.api_server.extra.port`` in that
    profile's config.yaml, then ``API_SERVER_PORT`` (process env for the
    active profile, the profile's own .env otherwise), then the adapter
    default 8642. The bind host is the adapter's loopback default — the
    dashboard and gateway share a network namespace in every supported
    deployment (same host process tree, or the same container under s6).

    Multiplex mode (one gateway serving several profiles) exposes per-profile
    mirrors under ``/p/<profile>/…``, so a non-default profile routes through
    the default gateway's port with that prefix — only the DEFAULT profile's
    api_server is bound in that mode, so the port must be read from the
    default home, never the target profile's (a secondary's own
    ``API_SERVER_PORT`` is a port nothing listens on). Per-profile-gateway
    mode (each profile its own process/port) uses the bare path on the
    profile's own port.
    """
    from hermes_cli.web_server import _cron_default_profile, load_config
    import os as _os

    multiplex = False
    try:
        from gateway.config import _env_multiplex_profiles_override

        cfg = load_config()
        multiplex = bool(cfg_get(cfg, "gateway", "multiplex_profiles", default=False))
        env_flag = _env_multiplex_profiles_override()
        if env_flag is not None:
            multiplex = env_flag
    except Exception:
        _log.debug("cron fire: multiplex detection failed; assuming single-profile", exc_info=True)

    listener_profile, listener_home = profile, home
    if multiplex and profile != "default":
        from hermes_constants import get_default_hermes_root

        listener_profile, listener_home = "default", get_default_hermes_root()
        _log.info(
            "cron fire: multiplex gateway — resolving api_server port for %s "
            "from the default profile's listener (%s)",
            profile,
            listener_home,
        )

    port = 0
    try:
        # Profile-scoped read through the CANONICAL loader (managed-scope
        # overlay, ${ENV_VAR} expansion, profile pathing) — never a raw
        # yaml.safe_load of config.yaml (tests/hermes_cli/
        # test_config_read_guard.py). The HERMES_HOME override scopes
        # get_config_path() to the LISTENER-OWNER profile, same pattern the
        # deprecated _fire_cron_job_for_profile used for its store scope.
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(str(listener_home))
        try:
            profile_cfg = load_config()
        finally:
            reset_hermes_home_override(token)
        raw = cfg_get(
            profile_cfg, "platforms", "api_server", "extra", "port", default=None
        )
        if raw:
            port = int(raw)
    except Exception:
        port = 0
    if not port:
        raw = (
            _os.getenv("API_SERVER_PORT", "")
            if listener_profile == _cron_default_profile()
            else _profile_env_value(listener_home, "API_SERVER_PORT")
        )
        try:
            port = int(raw) if raw else 0
        except ValueError:
            port = 0
    if not port:
        port = 8642

    if multiplex and profile != "default":
        return f"http://127.0.0.1:{port}/p/{profile}/api/cron/fire"
    return f"http://127.0.0.1:{port}/api/cron/fire"


async def _forward_cron_fire_to_gateway(
    profile: str, job_id: str, authorization: str
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """Forward a Chronos fire callback to the gateway api_server on loopback.

    The dashboard is the hosted deployment's only public HTTP door (Fly proxy
    → internal_port 9119), but cron execution belongs to the GATEWAY process:
    it owns the live platform adapters, so delivery works for relay-fronted
    logical platforms and E2EE rooms — the standalone path the dashboard used
    to run cannot serve either. This forwards the fire byte-preserved (same
    job_id, same NAS bearer — the gateway re-verifies the JWT itself) and
    passes the gateway's response through.

    Returns ``(status_code, body)`` from the gateway, or ``None`` when the
    gateway is unreachable (not started yet after a scale-to-zero wake,
    restarting, or api_server disabled) — the caller maps that to 503 so NAS
    retries per the Chronos contract (non-2xx = retryable; the store CAS
    de-dupes the eventual double fire), UNLESS the profile's gateway was
    deliberately stopped (see :func:`_gateway_intentionally_stopped`), in
    which case the caller drops the fire with 200 — retrying into an
    operator-stopped gateway can never succeed and only burns scheduler
    retries (OOF-266).
    """
    from hermes_cli.web_server import _cron_profile_home
    _profile_name, home = _cron_profile_home(profile)
    url = _gateway_fire_endpoint(_profile_name, home)
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={"job_id": job_id},
                headers={"Authorization": authorization},
            )
    except Exception as exc:
        _log.warning(
            "cron fire forward to %s failed (%s: %s); returning 503 for NAS retry",
            url, type(exc).__name__, exc,
        )
        return None
    try:
        body = resp.json()
    except Exception:
        body = {"raw": (resp.text or "")[:500]}
    if not isinstance(body, dict):
        body = {"raw": body}
    return resp.status_code, body


def _gateway_intentionally_stopped(profile: Optional[str]) -> bool:
    """True when the profile's gateway is stopped BY OPERATOR INTENT.

    Reads the durable ``desired_state`` field of the profile's
    ``gateway_state.json`` — written exclusively by the s6 lifecycle
    commands (``hermes gateway stop`` persists ``"stopped"``; start and
    restart persist ``"running"``, see service_manager's
    ``_write_gateway_desired_state``). This is the same operator-intent
    signal container-boot reconciliation trusts, and it is precisely NOT
    set to "stopped" during transient windows (crash loops, drains,
    scale-to-zero wakes, restarts) — so it cleanly splits "retry will
    eventually succeed" from "retry can never succeed".

    Deliberately does NOT fall back to the volatile ``gateway_state``
    runtime field: a legacy file without ``desired_state`` (or a gateway
    that crashed before persisting) must stay on the retryable-503 path.
    Failing open to "not intentionally stopped" is the safe direction —
    the worst case is retries against a dead gateway, which is exactly
    today's behavior.

    Exception-safe: any resolution or parse failure returns False.
    """
    from hermes_cli.web_server import _cron_profile_home
    import json as _json

    try:
        _name, home = _cron_profile_home(profile)
        state_file = home / "gateway_state.json"
        if not state_file.exists():
            return False
        data = _json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        return data.get("desired_state") == "stopped"
    except Exception:
        return False
