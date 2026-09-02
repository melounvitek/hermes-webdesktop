"""Pairing, webhooks, gateway lifecycle, credential pool, memory provider and operations (doctor/backup/import/hooks/checkpoints) dashboard routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import logging
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from hermes_cli.config import redact_key
from hermes_cli.web_models import PairingApprove, PairingRevoke, WebhookCreate, WebhookEnabledToggle, CredentialPoolAdd, MemoryProviderSelect, MemoryReset, BackupRequest, ImportRequest, HookCreate, HookDelete
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()


# ---------------------------------------------------------------------------
# Pairing endpoints — approve / revoke / list messaging pairing codes.
#
# These are how a remote admin onboards messaging users (Telegram, Discord, …)
# without shell access.  Wraps gateway.pairing.PairingStore directly.
# ---------------------------------------------------------------------------


def _pairing_store(profile: Optional[str] = None):
    """Pairing store for ``profile`` — the dashboard's own when unspecified.

    Every other admin endpoint scopes by profile, and the gateway already
    keeps one store per served profile (``gateway/run.py``). Without this the
    dashboard and desktop always read the global store, so an operator on a
    named profile approves into a whitelist their gateway never consults.

    ``PairingStore`` resolves the profile's home itself (``default`` maps back
    to the global store), so this only needs to validate the name — no
    ``_profile_scope`` needed, and nothing process-global is swapped across
    the ``await`` boundary.
    """
    from hermes_cli.web_server import _resolve_profile_dir
    from gateway.pairing import PairingStore

    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return PairingStore()

    _resolve_profile_dir(requested)  # 400/404 on an unknown profile

    return PairingStore(profile=requested)


@router.get("/api/pairing")
async def list_pairing(profile: Optional[str] = None):
    store = _pairing_store(profile)
    return {
        "pending": store.list_pending(),
        "approved": store.list_approved(),
    }


@router.post("/api/pairing/approve")
async def approve_pairing(body: PairingApprove):
    store = _pairing_store(body.profile)
    platform = (body.platform or "").lower().strip()
    # `request_id` is what an admin surface sends after listing pending
    # requests; `code` is the one-time code the user relays from their DM.
    # A GUI that only knows the older field name still works — a value with
    # request-id shape routes to the request path either way.
    target = (body.request_id or body.code or "").strip()
    if not platform or not target:
        raise HTTPException(
            status_code=400, detail="platform and request_id or code are required"
        )

    by_request_id = bool(body.request_id) or store.looks_like_request_id(target)
    if by_request_id:
        result = store.approve_request(platform, target)
    else:
        result = store.approve_code(platform, target.upper())

    if result:
        return {"ok": True, "user": result}
    # Lockout only gates the code path, so only report it there — otherwise a
    # stale request id would surface as a bogus 429 while the platform sat
    # locked out for an unrelated reason.
    if not by_request_id and store._is_locked_out(platform):
        raise HTTPException(
            status_code=429,
            detail=f"Platform '{platform}' is locked out after too many failed approvals.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"Pairing request or code not found or expired for platform '{platform}'.",
    )


@router.post("/api/pairing/revoke")
async def revoke_pairing(body: PairingRevoke):
    store = _pairing_store(body.profile)
    platform = (body.platform or "").lower().strip()
    if not platform or not body.user_id:
        raise HTTPException(status_code=400, detail="platform and user_id are required")
    if store.revoke(platform, body.user_id):
        return {"ok": True}
    raise HTTPException(
        status_code=404,
        detail=f"User {body.user_id} not found in approved list for {platform}.",
    )


@router.post("/api/pairing/clear-pending")
async def clear_pending_pairing(profile: Optional[str] = None):
    store = _pairing_store(profile)
    count = store.clear_pending()
    return {"ok": True, "cleared": count}


# ---------------------------------------------------------------------------
# Webhook subscription endpoints — list / subscribe / remove.
#
# Wraps the same JSON store the CLI uses (hermes_cli.webhook); the webhook
# adapter hot-reloads it without a gateway restart.  Per-route HMAC secrets
# are redacted on read and surfaced once on create.
# ---------------------------------------------------------------------------


def _webhook_route_summary(name: str, route: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": route.get("description", ""),
        "events": list(route.get("events") or []),
        "deliver": route.get("deliver", "log"),
        "deliver_only": bool(route.get("deliver_only")),
        "prompt": route.get("prompt", ""),
        "script": route.get("script", ""),
        "skills": list(route.get("skills") or []),
        "created_at": route.get("created_at"),
        "url": f"{base_url}/webhooks/{name}",
        # Secret is masked on read; full value only returned on create.
        "secret_set": bool(route.get("secret")),
        # Default-enabled; only an explicit enabled:false turns a route off.
        "enabled": route.get("enabled", True) is not False,
    }


@router.get("/api/webhooks")
async def list_webhooks():
    import hermes_cli.webhook as wh

    base_url = wh._get_webhook_base_url()
    subs = wh._load_subscriptions()
    return {
        "enabled": wh._is_webhook_enabled(),
        "base_url": base_url,
        "subscriptions": [
            _webhook_route_summary(name, route, base_url)
            for name, route in subs.items()
        ],
    }


@router.post("/api/webhooks/enable")
async def enable_webhooks():
    from hermes_cli.web_server import _restart_gateway_after_webhook_enable, _write_platform_enabled
    try:
        _write_platform_enabled("webhook", True)
    except Exception as exc:
        _log.exception("Failed to enable webhook platform from dashboard")
        raise HTTPException(
            status_code=500,
            detail="Failed to enable webhook platform.",
        ) from exc

    restart_result = _restart_gateway_after_webhook_enable()
    return {
        "ok": True,
        "platform": "webhook",
        "enabled": True,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@router.post("/api/webhooks")
async def create_webhook(body: WebhookCreate):
    import re as _re
    import secrets as _secrets
    import time as _time
    import hermes_cli.webhook as wh

    if not wh._is_webhook_enabled():
        raise HTTPException(
            status_code=400,
            detail="Webhook platform is not enabled. Enable it from the Webhooks page first.",
        )

    name = (body.name or "").strip().lower().replace(" ", "-")
    if not _re.match(r"^[a-z0-9][a-z0-9_-]*$", name):
        raise HTTPException(
            status_code=400,
            detail="Invalid name. Use lowercase alphanumeric with hyphens/underscores.",
        )

    if body.deliver_only and body.deliver == "log":
        raise HTTPException(
            status_code=400,
            detail="Direct delivery requires a real target (telegram, discord, …), not 'log'.",
        )

    secret = body.secret or _secrets.token_urlsafe(32)
    route: Dict[str, Any] = {
        "description": body.description or f"Dashboard-created subscription: {name}",
        "events": [e.strip() for e in body.events if e.strip()],
        "secret": secret,
        "prompt": body.prompt or "",
        "skills": [s.strip() for s in body.skills if s.strip()],
        "deliver": body.deliver or "log",
        "created_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    if body.script and body.script.strip():
        route["script"] = body.script.strip()
    if body.deliver_only:
        route["deliver_only"] = True
    if body.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": body.deliver_chat_id}

    subs = wh._load_subscriptions()
    subs[name] = route
    wh._save_subscriptions(subs)

    base_url = wh._get_webhook_base_url()
    summary = _webhook_route_summary(name, route, base_url)
    # Surface the secret exactly once, on create.
    summary["secret"] = secret
    return summary


@router.delete("/api/webhooks/{name}")
async def delete_webhook(name: str):
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    del subs[key]
    wh._save_subscriptions(subs)
    return {"ok": True}


@router.put("/api/webhooks/{name}/enabled")
async def set_webhook_enabled(name: str, body: WebhookEnabledToggle):
    """Enable or disable a webhook route.

    Disabled routes stay in the subscriptions file (so they can be
    re-enabled) but the gateway rejects incoming events with 403.  The
    gateway hot-reloads the subscriptions file, so this takes effect on the
    next event without a restart.
    """
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    subs[key]["enabled"] = bool(body.enabled)
    wh._save_subscriptions(subs)
    return {"ok": True, "name": key, "enabled": bool(body.enabled)}


# ---------------------------------------------------------------------------
# Gateway lifecycle endpoints — start / stop.
#
# restart + update already exist above; these complete the lifecycle so a
# remote admin can bring the gateway up or down without shell access.  Both
# spawn the real `hermes gateway <verb>` so behaviour matches the CLI exactly.
# Status is already surfaced by /api/status (gateway_running/state/platforms).
# ---------------------------------------------------------------------------


@router.post("/api/gateway/start")
async def start_gateway(profile: Optional[str] = None):
    from hermes_cli.web_server import _gateway_subcommand, _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "start"), "gateway-start")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway start")
        raise HTTPException(status_code=500, detail=f"Failed to start gateway: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "gateway-start"}


@router.post("/api/gateway/stop")
async def stop_gateway(profile: Optional[str] = None):
    from hermes_cli.web_server import _gateway_subcommand, _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "stop"), "gateway-stop")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway stop")
        raise HTTPException(status_code=500, detail=f"Failed to stop gateway: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "gateway-stop"}


# ---------------------------------------------------------------------------
# Credential pool endpoints — list / add / remove rotation keys.
#
# The credential pool (auth.json -> credential_pool.<provider>[]) holds the
# rotating API keys the agent round-robins through.  Secrets are redacted on
# read; only the agent ever sees the raw values at session start.
# ---------------------------------------------------------------------------


def _pool_entry_summary(entry: Any, index: int) -> Dict[str, Any]:
    """Redacted, display-safe view of one PooledCredential.

    ``index`` is 1-based to match CredentialPool.remove_index().
    """
    token = getattr(entry, "access_token", "") or ""
    return {
        "index": index,
        "id": getattr(entry, "id", None),
        "label": getattr(entry, "label", None),
        "auth_type": getattr(entry, "auth_type", None),
        "source": getattr(entry, "source", None),
        "priority": getattr(entry, "priority", 0),
        "last_status": getattr(entry, "last_status", None),
        "request_count": getattr(entry, "request_count", 0),
        "token_preview": redact_key(token) if token else "",
        "has_refresh": bool(getattr(entry, "refresh_token", None)),
    }


@router.get("/api/credentials/pool")
async def list_credential_pool():
    from agent.credential_pool import load_pool
    from hermes_cli.auth import read_credential_pool

    # load_pool() may hit the network synchronously (Copilot token exchange
    # over raw urllib). urllib's timeout does NOT bound DNS resolution
    # (getaddrinfo blocks in C), so on a networkless Windows host this froze
    # the uvicorn event loop for 17 minutes (2026-08-22 00:03-00:20 stall).
    # Keep every provider load off the loop - same pattern as
    # get_memory_status below.
    def _run():
        providers = []
        # read_credential_pool(None) lists every provider that has pooled entries;
        # load_pool() then gives us the rich PooledCredential objects per provider.
        raw_pool = read_credential_pool()
        for provider_id in sorted(raw_pool.keys()):
            try:
                pool = load_pool(provider_id)
            except Exception:
                _log.exception("load_pool(%s) failed", provider_id)
                continue
            entries = pool.entries()
            if not entries:
                continue
            providers.append({
                "provider": provider_id,
                "entries": [
                    _pool_entry_summary(e, i) for i, e in enumerate(entries, start=1)
                ],
            })
        return {"providers": providers}

    return await asyncio.to_thread(_run)


@router.post("/api/credentials/pool")
async def add_credential_pool_entry(body: CredentialPoolAdd):
    import uuid as _uuid
    from agent.credential_pool import (
        load_pool,
        PooledCredential,
        AUTH_TYPE_API_KEY,
        CUSTOM_POOL_PREFIX,
        SOURCE_MANUAL,
    )

    provider = (body.provider or "").strip().lower()
    api_key = (body.api_key or "").strip()
    if not provider or not api_key:
        raise HTTPException(status_code=400, detail="provider and api_key are required")

    # load_pool() may run synchronous OAuth token exchanges (network I/O);
    # keep it off the event loop - see list_credential_pool (2026-08-22
    # 17-minute stall fix).
    def _run():
        try:
            pool = load_pool(provider)
            label = (body.label or "").strip() or f"key #{len(pool.entries()) + 1}"
            entry = PooledCredential(
                provider=provider,
                id=_uuid.uuid4().hex[:6],
                label=label,
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token=api_key,
            )
            pool.add_entry(entry)
            # Re-adding a credential is an explicit re-engagement signal: lift
            # every suppression for this provider so a source deleted earlier
            # (via DELETE below or `hermes auth add`) can seed again.
            # Mirrors the `hermes auth add` behaviour in auth_commands.py.
            if not provider.startswith(CUSTOM_POOL_PREFIX):
                try:
                    from hermes_cli.auth import (
                        _load_auth_store,
                        unsuppress_credential_source,
                    )
                    suppressed = _load_auth_store().get("suppressed_sources", {})
                    for src in list(suppressed.get(provider, []) or []):
                        unsuppress_credential_source(provider, src)
                except Exception:
                    _log.exception("unsuppress after pool add failed (non-fatal)")
            return {"ok": True, "provider": provider, "count": len(pool.entries())}
        except HTTPException:
            raise
        except Exception as exc:
            _log.exception("POST /api/credentials/pool failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await asyncio.to_thread(_run)


@router.delete("/api/credentials/pool/{provider}/{index}")
async def remove_credential_pool_entry(provider: str, index: int):
    """Remove a pool entry.  ``index`` is 1-based (matches the list response).

    Removal must be sticky (#55217): ``load_pool()`` re-seeds entries from
    their backing source (.env var, OAuth singleton file, custom-provider
    config) on every call, so deleting only the pool row silently reverts on
    the next dashboard refresh.  We dispatch through the same RemovalStep
    registry the CLI ``hermes auth remove`` uses: each source cleans up its
    external state and suppresses ``(provider, source)`` so the seeders skip
    it.  Manual entries have no registered step — nothing external to clean,
    no suppression needed (they aren't re-seeded).
    """
    from agent.credential_pool import load_pool
    from agent.credential_sources import find_removal_step
    from hermes_cli.auth import suppress_credential_source

    provider = (provider or "").strip().lower()
    # load_pool() may run synchronous token exchanges and the removal steps do
    # blocking disk writes - keep them off the event loop (see
    # list_credential_pool; 2026-08-22 17-minute stall fix).
    def _run():
        try:
            pool = load_pool(provider)
            removed = pool.remove_index(index)
        except Exception as exc:
            _log.exception("DELETE /api/credentials/pool failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if removed is None:
            raise HTTPException(status_code=404, detail="No pool entry at that index")

        cleaned: List[str] = []
        hints: List[str] = []
        step = find_removal_step(provider, removed.source or "")
        if step is not None:
            try:
                result = step.remove_fn(provider, removed)
                cleaned = list(result.cleaned)
                hints = list(result.hints)
                if result.suppress:
                    suppress_credential_source(provider, removed.source)
            except Exception:
                # Cleanup is best-effort, but suppression is the actual bug fix -
                # without it the entry resurrects on the next load_pool(). Apply
                # it even when source-specific cleanup blew up.
                _log.exception(
                    "credential source cleanup failed for %s/%s; suppressing anyway",
                    provider, removed.source,
                )
                try:
                    suppress_credential_source(provider, removed.source)
                except Exception:
                    _log.exception("suppress_credential_source failed")
        return {
            "ok": True,
            "provider": provider,
            "count": len(pool.entries()),
            "cleaned": cleaned,
            "hints": hints,
        }

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Memory provider endpoints — status / list providers / select / disable / reset.
#
# Provider setup is dashboard-native when a provider exposes get_config_schema().
# The dashboard never runs interactive provider setup hooks; activation is only
# allowed once the provider is discoverable, available, and has required config.
# ---------------------------------------------------------------------------


@router.get("/api/memory")
async def get_memory_status():
    # load_config(), file stats and provider discovery are disk reads — keep
    # them off the event loop.
    from hermes_cli.web_server import (
        _discover_memory_provider_statuses,
        _normalize_memory_provider_name,
        asyncio,
        get_hermes_home,
        load_config,
    )
    def _run():
        cfg = load_config()
        active = ""
        mem = cfg.get("memory")
        if isinstance(mem, dict):
            active = _normalize_memory_provider_name(mem.get("provider"))

        # Built-in memory file sizes (so the UI can show what a reset would erase).
        mem_dir = get_hermes_home() / "memories"
        files = {}
        for fname, key in (("MEMORY.md", "memory"), ("USER.md", "user")):
            path = mem_dir / fname
            files[key] = path.stat().st_size if path.exists() else 0

        return {
            "active": active,
            "providers": _discover_memory_provider_statuses(),
            "builtin_files": files,
        }

    return await asyncio.to_thread(_run)


@router.put("/api/memory/provider")
async def set_memory_provider(body: MemoryProviderSelect):
    from hermes_cli.web_server import (
        _CONFIG_MUTATION_LOCK,
        _normalize_memory_provider_name,
        _require_memory_provider_ready,
        asyncio,
        load_config,
        save_config,
    )
    provider = _normalize_memory_provider_name(body.provider)

    def _run():
        _require_memory_provider_ready(provider)

        with _CONFIG_MUTATION_LOCK:
            cfg = load_config()
            if not isinstance(cfg.get("memory"), dict):
                cfg["memory"] = {}
            cfg["memory"]["provider"] = provider
            save_config(cfg)
        return {"ok": True, "active": provider}

    return await asyncio.to_thread(_run)


@router.post("/api/memory/reset")
async def reset_memory(body: MemoryReset):
    from hermes_cli.web_server import get_hermes_home
    target = (body.target or "all").strip().lower()
    if target not in {"all", "memory", "user"}:
        raise HTTPException(status_code=400, detail="target must be all, memory, or user")

    mem_dir = get_hermes_home() / "memories"
    deleted = []
    targets = []
    if target in {"all", "memory"}:
        targets.append("MEMORY.md")
    if target in {"all", "user"}:
        targets.append("USER.md")
    for fname in targets:
        path = mem_dir / fname
        if path.exists():
            try:
                path.unlink()
                deleted.append(fname)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not delete {fname}: {exc}")
    return {"ok": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Operations endpoints — doctor / security audit / backup / import /
# checkpoints / hooks.
#
# Diagnostic and maintenance commands.  The long-running / text-output ones
# (doctor, security audit, backup, import, skills install) are spawned as
# background actions whose logs the dashboard tails via
# /api/actions/{name}/status — same pattern as gateway restart and update.
# The cheap, structured reads (hooks list, checkpoints list) return JSON
# directly.
# ---------------------------------------------------------------------------


@router.post("/api/ops/doctor")
async def run_doctor():
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["doctor"], "doctor")
    except Exception as exc:
        _log.exception("Failed to spawn doctor")
        raise HTTPException(status_code=500, detail=f"Failed to run doctor: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "doctor"}


@router.post("/api/ops/security-audit")
async def run_security_audit():
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["security", "audit"], "security-audit")
    except Exception as exc:
        _log.exception("Failed to spawn security audit")
        raise HTTPException(status_code=500, detail=f"Failed to run security audit: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "security-audit"}


def _dashboard_backup_dir() -> Path:
    from hermes_cli.web_server import get_hermes_home
    return get_hermes_home() / "backups"


def _new_dashboard_backup_path() -> Path:
    from hermes_cli.web_server import secrets
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return _dashboard_backup_dir() / f"hermes-backup-{stamp}-{secrets.token_hex(4)}.zip"


@router.post("/api/ops/backup")
async def run_backup(body: BackupRequest):
    from hermes_cli.web_server import _spawn_hermes_action
    args = ["backup"]
    archive: Optional[Path] = None
    output = (body.output or "").strip()
    if output:
        args.extend(["-o", output])
    else:
        archive = _new_dashboard_backup_path()
        try:
            archive.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not create backup directory: {exc}",
            )
        args.extend(["-o", str(archive)])
    try:
        proc = _spawn_hermes_action(args, "backup")
    except Exception as exc:
        _log.exception("Failed to spawn backup")
        raise HTTPException(status_code=500, detail=f"Failed to run backup: {exc}")
    response = {"ok": True, "pid": proc.pid, "name": "backup"}
    if archive is not None:
        response["archive"] = str(archive)
    return response


@router.get("/api/ops/backup/download")
async def download_dashboard_backup(archive: str):
    from hermes_cli.web_server import _path_is_under
    try:
        backup_dir = _dashboard_backup_dir().expanduser().resolve(strict=False)
        target = Path(archive).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backup not found")
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid backup path")

    if not _path_is_under(backup_dir, target):
        raise HTTPException(status_code=403, detail="Backup is outside the dashboard backup directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(
        path=str(target),
        media_type="application/zip",
        filename=target.name,
        content_disposition_type="attachment",
    )


@router.post("/api/ops/import")
async def run_import(body: ImportRequest):
    from hermes_cli.web_server import _spawn_hermes_action, os
    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")
    if not os.path.isfile(archive):
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive}")
    args = ["import", archive]
    if body.force:
        args.append("--force")
    try:
        proc = _spawn_hermes_action(args, "import")
    except Exception as exc:
        _log.exception("Failed to spawn import")
        raise HTTPException(status_code=500, detail=f"Failed to run import: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "import"}


def _safe_backup_upload_name(filename: str | None) -> str:
    name = Path(filename or "backup.zip").name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name:
        name = "backup.zip"
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    return name


@router.post("/api/ops/import-upload")
async def run_import_upload(
    file: UploadFile = File(...),
    force: bool = Form(False),
):
    from hermes_cli.web_server import (
        _MANAGED_FILE_MAX_BYTES,
        _UPLOAD_CHUNK_BYTES,
        _spawn_hermes_action,
        os,
        secrets,
    )
    staging_dir = _dashboard_backup_dir()
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create import staging directory: {exc}",
        )

    safe_name = _safe_backup_upload_name(file.filename)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = staging_dir / f"dashboard-import-{stamp}-{secrets.token_hex(4)}-{safe_name}"
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".upload",
        dir=str(staging_dir),
    )
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Archive is too large")
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Import staging directory is not writable",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not write uploaded archive: {exc}",
        )
    finally:
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()

    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded archive is not a valid zip file",
        )

    args = ["import", str(target)]
    if force:
        args.append("--force")
    try:
        proc = _spawn_hermes_action(args, "import")
    except Exception as exc:
        _log.exception("Failed to spawn import")
        raise HTTPException(status_code=500, detail=f"Failed to run import: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "import",
        "archive": str(target),
        "uploaded_bytes": total,
    }


@router.get("/api/ops/hooks")
async def list_hooks():
    """List configured shell hooks from config.yaml with consent + health.

    Reports each hook's allowlist (consent) status and whether the script is
    currently executable, plus the set of valid hook events so the create
    form can offer them.
    """
    from hermes_cli.web_server import asyncio
    def _run():
        from hermes_cli.config import load_config as _load_config
        from agent import shell_hooks

        try:
            from hermes_cli.plugins import VALID_HOOKS
            valid_events = sorted(VALID_HOOKS)
        except Exception:
            valid_events = []

        specs = []
        try:
            specs = shell_hooks.iter_configured_hooks(_load_config())
        except Exception:
            _log.exception("iter_configured_hooks failed")

        out = []
        for spec in specs:
            entry = None
            try:
                entry = shell_hooks.allowlist_entry_for(spec.event, spec.command)
            except Exception:
                pass
            executable = False
            try:
                executable = shell_hooks.script_is_executable(spec.command)
            except Exception:
                pass
            out.append({
                "event": spec.event,
                "matcher": spec.matcher,
                "command": spec.command,
                "timeout": spec.timeout,
                "allowed": entry is not None,
                "approved_at": (entry or {}).get("approved_at"),
                "executable": executable,
            })

        return {"hooks": out, "valid_events": valid_events}

    return await asyncio.to_thread(_run)


@router.post("/api/ops/hooks")
async def create_hook(body: HookCreate):
    """Add a shell hook to config.yaml (and optionally approve it).

    Shell hooks run arbitrary commands, so this is a privileged action: it
    writes to the ``hooks:`` config block and, when ``approve`` is set, records
    consent in the allowlist so the hook actually fires.  Takes effect on the
    next session / gateway restart.
    """
    from hermes_cli.web_server import _CONFIG_MUTATION_LOCK, asyncio, load_config, save_config
    from agent import shell_hooks

    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")

    try:
        from hermes_cli.plugins import VALID_HOOKS
        if event not in VALID_HOOKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown event '{event}'. Valid: {', '.join(sorted(VALID_HOOKS))}",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    def _run():
        with _CONFIG_MUTATION_LOCK:
            cfg = load_config()
            hooks_cfg = cfg.get("hooks")
            if not isinstance(hooks_cfg, dict):
                hooks_cfg = {}
                cfg["hooks"] = hooks_cfg
            entries = hooks_cfg.get(event)
            if not isinstance(entries, list):
                entries = []
                hooks_cfg[event] = entries

            new_entry: Dict[str, Any] = {"command": command}
            if body.matcher:
                new_entry["matcher"] = body.matcher
            if body.timeout is not None:
                new_entry["timeout"] = int(body.timeout)
            entries.append(new_entry)
            save_config(cfg)

        approved = False
        if body.approve:
            try:
                shell_hooks._record_approval(event, command)
                approved = True
            except Exception:
                _log.exception("hook consent record failed")

        return {"ok": True, "event": event, "command": command, "approved": approved}

    return await asyncio.to_thread(_run)


@router.delete("/api/ops/hooks")
async def delete_hook(body: HookDelete):
    """Remove a hook from config.yaml and revoke its consent allowlist entry."""
    from hermes_cli.web_server import _CONFIG_MUTATION_LOCK, asyncio, load_config, save_config
    from agent import shell_hooks

    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")

    def _run():
        removed = False
        with _CONFIG_MUTATION_LOCK:
            cfg = load_config()
            hooks_cfg = cfg.get("hooks")
            if isinstance(hooks_cfg, dict) and isinstance(hooks_cfg.get(event), list):
                before = len(hooks_cfg[event])
                hooks_cfg[event] = [
                    e for e in hooks_cfg[event]
                    if not (isinstance(e, dict) and e.get("command") == command)
                ]
                removed = len(hooks_cfg[event]) < before
                if not hooks_cfg[event]:
                    del hooks_cfg[event]
                if not hooks_cfg:
                    cfg.pop("hooks", None)
                save_config(cfg)

        # Revoke consent regardless so a re-add re-prompts.
        try:
            shell_hooks.revoke(command)
        except Exception:
            pass
        return removed

    removed = await asyncio.to_thread(_run)

    if not removed:
        raise HTTPException(status_code=404, detail="No matching hook found")
    return {"ok": True}


@router.get("/api/ops/checkpoints")
async def list_checkpoints():
    """List the /rollback shadow store checkpoints (read-only)."""
    from hermes_cli.web_server import get_hermes_home, os
    # Checkpoints live under <hermes_home>/checkpoints/.  Surface a count +
    # total size so the dashboard can show what a prune would reclaim; the
    # actual prune is a spawned action so confirmation/pruning logic stays
    # in one place (the CLI).
    cp_dir = get_hermes_home() / "checkpoints"
    sessions = []
    total_bytes = 0
    if cp_dir.is_dir():
        with os.scandir(cp_dir) as scan:
            children = sorted((Path(e.path) for e in scan), key=lambda p: p.name)
        for child in children:
            if not child.is_dir():
                continue
            size = 0
            count = 0
            for f in child.rglob("*"):
                if f.is_file():
                    try:
                        size += f.stat().st_size
                        count += 1
                    except OSError:
                        pass
            total_bytes += size
            sessions.append({
                "session": child.name,
                "files": count,
                "bytes": size,
            })
    return {"sessions": sessions, "total_bytes": total_bytes}


@router.post("/api/ops/checkpoints/prune")
async def prune_checkpoints():
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["checkpoints", "prune"], "checkpoints-prune")
    except Exception as exc:
        _log.exception("Failed to spawn checkpoints prune")
        raise HTTPException(status_code=500, detail=f"Failed to prune checkpoints: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "checkpoints-prune"}
