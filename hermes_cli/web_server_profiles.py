"""Profile-scoped helpers: profile discovery fallback, profile dir/MCP-server writes, the profile/config scope context managers, skills-hub and tools/analytics catalog helpers.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import hashlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from fastapi import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional
from hermes_cli.config import DEFAULT_CONFIG, get_process_hermes_home
from hermes_cli.web_models import MCPServerCreate
from hermes_cli.web_server_gateway import _ACTION_LOG_FILES
from hermes_cli.web_server_mcp import _normalize_mcp_server_create

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _is_other_profile(profile: Optional[str]) -> bool:
    """True when ``profile`` names a profile other than this process's own."""
    from hermes_cli.web_server import _resolve_profile_dir
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return False
    try:
        target = _resolve_profile_dir(requested)
    except HTTPException:
        return True
    return target.resolve() != get_process_hermes_home().resolve()


def _approval_mode_of(config: Dict[str, Any]) -> str:
    """Normalize approvals.mode from an in-memory config document.

    Both sides of the broadcast comparison use in-memory documents (the raw
    on-disk dict and the about-to-be-saved dict): re-reading through the
    config cache after a save can serve the pre-save document when the
    replacement file collides on the (mtime_ns, size) cache key, which would
    suppress the broadcast exactly when the mode changed. Absent block or
    key normalizes to the same default the approval gate uses.
    """
    from tools.approval import _normalize_approval_mode

    approvals = config.get("approvals")
    default_mode = (DEFAULT_CONFIG.get("approvals") or {}).get("mode", "manual")
    mode = approvals.get("mode", default_mode) if isinstance(approvals, dict) else default_mode
    return _normalize_approval_mode(mode)


def _broadcast_gateway_session_info() -> None:
    """Broadcast session.info on the in-process gateway when it's loaded.

    ``sys.modules`` guard, not an import: gateway never imported means no
    live sessions in this process to notify.
    """
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return
    try:
        server.broadcast_session_info()
    except Exception:
        _log.exception("session.info broadcast after config save failed")


def _parse_model_ids(resp: "Any") -> List[str]:
    """Extract model ids from an OpenAI-compatible ``/v1/models`` response.

    Tolerant of the common shapes: ``{"data": [{"id": ...}]}`` (OpenAI / vLLM /
    llama.cpp) and a bare ``{"data": ["id", ...]}``. Returns ``[]`` on any
    parse/HTTP error so a slightly non-standard endpoint never hard-blocks.
    """
    try:
        if not resp.is_success:
            return []
        payload = resp.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = str(item.get("id") or "").strip()
        else:
            mid = str(item or "").strip()
        if mid:
            ids.append(mid)
    return ids


def _fallback_profile_dicts(profiles_mod) -> List[Dict[str, Any]]:
    def _safe(callable_, default):
        try:
            return callable_()
        except Exception:
            return default

    profiles: List[Dict[str, Any]] = []
    default_home = profiles_mod._get_default_hermes_home()
    if default_home.is_dir():
        model, provider = _safe(lambda: profiles_mod._read_config_model(default_home), (None, None))
        profiles.append({
            "name": "default",
            "path": str(default_home),
            "is_default": True,
            "model": model,
            "provider": provider,
            "has_env": (default_home / ".env").exists(),
            "skill_count": _safe(lambda: profiles_mod._count_skills(default_home), 0),
            "gateway_running": _safe(lambda: profiles_mod._check_gateway_running(default_home), False),
            "description": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description", ""), ""),
            "description_auto": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description_auto", False), False),
            "distribution_name": None,
            "distribution_version": None,
            "distribution_source": None,
            "has_alias": False,
        })

    profiles_root = profiles_mod._get_profiles_root()
    if profiles_root.is_dir():
        # Use os.scandir (context-managed) instead of Path.iterdir to avoid
        # leaking directory fds when an exception interrupts iteration — the
        # sidebar polls every few seconds so an fd leak exhausts RLIMIT_NOFILE
        # within days (#81547).
        with os.scandir(profiles_root) as scan:
            entries = sorted(scan, key=lambda e: e.name)
        for entry in entries:
            entry_path = Path(entry.path)
            if not entry.is_dir() or not profiles_mod._PROFILE_ID_RE.match(entry.name):
                continue
            model, provider = _safe(lambda entry=entry_path: profiles_mod._read_config_model(entry), (None, None))
            profiles.append({
                "name": entry.name,
                "path": str(entry_path),
                "is_default": False,
                "model": model,
                "provider": provider,
                "has_env": _safe(lambda entry=entry_path: (entry / ".env").exists(), False),
                "skill_count": _safe(lambda entry=entry_path: profiles_mod._count_skills(entry), 0),
                "gateway_running": _safe(
                    lambda entry=entry_path, name=entry.name: (
                        profiles_mod._check_gateway_running(entry)
                        or profiles_mod._served_by_running_multiplexer(name)
                    ),
                    False,
                ),
                "description": _safe(lambda entry=entry_path: profiles_mod.read_profile_meta(entry).get("description", ""), ""),
                "description_auto": _safe(lambda entry=entry_path: profiles_mod.read_profile_meta(entry).get("description_auto", False), False),
                "distribution_name": None,
                "distribution_version": None,
                "distribution_source": None,
                "has_alias": False,
            })

    return profiles


def _resolve_profile_dir(name: str) -> Path:
    """Validate ``name`` and resolve to its directory or raise an HTTPException."""
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.validate_profile_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' does not exist.")
    return profiles_mod.get_profile_dir(name)


def _write_profile_mcp_servers(profile_dir: Path, servers: List["MCPServerCreate"]) -> int:
    """Write MCP server entries into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override (same mechanism as
    ``_write_profile_model``) so the entries land in the target profile's
    config rather than the dashboard process's active profile.

    Mirrors the per-server shape the ``POST /api/mcp/servers`` endpoint builds,
    but batched so the whole profile-create write is a single config save.
    Returns the number of servers written.
    """
    from hermes_cli.web_server import load_config, save_config
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.mcp_config import _save_bearer_auth_token

    written = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            try:
                name, entry, bearer_token = _normalize_mcp_server_create(server)
            except ValueError as exc:
                display_name = (server.name or "").strip() or "<unnamed>"
                _log.warning(
                    "Profile-create: skipping MCP server '%s': %s",
                    display_name,
                    exc,
                )
                continue
            if bearer_token is not None:
                entry["headers"] = _save_bearer_auth_token(name, bearer_token)
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            # We created an empty mcp_servers dict but wrote nothing — don't
            # leave a stray empty key in the new profile's config.
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    finally:
        reset_hermes_home_override(token)
    return written


# ---------------------------------------------------------------------------
# Skills & Tools endpoints
#
# Every read/write below accepts an optional ``profile`` query param so the
# dashboard can manage ANY profile's skills/toolsets, not just the profile
# the dashboard process happens to be running under. Without this, "Set as
# active" on the Profiles page (which only flips the sticky ``active_profile``
# file for FUTURE CLI/gateway invocations) misled users into thinking skill
# toggles would land in the activated profile — they silently wrote into the
# dashboard's own config instead. See _profile_scope() for the mechanism.
# ---------------------------------------------------------------------------


_SKILLS_PROFILE_LOCK = threading.RLock()


@contextmanager
def _profile_scope(profile: Optional[str]):
    """Scope config + skill-directory resolution to ``profile`` for one request.

    Two seams must be redirected for skills/toolsets endpoints:

    1. ``load_config``/``save_config`` resolve ``get_hermes_home()`` at call
       time — the context-local override from ``set_hermes_home_override``
       reaches them (same pattern as ``_write_profile_model``).
    2. ``tools.skills_tool`` and ``tools.skill_manager_tool`` bind
       ``SKILLS_DIR`` at import time, so the override CANNOT reach them.
       Like ``_call_cron_for_profile`` does for cron's module globals,
       temporarily retarget both under a lock and restore them
       immediately after.

    ``tools.skills_sync`` (reset/diff/list-modified/opt-in/opt-out/
    repair-official) needs NO retargeting: since #65828 its directory
    lookups resolve at call time through the same contextvar override
    set in step 1.

    ``profile`` of None/""/"current" means "the dashboard's own profile" —
    config resolution is untouched, but the skill-module globals are still
    retargeted to the *current* ``get_hermes_home()`` so writes land in the
    live home even when the import-time binding is stale (e.g. the process
    imported the modules before a HERMES_HOME override, or under test
    isolation).
    """
    from hermes_cli.web_server import _resolve_profile_dir
    requested = (profile or "").strip()

    from hermes_constants import (
        get_hermes_home,
        set_hermes_home_override,
        reset_hermes_home_override,
    )
    from tools import skills_tool as _skills_tool
    from tools import skill_manager_tool as _skill_mgr

    token = None
    if not requested or requested.lower() == "current":
        profile_dir = get_hermes_home()
    else:
        profile_dir = _resolve_profile_dir(requested)
        token = set_hermes_home_override(str(profile_dir))

    with _SKILLS_PROFILE_LOCK:
        old_home = _skills_tool.HERMES_HOME
        old_skills_dir = _skills_tool.SKILLS_DIR
        old_mgr_home = _skill_mgr.HERMES_HOME
        old_mgr_skills_dir = _skill_mgr.SKILLS_DIR
        _skills_tool.HERMES_HOME = profile_dir
        _skills_tool.SKILLS_DIR = profile_dir / "skills"
        _skill_mgr.HERMES_HOME = profile_dir
        _skill_mgr.SKILLS_DIR = profile_dir / "skills"
        try:
            yield profile_dir if token is not None else None
        finally:
            _skills_tool.HERMES_HOME = old_home
            _skills_tool.SKILLS_DIR = old_skills_dir
            _skill_mgr.HERMES_HOME = old_mgr_home
            _skill_mgr.SKILLS_DIR = old_mgr_skills_dir
            if token is not None:
                reset_hermes_home_override(token)


@contextmanager
def _config_profile_scope(profile: Optional[str]):
    """Await-safe, config-only profile scope for handlers that ``await``.

    Unlike ``_profile_scope`` this touches ONLY the context-local
    ``set_hermes_home_override`` contextvar — it does NOT swap the
    process-global ``skills_tool``/``skill_manager`` module attributes.
    Those globals are shared across all event-loop tasks, so holding them
    across an ``await`` lets a concurrent skills request restore THIS
    request's profile dir on its ``finally`` (cross-contamination). The
    contextvar override is task-local and survives an ``await`` cleanly,
    which is all endpoints that resolve ``get_hermes_home()`` at call time
    (config, env, gateway status) actually need.

    None/""/"current" means the dashboard's own profile — no override.
    """
    from hermes_cli.web_server import _resolve_profile_dir
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        yield None
        return

    from hermes_constants import (
        set_hermes_home_override,
        reset_hermes_home_override,
    )

    profile_dir = _resolve_profile_dir(requested)
    token = set_hermes_home_override(str(profile_dir))
    try:
        yield profile_dir
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# Terminal execution backend picker — the GUI counterpart of terminal.backend
# in config.yaml. Each row carries a fast, defensive health probe (Docker
# daemon reachable, SSH host configured, Modal/Daytona credentials present) so
# the Capabilities panel can render Ready / Needs setup guidance instead of a
# bare enum (issues #57738 / #63783). Probes must never raise — a probe
# failure renders as a status, not a 500.
# ---------------------------------------------------------------------------

# Table-driven backend metadata — kept in sync with the dispatch ladder in
# tools/terminal_tool.py::_create_environment and the terminal.backend enum
# surfaced in the desktop raw-config settings.
_TERMINAL_BACKENDS: List[Dict[str, str]] = [
    {
        "name": "local",
        "label": "Local",
        "description": "Run commands directly on this machine. No isolation.",
    },
    {
        "name": "docker",
        "label": "Docker",
        "description": "Run commands in an isolated Docker container with a persistent workspace.",
    },
    {
        "name": "singularity",
        "label": "Singularity / Apptainer",
        "description": "Run commands in a Singularity/Apptainer container (HPC-friendly, rootless).",
    },
    {
        "name": "modal",
        "label": "Modal",
        "description": "Run commands in a Modal cloud sandbox.",
    },
    {
        "name": "daytona",
        "label": "Daytona",
        "description": "Run commands in a Daytona cloud sandbox.",
    },
    {
        "name": "ssh",
        "label": "SSH",
        "description": "Run commands on a remote host over SSH.",
    },
]


def _plugin_terminal_backend_rows() -> List[Dict[str, str]]:
    """Picker rows for plugin-registered terminal backends (fail-soft)."""
    rows: List[Dict[str, str]] = []
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()  # idempotent — plugin state may not be loaded yet
    except Exception:
        pass
    try:
        from agent.terminal_env_registry import list_providers

        for provider in list_providers():
            try:
                rows.append({
                    "name": provider.name.strip().lower(),
                    "label": provider.display_name,
                    "description": provider.description,
                })
            except Exception:
                continue
    except Exception:
        return rows
    return rows


# ---------------------------------------------------------------------------
# Token / cost analytics endpoint
# ---------------------------------------------------------------------------


def _aux_usage_rows(db, cutoff: float) -> List[Dict[str, Any]]:
    """Per-(model, task) auxiliary usage within the window (issue #23270).

    Reads the task-dimension rows (task != '') that record_auxiliary_usage
    writes into session_model_usage. Returns [] when the table predates the
    task column (older DB opened read-only by newer code).
    """
    try:
        cur = db._conn.execute("""
            SELECT u.model,
                   u.task,
                   u.billing_provider,
                   SUM(u.input_tokens) as input_tokens,
                   SUM(u.output_tokens) as output_tokens,
                   SUM(u.cache_read_tokens) as cache_read_tokens,
                   SUM(u.reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(u.estimated_cost_usd), 0) as estimated_cost,
                   COUNT(DISTINCT u.session_id) as sessions,
                   SUM(COALESCE(u.api_call_count, 0)) as api_calls,
                   MAX(u.last_seen) as last_used_at
            FROM session_model_usage u
            JOIN sessions s ON s.id = u.session_id
            WHERE s.started_at > ? AND u.task != ''
            GROUP BY u.model, u.task, u.billing_provider
            ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC
        """, (cutoff,))
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        # Table predates the task column (older DB opened by newer code) —
        # aux breakdown is simply unavailable.
        return []


def _merge_aux_into_by_model(
    by_model: List[Dict[str, Any]], aux_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fold aux usage rows into the sessions-derived per-model list.

    Aux usage lives only in session_model_usage (never in the sessions
    counters), so adding it here cannot double-count. Models that ONLY
    appear via aux calls (e.g. a dedicated vision model) get their own
    entry — previously they were entirely invisible.
    """
    if not aux_rows:
        return by_model
    merged: Dict[str, Dict[str, Any]] = {}
    for row in by_model:
        merged[row.get("model") or "unknown"] = row
    for aux in aux_rows:
        model = aux.get("model") or "unknown"
        target = merged.get(model)
        if target is None:
            target = {
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0,
                "sessions": 0,
                "api_calls": 0,
            }
            merged[model] = target
        target["input_tokens"] = (target.get("input_tokens") or 0) + (aux.get("input_tokens") or 0)
        target["output_tokens"] = (target.get("output_tokens") or 0) + (aux.get("output_tokens") or 0)
        target["estimated_cost"] = (target.get("estimated_cost") or 0) + (aux.get("estimated_cost") or 0)
        target["api_calls"] = (target.get("api_calls") or 0) + (aux.get("api_calls") or 0)
        tasks = target.setdefault("aux_tasks", [])
        tasks.append({
            "task": aux.get("task") or "",
            "input_tokens": aux.get("input_tokens") or 0,
            "output_tokens": aux.get("output_tokens") or 0,
            "estimated_cost": aux.get("estimated_cost") or 0,
            "api_calls": aux.get("api_calls") or 0,
        })
    result = list(merged.values())
    result.sort(
        key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
        reverse=True,
    )
    return result


def _aux_task_summary(aux_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate aux usage rows across models into a per-task summary."""
    by_task: Dict[str, Dict[str, Any]] = {}
    for aux in aux_rows:
        task = aux.get("task") or ""
        d = by_task.setdefault(task, {
            "task": task,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
            "api_calls": 0,
            "models": [],
        })
        d["input_tokens"] += aux.get("input_tokens") or 0
        d["output_tokens"] += aux.get("output_tokens") or 0
        d["estimated_cost"] += aux.get("estimated_cost") or 0
        d["api_calls"] += aux.get("api_calls") or 0
        model = aux.get("model") or "unknown"
        if model not in d["models"]:
            d["models"].append(model)
    result = list(by_task.values())
    result.sort(
        key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
        reverse=True,
    )
    return result


def _profile_cli_args(profile: Optional[str]) -> List[str]:
    """Return ``["-p", <name>]`` for a validated non-default profile.

    Hub install/uninstall/update run in a fresh ``hermes`` subprocess, and
    ``_apply_profile_override()`` reads ``-p`` from argv in the child — the
    only mechanism that reaches import-time-bound globals like
    ``skills_hub.SKILLS_DIR``. Empty/"current" means the dashboard's own
    profile (no args, legacy behavior).
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() in {"current", "default"}:
        return []
    from hermes_cli import profiles as profiles_mod
    _resolve_profile_dir(requested)
    return ["-p", profiles_mod.normalize_profile_name(requested)]


def _hub_action_name(verb: str, key: str) -> str:
    """Unique per-skill hub action name (+ registered log file).

    ``_spawn_hermes_action`` tracks one process/log per name, so a shared
    "skills-install"/"skills-uninstall" would make concurrent row-level actions
    overwrite each other's status/log while the UI polls per identifier. Slug
    (readable) + hash (collision-proof) keys each action to its own row.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:48] or "skill"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    name = f"skills-{verb}-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(name, f"action-{name}.log")
    return name


def _installed_hub_identifiers(profile: Optional[str] = None) -> dict:
    """Map identifier -> installed lock entry for hub-installed skills.

    Lets the UI mark search results that are already installed.  Scoped to
    ``profile``'s skills/.hub/lock.json when provided (HubLockFile takes an
    explicit path, sidestepping the import-time LOCK_FILE binding).
    Best-effort: returns an empty dict if the lock file can't be read.
    """
    try:
        from tools.skills_hub import HubLockFile

        requested = (profile or "").strip()
        if requested and requested.lower() != "current":
            profile_dir = _resolve_profile_dir(requested)
            lock = HubLockFile(profile_dir / "skills" / ".hub" / "lock.json")
        else:
            lock = HubLockFile()
        out = {}
        for entry in lock.list_installed():
            ident = entry.get("identifier")
            if ident:
                out[ident] = {
                    "name": entry.get("name"),
                    "trust_level": entry.get("trust_level"),
                    "scan_verdict": entry.get("scan_verdict"),
                }
        return out
    except Exception:
        return {}
