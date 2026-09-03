#!/usr/bin/env python3
"""Skills Tool — list and view skill documents (progressive disclosure).

A skill is a directory holding SKILL.md (YAML frontmatter + instructions) plus
optional references/, templates/, assets/, scripts/. Tier 1 (`skills_list`)
returns name/description only; tier 2-3 (`skill_view`) returns full content
and linked files on demand. Sibling modules (skills_tool_setup / _plugin /
_dedup) re-export every name here so ``from tools.skills_tool import X`` and
``patch("tools.skills_tool.X")`` keep working.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS, is_skill_support_path as _is_skill_support_path)
from tools.skills_tool_setup import (  # noqa: F401
    SkillReadinessStatus, _ENV_VAR_NAME_RE, _REMOTE_ENV_BACKENDS, _build_setup_note,
    _capture_required_environment_variables, _collect_prerequisite_values,
    _gateway_setup_hint, _get_required_environment_variables, _get_terminal_backend_name,
    _is_env_var_persisted, _is_gateway_surface, _is_remote_env_backend,
    _normalize_prerequisite_values, _normalize_setup_metadata,
    _remaining_required_environment_names)
from tools.skills_tool_plugin import (  # noqa: F401
    MAX_DESCRIPTION_LENGTH, MAX_NAME_LENGTH, _INJECTION_PATTERNS, _available_skill_files,
    _fail, _json, _mark_background_review_read, _plugin_skill_linked_files,
    _preprocess_skill, _read_skill_text, _serve_plugin_skill, _serve_skill_file,
    _truncate_description)
from tools.skills_tool_dedup import (  # noqa: F401
    _SKILL_VIEW_DEDUP_CAP, _SKILL_VIEW_DEDUP_MESSAGE, _check_skill_view_dedup,
    _record_skill_view, _skill_view_fingerprint, _skill_view_tracker,
    _skill_view_tracker_lock, reset_skill_view_dedup)

logger = logging.getLogger(__name__)

# Per-session discovery cache: {cache_key: (signature, timestamp, skills_list)}.
# Signature = per-dir max mtime of the dir and its immediate children (add/remove
# inside a category does NOT bump the root mtime) + the disabled set (config-only
# change, no mtime at all) + platform; the TTL bounds staleness from in-place
# SKILL.md edits, which no directory signature can see.
_SKILLS_CACHE: dict = {}
_SKILLS_CACHE_TTL_SECONDS = 30.0


def _skills_scan_signature(dirs_to_scan, disabled) -> tuple:
    """O(#dirs + #categories) stat-based change signature; includes the platform
    ``skill_matches_platform`` will use (via ``agent.skill_utils.sys`` so test
    patches are honored) because the scan result is platform-dependent."""
    from agent import skill_utils as _skill_utils

    platform = getattr(getattr(_skill_utils, "sys", None), "platform", "")
    sig = []
    for d in dirs_to_scan:
        try:
            m = d.stat().st_mtime
        except OSError:
            continue
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            m = max(m, entry.stat(follow_symlinks=False).st_mtime)
                    except OSError:
                        continue
        except OSError:
            pass
        sig.append((str(d), m))
    return (tuple(sig), frozenset(disabled), platform)


# All skills live in ~/.hermes/skills/ (seeded from bundled skills/ on install).
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time: the patched ``SKILLS_DIR`` when a
    test/external patcher changed it, else the live profile-scoped HERMES_HOME
    (long-lived runtimes may import this module before the profile is set)."""
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


_secret_capture_callback = None
_LOOKUP_HINT = "Use a skill name or relative path within the skills directory."


def _skill_lookup_path_error(name: str) -> Optional[str]:
    """Error if lookup *name* could escape the search roots it is joined onto.
    Windows drive paths are rejected too: their ``:`` would be misread as a
    plugin namespace separator."""
    from tools.path_security import has_traversal_component

    if not isinstance(name, str):
        return "Skill name must be a string."
    candidate = name.strip()
    win = PureWindowsPath(candidate)
    if PurePosixPath(candidate).is_absolute() or win.is_absolute() or win.drive:
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


def load_env() -> Dict[str, str]:
    """Load profile-scoped environment variables from HERMES_HOME/.env."""
    env_path = get_hermes_home() / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    # utf-8-sig: a Notepad BOM would otherwise glue U+FEFF onto the first key.
    with env_path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                if line.startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


# Lazy delegates to ``agent.skill_utils`` — public re-exports so existing
# callers (and tests patching either module) don't need updating.
def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    from agent.skill_utils import skill_matches_platform as _impl
    return _impl(frontmatter)


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """Offer-time relevance gate (kanban/docker/s6), NOT a hard-compatibility
    gate; explicit skill loads bypass it."""
    from agent.skill_utils import skill_matches_environment as _impl
    return _impl(frontmatter)


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    from agent.skill_utils import parse_frontmatter
    return parse_frontmatter(content)


def _get_disabled_skill_names() -> Set[str]:
    from agent.skill_utils import get_disabled_skill_names
    return get_disabled_skill_names()


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """``~/.hermes/skills/mlops/axolotl/SKILL.md`` -> ``"mlops"``; active profile
    dir first (respects test monkeypatching), then skills.external_dirs."""
    dirs_to_check = [_skills_dir()]
    try:
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    except Exception:
        pass
    for skills_dir in dirs_to_check:
        try:
            parts = skill_path.relative_to(skills_dir).parts
        except ValueError:
            continue
        if len(parts) >= 3:
            return parts[0]
    return None


def _parse_tags(tags_value) -> List[str]:
    """Tags from frontmatter: a parsed list, a bracket-wrapped string "[a, b]",
    or a comma-separated string "a, b"."""
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]
    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]


def _get_session_platform() -> str:
    """Platform from gateway session context (mirrors
    ``agent.skill_utils.get_disabled_skill_names``)."""
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_PLATFORM") or ""
    except Exception:
        return ""


def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """Disabled in config? Platform precedence: explicit arg, ``HERMES_PLATFORM``
    env, ``HERMES_SESSION_PLATFORM`` from session context. A globally-disabled
    skill stays disabled on every platform; the platform list adds to it (keep in
    sync with agent.skill_utils.get_disabled_skill_names)."""
    try:
        from hermes_cli.config import load_config
        skills_cfg = load_config().get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM") or _get_session_platform()
        global_disabled = skills_cfg.get("disabled", [])
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
            if platform_disabled is not None:
                return name in platform_disabled or name in global_disabled
        return name in global_disabled
    except Exception:
        return False


def _skill_search_dirs() -> Tuple[list, list, Path]:
    """(project_dirs, all_dirs, active_skills_dir). Trusted project-local dirs come
    FIRST so first-wins dedup / the collision resolver prefer them over
    same-named local/external skills."""
    from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs

    project_dirs = list(get_project_skills_dirs())
    all_dirs: list = list(project_dirs)
    active_skills_dir = _skills_dir()
    if active_skills_dir.exists():
        all_dirs.append(active_skills_dir)
    all_dirs.extend(get_external_skills_dirs())
    return project_dirs, all_dirs, active_skills_dir


def _find_all_skills(*, skip_disabled: bool = False) -> List[Dict[str, Any]]:
    """All skills (name, description, category) across local/project/external dirs,
    first-wins by name. ``skip_disabled=True`` returns ALL skills regardless of
    disabled state (used by the ``hermes skills`` config UI). Cached per session."""
    from agent.skill_utils import iter_project_skill_files, iter_skill_index_files

    cache_key = "with_disabled" if skip_disabled else "filtered"
    disabled = set() if skip_disabled else _get_disabled_skill_names()
    project_dirs, dirs_to_scan, _ = _skill_search_dirs()
    signature = _skills_scan_signature(dirs_to_scan, disabled)
    now = time.monotonic()

    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature and (now - cached[1]) < _SKILLS_CACHE_TTL_SECONDS:
        # Shallow copies: callers mutate the returned dicts (web_server annotates
        # s["enabled"]/s["usage"]); handing out cached objects would poison the cache.
        return [dict(s) for s in cached[2]]

    skills = []
    seen_names: set = set()
    # Project dirs iterate through the quarantine chokepoint (scan-time injection gate).
    for scan_dir in dirs_to_scan:
        _iter = iter_project_skill_files if scan_dir in project_dirs else lambda d: iter_skill_index_files(d, "SKILL.md")
        for skill_md in _iter(scan_dir):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                frontmatter, body = _parse_frontmatter(_read_skill_text(skill_md)[:4000])
                if not skill_matches_platform(frontmatter) or not skill_matches_environment(frontmatter):
                    continue
                name = frontmatter.get("name", skill_md.parent.name)[:MAX_NAME_LENGTH]
                if name in seen_names or name in disabled:
                    continue
                description = frontmatter.get("description", "")
                if not description:
                    for line in body.strip().split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            description = line
                            break
                seen_names.add(name)
                skills.append({
                    "name": name,
                    "description": _truncate_description(description),
                    "category": _get_category_from_path(skill_md)})
            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
            except Exception as e:
                logger.debug("Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True)

    # Keyed by the signature computed BEFORE the scan: a write racing the scan
    # changes the signature, so the next call re-scans instead of serving a torn result.
    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(s) for s in skills]


def _sort_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))


def skills_list(category: str = None, task_id: str = None) -> str:
    """Tier 1 listing: name + description (+ category) only, to minimize tokens.
    ``task_id`` is accepted for handler parity."""
    try:
        active_skills_dir = _skills_dir()
        if not active_skills_dir.exists():
            active_skills_dir.mkdir(parents=True, exist_ok=True)

        all_skills = _find_all_skills()
        try:
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            discover_plugins()
            for plugin_skill in get_plugin_manager().list_plugin_skill_metadata():
                frontmatter = plugin_skill.pop("frontmatter", {})
                if not skill_matches_platform(frontmatter) or _is_skill_disabled(plugin_skill["name"]):
                    continue
                all_skills.append(plugin_skill)
        except Exception:
            logger.debug("Plugin skill listing failed", exc_info=True)

        if not all_skills:
            return _json({
                "success": True, "skills": [], "categories": [],
                "message": "No skills found in skills/ directory."})
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]
        all_skills = _sort_skills(all_skills)
        categories = sorted({s.get("category") for s in all_skills if s.get("category")})
        return _json({
            "success": True,
            "skills": all_skills,
            "categories": categories,
            "count": len(all_skills),
            "hint": "Use skill_view(name) to see full content, tags, and linked files"})
    except Exception as e:
        return tool_error(str(e), success=False)


# ── skill_view helpers ─────────────────────────────────────────────────────

def _resolve_plugin_skill(name, file_path, task_id, preprocess):
    """Qualified-name dispatch for ``plugin:skill`` names.

    Returns ``(result_json, None)`` when answered, or ``(None, local_category_name)``
    to fall through to the flat-tree scan: categorized local skills also use
    ``category:skill`` in config and gateway prompts, so the on-disk
    ``category/skill`` translation is returned (None when there's no bare part)."""
    from agent.skill_utils import is_valid_namespace, parse_qualified_name
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    namespace, bare = parse_qualified_name(name)
    if not is_valid_namespace(namespace):
        return _fail(f"Invalid namespace '{namespace}' in '{name}'. Namespaces must match [a-zA-Z0-9_-]+."), None

    discover_plugins()  # idempotent
    pm = get_plugin_manager()
    active_memory_provider = None
    try:
        from plugins.memory import _get_active_memory_provider, _prune_inactive_memory_provider_skills
        active_memory_provider = _get_active_memory_provider()
        _prune_inactive_memory_provider_skills(active_memory_provider)
    except Exception as exc:
        logger.debug("Failed pruning inactive memory-provider skills: %s", exc)

    plugin_skill_md = pm.find_plugin_skill(name)
    # Memory providers load through plugins.memory, not the general PluginManager:
    # load the namespaced provider once so its collector can forward its skills
    # into the registry before declaring the qualified skill missing.
    if plugin_skill_md is None:
        try:
            if namespace == active_memory_provider:
                from plugins.memory import load_memory_provider

                load_memory_provider(namespace)
                plugin_skill_md = pm.find_plugin_skill(name)
        except Exception as exc:
            logger.debug("Failed lazy memory-provider skill load for %s: %s", namespace, exc)

    if plugin_skill_md is not None:
        if not plugin_skill_md.exists():
            pm.remove_plugin_skill(name)  # stale registry entry — file deleted out of band
            return _fail(
                f"Skill '{name}' file no longer exists at {plugin_skill_md}. The registry entry "
                f"has been cleaned up — try again after the plugin is reloaded."
            ), None
        return _serve_plugin_skill(
            plugin_skill_md, namespace, bare,
            file_path=file_path, preprocess=preprocess, session_id=task_id,
        ), None

    available = pm.list_plugin_skills(namespace)
    if available:  # plugin exists but this specific skill is missing
        return _fail(
            f"Skill '{bare}' not found in plugin '{namespace}'.",
            available_skills=[f"{namespace}:{s}" for s in available],
            hint=f"The '{namespace}' plugin provides {len(available)} skill(s).",
        ), None
    return None, (f"{namespace}/{bare}" if bare else None)  # plugin not found → local scan


def _under_any(path: Path, dirs) -> bool:
    """True when ``path`` (resolved where possible) lives under one of ``dirs``."""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    for d in dirs:
        try:
            resolved.relative_to(d)
            return True
        except ValueError:
            continue
    return False


def _collect_skill_candidates(name, local_category_name, all_dirs):
    """ALL (skill_dir, skill_md) candidates across every dir using every lookup
    strategy (direct path, recursive by parent dir / frontmatter name, legacy flat
    <name>.md), deduped by resolved path. Collision detection is the point: silent
    shadowing of a local skill by a same-named external one is a real bug class,
    so the caller refuses when more than one matches."""
    from agent.skill_utils import iter_skill_index_files

    candidates: List[Tuple[Optional[Path], Path]] = []
    seen_md: set = set()

    def _record(sd: Optional[Path], smd: Path) -> None:
        try:
            key = smd.resolve()
        except Exception:
            key = smd
        if key not in seen_md:
            seen_md.add(key)
            candidates.append((sd, smd))

    def _record_direct(direct_path: Path) -> None:
        # Direct path ("mlops/axolotl" or bare "axolotl") or its legacy flat .md sibling.
        flat = direct_path.with_suffix(".md")
        if not _is_skill_support_path(direct_path) and direct_path.is_dir() and (direct_path / "SKILL.md").exists():
            _record(direct_path, direct_path / "SKILL.md")
        elif flat.exists() and not _is_skill_support_path(flat):
            _record(None, flat)

    for search_dir in all_dirs:
        _record_direct(search_dir / name)
        if local_category_name:  # "myplugin:explore" with no such plugin → "myplugin/explore"
            _record_direct(search_dir / local_category_name)
        # Recursive by directory name plus frontmatter `name:` — skills_list()
        # exposes the frontmatter name, so skill_view(name) must accept it too.
        for found_skill_md in iter_skill_index_files(search_dir, "SKILL.md"):
            if found_skill_md.parent.name == name:
                _record(found_skill_md.parent, found_skill_md)
                continue
            try:
                fm, _ = _parse_frontmatter(_read_skill_text(found_skill_md))
            except Exception:
                fm = {}
            if fm.get("name") == name:
                _record(found_skill_md.parent, found_skill_md)
        # Legacy flat <name>.md anywhere under the dir; support docs are excluded
        # (they load via file_path and must not shadow real skills sharing the basename).
        for found_md in search_dir.rglob(f"{name}.md"):
            if found_md.name != "SKILL.md" and not _is_skill_support_path(found_md):
                _record(None, found_md)
    return candidates


_TEMPLATE_GLOBS = ["*.md", "*.py", "*.yaml", "*.yml", "*.json", "*.tex", "*.sh"]
_SCRIPT_GLOBS = ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"]


def _skill_linked_files(skill_dir: Optional[Path]) -> dict:
    """references/templates/assets/scripts of a directory skill (empty groups dropped)."""
    if not skill_dir:
        return {}

    def _rel(paths):
        return [str(f.relative_to(skill_dir)) for f in paths]
    def _multi(sub: Path, globs, rglob: bool):
        out: list = []
        for ext in globs:
            out.extend(_rel(sub.rglob(ext) if rglob else sub.glob(ext)))
        return out

    refs, tmpl, assets, scripts = (skill_dir / n for n in ("references", "templates", "assets", "scripts"))
    files = {
        "references": _rel(refs.glob("*.md")) if refs.exists() else [],
        "templates": _multi(tmpl, _TEMPLATE_GLOBS, True) if tmpl.exists() else [],
        "assets": _rel(f for f in assets.rglob("*") if f.is_file()) if assets.exists() else [],
        "scripts": _multi(scripts, _SCRIPT_GLOBS, False) if scripts.exists() else []}
    return {k: v for k, v in files.items() if v}


def _org_provenance_header(skill_dir: Path, active_skills_dir: Path):
    """(org_provenance dict, header text) for an org-mirror skill, else (None, "").
    Provenance is announced IN the content the model consumes; the author is
    token-verified at push time by the sync plane, so the header is trustworthy."""
    from agent.skill_utils import ORG_PROVENANCE_FILE, is_org_mirror_path, org_id_of_path

    if not is_org_mirror_path(skill_dir, active_skills_dir):
        return None, ""
    prov_org = org_id_of_path(skill_dir, active_skills_dir)
    author = ts = ""
    if prov_org:
        try:
            prov = json.loads(_read_skill_text(active_skills_dir / "_org" / prov_org / ORG_PROVENANCE_FILE))
            author = str(prov.get("author_device") or prov.get("author_user_id") or "")
            ts = str(prov.get("ts") or "")
        except Exception:
            pass
    header = (
        "> [!NOTE] ORG-SHARED SKILL — provenance\n"
        f"> This skill is shared by your organisation (org `{prov_org}`"
        + (f", last updated by `{author}`" if author else "")
        + (f", as of {ts}" if ts else "")
        + "). It was reviewed and approved for the whole\n"
        "> team — treat it as third-party instructions rather than your own notes.\n"
        "> You MAY improve it in place like any other skill. Your edits are kept locally\n"
        "> and are never overwritten by org updates; share them back with\n"
        "> `hermes sync propose` (or automatically, if your org enables it).\n\n")
    return {"org_id": prov_org, "shared_by": author or None, "as_of": ts or None}, header


def _skill_readiness(frontmatter: Dict[str, Any], skill_name: str) -> Tuple[dict, dict]:
    """Resolve required env vars / credential files (prompting for secrets where
    the surface allows) and register what's available for sandboxes.

    Returns ``(fields, extras)``: ``fields`` go before ``_source_path`` in the
    skill_view result, ``extras`` (setup_help, gateway_setup_hint, setup_note)
    after it — result key order is part of the tool output."""
    legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    required_env_vars = _get_required_environment_variables(frontmatter, legacy_env_vars)
    backend = _get_terminal_backend_name()
    env_snapshot = load_env()
    missing_required_env_vars = [
        e for e in required_env_vars
        if not e.get("optional") and not _is_env_var_persisted(e["name"], env_snapshot)]
    capture_result = _capture_required_environment_variables(skill_name, missing_required_env_vars)
    if missing_required_env_vars:
        env_snapshot = load_env()
    remaining = _remaining_required_environment_names(required_env_vars, capture_result, env_snapshot=env_snapshot)
    setup_needed = bool(remaining)

    # Only vars actually set pass through to sandboxed execution (execute_code, terminal).
    available_env_names = [e["name"] for e in required_env_vars if e["name"] not in remaining]
    if available_env_names:
        try:
            from tools.env_passthrough import register_env_passthrough

            register_env_passthrough(available_env_names)
        except Exception:
            logger.debug("Could not register env passthrough for skill %s", skill_name, exc_info=True)

    # Credential files for mounting into remote sandboxes: existing host files are
    # registered, missing ones flag setup_needed.
    required_cred_files_raw = frontmatter.get("required_credential_files", [])
    missing_cred_files: list = []
    if isinstance(required_cred_files_raw, list) and required_cred_files_raw:
        try:
            from tools.credential_files import register_credential_files

            missing_cred_files = register_credential_files(required_cred_files_raw)
            if missing_cred_files:
                setup_needed = True
        except Exception:
            logger.debug("Could not register credential files for skill %s", skill_name, exc_info=True)

    fields = {
        "required_environment_variables": required_env_vars,
        "required_commands": [],
        "missing_required_environment_variables": remaining,
        "missing_credential_files": missing_cred_files,
        "missing_required_commands": [],
        "setup_needed": setup_needed,
        "setup_skipped": capture_result["setup_skipped"],
        "readiness_status": (
            SkillReadinessStatus.SETUP_NEEDED if setup_needed else SkillReadinessStatus.AVAILABLE
        ).value}
    extras: dict = {}
    setup_help = next((e["help"] for e in required_env_vars if e.get("help")), None)
    if setup_help:
        extras["setup_help"] = setup_help
    if capture_result["gateway_setup_hint"]:
        extras["gateway_setup_hint"] = capture_result["gateway_setup_hint"]
    if setup_needed:
        missing_items = [f"env ${n}" for n in remaining] + [f"file {p}" for p in missing_cred_files]
        setup_note = _build_setup_note(SkillReadinessStatus.SETUP_NEEDED, missing_items, setup_help)
        if _is_remote_env_backend(backend) and setup_note:
            setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
        if setup_note:
            extras["setup_note"] = setup_note
    return fields, extras


@dataclass
class _Located:
    """Outcome of the local skill lookup phase of ``skill_view``."""

    project_dirs: list
    all_dirs: list
    active_skills_dir: Path
    skill_dir: Optional[Path] = None
    skill_md: Any = None  # set whenever error is None
    error: Optional[str] = None  # JSON failure payload to return as-is


def _locate_skill(name: str, local_category_name: Optional[str]) -> _Located:
    """Find the unique on-disk skill for *name*: collision refusal, project-tier
    precedence, quarantine gate, not-found listing."""
    project_dirs, all_dirs, active_skills_dir = _skill_search_dirs()
    loc = _Located(project_dirs, all_dirs, active_skills_dir)
    if not all_dirs:
        loc.error = _fail("Skills directory does not exist yet. It will be created on first install.")
        return loc

    candidates = _collect_skill_candidates(name, local_category_name, all_dirs)
    if len(candidates) > 1 and project_dirs:
        # A project skill intentionally overrides a same-named local/external skill,
        # so narrow to project candidates when any exist. Ambiguity WITHIN that tier still refuses.
        project_candidates = [(sd, smd) for sd, smd in candidates if _under_any(smd, project_dirs)]
        if project_candidates:
            candidates = project_candidates
    if len(candidates) > 1:
        paths = [str(smd) for _, smd in candidates]
        logger.warning("Skill name collision for '%s': %d candidates — %s", name, len(candidates), "; ".join(paths))
        loc.error = _fail(
            f"Ambiguous skill name '{name}': {len(candidates)} skills match across your local skills dir "
            "and external_dirs. Refusing to guess — load one explicitly by its categorized path.",
            matches=paths,
            hint="Pass the full relative path instead of the bare name (e.g., 'category/skill-name'), "
            "or rename one of the colliding skills so each name is unique.")
        return loc
    if candidates:
        loc.skill_dir, loc.skill_md = candidates[0]

    # Quarantine gate: a project-tier skill with a dangerous scan verdict must not
    # load even by explicit name (same chokepoint the index and skills_list use).
    if loc.skill_md is not None and project_dirs:
        from agent.skill_utils import is_quarantined_project_skill

        if _under_any(loc.skill_md, project_dirs) and is_quarantined_project_skill(loc.skill_md):
            loc.error = _fail(
                f"Project skill '{name}' is quarantined: the security scan flagged its content as "
                "dangerous. It will not load until the repo's skill content changes and passes a re-scan.",
                hint="Inspect the skill in the repo checkout, or untrust the repo with `hermes skills untrust`.",
            )
            return loc

    if not loc.skill_md or not loc.skill_md.exists():
        available = [s["name"] for s in _sort_skills(_find_all_skills())[:20]]
        loc.error = _fail(f"Skill '{name}' not found.", available_skills=available, hint="Use skills_list to see all available skills")
    return loc


def _log_security_warnings(name: str, skill_md: Path, content: str, loc: _Located) -> None:
    """Warn when loaded from outside the trusted dirs (project + local + external)
    and/or when common prompt-injection patterns appear. Never blocks."""
    trusted_dirs = [loc.active_skills_dir.resolve()]
    try:
        trusted_dirs.extend(d.resolve() for d in loc.all_dirs)
    except Exception:
        pass
    warnings = []
    if not _under_any(skill_md, trusted_dirs):
        warnings.append(f"skill file is outside the trusted skills directory (~/.hermes/skills/): {skill_md}")
    content_lower = content.lower()
    if any(p in content_lower for p in _INJECTION_PATTERNS):
        warnings.append("skill content contains patterns that may indicate prompt injection")
    if warnings:
        logger.warning("Skill security warning for '%s': %s", name, "; ".join(warnings))


def skill_view(
    name: str, file_path: str = None, task_id: str = None, preprocess: bool = True) -> str:
    """View a skill (SKILL.md) or a specific file within its directory, as JSON.

    ``name`` is a skill name or path ("axolotl", "03-fine-tuning/axolotl"); the
    qualified form "plugin:skill" resolves plugin-provided skills. ``preprocess``
    applies the configured SKILL.md template / inline shell rendering; internal
    slash/preload callers disable it because they render the skill message themselves."""
    try:
        # Validate before the ':' dispatch so a Windows drive path (C:\skills\foo)
        # can't be reinterpreted as a plugin namespace.
        lookup_error = _skill_lookup_path_error(name)
        if lookup_error:
            return _fail(lookup_error, hint=_LOOKUP_HINT)

        local_category_name: str | None = None
        if ":" in name:  # plugin registry; bare names use the flat-tree scan below
            served, local_category_name = _resolve_plugin_skill(name, file_path, task_id, preprocess)
            if served is not None:
                return served
        # The fall-through form (namespace/bare) joins onto each search dir too;
        # re-validate it since `bare` is not namespace-checked.
        if local_category_name:
            lookup_error = _skill_lookup_path_error(local_category_name)
            if lookup_error:
                return _fail(lookup_error, hint=_LOOKUP_HINT)

        loc = _locate_skill(name, local_category_name)
        if loc.error is not None:
            return loc.error
        skill_dir, skill_md, active_skills_dir = loc.skill_dir, loc.skill_md, loc.active_skills_dir

        try:  # read once — reused for platform check and main content
            content = _read_skill_text(skill_md)
        except Exception as e:
            return _fail(f"Failed to read skill '{name}': {e}")
        _log_security_warnings(name, skill_md, content, loc)

        try:
            frontmatter, _ = _parse_frontmatter(content)
        except Exception:
            frontmatter = {}

        if not skill_matches_platform(frontmatter):
            return _fail(f"Skill '{name}' is not supported on this platform.", readiness_status=SkillReadinessStatus.UNSUPPORTED.value)
        resolved_name = frontmatter.get("name", skill_md.parent.name)
        if _is_skill_disabled(resolved_name):
            return _fail(f"Skill '{resolved_name}' is disabled. Enable it with `hermes skills` or inspect the files directly on disk.")

        if file_path and skill_dir:
            return _serve_skill_file(
                skill_dir, file_path, name,
                hint="Use a relative path within the skill directory",
                list_available=True, mark_read=True)

        # tags/related_skills: metadata.hermes.* (agentskills.io) first, then top-level.
        metadata = frontmatter.get("metadata")
        hermes_meta = (metadata.get("hermes", {}) or {}) if isinstance(metadata, dict) else {}
        tags = _parse_tags(hermes_meta.get("tags") or frontmatter.get("tags", ""))
        related_skills = _parse_tags(hermes_meta.get("related_skills") or frontmatter.get("related_skills", ""))
        linked_files = _skill_linked_files(skill_dir)

        try:
            rel_path = str(skill_md.relative_to(active_skills_dir))
        except ValueError:  # external skill — relative to its own parent dir
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get("name", skill_md.stem if not skill_dir else skill_dir.name)
        readiness, readiness_extras = _skill_readiness(frontmatter, skill_name)

        rendered_content = content
        if preprocess:
            rendered_content = _preprocess_skill(
                content, skill_dir, task_id, "Could not preprocess skill content for %s", skill_name,
            )
        org_provenance = None
        if skill_dir:
            try:
                org_provenance, header = _org_provenance_header(skill_dir, active_skills_dir)
                rendered_content = header + rendered_content
            except Exception:
                logger.debug("Could not resolve org provenance for %s", skill_name, exc_info=True)

        result = {
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": tags,
            "related_skills": related_skills,
            "content": rendered_content,
            "path": rel_path,
            "skill_dir": str(skill_dir) if skill_dir else None,
            "org_provenance": org_provenance,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'" if linked_files else None,
            **readiness,
            # Internal: absolute source path for the repeat-view dedup fingerprint.
            "_source_path": str(skill_md),
            **readiness_extras}
        _mark_background_review_read(skill_md)
        # agentskills.io optional fields
        if frontmatter.get("compatibility"):
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata
        return _json(result)
    except Exception as e:
        return tool_error(str(e), success=False)


# ── Registry ────────────────────────────────────────────────────────────────

SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans').",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(category=args.get("category"), task_id=kw.get("task_id")),
    check_fn=check_skills_requirements,
    emoji="📚")


def _skill_view_with_bump(args, **kw):
    """Invoke skill_view, then bump view_count/use on success (best-effort).

    Repeat-view dedup mirrors read_file's unchanged-stub: when this session already
    loaded the SAME unchanged skill file, return a short stub instead of the full
    content. The stub only replaces content already present earlier in this
    conversation, and the cache is cleared on context compression."""
    name = args.get("name", "")
    task_id = kw.get("task_id")
    stub = _check_skill_view_dedup(task_id, name, args.get("file_path"))
    if stub is not None:
        return stub
    result = skill_view(name, file_path=args.get("file_path"), task_id=task_id)
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("success"):
            _record_skill_view(task_id, name, args.get("file_path"), parsed)
            # Qualified forms ("plugin:skill") return with the canonical name.
            resolved = parsed.get("name") or name
            if resolved:
                from tools.skill_usage import bump_use, bump_view
                bump_view(str(resolved))
                # Viewing is the agent actively loading the skill to act on it — that
                # counts as use; the curator's stale timer keys off last_used_at.
                bump_use(str(resolved), task_id=kw.get("task_id"), session_id=kw.get("session_id"))
    except Exception:
        pass
    return result


registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=_skill_view_with_bump,
    check_fn=check_skills_requirements,
    emoji="📚")
