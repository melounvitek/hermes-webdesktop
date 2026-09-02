"""Skill bundles — aliases that load multiple skills under one slash command.

Bundles are YAML files in ``<HERMES_HOME>/skill-bundles/`` (``name``,
``description``, ``skills: [...]``, optional ``instruction``; the file stem is
the fallback name). ``/<bundle>`` loads every member skill into one user
message. If a bundle and a skill share a slug, the bundle wins — slash dispatch
checks bundles first, on purpose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home
from agent.skill_commands import diff_command_snapshots, slugify_skill_name as _slugify

logger = logging.getLogger(__name__)

_bundles_cache: Dict[str, Dict[str, Any]] = {}
_bundles_cache_mtime: Optional[float] = None


def _bundles_dir() -> Path:
    """Bundles directory: ``HERMES_BUNDLES_DIR`` override (tests) or ``<HERMES_HOME>/skill-bundles``."""
    override = os.environ.get("HERMES_BUNDLES_DIR")
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "skill-bundles"


def _iter_bundle_files() -> List[Path]:
    base = _bundles_dir()
    if not base.exists():
        return []
    files: List[Path] = []
    for ext in ("*.yaml", "*.yml"):
        files.extend(sorted(base.glob(ext)))
    return files


def _max_mtime(files: List[Path]) -> float:
    """Highest mtime across the bundle files plus the dir itself (dir mtime catches deletions)."""
    mtimes = []
    for f in [_bundles_dir(), *files]:
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def _load_bundle_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse one bundle YAML; ``None`` (logged) on any error so a broken bundle can't break discovery."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read bundle %s: %s", path, exc)
        return None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in bundle %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Bundle %s is not a mapping; skipping", path)
        return None

    name = str(data.get("name") or path.stem).strip()
    if not name:
        logger.warning("Bundle %s has no name; skipping", path)
        return None

    skills = data.get("skills") or []
    if not isinstance(skills, list) or not skills:
        logger.warning("Bundle %s has no skills list; skipping", path)
        return None
    skills = [str(s).strip() for s in skills if str(s).strip()]
    if not skills:
        logger.warning("Bundle %s has empty skills list; skipping", path)
        return None

    description = str(data.get("description") or "").strip()
    instruction = str(data.get("instruction") or "").strip()

    slug = _slugify(name)
    if not slug:
        logger.warning("Bundle %s yielded empty slug; skipping", path)
        return None

    return {
        "name": name,
        "slug": slug,
        "description": description or f"Load {len(skills)} skills as a bundle",
        "skills": skills,
        "instruction": instruction,
        "path": str(path),
    }


def scan_bundles() -> Dict[str, Dict[str, Any]]:
    """Rebuild the ``"/slug"`` -> bundle info cache; duplicate slugs keep the first (alphabetical)."""
    global _bundles_cache, _bundles_cache_mtime
    files = _iter_bundle_files()
    out: Dict[str, Dict[str, Any]] = {}
    for f in files:
        info = _load_bundle_file(f)
        if not info:
            continue
        key = f"/{info['slug']}"
        if key in out:
            logger.warning(
                "Duplicate bundle slug %s from %s; keeping %s",
                key, f, out[key]["path"],
            )
            continue
        out[key] = info
    _bundles_cache = out
    _bundles_cache_mtime = _max_mtime(files)
    return out


def get_skill_bundles() -> Dict[str, Dict[str, Any]]:
    """Current bundle mapping; rescans only when a bundle file or the dir mtime changed."""
    current_mtime = _max_mtime(_iter_bundle_files())
    if not _bundles_cache or _bundles_cache_mtime != current_mtime:
        scan_bundles()
    return _bundles_cache


def resolve_bundle_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed command to its ``/slug`` key (``_`` ≡ ``-``, as Telegram rewrites hyphens)."""
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in get_skill_bundles() else None


def reload_bundles() -> Dict[str, Any]:
    """Re-scan and return an ``added``/``removed``/``unchanged``/``total`` diff (same shape as reload_skills)."""
    def _snapshot(cmds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        return {k.lstrip("/"): (v or {}).get("description", "") for k, v in cmds.items()}

    before = _snapshot(_bundles_cache)
    return diff_command_snapshots(before, _snapshot(scan_bundles()))


def list_bundles() -> List[Dict[str, Any]]:
    """Return a sorted list of bundle info dicts for display."""
    return sorted(get_skill_bundles().values(), key=lambda b: b["slug"])


def build_bundle_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    platform: str | None = None,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """Build the user message for a bundle invocation.

    Returns ``(message, loaded_skill_names, missing_skill_names)`` or ``None``
    if the bundle wasn't found. Uninstalled members are skipped with a note.
    Disabled members are skipped too: bundles load via ``_load_skill_payload``,
    bypassing the scan-time disabled filter, so the list is re-applied here.
    ``platform`` scopes that check (gateway passes it; None resolves from env).
    """
    info = get_skill_bundles().get(cmd_key)
    if not info:
        return None

    # Late import keeps skill_bundles cheap to import (no tools/* at import time).
    from agent.skill_commands import _load_skill_payload, _render_skill_block, _scaffold_header

    try:
        from agent.skill_utils import get_disabled_skill_names
        disabled_names = get_disabled_skill_names(platform=platform)
    except Exception:
        disabled_names = set()

    loaded_names: List[str] = []
    missing: List[str] = []
    disabled: List[str] = []
    skill_blocks: List[str] = []
    seen: set[str] = set()

    bundle_name = info["name"]

    for skill_id in info["skills"]:
        identifier = (skill_id or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue
        skill_name = loaded[2]

        # Gate on the loaded skill's canonical name (identifiers may be paths or aliases).
        if skill_name in disabled_names or identifier in disabled_names:
            disabled.append(skill_name or identifier)
            continue

        skill_blocks.append(_render_skill_block(
            loaded,
            f'[Loaded as part of the "{bundle_name}" skill bundle.]',
            task_id,
        ))
        loaded_names.append(skill_name)

    if not skill_blocks:
        return None

    header = _scaffold_header(
        f'"{bundle_name}" skill bundle',
        loaded_names,
        lead_lines=[f"Bundle: {bundle_name}"],
        missing=missing,
        disabled=disabled,
        extra_instruction=info.get("instruction") or "",
        user_instruction=user_instruction,
    )
    return ("\n\n".join([header, *skill_blocks]), loaded_names, missing)


# ── File-level CRUD — used by `hermes bundles` ─────────────────────────────


def bundle_path_for(name: str) -> Path:
    """Return the canonical filesystem path for a bundle name."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Bundle name {name!r} normalizes to an empty slug")
    return _bundles_dir() / f"{slug}.yaml"


def save_bundle(
    name: str,
    skills: List[str],
    description: str = "",
    instruction: str = "",
    overwrite: bool = False,
) -> Path:
    """Write a bundle to disk and refresh the cache.

    Raises ``FileExistsError`` if the target exists and not ``overwrite``;
    ``ValueError`` for unusable inputs.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Bundle name is required")
    cleaned_skills = [str(s).strip() for s in skills if str(s).strip()]
    if not cleaned_skills:
        raise ValueError("Bundle must reference at least one skill")

    path = bundle_path_for(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Bundle already exists at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"name": name, "skills": cleaned_skills}
    if description:
        payload["description"] = description
    if instruction:
        payload["instruction"] = instruction

    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    scan_bundles()
    return path


def delete_bundle(name: str) -> Path:
    """Delete a bundle by name and return its path; ``FileNotFoundError`` if absent."""
    path = bundle_path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No bundle at {path}")
    path.unlink()
    scan_bundles()
    return path


def get_bundle(name: str) -> Optional[Dict[str, Any]]:
    """Look up a bundle by name (slug-normalized)."""
    return get_skill_bundles().get(f"/{_slugify(name)}")
