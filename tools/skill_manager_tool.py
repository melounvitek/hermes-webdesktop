#!/usr/bin/env python3
"""
Skill Manager Tool -- Agent-Managed Skill Creation & Editing

Lets the agent create, patch, and delete skills — its procedural memory
(narrow, actionable "how to do X"), as opposed to MEMORY.md/USER.md (broad,
declarative). New skills land in ~/.hermes/skills/ (or ``skills.create_dir``);
existing skills (bundled, hub-installed, user-created) are modified in place.

Actions: create, edit (legacy full rewrite), patch, delete, write_file,
remove_file. Layout: ``<skills>/[category/]<skill>/SKILL.md`` plus optional
``references/ templates/ scripts/ assets/`` subdirs.
"""

import contextvars as _ctxvars
import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home, display_hermes_home
from utils import atomic_write_text, is_truthy_value
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    parse_frontmatter as _parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT,
)
from tools.skill_manager_guards import (  # noqa: F401 — re-exported for callers/tests
    _BackgroundReviewReadMarks,
    _background_review_has_read,
    _background_review_preflight,
    _background_review_read_before_write_guard,
    _background_review_read_paths,
    _background_review_write_guard,
    _containing_skills_root,
    _curator_consolidation_delete_guard,
    _is_path_redirect,
    _maybe_auto_propose_org_edit,
    _org_mirror_write_guard,
    _pinned_guard,
    _reset_background_review_read_marks,
    _validate_delete_target,
    _is_background_review,
    mark_background_review_skill_read,
)
from tools.skill_manager_batch import (  # noqa: F401
    _BATCH_MAX_OPS,
    _BATCH_OP_ACTIONS,
    _skill_manage_batch,
)

logger = logging.getLogger(__name__)

# External hub installs are always scanned; agent-created skills only when
# skills.guard_agent_created is on.
try:
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report
    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False


def _guard_agent_created_enabled() -> bool:
    """skills.guard_agent_created (default False): the agent can already run
    the same code via terminal() ungated, so the scan is opt-in belt-and-suspenders."""
    try:
        from hermes_cli.config import load_config
        return is_truthy_value(
            cfg_get(load_config(), "skills", "guard_agent_created"),
            default=False,
        )
    except Exception:
        return False


def _security_scan_skill(skill_dir: Path) -> Optional[str]:
    """Scan a skill directory after write. Returns error string if blocked, else None.

    No-op when skills.guard_agent_created is disabled (the default). An "ask"
    verdict means dangerous findings for an agent-created skill — surfaced as
    an error so the agent can retry with the flagged content removed.
    """
    if not _GUARD_AVAILABLE or not _guard_agent_created_enabled():
        return None
    try:
        result = scan_skill(skill_dir, source="agent-created")
        allowed, reason = should_allow_install(result)
        if allowed is None:
            logger.warning("Agent-created skill blocked (dangerous findings): %s", reason)
        if allowed is not True:
            return f"Security scan blocked this skill ({reason}):\n{format_scan_report(result)}"
    except Exception as e:
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
    return None


# All skills live in ~/.hermes/skills/ (single source of truth)
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills directory at call time.

    Long-lived multi-profile runtimes import this module once and later bind a
    different profile per session. Honor an explicitly patched module-level
    ``SKILLS_DIR`` (tests), otherwise resolve from the live HERMES_HOME.
    """
    configured = Path(SKILLS_DIR)
    if configured != _SKILLS_DIR_AT_IMPORT:
        return configured
    return get_hermes_home() / "skills"


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000   # ~36k tokens at 2.75 chars/token
MAX_SKILL_FILE_BYTES = 1_048_576    # 1 MiB per supporting file
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')  # filesystem-safe, URL-friendly
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}  # for write_file/remove_file
_FRONTMATTER_END_RE = re.compile(r'\n---\s*\n')


def _display_create_dir() -> str:
    """Display string for the skill-creation directory (schema/instruction text);
    follows ``skills.create_dir`` when configured."""
    try:
        from agent.skill_utils import display_skill_create_dir
        return display_skill_create_dir()
    except Exception:
        return f"{display_hermes_home()}/skills/"


def _err(message: str) -> Dict[str, Any]:
    return {"success": False, "error": message}


# =============================================================================
# Validation helpers
# =============================================================================

def _validate_name(name: str) -> Optional[str]:
    """Validate a skill name. Returns error message or None if valid."""
    if not name:
        return "Skill name is required."
    if len(name) > MAX_NAME_LENGTH:
        return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.match(name):
        return (
            f"Invalid skill name '{name}'. Use lowercase letters, numbers, "
            f"hyphens, dots, and underscores. Must start with a letter or digit."
        )
    return None


def _validate_category(category: Optional[str]) -> Optional[str]:
    """Validate an optional category name used as a single directory segment."""
    if category is None:
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    category = category.strip()
    if not category:
        return None
    invalid = (
        f"Invalid category '{category}'. Use lowercase letters, numbers, "
        "hyphens, dots, and underscores. Categories must be a single directory name."
    )
    if "/" in category or "\\" in category:
        return invalid
    if len(category) > MAX_NAME_LENGTH:
        return f"Category exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(category) else invalid


def _validate_frontmatter(content: str, *, new_skill: bool = False) -> Optional[str]:
    """Validate SKILL.md frontmatter (name + description) and a non-empty body.

    ``new_skill`` (create path only) also enforces the SKILL_PROMPT_DESC_LIMIT
    budget so new skills never lose routing signal to index truncation; edit
    and patch skip it so existing over-limit skills remain maintainable.
    """
    if not content.strip():
        return "Content cannot be empty."
    content = content.lstrip("\ufeff")  # tolerate a Windows UTF-8 BOM
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
    end_match = _FRONTMATTER_END_RE.search(content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
    try:
        parsed = yaml.safe_load(content[3:end_match.start() + 3])
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"
    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    if "name" not in parsed:
        return "Frontmatter must include 'name' field."
    if "description" not in parsed:
        return "Frontmatter must include 'description' field."
    desc = str(parsed["description"])
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if new_skill and len(desc.strip().strip("'\"")) > SKILL_PROMPT_DESC_LIMIT:
        return (
            f"Description is {len(desc.strip())} chars — new skills must fit the "
            f"{SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget (one sentence, "
            f"trigger first, ends with a period). The skill index truncates "
            f"longer descriptions to {SKILL_PROMPT_DESC_LIMIT - 3} chars + '...', "
            f"destroying the routing signal. Move detail into the skill body."
        )
    if not content[end_match.end() + 3:].strip():
        return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."
    return None


def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
    """Error message when content exceeds the agent-write character limit, else None."""
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        return (
            f"{label} content is {len(content):,} characters "
            f"(limit: {MAX_SKILL_CONTENT_CHARS:,}). "
            f"Consider splitting into a smaller SKILL.md with supporting files "
            f"in references/ or templates/."
        )
    return None


def _description_preview(content: str) -> str:
    """First 120 chars of the frontmatter description (verbose notifications); '' on any failure."""
    try:
        fm_end = _FRONTMATTER_END_RE.search(content[3:])
        if fm_end:
            parsed = yaml.safe_load(content[3:fm_end.start() + 3])
            return str(parsed.get("description", ""))[:120]
    except Exception:
        pass
    return ""


def _resolve_skill_dir(name: str, category: str = None) -> Path:
    """Directory for a new skill; honors ``skills.create_dir`` (e.g. a shared
    fleet directory) and falls back to the profile-local skills dir."""
    base = _skills_dir()
    try:
        from agent.skill_utils import get_skill_create_dir
        create_dir = get_skill_create_dir()
        if create_dir is not None:
            base = create_dir
    except Exception:
        logger.debug("skills.create_dir lookup failed", exc_info=True)
    return base / category / name if category else base / name


def _iter_skill_dirs(root: Path):
    """Yield every non-excluded skill directory (parent of a SKILL.md) under ``root``."""
    from agent.skill_utils import is_excluded_skill_path

    for skill_md in root.rglob("SKILL.md"):
        if not is_excluded_skill_path(skill_md):
            yield skill_md.parent


def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """Find a skill by name across the local skills dir then skills.external_dirs.

    Accepts the bare directory name (``axolotl``) and the categorized relative
    path (``mlops/axolotl``) — the two forms skill_view resolves. Bare lookups
    compare the skill's own dir name so category-nested skills still match.
    Returns ``{"path": Path}`` or None.
    """
    from agent.skill_utils import get_all_skills_dirs

    # The categorized form matches the skill dir RELATIVE to the local root
    # (never external dirs — relative_to raises there).
    local_root = None
    if "/" in name or "\\" in name:
        try:
            local_root = _skills_dir().resolve()
        except OSError:
            logger.debug(
                "skills dir resolve failed; categorized lookups fall back to the unresolved path",
                exc_info=True,
            )
            local_root = _skills_dir()

    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_dir in _iter_skill_dirs(skills_dir):
            if skill_dir.name == name:
                return {"path": skill_dir}
            if local_root is not None:
                try:
                    rel = skill_dir.resolve().relative_to(local_root)
                except ValueError:
                    continue
                if rel.as_posix() == name:  # POSIX form so it works on Windows too
                    return {"path": skill_dir}
    return None


def _find_skill_in_other_profiles(name: str) -> List[Tuple[str, Path]]:
    """``(profile_name, skill_dir)`` pairs for OTHER profiles holding ``name``.

    Lets the "not found" error explain a wrong-profile mistake. Fail-quiet:
    empty list when discovery fails.
    """
    matches: List[Tuple[str, Path]] = []
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
    except Exception:
        return matches

    _active = _skills_dir()
    active_dir = _active.resolve() if _active.exists() else _active

    # Every profile's skills dir EXCEPT the active one (already searched).
    candidates: List[Tuple[str, Path]] = [("default", root / "skills")]
    profiles_root = root / "profiles"
    try:
        if profiles_root.is_dir():
            candidates += [(e.name, e / "skills") for e in profiles_root.iterdir() if e.is_dir()]
    except OSError:
        pass

    for profile_name, skills_dir in candidates:
        try:
            if skills_dir.resolve() == active_dir or not skills_dir.is_dir():
                continue
            for skill_dir in _iter_skill_dirs(skills_dir):
                if skill_dir.name == name:
                    matches.append((profile_name, skill_dir))
                    break  # one match per profile is enough
        except (OSError, RuntimeError):
            continue
    return matches


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    """"Skill not found" error naming other profiles that hold the skill;
    ``suffix`` is appended after the hint."""
    from agent.file_safety import _resolve_active_profile_name
    active = _resolve_active_profile_name()
    base = f"Skill '{name}' not found in active profile '{active}'."

    others = _find_skill_in_other_profiles(name)
    if len(others) == 1:
        other_profile, other_path = others[0]
        base += (
            f" A skill by that name exists in profile "
            f"'{other_profile}' ({other_path}). To edit it, switch "
            f"profiles (`hermes -p {other_profile}`) or edit the file "
            f"directly (file tools / terminal)."
        )
    elif others:
        names = ", ".join(f"'{p}'" for p, _ in others)
        base += (
            f" Skills by that name exist in other profiles: {names}. "
            f"Switch profiles (`hermes -p <name>`) to edit there, or "
            f"edit the files directly (file tools / terminal)."
        )
    else:
        base += " Use skills_list() to see available skills."
    return base + suffix


def _validate_file_path(file_path: str) -> Optional[str]:
    """Validate a write_file/remove_file path: under an allowed subdir, no escape."""
    from tools.path_security import has_traversal_component

    if not file_path:
        return "file_path is required."
    normalized = Path(file_path)
    # Traversal is checked before any allow-listing so the SKILL.md exception
    # can never be reached by a traversal-laden path.
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."
    # SKILL.md lives at the skill root; accept 'SKILL.md' and '<skill>/SKILL.md'.
    if normalized.name == "SKILL.md" and len(normalized.parts) in (1, 2):
        return None
    if not normalized.parts or normalized.parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(normalized.parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{normalized.parts[0]}/myfile.md'"
    return None


def _resolve_skill_target(skill_dir: Path, file_path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a supporting-file path and ensure it stays within the skill directory."""
    from tools.path_security import validate_within_dir

    target = skill_dir / file_path
    error = validate_within_dir(target, skill_dir)
    return (None, error) if error else (target, None)


def _resolve_supporting_file(skill_dir: Path, file_path: str):
    """``_validate_file_path`` + ``_resolve_skill_target``. Returns (target, None) or (None, error_dict)."""
    err = _validate_file_path(file_path)
    if err:
        return None, _err(err)
    target, err = _resolve_skill_target(skill_dir, file_path)
    if err:
        return None, _err(err)
    return target, None


def _locate_for_write(name: str, action: str, not_found_suffix: str = ""):
    """Find the skill and run the org-mirror + background-review write guards.

    Returns ``(skill_dir, None)`` or ``(None, error_dict)``.
    """
    existing = _find_skill(name)
    if not existing:
        return None, _err(_skill_not_found_error(name, not_found_suffix))
    skill_dir = existing["path"]
    guard = (
        _org_mirror_write_guard(name, skill_dir, action)
        or _background_review_write_guard(name, skill_dir, action)
    )
    if guard:
        return None, guard
    return skill_dir, None


def _write_scanned(target: Path, content: str, skill_dir: Path,
                   original: Optional[str]) -> Optional[str]:
    """Atomically write ``content``, then security-scan the skill; on block,
    restore ``original`` (or unlink a newly created file) and return the error."""
    atomic_write_text(target, content, preserve_mode=True, create_mode=0o644)
    scan_error = _security_scan_skill(skill_dir)
    if scan_error:
        if original is not None:
            atomic_write_text(target, original, preserve_mode=True)
        else:
            target.unlink(missing_ok=True)
    return scan_error


def _attach_org_note(result: Dict[str, Any], name: str, skill_dir: Path) -> None:
    org_note = _maybe_auto_propose_org_edit(name, skill_dir)
    if org_note:
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"


# =============================================================================
# Core actions
# =============================================================================


def _add_description_prompt_preview(result: Dict[str, Any], content: str) -> None:
    """Append a system_prompt_preview field when the description will be truncated."""
    fm, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(fm):
        result["system_prompt_preview"] = (
            f"System prompt will show: \"{extract_skill_description(fm)}\" — "
            f"keep the trigger self-contained in the first "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars."
        )


def _attach_lint_findings(result: Dict[str, Any], skill_md: Path) -> None:
    """Attach ADVISORY skill-authoring-convention findings (never a hard block;
    hard rejects already ran in _validate_frontmatter)."""
    try:
        from tools.skill_linter import lint_skill  # local import: optional path

        findings = lint_skill(skill_md)
    except Exception:
        return
    if not findings:
        return
    result["lint_warnings"] = [
        {"severity": f.severity, "rule": f.rule, "message": f.message}
        for f in findings
    ]
    result["lint_hint"] = (
        "The skill was created. These are advisory authoring-convention "
        "findings (not blockers) — fix them with skill_manage(action='patch') "
        "to match Hermes skill standards."
    )


def _create_skill(name: str, content: str, category: str = None) -> Dict[str, Any]:
    """Create a new user skill with SKILL.md content."""
    err = (
        _validate_name(name)
        or _validate_category(category)
        or _validate_frontmatter(content, new_skill=True)
        or _validate_content_size(content)
    )
    if err:
        return _err(err)
    existing = _find_skill(name)
    if existing:
        return _err(f"A skill named '{name}' already exists at {existing['path']}.")

    skill_dir = _resolve_skill_dir(name, category)
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    atomic_write_text(skill_md, content, preserve_mode=True, create_mode=0o644)
    scan_error = _security_scan_skill(skill_dir)
    if scan_error:
        shutil.rmtree(skill_dir, ignore_errors=True)
        return _err(scan_error)

    try:
        _display_path = str(skill_dir.relative_to(_skills_dir()))
    except ValueError:
        _display_path = str(skill_dir)  # created under skills.create_dir
    result = {
        "success": True,
        "message": f"Skill '{name}' created.",
        "path": _display_path,
        "skill_md": str(skill_md),
        "_change": {"description": _description_preview(content)},
    }
    if category:
        result["category"] = category
    result["hint"] = (
        "To add reference files, templates, or scripts, use "
        "skill_manage(action='write_file', name='{}', file_path='references/example.md', file_content='...')".format(name)
    )
    _add_description_prompt_preview(result, content)
    _attach_lint_findings(result, skill_md)
    return result


def _edit_skill(name: str, content: str) -> Dict[str, Any]:
    """Replace the SKILL.md of any existing skill (full rewrite)."""
    err = _validate_frontmatter(content) or _validate_content_size(content)
    if err:
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "edit")
    if guard:
        return guard
    skill_md = skill_dir / "SKILL.md"
    read_guard = _background_review_read_before_write_guard(name, skill_md, "edit", "SKILL.md")
    if read_guard:
        return read_guard

    # SKILL.md always exists here (_find_skill requires it), so a blocked scan restores it.
    original_content = skill_md.read_text(encoding="utf-8") if skill_md.exists() else None
    scan_error = _write_scanned(skill_md, content, skill_dir, original_content)
    if scan_error:
        return _err(scan_error)

    result = {
        "success": True,
        "message": f"Skill '{name}' updated (full rewrite).",
        "path": str(skill_dir),
        "_change": {"description": _description_preview(content)},
    }
    _attach_org_note(result, name, skill_dir)
    _add_description_prompt_preview(result, content)
    return result


def _patch_skill(
    name: str,
    old_string: str,
    new_string: str,
    file_path: str = None,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """Targeted find-and-replace within SKILL.md (default) or a supporting file.
    Requires a unique match unless replace_all is True."""
    if not old_string:
        # A bare "required" error is a dead end: the model retries blindly and
        # often escapes to action='write_file', clobbering the whole file.
        return _err(
            "old_string is required for 'patch' and must be the EXACT text currently in the "
            "file. Read the target file first (read_file on the skill's SKILL.md, or the file "
            "named by file_path) and copy the snippet verbatim, then retry 'patch'. "
            "Do NOT fall back to action='write_file' — that rewrites the entire file and "
            "destroys unrelated content."
        )
    if new_string is None:
        return _err("new_string is required for 'patch'. Use an empty string to delete matched text.")
    # No old_string == new_string guard here: fuzzy_find_and_replace rejects
    # that with a richer error (file_preview) this layer cannot produce.

    skill_dir, guard = _locate_for_write(name, "patch")
    if guard:
        return guard

    if file_path:
        target, err = _resolve_supporting_file(skill_dir, file_path)
        if err:
            return err
    else:
        target = skill_dir / "SKILL.md"
    if not target.exists():
        return _err(f"File not found: {target.relative_to(skill_dir)}")

    target_label = file_path or "SKILL.md"
    read_guard = _background_review_read_before_write_guard(name, target, "patch", target_label)
    if read_guard:
        return read_guard

    content = target.read_text(encoding="utf-8")
    # Same fuzzy engine as the file patch tool (whitespace/indent/escape
    # normalization, block anchors) so minor formatting mismatches don't fail.
    from tools.fuzzy_match import fuzzy_find_and_replace

    new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all
    )
    if match_error:
        try:
            from tools.fuzzy_match import format_no_match_hint
            match_error += format_no_match_hint(match_error, match_count, old_string, content)
        except Exception:
            pass
        return _err(match_error) | {"file_preview": content[:500] + ("..." if len(content) > 500 else "")}

    err = _validate_content_size(new_content, label=target_label)
    if err:
        return _err(err)
    if not file_path:
        err = _validate_frontmatter(new_content)
        if err:
            return _err(f"Patch would break SKILL.md structure: {err}")

    scan_error = _write_scanned(target, new_content, skill_dir, content)
    if scan_error:
        return _err(scan_error)

    result = {
        "success": True,
        "message": f"Patched {target_label} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
        "_change": {
            "old": old_string[:200] + ("…" if len(old_string) > 200 else ""),
            "new": new_string[:200] + ("…" if len(new_string) > 200 else ""),
        },
    }
    _attach_org_note(result, name, skill_dir)
    return result


def _delete_skill(name: str, absorbed_into: Optional[str] = None) -> Dict[str, Any]:
    """Delete a skill.

    ``absorbed_into``: ``None`` = undeclared (legacy path, accepted);
    ``""`` = explicit prune with no forwarding target; ``"<skill>"`` = content
    absorbed into that umbrella, which must exist on disk (validated here so
    the model can't claim a nonexistent umbrella).
    """
    skill_dir, guard = _locate_for_write(name, "delete")
    if guard:
        return guard
    fail_closed = _curator_consolidation_delete_guard(name, absorbed_into)
    if fail_closed:
        return fail_closed
    pinned_err = _pinned_guard(name)
    if pinned_err:
        return _err(pinned_err)

    absorbed_target = absorbed_into.strip() if isinstance(absorbed_into, str) else ""
    is_consolidation = bool(absorbed_target)
    if is_consolidation:
        if absorbed_target == name:
            return _err(f"absorbed_into='{absorbed_target}' cannot equal the skill being deleted.")
        if not _find_skill(absorbed_target):
            return _err(
                f"absorbed_into='{absorbed_target}' does not exist. "
                f"Create or patch the umbrella skill first, then retry the delete."
            )

    skills_root = _containing_skills_root(skill_dir)
    unsafe = _validate_delete_target(skill_dir)  # defense-in-depth before rmtree
    if unsafe:
        return _err(unsafe)

    # During the curator consolidation pass a verified consolidation must be
    # RECOVERABLE (`hermes curator restore`), so route through the archive
    # primitive instead of rmtree. Foreground deletes keep hard-delete semantics.
    absorbed_note = f" Content absorbed into '{absorbed_target}'." if is_consolidation else ""
    if _is_background_review():
        try:
            from tools.skill_usage import archive_skill
            ok, archive_msg = archive_skill(name)
        except Exception as e:
            return _err(f"failed to archive '{name}': {e}")
        if not ok:
            return _err(archive_msg)
        return {
            "success": True,
            "message": f"Skill '{name}' archived ({archive_msg}).{absorbed_note}",
            "_archived": True,
        }

    shutil.rmtree(skill_dir)
    _rmdir_if_empty(skill_dir.parent, skills_root)  # empty category dir, never the root
    return {"success": True, "message": f"Skill '{name}' deleted.{absorbed_note}"}


def _rmdir_if_empty(parent: Path, stop: Path) -> None:
    if parent != stop and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _write_file(name: str, file_path: str, file_content: str) -> Dict[str, Any]:
    """Add or overwrite a supporting file within any skill directory."""
    err = _validate_file_path(file_path)
    if err:
        return _err(err)
    if not file_content and file_content != "":
        return _err("file_content is required.")
    content_bytes = len(file_content.encode("utf-8"))
    if content_bytes > MAX_SKILL_FILE_BYTES:
        return _err(
            f"File content is {content_bytes:,} bytes "
            f"(limit: {MAX_SKILL_FILE_BYTES:,} bytes / 1 MiB). "
            f"Consider splitting into smaller files."
        )
    err = _validate_content_size(file_content, label=file_path)
    if err:
        return _err(err)

    skill_dir, guard = _locate_for_write(name, "write_file", " Create it first with action='create'.")
    if guard:
        return guard
    target, err = _resolve_supporting_file(skill_dir, file_path)
    if err:
        return err
    if target.exists():
        read_guard = _background_review_read_before_write_guard(name, target, "write_file", file_path)
        if read_guard:
            return read_guard
    target.parent.mkdir(parents=True, exist_ok=True)
    original_content = target.read_text(encoding="utf-8") if target.exists() else None
    scan_error = _write_scanned(target, file_content, skill_dir, original_content)
    if scan_error:
        return _err(scan_error)

    result = {
        "success": True,
        "message": f"File '{file_path}' written to skill '{name}'.",
        "path": str(target),
    }
    _attach_org_note(result, name, skill_dir)
    return result


def _remove_file(name: str, file_path: str) -> Dict[str, Any]:
    """Remove a supporting file from any skill directory."""
    err = _validate_file_path(file_path)
    if err:
        return _err(err)
    existing = _find_skill(name)
    if not existing:
        return _err(_skill_not_found_error(name))
    skill_dir = existing["path"]
    guard = _background_review_write_guard(name, skill_dir, "remove_file")
    if guard:
        return guard

    target, err = _resolve_supporting_file(skill_dir, file_path)
    if err:
        return err
    if not target.exists():
        available = [
            str(f.relative_to(skill_dir))
            for subdir in ALLOWED_SUBDIRS
            if (skill_dir / subdir).exists()
            for f in (skill_dir / subdir).rglob("*")
            if f.is_file()
        ]
        return {
            "success": False,
            "error": f"File '{file_path}' not found in skill '{name}'.",
            "available_files": available if available else None,
        }
    read_guard = _background_review_read_before_write_guard(name, target, "remove_file", file_path)
    if read_guard:
        return read_guard

    target.unlink()
    _rmdir_if_empty(target.parent, skill_dir)
    return {"success": True, "message": f"File '{file_path}' removed from skill '{name}'."}


# =============================================================================
# Main entry point
# =============================================================================

# Set while replaying an already-approved staged skill write so skill_manage()
# does not re-gate (and re-stage) it.
_skill_gate_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False
)

_GATED_ACTIONS = {"create", "edit", "patch", "delete", "write_file", "remove_file"}


def _run_write_gate(build_staging):
    """Shared write-gate evaluation. Returns None to proceed, or a JSON tool
    result when blocked/staged. ``build_staging(wa)`` -> (payload, gist) is
    only called when the decision is to stage. Fails open if write_approval
    cannot be imported."""
    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open
    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)
    payload, gist = build_staging(wa)
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "gist": gist, "message": decision.message},
        ensure_ascii=False,
    )


def _apply_skill_write_gate(action, name, **payload_kwargs):
    """Flat-shape write gate: stage the full skill_manage kwargs so approval can
    replay them. Bypassed during approved-pending replay."""
    if action not in _GATED_ACTIONS or _skill_gate_bypass.get():
        return None

    def _staging(wa):
        payload = {"action": action, "name": name}
        payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
        gist = wa.skill_gist(
            action, name,
            content=payload_kwargs.get("content") or "",
            file_path=payload_kwargs.get("file_path") or "",
            old_string=payload_kwargs.get("old_string") or "",
            new_string=payload_kwargs.get("new_string") or "",
        )
        return payload, gist

    return _run_write_gate(_staging)


_FLAT_OP_KEYS = ("content", "category", "file_path", "file_content",
                 "old_string", "new_string")


def _skill_manage_from(payload: Dict[str, Any], **extra) -> str:
    """Call ``skill_manage`` with the flat-shape fields taken from ``payload``."""
    return skill_manage(
        action=payload.get("action", ""),
        name=payload.get("name", ""),
        replace_all=payload.get("replace_all", False),
        **{k: payload.get(k) for k in _FLAT_OP_KEYS},
        **extra,
    )


def apply_skill_pending(payload: Dict[str, Any]) -> str:
    """Replay a staged skill write, bypassing the gate. Returns the tool result
    JSON string. Called by the /skills approve handler."""
    token = _skill_gate_bypass.set(True)
    try:
        return _skill_manage_from(
            payload,
            absorbed_into=payload.get("absorbed_into"),
            operations=payload.get("operations"),
        )
    finally:
        _skill_gate_bypass.reset(token)


# Debounce state for the sync push hook: a burst of skill_manage writes
# collapses into one push after a quiet window, on a daemon timer.
_sync_push_timer = None
_sync_push_lock = None
_SYNC_PUSH_DEBOUNCE_S = 5.0


def _maybe_debounced_sync_push(skill_name: str) -> None:
    """Schedule a debounced best-effort sync push after a skill write.

    Fast-path: skills not opted into sync do nothing (no auth, no network).
    The push runs via ``skills_sync_client.maybe_push_skills`` which enforces
    the access gate and swallows errors. Never blocks the caller (M1-C).
    """
    global _sync_push_timer, _sync_push_lock
    try:
        from tools.skill_usage import is_sync_enabled

        if not is_sync_enabled(skill_name):
            return
    except Exception:
        return

    if _sync_push_lock is None:
        _sync_push_lock = threading.Lock()

    def _fire():
        try:
            from tools.skills_sync_client import maybe_push_skills

            maybe_push_skills(message=f"sync: {skill_name}")
        except Exception:
            pass

    with _sync_push_lock:
        if _sync_push_timer is not None:
            _sync_push_timer.cancel()  # only sets an Event; never raises
        _sync_push_timer = threading.Timer(_SYNC_PUSH_DEBOUNCE_S, _fire)
        _sync_push_timer.daemon = True
        _sync_push_timer.start()


def _act_create(a):
    if not a["content"]:
        return tool_error("content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).", success=False)
    return _create_skill(a["name"], a["content"], a["category"])


def _act_edit(a):
    # Legacy alias for a full rewrite (old transcripts/callers; not in the schema).
    if not a["content"]:
        return tool_error("content is required for a full rewrite. Provide the full updated SKILL.md text.", success=False)
    return _edit_skill(a["name"], a["content"])


def _act_patch(a):
    # Two shapes: old_string/new_string = targeted replacement;
    # content (alone) = full SKILL.md rewrite (absorbs the old 'edit').
    if a["content"] and (a["old_string"] or a["new_string"] is not None):
        return tool_error(
            "Pass EITHER content (full SKILL.md rewrite) OR "
            "old_string/new_string (targeted replacement), not both.",
            success=False,
        )
    if a["content"]:
        return _edit_skill(a["name"], a["content"])
    # Targeted-replacement validation lives in _patch_skill so the public
    # tool and the helper return the same actionable guidance.
    return _patch_skill(a["name"], a["old_string"], a["new_string"], a["file_path"], a["replace_all"])


def _act_write_file(a):
    if not a["file_path"]:
        return tool_error("file_path is required for 'write_file'. Example: 'references/api-guide.md'", success=False)
    if a["file_content"] is None:
        return tool_error("file_content is required for 'write_file'.", success=False)
    return _write_file(a["name"], a["file_path"], a["file_content"])


def _act_remove_file(a):
    if not a["file_path"]:
        return tool_error("file_path is required for 'remove_file'.", success=False)
    return _remove_file(a["name"], a["file_path"])


# action -> handler(args dict). Handlers return a result dict, or a JSON string
# (tool_error) for argument-shape errors.
_ACTION_HANDLERS = {
    "create": _act_create,
    "edit": _act_edit,
    "patch": _act_patch,
    "delete": lambda a: _delete_skill(a["name"], absorbed_into=a["absorbed_into"]),
    "write_file": _act_write_file,
    "remove_file": _act_remove_file,
}


def _record_success(action, name, result, *, file_path, absorbed_into, task_id,
                    session_id, ledger_before) -> None:
    """Post-mutation side effects, all best-effort (never break the tool):
    audit ledger, prompt-cache clear, curator telemetry, debounced sync push."""
    try:
        from tools import skill_ledger as _ledger
        _post = _find_skill(name)
        _evidence = {}
        if action == "delete":
            # consolidation vs prune, and whether the recoverable archive handled it
            _evidence["absorbed_into"] = absorbed_into
            _evidence["archived"] = bool(result.get("_archived"))
        if session_id:
            _evidence["session_id"] = session_id
        if file_path:
            _evidence["file_path"] = file_path
        _ledger.record_mutation(
            action,
            name,
            before=ledger_before if ledger_before is not None else [],
            after_root=_post["path"] if _post else None,
            evidence=_evidence,
        )
    except Exception:
        pass
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass
    # Curator telemetry: only the background review fork marks a skill as
    # agent-created — foreground creates are user-directed and belong to the
    # user. A recoverable curator archive keeps its record as STATE_ARCHIVED
    # (so `hermes curator status`/`restore` see it); only a hard delete forgets.
    try:
        from tools.skill_usage import bump_patch, forget, record_created
        from tools.skill_provenance import is_background_review
        if action == "create":
            record_created(name, agent_created=is_background_review(),
                           task_id=task_id, session_id=session_id)
        elif action in {"patch", "edit", "write_file", "remove_file"}:
            bump_patch(name, action=action, task_id=task_id, session_id=session_id)
        elif action == "delete" and not result.get("_archived"):
            forget(name)
    except Exception:
        pass
    # Sync push only runs AFTER the write gate passed (staged writes returned
    # early), so un-reviewed content is never pushed. Inert unless the access
    # gate is open, a sync URL is configured, and the skill is opted in.
    try:
        _maybe_debounced_sync_push(name)
    except Exception:
        pass


def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
    task_id: str = None,
    session_id: str = None,
    operations=None,
) -> str:
    """Manage user-created skills; dispatches to the action handler.

    ``operations``: batch shape — a list of {action, ...} dicts applied
    atomically (see _skill_manage_batch). When set, the flat single-op fields
    are ignored and ``action`` may be omitted/'batch'. Returns a JSON string.
    """
    if operations is not None:
        return _skill_manage_batch(
            operations, default_name=name or None,
            task_id=task_id, session_id=session_id,
        )
    preflight = _background_review_preflight(action, name)
    if preflight is not None:
        return json.dumps(preflight, ensure_ascii=False)

    # Approval gate: when on, stages the write for review (skills are too large
    # to review inline, so they always stage regardless of origin); bypassed
    # when replaying an already-approved staged write.
    args = dict(
        content=content, category=category, file_path=file_path,
        file_content=file_content, old_string=old_string, new_string=new_string,
        replace_all=replace_all, absorbed_into=absorbed_into,
    )
    gate_result = _apply_skill_write_gate(action, name, **args)
    if gate_result is not None:
        return gate_result

    # Audit ledger: capture pre-mutation state. Telemetry, not a gate — failures
    # must NEVER block the mutation. delete destroys the whole package and
    # consolidation may have re-homed support files first, so complete the
    # capture from the newest curator backup or a restore is hollow.
    _ledger_before = None
    try:
        from tools import skill_ledger as _ledger
        _pre = _find_skill(name)
        _ledger_before = _ledger.capture_before(
            _pre["path"] if _pre else None,
            complete_package=(action == "delete"),
            skill=name,
        )
    except Exception:
        pass

    handler = _ACTION_HANDLERS.get(action)
    if handler is None:
        result = _err(f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file")
    else:
        result = handler({"name": name, **args})
    if isinstance(result, str):
        return result  # tool_error JSON for argument-shape problems

    if result.get("success"):
        _record_success(
            action, name, result, file_path=file_path, absorbed_into=absorbed_into,
            task_id=task_id, session_id=session_id, ledger_before=_ledger_before,
        )
    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    # ONE call shape (memory-tool pattern, maintainer-directed): the call
    # IS an operations array — each op names its skill and action; a
    # single edit is a list of one. The legacy flat shape (top-level
    # action/name/content/...) is still ACCEPTED by the handler for old
    # transcripts and staged-write replay, but no longer advertised.
    "description": (
        "Create, update, or delete skills — your procedural memory for "
        "recurring task types. The call is an operations array (a single "
        "edit is a list of one); it applies atomically — any failure rolls "
        "every touched skill back. Ops: create (full SKILL.md; lands in "
        f"{_display_create_dir()}; must precede that skill's other "
        "ops), patch (targeted old_string/new_string fix — preferred; "
        "content alone REPLACES the whole file, read it via skill_view() "
        "first), write_file/remove_file (supporting files), delete (sole "
        "op only). Existing skills are modified wherever they live. Keep "
        "the description's first 57 chars a self-contained trigger: 'Use "
        "when <trigger>. <one-line behavior>.' — skill_view() shows "
        "format conventions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "Ordered ops; each names its target skill.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Skill name (lowercase, hyphens/underscores, "
                                "max 64 chars); an existing skill's name "
                                "unless creating."
                            )
                        },
                        "action": {
                            "type": "string",
                            "enum": ["create", "patch", "delete", "write_file", "remove_file"]
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Full SKILL.md text (YAML frontmatter + "
                                "markdown body) for create, or a full "
                                "rewrite on patch."
                            )
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category subdir for create (e.g. 'devops')."
                        },
                        # patch args: same fuzzy-matching semantics as the
                        # `patch` tool — teach only skill-specific facts here.
                        "old_string": {
                            "type": "string",
                            "description": "Text to find (patch; same matching semantics as the patch tool)."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement (patch); empty string deletes the match."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "patch: replace all occurrences (default false)."
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Path RELATIVE to the skill's own directory, "
                                "e.g. 'references/api.md' — no leading slash, "
                                "never absolute. write_file/remove_file: "
                                "required; first segment references/, "
                                "templates/, scripts/, or assets/. patch: "
                                "optional (default SKILL.md)."
                            )
                        },
                        "file_content": {
                            "type": "string",
                            "description": "Content for write_file."
                        }
                    },
                    "required": ["name", "action"]
                }
            },
            # NOTE: the handler also accepts the legacy flat single-op shape
            # (top-level action/name/content/old_string/new_string/
            # replace_all/category/file_path/file_content) — old transcripts
            # and staged-write replay depend on it — plus `absorbed_into` on
            # delete ops (curator-only vocabulary; the curator's prompt
            # documents it and the delete guard's error re-teaches it).
            # None are advertised.
        },
        "required": ["operations"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="skill_manage",
    toolset="skills",
    schema=SKILL_MANAGE_SCHEMA,
    handler=lambda args, **kw: _skill_manage_from(
        args,
        absorbed_into=args.get("absorbed_into"),
        operations=args.get("operations"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id")),
    emoji="📝",
)
