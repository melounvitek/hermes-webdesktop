"""Shared slash command helpers for skills (CLI and gateway both invoke /skill-name)."""

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import display_hermes_home
from agent.prompt_cache_boundary import register_stable_prefix
from agent.skill_preprocessing import (
    load_skills_config as _load_skills_config,
    preprocess_skill_content,
)

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}
_skill_commands_platform: Optional[str] = None
_skill_commands_home: Optional[str] = None
# Guards the (map, platform-tag, home-tag) triple so publication and the
# freshness lookup always see a consistent snapshot. Scanning stays outside.
_publish_lock = threading.Lock()
# Patterns for sanitizing skill names into clean hyphen-separated slugs.
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")

# ---------------------------------------------------------------------------
# Skill-scaffolding markers and the canonical extractor.
#
# A /skill (or /bundle) turn is expanded into a model-facing message embedding
# the full skill body. Memory providers that store the raw user turn would
# capture the body instead of what the user asked, so
# ``extract_user_instruction_from_skill_message`` recovers just the instruction.
#
# These markers MUST stay byte-identical to the builders (``_build_skill_message``
# here, ``build_bundle_invocation_message`` in agent/skill_bundles.py); the
# bundle markers are asserted in tests/openviking_plugin/test_openviking.py.
# ---------------------------------------------------------------------------
_SKILL_INVOCATION_PREFIX = "[IMPORTANT: The user has invoked the "
_SINGLE_SKILL_MARKER = "The full skill content is loaded below.]"
_SINGLE_SKILL_INSTRUCTION = (
    "The user has provided the following instruction alongside the skill invocation: "
)
_RUNTIME_NOTE = "\n\n[Runtime note:"
_BUNDLE_MARKER = " skill bundle,"
_BUNDLE_USER_INSTRUCTION = "\nUser instruction: "
_BUNDLE_FIRST_SKILL_BLOCK = "\n\n[Loaded as part of the "

# The skill name sits in the first quoted span of the activation note, for both
# the single-skill and the bundle header ("work" / "/clean /work").
_SKILL_NAME_RE = re.compile(re.escape(_SKILL_INVOCATION_PREFIX) + r'"([^"]*)"')

# SQL LIKE pattern matching a skill-expanded turn, for listing queries that
# recognize scaffolding before the row reaches Python. The prefix contains no
# LIKE wildcards, so it needs no ESCAPE clause.
SKILL_SCAFFOLD_SQL_LIKE = _SKILL_INVOCATION_PREFIX + "%"

# Marks where a preview query joined the head and tail of a long scaffolded
# message; ``describe_skill_invocation`` cuts a description there rather than
# show the skill body on the far side.
SKILL_EXCERPT_JOINT = "\x1e"


def slugify_skill_name(name: str) -> str:
    """Normalize a skill/bundle name to a ``/command`` slug (``Foo Bar`` -> ``foo-bar``).

    Strips non-alnum chars (``+``, ``/``) that would make invalid Telegram
    command names downstream.
    """
    cmd = name.lower().replace(" ", "-").replace("_", "-")
    cmd = _SKILL_INVALID_CHARS.sub("", cmd)
    return _SKILL_MULTI_HYPHEN.sub("-", cmd).strip("-")


def append_user_instruction(parts: list, instruction: str) -> str:
    """Append the instruction line to ``parts``; return the stable prefix.

    The prefix ends exactly at the instruction marker so, registered with
    ``agent.prompt_cache_boundary``, the Anthropic cache planner can break on
    the scaffold instead of caching the whole message as one atomic block.
    Single construction site guarantees the prefix is a byte-prefix of the message.
    """
    stable_prefix = "\n".join(parts) + "\n" + _SINGLE_SKILL_INSTRUCTION
    parts.append(f"{_SINGLE_SKILL_INSTRUCTION}{instruction}")
    return stable_prefix


def extract_user_instruction_from_skill_message(content: Any) -> Optional[str]:
    """Recover the user's instruction from a slash-skill-expanded turn.

    Returns the string unchanged when it is NOT scaffolding, the extracted
    instruction when the scaffolding carried one, or ``None`` for a bare
    ``/skill`` invocation (nothing worth storing in memory).
    """
    if not isinstance(content, str):
        return None

    if not content.startswith(_SKILL_INVOCATION_PREFIX):
        return content

    if _BUNDLE_MARKER in content:
        return _extract_bundle_user_instruction(content)

    if _SINGLE_SKILL_MARKER in content:
        return _extract_single_skill_user_instruction(content)

    return None


def describe_skill_invocation(content: Any, separator: str = " — ") -> Optional[str]:
    """Render a slash-skill-expanded turn the way the user typed it.

    Returns ``"/work — fix the title leak"``, ``"/work"`` for a bare invocation,
    or ``None`` when *content* is not skill scaffolding. Surfaces that summarize
    a user turn (session titles, previews, ``/rewind``) use this so the skill's
    own prose never masquerades as the user's. Pass ``separator=" "`` for the
    literal invocation as typed (chat transcripts).
    """
    if not isinstance(content, str) or not content.startswith(_SKILL_INVOCATION_PREFIX):
        return None

    match = _SKILL_NAME_RE.match(content)
    name = (match.group(1) if match else "").strip()
    # Bundle headers already carry their typed "/a /b" keys; a single skill is a bare name.
    label = name if name.startswith("/") else f"/{name}"

    instruction = extract_user_instruction_from_skill_message(content)
    if instruction and instruction is not content:
        # An excerpt (head + tail joined by SKILL_EXCERPT_JOINT) can put the
        # joint inside the span — keep only the side the marker was found on.
        instruction = " ".join(instruction.split(SKILL_EXCERPT_JOINT)[0].split())
        if instruction:
            return f"{label}{separator}{instruction}" if name else instruction

    return label if name else None


def _extract_single_skill_user_instruction(message: str) -> Optional[str]:
    # The instruction follows the skill body, so the LAST marker is the user's
    # (the body may quote the marker text).
    marker_idx = message.rfind(_SINGLE_SKILL_INSTRUCTION)
    if marker_idx < 0:
        return None
    instruction = message[marker_idx + len(_SINGLE_SKILL_INSTRUCTION):]
    return _cut_at(instruction, _RUNTIME_NOTE)


def _extract_bundle_user_instruction(message: str) -> Optional[str]:
    # Bundles put the instruction before the loaded skills, so the FIRST marker is the user's.
    marker_idx = message.find(_BUNDLE_USER_INSTRUCTION)
    if marker_idx < 0:
        return None
    instruction = message[marker_idx + len(_BUNDLE_USER_INSTRUCTION):]
    return _cut_at(instruction, _BUNDLE_FIRST_SKILL_BLOCK)


def _cut_at(text: str, stop_marker: str) -> Optional[str]:
    idx = text.find(stop_marker)
    if idx >= 0:
        text = text[:idx]
    return text.strip() or None


def _resolve_skill_commands_platform() -> Optional[str]:
    """Current platform scope for disabled-skill filtering, or None (CLI, RL, scripts).

    A change here invalidates the scan cache so each platform sees its own
    ``skills.platform_disabled`` view.
    """
    try:
        from gateway.session_context import get_session_env

        resolved_platform = (
            os.getenv("HERMES_PLATFORM")
            or get_session_env("HERMES_SESSION_PLATFORM")
        )
    except Exception:
        resolved_platform = os.getenv("HERMES_PLATFORM")
    return resolved_platform or None


def _resolve_skill_commands_home() -> str:
    """Effective Hermes home the scan is scoped to.

    Profiles carry their own ``skills.external_dirs``; a profile switch without
    a platform change must still invalidate the scan cache.
    """
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _load_skill_payload(skill_identifier: str, task_id: str | None = None) -> tuple[dict[str, Any], Path | None, str] | None:
    """Load a skill by name/path and return (loaded_payload, skill_dir, display_name)."""
    raw_identifier = (skill_identifier or "").strip()
    if not raw_identifier:
        return None

    try:
        from tools.skills_tool import _skills_dir, skill_view
        from agent.skill_utils import normalize_skill_lookup_name

        normalized = normalize_skill_lookup_name(raw_identifier)

        loaded_skill = json.loads(
            skill_view(normalized, task_id=task_id, preprocess=False)
        )
    except Exception:
        return None

    if not loaded_skill.get("success"):
        return None

    skill_name = str(loaded_skill.get("name") or normalized)
    skill_path = str(loaded_skill.get("path") or "")
    skill_dir = None
    # Prefer the absolute skill_dir from skill_view() (correct for external
    # skills too); fall back to SKILLS_DIR-relative reconstruction for legacy responses.
    abs_skill_dir = loaded_skill.get("skill_dir")
    if abs_skill_dir:
        skill_dir = Path(abs_skill_dir)
    elif skill_path:
        try:
            skill_dir = _skills_dir() / Path(skill_path).parent
        except Exception:
            skill_dir = None

    return loaded_skill, skill_dir, skill_name


def _bump_use(skill_name: str, task_id: str | None) -> None:
    """Track active usage for Curator lifecycle management; never fatal."""
    try:
        from tools.skill_usage import bump_use
        bump_use(skill_name, task_id=task_id)
    except Exception:
        pass


def _inject_skill_config(loaded_skill: dict[str, Any], parts: list[str]) -> None:
    """Append a ``[Skill config: ...]`` block with resolved ``metadata.hermes.config`` values.

    Lets the agent see configured values without reading config.yaml itself.
    Non-critical: any failure leaves the message without the block.
    """
    try:
        from agent.skill_utils import (
            extract_skill_config_vars,
            parse_frontmatter,
            resolve_skill_config_values,
        )

        raw_content = str(loaded_skill.get("raw_content") or loaded_skill.get("content") or "")
        if not raw_content:
            return

        frontmatter, _ = parse_frontmatter(raw_content)
        config_vars = extract_skill_config_vars(frontmatter)
        if not config_vars:
            return

        resolved = resolve_skill_config_values(config_vars)
        if not resolved:
            return

        lines = ["", f"[Skill config (from {display_hermes_home()}/config.yaml):"]
        for key, value in resolved.items():
            display_val = str(value) if value else "(not set)"
            lines.append(f"  {key} = {display_val}")
        lines.append("]")
        parts.extend(lines)
    except Exception:
        pass


def _setup_note(loaded_skill: dict[str, Any]) -> Optional[str]:
    if loaded_skill.get("setup_skipped"):
        return (
            "Required environment setup was skipped. Continue loading the skill "
            "and explain any reduced functionality if it matters."
        )
    if loaded_skill.get("gateway_setup_hint"):
        return loaded_skill["gateway_setup_hint"]
    if loaded_skill.get("setup_needed") and loaded_skill.get("setup_note"):
        return loaded_skill["setup_note"]
    return None


def _supporting_files(loaded_skill: dict[str, Any], skill_dir: Path | None) -> list[str]:
    """Skill-relative support file paths: from ``linked_files`` or a disk walk."""
    supporting = []
    for entries in (loaded_skill.get("linked_files") or {}).values():
        if isinstance(entries, list):
            supporting.extend(entries)

    if not supporting and skill_dir:
        for subdir in ("references", "templates", "scripts", "assets"):
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                for f in sorted(subdir_path.rglob("*")):
                    if f.is_file() and not f.is_symlink():
                        supporting.append(str(f.relative_to(skill_dir)))
    return supporting


def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path | None,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
    session_id: str | None = None,
) -> str:
    """Format a loaded skill into a user/system message payload."""
    from tools.skills_tool import _skills_dir

    # Preprocess first so downstream blocks see the expanded content.
    content = preprocess_skill_content(
        str(loaded_skill.get("content") or ""),
        skill_dir,
        session_id,
        skills_cfg=_load_skills_config(),
    )

    parts = [activation_note, "", content.strip()]

    # Absolute skill dir lets the agent run bundled scripts without a skill_view() round-trip.
    if skill_dir:
        parts.append("")
        parts.append(f"[Skill directory: {skill_dir}]")
        parts.append(
            "Resolve any relative paths in this skill (e.g. `scripts/foo.js`, "
            "`templates/config.yaml`) against that directory, then run them "
            "with the terminal tool using the absolute path."
        )

    _inject_skill_config(loaded_skill, parts)

    setup_note = _setup_note(loaded_skill)
    if setup_note:
        parts.extend(["", f"[Skill setup note: {setup_note}]"])

    supporting = _supporting_files(loaded_skill, skill_dir)
    if supporting and skill_dir:
        try:
            skill_view_target = str(skill_dir.relative_to(_skills_dir()))
        except ValueError:
            skill_view_target = skill_dir.name  # external dir — use the skill name
        parts.append("")
        parts.append(
            "[This skill has supporting files (paths relative to the skill "
            "directory above):]"
        )
        for sf in supporting:
            parts.append(f"- {sf}")
        parts.append(
            f'\nLoad any of these with skill_view(name="{skill_view_target}", '
            f'file_path="<path>"), or run scripts directly by absolute path '
            f"(e.g. `node {skill_dir}/scripts/foo.js`)."
        )

    stable_prefix = None
    if user_instruction:
        parts.append("")
        # Everything before the volatile instruction is a stable scaffold; the
        # registered boundary lets the cache planner break there (see append_user_instruction).
        stable_prefix = append_user_instruction(parts, user_instruction)

    if runtime_note:
        parts.append("")
        parts.append(f"[Runtime note: {runtime_note}]")

    message = "\n".join(parts)
    if stable_prefix is not None and message.startswith(stable_prefix) and len(message) > len(stable_prefix):
        register_stable_prefix(stable_prefix)
    return message


def _render_skill_block(
    loaded: tuple[dict[str, Any], Path | None, str],
    activation_note: str,
    task_id: str | None,
    **message_kwargs: str,
) -> str:
    """Bump usage and build the message block for one loaded skill."""
    loaded_skill, skill_dir, skill_name = loaded
    _bump_use(skill_name, task_id)
    return _build_skill_message(loaded_skill, skill_dir, activation_note, session_id=task_id, **message_kwargs)


def _scaffold_header(
    subject: str,
    loaded_names: list[str],
    *,
    lead_lines: list[str] | None = None,
    missing: list[str] | None = None,
    disabled: list[str] | None = None,
    extra_instruction: str = "",
    user_instruction: str = "",
) -> str:
    """Header for multi-skill messages (bundles and stacked invocations).

    ``subject`` (e.g. ``'"name" skill bundle'``) must end in " skill bundle" so
    the bundle-format extractor in extract_user_instruction_from_skill_message()
    applies unchanged.
    """
    lines = [
        f"[IMPORTANT: The user has invoked the {subject}, "
        f"loading {len(loaded_names)} skills together. Treat every skill below "
        "as active guidance for this turn.]",
        "",
        *(lead_lines or []),
        f"Skills loaded: {', '.join(loaded_names)}",
    ]
    if missing:
        lines.append(f"Skills missing (skipped): {', '.join(missing)}")
    if disabled:
        lines.append(
            f"Skills disabled for this platform (skipped): {', '.join(disabled)}"
        )
    if extra_instruction:
        lines.extend(["", f"Bundle instruction: {extra_instruction}"])
    if user_instruction:
        lines.extend(["", f"User instruction: {user_instruction}"])
    return "\n".join(lines)


_SCAN_SKIP_PARTS = {'.git', '.github', '.hub', '.archive'}


def _scan_skill_md(skill_md: Path, disabled: set, seen_names: set, commands: Dict[str, Dict[str, Any]], resolve_command) -> None:
    """Register one SKILL.md in *commands* (no-op when filtered or colliding)."""
    from tools.skills_tool import _parse_frontmatter, skill_matches_platform, skill_matches_environment

    if any(part in _SCAN_SKIP_PARTS for part in skill_md.parts):
        return
    frontmatter, body = _parse_frontmatter(skill_md.read_text(encoding='utf-8'))
    # OS gate is hard; environment gate (kanban/docker/s6) is offer-time only.
    if not skill_matches_platform(frontmatter) or not skill_matches_environment(frontmatter):
        return
    name = frontmatter.get('name', skill_md.parent.name)
    if name in seen_names or name in disabled:
        return
    description = frontmatter.get('description', '')
    if not description:
        for line in body.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                description = line[:80]
                break
    seen_names.add(name)
    cmd_name = slugify_skill_name(name)
    if not cmd_name:
        return
    # A collision with a core command (name or alias, via resolve_command) skips
    # auto-registration; the skill stays loadable via /skill <name>.
    if resolve_command(cmd_name) is not None:
        logger.warning(
            "Skill %r generates slash command '/%s' which "
            "collides with a core Hermes command; skipping "
            "auto-registration. Use '/skill %s' instead.",
            name, cmd_name, name,
        )
        return
    # Dedup on the slug too: "git_helper" and "git-helper" normalize the same.
    # First-wins preserves project > local > external precedence.
    cmd_key = f"/{cmd_name}"
    if cmd_key in commands:
        logger.warning(
            "Skill %r maps to slash command %s already claimed "
            "by %r; keeping the first and skipping this one.",
            name, cmd_key, commands[cmd_key]["name"],
        )
        return
    commands[cmd_key] = {
        "name": name,
        "description": description or f"Invoke the {name} skill",
        "skill_md_path": str(skill_md),
        "skill_dir": str(skill_md.parent),
    }


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan skill dirs and return {"/skill-name": {name, description, skill_md_path, skill_dir}}.

    Builds into a local map and publishes once at the end: writing straight into
    the global exposed partial results to overlapping scans, which then logged
    bogus "already claimed" collisions against their own incumbents.
    """
    global _skill_commands, _skill_commands_platform, _skill_commands_home
    platform = _resolve_skill_commands_platform()
    home = _resolve_skill_commands_home()
    commands: Dict[str, Dict[str, Any]] = {}
    try:
        from tools.skills_tool import _skills_dir, _get_disabled_skill_names
        from agent.skill_utils import (
            get_external_skills_dirs,
            get_project_skills_dirs,
            iter_project_skill_files,
            iter_skill_index_files,
        )
        from hermes_cli.commands import resolve_command
        disabled = _get_disabled_skill_names()
        seen_names: set = set()

        # Precedence: project (through the quarantine chokepoint) > local > external.
        # Resolve the local dir at call time: import-time SKILLS_DIR is frozen to
        # the launch home, but a multiplexed profile scope may have changed it.
        project_dirs = list(get_project_skills_dirs())
        dirs_to_scan = list(project_dirs)
        skills_dir = _skills_dir()
        if skills_dir.exists():
            dirs_to_scan.append(skills_dir)
        dirs_to_scan.extend(get_external_skills_dirs())

        for scan_dir in dirs_to_scan:
            _iter = (
                iter_project_skill_files(scan_dir)
                if scan_dir in project_dirs
                else iter_skill_index_files(scan_dir, "SKILL.md")
            )
            for skill_md in _iter:
                try:
                    _scan_skill_md(skill_md, disabled, seen_names, commands, resolve_command)
                except Exception:
                    continue
    except Exception:
        pass
    # Publish map + tags as ONE step: a reader landing between bare assignments
    # could accept the new map under a stale platform tag and serve another
    # platform's disabled-skill view.
    with _publish_lock:
        _skill_commands = commands
        _skill_commands_platform = platform
        _skill_commands_home = home
    return commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty).

    Rescans when the platform scope changes (one gateway serving Telegram and
    Discord) or the active profile's home changes (Desktop profile switch), so
    each sees its own ``platform_disabled`` / ``external_dirs`` view.
    """
    current_platform = _resolve_skill_commands_platform()
    current_home = _resolve_skill_commands_home()
    with _publish_lock:
        commands = _skill_commands
        is_fresh = (
            bool(commands)
            and _skill_commands_platform == current_platform
            and _skill_commands_home == current_home
        )
    if is_fresh:
        return commands
    # Scan outside the lock — file I/O and deferred imports; concurrent scans
    # are safe since each builds its own map.
    return scan_skill_commands()


def diff_command_snapshots(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, Any]:
    """Diff two {name: description} snapshots into added/removed/unchanged/total.

    Removed entries carry the pre-rescan description (the file may be gone).
    """
    added_names = sorted(set(after) - set(before))
    removed_names = sorted(set(before) - set(after))
    return {
        "added": [{"name": n, "description": after[n]} for n in added_names],
        "removed": [{"name": n, "description": before[n]} for n in removed_names],
        "unchanged": sorted(set(after) & set(before)),
        "total": len(after),
    }


def reload_skills() -> Dict[str, Any]:
    """Re-scan skill dirs and return a diff of the slash-command map.

    Does NOT invalidate the skills system-prompt cache: skills are called by
    name, so keeping the prompt cache intact means ``/reload-skills`` costs no
    cache reset.

    Returns ``{"added": [{name, description}], "removed": [...], "unchanged":
    [names], "total": int, "commands": int}``; ``description`` is the full
    frontmatter field (the system prompt index truncates it).
    """
    def _snapshot(cmds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        return {
            slash_key.lstrip("/"): (info or {}).get("description") or ""
            for slash_key, info in cmds.items()
        }

    before = _snapshot(_skill_commands)
    new_commands = scan_skill_commands()
    result = diff_command_snapshots(before, _snapshot(new_commands))
    result["commands"] = len(new_commands)
    return result


def resolve_skill_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed /command to its canonical ``/slug`` key, or None.

    Underscores map to hyphens: Telegram disallows hyphens in bot commands, so
    ``/claude-code`` comes back as ``/claude_code``.
    """
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in get_skill_commands() else None


def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    runtime_note: str = "",
) -> Optional[str]:
    """Build the user message for a skill slash command, or None if not found."""
    skill_info = get_skill_commands().get(cmd_key)
    if not skill_info:
        return None

    loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
    if not loaded:
        return None

    return _render_skill_block(
        loaded,
        f'[IMPORTANT: The user has invoked the "{loaded[2]}" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]",
        task_id,
        user_instruction=user_instruction,
        runtime_note=runtime_note,
    )


# ---------------------------------------------------------------------------
# Stacked slash-skill invocations — `/skill-a /skill-b do XYZ` loads every
# leading skill (up to _MAX_STACKED_SKILLS). The message reuses the BUNDLE
# scaffolding markers so the memory extractor needs no new plumbing.
# ---------------------------------------------------------------------------
_MAX_STACKED_SKILLS = 5


def split_stacked_skill_commands(rest: str) -> tuple[list[str], str]:
    """Consume further leading ``/skill`` tokens from *rest* (text after the first matched command).

    Stops at the first token that is not a resolvable skill command (or a
    repeat); that token onward is the user instruction. Returns
    ``(extra_cmd_keys, remaining_instruction)``.
    """
    keys: list[str] = []
    remaining = rest or ""
    while len(keys) < _MAX_STACKED_SKILLS - 1:
        stripped = remaining.lstrip()
        if not stripped.startswith("/"):
            break
        parts = stripped.split(None, 1)
        token = parts[0]
        tail = parts[1] if len(parts) > 1 else ""
        cmd_key = resolve_skill_command_key(token.lstrip("/"))
        if cmd_key is None or cmd_key in keys:
            break
        keys.append(cmd_key)
        remaining = tail
    return keys, remaining.strip()


def build_stacked_skill_invocation_message(
    cmd_keys: list[str],
    user_instruction: str = "",
    task_id: str | None = None,
) -> Optional[tuple[str, list[str], list[str]]]:
    """Build the user message for a stacked multi-skill slash invocation.

    Returns ``(message, loaded_skill_names, missing_skill_names)`` or ``None``
    when no skill could be loaded at all.
    """
    commands = get_skill_commands()

    loaded_names: list[str] = []
    missing: list[str] = []
    skill_blocks: list[str] = []
    seen: set[str] = set()

    for cmd_key in cmd_keys:
        if not cmd_key or cmd_key in seen:
            continue
        seen.add(cmd_key)

        skill_info = commands.get(cmd_key)
        loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id) if skill_info else None
        if not loaded:
            missing.append(cmd_key.lstrip("/"))
            continue
        skill_name = loaded[2]
        # Must start with "[Loaded as part of the " — the bundle block marker.
        skill_blocks.append(_render_skill_block(
            loaded,
            f'[Loaded as part of the stacked skill invocation "{skill_name}".]',
            task_id,
        ))
        loaded_names.append(skill_name)

    if not skill_blocks:
        return None

    typed = " ".join(k for k in cmd_keys if k)
    header = _scaffold_header(
        f'"{typed}" stacked skill bundle',
        loaded_names,
        missing=missing,
        user_instruction=user_instruction,
    )
    return ("\n\n".join([header, *skill_blocks]), loaded_names, missing)


def build_preloaded_skills_prompt(
    skill_identifiers: list[str],
    task_id: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Load skills for session-wide CLI/TUI preloading.

    Returns (prompt_text, loaded_skill_names, missing_identifiers). Disabled
    skills count as missing: this path bypasses the scan-time disabled filter,
    so ``hermes -s <skill>`` must not force-load an operator-disabled skill.
    """
    prompt_parts: list[str] = []
    loaded_names: list[str] = []
    missing: list[str] = []

    try:
        from agent.skill_utils import get_disabled_skill_names
        disabled_names = get_disabled_skill_names()
    except Exception:
        disabled_names = set()

    seen: set[str] = set()
    for raw_identifier in skill_identifiers:
        identifier = (raw_identifier or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = _load_skill_payload(identifier, task_id=task_id)
        if not loaded or loaded[2] in disabled_names or identifier in disabled_names:
            missing.append(identifier)
            continue
        skill_name = loaded[2]
        prompt_parts.append(_render_skill_block(
            loaded,
            f'[IMPORTANT: The user launched this CLI session with the "{skill_name}" skill '
            "preloaded. Treat its instructions as active guidance for the duration of this "
            "session unless the user overrides them.]",
            task_id,
        ))
        loaded_names.append(skill_name)

    return "\n\n".join(prompt_parts), loaded_names, missing
