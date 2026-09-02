"""System-prompt assembly for :class:`AIAgent`.

Built once per session and reused across turns (only context compression
triggers a rebuild) so the upstream prefix cache stays warm.  Three tiers
are joined with ``\\n\\n``: ``stable`` (identity, guidance, env hints, coding
brief, platform hints), ``context`` (workspace snapshot, caller
``system_message``, context files) and ``volatile`` (skills index, memory,
USER.md, external memory provider, timestamp line).  See the
``hermes-agent-dev`` skill's ``references/system-prompt-invariant.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    EXECUTION_GUIDANCE_MODELS,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    USER_PROFILE_GUIDANCE,
    PARALLEL_TOOL_CALL_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    STEER_CHANNEL_NOTE,
    TASK_COMPLETION_GUIDANCE,
    TELEGRAM_RICH_MESSAGES_HINT,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
    drain_truncation_warnings,
)
from agent.runtime_cwd import resolve_context_cwd
from hermes_constants import get_default_hermes_root, get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)
_PLUGIN_SECTION_FRAME_RE = re.compile(
    r"^## Plugin Context: (?P<id>[a-z0-9][a-z0-9._-]{0,127})\n"
    r"<!-- hermes-plugin-section-chars:(?P<chars>[0-9]{1,4}) -->\n\n",
    re.MULTILINE,
)
_TRUTHY_GATES = {"true", "always", "yes", "on"}
_FALSY_GATES = {"false", "never", "no", "off"}


def _ra():
    """Lazy ``run_agent`` handle: tests ``patch("run_agent.load_soul_md")`` etc.,
    so the helpers must be resolved through that namespace on every call."""
    import run_agent
    return run_agent


def _model_gate(setting: Any, model: Optional[str], default_models) -> bool:
    """Resolve a config gate: True/"true"-ish -> on, False/"false"-ish -> off,
    list -> case-insensitive model-substring match, anything else ("auto") ->
    match against *default_models*."""
    model_lower = (model or "").lower()
    if setting is True or (isinstance(setting, str) and setting.lower() in _TRUTHY_GATES):
        return True
    if setting is False or (isinstance(setting, str) and setting.lower() in _FALSY_GATES):
        return False
    if isinstance(setting, list):
        return any(p.lower() in model_lower for p in setting if isinstance(p, str))
    return any(p in model_lower for p in default_models)


def _resolve_platform_hint(agent: Any, platform_key: str, default_hint: str) -> str:
    """Apply the ``platform_hints.<platform>`` config override to *default_hint*.

    ``replace`` substitutes the default, ``append`` adds text (a bare string is
    shorthand for append); ``replace`` wins when both are present.  Malformed
    entries fall back to the unmodified default so bad config can never break
    prompt assembly or leak across platforms.
    """
    if not platform_key:
        return default_hint
    overrides = getattr(agent, "_platform_hint_overrides", None)
    if not isinstance(overrides, dict) or not overrides:
        return default_hint
    spec = overrides.get(platform_key)
    if spec is None:
        return default_hint
    if isinstance(spec, str):
        extra = spec.strip()
        return f"{default_hint}\n\n{extra}".strip() if extra else default_hint
    if not isinstance(spec, dict):
        return default_hint
    replace_text = spec.get("replace")
    base = replace_text.strip() if isinstance(replace_text, str) and replace_text.strip() else default_hint
    append_text = spec.get("append")
    if isinstance(append_text, str) and append_text.strip():
        return f"{base}\n\n{append_text.strip()}".strip()
    return base


_TUI_EMBEDDED_PANE_CLARIFIER = (
    " You're in its embedded terminal pane, beside the GUI chat — the user can "
    "select your output (Option-drag on macOS, Shift-drag elsewhere) and press "
    "Cmd/Ctrl+L to send it to the chat composer."
)


def _tui_embedded_pane_clarifier(hint: str) -> str:
    """Append the desktop embedded-terminal clarifier to a tui hint when
    ``HERMES_DESKTOP_TERMINAL`` is set (only on the desktop's TUI PTY, never the
    chat backend).  Idempotent; empty input stays empty."""
    if not hint or _TUI_EMBEDDED_PANE_CLARIFIER in hint:
        return hint
    if not is_truthy_value(os.getenv("HERMES_DESKTOP_TERMINAL")):
        return hint
    return hint + _TUI_EMBEDDED_PANE_CLARIFIER


def _plugin_session_info(agent: Any) -> Dict[str, str]:
    """Return immutable-at-render-time metadata exposed to prompt sections."""
    try:
        cwd = str(resolve_context_cwd() or "")
    except Exception:
        cwd = ""
    try:
        # Prefer the agent's own home: ambient get_active_profile_name()
        # misreports on threads that lost the HERMES_HOME ContextVar.
        _home = _agent_home(agent)
        if _home is not None:
            profile_name = _profile_name_for_home(_home)
        else:
            from hermes_cli.profiles import get_active_profile_name

            profile_name = str(get_active_profile_name() or "default")
    except Exception:
        profile_name = "default"
    return {
        "session_id": str(getattr(agent, "session_id", None) or ""),
        "model": str(getattr(agent, "model", None) or ""),
        "provider": str(getattr(agent, "provider", None) or ""),
        "platform": str(getattr(agent, "platform", None) or ""),
        "profile_name": profile_name,
        "cwd": cwd,
    }


def _frozen_plugin_prompt_sections(agent: Any) -> tuple:
    """Render plugin sections once per session and freeze them on the agent.

    A restored ``_cached_system_prompt`` is parsed instead of re-running plugin
    code; a render that raises at a rebuild boundary keeps the previous bytes
    (stashed by ``invalidate_system_prompt``) instead of silently vanishing.
    """
    attr = "_plugin_system_prompt_sections_snapshot"
    if hasattr(agent, attr):
        return getattr(agent, attr)
    stored_prompt = getattr(agent, "_cached_system_prompt", None)
    if isinstance(stored_prompt, str) and stored_prompt:
        rendered = _restore_plugin_prompt_sections(stored_prompt)
        setattr(agent, attr, rendered)
        return rendered
    try:
        from hermes_cli.plugins import render_system_prompt_sections

        rendered = tuple(render_system_prompt_sections(_plugin_session_info(agent)))
    except Exception as exc:
        previous = getattr(agent, "_plugin_system_prompt_sections_previous", None)
        if previous:
            logger.warning(
                "Plugin system prompt sections failed to re-render (%s); "
                "keeping the previous frozen sections", exc,
            )
            rendered = previous
        else:
            logger.warning("Plugin system prompt sections could not be rendered: %s", exc)
            rendered = ()
    setattr(agent, attr, rendered)
    return rendered


def _restore_plugin_prompt_sections(prompt: str) -> tuple:
    """Recover frozen section bytes from the persisted full prompt.  Only the
    exact canonical container emitted by core is accepted — user/project text
    may resemble a frame."""
    from hermes_cli.plugins import (
        MAX_SYSTEM_PROMPT_SECTION_CHARS,
        PLUGIN_SECTIONS_END,
        PLUGIN_SECTIONS_START,
        RenderedPluginSystemPromptSection,
        format_system_prompt_sections,
    )

    start = prompt.rfind(PLUGIN_SECTIONS_START)
    if start < 0:
        return ()
    end = prompt.find(PLUGIN_SECTIONS_END, start + len(PLUGIN_SECTIONS_START))
    if end < 0:
        return ()
    after_end = end + len(PLUGIN_SECTIONS_END)
    if not prompt[after_end:].startswith("\n\nConversation started:"):
        return ()
    framed = prompt[start:after_end]

    restored = []
    for match in _PLUGIN_SECTION_FRAME_RE.finditer(framed):
        content_len = int(match.group("chars"))
        if content_len > MAX_SYSTEM_PROMPT_SECTION_CHARS:
            continue
        content = framed[match.end() : match.end() + content_len]
        if len(content) != content_len:
            continue
        restored.append(
            RenderedPluginSystemPromptSection(
                id=match.group("id"),
                content=content,
                position="after_memory",
                plugin="persisted-prompt",
            )
        )
    if format_system_prompt_sections(restored) != framed:
        return ()
    return tuple(restored)


def restore_plugin_prompt_sections(agent: Any, prompt: str) -> None:
    """Seed a resumed agent's frozen snapshot from persisted prompt bytes."""
    agent._plugin_system_prompt_sections_snapshot = _restore_plugin_prompt_sections(prompt)


def _plugin_section_blocks(sections: tuple, position: str) -> List[str]:
    from hermes_cli.plugins import format_system_prompt_sections

    block = format_system_prompt_sections([s for s in sections if s.position == position])
    return [block] if block else []


def _session_start_like(agent: Any, now: Any) -> Any:
    """Best-known conversation start time, or ``now`` as a fallback.

    ``Conversation started:`` must be byte-stable across rebuilds (compression,
    resume, fresh gateway turns), so prefer immutable sources in order: the
    lineage-root session id's embedded stamp (compaction rotates ids, each with
    its own mint time), the current session id's stamp, ``agent.session_start``,
    then ``now``.  Stamps are box-local wall-clock: attach that zone first, then
    convert to ``now``'s zone so the date matches the per-turn clock.
    """
    from datetime import datetime

    try:
        machine_local_tz = datetime.now().astimezone().tzinfo
    except (ValueError, OSError):
        machine_local_tz = None

    def _to_display_tz(dt: Any) -> Any:
        if machine_local_tz is not None and dt.tzinfo is None:
            try:
                dt = dt.replace(tzinfo=machine_local_tz)
            except ValueError:
                pass
        if getattr(now, "tzinfo", None) is not None and dt.tzinfo is not None:
            try:
                dt = dt.astimezone(now.tzinfo)
            except (ValueError, OSError):
                pass
        return dt

    session_id = getattr(agent, "session_id", None)
    root_id = None
    try:
        db = getattr(agent, "_session_db", None)
        if db is not None and isinstance(session_id, str) and session_id:
            root_id = db.get_conversation_root(session_id)
    except Exception:
        root_id = None
    for candidate in (root_id, session_id):
        if isinstance(candidate, str) and candidate:
            m = re.match(r"^(\d{8})_(\d{6})", candidate)
            if m:
                try:
                    embedded = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
                    return _to_display_tz(embedded)
                except ValueError:
                    pass

    session_start = getattr(agent, "session_start", None)
    if hasattr(session_start, "astimezone"):
        return _to_display_tz(session_start)
    return now


def _agent_home(agent: Any) -> Optional[Path]:
    """The agent's OWN profile home, or None to use ambient resolution.

    A bound HERMES_HOME ContextVar override wins (the gateway multiplexes
    profiles over one shared session DB and binds the home per turn); else the
    parent of ``_session_db.db_path`` — ground truth on threads that lost the
    ContextVar, where ambient resolution would leak the launch profile.
    """
    try:
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            return Path(override)
    except Exception:
        pass
    try:
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        if db_path:
            return Path(db_path).parent
    except Exception:
        pass
    return None


def _agent_skills_dir(agent: Any) -> Optional[Path]:
    """The agent's own ``<home>/skills`` dir, or None to use ambient home."""
    home = _agent_home(agent)
    return (home / "skills") if home is not None else None


def _profile_name_for_home(home: Path) -> str:
    """``<root>/profiles/X`` -> ``"X"``; anything else -> ``"default"``.

    Uses ``get_default_hermes_root()`` (NOT ``get_hermes_home()``): on a bound
    profile session the ambient home IS the profile dir, so every profile would
    misreport as "default".
    """
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()
        rel = home.resolve().relative_to((root / "profiles").resolve())
        return rel.parts[0] if rel.parts else "default"
    except (ValueError, OSError):
        return "default"


def _tool_guidance_block(agent: Any) -> Optional[str]:
    """Tool-aware behavioral guidance, injected only when the tools are loaded."""
    tool_guidance = []
    # With both memory stores disabled no store is built, so the full guidance
    # would steer the model at a tool that always answers "Memory is not
    # available"; with only USER.md enabled the narrower block is used.
    if "memory" in agent.valid_tool_names:
        if getattr(agent, "_memory_enabled", True):
            tool_guidance.append(MEMORY_GUIDANCE)
        elif getattr(agent, "_user_profile_enabled", True):
            tool_guidance.append(USER_PROFILE_GUIDANCE)
    if "session_search" in agent.valid_tool_names:
        tool_guidance.append(SESSION_SEARCH_GUIDANCE)
    if "skill_manage" in agent.valid_tool_names:
        tool_guidance.append(SKILLS_GUIDANCE)
    # Kanban lifecycle: resolved once at __init__ (_kanban_worker_guidance);
    # the kanban_show fallback covers code paths that bypass agent_init.
    _kanban_guidance = getattr(agent, "_kanban_worker_guidance", None)
    if _kanban_guidance:
        tool_guidance.append(_kanban_guidance)
    elif _kanban_guidance is None and "kanban_show" in agent.valid_tool_names:
        tool_guidance.append(KANBAN_GUIDANCE)
    return " ".join(tool_guidance) if tool_guidance else None


def _skills_prompt(agent: Any, _r: Any) -> str:
    """Skills index (empty without skills tools).  Focus mode demotes non-coding
    categories to names-only — never hidden, every name stays visible."""
    if not any(name in agent.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage']):
        return ""
    avail_toolsets = {
        toolset
        for toolset in (_r.get_toolset_for_tool(tool_name) for tool_name in agent.valid_tool_names)
        if toolset
    }
    _compact_cats = frozenset()
    try:
        from agent.coding_context import coding_compact_skill_categories

        _compact_cats = coding_compact_skill_categories(platform=agent.platform, cwd=resolve_context_cwd())
    except Exception:
        _compact_cats = frozenset()
    return _r.build_skills_system_prompt(
        available_tools=agent.valid_tool_names,
        available_toolsets=avail_toolsets,
        compact_categories=_compact_cats or None,
        skills_dir_override=_agent_skills_dir(agent),
    )


def _bot_mode_parts(agent: Any) -> List[str]:
    """Bot Mode teammate protocol — only in a bot's canonical "Bot Chat" session.
    Marks the prompt timeless (the volatile date line is dropped) since a birth
    date pinned in a months-long session is misinformation."""
    parts: List[str] = []
    try:
        from tools.bot_mode_probe import (
            BOT_CHAT_TITLE,
            epoch_line,
            get_bot_mode_protocol_section,
        )
        _title = str(getattr(agent, "_session_title_hint", "") or "").strip()
        if not _title:
            _sdb = getattr(agent, "_session_db", None)
            _sid = getattr(agent, "session_id", None)
            _title = str((_sdb.get_session_title(_sid) if (_sdb and _sid) else None) or "").strip()
        if _title == BOT_CHAT_TITLE:
            _bot_section = get_bot_mode_protocol_section(_agent_home(agent))
            if _bot_section:
                parts.append(_bot_section)
                # Capability epoch lets the restore path rebuild ONCE per
                # user-initiated capability change in an eternal session.
                parts.append(epoch_line(_agent_home(agent)))
                agent._bot_chat_timeless_prompt = True
    except Exception:
        pass
    return parts


def _active_profile_line(agent: Any) -> str:
    """Name the running profile so the agent doesn't conflate ``~/.hermes/skills``
    (default) with ``~/.hermes/profiles/<active>/skills``.  Resolved from the
    agent's OWN home first (a build thread that lost the ContextVar would
    otherwise print "default" for a bot profile)."""
    _agent_home_path = _agent_home(agent)
    active_profile = "default"
    try:
        if _agent_home_path is not None:
            active_profile = _profile_name_for_home(_agent_home_path)
        else:
            from agent.file_safety import _resolve_active_profile_name
            active_profile = _resolve_active_profile_name()
    except Exception:
        active_profile = "default"
    # With an explicit agent home, the default profile's data lives at the
    # ROOT (get_hermes_home() on a bound profile session is the PROFILE dir).
    # Without one, keep the ambient (patchable) resolution byte-identical.
    if _agent_home_path is not None:
        _home_str = str(_agent_home_path)
        _root_str = str(get_default_hermes_root())
    else:
        _home_str = _root_str = str(get_hermes_home())
    if active_profile == "default":
        return (
            "Active Hermes profile: default. Other profiles (if any) live "
            "under " + _root_str + "/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    # A non-default name is only returned when the resolved home is ALREADY
    # <root>/profiles/<name>, so the profile home is the session home itself.
    profile_home = _home_str
    default_root = get_default_hermes_root()
    return (
        f"Active Hermes profile: {active_profile}. This session reads "
        f"and writes {profile_home}/. The default "
        f"profile's data lives at {default_root}/skills/, {default_root}/plugins/, "
        f"{default_root}/cron/, {default_root}/memories/ — those belong to a "
        f"different session run from a different shell. Do NOT modify "
        f"another profile's skills/plugins/cron/memories unless the user "
        f"explicitly directs you to."
    )


def _platform_hint(agent: Any) -> str:
    """Built-in/plugin platform hint + Telegram rich-messages opt-in + config
    override + desktop TUI clarifier."""
    platform_key = (agent.platform or "").lower().strip()
    _default_hint = ""
    if platform_key in PLATFORM_HINTS:
        _default_hint = PLATFORM_HINTS[platform_key]
    elif platform_key:
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            if _entry and _entry.platform_hint:
                _default_hint = _entry.platform_hint
        except Exception:
            pass

    # Same precedence the adapter uses: top-level platforms.telegram.extra
    # overrides gateway.platforms.telegram.extra at the leaf.
    if platform_key == "telegram" and _default_hint:
        try:
            from hermes_cli.config import load_config_readonly
            _cfg = load_config_readonly()
            _gw_tg_extra = (((_cfg.get("gateway") or {}).get("platforms") or {}).get("telegram") or {}).get("extra")
            _top_tg_extra = ((_cfg.get("platforms") or {}).get("telegram") or {}).get("extra")
            if not isinstance(_gw_tg_extra, dict):
                _gw_tg_extra = {}
            if not isinstance(_top_tg_extra, dict):
                _top_tg_extra = {}
            if {**_gw_tg_extra, **_top_tg_extra}.get("rich_messages"):
                _default_hint = _default_hint.rstrip() + " " + TELEGRAM_RICH_MESSAGES_HINT
        except Exception:
            pass  # Config read failure — fall back to base hint only

    _effective_hint = _resolve_platform_hint(agent, platform_key, _default_hint)
    if platform_key == "tui" and _effective_hint:
        _effective_hint = _tui_embedded_pane_clarifier(_effective_hint)
    return _effective_hint


def _zone_bits(now: Any, tz: Any) -> List[str]:
    """IANA key, abbreviation (if different) and UTC offset — all constant for
    the day, so the byte-stable date line stays cacheable."""
    bits = []
    _iana = getattr(tz, "key", None)
    if _iana:
        bits.append(_iana)
    _abbrev = now.strftime("%Z")
    if _abbrev and _abbrev != _iana:
        bits.append(_abbrev)
    _offset = now.strftime("%z")
    if _offset:  # '-0400' -> 'UTC-04:00'
        bits.append(f"UTC{_offset[:3]}:{_offset[3:]}")
    return bits


def _timestamp_line(agent: Any) -> str:
    """Date-only (not minute-precision) so the prompt is byte-stable for the
    day; zone + offset included so tools that need an explicit offset don't
    have to guess EST vs EDT.  Long-lived sessions get an "as of" line on
    rebuild days (the cache prefix is already invalidated at that boundary)."""
    from hermes_time import get_timezone as _hermes_tz, now as _hermes_now
    now = _hermes_now()
    _bits = _zone_bits(now, _hermes_tz())
    _zone_suffix = f" ({', '.join(_bits)})" if _bits else ""
    _start = _session_start_like(agent, now)
    timestamp_line = f"Conversation started: {_start.strftime('%A, %B %d, %Y')}{_zone_suffix}"
    if now.strftime("%Y%m%d") != _start.strftime("%Y%m%d"):
        timestamp_line += (
            f"\nToday's date (as of the last context rebuild): "
            f"{now.strftime('%A, %B %d, %Y')} — trust this over the start "
            f"date for what day it is now; query tools for exact time."
        )
    if getattr(agent, "_bot_chat_timeless_prompt", False):
        timestamp_line = f"Timezone: {', '.join(_bits)}" if _bits else ""
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    if agent.platform:
        timestamp_line += f"\nPlatform: {agent.platform}"
    return timestamp_line


def _memory_parts(agent: Any) -> List[str]:
    """Built-in memory/USER.md blocks plus the external provider block (gated on
    the same check ``inject_memory_provider_tools`` uses, so we never advertise
    tools the toolset config gated off)."""
    parts: List[str] = []
    if agent._memory_store:
        if agent._memory_enabled:
            mem_block = agent._memory_store.format_for_system_prompt("memory")
            if mem_block:
                parts.append(mem_block)
        if agent._user_profile_enabled:
            user_block = agent._memory_store.format_for_system_prompt("user")
            if user_block:
                parts.append(user_block)
    if agent._memory_manager:
        try:
            from agent.memory_manager import memory_provider_tools_exposed as _mem_exposed
        except Exception:
            _mem_exposed = None
        if _mem_exposed is None or _mem_exposed(agent):
            try:
                _ext_mem_block = agent._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    parts.append(_ext_mem_block)
            except Exception:
                pass
    return parts


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, str]:
    """Assemble the system prompt as three ordered cache tiers.

    ``stable`` runs through the coding operating brief when a workspace
    snapshot follows; ``context`` holds the snapshot, the remaining
    session-stable guidance, context files and the caller ``system_message``;
    ``volatile`` holds skills index, memory, user profile, external memory
    block and the timestamp line.  Never re-rendered mid-session.
    """
    _r = _ra()

    # Model context window scales the context-file caps; stable per conversation.
    _ctx_len: Optional[int] = None
    _cc = getattr(agent, "context_compressor", None)
    if _cc is not None:
        _cc_len = getattr(_cc, "context_length", None)
        if isinstance(_cc_len, int) and _cc_len > 0:
            _ctx_len = _cc_len

    # ── Stable tier ────────────────────────────────────────────────
    stable_parts: List[str] = []
    # SOUL.md is primary identity (cron keeps the persona while skipping cwd
    # instructions).  Scoped to the agent's OWN home — see _agent_home.
    _soul_loaded = False
    if agent.load_soul_identity or not agent.skip_context_files:
        _soul_content = _r.load_soul_md(_ctx_len, home_override=_agent_home(agent))
        if _soul_content:
            stable_parts.append(_soul_content)
            _soul_loaded = True
    if not _soul_loaded:
        stable_parts.append(DEFAULT_AGENT_IDENTITY)

    # The skill_view() pointer dangles without skill tools OR without the
    # hermes-agent skill installed, so the variant is chosen after the skills
    # index is built; this slot holds its position.
    _has_skill_view = "skill_view" in (agent.valid_tool_names or set())
    _help_guidance_slot = len(stable_parts)
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS)

    # Universal (model-agnostic) guidance, each gated by its config.yaml key.
    if getattr(agent, "_task_completion_guidance", True) and agent.valid_tool_names:
        stable_parts.append(TASK_COMPLETION_GUIDANCE)
    if getattr(agent, "_parallel_tool_call_guidance", True) and agent.valid_tool_names:
        stable_parts.append(PARALLEL_TOOL_CALL_GUIDANCE)
    _tool_block = _tool_guidance_block(agent)
    if _tool_block:
        stable_parts.append(_tool_block)
    # Steering only lands inside tool results, so only reachable with tools.
    if agent.valid_tool_names:
        stable_parts.append(STEER_CHANNEL_NOTE)

    # agent.tool_use_enforcement / agent.execution_guidance: "auto" (default)
    # matches the hardcoded model lists; true/false force; a list gives custom
    # model-name substrings.  Execution guidance is an independent gate so
    # DeepSeek/Kimi/Qwen-class models get it even with enforcement off.
    if agent.valid_tool_names:
        if _model_gate(agent._tool_use_enforcement, agent.model, TOOL_USE_ENFORCEMENT_MODELS):
            stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
            _model_lower = (agent.model or "").lower()
            if "gemini" in _model_lower or "gemma" in _model_lower:
                stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
        if _model_gate(getattr(agent, "_execution_guidance", "auto"), agent.model, EXECUTION_GUIDANCE_MODELS):
            from agent.prompt_builder import execution_guidance_text
            stable_parts.append(execution_guidance_text(agent.valid_tool_names))

    skills_prompt = _skills_prompt(agent, _r)
    # Skill-pointer variant requires BOTH skill_view AND the hermes-agent skill
    # in the rendered index (pure string check — inherits the index's stability).
    if _has_skill_view and "- hermes-agent:" in skills_prompt:
        stable_parts[_help_guidance_slot] = HERMES_AGENT_HELP_GUIDANCE

    # Alibaba Coding Plan always reports "glm-4.7" as the model name; inject
    # the real identity so the agent can answer correctly.
    if agent.provider == "alibaba":
        _model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model
        stable_parts.append(
            f"You are powered by the model named {_model_short}. "
            f"The exact model ID is {agent.model}. "
            f"When asked what model you are, always answer based on this information, "
            f"not on any model name returned by the API."
        )

    _env_hints = _r.build_environment_hints()
    if _env_hints:
        stable_parts.append(_env_hints)

    # Coding posture: operating brief stays in the stable prefix; the live
    # git/workspace snapshot sits behind its own cache boundary, and the blocks
    # below it must keep their historical post-snapshot position.
    coding_workspace_parts: List[str] = []
    coding_trailing_parts: List[str] = []
    if agent.valid_tool_names:
        try:
            from agent.coding_context import coding_system_prompt_parts

            coding_prefix_parts, coding_workspace_parts, coding_trailing_parts = coding_system_prompt_parts(
                platform=agent.platform,
                cwd=resolve_context_cwd(),
                model=agent.model,
                valid_tool_names=agent.valid_tool_names,
            )
            stable_parts.extend(coding_prefix_parts)
        except Exception:
            pass  # Coding-context probing must never block prompt build.
    if coding_workspace_parts:
        post_workspace_parts: List[str] = []
    else:
        stable_parts.extend(coding_trailing_parts)
        post_workspace_parts = stable_parts

    # Local Python toolchain probe: one line, nothing when the env is clean,
    # skipped for remote terminal backends.  config.yaml agent.environment_probe.
    if getattr(agent, "_environment_probe", True):
        try:
            from tools.env_probe import get_environment_probe_line
            _probe_line = get_environment_probe_line()
            if _probe_line:
                post_workspace_parts.append(_probe_line)
        except Exception:
            pass  # Probe failure must never block prompt build.
    if getattr(agent, "_bot_mode_protocol", True):
        post_workspace_parts.extend(_bot_mode_parts(agent))
    post_workspace_parts.append(_active_profile_line(agent))
    _effective_hint = _platform_hint(agent)
    if _effective_hint:
        post_workspace_parts.append(_effective_hint)

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []
    if coding_workspace_parts:
        context_parts.extend(coding_workspace_parts)
        context_parts.extend(coding_trailing_parts)
        context_parts.extend(post_workspace_parts)
    # ephemeral_system_prompt is injected at API-call time only, never cached.
    if system_message is not None:
        context_parts.append(system_message)
    if not agent.skip_context_files:
        # TERMINAL_CWD when set (gateway); None lets discovery fall back to the
        # launch dir.  The install-tree fallback is only legitimate for cli/tui
        # where the launch dir IS the user's shell cwd; desktop-pinned launch
        # dirs are treated as the fallback they really are so the guard can
        # reject Hermes's bundled contributor AGENTS.md.
        context_cwd = resolve_context_cwd()
        if getattr(agent, "_context_cwd_is_launch_artifact", False):
            context_cwd = None
        context_files_prompt = _r.build_context_files_prompt(
            cwd=context_cwd, skip_soul=_soul_loaded,
            context_length=_ctx_len,
            allow_install_tree_fallback=agent.platform in ("cli", "tui"),
            home_override=_agent_home(agent))
        if context_files_prompt:
            context_parts.append(context_files_prompt)

    # ── Volatile tier (most likely to differ on a rebuild; kept last so the stable prefix stays reusable) ──
    # Skills are runtime-mutable, so the index leads the volatile band: on a
    # longest-prefix backend an unchanged index still falls inside the reused
    # prefix and a changed one only re-prefills from here on.
    volatile_parts: List[str] = []
    if skills_prompt:
        volatile_parts.append(skills_prompt)
    volatile_parts.extend(_memory_parts(agent))
    # Plugin sections are confined to one coarse anchor in the volatile tail so
    # a resumed process can reconstruct the stable prefix without re-running plugins.
    volatile_parts.extend(_plugin_section_blocks(_frozen_plugin_prompt_sections(agent), "after_memory"))
    volatile_parts.append(_timestamp_line(agent))

    return {
        "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
        "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
        "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
    }


def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """Assemble the full prompt; cached on ``agent._cached_system_prompt`` and
    only rebuilt after compression.  Tiers are ordered stable -> context ->
    volatile so implicit longest-prefix caches keep the unchanged scaffold."""
    parts = build_system_prompt_parts(agent, system_message=system_message)
    joined = "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)
    agent._cached_system_prompt_static = parts["stable"]
    # Surface context-file truncation warnings in chat, not only in logs.
    for warning in drain_truncation_warnings():
        agent._emit_status(warning)
    return joined


def invalidate_system_prompt(agent: Any) -> None:
    """Force a rebuild on the next turn (after compression).  Reloads memory
    from disk and clears the frozen plugin snapshot so plugins re-render at the
    same boundary; the previous bytes are stashed as the fail-open fallback."""
    agent._cached_system_prompt = None
    agent._cached_system_prompt_static = None
    _snapshot_attr = "_plugin_system_prompt_sections_snapshot"
    if hasattr(agent, _snapshot_attr):
        agent._plugin_system_prompt_sections_previous = getattr(agent, _snapshot_attr)
        delattr(agent, _snapshot_attr)
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def reconstruct_static_prefix(
    agent: Any,
    system_message: Optional[str] = None,
    *,
    log_label: str = "restore",
) -> None:
    """Reconstruct ``_cached_system_prompt_static`` for a stored prompt.

    Only the full prompt is persisted, so restore / keep-prompt compression /
    mid-turn failover to a cache-on provider must rebuild the stable tier to
    regain the ``[static, volatile]`` layout.  The rebuilt tier is used ONLY
    when the stored prompt literally starts with it; otherwise static stays
    None and the stored bytes are sent untouched.  A failed rebuild is memoized
    per stored prompt so the retry-loop hot path doesn't redo the file I/O.
    """
    if not getattr(agent, "_use_prompt_caching", False):
        return
    stored = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(stored, str) or not stored:
        return
    existing = getattr(agent, "_cached_system_prompt_static", None)
    if isinstance(existing, str) and existing and stored.startswith(existing):
        return
    if getattr(agent, "_static_rebuild_failed_for", None) == stored:
        return
    try:
        static = build_system_prompt_parts(agent, system_message=system_message)["stable"]
        if static and stored.startswith(static):
            agent._cached_system_prompt_static = static
            agent._static_rebuild_failed_for = None
            return
    except Exception:
        logger.debug("static system-prefix reconstruction failed on %s", log_label, exc_info=True)
    agent._cached_system_prompt_static = None
    agent._static_rebuild_failed_for = stored


def format_tools_for_system_message(agent: Any) -> str:
    """JSON tool definitions in the trajectory format."""
    if not agent.tools:
        return "[]"
    return json.dumps(
        [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {}),
                "required": None,  # Match the format in the example
            }
            for tool in agent.tools
        ],
        ensure_ascii=False,
    )


__all__ = [
    "build_system_prompt_parts",
    "build_system_prompt",
    "invalidate_system_prompt",
    "restore_plugin_prompt_sections",
    "format_tools_for_system_message",
]
