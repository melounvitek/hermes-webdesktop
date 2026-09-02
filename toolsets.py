"""Toolset helpers: get/resolve/validate named tool groups (static TOOLSETS + registry-registered)."""

from typing import Dict, List, Any, Set, Optional, Tuple


# Shared tool list for CLI and all messaging platform toolsets.
# Edit this once to update all platforms simultaneously.
_HERMES_CORE_TOOLS = [
    # Web
    "web_search", "web_extract",
    # Terminal + process management
    "terminal", "process_manage",
    # Desktop GUI affordances (read_terminal, open_preview, project_*) are NOT
    # here: they live in `desktop_ui`/`project`, enabled only by the GUI gateway
    # per desktop-sourced session (tui_gateway/server.py::_load_enabled_toolsets).
    # File manipulation
    "read_file", "write_file", "patch", "search_files",
    # Vision + image generation
    "vision_analyze", "image_generate",
    # Skills
    "skills_list", "skill_view", "skill_manage",
    # Browser automation
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images",
    "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
    # replaces other tools when browser.backend is "browser-use"
    "browser_exec",
    # Text-to-speech
    "text_to_speech",
    # Planning & memory
    "todo_list", "memory",
    # Session history search
    "session_search",
    # Clarifying questions
    "clarify",
    # Code execution + delegation
    "execute_code", "delegate_task",
    # Cronjob management
    "cronjob_manage",
    # Home Assistant smart home control (gated on HASS_TOKEN via check_fn)
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
    # Kanban coordination — check_fn in tools/kanban_tools.py admits these only
    # for kanban workers (HERMES_KANBAN_TASK) or profiles enabling `kanban`.
    "kanban_show", "kanban_list",
    "kanban_complete", "kanban_block", "kanban_request_review",
    "kanban_request_changes",
    "kanban_heartbeat",
    "kanban_comment", "kanban_create", "kanban_link",
    "kanban_unblock",
    "kanban_attach", "kanban_attach_url", "kanban_attachments",
    # Computer use (macOS, gated on cua-driver being installed via check_fn)
    "computer_use",
]

# Webhook payloads are untrusted third-party content: no file/system execution.
_HERMES_WEBHOOK_SAFE_TOOLS = [
    "web_search",
    "web_extract",
    "vision_analyze",
    "clarify",
]


# Core toolset definitions
# These can include individual tools or reference other toolsets
TOOLSETS = {
    # Basic toolsets - individual tool categories
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []
    },

    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": []
    },

    "x_search": {
        "description": (
            "Search X (Twitter) posts and threads via xAI's built-in "
            "x_search Responses tool. Read-only public X discovery; use the "
            "xurl skill for authenticated X API reads and account actions. "
            "Available when xAI credentials are configured (SuperGrok OAuth "
            "or XAI_API_KEY). Off by default; enable in `hermes tools` → "
            "X (Twitter) Search."
        ),
        "tools": ["x_search"],
        "includes": []
    },

    "vision": {
        "description": "Image analysis and vision tools",
        "tools": ["vision_analyze"],
        "includes": []
    },

    "video": {
        "description": "Video analysis and understanding tools (opt-in, not in default toolset)",
        "tools": ["video_analyze"],
        "includes": []
    },

    "image_gen": {
        "description": "Creative generation tools (images)",
        "tools": ["image_generate"],
        "includes": []
    },

    "video_gen": {
        "description": (
            "Video generation tools. Single ``video_generate`` tool covers "
            "text-to-video (prompt only) and image-to-video (prompt + "
            "image_url), plus reference-to-video. Provider-specific edit/"
            "extend workflows may appear as separate tools. Configure via "
            "``hermes tools`` → Video Generation."
        ),
        "tools": ["video_generate", "xai_video_edit", "xai_video_extend"],
        "includes": []
    },

    "computer_use": {
        "description": (
            "Background desktop control via cua-driver (macOS/Windows/Linux) — "
            "screenshots, mouse, keyboard, scroll, drag. Does NOT steal the "
            "user's cursor or keyboard focus. Works with any tool-capable model."
        ),
        "tools": ["computer_use"],
        "includes": []
    },

    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": ["terminal", "process_manage"],
        "includes": []
    },

    "skills": {
        "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
        "tools": ["skills_list", "skill_view", "skill_manage"],
        "includes": []
    },

    "browser": {
        "description": "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
        "tools": [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "browser_cdp",
            "browser_dialog", "browser_exec", "web_search"
        ],
        "includes": []
    },

    "cronjob": {
        "description": "Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks",
        "tools": ["cronjob_manage"],
        "includes": []
    },


    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": ["read_file", "write_file", "patch", "search_files"],
        "includes": []
    },

    "tts": {
        "description": "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI",
        "tools": ["text_to_speech"],
        "includes": []
    },

    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": ["todo_list"],
        "includes": []
    },

    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": ["memory"],
        "includes": []
    },

    "context_engine": {
        "description": "Runtime tools exposed by the active context engine",
        "tools": [],
        "includes": []
    },

    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": ["session_search"],
        "includes": []
    },

    "project": {
        "description": "Desktop Projects — create/switch named workspaces (GUI sessions only)",
        "tools": ["desktop_project"],
        "includes": []
    },

    "bot_room": {
        "description": "Verified text-only Group Chat turn capabilities",
        "tools": [],
        "includes": [],
    },

    # GUI-renderer affordances, enabled per desktop-sourced SESSION by the GUI
    # gateway (tui_gateway/server.py::_load_enabled_toolsets) — never by a
    # process env var, which is blind to a desktop client on a remote backend.
    "desktop_ui": {
        "description": "Desktop GUI affordances — in-app terminal/browser panes, pane focus, reactions (GUI sessions only)",
        "tools": [
            "read_terminal", "close_terminal",
            "desktop_preview", "drive_preview", "annotate_preview",
            "read_window_below",
            "focus_pane", "react_to_message",
            "setup_mcp", "gui_tour", "show_tip",
        ],
        "includes": []
    },

    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": ["clarify"],
        "includes": []
    },

    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": ["execute_code"],
        "includes": []
    },

    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": ["delegate_task"],
        "includes": []
    },


    "homeassistant": {
        "description": "Home Assistant smart home control and monitoring",
        "tools": ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"],
        "includes": []
    },

    "kanban": {
        "description": (
            "Kanban multi-agent coordination — only active when the agent "
            "is spawned by the kanban dispatcher (HERMES_KANBAN_TASK env "
            "set). The dispatcher runs inside the gateway by default; see "
            "`kanban.dispatch_in_gateway` in config.yaml. Lets workers mark "
            "tasks done with structured handoffs, enter first-class review "
            "(request_review — not a block), return review changes, block for human input, "
            "heartbeat during long ops, comment on threads, attach files, and "
            "(for orchestrators) list, unblock, and fan out tasks."
        ),
        "tools": [
            "kanban_show", "kanban_list", "kanban_complete", "kanban_block",
            "kanban_request_review", "kanban_request_changes",
            "kanban_heartbeat", "kanban_comment",
            "kanban_create", "kanban_link",
            "kanban_unblock",
            "kanban_attach", "kanban_attach_url", "kanban_attachments",
        ],
        "includes": [],
    },

    "discord": {
        "description": "Discord read and participate tools (fetch messages, search members, create threads)",
        "tools": ["discord"],
        "includes": [],
    },

    "discord_admin": {
        "description": "Discord server management (list channels/roles, pin messages, assign roles)",
        "tools": ["discord_admin"],
        "includes": [],
    },

    "yuanbao": {
        "description": "Yuanbao platform tools - group info, member queries, DM, stickers",
        "tools": [
            "yb_query_group_info",
            "yb_query_group_members",
            "yb_send_dm",
            "yb_search_sticker",
            "yb_send_sticker",
        ],
        "includes": []
    },

    "feishu_doc": {
        "description": "Read Feishu/Lark document content",
        "tools": ["feishu_doc_read"],
        "includes": []
    },

    "feishu_drive": {
        "description": "Feishu/Lark document comment operations (list, reply, add)",
        "tools": [
            "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
            "feishu_drive_reply_comment", "feishu_drive_add_comment",
        ],
        "includes": []
    },

    "spotify": {
        "description": "Native Spotify playback, search, playlist, album, and library tools",
        "tools": [
            "spotify_playback", "spotify_devices", "spotify_queue", "spotify_search",
            "spotify_playlists", "spotify_albums", "spotify_library",
        ],
        "includes": []
    },


    # Scenario-specific toolsets

    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal", "process_manage"],
        "includes": ["web", "file"]  # For searching error messages and solutions, and file operations
    },

    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": [],
        "includes": ["web", "vision", "image_gen"]
    },

    # Coding posture, auto-selected in a code workspace (agent/coding_context.py).
    # `desktop_ui` is folded in separately by the GUI gateway for desktop sessions.
    "coding": {
        "description": "Coding-focused toolset: files, terminal, search, web docs, skills, todo, delegate, vision, browser",
        "tools": [
            "web_search", "web_extract",
            "terminal", "process_manage",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze",
            "skills_list", "skill_view", "skill_manage",
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
            "browser_exec",
            "todo_list", "memory",
            "session_search", "clarify",
            "execute_code", "delegate_task",
        ],
        "includes": [],
        # Per-session posture; never auto-recovered into platform tool config.
        "posture": True,
    },

    # Full Hermes toolsets (CLI + messaging platforms). All share the core tools;
    # there is deliberately no agent-callable send_message tool.

    "hermes-acp": {
        "description": "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI",
        "tools": [
            "web_search", "web_extract",
            "terminal", "process_manage",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze",
            "skills_list", "skill_view", "skill_manage",
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
            "browser_exec",
            "todo_list", "memory",
            "session_search",
            "execute_code", "delegate_task",
        ],
        "includes": []
    },

    "hermes-api-server": {
        "description": "OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message)",
        "tools": [
            "web_search", "web_extract",
            "terminal", "process_manage",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "skills_list", "skill_view", "skill_manage",
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
            "browser_exec",
            "todo_list", "memory",
            "session_search",
            "execute_code", "delegate_task",
            "cronjob_manage",
            "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
        ],
        "includes": []
    },

    "hermes-cli": {
        "description": "Full interactive CLI toolset - all default tools plus cronjob management",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-cron": {
        # Mirrors hermes-cli; `hermes tools` platform config filters it down and
        # _get_platform_tools() drops _DEFAULT_OFF_TOOLSETS unless user-enabled.
        "description": "Default cron toolset - same core tools as hermes-cli; gated by `hermes tools`",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-telegram": {
        "description": "Telegram bot toolset - full access for personal use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-discord": {
        "description": "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
        "tools": _HERMES_CORE_TOOLS + [
            "discord",
            "discord_admin",
        ],
        "includes": []
    },

    "hermes-whatsapp": {
        "description": "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-slack": {
        "description": "Slack bot toolset - full access for workspace use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-signal": {
        "description": "Signal bot toolset - encrypted messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-bluebubbles": {
        "description": "BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-homeassistant": {
        "description": "Home Assistant bot toolset - smart home event monitoring and control",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-email": {
        "description": "Email bot toolset - interact with Hermes via email (IMAP/SMTP)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-mattermost": {
        "description": "Mattermost bot toolset - self-hosted team messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-matrix": {
        "description": "Matrix bot toolset - decentralized encrypted messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-dingtalk": {
        "description": "DingTalk bot toolset - enterprise messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-feishu": {
        "description": "Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)",
        "tools": _HERMES_CORE_TOOLS + [
            "feishu_doc_read",
            "feishu_drive_list_comments",
            "feishu_drive_list_comment_replies",
            "feishu_drive_reply_comment",
            "feishu_drive_add_comment",
        ],
        "includes": []
    },

    "hermes-weixin": {
        "description": "Weixin bot toolset - personal WeChat messaging via iLink (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-qqbot": {
        "description": "QQBot toolset - QQ messaging via Official Bot API v2 (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom": {
        "description": "WeCom bot toolset - enterprise WeChat messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom-callback": {
        "description": "WeCom callback toolset - enterprise self-built app messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-yuanbao": {
        "description": "Yuanbao Bot 元宝消息平台工具集 - 群信息、成员查询、私聊、贴纸表情",
        "tools": _HERMES_CORE_TOOLS + [
            "yb_query_group_info",
            "yb_query_group_members",
            "yb_send_dm",
            "yb_search_sticker",
            "yb_send_sticker",
        ],
        "module": "tools.yuanbao_tools",
        "includes": []
    },

    "hermes-sms": {
        "description": "SMS bot toolset - interact with Hermes via SMS (Twilio)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-webhook": {
        "description": "Webhook toolset - receive and process external webhook events",
        "tools": _HERMES_WEBHOOK_SAFE_TOOLS,
        "includes": []
    },

    "hermes-gateway": {
        "description": "Gateway toolset - union of all messaging platform tools",
        "tools": [],
        "includes": ["hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack", "hermes-signal", "hermes-bluebubbles", "hermes-homeassistant", "hermes-email", "hermes-sms", "hermes-mattermost", "hermes-matrix", "hermes-dingtalk", "hermes-feishu", "hermes-wecom", "hermes-wecom-callback", "hermes-weixin", "hermes-qqbot", "hermes-webhook", "hermes-yuanbao"]
    }
}


def _registry():
    """Live tool registry, or None when tools.registry can't be imported."""
    try:
        from tools.registry import registry
        return registry
    except Exception:
        return None


def _registry_generation() -> Tuple[int, int]:
    reg = _registry()
    return (id(reg), getattr(reg, "_generation", 0)) if reg is not None else (0, 0)


def _static_copy(toolset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **toolset,
        "tools": list(toolset.get("tools", [])),
        "includes": list(toolset.get("includes", [])),
    }


def get_toolset(name: str, *, include_registry: bool = True) -> Optional[Dict[str, Any]]:
    """Return a toolset definition, or None if unknown.

    include_registry=True merges tools plugins/overlays registered into this
    toolset and resolves registry-only (plugin/MCP) toolsets and aliases.
    include_registry=False returns only the static TOOLSETS entry (copied), so
    platform reverse-mapping (#49622) is unaffected by registry additions.
    """
    toolset = TOOLSETS.get(name)
    if not include_registry:
        return _static_copy(toolset) if toolset else None

    registry = _registry()
    if registry is None:
        return toolset if toolset else None

    if toolset:
        merged_tools = sorted(set(toolset.get("tools", [])) | set(registry.get_tool_names_for_toolset(name)))
        return {**toolset, "tools": merged_tools}

    registry_toolset = name
    description = f"Plugin toolset: {name}"
    if name not in _get_plugin_toolset_names():
        registry_toolset = registry.get_toolset_alias_target(name)
        if not registry_toolset:
            return None
        description = f"MCP server '{name}' tools"
    else:
        reverse_aliases = {
            canonical: alias
            for alias, canonical in _get_registry_toolset_aliases().items()
            if alias not in TOOLSETS
        }
        if reverse_aliases.get(name):
            description = f"MCP server '{reverse_aliases[name]}' tools"

    return {
        "description": description,
        "tools": registry.get_tool_names_for_toolset(registry_toolset),
        "includes": [],
    }


def bundle_non_core_tools(toolset_name: str) -> Set[str]:
    """A bundle's tools minus _HERMES_CORE_TOOLS (one level of includes).

    Bundles are `_HERMES_CORE_TOOLS + extras`; disabling one must not strip the
    core tools every other toolset shares. One `includes` pass suffices because
    only hermes-gateway nests bundles. Unknown names: full resolution minus core.
    """
    core = set(_HERMES_CORE_TOOLS)
    ts_def = get_toolset(toolset_name)
    if not (ts_def and "tools" in ts_def):
        return set(resolve_toolset(toolset_name)) - core
    to_remove = set(ts_def["tools"]) - core
    for inc in ts_def.get("includes", []):
        inc_def = get_toolset(inc)
        if inc_def and "tools" in inc_def:
            to_remove.update(set(inc_def["tools"]) - core)
    return to_remove


# Memo keyed on (name, include_registry, id(registry), registry generation).
# Engages only at the public entry (visited is None); recursion is untouched.
_resolve_toolset_memo: Dict[Tuple[str, bool, int, int], List[str]] = {}


def resolve_toolset(name: str, visited: Set[str] = None, *, include_registry: bool = True) -> List[str]:
    """Recursively resolve a toolset (and its includes) to a sorted tool-name list.

    include_registry=False resolves the static TOOLSETS view only (#49622).
    """
    external_call = visited is None
    if external_call:
        memo_key = (name, include_registry, *_registry_generation())
        cached = _resolve_toolset_memo.get(memo_key)
        if cached is not None:
            return list(cached)
        visited = set()

    # "all"/"*" span every toolset so new toolsets are included automatically.
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            all_tools.update(resolve_toolset(toolset_name, visited.copy(), include_registry=include_registry))
        return sorted(all_tools)

    # Diamond include or cycle: return [] silently — the tools were (or will
    # be) collected via another path, so this is not an error.
    if name in visited:
        return []
    visited.add(name)

    toolset = get_toolset(name, include_registry=include_registry)
    if not toolset:
        # Registered plugin platforms get an implicit hermes-<platform> bundle:
        # core tools + whatever the plugin registered under the platform name.
        if include_registry and name.startswith("hermes-"):
            platform_name = name[len("hermes-"):]
            try:
                from gateway.platform_registry import platform_registry
                if platform_registry.is_registered(platform_name):
                    plugin_tools = set(_HERMES_CORE_TOOLS)
                    registry = _registry()
                    if registry is not None:
                        try:
                            plugin_tools.update(e.name for e in registry.get_all_entries() if e.toolset == platform_name)
                        except Exception:
                            pass
                    return list(plugin_tools)
            except Exception:
                pass
        return []

    tools = set(toolset.get("tools", []))
    for included_name in toolset.get("includes", []):
        tools.update(resolve_toolset(included_name, visited, include_registry=include_registry))

    result = sorted(tools)
    if external_call:
        # Stale-generation entries are never hit again; bound the memo.
        if len(_resolve_toolset_memo) >= 256:
            _resolve_toolset_memo.clear()
        _resolve_toolset_memo[(name, include_registry, *_registry_generation())] = list(result)
    return result


def _get_plugin_toolset_names() -> Set[str]:
    """Registry toolset names absent from the static TOOLSETS dict."""
    registry = _registry()
    if registry is None:
        return set()
    try:
        return {n for n in registry.get_registered_toolset_names() if n not in TOOLSETS}
    except Exception:
        return set()


def _get_registry_toolset_aliases() -> Dict[str, str]:
    registry = _registry()
    if registry is None:
        return {}
    try:
        return registry.get_registered_toolset_aliases()
    except Exception:
        return {}


def _plugin_display_names() -> List[str]:
    """Plugin toolset names, shown under their first non-static alias when one exists."""
    aliases = _get_registry_toolset_aliases()
    names = []
    for ts_name in _get_plugin_toolset_names():
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                names.append(alias)
                break
        else:
            names.append(ts_name)
    return names


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """All toolset definitions: static plus plugin-registered."""
    result = dict(TOOLSETS)
    for display_name in _plugin_display_names():
        if display_name in result:
            continue
        toolset = get_toolset(display_name)
        if toolset:
            result[display_name] = toolset
    return result


def get_toolset_names() -> List[str]:
    """Sorted names of all toolsets (static + plugin), excluding aliases."""
    return sorted(set(TOOLSETS.keys()) | set(_plugin_display_names()))


def validate_toolset(name: str) -> bool:
    if name in {"all", "*"} or name in TOOLSETS:
        return True
    return name in _get_plugin_toolset_names() or name in _get_registry_toolset_aliases()


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None
) -> None:
    """Register a runtime toolset in TOOLSETS."""
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }


def get_toolset_info(name: str) -> Dict[str, Any]:
    """Toolset definition plus its resolved tools, or None if unknown."""
    toolset = get_toolset(name)
    if not toolset:
        return None
    resolved_tools = resolve_toolset(name)
    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved_tools,
        "tool_count": len(resolved_tools),
        "is_composite": bool(toolset["includes"])
    }
