"""Slash command definitions and autocomplete for the Hermes CLI.

Central registry for all slash commands. Every consumer -- CLI help, gateway
dispatch, Telegram BotCommands, Slack subcommand mapping, autocomplete --
derives its data from ``COMMAND_REGISTRY``.

To add a command: add a ``CommandDef`` entry to ``COMMAND_REGISTRY``.
To add an alias: set ``aliases=("short",)`` on the existing ``CommandDef``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from utils import is_truthy_value
from hermes_constants import INDICATOR_STYLES

# (config-file signature, personalities) memo for /personality completion.
_personalities_memo: Optional[
    Tuple[Tuple[Optional[str], Optional[int], Optional[int]], Dict[str, Any]]
] = None


def _personalities_from_cli_config() -> Dict[str, Any]:
    """Return the available personalities map, memoised on config mtime.

    Wraps ``available_personalities(load_cli_config())`` — the single owner of
    built-ins + user overrides. load_cli_config() does a full YAML parse + deep
    merge on every call and the completer runs per keystroke; the result only
    changes when config.yaml changes on disk, so keying on path+mtime+size
    keeps the memo freshness-correct (same pattern as load_env). Falls back to
    a fresh load when the file cannot be stat'ed.
    """
    global _personalities_memo
    from cli import load_cli_config
    from hermes_cli.personality import available_personalities

    try:
        from hermes_cli.config import get_config_path

        cfg_path = get_config_path()
        st = cfg_path.stat()
        sig = (str(cfg_path), st.st_mtime_ns, st.st_size)
    except Exception:
        sig = (None, None, None)

    if _personalities_memo is not None and _personalities_memo[0] == sig:
        return _personalities_memo[1]

    personalities = available_personalities(load_cli_config())
    _personalities_memo = (sig, personalities)
    return personalities

logger = logging.getLogger(__name__)

# prompt_toolkit is optional (only the completer/auto-suggest need it); the
# gateway must still import this module for the registry without it.
try:
    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    from prompt_toolkit.completion import Completer, Completion
except ImportError:  # pragma: no cover
    AutoSuggest = object  # type: ignore[assignment,misc]
    Completer = object    # type: ignore[assignment,misc]
    Suggestion = None     # type: ignore[assignment]
    Completion = None     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CommandDef dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command."""

    name: str                          # canonical name without slash: "background"
    description: str                   # human-readable description
    category: str                      # "Session", "Configuration", etc.
    aliases: tuple[str, ...] = ()      # alternative names: ("bg",)
    args_hint: str = ""                # argument placeholder: "<prompt>", "[name]"
    subcommands: tuple[str, ...] = ()  # tab-completable subcommands
    cli_only: bool = False             # only available in CLI
    gateway_only: bool = False         # only available in gateway/messaging
    gateway_config_gate: str | None = None  # config dotpath; when truthy, overrides cli_only for gateway
    # Mid-run (agent busy) gateway behavior, driving the Guard-2 dispatcher in
    # gateway/run.py (_dispatch_busy_slash_command):
    #   "dispatch"                — run while busy (normal handler, or the
    #                               mid-run variant named by ``busy_handler``).
    #   "reject"                  — refuse mid-run; generic "Agent is running"
    #                               catch-all unless ``busy_handler`` names a
    #                               command-specific reject message.
    #   "interrupt_then_dispatch" — interrupt the agent first (/stop, /new,
    #                               /reset); Guard 1 (platforms/base.py) routes
    #                               these via is_interrupt_then_dispatch().
    busy_policy: str = "reject"
    # Key of a special mid-run handler in gateway/run.py's Guard-2 table for
    # commands whose busy behavior differs from their normal handler.
    busy_handler: str | None = None
    # Key in ``hermes_cli.slash_exec.EXECUTORS`` — a pure formatter producing
    # the surface-independent core text; surfaces apply only their decoration.
    # A string key (not a callable) keeps this module import-light for the
    # gateway (no prompt_toolkit, no executor dependencies).
    execute: str | None = None
    # Desktop composer: ``options`` | ``text`` | ``mixed``. ``None`` is inferred.
    argument_mode: str | None = None
    # Desktop availability. ``None`` = offered; ``hidden`` = runs but stays out
    # of the popover; otherwise a reason (terminal / messaging / settings / …).
    desktop: str | None = None


# Valid values for CommandDef.busy_policy (see field docs above).
VALID_BUSY_POLICIES: frozenset[str] = frozenset(
    {"dispatch", "reject", "interrupt_then_dispatch"}
)


# ---------------------------------------------------------------------------
# Central registry -- single source of truth
# ---------------------------------------------------------------------------

COMMAND_REGISTRY: list[CommandDef] = [
    # Session
    CommandDef("start", "Acknowledge platform start pings without a reply", "Session",
               gateway_only=True, busy_policy="dispatch", busy_handler="start"),
    CommandDef("new", "Start a new session (fresh session ID + history)", "Session",
               aliases=("reset",), args_hint="[name]",
               busy_policy="interrupt_then_dispatch", busy_handler="new"),
    CommandDef("topic", "Enable or inspect Telegram DM topic sessions", "Session",
               gateway_only=True, args_hint="[off|help|session-id]"),
    CommandDef("clear", "Clear screen and start a new session", "Session",
               cli_only=True, desktop="terminal"),
    CommandDef("redraw", "Force a full UI repaint (recovers from terminal drift)", "Session",
               cli_only=True, desktop="terminal"),
    CommandDef("history", "Show conversation history", "Session",
               cli_only=True, desktop="terminal"),
    CommandDef("save", "Export the current conversation (bare /save shows usage)", "Session",
               args_hint="<json|md|html> [filename] [redact]"),
    CommandDef("retry", "Retry the last message (resend to agent)", "Session"),
    CommandDef("prompt", "Compose your next prompt in $EDITOR (markdown), then send it", "Session",
               cli_only=True, args_hint="[initial text]", aliases=("compose",)),
    CommandDef("undo", "Back up N user turns and re-prompt (default 1)", "Session",
               args_hint="[N]"),
    CommandDef("title", "Set a title for the current session", "Session",
               args_hint="[name]"),
    CommandDef("handoff", "Hand off this session to a messaging platform (Telegram, Discord, etc.)", "Session",
               args_hint="<platform>", cli_only=True, argument_mode="options"),
    CommandDef("branch", "Branch the current session (explore a different path)", "Session",
               aliases=("fork",), args_hint="[name]"),
    CommandDef("worktree", "Show, list, create, or prune isolated git worktrees", "Session",
               cli_only=True, args_hint="[new [name]|list|prune [--dry-run]]",
               subcommands=("new", "list", "prune")),
    CommandDef("compress", "Compress conversation context (add 'here [N]' to keep recent N turns; --preview shows what would happen)", "Session",
               aliases=("compact",), args_hint="[here [N] | focus topic | --preview|--dry-run]"),
    CommandDef("rollback", "List or restore filesystem checkpoints (restores keep your hand-edits; --all overrides)", "Session",
               args_hint="[number] [--all]"),
    CommandDef("snapshot", "Create or restore state snapshots of Hermes config/state", "Session",
               cli_only=True, aliases=("snap",), args_hint="[create|restore <id>|prune]",
               desktop="terminal"),
    CommandDef("export", "Export a profile (config, skills, theme) to a shareable archive", "Configuration",
               cli_only=True, args_hint="[profile] [-o output.tar.gz]"),
    CommandDef("import", "Import a shared profile archive as a new profile", "Configuration",
               cli_only=True, args_hint="<archive.tar.gz> [--name <name>]"),
    CommandDef("stop", "Kill all running background processes", "Session",
               busy_policy="interrupt_then_dispatch", busy_handler="stop"),
    CommandDef("pause", "Pause new work globally (emergency stop); '/pause off' resumes", "Session",
               gateway_only=True, args_hint="[reason | off]",
               busy_policy="dispatch"),
    CommandDef("approve", "Approve a pending dangerous command", "Session",
               gateway_only=True, args_hint="[session|always]", busy_policy="dispatch",
               desktop="messaging"),
    CommandDef("deny", "Deny a pending dangerous command (optionally with a reason)", "Session",
               gateway_only=True, args_hint="[all] [reason]", busy_policy="dispatch",
               desktop="messaging"),
    CommandDef("bg", "Run a prompt in a separate background session", "Session",
               args_hint="<prompt>", busy_policy="dispatch"),
    CommandDef("btw", "Ask a side question about the current conversation without interrupting it", "Session",
               args_hint="<question>", busy_policy="dispatch"),
    CommandDef("agents", "Show active agents and running tasks", "Session",
               aliases=("tasks",), busy_policy="dispatch"),
    CommandDef("journey", "Open the learning journey timeline",
               "Session", aliases=("learning", "memory-graph"), cli_only=True,
               args_hint="[list|delete <id>|edit <id>]",
               subcommands=("list", "delete", "edit")),
    CommandDef("queue", "Queue a prompt for the next turn (doesn't interrupt)", "Session",
               aliases=("q",), args_hint="<prompt>",
               busy_policy="dispatch", busy_handler="queue"),
    CommandDef("steer", "Inject a message after the next tool call without interrupting", "Session",
               args_hint="<prompt>", busy_policy="dispatch", busy_handler="steer"),
    CommandDef("goal", "Set a standing goal Hermes works on across turns until achieved", "Session",
               args_hint="[text | draft <text> | show | gate add <cmd> | pause | resume | clear | status | wait <pid> | unwait]",
               argument_mode="mixed", busy_policy="dispatch", busy_handler="goal"),
    CommandDef("heartbeat", "Set a recurring prompt that re-enters this session when idle", "Session",
               aliases=("hb",), args_hint="[every <interval> <prompt> | status | pause | resume | clear]",
               subcommands=("status", "pause", "resume", "clear"),
               busy_policy="dispatch"),
    CommandDef("refine", "Review this conversation now and save lessons to memory/skills", "Session",
               args_hint="[focus instructions]"),
    CommandDef("review", "Spawn an independent subagent to review the work just discussed (PR, code, docs)", "Session",
               args_hint="[review instructions]"),
    CommandDef("loop", "Re-run a prompt on a recurring interval in this session", "Session",
               aliases=("proactive",),
               args_hint="[interval] <prompt> [--times N] [--until <condition>] | status | pause | resume | stop",
               argument_mode="mixed", busy_policy="dispatch", busy_handler="loop"),
    CommandDef("plan", "Write a markdown implementation plan to .hermes/plans/ without executing anything", "Session",
               args_hint="[task]"),
    CommandDef("moa", "Run one prompt through the default Mixture of Agents preset, then restore your model", "Session",
               args_hint="<prompt>", busy_policy="reject", busy_handler="moa"),
    CommandDef("subgoal", "Add or manage extra criteria on the active goal", "Session",
               args_hint="[text | remove N | clear]", busy_policy="dispatch"),
    CommandDef("status", "Show session, model, token, and context info", "Session",
               busy_policy="dispatch"),
    CommandDef("egress", "Show Docker egress proxy status", "Session",
               args_hint="[status]", subcommands=("status",),
               busy_policy="dispatch", busy_handler="egress",
               execute="egress"),
    CommandDef("context", "Show detailed context window view with usage gauge, category breakdown, compression stats, and throughput", "Session",
               aliases=("ctx",), args_hint="[all]", subcommands=("all",),
               busy_policy="dispatch"),
    CommandDef("whoami", "Show your slash command access (admin / user)", "Info"),
    CommandDef("profile", "Show active profile name and home directory", "Info",
               busy_policy="dispatch", execute="profile"),
    CommandDef("sethome", "Set this chat as the home channel", "Session",
               gateway_only=True, aliases=("set-home",), desktop="terminal"),
    CommandDef("resume", "Resume a previously-named session", "Session",
               args_hint="[name]", argument_mode="mixed"),

    # Configuration
    CommandDef("sessions", "Browse and resume previous sessions", "Session"),

    # Configuration
    CommandDef("config", "Show current configuration", "Configuration",
               cli_only=True, desktop="terminal"),
    CommandDef("model", "Switch model (session-scoped; --global to persist)", "Configuration",
               args_hint="[model] [--provider name] [--global|--session] [--refresh]",
               busy_policy="reject", busy_handler="model", desktop="hidden"),
    CommandDef("codex-runtime", "Toggle codex app-server runtime for OpenAI/Codex models",
               "Configuration", aliases=("codex_runtime",),
               args_hint="[auto|codex_app_server]",
               busy_policy="reject", busy_handler="codex-runtime"),

    CommandDef("personality", "Set a predefined personality", "Configuration",
               args_hint="[name]", argument_mode="options"),
    CommandDef("statusbar", "Toggle the context/model status bar", "Configuration",
               cli_only=True, aliases=("sb",), desktop="terminal"),
    CommandDef("battery", "Toggle a color-coded battery indicator in the status bar",
               "Configuration", cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status")),
    CommandDef("timestamps", "Toggle [HH:MM] timestamps on messages and /history", "Configuration",
               cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status"), aliases=("ts",)),
    CommandDef("diff", "Show git changes in the working directory", "Info",
               args_hint="[staged|all|session] [--stat] [path...]",
               subcommands=("staged", "all", "session")),
    CommandDef("verbose", "Cycle tool progress display: off -> new -> all -> verbose",
               "Configuration", cli_only=True,
               gateway_config_gate="display.tool_progress_command",
               busy_policy="dispatch", desktop="terminal"),
    CommandDef("focus", "Toggle focus view — show only your prompt and the final response",
               "Configuration", cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status")),
    CommandDef("footer", "Toggle gateway runtime-metadata footer on final replies",
               "Configuration", args_hint="[on|off|status]",
               subcommands=("on", "off", "status"), busy_policy="dispatch",
               desktop="terminal"),
    CommandDef("yolo", "Toggle YOLO mode (skip all dangerous command approvals)",
               "Configuration", busy_policy="dispatch"),
    CommandDef("approvals", "Show or set the persistent dangerous-command approval mode",
               "Configuration", args_hint="[manual|smart|off]",
               subcommands=("manual", "smart", "off")),
    CommandDef("reasoning", "Manage reasoning effort and display", "Configuration",
               args_hint="[level|show|hide|full|clamp] [--global]",
               subcommands=("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "show", "hide", "on", "off", "full", "clamp", "--global"),
               desktop="advanced"),
    CommandDef("fast", "Fast mode — OpenAI Priority Processing / Anthropic Fast Mode (normal/fast/auto/cold)", "Configuration",
               args_hint="[normal|fast|auto|cold|status] [--global]",
               subcommands=("normal", "fast", "auto", "cold", "status", "on", "off", "--global"),
               desktop="advanced"),
    CommandDef("skin", "Show or change the display skin/theme", "Configuration",
               cli_only=True, args_hint="[name]", argument_mode="options"),
    CommandDef("indicator", "Pick the TUI busy-indicator style", "Configuration",
               cli_only=True, args_hint=f"[{'|'.join(INDICATOR_STYLES)}]",
               subcommands=INDICATOR_STYLES, desktop="terminal"),
    CommandDef("voice", "Toggle voice mode", "Configuration",
               args_hint="[on|off|tts|status]", subcommands=("on", "off", "tts", "status"),
               desktop="composer-voice"),
    CommandDef("wake", "Toggle the 'Hey Hermes' wake word listener", "Configuration",
               cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status")),
    CommandDef("busy", "Control how messages behave while Hermes is working", "Configuration",
               args_hint="[queue|steer|interrupt|status]",
               subcommands=("queue", "steer", "interrupt", "status"),
               busy_policy="dispatch", desktop="terminal"),

    # Tools & Skills
    CommandDef("tools", "Manage tools: /tools [list|disable|enable] [name...]", "Tools & Skills",
               args_hint="[list|disable|enable] [name...]", cli_only=True,
               argument_mode="options"),
    CommandDef("toolsets", "List available toolsets", "Tools & Skills",
               cli_only=True, desktop="terminal"),
    CommandDef("skills", "Search, install, inspect, or manage skills",
               "Tools & Skills", cli_only=True,
               gateway_config_gate="skills.write_approval",
               subcommands=("search", "browse", "inspect", "install", "audit",
                            "pending", "approve", "reject", "diff", "approval"),
               desktop="settings"),
    CommandDef("memory", "Review pending memory writes / toggle the approval gate",
               "Tools & Skills",
               args_hint="[pending|approve|reject|approval] [id|on|off]",
               subcommands=("pending", "approve", "reject", "approval")),
    CommandDef("bundles", "List skill bundles (aliases /<name> for multiple skills)",
               "Tools & Skills", execute="bundles"),
    CommandDef("pet", "Toggle or adopt a petdex mascot (/pet, /pet list, /pet <slug>)", "Tools & Skills",
               cli_only=True, args_hint="[toggle|list|scale <n>|<slug>]", subcommands=("toggle", "list", "scale", "off")),
    CommandDef("hatch", "Generate a new petdex pet from a description",
               "Tools & Skills", cli_only=True, aliases=("generate-pet",), args_hint="[description]"),
    CommandDef("learn", "Learn a reusable skill from anything you describe (dirs, URLs, this chat, notes)",
               "Tools & Skills", args_hint="<what to learn from>"),
    CommandDef("init", "Generate or update AGENTS.md project instructions from a repo scan",
               "Tools & Skills", args_hint="[notes]"),
    CommandDef("cron", "Manage scheduled tasks", "Tools & Skills",
               cli_only=True, args_hint="[subcommand]",
               subcommands=("list", "add", "create", "edit", "pause", "resume", "run", "remove"),
               desktop="terminal"),
    CommandDef("suggestions", "Review suggested automations (accept/dismiss)",
               "Tools & Skills", aliases=("suggest",), args_hint="[accept|dismiss N | catalog]",
               subcommands=("accept", "dismiss", "catalog", "clear")),
    CommandDef("blueprint", "Set up an automation from a blueprint template",
               "Tools & Skills", aliases=("bp",), args_hint="[name] [slot=value ...]"),
    CommandDef("curator", "Background skill maintenance (status, run, pin, archive, list-archived)",
               "Tools & Skills", args_hint="[subcommand]",
               subcommands=("status", "run", "pause", "resume", "pin", "unpin", "restore", "list-archived"),
               desktop="advanced"),
    CommandDef("kanban", "Multi-profile collaboration board (tasks, links, comments)",
               "Tools & Skills", args_hint="[subcommand]",
               subcommands=("init", "boards", "create", "list", "ls", "show", "assign",
                            "reclaim", "reassign", "diagnostics", "diag", "link", "unlink",
                            "claim", "comment", "complete", "edit", "block", "unblock",
                            "archive", "tail", "dispatch", "stats", "notify-subscribe",
                            "notify-list", "notify-unsubscribe", "log", "runs",
                            "heartbeat", "assignees", "context", "specify", "gc"),
               busy_policy="dispatch", desktop="advanced"),
    CommandDef("reload", "Reload .env variables into the running session", "Tools & Skills",
               cli_only=True, desktop="terminal"),
    CommandDef("reload-mcp", "Reload MCP servers from config", "Tools & Skills",
               aliases=("reload_mcp",), desktop="advanced"),
    CommandDef("reload-skills", "Re-scan ~/.hermes/skills/ for newly installed or removed skills",
               "Tools & Skills", aliases=("reload_skills",), desktop="advanced"),
    CommandDef("browser", "Connect browser tools to your live Chromium-family browser via CDP, or switch to Browser Use mode", "Tools & Skills",
               cli_only=True, args_hint="[connect|disconnect|status|use]",
               subcommands=("connect", "disconnect", "status", "use")),
    CommandDef("plugins", "List installed plugins and their status",
               "Tools & Skills", cli_only=True, desktop="terminal"),

    # Info
    CommandDef("commands", "Browse all commands and skills (paginated)", "Info",
               gateway_only=True, args_hint="[page]", busy_policy="dispatch",
               execute="gateway_commands"),
    CommandDef("help", "Show available commands (/help skills lists skill commands, /help <text> filters)", "Info", busy_policy="dispatch",
               execute="gateway_help", args_hint="[skills|<filter>]"),
    CommandDef("palette", "Open the fuzzy command palette (also Ctrl+P)", "Info",
               cli_only=True, busy_policy="dispatch"),
    CommandDef("restart", "Gracefully restart the gateway after draining active runs", "Session",
               gateway_only=True, busy_policy="dispatch", desktop="terminal"),
    CommandDef("usage", "Show token usage and rate limits; `reset` redeems a banked Codex limit reset", "Info",
               args_hint="[reset [--force]]"),
    CommandDef("subscription", "View your Nous plan and change it in the browser", "Info",
               cli_only=True, aliases=("upgrade",)),
    CommandDef("topup", "Show your Nous balance and manage billing on the portal", "Info"),
    CommandDef("insights", "Show usage insights and analytics", "Info",
               args_hint="[days]", desktop="advanced"),
    CommandDef("platforms", "Show gateway/messaging platform status", "Info",
               cli_only=True, aliases=("gateway",), desktop="terminal"),
    CommandDef("platform", "Pause, resume, or list a failing gateway platform", "Info",
               gateway_only=True, args_hint="<pause|resume|list> [name]"),
    CommandDef("copy", "Copy the last assistant response to clipboard", "Info",
               cli_only=True, args_hint="[number]", desktop="terminal"),
    CommandDef("paste", "Attach clipboard image from your clipboard", "Info",
               cli_only=True, desktop="terminal"),
    CommandDef("image", "Attach a local image file for your next prompt", "Info",
               cli_only=True, args_hint="<path>", desktop="terminal"),
    CommandDef("update", "Update Hermes Agent to the latest version", "Info",
               busy_policy="dispatch", desktop="terminal"),
    CommandDef("version", "Show Hermes Agent version", "Info", aliases=("v",),
               busy_policy="dispatch", execute="version"),
    CommandDef("debug", "Upload debug report (system info + logs) and get shareable links", "Info",
               args_hint="[nous|local]"),

    # Exit
    CommandDef("quit", "Exit the CLI (use --delete to also remove session history)", "Exit",
               cli_only=True, aliases=("exit",), args_hint="[--delete]",
               desktop="terminal"),
]


# Used only to distinguish ``mixed`` (subcommands plus free-text) from
# ``options`` (subcommand list only). A bare ``args_hint`` with no
# subcommands is always ``text`` — do not add tokens here for that path.
_PROSE_HINTS = ("<prompt>", "[text", "instructions", "[interval]", "<what")


def infer_argument_mode(cmd: CommandDef) -> str | None:
    """Composer mode: explicit on the CommandDef, else inferred from its args."""
    if cmd.argument_mode in {"options", "text", "mixed"}:
        return cmd.argument_mode
    hint = (cmd.args_hint or "").strip()
    if cmd.subcommands and hint and any(token in hint.lower() for token in _PROSE_HINTS):
        return "mixed"
    if cmd.subcommands:
        return "options"
    if hint:
        return "text"
    return None


def command_desktop_meta(cmd: CommandDef) -> dict[str, str | None]:
    """Wire shape for ``commands.catalog`` — reads the CommandDef, nothing else."""
    return {"argument_mode": infer_argument_mode(cmd), "desktop": cmd.desktop}


# ---------------------------------------------------------------------------
# Derived lookups -- rebuilt once at import time, refreshed by rebuild_lookups()
# ---------------------------------------------------------------------------

def _build_command_lookup() -> dict[str, CommandDef]:
    """Map every name and alias to its CommandDef."""
    lookup: dict[str, CommandDef] = {}
    for cmd in COMMAND_REGISTRY:
        lookup[cmd.name] = cmd
        for alias in cmd.aliases:
            lookup[alias] = cmd
    return lookup


_COMMAND_LOOKUP: dict[str, CommandDef] = _build_command_lookup()


def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to its CommandDef.

    Accepts names with or without the leading slash.
    """
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))


def _build_description(cmd: CommandDef) -> str:
    """Build a CLI-facing description string including usage hint."""
    if cmd.args_hint:
        return f"{cmd.description} (usage: /{cmd.name} {cmd.args_hint})"
    return cmd.description


# Backwards-compatible flat dict ("/command" -> description) and the same
# grouped by category. Both exclude gateway_only commands.
COMMANDS: dict[str, str] = {}
COMMANDS_BY_CATEGORY: dict[str, dict[str, str]] = {}
# Subcommands lookup: "/cmd" -> ["sub1", "sub2", ...]
SUBCOMMANDS: dict[str, list[str]] = {}
for _cmd in COMMAND_REGISTRY:
    if _cmd.subcommands:
        SUBCOMMANDS[f"/{_cmd.name}"] = list(_cmd.subcommands)
    if _cmd.gateway_only:
        continue
    _entries = {f"/{_cmd.name}": _build_description(_cmd)}
    for _alias in _cmd.aliases:
        _entries[f"/{_alias}"] = f"{_cmd.description} (alias for /{_cmd.name})"
    COMMANDS.update(_entries)
    COMMANDS_BY_CATEGORY.setdefault(_cmd.category, {}).update(_entries)


# /help renderer sub-groups for the ~46-command "Session" category. Category
# itself is load-bearing for gateway help, so commands are not re-tagged;
# unlisted Session commands fall under the base header. Bare names (no /).
HELP_SESSION_SUBGROUPS: dict[str, tuple[str, ...]] = {
    "Context": (
        "compress", "compact", "context", "ctx", "status",
    ),
    "Background & Automation": (
        "bg", "btw", "agents", "tasks", "queue", "q", "steer",
        "goal", "subgoal", "heartbeat", "hb", "refine", "loop", "proactive",
        "moa", "journey", "learning", "memory-graph",
    ),
}

# Fallback: derive subcommands from pipe patterns in args_hint ("[on|off|status]")
# for commands without an explicit ``subcommands`` field.
_PIPE_SUBS_RE = re.compile(r"[a-z]+(?:\|[a-z]+)+")
for _cmd in COMMAND_REGISTRY:
    key = f"/{_cmd.name}"
    if key in SUBCOMMANDS or not _cmd.args_hint:
        continue
    m = _PIPE_SUBS_RE.search(_cmd.args_hint)
    if m:
        SUBCOMMANDS[key] = m.group(0).split("|")


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------

# All names + aliases the gateway dispatches. Config-gated commands are
# included; their handler checks the gate at runtime.
GATEWAY_KNOWN_COMMANDS: frozenset[str] = frozenset(
    name
    for cmd in COMMAND_REGISTRY
    if not cmd.cli_only or cmd.gateway_config_gate
    for name in (cmd.name, *cmd.aliases)
)


def is_gateway_known_command(name: str | None) -> bool:
    """Return True if ``name`` is a built-in or plugin gateway slash command.

    Plugin commands are looked up lazily so importing this module never forces
    plugin discovery. Gateway code uses this to decide whether to emit
    ``command:<name>`` hooks — plugins get the same lifecycle events as built-ins.
    """
    if not name:
        return False
    return name in GATEWAY_KNOWN_COMMANDS or any(
        plugin_name == name for plugin_name, _d, _h in _iter_plugin_command_entries()
    )


# Commands with explicit mid-run handling (busy_policy != "reject"). Kept
# under its historical name for introspection/tests; the real bypass set is
# every resolvable command (see should_bypass_active_session).
ACTIVE_SESSION_BYPASS_COMMANDS: frozenset[str] = frozenset(
    cmd.name for cmd in COMMAND_REGISTRY if cmd.busy_policy != "reject"
)


def is_interrupt_then_dispatch(command_name: str | None) -> bool:
    """True when *command_name* (or alias) has busy_policy "interrupt_then_dispatch".

    Guard 1 (gateway/platforms/base.py) routes these through the cancel-handoff
    path that serializes cancellation + runner response + pending drain.
    """
    if not command_name:
        return False
    cmd = resolve_command(command_name)
    return cmd is not None and cmd.busy_policy == "interrupt_then_dispatch"


def should_bypass_active_session(command_name: str | None) -> bool:
    """Return True for any resolvable slash command.

    Every recognized slash command is dispatched mid-run — either by its
    Level-2 handler in gateway/run.py or by the "busy — wait or /stop first"
    catch-all — never queued: gateway.run's safety net discards command text
    that reaches the pending queue, so a queued mid-run /model would interrupt
    the agent AND vanish with a zero-char response.
    ACTIVE_SESSION_BYPASS_COMMANDS is the subset with explicit handlers.
    """
    return resolve_command(command_name) is not None if command_name else False


def _resolve_config_gates() -> set[str]:
    """Return canonical names of commands whose ``gateway_config_gate`` dotpath is truthy in config.yaml (empty set on any error)."""
    gated = [c for c in COMMAND_REGISTRY if c.gateway_config_gate]
    if not gated:
        return set()
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
    except Exception:
        return set()
    result: set[str] = set()
    for cmd in gated:
        val: Any = cfg
        for key in cmd.gateway_config_gate.split("."):
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if is_truthy_value(val, default=False):
            result.add(cmd.name)
    return result


def _is_gateway_available(cmd: CommandDef, config_overrides: set[str] | None = None) -> bool:
    """True if *cmd* appears in gateway surfaces: not ``cli_only``, or its config gate is truthy.

    Pass *config_overrides* from ``_resolve_config_gates()`` to avoid
    re-reading config per command.
    """
    if not cmd.cli_only:
        return True
    if cmd.gateway_config_gate:
        overrides = config_overrides if config_overrides is not None else _resolve_config_gates()
        return cmd.name in overrides
    return False


def _requires_argument(args_hint: str) -> bool:
    """Return True when selecting a command without text would be incomplete."""
    return args_hint.strip().startswith("<")


def gateway_help_lines() -> list[str]:
    """Generate gateway help text lines from the registry."""
    overrides = _resolve_config_gates()
    lines: list[str] = []
    for cmd in COMMAND_REGISTRY:
        if not _is_gateway_available(cmd, overrides):
            continue
        args = f" {cmd.args_hint}" if cmd.args_hint else ""
        alias_parts: list[str] = []
        for a in cmd.aliases:
            # Skip internal aliases like reload_mcp (underscore variant)
            if a.replace("-", "_") == cmd.name.replace("-", "_") and a != cmd.name:
                continue
            alias_parts.append(f"`/{a}`")
        alias_note = f" (alias: {', '.join(alias_parts)})" if alias_parts else ""
        lines.append(f"`/{cmd.name}{args}` -- {cmd.description}{alias_note}")
    return lines


def _iter_plugin_command_entries() -> list[tuple[str, str, str]]:
    """Return (name, description, args_hint) tuples for all plugin slash commands.

    Registered via :func:`hermes_cli.plugins.PluginContext.register_command`;
    surfaced like ``CommandDef`` entries (Telegram menu, Slack ``/hermes``
    map, Discord picker). Lookup is lazy so importing this module never
    forces plugin discovery (filesystem scans, env-dependent behavior).
    """
    try:
        from hermes_cli.plugins import get_plugin_commands
    except Exception:
        return []
    try:
        commands = get_plugin_commands() or {}
    except Exception:
        return []
    entries: list[tuple[str, str, str]] = []
    for name, meta in commands.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            continue
        description = str(meta.get("description") or f"Run /{name}")
        args_hint = str(meta.get("args_hint") or "").strip()
        entries.append((name, description, args_hint))
    return entries


def telegram_bot_commands(*, include_plugins: bool = True) -> list[tuple[str, str]]:
    """Return (command_name, description) pairs for Telegram setMyCommands.

    Names are Telegram-sanitized (hyphens → underscores); aliases are skipped
    (one menu entry per canonical command). Built-ins that require arguments
    are **included** — their handlers show usage text when selected bare, so
    hiding them hurts discoverability. Plugin commands requiring arguments are
    **excluded** because plugins may lack a no-arg fallback; callers needing
    source metadata pass ``include_plugins=False`` and use
    :func:`_collect_gateway_skill_entries`.
    """
    overrides = _resolve_config_gates()
    result: list[tuple[str, str]] = []
    for cmd in COMMAND_REGISTRY:
        if not _is_gateway_available(cmd, overrides):
            continue
        tg_name = _sanitize_telegram_name(cmd.name)
        if tg_name:
            result.append((tg_name, cmd.description))
    if include_plugins:
        for name, description, args_hint in _iter_plugin_command_entries():
            if _requires_argument(args_hint):
                continue
            tg_name = _sanitize_telegram_name(name)
            if tg_name:
                result.append((tg_name, description))
    return result


# Telegram allows 100 BotCommands; the 60-slot default keeps every built-in
# plus common skill commands visible while staying under the ~4KB payload
# limit. Tunable via platforms.telegram.extra.command_menu.max_commands.
_DEFAULT_TELEGRAM_MENU_MAX_COMMANDS = 60
_TELEGRAM_BOT_API_MAX_COMMANDS = 100
# priority_mode -> rank tables consulted in order ("configured" = user list,
# "default" = _TELEGRAM_MENU_PRIORITY); unranked candidates keep stable order after.
_TELEGRAM_PRIORITY_TIERS: dict[str, tuple[str, ...]] = {
    "prepend": ("configured", "default"),
    "append": ("default", "configured"),
    "replace": ("configured",),
}
_TELEGRAM_PRIORITY_MODES = frozenset(_TELEGRAM_PRIORITY_TIERS)

_TELEGRAM_MENU_PRIORITY = (
    # Most-typed everyday commands first.
    "help",
    "new",
    "stop",
    "status",
    "egress",
    "resume",
    "sessions",
    "model",
    # Maintenance / diagnostics — the ones that prompted this priority list.
    "debug",
    "restart",
    "update",
    "verbose",
    "commands",
    # Mid-turn session control.
    "approve",
    "deny",
    "queue",
    "steer",
    "bg",
    "btw",
    # Lower-priority but still useful operational built-ins.
    "reasoning",
    "usage",
    "platforms",
    "platform",
    "profile",
    "whoami",
)
"""Built-ins that must survive Telegram's small visible menu cap; everything
else stays dispatchable when typed manually."""


def _nested_mapping(root: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    node: Any = root
    for key in path:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key)
    return node if isinstance(node, Mapping) else {}


def _telegram_command_menu_config() -> dict[str, Any]:
    """Return normalized Telegram command-menu config with safe defaults.

    Canonical user-facing path:
    ``platforms.telegram.extra.command_menu``.
    """
    try:
        from hermes_cli.config import read_raw_config
        raw_cfg = read_raw_config() or {}
    except Exception:
        raw_cfg = {}
    if not isinstance(raw_cfg, Mapping):
        raw_cfg = {}

    menu_cfg = dict(_nested_mapping(raw_cfg, "platforms", "telegram", "extra", "command_menu"))

    max_commands = menu_cfg.get("max_commands", _DEFAULT_TELEGRAM_MENU_MAX_COMMANDS)
    try:
        max_commands = int(max_commands)
    except (TypeError, ValueError):
        max_commands = _DEFAULT_TELEGRAM_MENU_MAX_COMMANDS
    max_commands = max(1, min(_TELEGRAM_BOT_API_MAX_COMMANDS, max_commands))

    priority_mode = str(menu_cfg.get("priority_mode") or "prepend").strip().lower()
    if priority_mode not in _TELEGRAM_PRIORITY_MODES:
        priority_mode = "prepend"

    raw_priority = menu_cfg.get("priority")
    if isinstance(raw_priority, list):
        priority = [str(item) for item in raw_priority if str(item).strip()]
    elif isinstance(raw_priority, str) and raw_priority.strip():
        priority = [raw_priority]
    else:
        priority = []

    return {
        "max_commands": max_commands,
        "priority_mode": priority_mode,
        "priority": priority,
    }


def telegram_menu_max_commands() -> int:
    """Return configured Telegram BotCommand menu cap with safe bounds."""
    return int(_telegram_command_menu_config()["max_commands"])


def _dedupe_sanitized_names(raw_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_names:
        name = _sanitize_telegram_name(str(raw_name))
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return tuple(result)


def _prioritize_telegram_menu_candidates(
    candidates: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    """Order Telegram candidates while keeping default priority core-only.

    Candidate tuples contain ``(final_name, description, source, raw_name)``.
    ``raw_name`` preserves the pre-clamp command name so an explicitly
    configured long command remains addressable after Telegram name clamping.
    """
    menu_cfg = _telegram_command_menu_config()
    configured_rank = {n: i for i, n in enumerate(_dedupe_sanitized_names(menu_cfg["priority"]))}
    default_rank = {n: i for i, n in enumerate(_dedupe_sanitized_names(_TELEGRAM_MENU_PRIORITY))}
    # Tier order per mode: which rank tables win, in precedence order.
    # "replace" ignores the built-in defaults entirely.
    tiers = _TELEGRAM_PRIORITY_TIERS[menu_cfg["priority_mode"]]

    def _rank(stable_index: int, candidate: tuple[str, str, str, str]) -> tuple[int, int, int]:
        final_name, _desc, source, raw_name = candidate
        indexes = {
            "configured": configured_rank.get(raw_name, configured_rank.get(final_name)),
            "default": default_rank.get(final_name) if source == "core" else None,
        }
        for tier, table in enumerate(tiers):
            if indexes[table] is not None:
                return (tier, indexes[table], stable_index)
        return (len(tiers), 0, stable_index)

    return [c for _i, c in sorted(enumerate(candidates), key=lambda item: _rank(*item))]


_CMD_NAME_LIMIT = 32
"""Max command name length shared by Telegram and Discord."""

_TG_INVALID_CHARS = re.compile(r"[^a-z0-9_]")
_TG_MULTI_UNDERSCORE = re.compile(r"_{2,}")


def _sanitize_telegram_name(raw: str) -> str:
    """Convert a command/skill/plugin name to a valid Telegram command name.

    Telegram allows only lowercase a-z, 0-9 and underscores: lowercase →
    hyphens to underscores → strip other chars → collapse/strip underscores.
    """
    name = raw.lower().replace("-", "_")
    name = _TG_INVALID_CHARS.sub("", name)
    name = _TG_MULTI_UNDERSCORE.sub("_", name)
    return name.strip("_")


def _clamp_command_names(
    entries: Sequence[tuple[str, ...]],
    reserved: set[str],
) -> list[tuple[str, ...]]:
    """Enforce the 32-char Telegram/Discord command-name limit with collision avoidance.

    Over-long names are truncated; if that collides with *reserved* or an
    earlier entry, a 31-char prefix + digit ``0``-``9`` is tried, and the entry
    is silently dropped when all ten are taken. Duplicate names are dropped.
    Elements beyond ``(name, desc)`` pass through unchanged.
    """
    used: set[str] = set(reserved)
    result: list = []
    for entry in entries:
        name, desc, *extra = entry
        if len(name) > _CMD_NAME_LIMIT:
            candidate = name[:_CMD_NAME_LIMIT]
            if candidate in used:
                prefix = name[:_CMD_NAME_LIMIT - 1]
                for digit in range(10):
                    candidate = f"{prefix}{digit}"
                    if candidate not in used:
                        break
                else:
                    continue
            name = candidate
        if name in used:
            continue
        used.add(name)
        result.append((name, desc, *extra))
    return result


# ---------------------------------------------------------------------------
# Shared skill/plugin collection for gateway platforms
# ---------------------------------------------------------------------------

def _truncate_desc(desc: str, limit: int) -> str:
    """Clamp a menu description to *limit* chars with a ``...`` tail."""
    return desc if len(desc) <= limit else desc[:limit - 3] + "..."


def _iter_gateway_skills(platform: str):
    """Yield ``(cmd_key, info, rel_parts)`` for skills eligible as gateway slash commands.

    Scan roots are the local ``SKILLS_DIR`` plus every configured
    ``skills.external_dirs`` / trusted project skills dir (#8110, #18741) —
    a skill anywhere else, or under the hub dir (``SKILLS_DIR/.hub``), is
    skipped, as are skills disabled for *platform*. Paths are resolved on both
    sides so symlinked roots (macOS ``/var`` → ``/private/var``) still match,
    and matching is per path component so ``/my-skills`` never admits
    ``/my-skills-extra``. ``rel_parts`` is the skill dir relative to its root
    (``("creative", "ascii-art")``) for category derivation. Iterates
    ``sorted(skill_cmds)`` so first-wins collision handling is alphabetical.
    """
    from pathlib import Path

    from agent.skill_commands import get_skill_commands
    from agent.skill_utils import get_disabled_skill_names, get_external_skills_dirs, get_project_skills_dirs
    from tools.skills_tool import SKILLS_DIR

    try:
        disabled = get_disabled_skill_names(platform=platform)
    except Exception:
        disabled = set()
    hub_dir = (SKILLS_DIR / ".hub").resolve()
    roots = [SKILLS_DIR.resolve()]
    for getter in (get_external_skills_dirs, get_project_skills_dirs):
        try:
            for d in getter():
                try:
                    roots.append(Path(d).resolve())
                except Exception:
                    continue
        except Exception:
            pass

    skill_cmds = get_skill_commands()
    for cmd_key in sorted(skill_cmds):
        info = skill_cmds[cmd_key]
        skill_path = info.get("skill_md_path", "")
        if not skill_path:
            continue
        sp = Path(skill_path).resolve()
        if sp.is_relative_to(hub_dir):
            continue
        root = next((r for r in roots if sp.is_relative_to(r)), None)
        if root is None or info.get("name", "") in disabled:
            continue
        yield cmd_key, info, sp.parent.relative_to(root).parts


def _collect_gateway_skill_entries(
    platform: str,
    max_slots: int | None,
    reserved_names: set[str],
    desc_limit: int = 100,
    sanitize_name: "Callable[[str], str] | None" = None,
) -> tuple[list[tuple[str, str, str, str]], int]:
    """Collect plugin + skill entries for a gateway platform.

    Plugin slash commands come first and are never trimmed; skill commands
    (alphabetical, see :func:`_iter_gateway_skills`) fill the remaining
    *max_slots* — ``None`` returns every candidate for a caller applying its
    own global cap. *reserved_names* (built-in names) is mutated in place as
    names are claimed. *sanitize_name* runs before clamping and may return ""
    to skip an entry. Returns ``(entries, hidden_count)`` with entries of
    ``(name, description, cmd_key, raw_name)`` — ``cmd_key`` is the original
    skill key ("" for plugins); ``raw_name`` the sanitized pre-clamp name used
    for configured-priority matching.
    """
    # --- Tier 1: Plugin slash commands (never trimmed) ---------------------
    # Plugins have no cmd_key — "" placeholder; raw_name is the sanitized pre-clamp name.
    plugin_entries: list[tuple[str, str, str, str]] = []
    try:
        from hermes_cli.plugins import get_plugin_commands
        plugin_cmds = get_plugin_commands()
        for cmd_name in sorted(plugin_cmds):
            if platform == "telegram":
                args_hint = str(plugin_cmds[cmd_name].get("args_hint") or "").strip()
                if _requires_argument(args_hint):
                    continue
            name = sanitize_name(cmd_name) if sanitize_name else cmd_name
            if not name:
                continue
            desc = _truncate_desc(plugin_cmds[cmd_name].get("description", "Plugin command"), desc_limit)
            plugin_entries.append((name, desc, "", name))
    except Exception:
        pass
    plugin_entries = _clamp_command_names(plugin_entries, reserved_names)
    reserved_names.update(n for n, *_rest in plugin_entries)

    # --- Tier 2: Built-in skill commands (trimmed at cap) -----------------
    skill_entries: list[tuple[str, str, str, str]] = []
    try:
        for cmd_key, info, _rel in _iter_gateway_skills(platform):
            raw_name = cmd_key.lstrip("/")
            name = sanitize_name(raw_name) if sanitize_name else raw_name
            if not name:
                continue
            skill_entries.append((name, _truncate_desc(info.get("description", ""), desc_limit), cmd_key, name))
    except Exception:
        pass
    # Clamp names; cmd_key and raw_name survive any clamp-induced rename.
    skill_entries = _clamp_command_names(skill_entries, reserved_names)

    if max_slots is None:
        return plugin_entries + skill_entries, 0

    # Skills fill remaining slots — only tier that gets trimmed
    remaining = max(0, max_slots - len(plugin_entries))
    hidden_count = max(0, len(skill_entries) - remaining)
    return (plugin_entries + skill_entries[:remaining])[:max_slots], hidden_count


# ---------------------------------------------------------------------------
# Platform-specific wrappers
# ---------------------------------------------------------------------------

def telegram_menu_commands(max_commands: int = 100) -> tuple[list[tuple[str, str]], int]:
    """Return ``(menu_commands, hidden_count)`` for Telegram, capped to the Bot API limit.

    Tier order: core CommandDefs, then plugin slash commands, then skill
    commands (alphabetical; hub skills and telegram-disabled skills excluded).
    Tiers keep their relative order unless named in
    ``platforms.telegram.extra.command_menu.priority`` — explicit priority is
    applied to the combined list *before* the cap, so a prioritized dynamic
    command can displace an unprioritized core command.
    """
    core_commands = list(telegram_bot_commands(include_plugins=False))
    reserved_names = {n for n, _ in core_commands}
    entries, hidden_count = _collect_gateway_skill_entries(
        platform="telegram",
        max_slots=None,
        reserved_names=reserved_names,
        desc_limit=40,
        sanitize_name=_sanitize_telegram_name,
    )
    candidates = [(name, desc, "core", name) for name, desc in core_commands]
    for name, desc, cmd_key, raw_name in entries:
        source = "skill" if cmd_key else "plugin"
        candidates.append((name, desc, source, raw_name))

    candidates = _prioritize_telegram_menu_candidates(candidates)
    overflow_count = max(0, len(candidates) - max_commands)
    menu = [(name, desc) for name, desc, _source, _raw_name in candidates[:max_commands]]
    return menu, hidden_count + overflow_count


def discord_skill_commands_by_category(
    reserved_names: set[str],
) -> tuple[dict[str, list[tuple[str, str, str]]], list[tuple[str, str, str]], int]:
    """Return ``(categories, uncategorized, hidden_count)`` for Discord ``/skill`` autocomplete.

    Skills nested >= 2 levels under a scan root (``creative/ascii-art/SKILL.md``)
    are grouped under ``categories[top_level]``; root-level skills are
    *uncategorized*. Entries are ``(name, description, cmd_key)`` with names
    clamped to 32 chars and descriptions to 100. Eligibility follows
    :func:`_iter_gateway_skills`. No per-group cap is applied — the caller
    flattens everything into one autocomplete callback, which scales to
    thousands of entries; ``hidden_count`` only reports 32-char clamp
    collisions against reserved names or earlier skills.
    """
    categories: dict[str, list[tuple[str, str, str]]] = {}
    uncategorized: list[tuple[str, str, str]] = []
    # clamped-32-char-name → origin, so collisions get an actionable warning.
    # Reserved (gateway-builtin) names carry a sentinel so the warning can
    # distinguish "collided with a reserved command" from "two skills collided
    # on the 32-char clamp" — the latter is the rename-worthy case.
    _names_used: dict[str, str] = dict.fromkeys(reserved_names, "<reserved>")
    hidden = 0

    try:
        for cmd_key, info, rel_parts in _iter_gateway_skills("discord"):
            # Clamp to 32 chars (Discord per-command name limit). On collision
            # the first (alphabetical) skill wins and the loser is dropped from
            # the picker; warn loudly, since a silent ``hidden`` count gave
            # skill authors no way to discover the drop.
            discord_name = cmd_key.lstrip("/")[:32]
            if discord_name in _names_used:
                prior = _names_used[discord_name]
                if prior == "<reserved>":
                    logger.warning(
                        "Discord /skill: %r (from %r) collides on its 32-char "
                        "clamp with a reserved gateway command name %r — the "
                        "skill will not appear in the /skill autocomplete. "
                        "Rename the skill's frontmatter ``name:`` to differ "
                        "in its first 32 chars.",
                        discord_name, cmd_key, discord_name,
                    )
                else:
                    logger.warning(
                        "Discord /skill: %r and %r both clamp to %r on "
                        "Discord's 32-char command-name limit — only %r "
                        "will appear in the /skill autocomplete. Rename "
                        "one skill's frontmatter ``name:`` to differ in "
                        "its first 32 chars.",
                        prior, cmd_key, discord_name, prior,
                    )
                hidden += 1
                continue
            _names_used[discord_name] = cmd_key
            entry = (discord_name, _truncate_desc(info.get("description", ""), 100), cmd_key)
            # creative/ascii-art/SKILL.md → category "creative"; root-level skills are uncategorized.
            if len(rel_parts) >= 2:
                categories.setdefault(rel_parts[0], []).append(entry)
            else:
                uncategorized.append(entry)
    except Exception:
        pass

    return categories, uncategorized, hidden


# ---------------------------------------------------------------------------
# Slack native slash commands
# ---------------------------------------------------------------------------

# Slack slash command name constraints: lowercase a-z, 0-9, hyphens,
# underscores. Max 32 chars. Slack app manifest accepts up to 50 slash
# commands per app.
_SLACK_MAX_SLASH_COMMANDS = 50
_SLACK_NAME_LIMIT = 32
_SLACK_INVALID_CHARS = re.compile(r"[^a-z0-9_\-]")
_SLACK_RESERVED_COMMANDS = frozenset({
    # Built-in Slack slash commands that cannot be registered by apps.
    # https://slack.com/help/articles/201259356-Use-built-in-slash-commands
    "me", "status", "away", "dnd", "shrug", "remind", "msg", "feed",
    "who", "collapse", "expand", "leave", "join", "open", "search",
    "topic", "mute", "pro", "shortcuts",
})

# Canonical commands intentionally NOT given a native Slack slash slot. Slack
# caps apps at 50 slash commands and the registry is at that ceiling; rather
# than let the clamp silently drop whichever command sorts last (breaking the
# Telegram-parity test), low-frequency commands are routed through
# ``/hermes <command>`` on Slack only. They stay native on every other surface.
# Rule: when a new canonical command tips the registry past the cap, demote a
# rarer one-off lookup here (version, whoami, platform, diff, update, ...)
# rather than a recurring interactive surface (context, loop, save, approvals).
# Keep TIGHT and intentional — the parity test reads this set. (Aliases are
# never pinned ahead of canonicals: /bg and /btw became canonical commands
# instead, so they win first-pass slots on their own.)
_SLACK_VIA_HERMES_ONLY = frozenset({"topup", "moa", "debug", "egress", "init", "version", "diff", "update", "heartbeat", "refine", "review", "pause", "whoami", "platform", "insights"})


def _sanitize_slack_name(raw: str) -> str:
    """Convert a command name to a valid Slack slash command name.

    Slack allows lowercase a-z, digits, hyphens, and underscores. Max 32
    chars. Uppercase is lowercased; invalid chars are stripped.
    """
    name = raw.lower()
    name = _SLACK_INVALID_CHARS.sub("", name)
    name = name.strip("-_")
    return name[:_SLACK_NAME_LIMIT]


def slack_native_slashes() -> list[tuple[str, str, str]]:
    """Return (slash_name, description, usage_hint) triples for Slack.

    Every gateway-available command (canonical names first, then aliases,
    then plugin commands) becomes a standalone Slack slash, clamped to the
    50-command cap with duplicate avoidance. Names colliding with a Slack
    built-in (``/status``, ``/me``, ...) or listed in _SLACK_VIA_HERMES_ONLY
    are skipped. ``/hermes`` is always the first entry so the
    ``/hermes <command>`` form keeps working for anything dropped.
    """
    overrides = _resolve_config_gates()
    entries: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    # Reserve /hermes as the catch-all top-level command.
    entries.append(("hermes", "Talk to Hermes or run a subcommand", "[subcommand] [args]"))
    seen.add("hermes")

    def _add(name: str, desc: str, hint: str) -> None:
        slack_name = _sanitize_slack_name(name)
        if (
            not slack_name
            or slack_name in seen
            or slack_name in _SLACK_RESERVED_COMMANDS
            or slack_name in _SLACK_VIA_HERMES_ONLY
            or len(entries) >= _SLACK_MAX_SLASH_COMMANDS
        ):
            return
        # Slack description cap is 2000 chars; keep it short.
        entries.append((slack_name, desc[:140], hint[:100]))
        seen.add(slack_name)

    available = [cmd for cmd in COMMAND_REGISTRY if _is_gateway_available(cmd, overrides)]
    # Canonical names first so they win slots at the cap; aliases second.
    for cmd in available:
        _add(cmd.name, cmd.description, cmd.args_hint or "")
    for cmd in available:
        for alias in cmd.aliases:
            _add(alias, f"Alias for /{cmd.name} — {cmd.description}", cmd.args_hint or "")

    # Third pass: plugin commands.
    for name, description, args_hint in _iter_plugin_command_entries():
        _add(name, description, args_hint or "")

    return entries


def slack_app_manifest(request_url: str = "https://hermes-agent.local/slack/commands") -> dict[str, Any]:
    """Return the ``features.slash_commands`` manifest portion for all gateway slashes.

    ``request_url`` is schema-required but ignored in Socket Mode (a
    placeholder is fine). Only this portion is returned so we stay decoupled
    from the rest of the manifest users configure once in the Slack UI.
    """
    slashes = []
    for name, desc, usage in slack_native_slashes():
        entry = {
            "command": f"/{name}",
            "description": desc or f"Run /{name}",
            "should_escape": False,
            "url": request_url,
        }
        if usage:
            entry["usage_hint"] = usage
        slashes.append(entry)
    return {"features": {"slash_commands": slashes}}


def slack_subcommand_map() -> dict[str, str]:
    """Return name/alias -> "/command" mapping for the Slack ``/hermes`` handler, plugin commands included."""
    overrides = _resolve_config_gates()
    mapping: dict[str, str] = {}
    for cmd in COMMAND_REGISTRY:
        if not _is_gateway_available(cmd, overrides):
            continue
        mapping[cmd.name] = f"/{cmd.name}"
        for alias in cmd.aliases:
            mapping[alias] = f"/{alias}"
    for name, _description, _args_hint in _iter_plugin_command_entries():
        if name not in mapping:
            mapping[name] = f"/{name}"
    return mapping


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------


class SlashCommandCompleter(Completer):
    """Autocomplete for built-in slash commands, subcommands, and skill commands."""

    def __init__(
        self,
        skill_commands_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None,
        command_filter: Callable[[str], bool] | None = None,
        skill_bundles_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._skill_commands_provider = skill_commands_provider
        self._command_filter = command_filter
        self._skill_bundles_provider = skill_bundles_provider
        # Cached project file list for fuzzy @ completions
        self._file_cache: list[str] = []
        self._file_cache_time: float = 0.0
        self._file_cache_cwd: str = ""

    def _command_allowed(self, slash_command: str) -> bool:
        if self._command_filter is None:
            return True
        try:
            return bool(self._command_filter(slash_command))
        except Exception:
            return True

    def _iter_skill_commands(self) -> Mapping[str, dict[str, Any]]:
        if self._skill_commands_provider is None:
            return {}
        try:
            return self._skill_commands_provider() or {}
        except Exception:
            return {}

    def _iter_skill_bundles(self) -> Mapping[str, dict[str, Any]]:
        if self._skill_bundles_provider is None:
            return {}
        try:
            return self._skill_bundles_provider() or {}
        except Exception:
            return {}

    # -- stacked slash-skill completion helpers ---------------------------

    @staticmethod
    def _normalize_skill_token(token: str) -> str:
        """Canonicalize a typed skill token to its hyphenated /slug form.

        Mirrors resolve_skill_command_key() in agent/skill_commands.py:
        underscores (Telegram bot-command form) are interchangeable with
        hyphens.
        """
        return "/" + token.lstrip("/").replace("_", "-").lower()

    def _is_skill_command(self, token: str) -> bool:
        return self._normalize_skill_token(token) in self._iter_skill_commands()

    def _stacked_skill_completions(self, text: str):
        """Offer skill-command completions for stacked invocations (``/skill-a /skill-b do XYZ``).

        Keep suggesting while every completed token is a distinct skill
        command, the cap is not reached, and the current word starts with
        ``/``; once the chain breaks, offer nothing — instruction text must
        never be polluted with skill suggestions.
        """
        try:
            from agent.skill_commands import _MAX_STACKED_SKILLS as _cap
        except Exception:
            _cap = 5

        tokens = text.split()
        if text.endswith(" "):
            completed, current_word = tokens, ""
        else:
            completed, current_word = tokens[:-1], tokens[-1]

        # The chain must be unbroken: every completed token is a distinct
        # skill command, and there's room left under the cap.
        seen: set[str] = set()
        for token in completed:
            key = self._normalize_skill_token(token)
            if key not in self._iter_skill_commands() or key in seen:
                return
            seen.add(key)
        if len(seen) >= _cap:
            return

        # Only suggest while the user is typing another /token — a bare
        # space after the chain means they may be starting the instruction.
        if not current_word.startswith("/"):
            return

        word_key = self._normalize_skill_token(current_word)
        for cmd, info in self._iter_skill_commands().items():
            if cmd in seen or not cmd.startswith(word_key):
                continue
            # Exact match: append a trailing space so the dropdown stays
            # visible and the next stacked token can be typed immediately
            # (mirrors _completion_text semantics).
            replacement = f"{cmd} " if cmd == word_key else cmd
            yield Completion(
                replacement,
                start_position=-len(current_word),
                display=cmd,
                display_meta=f"⚡ {_short_desc(info, 'Skill command')}",
            )

    # Commands that open pickers when run bare. No trailing space for these:
    # the TUI applies the completion on Enter, and "/model " blocks the picker.
    _PICKER_COMMANDS = frozenset({"model", "skin", "personality"})

    @staticmethod
    def _completion_text(cmd_name: str, word: str) -> str:
        """Return replacement text for a completion.

        On an exact match (``/help`` fully typed) a no-op replacement makes
        prompt_toolkit suppress the menu, so a trailing space is appended to
        keep the dropdown visible — except for _PICKER_COMMANDS.
        """
        if cmd_name != word or cmd_name in SlashCommandCompleter._PICKER_COMMANDS:
            return cmd_name
        return f"{cmd_name} "

    @staticmethod
    def _extract_path_word(text: str) -> str | None:
        """Return the path-like word under the cursor, else None.

        Path-like: starts with ``./``, ``../``, ``~/``, ``/`` or contains a
        ``/``. Tokens with a ``://`` scheme are excluded — treating a pasted
        URL as a path fires os.listdir per keystroke for no useful result.
        """
        # Words are space-delimited, but paths can contain almost anything.
        word = text.rpartition(" ")[2]
        if not word or "://" in word:
            return None
        # Only trigger path completion for path-like tokens
        if word.startswith(("./", "../", "~/", "/")) or "/" in word:
            return word
        return None

    @staticmethod
    def _dir_completions(
        expanded: str,
        word: str,
        limit: int,
        text_for: Callable[[str], str],
        want_dir: bool | None = None,
    ):
        """Yield directory-listing completions for the path *expanded*.

        Entries of the parent dir are matched case-insensitively on the typed
        basename (all entries after a trailing ``/``), sorted by name, and
        limited to *limit*. ``text_for(full_path)`` builds the completion text
        (without the trailing ``/``); *want_dir* restricts to dirs / files.
        """
        if expanded.endswith("/"):
            search_dir, prefix = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)
        try:
            entries = os.listdir(search_dir)
        except OSError:
            return
        prefix_lower = prefix.lower()
        count = 0
        for entry in sorted(entries):
            if prefix and not entry.lower().startswith(prefix_lower):
                continue
            full_path = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full_path)
            if want_dir is not None and want_dir != is_dir:
                continue
            if count >= limit:
                break
            suffix = "/" if is_dir else ""
            yield Completion(
                text_for(full_path) + suffix,
                start_position=-len(word),
                display=entry + suffix,
                display_meta="dir" if is_dir else _file_size_label(full_path),
            )
            count += 1

    @staticmethod
    def _path_completions(word: str, limit: int = 30):
        """Yield Completion objects for file paths matching *word*."""
        # Completion text keeps the user's own path style (~, absolute, relative).
        if word.startswith("~"):
            text_for = lambda fp: "~/" + os.path.relpath(fp, os.path.expanduser("~"))  # noqa: E731
        elif os.path.isabs(word):
            text_for = lambda fp: fp  # noqa: E731
        else:
            text_for = os.path.relpath
        yield from SlashCommandCompleter._dir_completions(
            os.path.expanduser(word), word, limit, text_for
        )

    @staticmethod
    def _extract_context_word(text: str) -> str | None:
        """Extract a bare ``@`` token for context reference completions."""
        word = text.rpartition(" ")[2]
        return word if word.startswith("@") else None

    def _context_completions(self, word: str, limit: int = 30):
        """Yield Claude Code-style @ context completions.

        Bare ``@`` or ``@partial`` shows static references and matching
        files/folders.  ``@file:path`` and ``@folder:path`` are handled
        by the existing path completion path.
        """
        lowered = word.lower()

        # Static context references
        _STATIC_REFS = (
            ("@diff", "Git working tree diff"),
            ("@staged", "Git staged diff"),
            ("@file:", "Attach a file"),
            ("@folder:", "Attach a folder"),
            ("@git:", "Git log with diffs (e.g. @git:5)"),
            ("@url:", "Fetch web content"),
        )
        for candidate, meta in _STATIC_REFS:
            if candidate.lower().startswith(lowered) and candidate.lower() != lowered:
                yield Completion(
                    candidate,
                    start_position=-len(word),
                    display=candidate,
                    display_meta=meta,
                )

        # If the user typed @file: / @folder: (or just @file / @folder with
        # no colon yet), delegate to path completions.  Accepting the bare
        # form lets the picker surface directories as soon as the user has
        # typed `@folder`, without requiring them to first accept the static
        # `@folder:` hint and re-trigger completion.
        for prefix in ("@file:", "@folder:"):
            bare = prefix[:-1]

            if word == bare or word.startswith(prefix):
                path_part = '' if word == bare else word[len(prefix):]
                expanded = os.path.expanduser(path_part)
                if not expanded or expanded == ".":
                    expanded = "./"
                # `@folder:` surfaces only directories, `@file:` only regular
                # files — otherwise `@folder:` lists every dotfile in cwd.
                yield from self._dir_completions(
                    expanded, word, limit,
                    lambda fp: f"{prefix}{os.path.relpath(fp)}",
                    want_dir=(prefix == "@folder:"),
                )
                return

        # Bare @ or @partial — fuzzy project-wide file search
        query = word[1:]  # strip the @
        yield from self._fuzzy_file_completions(word, query, limit)

    def _get_project_files(self) -> list[str]:
        """Return cached list of project files (refreshed every 5s)."""
        cwd = os.getcwd()
        now = time.monotonic()
        if (
            self._file_cache
            and self._file_cache_cwd == cwd
            and now - self._file_cache_time < 5.0
        ):
            return self._file_cache

        files: list[str] = []
        # Try rg first (fast, respects .gitignore), then fd, then find.
        for cmd in [
            ["rg", "--files", "--sortr=modified", cwd],
            ["rg", "--files", cwd],
            ["fd", "--type", "f", "--base-directory", cwd],
        ]:
            tool = cmd[0]
            if not shutil.which(tool):
                continue
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=2,
                    cwd=cwd, encoding="utf-8", errors="replace",
                )
                if proc.returncode == 0 and proc.stdout and proc.stdout.strip():
                    raw = proc.stdout.strip().split("\n")
                    # Store relative paths
                    for p in raw[:5000]:
                        try:
                            rel = os.path.relpath(p, cwd) if os.path.isabs(p) else p
                        except ValueError:
                            # Windows: relpath raises for paths on a different
                            # mount than cwd — device paths (\\.\nul, \\.\con)
                            # or another drive letter. One bad entry must not
                            # crash the @ autocomplete event loop (#42016).
                            continue
                        files.append(rel)
                    break
            except (subprocess.TimeoutExpired, OSError):
                continue

        self._file_cache = files
        self._file_cache_time = now
        self._file_cache_cwd = cwd
        return files

    @staticmethod
    def _score_path(filepath: str, query: str) -> int:
        """Score a file path against a fuzzy query. Higher = better match."""
        if not query:
            return 1  # show everything when query is empty

        filename = os.path.basename(filepath)
        lower_file = filename.lower()
        lower_path = filepath.lower()
        lower_q = query.lower()

        # Exact filename match
        if lower_file == lower_q:
            return 100
        # Filename starts with query
        if lower_file.startswith(lower_q):
            return 80
        # Filename contains query as substring
        if lower_q in lower_file:
            return 60
        # Full path contains query
        if lower_q in lower_path:
            return 40
        # Initials / abbreviation match: e.g. "fo" matches "file_operations"
        # Check if query chars appear in order in filename
        qi = 0
        for c in lower_file:
            if qi < len(lower_q) and c == lower_q[qi]:
                qi += 1
        if qi == len(lower_q):
            # Bonus if matches land on word boundaries (after _, -, /, .)
            boundary_hits = 0
            qi = 0
            prev = "_"  # treat start as boundary
            for c in lower_file:
                if qi < len(lower_q) and c == lower_q[qi]:
                    if prev in "_-./":
                        boundary_hits += 1
                    qi += 1
                prev = c
            if boundary_hits >= len(lower_q) * 0.5:
                return 35
            return 25
        return 0

    def _fuzzy_file_completions(self, word: str, query: str, limit: int = 20):
        """Yield fuzzy file completions for bare @query."""
        files = self._get_project_files()

        if not query:
            # No query — recently modified files (already mtime-sorted).
            ranked = files[:limit]
        else:
            scored = [(s, fp) for fp in files if (s := self._score_path(fp, query)) > 0]
            scored.sort(key=lambda x: (-x[0], x[1]))
            ranked = [fp for _, fp in scored[:limit]]

        for fp in ranked:
            is_dir = fp.endswith("/")
            kind = "folder" if is_dir else "file"
            meta = "dir" if is_dir else _file_size_label(os.path.join(os.getcwd(), fp))
            if query:
                meta = f"{fp}  {meta}" if meta else fp
            yield Completion(
                f"@{kind}:{fp}",
                start_position=-len(word),
                display=os.path.basename(fp),
                display_meta=meta,
            )

    @staticmethod
    def _skin_completions(sub_text: str, sub_lower: str):
        """Yield completions for /skin from available skins."""
        try:
            from hermes_cli.skin_engine import list_skins
            for s in list_skins():
                name = s["name"]
                if name.startswith(sub_lower) and name != sub_lower:
                    yield Completion(
                        name,
                        start_position=-len(sub_text),
                        display=name,
                        display_meta=s.get("description", "") or s.get("source", ""),
                    )
        except Exception:
            pass

    @staticmethod
    def _tools_completions(sub_text: str, sub_lower: str):
        """Yield completions for /tools — subcommand + toolset/MCP-server name.

        Handles both ``/tools <tab>`` (suggesting ``list|disable|enable``) and
        ``/tools enable <tab>`` / ``/tools disable <tab>`` (suggesting toolset
        keys and MCP server prefixes, filtered by current enable state so the
        user only sees actionable options).
        """
        SUBS = ("list", "disable", "enable")
        parts = sub_text.split()
        trailing_space = sub_text.endswith(" ")

        # Subcommand stage: zero words typed, or completing the first word.
        if len(parts) == 0 or (len(parts) == 1 and not trailing_space):
            partial = sub_text if not trailing_space else ""
            for sub in SUBS:
                if sub.startswith(partial.lower()) and sub != partial.lower():
                    yield Completion(sub, start_position=-len(partial), display=sub)
            return

        subcommand = parts[0].lower()
        if subcommand not in ("enable", "disable"):
            return

        partial = "" if trailing_space else parts[-1]
        partial_lower = partial.lower()
        already = set(parts[1:] if trailing_space else parts[1:-1])

        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.tools_config import (
                CONFIGURABLE_TOOLSETS,
                _get_platform_tools,
                _get_plugin_toolset_keys,
            )

            # Readonly loader: this runs per keystroke and never mutates config,
            # so skip the defensive deepcopy of load_config().
            config = load_config_readonly()
            enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
            mcp_servers = config.get("mcp_servers") or {}

            # (candidate, meta, actionable): toolsets are only offered when the
            # subcommand would change their state (enable → off, disable → on);
            # MCP server prefixes are always offered.
            rows = [(k, label, (subcommand == "enable") != (k in enabled)) for k, label, _d in CONFIGURABLE_TOOLSETS]
            rows += [(k, "plugin toolset", (subcommand == "enable") != (k in enabled)) for k in sorted(_get_plugin_toolset_keys())]
            if isinstance(mcp_servers, dict):
                rows += [(f"{srv}:", f"MCP server '{srv}'", True) for srv in sorted(mcp_servers)]
            for key, meta, actionable in rows:
                if actionable and key not in already and key.startswith(partial_lower):
                    yield Completion(key, start_position=-len(partial), display=key, display_meta=meta)
        except Exception:
            return

    @staticmethod
    def _handoff_completions(sub_text: str, sub_lower: str):
        """Yield platform completions for /handoff.

        Offers connected (enabled + configured) gateway platforms. A recorded
        home channel is NOT required to list a platform — it's often learned at
        runtime — so the meta hints whether one is set yet. Completes only the
        first arg (the platform); once one is chosen, stop.
        """
        parts = sub_text.split()
        trailing_space = sub_text.endswith(" ")
        if len(parts) > 1 or (len(parts) == 1 and trailing_space):
            return
        partial = "" if (not parts or trailing_space) else parts[-1]
        partial_lower = partial.lower()
        try:
            from gateway.config import load_gateway_config

            gw = load_gateway_config()
            platforms = gw.get_connected_platforms()
        except Exception:
            return
        for platform in platforms:
            name = platform.value
            if not name.startswith(partial_lower):
                continue
            try:
                home = gw.get_home_channel(platform)
            except Exception:
                home = None
            meta = f"→ {home.name}" if home and getattr(home, "name", None) else "send this session here"
            yield Completion(
                name,
                start_position=-len(partial),
                display=name,
                display_meta=meta,
            )

    @staticmethod
    def _personality_completions(sub_text: str, sub_lower: str):
        """Yield completions for /personality via hermes_cli.personality."""
        try:
            from hermes_cli.personality import describe_personality

            personalities = _personalities_from_cli_config()

            if "none".startswith(sub_lower) and "none" != sub_lower:
                yield Completion(
                    "none",
                    start_position=-len(sub_text),
                    display="none",
                    display_meta="clear personality overlay",
                )
            for name, prompt in personalities.items():
                if name.startswith(sub_lower) and name != sub_lower:
                    yield Completion(
                        name,
                        start_position=-len(sub_text),
                        display=name,
                        display_meta=describe_personality(prompt),
                    )
        except Exception:
            pass

    # base command -> (handler(sub_text, sub_lower), single_word_only).
    # Single-word handlers only run while the first argument is being typed;
    # /tools and /handoff parse multi-word input themselves, bypassing the
    # static SUBCOMMANDS branch.
    _DYNAMIC_COMPLETIONS: dict[str, tuple[Callable[..., Any], bool]] = {
        "/skin": (_skin_completions, True),
        "/personality": (_personality_completions, True),
        "/tools": (_tools_completions, False),
        "/handoff": (_handoff_completions, False),
    }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            # Try @ context completion (Claude Code-style)
            ctx_word = self._extract_context_word(text)
            if ctx_word is not None:
                yield from self._context_completions(ctx_word)
                return
            # Try file path completion for non-slash input
            path_word = self._extract_path_word(text)
            if path_word is not None:
                yield from self._path_completions(path_word)
            return

        # Check if we're completing a subcommand (base command already typed)
        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()
        if len(parts) > 1 or (len(parts) == 1 and text.endswith(" ")):
            sub_text = parts[1] if len(parts) > 1 else ""
            sub_lower = sub_text.lower()

            # Stacked slash-skill chain (`/skill-a /skill-b …`), see
            # split_stacked_skill_commands in agent/skill_commands.py.
            if self._is_skill_command(base_cmd):
                yield from self._stacked_skill_completions(text)
                return

            # Dynamic completions for commands with runtime lists.
            dynamic = self._DYNAMIC_COMPLETIONS.get(base_cmd)
            if dynamic is not None:
                handler, single_word = dynamic
                if not single_word or " " not in sub_text:
                    yield from handler(sub_text, sub_lower)
                    return

            # Static subcommand completions
            if " " not in sub_text and base_cmd in SUBCOMMANDS and self._command_allowed(base_cmd):
                for sub in SUBCOMMANDS[base_cmd]:
                    if sub.startswith(sub_lower) and sub != sub_lower:
                        yield Completion(
                            sub,
                            start_position=-len(sub_text),
                            display=sub,
                        )
            return

        word = text[1:]

        def _cmd_completion(cmd_name: str, meta: str):
            return Completion(
                self._completion_text(cmd_name, word),
                start_position=-len(word),
                display=f"/{cmd_name}",
                display_meta=meta,
            )

        for cmd, desc in COMMANDS.items():
            if not self._command_allowed(cmd):
                continue
            if cmd[1:].startswith(word):
                yield _cmd_completion(cmd[1:], desc)

        for cmd, info in self._iter_skill_bundles().items():
            if cmd[1:].startswith(word):
                skill_count = len(info.get("skills", []))
                yield _cmd_completion(
                    cmd[1:],
                    f"▣ {_short_desc(info, 'Skill bundle')} ({skill_count} skills)",
                )

        for cmd, info in self._iter_skill_commands().items():
            if cmd[1:].startswith(word):
                yield _cmd_completion(cmd[1:], f"⚡ {_short_desc(info, 'Skill command')}")

        # Plugin-registered slash commands
        try:
            from hermes_cli.plugins import get_plugin_commands
            for cmd_name, cmd_info in get_plugin_commands().items():
                if cmd_name.startswith(word):
                    yield _cmd_completion(cmd_name, f"🔌 {_short_desc(cmd_info, 'Plugin command')}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Inline auto-suggest (ghost text) for slash commands
# ---------------------------------------------------------------------------

class SlashCommandAutoSuggest(AutoSuggest):
    """Inline ghost-text suggestions for slash commands and their subcommands.

    Shows the rest of a command or subcommand in dim text as you type.
    Falls back to history-based suggestions for non-slash input.
    """

    def __init__(
        self,
        history_suggest: AutoSuggest | None = None,
        completer: SlashCommandCompleter | None = None,
    ) -> None:
        self._history = history_suggest
        self._completer = completer  # Reuse its model cache

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return self._history.get_suggestion(buffer, document) if self._history else None

        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()

        if len(parts) == 1 and not text.endswith(" "):
            # Still typing the command name: /upd → "ate". Prefer the SHORTEST
            # match so /he ghosts "lp" (/help), not "artbeat" (/heartbeat).
            word = text[1:].lower()
            for cmd in sorted(COMMANDS, key=len):
                if self._completer is not None and not self._completer._command_allowed(cmd):
                    continue
                cmd_name = cmd[1:]  # strip leading /
                if cmd_name.startswith(word) and cmd_name != word:
                    return Suggestion(cmd_name[len(word):])
            return None

        # Command is complete — suggest subcommands
        sub_text = parts[1] if len(parts) > 1 else ""
        sub_lower = sub_text.lower()

        # Stacked skill chain: ghost-suggest the rest of the next skill name;
        # otherwise fall through to the history fallback for instruction text.
        if self._completer is not None and self._completer._is_skill_command(base_cmd):
            for completion in self._completer._stacked_skill_completions(text):
                remainder = completion.text[-completion.start_position:] \
                    if completion.start_position else completion.text
                if remainder.strip():
                    return Suggestion(remainder)

        # Static subcommands
        if self._completer is not None and not self._completer._command_allowed(base_cmd):
            return None
        if " " not in sub_text:
            for sub in SUBCOMMANDS.get(base_cmd, ()):
                if sub.startswith(sub_lower) and sub != sub_lower:
                    return Suggestion(sub[len(sub_text):])

        return self._history.get_suggestion(buffer, document) if self._history else None


def _short_desc(info: Mapping[str, Any], default: str) -> str:
    """50-char description preview used in completion menus."""
    description = str(info.get("description", default))
    return description[:50] + ("..." if len(description) > 50 else "")


def _file_size_label(path: str) -> str:
    """Return a compact human-readable file size, or '' on error."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f}K"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}M"
    return f"{size / (1024 * 1024 * 1024):.1f}G"
