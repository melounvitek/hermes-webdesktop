#!/usr/bin/env python3
"""
Hermes Agent CLI - Interactive Terminal Interface

A beautiful command-line interface for the Hermes Agent, inspired by Claude Code.
Features ASCII art branding, interactive REPL, toolset selection, and rich formatting.

Usage:
    python cli.py                          # Start interactive mode with all tools
    python cli.py --toolsets web,terminal  # Start with specific toolsets
    python cli.py --skills hermes-agent-dev,github-auth
    python cli.py --list-tools             # List available tools and exit
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass

import logging
import os
import functools
import shutil
import sys
import json
import re
import concurrent.futures
import atexit
import errno
import time
import uuid
import textwrap
from collections import deque
from dataclasses import dataclass
from urllib.parse import unquote, urlparse
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Mapping

logger = logging.getLogger(__name__)

# Suppress startup messages for clean CLI experience
os.environ["HERMES_QUIET"] = "1"  # Our own modules

from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_billing_mixin import CLIBillingMixin
from hermes_cli.cli_loops_mixin import CLILoopsMixin
from hermes_cli.cli_info_mixin import CLIInfoMixin
from hermes_cli.cli_terminal_mixin import CLITerminalMixin
from hermes_cli.cli_modal_mixin import CLIModalMixin
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from hermes_cli.cli_voice_mixin import CLIVoiceMixin
from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from agent.interrupt_compat import request_hard_interrupt
from agent.pet import render as pet_render

# prompt_toolkit for fixed input area TUI
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import Application
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
try:
    from prompt_toolkit.cursor_shapes import CursorShape
    _STEADY_CURSOR = CursorShape.BLOCK  # Non-blinking block cursor
except (ImportError, AttributeError):
    _STEADY_CURSOR = None

try:
    from hermes_cli.pt_input_extras import (
        install_cmd_backspace_alias,
        install_ctrl_enter_alias,
        install_ignored_terminal_sequences,
        install_keypress_data_normalization,
        install_modify_other_keys_aliases,
        install_shift_enter_alias,
    )
    install_shift_enter_alias()
    install_ctrl_enter_alias()
    install_cmd_backspace_alias()
    install_modify_other_keys_aliases()
    install_keypress_data_normalization()
    install_ignored_terminal_sequences()
    del install_shift_enter_alias, install_ctrl_enter_alias, install_cmd_backspace_alias, install_modify_other_keys_aliases, install_keypress_data_normalization, install_ignored_terminal_sequences
except Exception:
    pass
import threading
import queue

def CanonicalUsage(*args, **kwargs):
    from agent.usage_pricing import CanonicalUsage as _CanonicalUsage

    return _CanonicalUsage(*args, **kwargs)


def estimate_usage_cost(*args, **kwargs):
    from agent.usage_pricing import estimate_usage_cost as _estimate_usage_cost

    return _estimate_usage_cost(*args, **kwargs)


def format_duration_compact(*args, **kwargs):
    seconds = float(args[0] if args else kwargs.get("seconds", 0.0))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    days = hours / 24
    return f"{days:.1f}d"


# Cached reverse map of config.yaml ``model_aliases:`` so the TUI can show
# friendly names instead of full Palantir RIDs / long catalog IDs. Built
# lazily on first call; cache is process-lifetime (config is read once at
# session start, so further invalidation is unnecessary).
_REVERSE_ALIAS_CACHE: dict[str, str] | None = None


def _reverse_alias_for_display(model_name: str) -> str:
    """Return the shortest configured alias for ``model_name``, or ``model_name``.

    Looks up both ``model_aliases:`` (dict-based, full DirectAlias entries)
    and ``model.aliases:`` (string-based, set via ``hermes config set``)
    from config.yaml. Multiple aliases pointing at the same model — the
    shortest wins, so ``opus47`` beats ``palantir-claude47``.
    """
    global _REVERSE_ALIAS_CACHE
    if not model_name:
        return model_name
    if _REVERSE_ALIAS_CACHE is None:
        rmap: dict[str, str] = {}
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            ma = cfg.get("model_aliases")
            if isinstance(ma, dict):
                for alias, entry in ma.items():
                    if isinstance(entry, dict):
                        m = str(entry.get("model", "") or "").strip()
                        if m and (m not in rmap or len(alias) < len(rmap[m])):
                            rmap[m] = alias
            mdl = cfg.get("model", {}) or {}
            if isinstance(mdl, dict):
                simple = mdl.get("aliases")
                if isinstance(simple, dict):
                    for alias, val in simple.items():
                        if isinstance(val, str) and val.strip():
                            v = val.strip()
                            m = v.split("/", 1)[1] if "/" in v else v
                            if m and (m not in rmap or len(alias) < len(rmap[m])):
                                rmap[m] = alias
        except Exception:
            pass
        _REVERSE_ALIAS_CACHE = rmap
    return _REVERSE_ALIAS_CACHE.get(model_name, model_name)


def format_token_count_compact(*args, **kwargs):
    value = int(args[0] if args else kwargs.get("value", 0))
    abs_value = abs(value)
    if abs_value < 1_000:
        return str(value)

    sign = "-" if value < 0 else ""
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = abs_value / threshold
            if scaled < 10:
                text = f"{scaled:.2f}"
            elif scaled < 100:
                text = f"{scaled:.1f}"
            else:
                text = f"{scaled:.0f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"

    return f"{value:,}"


def is_table_divider(*args, **kwargs):
    from agent.markdown_tables import is_table_divider as _is_table_divider

    return _is_table_divider(*args, **kwargs)


def looks_like_table_row(*args, **kwargs):
    from agent.markdown_tables import looks_like_table_row as _looks_like_table_row

    return _looks_like_table_row(*args, **kwargs)


def realign_markdown_tables(*args, **kwargs):
    from agent.markdown_tables import realign_markdown_tables as _realign_markdown_tables

    return _realign_markdown_tables(*args, **kwargs)
# NOTE: `from agent.account_usage import ...` is deliberately NOT at module
# top — it transitively pulls the OpenAI SDK chain (~230 ms cold) and is only
# needed when the user runs `/limits`. Lazy-imported inside the handler below.
from hermes_cli.banner import format_banner_version_label

_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_constants import get_hermes_home, display_hermes_home  # noqa: F401  (mixins import via cli)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import base_url_host_matches, base_url_hostname, fast_safe_load

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)


_REASONING_TAGS = (
    "REASONING_SCRATCHPAD",
    "think",
    "thinking",
    "reasoning",
    "thought",
)


def _strip_reasoning_tags(text: str) -> str:
    """Remove reasoning/thinking blocks from displayed text.

    Handles every case:
      * Closed pairs ``<tag>…</tag>`` (case-insensitive, multi-line).
      * Unterminated open tags that run to end-of-text (e.g. truncated
        generations on NIM/MiniMax where the close tag is dropped).
      * Stray orphan close tags (``stuff</think>answer``) left behind by
        partial-content dumps.

    Covers the variants emitted by reasoning models today: ``<think>``,
    ``<thinking>``, ``<reasoning>``, ``<REASONING_SCRATCHPAD>``, and
    ``<thought>`` (Gemma 4).  Must stay in sync with
    ``run_agent.py::_strip_think_blocks`` and the stream consumer's
    ``_OPEN_THINK_TAGS`` / ``_CLOSE_THINK_TAGS`` tuples.

    Also strips tool-call XML blocks some open models leak into visible
    content (``<tool_call>``, ``<function_calls>``, Gemma-style
    ``<function name="…">…</function>``). Ported from
    openclaw/openclaw#67318.
    """
    cleaned = text
    for tag in _REASONING_TAGS:
        # Closed pair — case-insensitive so <THINK>…</THINK> is handled too.
        cleaned = re.sub(
            rf"<{tag}>.*?</{tag}>\s*",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Unterminated open tag — strip from the tag to end of text.
        cleaned = re.sub(
            rf"<{tag}>.*$",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Stray orphan close tag left behind by partial dumps.
        cleaned = re.sub(
            rf"</{tag}>\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    # Tool-call XML blocks (openclaw/openclaw#67318).
    for tc_tag in ("tool_call", "tool_calls", "tool_result",
                   "function_call", "function_calls"):
        cleaned = re.sub(
            rf"<{tc_tag}\b[^>]*>.*?</{tc_tag}>\s*",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
    # <function name="..."> — boundary + attribute gated to avoid prose FPs.
    cleaned = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
        r'<function\b[^>]*\bname\s*=[^>]*>'
        r'(?:(?:(?!</function>).)*)</function>\s*',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Stray tool-call close tags.
    cleaned = re.sub(
        r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
        '',
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _assistant_content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _assistant_copy_text(content: Any) -> str:
    return _strip_reasoning_tags(_assistant_content_as_text(content))


# =============================================================================
# Configuration Loading
# =============================================================================

def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from a JSON file.
    
    The file should contain a JSON array of {role, content} dicts, e.g.:
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    
    Relative paths are resolved from ~/.hermes/.
    Returns an empty list if the path is empty or the file doesn't exist.
    """
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = _hermes_home / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []


def _resolve_prefill_messages_file(config: Dict[str, Any]) -> str:
    """Resolve the prefill file path from env/config.

    ``prefill_messages_file`` at the top level is the canonical config key.
    ``agent.prefill_messages_file`` remains a legacy fallback for older CLI and
    godmode-generated configs.
    """
    env_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "").strip()
    if env_path:
        return env_path
    top_level = str(config.get("prefill_messages_file", "") or "").strip()
    if top_level:
        return top_level
    agent_cfg = config.get("agent", {})
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("prefill_messages_file", "") or "").strip()
    return ""


def _parse_reasoning_config(effort) -> dict | None:
    """Parse a reasoning effort level into an OpenRouter reasoning config dict.

    Accepts the raw config value (string or YAML boolean — ``false``/``off``
    parse as thinking disabled, see parse_reasoning_effort).
    """
    from hermes_constants import parse_reasoning_effort
    result = parse_reasoning_effort(effort)
    if effort and str(effort).strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def _parse_service_tier_config(raw: str) -> str | None:
    """Parse a persisted fast-mode preference: None, "priority", "auto", or "cold"."""
    value = str(raw or "").strip().lower()
    if not value or value in {"normal", "default", "standard", "off", "none"}:
        return None
    if value in {"fast", "priority", "on"}:
        return "priority"
    if value in {"auto", "cold"}:
        return value
    logger.warning("Unknown service_tier '%s', ignoring", raw)
    return None

def _mirror_config_to_env(defaults, _file_has_terminal_config):
    """Project config.yaml values into the env vars the tool modules read (terminal/browser/auxiliary/security/sessions). Env always wins when already set."""
    # Apply terminal config to environment variables (so terminal_tool picks them up)
    terminal_config = defaults.get("terminal", {})

    # Normalize config key: the new config system (hermes_cli/config.py) and all
    # documentation use "backend", the legacy cli-config.yaml uses "env_type".
    # Accept both, with "backend" taking precedence (it's the documented key).
    if "backend" in terminal_config:
        terminal_config["env_type"] = terminal_config["backend"]

    # CWD resolution for CLI/TUI. The gateway has its own config bridge in
    # gateway/run.py but may lazily import cli.py (triggering this code).
    # Local backend: always os.getcwd(). Use `cd /dir && hermes` to control it.
    # Non-local with placeholder: pop so terminal_tool uses its per-backend default.
    # Non-local with explicit path: keep as-is.
    _CWD_PLACEHOLDERS = (".", "auto", "cwd")
    effective_backend = terminal_config.get("env_type", "local")

    if effective_backend == "local":
        terminal_config["cwd"] = os.getcwd()
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        terminal_config.pop("cwd", None)

    env_mappings = {
        "env_type": "TERMINAL_ENV",
        "degraded_mode": "TERMINAL_DEGRADED_MODE",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "home_mode": "TERMINAL_HOME_MODE",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        "daytona_image": "TERMINAL_DAYTONA_IMAGE",
        "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
        # SSH config
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        # Container resource config (docker, singularity, modal, daytona, vercel_sandbox -- ignored for local/ssh)
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
        "docker_env": "TERMINAL_DOCKER_ENV",
        "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
        "docker_shm_size": "TERMINAL_DOCKER_SHM_SIZE",
        "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "docker_network": "TERMINAL_DOCKER_NETWORK",
        "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
        "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
        "docker_shared_container_key": "TERMINAL_DOCKER_SHARED_CONTAINER_KEY",
        "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        # Persistent shell (non-local backends)
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
        # Sudo support (works with all backends)
        "sudo_password": "SUDO_PASSWORD",
    }

    # Bridge config → env vars for terminal_tool. TERMINAL_CWD is force-exported
    # UNLESS we're inside a gateway process (detected by _HERMES_GATEWAY marker)
    # where it was already set correctly by gateway/run.py's config bridge.
    _is_gateway = os.environ.get("_HERMES_GATEWAY") == "1"
    for config_key, env_var in env_mappings.items():
        if config_key in terminal_config:
            if env_var == "TERMINAL_CWD":
                if _is_gateway:
                    continue
                # CLI: always export (overrides stale .env or inherited values)
                os.environ[env_var] = str(terminal_config[config_key])
                continue
            if _file_has_terminal_config or env_var not in os.environ:
                val = terminal_config[config_key]
                if isinstance(val, (list, dict)):
                    os.environ[env_var] = json.dumps(val)
                else:
                    os.environ[env_var] = str(val)

    # Apply browser config to environment variables
    browser_config = defaults.get("browser", {})
    browser_env_mappings = {
        "inactivity_timeout": "BROWSER_INACTIVITY_TIMEOUT",
    }

    for config_key, env_var in browser_env_mappings.items():
        if config_key in browser_config:
            os.environ[env_var] = str(browser_config[config_key])

    # Apply auxiliary model/direct-endpoint overrides to environment variables.
    # Vision and web_extract each have their own provider/model/base_url/api_key tuple.
    # Compression config is read directly from config.yaml by run_agent.py and
    # auxiliary_client.py — no env var bridging needed.
    # Only set env vars for non-empty / non-default values so auto-detection
    # still works.
    auxiliary_config = defaults.get("auxiliary", {})
    auxiliary_task_env = {
        # config key → env var mapping
        "vision": {
            "provider": "AUXILIARY_VISION_PROVIDER",
            "model": "AUXILIARY_VISION_MODEL",
            "base_url": "AUXILIARY_VISION_BASE_URL",
            "api_key": "AUXILIARY_VISION_API_KEY",
        },
        "approval": {
            "provider": "AUXILIARY_APPROVAL_PROVIDER",
            "model": "AUXILIARY_APPROVAL_MODEL",
            "base_url": "AUXILIARY_APPROVAL_BASE_URL",
            "api_key": "AUXILIARY_APPROVAL_API_KEY",
        },
    }

    for task_key, env_map in auxiliary_task_env.items():
        task_cfg = auxiliary_config.get(task_key, {})
        if not isinstance(task_cfg, dict):
            continue
        prov = str(task_cfg.get("provider", "")).strip()
        model = str(task_cfg.get("model", "")).strip()
        base_url = str(task_cfg.get("base_url", "")).strip()
        api_key = str(task_cfg.get("api_key", "")).strip()
        if prov and prov != "auto":
            os.environ[env_map["provider"]] = prov
        if model:
            os.environ[env_map["model"]] = model
        if base_url:
            os.environ[env_map["base_url"]] = base_url
        if api_key:
            os.environ[env_map["api_key"]] = api_key

    # Security settings
    security_config = defaults.get("security", {})
    if isinstance(security_config, dict):
        redact = security_config.get("redact_secrets")
        if redact is not None:
            os.environ["HERMES_REDACT_SECRETS"] = str(redact).lower()

    # Session-search index knobs (hermes_state reads the env carriers).
    sessions_config = defaults.get("sessions", {})
    if isinstance(sessions_config, dict):
        if "cjk_fts" in sessions_config:
            os.environ["HERMES_CJK_FTS"] = str(sessions_config["cjk_fts"])
        if "search_slow_ms" in sessions_config:
            os.environ["HERMES_SEARCH_SLOW_MS"] = str(
                sessions_config["search_slow_ms"]
            )


def _cli_config_defaults():
    """Built-in defaults for every config key the CLI reads (the file overlays these)."""
    # Default configuration
    defaults = {
        "model": {
            "default": "",
            "base_url": "",
            "provider": "auto",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "home_mode": "auto",
            "lifetime_seconds": 300,
            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_forward_env": [],
            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_volumes": [],  # host:container volume mounts for Docker backend
            "docker_mount_cwd_to_workspace": False,  # explicit opt-in only; default off for sandbox isolation
            "docker_shared_container_key": "",
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
            "record_sessions": False,  # Auto-record browser sessions as WebM videos
            "engine": "auto",  # Browser engine: auto (Chrome), lightpanda, chrome
            "camofox": {
                "rewrite_loopback_urls": False,
                "loopback_host_alias": "host.docker.internal",
            },
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.50,    # Compress at 50% of model's context limit
            "min_tail_user_messages": 1,  # Real user messages guaranteed in the tail (1 = existing single anchor)
        },
        "agent": {
            "max_turns": 500,  # Default max tool-calling iterations (shared with subagents)
            "verbose": False,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "service_tier": "",
            # Built-in personalities live in hermes_cli.personality
            # (BUILTIN_PERSONALITIES) — the single owner. Entries here are
            # user-defined additions/overrides merged on top by name.
            "personalities": {},
        },

        "display": {
            "compact": False,
            "resume_display": "full",
            # Recap tuning for /resume — see hermes_cli/config.py DEFAULT_CONFIG.
            "resume_exchanges": 10,
            "resume_max_user_chars": 300,
            "resume_max_assistant_chars": 200,
            "resume_max_assistant_lines": 3,
            "resume_skip_tool_only": True,
            # Live reasoning display default ON — keep in sync with
            # hermes_cli/config.py DEFAULT_CONFIG (display.show_reasoning).
            "show_reasoning": True,
            "reasoning_full": False,
            "streaming": True,
            "busy_input_mode": "interrupt",
            "persistent_output": True,
            "persistent_output_max_lines": 200,
            # Clear terminal scrollback as well as the visible viewport when the
            # classic CLI performs a full redraw/resize recovery. Disabled by
            # default because some users prefer preserving terminal history;
            # enable when a terminal/tmux stack stamps stale prompt chrome into
            # scrollback during fullscreen/restore resizes.
            "cli_rebuild_scrollback_on_redraw": False,
            # Print a one-line summary of resolved modal prompts (approval /
            # clarify) into scrollback so the decision survives the repaint.
            "persist_prompts": True,

            "skin": "default",
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "auxiliary": {
            "vision": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "model": "",       # Subagent model override (empty = inherit parent model)
            "provider": "",    # Subagent provider override (empty = inherit parent provider)
            "base_url": "",    # Direct OpenAI-compatible endpoint for subagents
            "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        },
        "onboarding": {
            # First-touch hint flags (see agent/onboarding.py).  Each hint is
            # shown once per install then latched here.
            "seen": {},
        },
    }
    return defaults


def load_cli_config() -> Dict[str, Any]:
    """
    Load CLI configuration from config files.
    
    Config lookup order:
    1. ~/.hermes/config.yaml (user config - preferred)
    2. ./cli-config.yaml (project config - fallback)
    
    Environment variables take precedence over config file values.
    Returns default values if no config file exists.

    If HERMES_IGNORE_USER_CONFIG=1 is set (via ``hermes chat --ignore-user-config``),
    the user config at ``~/.hermes/config.yaml`` is skipped entirely and only the
    built-in defaults plus the project-level ``cli-config.yaml`` (if any) are used.
    Credentials in ``.env`` are still loaded — this flag only suppresses
    behavioral/config settings.
    """
    # Check user config first ({HERMES_HOME}/config.yaml)
    user_config_path = _hermes_home / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'

    # --ignore-user-config: force-skip the user config.yaml (still honor project
    # config as a fallback so defaults stay sensible).
    ignore_user_config = os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1"

    # Use user config if it exists, otherwise project config
    if user_config_path.exists() and not ignore_user_config:
        config_path = user_config_path
    else:
        config_path = project_config_path

    defaults = _cli_config_defaults()
    
    # Track whether the config file explicitly set terminal config.
    # When using defaults (no config file / no terminal section), we should NOT
    # overwrite env vars that were already set by .env -- only a user's config
    # file should be authoritative.
    _file_has_terminal_config = False

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                from hermes_cli.config import _normalize_root_model_keys

                file_config = _normalize_root_model_keys(fast_safe_load(f) or {})
            
            _file_has_terminal_config = "terminal" in file_config

            # Handle model config - can be string (new format) or dict (old format)
            if "model" in file_config:
                if isinstance(file_config["model"], str):
                    # New format: model is just a string, convert to dict structure
                    defaults["model"]["default"] = file_config["model"]
                elif isinstance(file_config["model"], dict):
                    # Old format: model is a dict with default/base_url
                    defaults["model"].update(file_config["model"])
                    # If the user config sets model.model but not model.default,
                    # promote model.model to model.default so the user's explicit
                    # choice isn't shadowed by the hardcoded default.  Without this,
                    # profile configs that only set "model:" (not "default:") silently
                    # fall back to claude-opus because the merge preserves the
                    # hardcoded default and HermesCLI.__init__ checks "default" first.
                    if "model" in file_config["model"] and "default" not in file_config["model"]:
                        defaults["model"]["default"] = file_config["model"]["model"]

            # Deep merge file_config into defaults.
            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and file_config[key] is None:
                        continue
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
            
            # Second: carry over keys from file_config that aren't in defaults
            # (e.g. platform_toolsets, provider_routing, memory, honcho, etc.)
            for key in file_config:
                if key not in defaults and key != "model":
                    defaults[key] = file_config[key]
            
            # Handle legacy root-level max_turns (backwards compat) - copy to
            # agent.max_turns whenever the nested key is missing.
            agent_file_config = file_config.get("agent")
            if "max_turns" in file_config and not (
                isinstance(agent_file_config, dict)
                and agent_file_config.get("max_turns") is not None
            ):
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)

    # Expand ${ENV_VAR} references in config values before bridging to env vars.
    from hermes_cli.config import _expand_env_vars
    defaults = _expand_env_vars(defaults)

    # Managed scope: overlay administrator-pinned values LAST so they win over
    # the user's config here too. cli.py builds its config independently of
    # hermes_cli.config._load_config_impl (which has its own managed merge), so
    # without this the entire interactive CLI/TUI surface — skin, display prefs,
    # etc. read from CLI_CONFIG — would silently ignore managed scope while
    # `hermes config`/`doctor`/guards (which use load_config) honor it. The
    # shared helper mirrors _load_config_impl (env-only expansion, root-model
    # normalization, leaf-merge) and is fail-open.
    from hermes_cli import managed_scope

    defaults = managed_scope.apply_managed_overlay(defaults)

    _mirror_config_to_env(defaults, _file_has_terminal_config)

    return defaults

# Load configuration at module startup
CLI_CONFIG = load_cli_config()


# Initialize centralized logging early — agent.log + errors.log in ~/.hermes/logs/.
# This ensures CLI sessions produce a log trail even before AIAgent is instantiated.
try:
    from hermes_logging import setup_logging
    setup_logging(mode="cli")
except Exception:
    pass  # Logging setup is best-effort — don't crash the CLI

# Validate config structure early — print warnings before user hits cryptic errors
try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception:
    pass

# Initialize the skin engine from config
try:
    from hermes_cli.skin_engine import init_skin_from_config
    init_skin_from_config(CLI_CONFIG)
except Exception:
    pass  # Skin engine is optional — default skin used if unavailable

# Initialize tool preview length from config
try:
    from agent.display import set_tool_preview_max_len
    _tpl = CLI_CONFIG.get("display", {}).get("tool_preview_length", 0)
    set_tool_preview_max_len(int(_tpl) if _tpl else 0)
except Exception:
    pass

# Initialize friendly tool labels from config (default on)
try:
    from agent.display import set_friendly_tool_labels
    _ftl = CLI_CONFIG.get("display", {}).get("friendly_tool_labels", True)
    set_friendly_tool_labels(bool(_ftl))
except Exception:
    pass

# Neuter AsyncHttpxClientWrapper.__del__ before any AsyncOpenAI clients are
# created.  The SDK's __del__ schedules aclose() on asyncio.get_running_loop()
# which, during CLI idle time, finds prompt_toolkit's event loop and tries to
# close TCP transports bound to dead worker loops — producing
# "Event loop is closed" / "Press ENTER to continue..." errors.
#
# We install a sys.meta_path finder that defers the actual import + patch
# until ``openai._base_client`` is first loaded by the rest of the codebase.
# Eagerly importing it here (the old approach) cost ~166ms / ~30MB on every
# cold CLI start because openai's type tree (responses/*, graders/*) is huge.
# The finder approach pays nothing until the SDK is genuinely needed and
# still guarantees the patch is applied before any AsyncOpenAI instance can
# be constructed (the import-then-instantiate ordering is enforced by
# Python's import system).
try:
    import sys as _httpx_neuter_sys
    import importlib.util as _httpx_neuter_imp_util

    class _AsyncHttpxDelNeuter:
        """Defer ``AsyncHttpxClientWrapper.__del__`` neutering until import.

        Saves ~166ms on cold CLI start where openai is never used (e.g.
        ``hermes --help`` paths inside the chat command flow).  See
        ``agent.auxiliary_client.neuter_async_httpx_del`` for full rationale
        on why ``__del__`` must be a no-op.
        """

        _armed = True

        def find_spec(self, fullname, path=None, target=None):
            if not self._armed or fullname != "openai._base_client":
                return None
            # Disarm before delegating so the recursive find_spec call
            # below doesn't loop through us.
            self._armed = False
            try:
                _httpx_neuter_sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = _httpx_neuter_imp_util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def _patched_exec(module):
                _orig_exec(module)
                try:
                    cls = getattr(module, "AsyncHttpxClientWrapper", None)
                    if cls is not None:
                        cls.__del__ = lambda self: None  # type: ignore[assignment]
                except Exception:
                    pass

            spec.loader.exec_module = _patched_exec  # type: ignore[method-assign]
            return spec

    _httpx_neuter_sys.meta_path.insert(0, _AsyncHttpxDelNeuter())
except Exception:
    pass

from rich import box as rich_box
from rich.console import Console
from rich.markup import escape as _escape
from rich.panel import Panel
from rich.text import Text as _RichText

# Import agent and tool systems lazily. Bare interactive startup only needs the
# prompt; the full agent/tool registry is initialized on first use.
def AIAgent(*args, **kwargs):
    from run_agent import AIAgent as _AIAgent

    return _AIAgent(*args, **kwargs)


def get_tool_definitions(*args, **kwargs):
    from hermes_cli.mcp_startup import wait_for_mcp_discovery
    from model_tools import get_tool_definitions as _get_tool_definitions

    wait_for_mcp_discovery()
    return _get_tool_definitions(*args, **kwargs)


def get_toolset_for_tool(*args, **kwargs):
    from model_tools import get_toolset_for_tool as _get_toolset_for_tool

    return _get_toolset_for_tool(*args, **kwargs)

from hermes_cli.banner import build_welcome_banner  # noqa: F401  (CLIInfoMixin imports via cli)


def get_all_toolsets(*args, **kwargs):
    from toolsets import get_all_toolsets as _get_all_toolsets

    return _get_all_toolsets(*args, **kwargs)


def get_toolset_info(*args, **kwargs):
    from toolsets import get_toolset_info as _get_toolset_info

    return _get_toolset_info(*args, **kwargs)


def validate_toolset(*args, **kwargs):
    from toolsets import validate_toolset as _validate_toolset

    return _validate_toolset(*args, **kwargs)


def _sync_process_session_id(session_id: str) -> None:
    """Keep process-local session-id consumers aligned after CLI switches."""
    from gateway.session_context import set_current_session_id

    set_current_session_id(session_id)

# Cron job system for scheduled tasks (execution is handled by the gateway)
def get_job(*args, **kwargs):
    from cron import get_job as _get_job

    return _get_job(*args, **kwargs)



def _cleanup_all_terminals(*args, **kwargs):
    from tools.terminal_tool import cleanup_all_environments

    return cleanup_all_environments(*args, **kwargs)


def set_sudo_password_callback(*args, **kwargs):
    from tools.terminal_tool import set_sudo_password_callback as _set_sudo_password_callback

    return _set_sudo_password_callback(*args, **kwargs)


def set_approval_callback(*args, **kwargs):
    from tools.terminal_tool import set_approval_callback as _set_approval_callback

    return _set_approval_callback(*args, **kwargs)


def set_secret_capture_callback(*args, **kwargs):
    from tools.skills_tool import set_secret_capture_callback as _set_secret_capture_callback

    return _set_secret_capture_callback(*args, **kwargs)


def _cleanup_all_browsers(*args, **kwargs):
    from tools.browser_tool import _emergency_cleanup_all_sessions

    return _emergency_cleanup_all_sessions(*args, **kwargs)

# Guard to prevent cleanup from running multiple times on exit
_cleanup_done = False
_cleanup_in_progress = False
_cli_wake_owner = None
# One-shot CLI finalization runs before process cleanup so plugins can observe
# the session boundary while the agent is still attached. If a signal lands in
# that narrow window, atexit cleanup must not emit that session finalization again.
_single_query_finalize_attempted_session_ids: set[str | None] = set()
# Session IDs that were handed off to the gateway via /handoff.  The CLI
# process exits after a successful handoff, but the gateway now owns the
# session lifecycle — _run_cleanup must NOT call finalize_session on these,
# because doing so sets end_reason on a row the gateway just reopened and is
# actively writing to (#88234).  The race made the handoff leg vanish from
# session history and broke session_search recall for the handed-off session.
_handed_off_session_ids: set[str | None] = set()
# Weak reference to the active AIAgent for memory provider shutdown at exit
_active_agent_ref = None
_deferred_agent_startup_done = False
# Set True once the TUI's prompt_toolkit app starts (which enables focus
# reporting + mouse tracking). Gates the on-exit terminal reset so non-TUI
# one-shot CLI runs — which also register _run_cleanup via atexit — don't emit
# escape codes for modes they never enabled (#36823).
_tui_input_modes_active = False


def _mark_tui_input_modes_active() -> None:
    """Record that the TUI app started, so _run_cleanup resets input modes."""
    global _tui_input_modes_active
    _tui_input_modes_active = True


def _prepare_deferred_agent_startup() -> None:
    """Run Termux-deferred agent discovery before the first real agent turn."""
    global _deferred_agent_startup_done
    if _deferred_agent_startup_done:
        return
    if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
        return
    _deferred_agent_startup_done = True
    _accept_hooks = os.environ.get("HERMES_ACCEPT_HOOKS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception:
        logger.warning(
            "plugin discovery failed at deferred CLI startup",
            exc_info=True,
        )
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(
            logger=logger,
            thread_name="termux-cli-mcp-discovery",
        )
    except Exception:
        logger.debug(
            "MCP tool discovery failed at deferred CLI startup",
            exc_info=True,
        )
    try:
        from agent.shell_hooks import register_from_config
        from hermes_cli.config import load_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)

        from agent.outbound_webhooks import (
            register_from_config as register_outbound_webhooks,
        )

        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug(
            "shell-hook registration failed at deferred CLI startup",
            exc_info=True,
        )

def _arm_exit_watchdog(timeout_s: float | None = None, *, from_signal: bool = False) -> None:
    """Guarantee the process actually exits once shutdown has begun.

    Two hang classes have kept "dead" CLI processes alive for minutes:

      1. A cleanup step wedged on network I/O (memory provider
         ``on_session_end``, MCP teardown, remote terminal cleanup).
      2. Interpreter teardown blocked joining non-daemon threads —
         stdlib ``ThreadPoolExecutor`` workers are joined unconditionally
         by ``concurrent.futures``' atexit hook even after
         ``shutdown(wait=False)``, so one tool thread wedged on a socket
         held the process open forever (#27563 class).

    The shared daemon pool (``tools.daemon_pool``) removes the main cause
    of (2); this watchdog is the backstop for both. It arms a daemon
    timer when ``_run_cleanup`` starts; if the process is still alive
    after ``timeout_s`` it flushes logging/stdio and calls ``os._exit(0)``.
    Daemon threads keep running through ``Py_FinalizeEx``'s thread joins,
    so the timer fires even when the main thread is stuck in teardown.

    Tune with ``HERMES_EXIT_WATCHDOG_S`` (seconds); ``0`` disables.
    """
    if timeout_s is None:
        try:
            timeout_s = float(os.getenv("HERMES_EXIT_WATCHDOG_S", "30"))
        except (TypeError, ValueError):
            timeout_s = 30.0
    if timeout_s <= 0:
        return
    # Never arm under pytest: tests invoke _run_cleanup() directly and a
    # 30s-delayed os._exit(0) would silently kill the test worker.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    def _watchdog():
        time.sleep(timeout_s)
        # If this is the outer, signal-armed watchdog and cleanup is already in
        # progress, let the cleanup-owned timer enforce shutdown for the current
        # cycle. The signal timer is a broader backstop when graceful unwind
        # never starts.
        if from_signal and _cleanup_in_progress:
            return

        # Still alive — cleanup or interpreter teardown is wedged.
        try:
            logger.warning(
                "Exit watchdog fired after %.0fs — forcing process exit "
                "(a cleanup step or non-daemon thread is wedged).",
                timeout_s,
            )
        except Exception:
            pass
        try:
            import logging as _lg
            _lg.shutdown()
        except Exception:
            pass
        for _stream in (sys.stdout, sys.stderr):
            try:
                _stream.flush()
            except Exception:
                pass
        os._exit(0)

    try:
        threading.Thread(
            target=_watchdog, daemon=True, name="exit-watchdog"
        ).start()
    except Exception:
        pass  # best-effort — never block shutdown on watchdog setup


_signal_watchdog_armed = False


def _arm_exit_watchdog_on_shutdown_signal() -> None:
    """Arm the exit backstop the moment a termination signal arrives.

    SIGTERM/SIGHUP establish unambiguous shutdown intent, but the graceful
    path from signal → ``agent.interrupt()`` → ``app.exit()`` /
    ``KeyboardInterrupt`` → ``finally`` → ``_run_cleanup`` has several wedge
    points BEFORE ``_run_cleanup`` arms the normal watchdog: a main thread
    parked in a syscall that never observes the unwind, a prompt_toolkit
    teardown that never returns, or an agent worker blocking the ``finally``.
    When that happens the process has NO backstop and a "dead" CLI lingers
    (observed: ``hermes --tui`` alive ~47 min at 4% CPU after terminal close —
    the #65998 class).

    Arming at signal time closes that window. The leash is 2× the normal
    cleanup timeout so a slow-but-progressing ``_run_cleanup`` (which arms
    its own tighter timer when it starts) is never cut short by this outer
    backstop — this timer only wins when cleanup was never reached at all.

    Deliberately NOT armed at chat startup: the watchdog thread calls
    ``os._exit(0)`` unconditionally after its sleep, so arming without
    shutdown intent would hard-kill every session that outlives the timeout.

    Idempotent (module flag) so repeated signals don't stack timer threads.
    Never raises — safe to call from a signal handler.
    """
    global _signal_watchdog_armed
    if _signal_watchdog_armed:
        return
    _signal_watchdog_armed = True
    try:
        base = float(os.getenv("HERMES_EXIT_WATCHDOG_S", "30"))
    except (TypeError, ValueError):
        base = 30.0
    if base <= 0:
        return  # explicitly disabled
    try:
        _arm_exit_watchdog(timeout_s=base * 2, from_signal=True)
    except Exception:
        pass  # never let the backstop break signal handling


def _run_cleanup(*, notify_session_finalize: bool = True):
    """Run resource cleanup exactly once."""
    global _cleanup_done, _cleanup_in_progress
    if _cleanup_done:
        return
    _cleanup_done = True
    _cleanup_in_progress = True

    try:
        # Bound total shutdown time: if cleanup (or the interpreter's
        # thread-join teardown after it) wedges, force-exit instead of
        # leaving a zombie CLI holding the terminal for minutes.
        _arm_exit_watchdog()

        # Reset terminal input modes first, before the slower resource teardown
        # below (MCP / browser / memory shutdown can take seconds). On Ctrl+C the
        # user's terminal becomes usable immediately, and a later step raising
        # can't skip the reset (#36823). No-op unless the TUI actually ran.
        _reset_terminal_input_modes_on_exit()

        try:
            from tools.wake_word import stop_listening as _stop_wake_word
            if _cli_wake_owner is not None:
                _stop_wake_word(owner=_cli_wake_owner)
        except Exception:
            pass
        try:
            _cleanup_all_terminals()
        except Exception:
            pass
        try:
            from tools.async_delegation import interrupt_all as _interrupt_async_delegations
            _interrupt_async_delegations(reason="CLI shutdown")
        except Exception:
            pass
        try:
            _cleanup_all_browsers()
        except Exception:
            pass
        try:
            from tools.mcp_tool import shutdown_mcp_servers
            shutdown_mcp_servers()
        except BaseException:
            pass
        # Close cached auxiliary LLM clients (sync + async) so that
        # AsyncHttpxClientWrapper.__del__ doesn't fire on a closed event loop
        # and trigger prompt_toolkit's "Press ENTER to continue..." handler.
        try:
            from agent.auxiliary_client import shutdown_cached_clients
            shutdown_cached_clients()
        except Exception:
            pass
        # Shut down memory provider (on_session_end + shutdown_all) at actual
        # session boundary — NOT per-turn inside run_conversation().
        if notify_session_finalize:
            cleanup_session_id = _active_agent_ref.session_id if _active_agent_ref else None
            if _should_emit_cleanup_session_finalize(cleanup_session_id):
                _notify_session_finalize(
                    session_id=cleanup_session_id,
                    platform="cli",
                    reason="shutdown",
                )
        try:
            if _active_agent_ref and hasattr(_active_agent_ref, 'shutdown_memory_provider'):
                # A /new shortly before exit leaves its end→switch boundary task
                # (old-session extraction, LLM-bound) queued on the memory
                # manager's serialized worker. shutdown_all()'s drain only waits
                # ~5s and cancels queued tasks, so give pending work a bounded
                # head start via the manager's own barrier — otherwise a
                # "/new then quit" silently drops the old session's extraction.
                # The 30s exit watchdog remains the hard backstop.
                _mm = getattr(_active_agent_ref, '_memory_manager', None)
                if _mm is not None and hasattr(_mm, 'flush_pending'):
                    try:
                        _mm.flush_pending(timeout=10)
                    except Exception:
                        pass
                # Forward the agent's own transcript so memory providers'
                # on_session_end hooks see the real conversation instead of
                # an empty list (#15165). ``_session_messages`` is set on
                # ``AIAgent.__init__`` and refreshed every turn via
                # ``_persist_session``. Fall back to no-arg on test stubs /
                # partially-initialised agents where the attribute is missing.
                _session_msgs = getattr(_active_agent_ref, '_session_messages', None)
                if isinstance(_session_msgs, list):
                    logger.info(
                        "CLI cleanup calling memory shutdown for session %s with %d message(s)",
                        getattr(_active_agent_ref, "session_id", None) or "<unknown>",
                        len(_session_msgs),
                    )
                    _active_agent_ref.shutdown_memory_provider(_session_msgs)
                else:
                    logger.info(
                        "CLI cleanup calling memory shutdown for session %s without session message list",
                        getattr(_active_agent_ref, "session_id", None) or "<unknown>",
                    )
                    _active_agent_ref.shutdown_memory_provider()
        except Exception as e:
            logger.warning("CLI cleanup memory shutdown failed: %s", e, exc_info=True)
    finally:
        _cleanup_in_progress = False


def _should_emit_cleanup_session_finalize(session_id: str | None) -> bool:
    # A session that was handed off to the gateway is now owned by the
    # gateway process.  The CLI must not finalize it on exit — that sets
    # end_reason on a row the gateway reopened and is actively writing
    # to, causing the handoff leg to vanish from session history (#88234).
    if session_id is not None and session_id in _handed_off_session_ids:
        return False
    if not _single_query_finalize_attempted_session_ids:
        return True
    if session_id is None:
        return False
    return session_id not in _single_query_finalize_attempted_session_ids


def _notify_session_finalize(
    *,
    session_id: str | None,
    platform: str = "cli",
    reason: str = "shutdown",
) -> None:
    try:
        from hermes_cli.lifecycle import finalize_session
        finalize_session(
            session_id=session_id,
            platform=platform,
            reason=reason,
        )
    except Exception:
        pass


def _emit_interrupted_session_end(cli, *, reason: str = "keyboard_interrupt") -> None:
    """Best-effort on_session_end hook for interrupted non-interactive runs."""
    agent = getattr(cli, "agent", None)
    if agent is None:
        return

    try:
        agent.interrupt(reason.replace("_", " "))
    except Exception:
        pass

    session_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
    # Don't emit session-end for a session that was handed off to the
    # gateway — the gateway owns the lifecycle now (#88234).
    if session_id in _handed_off_session_ids:
        return
    if session_id:
        try:
            cli.session_id = session_id
        except Exception:
            pass

    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=session_id,
            task_id=getattr(agent, "_current_task_id", "") or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            completed=False,
            interrupted=True,
            model=getattr(agent, "model", None),
            platform=getattr(agent, "platform", None) or "cli",
            reason=reason,
        )
    except Exception:
        pass


def _notify_single_query_session_finalize(cli, *, reason: str = "shutdown") -> None:
    agent = getattr(cli, "agent", None)
    session_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
    if session_id in _single_query_finalize_attempted_session_ids:
        return
    # Don't finalize a session that was handed off to the gateway —
    # the gateway owns the lifecycle now (#88234).
    if session_id in _handed_off_session_ids:
        return

    try:
        _notify_session_finalize(
            session_id=session_id,
            platform=getattr(agent, "platform", None) or "cli",
            reason=reason,
        )
    finally:
        _single_query_finalize_attempted_session_ids.add(session_id)


def _flush_one_shot_session_store(cli) -> None:
    """Durably flush + finalize the one-shot session row before process exit.

    The quiet/one-shot ``-q`` / ``-Q`` paths (including resume-or-create of a
    titled session via ``-c <name> --create-if-missing``, the Bot Mode
    bot-to-bot send) get exactly ONE turn and then exit. The interactive CLI
    finalizes its session row on quit (``end_session(..., "cli_close")``) and
    every later turn retries a transiently-failed transcript flush; the
    one-shot path had neither, so:

    - a turn whose in-loop ``_flush_messages_to_session_db`` failed under
      write-lock contention (e.g. a busy multiplex gateway sharing state.db)
      was silently lost — the reply reached stdout and agent.log but the
      resumed session's stored history never changed (#88583);
    - the resumed/created titled session row was left dangling open
      (``ended_at``/``end_reason`` NULL) on every one-shot exit;
    - queued async token-accounting deltas relied on interpreter-exit hooks,
      which the kanban SIGTERM path's ``os._exit(0)`` skips entirely.

    Idempotent and best-effort: ``_persist_session`` dedupes via the
    per-message ``_DB_PERSISTED_MARKER`` stamps (already-written turns are
    not re-written) and ``end_session`` no-ops on an already-ended row.
    Sessions handed off to the gateway are owned by the gateway process and
    are left strictly alone (#88234).
    """
    agent = getattr(cli, "agent", None)
    if agent is None:
        return
    session_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
    if not session_id or session_id in _handed_off_session_ids:
        return
    if getattr(agent, "_persist_disabled", False):
        return
    # Retry persistence for any rows the in-turn flush failed to write.
    # ``cli.conversation_history`` holds the resumed history's live dicts, so
    # passing it keeps restored messages identity-skipped even when the failed
    # first flush never got to stamp them.
    try:
        msgs = getattr(agent, "_session_messages", None)
        if isinstance(msgs, list) and msgs and hasattr(agent, "_persist_session"):
            agent._persist_session(
                msgs, getattr(cli, "conversation_history", None)
            )
    except Exception:
        logger.debug("one-shot final session persist retry failed", exc_info=True)
    db = getattr(agent, "_session_db", None) or getattr(cli, "_session_db", None)
    if db is None:
        return
    try:
        db.flush_token_counts()
    except Exception:
        logger.debug("one-shot token-count drain failed", exc_info=True)
    try:
        db.end_session(session_id, "cli_close")
    except Exception:
        logger.debug("one-shot end_session failed", exc_info=True)


def _wait_for_oneshot_background_completions(cli) -> None:
    """Bounded linger for notify_on_complete background processes (#90879).

    A one-shot run (``-q`` / ``-Q``) that spawned bounded background work —
    most importantly a Bot Mode handoff reply via ``message_agent`` /
    ``bot_relay``, spawned as ``terminal(background=true,
    notify_on_complete=true)`` — must not exit while that work is still
    running: the children write to pipes owned by this process and are
    destroyed shortly after it dies. Delegates the actual wait (and its
    ``terminal.oneshot_completion_wait_seconds`` bound) to the process
    registry. Cheap no-op when nothing is pending.
    """
    from tools.process_registry import process_registry

    agent = getattr(cli, "agent", None)
    task_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
    # Wait on the whole registry, not just this task's processes: a one-shot
    # CLI process hosts exactly one agent, so every tracked process in this
    # interpreter was spawned by this run (task_id filtering would silently
    # skip processes registered before the session id settled).
    result = process_registry.wait_for_pending_completions(None)
    if result.get("waited"):
        logger.info(
            "One-shot exit linger for session %s: completed=%s timed_out=%s",
            task_id or "<unknown>",
            result.get("completed"),
            result.get("timed_out"),
        )


def _finalize_single_query(cli) -> None:
    """Close one-shot CLI resources before releasing the active session lease."""
    try:
        # Linger (bounded) for background processes the turn spawned with
        # notify_on_complete=true BEFORE any teardown. The one-shot parent
        # owns those children's stdout pipes; exiting now kills the delivery
        # a few seconds later. Bot Mode handoff replies dispatched from a
        # short-lived `hermes -p <bot> chat -Q` recipient (message_agent /
        # bot_relay spawns) are exactly this shape and were silently
        # destroyed on parent exit (#90879).
        try:
            _wait_for_oneshot_background_completions(cli)
        except Exception:
            logger.debug("one-shot background completion wait failed", exc_info=True)
        # Durable flush FIRST: memory-provider shutdown inside _run_cleanup
        # can issue aux-LLM calls, and nothing after it may fail in a way
        # that loses the turn (#88583).
        try:
            _flush_one_shot_session_store(cli)
        except Exception:
            logger.debug("one-shot session store flush failed", exc_info=True)
        _notify_single_query_session_finalize(cli)
        _run_cleanup(notify_session_finalize=False)
    finally:
        cli._release_active_session()


def _reset_terminal_input_modes_on_exit() -> None:
    """Best-effort: disable focus reporting + mouse tracking on TUI exit so they
    don't leak into the next shell session sharing the tab.

    prompt_toolkit restores these on a clean teardown, but Ctrl+C, SIGTERM /
    SIGHUP and crashes can bypass its unwind, leaving the modes enabled. The
    terminal then emits raw ``ESC[I`` / ``ESC[O`` focus events and fragmented
    SGR mouse reports as visible text in whatever runs next in the same tab
    (#36823). Called from ``_run_cleanup`` (atexit-registered + invoked on the
    normal / EOF / interrupt exit paths) this covers normal quit, Ctrl+C and
    SIGTERM/SIGHUP. ``kill -9`` is uncatchable, and the kanban worker's
    ``os._exit(0)`` path bypasses ``atexit``; neither runs this — but both are
    non-TTY / non-TUI, so there is nothing to reset there.

    Gated on ``_tui_input_modes_active`` so one-shot non-TUI CLI runs (which
    share ``_run_cleanup`` via ``atexit``) never emit these codes. Writes to the
    controlling terminal directly: by exit, prompt_toolkit's own output is torn
    down, so ``sys.stdout`` is the real fd; falls back to ``/dev/tty`` when
    stdout is redirected away from the terminal.
    """
    global _tui_input_modes_active
    if not _tui_input_modes_active:
        return
    # About to disable the modes — clear the flag so a re-armed _run_cleanup (or
    # a long-lived process that reuses it) doesn't re-emit them.
    _tui_input_modes_active = False
    # Prefer stdout when it's the terminal; otherwise the TUI may have driven
    # /dev/tty while stdout was redirected — reset there instead of nowhere.
    try:
        stream = sys.stdout
        if stream is not None and stream.isatty():
            stream.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
            stream.flush()
            return
    except Exception:
        pass
    try:
        with open("/dev/tty", "w", encoding="ascii") as tty:
            tty.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
            tty.flush()
    except Exception:
        pass


# =============================================================================
# Git Worktree Isolation (#652)
# =============================================================================

# Tracks the active worktree for cleanup on exit
_active_worktree: Optional[Dict[str, str]] = None


def _normalize_git_bash_path(p: Optional[str]) -> Optional[str]:
    """Translate a Git Bash-style path (``/c/Users/...``) to the native
    Windows form (``C:\\Users\\...``) that Python's ``subprocess.Popen``
    and ``pathlib.Path`` accept.

    No-op on non-Windows and for paths that already look native.  Git on
    native Windows normally emits forward-slash Windows paths
    (``C:/Users/...``) which both bash and Python handle, but certain
    configurations (Git Bash shells, MSYS2, WSL-mounted repos) surface
    ``/c/...`` or ``/cygdrive/c/...`` variants.
    """
    if not p:
        return p
    if sys.platform != "win32":
        return p
    import re as _re
    # /c/Users/... or /C/Users/...
    m = _re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    # /cygdrive/c/... or /mnt/c/...
    m = _re.match(r"^/(?:cygdrive|mnt)/([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    return p


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD, or None if not in a repo.

    Runs through :func:`_normalize_git_bash_path` so callers can pass
    the result directly to ``Path``/``subprocess.Popen(cwd=...)`` on
    Windows without hitting ``C:\\c\\Users\\...`` style resolution
    mistakes.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if result.returncode == 0:
            return _normalize_git_bash_path(result.stdout.strip())
    except Exception:
        pass
    return None


def _path_is_within_root(path: Path, root: Path) -> bool:
    """Return True when a resolved path stays within the expected root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _cleanup_failed_worktree_add(repo_root: str, wt_path: Path, branch_name: str) -> None:
    """Make a failed/timed-out ``git worktree add`` atomic after the fact.

    ``git worktree add`` is not transactional: killed mid-checkout (the 30s
    timeout) it leaves (a) the partially-materialized worktree directory,
    (b) an admin entry under ``.git/worktrees/<name>`` that is LOCKED with a
    reason naming the *current, live* pid — so the startup pruner's
    dead-pid unlock will never touch it — and (c) sometimes the new branch.
    Any retry of the same name then fails on the leftovers. Sweep all three,
    quietly; every step is fail-soft because this runs on an error path.
    """
    import shutil
    import subprocess

    def _git(*args: str) -> None:
        try:
            subprocess.run(
                ["git", *args],
                capture_output=True, text=True, timeout=15, cwd=repo_root, check=False,
            )
        except Exception:
            pass

    try:
        # Unlock first: `worktree remove --force` refuses a locked tree.
        _git("worktree", "unlock", str(wt_path))
        _git("worktree", "remove", "--force", str(wt_path))
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        # Drop the orphaned admin entry when the dir is already gone
        # (`remove` needs the dir; `prune` handles the dirless case).
        _git("worktree", "prune")
        _git("branch", "-D", branch_name)
    except Exception as e:
        logger.debug("cleanup after failed worktree add: %s", e)


_PACK_SPRAWL_THRESHOLD = 15


def _maintain_pack_health(repo_root: str) -> None:
    """Repack the object store when pack files sprawl (background thread).

    On a multi-agent box every fetch/salvage session adds packs; git never
    consolidates them on its own aggressively enough (``gc --auto``'s
    threshold is 50 *and* it counts only non-kept packs). Past a few dozen
    packs every object lookup scans every pack index, and worktree creation
    can blow its 30s timeout under concurrent load (Aug 2026 incident: 39
    packs, 638MB → ``hermes -w`` timing out; a full repack halved the store
    and restored 0.5s creates). Threshold 15 keeps lookups fast without
    repacking on every startup; ``nice`` + background thread keeps it off
    the startup path. Fail-soft everywhere.
    """
    import subprocess

    try:
        pack_dir = Path(repo_root) / ".git" / "objects" / "pack"
        if not pack_dir.is_dir():
            return
        packs = len(list(pack_dir.glob("*.pack")))
        if packs < _PACK_SPRAWL_THRESHOLD:
            return
        logger.info("git pack sprawl (%d packs) — repacking in background", packs)
        cmd = ["git", "repack", "-a", "-d", "--quiet"]
        if os.name == "posix":
            cmd = ["nice", "-n", "19", *cmd]
        subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=1800, cwd=repo_root, check=False,
        )
        # Repacking can strand now-duplicated admin files; a prune here keeps
        # the worktree bookkeeping tight on the same maintenance pass.
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True, text=True, timeout=60, cwd=repo_root, check=False,
        )
    except Exception as e:
        logger.debug("pack maintenance skipped: %s", e)


def _resolve_worktree_base(
    repo_root: str,
    fetch_timeout: float = 5,
    freshness_window: float = 300,
) -> tuple:
    """Resolve the freshest base ref to branch a new worktree from.

    The standalone clone's ``HEAD`` can lag the remote by hundreds of commits
    (the ``~/.hermes/hermes-agent`` clone is updated only by ``hermes update``,
    not on every session). Branching a worktree from that stale ``HEAD`` roots
    every new branch on an old base — so the PR diff GitHub computes against
    current ``main`` balloons with unrelated changes, and the agent has to
    discover the staleness via the pre-push gate and rebase. Branching from the
    freshly-fetched remote tip instead means the worktree starts current.

    Strategy (each step falls back to the next on failure):
      1. If the current branch tracks an upstream, refresh and use that
         upstream ref — so a deliberate feature-branch worktree tracks its own
         remote, not the default branch.
      2. Else refresh the remote's default branch (``origin/HEAD`` → e.g.
         ``origin/main``) and use it.
      3. Else fall back to ``HEAD`` (offline, no remote, or detached) — the
         old behavior, never worse than before.

    "Refresh" is deliberately cheap on the startup path (the fetch here used
    to stall ``hermes -w`` launches for 30-60s on flaky smart-HTTP
    connections):

    - The fetch is SKIPPED entirely when the repo's ``FETCH_HEAD`` is younger
      than *freshness_window* seconds — a base fetched moments ago cannot have
      meaningfully moved, so repeated launches don't re-pay a network round
      trip.
    - The fetch is capped at *fetch_timeout* seconds. On timeout or failure we
      fall back to the locally-known remote-tracking ref (labelled "cached")
      instead of cascading into a second fetch attempt. Genuine staleness is
      backstopped by the pre-push stale-base gate.

    Returns ``(base_ref, label)`` where *base_ref* is a git revision suitable
    for ``git worktree add ... <base_ref>`` and *label* is a short
    human-readable description for the session banner.
    """
    import subprocess

    from hermes_cli._subprocess_compat import noninteractive_git_env

    def _git(args, timeout: float = 20):
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )

    def _ref_exists(ref: str) -> bool:
        try:
            return _git(["rev-parse", "--verify", "--quiet", ref + "^{commit}"]).returncode == 0
        except Exception:
            return False

    def _fetch_head_age() -> Optional[float]:
        """Seconds since the last fetch in this repo, or None if unknown."""
        try:
            gd = _git(["rev-parse", "--git-dir"])
            if gd.returncode != 0:
                return None
            git_dir = Path(gd.stdout.strip())
            if not git_dir.is_absolute():
                git_dir = Path(repo_root) / git_dir
            fetch_head = git_dir / "FETCH_HEAD"
            if not fetch_head.exists():
                return None
            return max(0.0, time.time() - fetch_head.stat().st_mtime)
        except Exception:
            return None

    def _refresh(remote: str, branch: str, ref: str) -> tuple:
        """Return (ref, label) after a cheap best-effort refresh of *ref*.

        Never raises, never fetches twice, never blocks longer than
        *fetch_timeout*.
        """
        age = _fetch_head_age()
        if age is not None and age < freshness_window and _ref_exists(ref):
            return ref, f"{ref} (fetched {int(age)}s ago)"
        try:
            fetched = _git(["fetch", remote, branch], timeout=fetch_timeout)
            if fetched.returncode == 0:
                return ref, f"{ref} (fetched)"
            reason = "fetch failed"
        except subprocess.TimeoutExpired:
            reason = f"fetch timed out after {fetch_timeout:g}s"
        except Exception as e:
            reason = f"fetch error: {e}"
        if _ref_exists(ref):
            logger.debug("worktree base: %s — using cached %s", reason, ref)
            return ref, f"{ref} (cached — {reason})"
        return "HEAD", f"HEAD (local — {reason}, no cached {ref})"

    # 1. Current branch's upstream, if it tracks one.
    try:
        up = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
        if up.returncode == 0:
            upstream = up.stdout.strip()  # e.g. "origin/main"
            if upstream and "/" in upstream:
                remote, branch = upstream.split("/", 1)
                return _refresh(remote, branch, upstream)
    except Exception as e:
        logger.debug("worktree base: upstream resolution failed: %s", e)

    # 2. Remote default branch (origin/HEAD).
    try:
        # Resolve the remote's default branch symref.
        head_ref = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
        default_ref = ""
        if head_ref.returncode == 0:
            default_ref = head_ref.stdout.strip().replace("refs/remotes/", "", 1)
        if not default_ref:
            # origin/HEAD not set locally; ask the remote (network — capped
            # like the fetch so a stalled connection can't hang startup).
            show = _git(["remote", "show", "origin"], timeout=max(fetch_timeout, 5))
            for line in show.stdout.splitlines():
                line = line.strip()
                if line.startswith("HEAD branch:"):
                    _branch = line.split(":", 1)[1].strip()
                    # A remote with no default branch reports "(unknown)";
                    # don't construct a bogus "origin/(unknown)" ref from it.
                    if _branch and _branch != "(unknown)":
                        default_ref = "origin/" + _branch
                    break
        if default_ref and "/" in default_ref:
            remote, branch = default_ref.split("/", 1)
            return _refresh(remote, branch, default_ref)
    except Exception as e:
        logger.debug("worktree base: default-branch resolution failed: %s", e)

    # 3. Fall back to local HEAD (offline / no remote / detached).
    return "HEAD", "HEAD (local — could not reach remote)"


def _setup_worktree(repo_root: str = None, sync_base: bool = True,
                    name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Create an isolated git worktree for this CLI session.

    Returns a dict with worktree metadata on success, None on failure.
    The dict contains: path, branch, repo_root.

    When *sync_base* is True (default), the worktree branches from the
    freshly-fetched remote tip rather than the (possibly stale) local ``HEAD``
    — see ``_resolve_worktree_base``. Set ``worktree_sync: false`` in config to
    branch from local ``HEAD`` (the pre-#10760-followup behavior).

    When *name* is given (``/worktree new <name>``), the worktree directory
    and branch use the sanitized name instead of a random ``hermes-<id>``.
    Named trees intentionally skip the ``hermes-`` prefix so the startup
    pruner ages them on its slower named-tree schedule.
    """
    import subprocess

    from hermes_cli._subprocess_compat import (
        noninteractive_git_env as _noninteractive_git_env,
    )

    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        _cprint("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    if name:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")[:40]
        wt_name = safe or f"hermes-{uuid.uuid4().hex[:8]}"
    else:
        wt_name = f"hermes-{uuid.uuid4().hex[:8]}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    wt_path = worktrees_dir / wt_name
    if name and wt_path.exists():
        _cprint(f"\033[31m✗ Worktree already exists: {wt_path}\033[0m")
        print("  Pick a different name, or remove it with: "
              f"git worktree remove {wt_path}")
        return None

    # Ensure .worktrees/ is in .gitignore
    gitignore = Path(repo_root) / ".gitignore"
    _ignore_entry = ".worktrees/"
    try:
        # utf-8-sig: git files are UTF-8 and Notepad prepends a BOM, which
        # would glue to the first line and defeat the membership check below
        # (duplicating the entry); the locale default also breaks non-ASCII
        # patterns on Windows. The append below already writes UTF-8.
        existing = (
            gitignore.read_text(encoding="utf-8-sig", errors="replace")
            if gitignore.exists()
            else ""
        )
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)

    # Resolve the base ref. By default branch from the freshly-fetched remote
    # tip so the worktree starts current with the project, not from the
    # (possibly stale) local HEAD of the standalone clone (#10760 follow-up).
    if sync_base:
        base_ref, base_label = _resolve_worktree_base(repo_root)
    else:
        base_ref, base_label = "HEAD", "HEAD (local — worktree_sync disabled)"

    # Create the worktree. checkout.workers parallelizes the file
    # materialization (~6k files on this repo): 0.6s serial → ~0.2s with 8
    # workers. Harmless on git builds without parallel-checkout support —
    # unknown -c keys are ignored for checkout, and the fallback retry
    # below drops the flags entirely.
    _wt_add_cfg = [
        "-c", "checkout.workers=8",
        "-c", "checkout.thresholdForParallelism=100",
    ]
    try:
        # 120s, not 30: on a multi-agent box the ~10k-file materialization
        # contends with sibling sessions' checkouts/fetches/Electron dev
        # builds for the same disk — measured 113s wall at near-zero CPU
        # under load vs 1.2s idle (Aug 2026). A too-tight timeout kills a
        # legitimately slow create and wastes the work already done.
        result = subprocess.run(
            ["git", *_wt_add_cfg, "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=repo_root,
            stdin=subprocess.DEVNULL, env=_noninteractive_git_env(),
        )
        if result.returncode != 0:
            # If branching from the resolved remote ref failed for any reason
            # (e.g. a partial fetch left the ref unusable), retry from local
            # HEAD so worktree creation never hard-fails on a sync hiccup.
            if base_ref != "HEAD":
                logger.warning(
                    "worktree add from %s failed (%s); retrying from local HEAD",
                    base_ref, result.stderr.strip(),
                )
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                base_ref, base_label = "HEAD", "HEAD (fallback — remote base failed)"
                result = subprocess.run(
                    ["git", "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=repo_root,
                    stdin=subprocess.DEVNULL, env=_noninteractive_git_env(),
                )
            if result.returncode != 0:
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                _cprint(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
                return None
    except Exception as e:
        # A timed-out/failed `worktree add` is NOT atomic: git leaves the
        # partially-materialized directory plus a LOCKED admin entry under
        # .git/worktrees/<name> whose lock pid is THIS live process — so the
        # startup pruner's dead-pid unlock never reaps it and every retry of
        # the same name fails. Clean up our own wreckage before surfacing
        # the error (Aug 2026 incident: 30s timeout during pack-sprawl left
        # exactly this poison).
        _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
        _cprint(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None

    # Copy files listed in .worktreeinclude (gitignored files the agent needs)
    include_file = Path(repo_root) / ".worktreeinclude"
    if include_file.exists():
        try:
            repo_root_resolved = Path(repo_root).resolve()
            wt_path_resolved = wt_path.resolve()
            # utf-8-sig, not the locale default: on a cp1251/GBK Windows
            # machine a UTF-8 include list either decodes to mojibake paths
            # (entries silently not copied) or raises UnicodeDecodeError,
            # which the enclosing handler swallows at DEBUG — no include is
            # copied at all. A Notepad BOM likewise glued to the first entry.
            for line in include_file.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                src = Path(repo_root) / entry
                dst = wt_path / entry
                # Prevent path traversal and symlink escapes: both the resolved
                # source and the resolved destination must stay inside their
                # expected roots before any file or symlink operation happens.
                try:
                    src_resolved = src.resolve(strict=False)
                    dst_resolved = dst.resolve(strict=False)
                except (OSError, ValueError):
                    logger.debug("Skipping invalid .worktreeinclude entry: %s", entry)
                    continue
                if not _path_is_within_root(src_resolved, repo_root_resolved):
                    logger.warning("Skipping .worktreeinclude entry outside repo root: %s", entry)
                    continue
                if not _path_is_within_root(dst_resolved, wt_path_resolved):
                    logger.warning("Skipping .worktreeinclude entry that escapes worktree: %s", entry)
                    continue
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                elif src.is_dir():
                    # Symlink directories (faster, saves disk).  On Windows,
                    # symlink creation requires Developer Mode or elevation,
                    # and fails with OSError otherwise — fall back to a
                    # recursive copy so the worktree is still usable.  The
                    # copy is slower and uses disk, but it doesn't require
                    # admin and matches the Linux/macOS symlink outcome
                    # functionally.
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.symlink(str(src_resolved), str(dst))
                        except (OSError, NotImplementedError) as _sym_err:
                            if sys.platform == "win32":
                                logger.info(
                                    ".worktreeinclude: symlink failed (%s) — "
                                    "falling back to copytree on Windows.",
                                    _sym_err,
                                )
                                try:
                                    shutil.copytree(
                                        str(src_resolved),
                                        str(dst),
                                        symlinks=True,
                                        dirs_exist_ok=False,
                                    )
                                except Exception as _copy_err:
                                    logger.warning(
                                        ".worktreeinclude: copy fallback "
                                        "also failed for %s -> %s: %s",
                                        src, dst, _copy_err,
                                    )
                            else:
                                raise
        except Exception as e:
            logger.debug("Error copying .worktreeinclude entries: %s", e)

    # Lock the worktree so other processes (and `git worktree remove`) can see
    # it is actively in use.  Fail-soft: a lock failure never blocks the session.
    try:
        subprocess.run(
            ["git", "worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(wt_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        logger.debug("Worktree locked: %s (pid=%s)", wt_path, os.getpid())
    except Exception as e:
        logger.debug("git worktree lock failed (non-fatal): %s", e)

    info = {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
        "base": base_ref,
    }

    _cprint(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")
    print(f"  Base:   {base_label}")

    return info


def _worktree_has_unpushed_commits(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has commits not reachable from any remote branch.

    ``git log HEAD --not --remotes`` compares against remote-tracking refs under
    ``refs/remotes/*``. If a repo has no remote-tracking refs yet, there is no
    usable remote baseline to compare against, so treat it as having no
    "unpushed" commits.

    SHALLOW-CLONE CAVEAT: in a shallow clone (the installer default) the
    shallow boundary can disconnect an older worktree HEAD from origin/*,
    making already-public commits look unpushed. The verdict here stays
    conservative (True) on purpose — deleting on unverifiable history would
    risk real work. Callers that can afford it should deepen first via
    ``_deepen_shallow_repo`` (the startup pruner does) or check
    ``_repo_is_shallow`` before presenting this verdict as fact.
    """
    import subprocess

    try:
        remote_refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", "refs/remotes"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if remote_refs.returncode != 0:
            return True
        if not remote_refs.stdout.strip():
            return False

        result = subprocess.run(
            ["git", "log", "--oneline", "HEAD", "--not", "--remotes"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _worktree_is_dirty(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has uncommitted changes (staged, unstaged, or
    untracked).

    Fails SAFE: on any error returns True so callers do not delete a worktree
    whose state they cannot determine.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _repo_is_shallow(repo_path: str, timeout: int = 5) -> bool:
    """Return whether *repo_path* belongs to a shallow clone.

    Shallowness poisons every history-connectivity verdict the worktree
    machinery relies on: an older worktree's HEAD (a past snapshot of main)
    is disconnected from current ``origin/main`` by the shallow boundary, so
    ``git log HEAD --not --remotes`` misreports thousands of already-public
    commits as "unpushed" and the worktree is preserved forever. The default
    installer clones with ``--depth 1``, so this is the normal state of a
    user install, not an edge case.

    Fails toward False: if git can't be queried we don't want callers to
    take shallow-specific branches on top of an unknown state.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_path,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def _deepen_shallow_repo(repo_root: str, timeout: int = 600) -> bool:
    """One-time blobless unshallow so history-based verdicts become correct.

    Fetches the full commit/tree graph (``--unshallow --filter=blob:none``)
    without downloading historical file contents, which keeps the transfer a
    small fraction of a full clone. Runs only from background paths (the
    startup pruner thread), never on the interactive session-close path.

    Falls back to a plain ``--unshallow`` if the server rejects partial-clone
    filters. Fail-soft: returns whether the repo is actually non-shallow
    afterwards; on failure (offline, no remote) callers keep today's
    preserve-everything behavior.
    """
    import subprocess

    if not _repo_is_shallow(repo_root):
        return True

    try:
        remotes = subprocess.run(
            ["git", "remote"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        names = [r.strip() for r in remotes.stdout.splitlines() if r.strip()]
        if remotes.returncode != 0 or not names:
            return False
        remote = "origin" if "origin" in names else names[0]

        for extra in (["--filter=blob:none"], []):
            try:
                result = subprocess.run(
                    ["git", "fetch", remote, "--unshallow", *extra],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode == 0:
                break
            logger.debug(
                "git fetch --unshallow%s failed: %s",
                " " + " ".join(extra) if extra else "",
                result.stderr.strip()[-500:],
            )
    except Exception as e:
        logger.debug("Deepening shallow repo failed (non-fatal): %s", e)
        return False

    deepened = not _repo_is_shallow(repo_root)
    if deepened:
        logger.info(
            "Deepened shallow clone at %s so worktree cleanup can verify "
            "push state", repo_root,
        )
    return deepened


# Upper bound on retained `git cherry` verdict entries (see
# _save_worktree_merge_cache). Each entry is ~90 bytes, so this caps the cache
# near 90 KB even on a repo that churns thousands of worktree branches.
_WORKTREE_MERGE_CACHE_MAX = 1000


def _worktree_merge_cache_path() -> Path:
    """Path of the patch-equivalence verdict cache (profile-aware)."""
    return get_hermes_home() / "cache" / "worktree_merge_verdicts.json"


def _load_worktree_merge_cache() -> Dict[str, bool]:
    """Load the ``git cherry`` verdict cache. Missing/corrupt cache = empty."""
    try:
        raw = json.loads(
            _worktree_merge_cache_path().read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("verdicts")
    if not isinstance(entries, dict):
        return {}
    # Only keep well-formed bool verdicts — a hand-edited or partially written
    # cache must never inject a non-bool into the prune decision.
    return {k: v for k, v in entries.items() if isinstance(v, bool)}


def _save_worktree_merge_cache(verdicts: Dict[str, bool]) -> None:
    """Persist the verdict cache atomically. Best-effort — never raises.

    Bounded to the most recent ``_WORKTREE_MERGE_CACHE_MAX`` entries so the
    file can't grow without limit across thousands of sessions.
    """
    path = _worktree_merge_cache_path()
    tmp = None
    try:
        items = list(verdicts.items())
        if len(items) > _WORKTREE_MERGE_CACHE_MAX:
            items = items[-_WORKTREE_MERGE_CACHE_MAX:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"version": 1, "verdicts": dict(items)}),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except Exception as e:
        logger.debug("Could not persist worktree merge cache: %s", e)
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def _worktree_commits_all_merged_upstream(
    worktree_path: str,
    timeout: int = 30,
    max_ahead: int = 20,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Return whether every local-only commit is patch-equivalent to a commit
    already on the default upstream branch.

    The dominant ``.worktrees/`` leak: a branch is pushed, its PR is
    squash-merged (or cherry-picked), and the remote branch is deleted. The
    local commits are then unreachable from ``refs/remotes/*`` forever, so the
    unpushed-commits guard preserves the worktree indefinitely even though its
    content is fully merged. ``git cherry`` detects patch-equivalence, letting
    the pruner reap these.

    Bounded: skips (returns False) when the branch is more than ``max_ahead``
    commits ahead — a stale-base tree, too expensive to diff-hash and unlikely
    to be a merged scratch branch. Fails SAFE toward False (preserve).

    ``git cherry`` diff-hashes every commit in the range, which on a large repo
    costs ~0.2-1.0s per worktree — and a tree preserved for unpushed work is
    re-tested on *every* startup, forever, always reaching the same answer. When
    *cache* is provided, the verdict is memoized against
    ``(base_sha, head_sha, max_ahead)``: the exact inputs ``git cherry``
    consumes. A cache hit is therefore identical to recomputation by
    construction — if either ref moves the key changes and the real git call
    runs again.
    """
    import subprocess

    base = None
    for candidate in ("origin/HEAD", "origin/main", "origin/master"):
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", candidate],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                base = candidate
                break
        except Exception:
            return False
    if base is None:
        return False

    try:
        # Resolve both endpoints to shas up front. These are the complete
        # inputs to the range below, so they form an exact cache key. Cheap
        # (~1ms) relative to the diff-hashing `git cherry` they guard.
        cache_key = None
        if cache is not None:
            revs = subprocess.run(
                ["git", "rev-parse", f"{base}^{{commit}}", "HEAD^{commit}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if revs.returncode == 0:
                shas = revs.stdout.split()
                if len(shas) == 2:
                    cache_key = f"{shas[0]}..{shas[1]}:{max_ahead}"
                    if cache_key in cache:
                        return cache[cache_key]

        def _memo(verdict: bool) -> bool:
            if cache is not None and cache_key is not None:
                cache[cache_key] = verdict
            return verdict

        ahead = subprocess.run(
            ["git", "rev-list", "--count", f"{base}..HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if ahead.returncode != 0:
            return False
        count = int(ahead.stdout.strip() or "0")
        if count == 0:
            return _memo(True)
        if count > max_ahead:
            return _memo(False)

        cherry = subprocess.run(
            ["git", "cherry", base, "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if cherry.returncode != 0:
            return False
        lines = [ln for ln in cherry.stdout.splitlines() if ln.strip()]
        # "-" = patch-equivalent commit exists upstream; "+" = unique local work
        return _memo(bool(lines) and all(ln.startswith("-") for ln in lines))
    except Exception:
        return False


def _worktree_branch_pr_merged(
    worktree_path: str,
    timeout: int = 15,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Return whether the worktree branch's PR is MERGED on GitHub.

    Escape hatch for the case ``git cherry`` cannot catch: a rebase-merge that
    altered the diff (conflict resolution against a moved base, follow-up
    commits added during salvage/CI-fix) changes the patch-id, so the local
    commits are no longer patch-equivalent to anything upstream even though
    the PR merged. Those trees survive the cherry check forever (Aug 2026:
    12 of 22 "unpushed" trees on a loaded box had MERGED PRs).

    GitHub's PR state is the authoritative merge signal, so a clean tree
    whose branch has a MERGED PR is reaped. Verdicts are memoized keyed on
    ``(branch, head_sha)`` — MERGED is monotonic, so a True verdict is cached
    permanently; False is never cached (the PR may merge later without new
    local commits, which would leave the key unchanged).

    Fails SAFE toward False (preserve): no gh binary, offline, rate-limited,
    detached HEAD, or any parse failure keeps the tree.
    """
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if head.returncode != 0:
            return False
        branch = head.stdout.strip()
        if not branch or branch == "HEAD":  # detached — no PR to look up
            return False

        cache_key = None
        if cache is not None:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if sha.returncode == 0 and sha.stdout.strip():
                cache_key = f"pr-merged:{branch}:{sha.stdout.strip()}"
                if cache.get(cache_key) is True:
                    return True

        result = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--state", "merged",
             "--json", "number", "--limit", "1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return False
        prs = json.loads(result.stdout or "[]")
        merged = isinstance(prs, list) and len(prs) > 0
        if merged and cache is not None and cache_key is not None:
            cache[cache_key] = True
        return merged
    except Exception:
        return False


def _fetch_remote_branch_heads(repo_root: str, timeout: int = 20) -> Optional[Dict[str, str]]:
    """Return ``{branch_name: sha}`` for every branch on origin, or None.

    One ``git ls-remote --heads origin`` call answers "is this branch pushed?"
    for EVERY worktree in the sweep, so callers pay a single bounded network
    round-trip instead of per-tree probes. Needed because managed installs
    fetch with a single-branch refspec (``+refs/heads/main:...``), so
    ``refs/remotes/origin/<branch>`` never exists for pushed PR branches and
    ``git log HEAD --not --remotes`` misreports them as unpushed forever —
    the dominant reason open-PR worktrees accumulate (Aug 2026: 24 of 33
    preserved trees on a loaded box were pushed branches with open PRs).

    Fails SAFE toward None (offline, no remote, timeout): callers must treat
    None as "cannot verify — preserve".
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "origin"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
        )
        if result.returncode != 0:
            return None
        heads: Dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                heads[parts[1][len("refs/heads/"):].strip()] = parts[0].strip()
        return heads
    except Exception:
        return None


def _worktree_branch_pushed_exact(
    worktree_path: str,
    remote_heads: Optional[Dict[str, str]],
    timeout: int = 10,
) -> bool:
    """Return whether the worktree's branch head is EXACTLY what origin holds.

    True means the working tree contains nothing origin doesn't already have:
    the checkout is redundant — the work lives on the remote (typically as an
    open PR) and in the local branch ref. Reaping the TREE while keeping the
    BRANCH loses nothing and is one ``git worktree add`` away from undone.

    Exact-match is deliberately the only True case: a local head that is
    ahead of (or diverged from) the pushed branch has commits origin lacks,
    and without remote-tracking refs we cannot cheaply prove ancestry — so
    anything but equality fails SAFE toward preserve.
    """
    import subprocess

    if not remote_heads:
        return False
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if head.returncode != 0:
            return False
        branch = head.stdout.strip()
        if not branch or branch == "HEAD":
            return False
        remote_sha = remote_heads.get(branch)
        if not remote_sha:
            return False
        local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if local.returncode != 0:
            return False
        return local.stdout.strip() == remote_sha
    except Exception:
        return False


def _worktree_lock_is_live(repo_root: str, worktree_path: str, timeout: int = 10):
    """Classify a worktree's git lock as live, dead, or absent.

    ``hermes -w`` locks each worktree with reason ``hermes pid=<pid>`` so a
    concurrent hermes process' startup prune leaves an in-use worktree alone.
    But a *crashed* session leaves the lock behind forever, and
    ``git worktree remove --force`` (single ``-f``) refuses to remove a locked
    worktree — so dead-locked worktrees accumulate indefinitely. This lets the
    pruner tell the two apart:

    - ``"live"``  — locked and the owning pid is still running (skip it).
    - ``"dead"``  — locked but the owning pid is gone, or the reason isn't a
                    parseable hermes lock (safe to unlock + reap).
    - ``None``    — not locked at all.

    Fails SAFE toward ``"live"``: if git can't be queried at all we cannot
    prove the worktree is safe to touch, so we report it as live.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
        )
        if result.returncode != 0:
            return "live"
    except Exception:
        return "live"

    target = Path(worktree_path).resolve()
    current: Optional[Path] = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                current = Path(line[len("worktree "):].strip()).resolve()
            except Exception:
                current = None
        elif line == "locked" or line.startswith("locked "):
            if current != target:
                continue
            reason = line[len("locked"):].strip()
            m = re.search(r"hermes pid=(\d+)", reason)
            if not m:
                # Locked by something we don't recognize as a hermes session
                # (or lock reason unavailable). Treat as dead — a foreign lock
                # on a hermes -w worktree is almost certainly a leftover, and
                # the age/dirty/unpushed gates already ran before we got here.
                return "dead"
            pid = int(m.group(1))
            if pid == os.getpid():
                return "live"
            try:
                from gateway.status import _pid_exists
                return "live" if _pid_exists(pid) else "dead"
            except Exception:
                # Can't determine liveness — fail safe toward keeping it.
                return "live"
    return None


def _cleanup_worktree(info: Dict[str, str] = None) -> None:
    """Remove a worktree and its branch on exit.

    Preserves the worktree only if it has unpushed commits (real work
    that hasn't been pushed to any remote).  Uncommitted changes alone
    (untracked files, test artifacts) are not enough to keep it — agent
    work lives in commits/PRs, not the working tree.
    """
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    import subprocess

    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    has_unpushed = _worktree_has_unpushed_commits(wt_path, timeout=10)

    if has_unpushed:
        if _repo_is_shallow(repo_root):
            # In a shallow clone the unpushed verdict is unreliable: the
            # shallow boundary disconnects this worktree's history from
            # origin/*, so already-public commits look "unpushed". Be honest
            # about why we're keeping it — the startup pruner deepens the
            # clone in the background and will reap it on a later startup.
            _cprint(f"\n\033[33m⚠ Shallow clone — cannot verify push state, keeping: {wt_path}\033[0m")
            print("  The next `hermes -w` session deepens the clone and prunes merged worktrees automatically.")
        else:
            _cprint(f"\n\033[33m⚠ Worktree has unpushed commits, keeping: {wt_path}\033[0m")
            print(f"  To clean up manually: git worktree remove --force {wt_path}")
        _active_worktree = None
        return

    # Remove worktree (even if working tree is dirty — uncommitted
    # changes without unpushed commits are just artifacts)
    # Unlock first so `git worktree remove` isn't blocked by the lock we
    # placed at creation time.  Fail-soft — never block cleanup.
    try:
        subprocess.run(
            ["git", "worktree", "unlock", wt_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("git worktree unlock failed (non-fatal): %s", e)

    try:
        subprocess.run(
            ["git", "worktree", "remove", wt_path, "--force"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to remove worktree: %s", e)

    # Delete the branch
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to delete branch %s: %s", branch, e)

    _active_worktree = None
    _cprint(f"\033[32m✓ Worktree cleaned up: {wt_path}\033[0m")


def _run_state_db_auto_maintenance(session_db) -> None:
    """Call ``SessionDB.maybe_auto_prune_and_vacuum`` using current config.

    Reads the ``sessions:`` section from config.yaml via
    :func:`hermes_cli.config.load_config` (the authoritative loader that
    deep-merges DEFAULT_CONFIG, so unmigrated configs still get default
    values). Honours ``auto_prune`` / ``retention_days`` /
    ``vacuum_after_prune`` / ``min_vacuum_interval_days`` /
    ``min_interval_hours``, and delegates to the DB. Never raises —
    maintenance must never block interactive startup.
    """
    if session_db is None:
        return
    try:
        from hermes_cli.config import load_config as _load_full_config
        from hermes_constants import get_hermes_home as _get_hermes_home
        _hermes_home_maint = _get_hermes_home()

        # One-time prune of empty TUI ghost sessions.
        try:
            if not session_db.get_meta("ghost_session_prune_v1"):
                pruned = session_db.prune_empty_ghost_sessions(
                    sessions_dir=_hermes_home_maint / "sessions"
                )
                session_db.set_meta("ghost_session_prune_v1", "1")
                if pruned:
                    logger.info("Pruned %d empty TUI ghost sessions", pruned)
        except Exception as _prune_exc:
            logger.debug("Ghost session prune skipped: %s", _prune_exc)

        # One-time finalize of orphaned compression continuations (#20001).
        try:
            if not session_db.get_meta("orphaned_compression_finalize_v1"):
                finalized = session_db.finalize_orphaned_compression_sessions()
                session_db.set_meta("orphaned_compression_finalize_v1", "1")
                if finalized:
                    logger.info(
                        "Finalized %d orphaned compression sessions", finalized
                    )
        except Exception as _finalize_exc:
            logger.debug("Orphan compression finalize skipped: %s", _finalize_exc)

        cfg = (_load_full_config().get("sessions") or {})

        # Auto-archive (soft-hide stale sessions) is independent of the
        # destructive auto_prune sweep — run it first, before prune's early
        # return, so enabling one doesn't require the other.
        if cfg.get("auto_archive", False):
            session_db.maybe_auto_archive(
                idle_days=float(cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            )

        if not cfg.get("auto_prune", False):
            return
        session_db.maybe_auto_prune_and_vacuum(
            retention_days=int(cfg.get("retention_days", 90)),
            min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            min_vacuum_interval_days=int(cfg.get("min_vacuum_interval_days", 30)),
            vacuum=bool(cfg.get("vacuum_after_prune", True)),
            sessions_dir=_hermes_home_maint / "sessions",
        )
    except Exception as exc:
        logger.debug("state.db auto-maintenance skipped: %s", exc)


def _run_checkpoint_auto_maintenance() -> None:
    """Call ``checkpoint_manager.maybe_auto_prune_checkpoints`` using current config.

    Reads the ``checkpoints:`` section from config.yaml via
    :func:`hermes_cli.config.load_config`. Honours ``auto_prune`` /
    ``retention_days`` / ``delete_orphans`` / ``min_interval_hours``.
    Never raises — maintenance must never block interactive startup.
    """
    try:
        from hermes_cli.config import load_config as _load_full_config
        cfg = (_load_full_config().get("checkpoints") or {})
        if not cfg.get("auto_prune", False):
            return
        from tools.checkpoint_manager import maybe_auto_prune_checkpoints
        # delete_orphans is intentionally never honoured here: a missing
        # workdir at startup is ambiguous (deleted project vs. an unmounted
        # external volume / network share / VPN not yet up) and this sweep
        # runs unattended. Orphan cleanup is only ever done via the explicit
        # `hermes checkpoints prune` command, which the user has to invoke.
        maybe_auto_prune_checkpoints(
            retention_days=int(cfg.get("retention_days", 7)),
            min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            delete_orphans=False,
            max_total_size_mb=int(cfg.get("max_total_size_mb", 500)),
        )
    except Exception as exc:
        logger.debug("checkpoint auto-maintenance skipped: %s", exc)


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove stale worktrees and orphaned branches on startup.

    Covers EVERY directory under ``.worktrees/`` except kanban task trees
    (``t_<hex>`` — owned by the kanban dispatcher's own gc). Scratch trees
    created by ``hermes -w`` (``hermes-*``) age out fast; named trees created
    manually for salvage/review lanes age out on a slower schedule:

    - ``hermes-*``: skip under 24h; reap 24h+ when clean and merged/pushed;
      72h+ is the aggressive tier (still never deletes real work).
    - named trees: same logic at 3x the timeline (72h soft / 9d hard).

    Work-preservation guards (all tiers, any age):
    - uncommitted changes (dirty) — never removed;
    - unpushed commits — never removed, UNLESS every local-only commit is
      patch-equivalent to a commit already on upstream (``git cherry``): the
      squash-merged-PR case, which is the dominant ``.worktrees/`` leak since
      those commits stay unreachable from ``refs/remotes/*`` forever;
    - pushed-branch tier: when the tree's branch head EXACTLY matches what
      origin holds (one ``git ls-remote`` per sweep, lazily), the checkout is
      redundant — typically an open-PR lane. The TREE is reaped but its
      BRANCH ref is kept (and shielded from the orphaned-branch pass), so
      the lane is one ``git worktree add`` away from restored. Needed because
      managed installs fetch with a single-branch refspec, leaving pushed PR
      branches with no ``refs/remotes/*`` entry — they read as "unpushed"
      forever and were the dominant survivor class (Aug 2026: 24 of 33
      preserved trees, ~18GB).

    Lock handling (orthogonal to age): ``hermes -w`` locks each worktree with
    reason ``hermes pid=<pid>`` so a concurrent hermes process leaves an in-use
    worktree alone. A *live*-locked worktree is skipped at any age; a
    *dead*-locked one (owning pid gone — a crashed session) is unlocked first
    so ``git worktree remove --force`` can actually reap it, otherwise those
    leftovers accumulate forever (``remove --force`` refuses a locked tree).

    Branch deletion is gated on ``git worktree remove`` succeeding, so a failed
    removal never orphans the branch (which would drop easy reachability of any
    commits still in the worktree).

    Preserved-work visibility: trees skipped for unpushed/dirty reasons that
    are older than 7 days are listed in a single WARNING so real in-flight
    work can't rot silently.

    Also prunes orphaned ``hermes/*`` and ``pr-*`` local branches that
    have no corresponding worktree.

    Performance: this runs on the startup path of every ``hermes -w`` session,
    and each candidate tree costs several git subprocesses (the ``git cherry``
    patch-equivalence probe dominates at ~0.2-1.0s on a large repo). With
    dozens of accumulated worktrees the serial version added ~11-18s of latency
    before the banner. Two changes keep the decisions byte-identical while
    removing nearly all of that:

    1. The read-only classification of each tree (dirty / unpushed / merged /
       lock state) is independent per tree, so it runs on a thread pool. Only
       the mutating phase (unlock, remove, branch -D) stays serial and ordered.
    2. ``git cherry`` verdicts are memoized on disk keyed by the exact
       ``(base_sha, head_sha)`` range they were computed from, so a tree
       preserved for unpushed work is not re-diff-hashed on every subsequent
       startup.
    """
    import re
    import subprocess
    import time

    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        _prune_orphaned_branches(repo_root)
        return

    # A shallow clone (the installer's default `--depth 1`) disconnects old
    # worktree HEADs from current origin/main, so the unpushed-commits guard
    # misclassifies every aged worktree as unpushed work and preserves it
    # forever. Deepen once — bloblessly, in this background thread — so all
    # history verdicts below (and the session-exit cleanup) become correct.
    # Fail-soft: offline, we just keep today's preserve-everything behavior.
    if _repo_is_shallow(repo_root):
        _deepen_shallow_repo(repo_root)

    now = time.time()
    stale_work_cutoff = now - (7 * 24 * 3600)
    preserved_stale: list = []
    # Kanban task worktrees (<repo>/.worktrees/t_<hex>) have their own
    # dispatcher-driven lifecycle (hermes kanban gc) — never touch them here.
    kanban_re = re.compile(r"^t_[0-9a-f]+$")

    # ── Phase 1: age filter (no subprocesses) ───────────────────────────────
    # Cheap stat-only pass so the thread pool below is sized to the trees that
    # actually need git work, not to everything on disk.
    candidates: list = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir() or kanban_re.match(entry.name):
            continue

        # Scratch trees (hermes-*) age out on the default schedule; named
        # trees (salvage/review lanes someone created deliberately) get 3x.
        scratch = entry.name.startswith("hermes-")
        tier_hours = max_age_hours if scratch else max_age_hours * 3
        soft_cutoff = now - (tier_hours * 3600)
        hard_cutoff = now - (tier_hours * 3 * 3600)

        try:
            mtime = entry.stat().st_mtime
            if mtime > soft_cutoff:
                continue  # Too recent — skip
        except Exception:
            continue

        candidates.append((entry, mtime, mtime <= hard_cutoff))

    if not candidates:
        _prune_orphaned_branches(repo_root)
        return

    # ── Phase 2: classify in parallel (read-only git queries) ───────────────
    # Every check here is a read-only git query against a distinct worktree, so
    # they are safe to run concurrently (git takes no repo-wide lock for these,
    # and each has its own index). Verdicts are collected and applied serially
    # below so removal order and log output stay deterministic.
    merge_cache = _load_worktree_merge_cache()
    cache_size_before = len(merge_cache)
    cache_lock = threading.Lock()

    # Lazy, once-per-sweep ls-remote: answers "is this branch pushed as-is?"
    # for every tree. Only paid when some tree actually reaches the
    # pushed-tier check (the TUI path runs this pruner synchronously, so an
    # unconditional network call would tax every launch; offline it costs
    # one bounded timeout at most, and None degrades verdicts to preserve).
    _remote_heads_memo: dict = {}
    _remote_heads_lock = threading.Lock()

    def _get_remote_heads():
        with _remote_heads_lock:
            if "heads" not in _remote_heads_memo:
                _remote_heads_memo["heads"] = _fetch_remote_branch_heads(
                    repo_root, timeout=10
                )
            return _remote_heads_memo["heads"]

    def _classify(item):
        entry, mtime, force = item
        # Never delete real work, regardless of age or tier. Uncommitted
        # changes and unpushed commits may be a crashed session's in-flight
        # work; only clean, fully-merged/pushed trees (the scratch trees that
        # actually cause .worktrees/ bloat) are ever reaped.
        if _worktree_is_dirty(str(entry), timeout=5):
            return (entry, mtime, force, "dirty", None)
        keep_branch = False
        if _worktree_has_unpushed_commits(str(entry), timeout=5):
            # Squash-merge escape hatch: commits unreachable from any remote
            # ref but patch-equivalent to upstream commits are merged work,
            # not unpushed work.
            with cache_lock:
                snapshot = dict(merge_cache)
            merged = _worktree_commits_all_merged_upstream(
                str(entry), timeout=30, cache=snapshot
            )
            if not merged:
                # Rebase-merge escape hatch: conflict resolution or follow-up
                # commits change the patch-id, so cherry misses them — but
                # GitHub knows the PR merged. Authoritative and cheap (~0.3s,
                # memoized on (branch, head_sha) so it's paid once per tree).
                merged = _worktree_branch_pr_merged(
                    str(entry), timeout=15, cache=snapshot
                )
            with cache_lock:
                merge_cache.update(snapshot)
            if not merged:
                # Pushed-branch tier: single-branch fetch refspecs (the
                # managed-install default) leave pushed PR branches with no
                # refs/remotes/* entry, so they read as "unpushed" forever
                # even though every byte is on origin. When the local head
                # EXACTLY matches the remote branch, the checkout is
                # redundant: reap the tree, keep the branch ref so the lane
                # is one `git worktree add` from restored.
                if _worktree_branch_pushed_exact(str(entry), _get_remote_heads(), timeout=10):
                    keep_branch = True
                else:
                    return (entry, mtime, force, "unpushed", None)

        # Respect git-native session locks. A lock owned by a still-running
        # hermes process means the worktree is actively in use — never touch
        # it. A lock whose owning pid is gone is a crashed session's leftover:
        # unlock it so `git worktree remove --force` (single -f) can reap it,
        # otherwise dead-locked worktrees pile up indefinitely.
        lock_state = _worktree_lock_is_live(repo_root, str(entry), timeout=5)
        if lock_state == "live":
            return (entry, mtime, force, "locked-live", None)
        return (entry, mtime, force,
                "reap-keep-branch" if keep_branch else "reap", lock_state)

    # Bounded pool: enough to hide git's per-process startup latency without
    # spawning dozens of concurrent git processes on a small machine.
    workers = max(1, min(8, (os.cpu_count() or 4), len(candidates)))
    try:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="hermes-wt-prune"
            ) as pool:
                verdicts = list(pool.map(_classify, candidates))
        else:
            verdicts = [_classify(c) for c in candidates]
    except Exception as e:
        # Never let a pool failure block startup — fall back to serial.
        logger.debug("Parallel worktree classification failed (%s); serial", e)
        verdicts = [_classify(c) for c in candidates]

    if len(merge_cache) != cache_size_before:
        _save_worktree_merge_cache(merge_cache)

    # ── Phase 3: mutate serially (unlock / remove / branch -D) ──────────────
    # Branch refs deliberately preserved by the pushed tier — must survive
    # the orphaned-branch pass even though their worktree is now gone.
    kept_branches: set = set()
    for entry, mtime, force, verdict, lock_state in verdicts:
        if verdict == "dirty":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (uncommitted changes)")
            continue
        if verdict == "unpushed":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (unpushed commits)")
            continue
        if verdict == "locked-live":
            logger.debug("Skipping live-locked worktree: %s", entry.name)
            continue

        if lock_state == "dead":
            try:
                subprocess.run(
                    ["git", "worktree", "unlock", str(entry)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
                )
            except Exception as e:
                logger.debug("Failed to unlock dead worktree %s: %s", entry.name, e)

        # Safe to remove
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()

            remove_result = subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, cwd=repo_root,
            )
            if remove_result.returncode != 0:
                # Removal failed — keep the branch so any commits stay
                # reachable rather than orphaning it.
                logger.debug(
                    "Failed to remove worktree %s: %s",
                    entry.name, remove_result.stderr.strip(),
                )
                continue
            if branch and verdict == "reap-keep-branch":
                # Pushed-tier trees keep their branch ref: the branch IS the
                # work (an open PR's local anchor) — only the checkout was
                # redundant. Also shield it from the orphaned-branch pass
                # below, which would otherwise delete hermes/*- and pr-*
                # named refs the moment their worktree is gone.
                kept_branches.add(branch)
            elif branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
                )
            logger.debug("Pruned stale worktree: %s (force=%s)", entry.name, force)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)

    if preserved_stale:
        logger.warning(
            "Preserving %d worktree(s) older than 7 days with unmerged work "
            "(run `hermes worktree prune` to review and reclaim): %s",
            len(preserved_stale), ", ".join(sorted(preserved_stale)),
        )

    _prune_orphaned_branches(repo_root, protect=kept_branches)

    # Escalation notice: the startup pass is deliberately conservative, so
    # installs accumulate preserved trees it can never reclaim. Once the
    # footprint is clearly a problem (many trees or multi-GB), say so once
    # per launch and name the attended reclaim command — silence here is how
    # boxes reach 15GB+ of .worktrees/ without anyone noticing.
    try:
        from hermes_cli.worktree_gc import worktrees_summary

        count, size_mb = worktrees_summary(repo_root)
        if count >= 10 or (size_mb or 0) >= 5120:
            size_txt = f"{size_mb / 1024:.1f}GB" if size_mb else "unknown size"
            logger.warning(
                ".worktrees/ holds %d tree(s) (%s) — run `hermes worktree list` "
                "to audit and `hermes worktree prune` to reclaim safely.",
                count, size_txt,
            )
    except Exception:
        pass


def _prune_orphaned_branches(repo_root: str, protect: Optional[set] = None) -> None:
    """Delete local ``hermes/hermes-*`` and ``pr-*`` branches with no worktree.

    These are auto-generated by ``hermes -w`` sessions and PR review
    workflows respectively.  Once their worktree is gone they serve no
    purpose and just accumulate.

    ``protect``: branch names to never delete this pass — the pushed-tier
    reap above removes a tree while deliberately keeping its branch (an open
    PR's local anchor), and some of those carry ``hermes/hermes-*`` names
    this sweep would otherwise collect immediately.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        if result.returncode != 0:
            return
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
    except Exception:
        return

    # Collect branches that are actively checked out in a worktree
    active_branches: set = set()
    try:
        wt_result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        for line in wt_result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())
    except Exception:
        return  # Can't determine active branches — bail

    # Also protect the currently checked-out branch and main
    try:
        head_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, cwd=repo_root,
        )
        current = head_result.stdout.strip()
        if current:
            active_branches.add(current)
    except Exception:
        pass
    active_branches.add("main")

    orphaned = [
        b for b in all_branches
        if b not in active_branches
        and b not in (protect or ())
        and (b.startswith("hermes/hermes-") or b.startswith("pr-"))
    ]

    if not orphaned:
        return

    # Delete in batches
    for i in range(0, len(orphaned), 50):
        batch = orphaned[i:i + 50]
        try:
            subprocess.run(
                ["git", "branch", "-D"] + batch,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=repo_root,
            )
        except Exception as e:
            logger.debug("Failed to prune orphaned branches: %s", e)

    logger.debug("Pruned %d orphaned branches", len(orphaned))

# ============================================================================
# ASCII Art & Branding
# ============================================================================

# Color palette (hex colors for Rich markup):
# - Gold: #FFD700 (headers, highlights)
# - Amber: #FFBF00 (secondary highlights)
# - Bronze: #CD7F32 (tertiary elements)
# - Light: #FFF8DC (text)
# - Dim: #B8860B (muted text)

# ANSI building blocks for conversation display
_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold — fallback
_BOLD = "\033[1m"
_RST = "\033[0m"
_STREAM_PAD = ""  # No indent for streamed response text — leading whitespace pollutes
# terminal copy/paste (every selected line carried 4 spaces).  Matches the
# response Panel's flush-left padding.
_STREAM_PARTIAL_PREVIEW_LEN = 60  # tail of an unfinished logical line mirrored
# into the spinner while streaming (TTFT perception without hard-wrapping)


def _hex_to_ansi(hex_color: str, *, bold: bool = False) -> str:
    """Convert a hex color like '#268bd2' to a true-color ANSI escape.

    Auto-remaps known dark-mode-tuned colors to readable light-mode
    equivalents when running on a light terminal (see
    _maybe_remap_for_light_mode + _LIGHT_MODE_REMAP).
    """
    hex_color = _maybe_remap_for_light_mode(hex_color)
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        prefix = "1;" if bold else ""
        return f"\033[{prefix}38;2;{r};{g};{b}m"
    except (ValueError, IndexError):
        return _ACCENT_ANSI_DEFAULT if bold else "\033[38;2;184;134;11m"


# ────────────────────────────────────────────────────────────────────────
# Light/dark terminal mode detection.
#
# Mirrors ui-tui/src/theme.ts detectLightMode().  Used to decide whether
# to remap "near-white" skin colors (e.g. #FFF8DC banner_text, #B8860B
# banner_dim) to darker equivalents that are readable on a light
# Terminal.app / iTerm2 background.
#
# Detection priority:
#   1. HERMES_LIGHT / HERMES_TUI_LIGHT env (true/false) — explicit override
#   2. HERMES_TUI_THEME=light|dark — explicit theme
#   3. HERMES_TUI_BACKGROUND=#RRGGBB — explicit bg hint
#   4. COLORFGBG env (set by xterm/Konsole/urxvt) — bg slot 7/15 = light
#   5. OSC 11 query (\x1b]11;?\x1b\\) — ask the terminal directly
#   6. Default: assume dark (matches the legacy Hermes assumption)
#
# Cached after first call so we don't query the terminal repeatedly.
_LIGHT_MODE_CACHE: bool | None = None
_TRUE_RE = re.compile(r"^(1|true|on|yes|y)$")
_FALSE_RE = re.compile(r"^(0|false|off|no|n)$")
_LIGHT_DEFAULT_TERM_PROGRAMS = frozenset()  # Apple_Terminal doesn't reliably indicate; require explicit


def _luminance_from_hex(hex_str: str) -> float | None:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    # Rec.709 luma
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


_DA1_REPLY_RE = re.compile(rb"\x1b\[\?[0-9;]*c")


def _query_osc11_background() -> str | None:
    """Ask the terminal for its background color via OSC 11.

    Most modern terminals reply with \x1b]11;rgb:RRRR/GGGG/BBBB\x1b\\
    within a few ms.  Returns "#RRGGBB" or None on timeout / non-tty.

    The OSC 11 query is fenced with a DA1 sentinel (\x1b[c) — the same
    pattern the Ink TUI's TerminalQuerier uses.  Terminals answer queries
    in order and virtually every terminal answers DA1, so seeing the DA1
    reply proves the terminal already ignored our OSC 11 (multiplexers
    like herdr answer DA1 in <1ms while swallowing OSC 11).  Without the
    fence we can only wait out a blind timeout, and a reply that arrives
    AFTER we stop listening leaks into prompt_toolkit's stdin as typed
    text — the "gibberish ANSI characters" seen inside terminal managers
    that relay color queries slowly (herdr, WSL bridges, some tmux
    setups).

    Skipped over SSH: the round-trip routinely exceeds our budget, so a
    late reply lands after prompt_toolkit has grabbed the tty — its payload
    leaks in as typed text and the BEL terminator reads as Ctrl+G (open
    editor), trapping the user in a stray editor. Remote sessions fall back
    to COLORFGBG / env hints / the dark default instead.

    After the main read + TCSAFLUSH, a short drain window (50 ms) catches
    late-arriving bytes that slipped past the flush — a race observed on VPS
    and container terminals under load (#40250).
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return None
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return None
    try:
        try:
            tty.setcbreak(fd)
        except Exception:
            return None
        try:
            # OSC 11 query + DA1 sentinel fence, in one write so no
            # reordering is possible.
            sys.stdout.write("\x1b]11;?\x1b\\\x1b[c")
            sys.stdout.flush()
        except Exception:
            return None
        # Read until the DA1 fence closes — proof the terminal has processed
        # everything up to and including our OSC 11, so nothing can arrive
        # late and leak into prompt_toolkit's stdin.  DA1 is answered by
        # effectively every terminal ever made (it predates color), and on
        # real terminals the fence closes in single-digit milliseconds
        # (herdr: <1ms, xterm/kitty/tmux: <5ms).  The 1s deadline is a
        # safety net for a hypothetical terminal that ignores DA1 — not a
        # window we ever expect to wait out.  A slow in-order relay that
        # delivers the OSC 11 reply at e.g. 400ms is handled correctly:
        # we keep listening until its DA1 reply follows, so the payload is
        # consumed here instead of leaking as typed input (the "gibberish
        # ANSI characters" seen inside terminal managers).
        import select
        deadline = time.monotonic() + 1.0
        buf = b""
        while time.monotonic() < deadline:
            r, _, _ = select.select([fd], [], [], deadline - time.monotonic())
            if not r:
                continue
            try:
                chunk = os.read(fd, 64)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if _DA1_REPLY_RE.search(buf):
                break
        # Parse: \x1b]11;rgb:RRRR/GGGG/BBBB\x1b\\
        m = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
        if not m:
            return None
        # Each component is 1-4 hex digits — normalize to 8-bit
        def norm(h: bytes) -> int:
            v = int(h, 16)
            # Scale to 0-255 based on hex length
            bits = len(h) * 4
            return (v * 255) // ((1 << bits) - 1) if bits else 0
        r, g, b = norm(m.group(1)), norm(m.group(2)), norm(m.group(3))
        return f"#{r:02X}{g:02X}{b:02X}"
    finally:
        # TCSAFLUSH discards any unread input as it restores the original
        # attributes — scrubs a slow/partial OSC 11 reply out of the tty
        # buffer before prompt_toolkit can read it as keystrokes.
        try:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        except Exception:
            pass
        # Race guard: on slow terminals (VPS, container, heavy load), the
        # OSC 11 reply can arrive *after* TCSAFLUSH completes.  Drain any
        # late bytes with a short post-flush window so they don't leak into
        # prompt_toolkit's input buffer as typed text.
        try:
            import select as _sel
            drain_deadline = time.monotonic() + 0.05
            while time.monotonic() < drain_deadline:
                r, _, _ = _sel.select([fd], [], [], drain_deadline - time.monotonic())
                if not r:
                    break
                late = os.read(fd, 64)
                if not late:
                    break
        except Exception:
            pass


def _heal_cooked_mode_drift(fd: int) -> bool:
    """Detect and heal cooked-mode termios drift on *fd* while prompt_toolkit
    expects raw mode.

    prompt_toolkit's ``run_in_terminal`` / ``in_terminal`` wraps every
    "print above the prompt" in a ``cooked_mode()`` context: it flips the
    tty back to cooked (ICANON/ECHO/ISIG), runs the function, then restores
    raw mode.  Hermes schedules those windows cross-thread constantly — the
    background self-review's ``💾`` summary, background process notification
    drains, curses pickers — and if a restore is ever lost (coroutine
    cancelled mid-window, racing chains, an external writer touching the
    shared tty), the terminal is left in cooked mode while the Application
    still believes it owns raw mode.  The kernel line-buffers every
    keystroke and the CLI appears to "stop taking input" even though the
    process is perfectly healthy (observed live: pts in ``icanon echo``
    while the event loop idled normally in ``ep_poll``).

    This helper is the last line of defense for that whole class: when the
    lflag has drifted back to cooked, re-apply prompt_toolkit's own raw-mode
    flag surgery (mirrors ``prompt_toolkit.input.vt100.raw_mode``) in place.
    Returns True when drift was detected and healed, False when the tty was
    already raw (or could not be inspected).

    POSIX-only by construction — callers must not invoke this on Windows
    (no termios; prompt_toolkit uses the win32 console API there instead).
    """
    try:
        import termios
        attrs = termios.tcgetattr(fd)
    except Exception:
        return False
    lflag = attrs[3]
    if not (lflag & (termios.ICANON | termios.ECHO)):
        return False  # still raw — nothing to do
    # Same surgery as prompt_toolkit.input.vt100.raw_mode._patch_lflag /
    # _patch_iflag, applied to the *current* attrs so any user settings
    # (speed, size-independent flags) are preserved.
    attrs[3] = lflag & ~(
        termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG
    )
    attrs[0] = attrs[0] & ~(
        termios.IXON
        | termios.IXOFF
        | termios.ICRNL
        | termios.INLCR
        | termios.IGNCR
    )
    # VMIN=1 so reads return per-byte (Solaris-derived systems default to 4;
    # prompt_toolkit sets this explicitly in raw_mode.__enter__).
    attrs[6][termios.VMIN] = 1
    try:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        return False
    return True


def _detect_light_mode() -> bool:
    global _LIGHT_MODE_CACHE
    if _LIGHT_MODE_CACHE is not None:
        return _LIGHT_MODE_CACHE
    result = False
    try:
        # 1. Explicit env override
        for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT"):
            v = (os.environ.get(var) or "").strip().lower()
            if _TRUE_RE.match(v):
                result = True
                _LIGHT_MODE_CACHE = result
                return result
            if _FALSE_RE.match(v):
                _LIGHT_MODE_CACHE = result
                return result
        # 2. Theme hint
        theme = (os.environ.get("HERMES_TUI_THEME") or "").strip().lower()
        if theme == "light":
            result = True
            _LIGHT_MODE_CACHE = result
            return result
        if theme == "dark":
            _LIGHT_MODE_CACHE = result
            return result
        # 3. Explicit bg hex
        bg_hint = os.environ.get("HERMES_TUI_BACKGROUND") or ""
        bg_lum = _luminance_from_hex(bg_hint)
        if bg_lum is not None:
            result = bg_lum >= 0.5
            _LIGHT_MODE_CACHE = result
            return result
        # 4. COLORFGBG (xterm/Konsole/urxvt)
        cfgbg = (os.environ.get("COLORFGBG") or "").strip()
        if cfgbg:
            last = cfgbg.split(";")[-1] if ";" in cfgbg else cfgbg
            if last.isdigit():
                bg = int(last)
                if bg in {7, 15}:
                    result = True
                    _LIGHT_MODE_CACHE = result
                    return result
                if 0 <= bg < 16:
                    _LIGHT_MODE_CACHE = result
                    return result
        # 5. OSC 11 query (best-effort, only when stdin/stdout are TTY)
        bg_color = _query_osc11_background()
        if bg_color:
            lum = _luminance_from_hex(bg_color)
            if lum is not None:
                result = lum >= 0.5
                _LIGHT_MODE_CACHE = result
                return result
        # 6. TERM_PROGRAM allow-list (currently empty)
        tp = (os.environ.get("TERM_PROGRAM") or "").strip()
        if tp in _LIGHT_DEFAULT_TERM_PROGRAMS:
            result = True
    except Exception:
        result = False
    _LIGHT_MODE_CACHE = result
    return result


# Light-mode equivalents of skin colors that are unreadable on cream
# Terminal.app backgrounds.  Used by _SkinAwareAnsi to remap colors
# at resolution time when light mode is detected.
#
# IMPORTANT: only remap colors that are used as STANDALONE foregrounds
# on the terminal's background.  Don't remap colors that are paired
# with a dark bg (e.g. status bar text on bg:#1a1a2e) — those would
# become invisible the OTHER direction (dark gray on dark navy).
_LIGHT_MODE_REMAP: dict[str, str] = {
    # Original (dark-mode) -> Light-mode replacement (darker, readable)
    "#FFF8DC": "#1A1A1A",   # cornsilk -> near-black
    "#FFD700": "#9A6B00",   # gold -> dark goldenrod (readable on cream)
    "#FFBF00": "#8A5A00",   # amber -> dark amber
    "#B8860B": "#5C4500",   # dark goldenrod -> deeper brown (more contrast)
    "#DAA520": "#6B4F00",   # goldenrod -> dark olive
    "#F1E6CF": "#1A1A1A",   # cream -> near-black
    "#c9d1d9": "#24292F",   # github-light fg
    "#EAF7FF": "#0F1B26",   # ice
    "#F5F5F5": "#1A1A1A",
    "#FFF0D4": "#1A1A1A",
    "#CD7F32": "#8A4F1A",   # bronze -> darker bronze
    "#FFEFB5": "#3A2A00",
    # NOTE: skipping #C0C0C0/#888888/#555555/#8B8682 — those are
    # status-bar foregrounds paired with dark navy bg, where dark
    # remap values would become invisible.
}


def _maybe_remap_for_light_mode(hex_color: str) -> str:
    """If we're in light mode, remap a dark-mode-tuned color to a
    higher-contrast equivalent.  No-op in dark mode."""
    if not _detect_light_mode():
        return hex_color
    if not hex_color or not hex_color.startswith("#"):
        return hex_color
    # Case-insensitive lookup
    upper = hex_color.upper()
    if upper in _LIGHT_MODE_REMAP_UPPER:
        return _LIGHT_MODE_REMAP_UPPER[upper]
    return hex_color


# Pre-uppercased lookup table for case-insensitive remapping
_LIGHT_MODE_REMAP_UPPER = {k.upper(): v for k, v in _LIGHT_MODE_REMAP.items()}


def _install_skin_light_mode_hook() -> None:
    """Wrap SkinConfig.get_color at import time so EVERY skin color read goes
    through the light-mode remap.  Idempotent."""
    try:
        from hermes_cli.skin_engine import SkinConfig  # type: ignore[import]
    except Exception:
        return
    if getattr(SkinConfig, "_hermes_light_mode_hook_installed", False):
        return
    _orig_get_color = SkinConfig.get_color

    def _wrapped_get_color(self, key, fallback=""):
        value = _orig_get_color(self, key, fallback)
        try:
            return _maybe_remap_for_light_mode(value)
        except Exception:
            return value

    SkinConfig.get_color = _wrapped_get_color  # type: ignore[method-assign]
    SkinConfig._hermes_light_mode_hook_installed = True  # type: ignore[attr-defined]


_install_skin_light_mode_hook()


# Prime the light-mode detection cache early (at module load) when
# we're running interactively so OSC 11 happens before pt grabs the
# tty.  Skip for non-tty contexts (subagents, gateway, tests).
try:
    if sys.stdin.isatty() and sys.stdout.isatty():
        _detect_light_mode()
except Exception:
    pass



class _SkinAwareAnsi:
    """Lazy ANSI escape that resolves from the skin engine on first use.

    Acts as a string in f-strings and concatenation.  Call ``.reset()`` to
    force re-resolution after a ``/skin`` switch.
    """

    def __init__(self, skin_key: str, fallback_hex: str = "#FFD700", *, bold: bool = False):
        self._skin_key = skin_key
        self._fallback_hex = fallback_hex
        self._bold = bold
        self._cached: str | None = None

    def __str__(self) -> str:
        if self._cached is None:
            try:
                from hermes_cli.skin_engine import get_active_skin
                self._cached = _hex_to_ansi(
                    get_active_skin().get_color(self._skin_key, self._fallback_hex),
                    bold=self._bold,
                )
            except Exception:
                self._cached = _hex_to_ansi(self._fallback_hex, bold=self._bold)
        return self._cached

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)

    def reset(self) -> None:
        """Clear cache so the next access re-reads the skin."""
        self._cached = None


_ACCENT = _SkinAwareAnsi("response_border", "#FFD700", bold=True)
# Use ANSI dim+italic attributes (\x1b[2;3m) instead of a hardcoded
# hex color so dim/thinking text inherits the terminal's default
# foreground color and stays readable in both light and dark
# Terminal.app modes.  Hardcoded skin colors like #B8860B
# (dark goldenrod) become invisible against light cream backgrounds.
_DIM = "\x1b[2;3m"


def _b(s: str) -> str:
    """Bold if stdout is a real TTY; plain text otherwise (slash-worker safe)."""
    import sys as _sys
    try:
        return f"\x1b[1m{s}\x1b[0m" if _sys.stdout.isatty() else str(s)
    except Exception:
        return str(s)


def _d(s: str) -> str:
    """Dim-italic if stdout is a real TTY; plain text otherwise."""
    import sys as _sys
    try:
        return f"\x1b[2;3m{s}\x1b[0m" if _sys.stdout.isatty() else str(s)
    except Exception:
        return str(s)


def _accent_hex() -> str:
    """Return the active skin accent color for legacy CLI output lines."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color("ui_accent", "#FFBF00")
    except Exception:
        return "#FFBF00"


def _rich_text_from_ansi(text: str) -> _RichText:
    """Safely render assistant/tool output that may contain ANSI escapes.

    Using Rich Text.from_ansi preserves literal bracketed text like
    ``[not markup]`` while still interpreting real ANSI color codes.
    """
    return _RichText.from_ansi(text or "")


def _strip_markdown_syntax(text: str) -> str:
    """Best-effort markdown marker removal for plain-text display."""
    plain = _rich_text_from_ansi(text or "").plain
    # Avoid stripping cron-style expressions like "* * * * *" as if they were
    # Markdown horizontal rules. CommonMark treats three or more "*" as an HR,
    # but in Hermes output it's common to display cron schedules verbatim.
    #
    # Keep the behavior for "-" / "_" HR markers, and only strip "*" HR lines
    # when there are exactly 3 asterisks (with optional whitespace).
    plain = re.sub(r"^\s{0,3}(?:[-_]\s*){3,}$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}(?:\*\s*){3}\s*$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", plain, flags=re.MULTILINE)
    # Preserve blockquotes, lists, and checkboxes because they carry structure.
    plain = re.sub(r"(```+|~~~+)", "", plain)
    plain = re.sub(r"`([^`]*)`", r"\1", plain)
    plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)___([^_]+)___(?!\w)", r"\1", plain)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", plain)
    # Only strip `*emphasis*` markers when the inner text is non-whitespace.
    # This avoids corrupting cron expressions like "* * * * *".
    plain = re.sub(r"\*([^\s*][^*]*?[^\s*])\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", plain)
    plain = re.sub(r"~~([^~]+)~~", r"\1", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip("\n")


_WINDOWS_PATH_WITH_DOT_SEGMENT_RE = re.compile(
    r"(?i)(?:\b[a-z]:\\|\\\\)[^\s`]*\\\.[^\s`]*"
)


def _preserve_windows_dot_segments_for_markdown(text: str) -> str:
    r"""Keep Windows path separators before hidden directories in Markdown.

    CommonMark treats ``\.`` as an escaped literal dot, so Rich Markdown would
    render ``D:\repo\.ai`` as ``D:\repo.ai``.  Doubling only that separator
    inside Windows path-looking tokens preserves the path without changing
    ordinary markdown escapes like ``1\. not a list``.
    """
    if "\\." not in text:
        return text

    def _protect(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)\\(?=\.)", r"\\\\", match.group(0))

    return _WINDOWS_PATH_WITH_DOT_SEGMENT_RE.sub(_protect, text)


def _terminal_width_for_streaming() -> int:
    """Display cells available inside the streamed response box.

    The streaming path prefixes every line with ``_STREAM_PAD`` (now
    empty — flush-left so copy/paste stays clean) inside an open
    response panel.  The realigner uses this number as its budget when
    deciding whether to keep a horizontal table or fall back to
    vertical key-value rendering.  We subtract a small safety margin
    so terminal-resize races don't push a borderline table into
    mid-cell soft-wrap.
    """

    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    return max(20, cols - len(_STREAM_PAD) - 2)


def _render_final_assistant_content(text: str, mode: str = "render"):
    """Render final assistant content as markdown, stripped text, or raw text."""
    from rich.markdown import Markdown

    # Estimate the cells available to the rendered table.  The Panel
    # used by the background-task / final-response path renders
    # flush-left (no horizontal padding — leading spaces pollute
    # terminal copy/paste) with 1 cell of border on each side.
    # Subtract a small safety margin so resize races don't push a
    # borderline table into soft-wrap.
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    panel_width = max(20, cols - 4)

    normalized_mode = str(mode or "render").strip().lower()
    if normalized_mode == "strip":
        # Strip first — inline markdown inside cells (`code`, **bold**, ~~strike~~)
        # changes cell display width — then re-align so the column padding
        # reflects the final visible text, not the marker-decorated source.
        return _RichText(
            realign_markdown_tables(_strip_markdown_syntax(text), panel_width)
        )
    if normalized_mode == "raw":
        return _rich_text_from_ansi(text or "")

    # `render` mode: Rich's Markdown renderer handles CJK width via wcwidth
    # internally, so a pre-pass through realign_markdown_tables would just
    # rewrite already-correct padding.  But on the way in we still want to
    # normalise model-emitted under-padded tables so that mid-render fallbacks
    # (narrow panels, etc.) at least see consistent input.
    plain = _rich_text_from_ansi(text or "").plain
    plain = _preserve_windows_dot_segments_for_markdown(plain)
    plain = realign_markdown_tables(plain, panel_width)
    return Markdown(plain)


def _post_stream_transform_output(response: str, result: dict | None) -> str:
    """Return text that still needs display after a streamed response transform.

    A transform hook is allowed to replace the final response, not merely append
    to it. When the transformed text retains the streamed response as a prefix,
    printing only its suffix avoids duplicating the already-rendered body. A
    replacement has no safe suffix, so deliberately print the complete final
    response rather than silently dropping it.
    """
    if not result or not result.get("response_transformed"):
        return ""

    original = result.get("pre_transform_response") or ""
    if original and response.startswith(original):
        return response[len(original):]

    return f"\n[Response transformed after streaming]\n{response}"


_OUTPUT_HISTORY_ENABLED = True
_OUTPUT_HISTORY_REPLAYING = False
_OUTPUT_HISTORY_SUPPRESSED = False
_OUTPUT_HISTORY_MAX_LINES = 200
_OUTPUT_HISTORY = deque(maxlen=_OUTPUT_HISTORY_MAX_LINES)


def _coerce_output_history_limit(value) -> int:
    try:
        return max(10, int(value))
    except (TypeError, ValueError):
        return 200


def _configure_output_history(enabled: bool, max_lines=200) -> None:
    """Configure recent CLI output replayed after terminal redraws."""
    global _OUTPUT_HISTORY_ENABLED, _OUTPUT_HISTORY_MAX_LINES, _OUTPUT_HISTORY
    _OUTPUT_HISTORY_ENABLED = bool(enabled)
    _OUTPUT_HISTORY_MAX_LINES = _coerce_output_history_limit(max_lines)
    _OUTPUT_HISTORY = deque(maxlen=_OUTPUT_HISTORY_MAX_LINES)


def _clear_output_history() -> None:
    _OUTPUT_HISTORY.clear()


@contextmanager
def _suspend_output_history():
    global _OUTPUT_HISTORY_SUPPRESSED
    old_value = _OUTPUT_HISTORY_SUPPRESSED
    _OUTPUT_HISTORY_SUPPRESSED = True
    try:
        yield
    finally:
        _OUTPUT_HISTORY_SUPPRESSED = old_value


def _record_output_history_entry(entry) -> None:
    if not _OUTPUT_HISTORY_ENABLED or _OUTPUT_HISTORY_REPLAYING or _OUTPUT_HISTORY_SUPPRESSED:
        return
    _OUTPUT_HISTORY.append(entry)


def _record_output_history(text: str) -> None:
    if not _OUTPUT_HISTORY_ENABLED or _OUTPUT_HISTORY_REPLAYING or _OUTPUT_HISTORY_SUPPRESSED:
        return
    normalized = str(text).replace("\r", "").rstrip("\n")
    if not normalized:
        return
    for line in normalized.splitlines():
        _record_output_history_entry(line)


def _replay_output_history() -> None:
    """Repaint recent output above the prompt after a full screen clear."""
    global _OUTPUT_HISTORY_REPLAYING
    if not _OUTPUT_HISTORY_ENABLED or not _OUTPUT_HISTORY:
        return
    _OUTPUT_HISTORY_REPLAYING = True
    try:
        rendered_lines = []
        for entry in tuple(_OUTPUT_HISTORY):
            if callable(entry):
                try:
                    lines = entry()
                except Exception:
                    continue
                if isinstance(lines, str):
                    lines = lines.splitlines()
            else:
                lines = [entry]
            rendered_lines.extend(str(line) for line in lines)
        if rendered_lines:
            # Replay after resize can contain hundreds of history lines. A
            # per-line prompt_toolkit print forces one synchronous terminal I/O
            # and redraw cycle per line, which users perceive as a waterfall of
            # old output. Keep the existing history contents unchanged, but
            # emit the replay as one ANSI payload so resize recovery does a
            # single prompt_toolkit print/redraw.
            _pt_print(_PT_ANSI("\n".join(rendered_lines)))
    except Exception:
        pass
    finally:
        _OUTPUT_HISTORY_REPLAYING = False


def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.

    When called from a background thread while a prompt_toolkit
    ``Application`` is running (the common case for the self-improvement
    background review's ``💾 …`` summary, curator summaries, and other
    bg-thread emissions), a direct ``_pt_print`` races with the input
    area's redraw and the line can end up visually buried behind the
    prompt.  Route those cases through ``run_in_terminal`` via
    ``loop.call_soon_threadsafe``, which pauses the input area, prints
    the line above it, and redraws the prompt cleanly.
    """
    _record_output_history(text)

    try:
        from prompt_toolkit.application import get_app_or_none, run_in_terminal
    except Exception:
        _pt_print(_PT_ANSI(text))
        return

    app = None
    try:
        app = get_app_or_none()
    except Exception:
        app = None

    # No active app, or we're already on the app's main thread: the
    # direct prompt_toolkit print is safe and matches existing behavior
    # (spinner frames, streamed tokens, tool activity prefixes, …).
    if app is None or not getattr(app, "_is_running", False):
        try:
            _pt_print(_PT_ANSI(text))
        except Exception:
            # Fallback when stdout is not a real console (e.g. subprocess
            # worker logging to a file). prompt_toolkit raises
            # NoConsoleScreenBufferError (Windows) or OSError (other).
            try:
                print(text)
            except Exception:
                pass
        return

    try:
        loop = app.loop  # type: ignore[attr-defined]
    except Exception:
        loop = None
    if loop is None:
        _pt_print(_PT_ANSI(text))
        return

    import asyncio as _asyncio
    try:
        # Use get_running_loop() instead of get_event_loop() to avoid the
        # DeprecationWarning / RuntimeWarning emitted by Python 3.10+ when
        # get_event_loop() is called from a thread that has no current event
        # loop set (e.g. the process_loop background thread).  Fixes #19285.
        current_loop = _asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    except Exception:
        current_loop = None
    # Same thread as the app's loop → safe to print directly.
    if current_loop is loop and loop.is_running():
        _pt_print(_PT_ANSI(text))
        return

    # Cross-thread emission: ask the app's event loop to schedule a
    # ``run_in_terminal`` that wraps ``_pt_print``.  This hides the
    # prompt, prints, and redraws.  Fire-and-forget — if scheduling
    # fails we fall back to a direct print so the line isn't lost.
    def _schedule():
        # run_in_terminal() may return either:
        #   • a coroutine / Future (prompt_toolkit ≥ 3.0) — must be scheduled
        #     via ensure_future so the coroutine is actually awaited; calling
        #     it bare would leave it unawaited and silently drop the output
        #     (fixes #23185 Bug A).
        #   • None (some mocks / older PT builds) — just call the inner
        #     function directly since PT already executed it synchronously.
        # Do NOT fall back to a bare _pt_print when ensure_future raises,
        # because run_in_terminal already invoked the lambda in that case
        # (the mock path), which would double-print the line.
        try:
            import asyncio as _aio
            import inspect as _inspect
            coro = run_in_terminal(lambda: _pt_print(_PT_ANSI(text)))
            if coro is not None and (_inspect.isawaitable(coro) or _inspect.iscoroutine(coro)):
                _aio.ensure_future(coro)
            # else: run_in_terminal ran the lambda synchronously; nothing more
            # to do (double-scheduling would print twice).
        except Exception:
            pass  # best-effort; the line may already have been printed

    try:
        loop.call_soon_threadsafe(_schedule)
    except Exception:
        try:
            _pt_print(_PT_ANSI(text))
        except Exception:
            pass


def _prepend_note_to_message(message, note: str):
    """Prepend a one-shot system-style note to a user message.

    ``message`` is normally a plain string, but when the user attaches an image
    to a vision-capable model it becomes a list of OpenAI-style content parts
    (text + ``image_url`` blocks). Naively doing ``note + "\\n\\n" + message``
    then raises ``TypeError: can only concatenate str (not "list") to str`` —
    e.g. running ``/model ...`` (which queues a model-switch note) and then
    sending a pasted image in the same turn.

    Returns the message with ``note`` prepended:
      * ``str``  → ``f"{note}\\n\\n{message}"`` (just ``note`` when empty)
      * ``list`` → note folded into the first text part, or inserted as a new
        leading ``{"type": "text"}`` part when there is no text part.
    Unknown shapes are returned unchanged (fail-open).
    """
    note = str(note or "").strip()
    if not note:
        return message
    if isinstance(message, str):
        return f"{note}\n\n{message}" if message else note
    if isinstance(message, list):
        parts = list(message)
        for i, part in enumerate(parts):
            if isinstance(part, dict) and part.get("type") == "text":
                merged = dict(part)
                text = merged.get("text", "")
                merged["text"] = f"{note}\n\n{text}" if text else note
                parts[i] = merged
                return parts
        # No text part (image-only) — insert the note as a leading text block.
        return [{"type": "text", "text": note}, *parts]
    return message


def _cli_visible_print(text: str = "") -> None:
    """Print normally unless prompt_toolkit owns the live terminal.

    Bare ``print()`` output is swallowed by ``patch_stdout`` while an
    interactive ``Application`` is running, so ``/sessions`` and ``/history``
    would render nothing. Route through ``_cprint`` (prompt_toolkit-native)
    in that case, and fall back to ``print`` otherwise.
    """
    try:
        from prompt_toolkit.application import get_app_or_none
        app = get_app_or_none()
    except Exception:
        app = None

    if app is not None and getattr(app, "_is_running", False):
        _cprint(text)
    else:
        print(text)


# ---------------------------------------------------------------------------
# File-drop / local attachment detection — extracted as pure helpers for tests.
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.bmp', '.tiff', '.tif', '.svg', '.ico',
})



def _termux_example_image_path(filename: str = "cat.png") -> str:
    """Return a realistic example media path for the current Termux setup."""
    candidates = [
        os.path.expanduser("~/storage/shared"),
        "/sdcard",
        "/storage/emulated/0",
        "/storage/self/primary",
    ]
    # Termux/Android roots are POSIX paths — join with literal forward
    # slashes so the hint stays correct even when this renders on Windows.
    for root in candidates:
        if os.path.isdir(root):
            return f"{root}/Pictures/{filename}"
    return f"~/storage/shared/Pictures/{filename}"


def _split_path_input(raw: str) -> tuple[str, str]:
    r"""Split a leading file path token from trailing free-form text.

    Supports quoted paths and backslash-escaped spaces so callers can accept
    inputs like:
      /tmp/pic.png describe this
      ~/storage/shared/My\ Photos/cat.png what is this?
      "/storage/emulated/0/DCIM/Camera/cat 1.png" summarize
    """
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == '\\' and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                token = raw[1:pos]
                remainder = raw[pos + 1 :].strip()
                return token, remainder
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == '\\' and pos + 1 < len(raw) and raw[pos + 1] == ' ':
            pos += 2
        elif ch == ' ':
            break
        else:
            pos += 1

    token = raw[:pos].replace('\\ ', ' ')
    remainder = raw[pos:].strip()
    return token, remainder


def _resolve_attachment_path(raw_path: str) -> Path | None:
    """Resolve a user-supplied local attachment path.

    Accepts quoted or unquoted paths, expands ``~`` and env vars, and resolves
    relative paths from ``TERMINAL_CWD`` when set (matching terminal tool cwd).
    Returns ``None`` when the path does not resolve to an existing file.
    """
    token = str(raw_path or "").strip()
    if not token:
        return None

    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()
    token = token.replace('\\ ', ' ')
    if not token:
        return None

    expanded = token
    if token.startswith("file://"):
        try:
            parsed = urlparse(token)
            if parsed.scheme == "file":
                expanded = unquote(parsed.path or "")
                if parsed.netloc and os.name == "nt":
                    expanded = f"//{parsed.netloc}{expanded}"
                elif (
                    os.name == "nt"
                    and len(expanded) >= 3
                    and expanded[0] == "/"
                    and expanded[1].isalpha()
                    and expanded[2] == ":"
                ):
                    # file:///C:/... parses to path "/C:/..." — drop the
                    # leading slash so it resolves as a drive-letter path.
                    expanded = expanded[1:]
        except Exception:
            expanded = token
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            expanded = f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    path = Path(expanded)
    if not path.is_absolute():
        base_dir = Path(os.getenv("TERMINAL_CWD", os.getcwd()))
        path = base_dir / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    # Path.exists() / is_file() invoke os.stat(), which raises OSError when
    # the candidate string is structurally invalid as a path — most commonly
    # ENAMETOOLONG (errno 63 on macOS, errno 36 on Linux) when the input
    # exceeds NAME_MAX (typically 255 bytes). This bites pasted slash
    # commands like `/goal <long prose>` because `_detect_file_drop()`'s
    # `starts_like_path` prefilter accepts any input starting with `/`,
    # then this resolver tries to stat it before short-circuiting on the
    # slash-command path. Without this guard the OSError propagates up to
    # the process_loop catch-all in _interactive_loop and the user input
    # is silently lost (the warning ends up in agent.log but the user sees
    # nothing — the prompt just hangs).
    try:
        if not resolved.exists() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved





def _detect_file_drop(user_input: str) -> "dict | None":
    """Detect if *user_input* starts with a real local file path.

    This catches dragged/pasted paths before they are mistaken for slash
    commands, and also supports Termux-friendly paths like ``~/storage/...``.

    Returns a dict on match::

        {
            "path": Path,          # resolved file path
            "is_image": bool,      # True when suffix is a known image type
            "remainder": str,      # any text after the path
        }

    Returns ``None`` when the input is not a real file path.
    """
    if not isinstance(user_input, str):
        return None

    stripped = user_input.strip()
    if not stripped:
        return None

    starts_like_path = (
        stripped.startswith("/")
        or stripped.startswith("~")
        or stripped.startswith("./")
        or stripped.startswith("../")
        or stripped.startswith("file://")
        or (len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in {"\\", "/"} and stripped[0].isalpha())
        or stripped.startswith('"/')
        or stripped.startswith('"~')
        or stripped.startswith("'/")
        or stripped.startswith("'~")
        or stripped.startswith('"./')
        or stripped.startswith('"../')
        or stripped.startswith("'./")
        or stripped.startswith("'../")
        or (len(stripped) >= 4 and stripped[0] in {"'", '"'} and stripped[2] == ":" and stripped[3] in {"\\", "/"} and stripped[1].isalpha())
    )
    if not starts_like_path:
        return None

    direct_path = _resolve_attachment_path(stripped)
    if direct_path is not None:
        return {
            "path": direct_path,
            "is_image": direct_path.suffix.lower() in _IMAGE_EXTENSIONS,
            "remainder": "",
        }

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None and " " in stripped and stripped[0] not in {"'", '"'}:
        space_positions = [idx for idx, ch in enumerate(stripped) if ch == " "]
        for pos in reversed(space_positions):
            candidate = stripped[:pos].rstrip()
            resolved = _resolve_attachment_path(candidate)
            if resolved is not None:
                drop_path = resolved
                remainder = stripped[pos + 1 :].strip()
                break
    if drop_path is None:
        return None

    return {
        "path": drop_path,
        "is_image": drop_path.suffix.lower() in _IMAGE_EXTENSIONS,
        "remainder": remainder,
    }


def _format_image_attachment_badges(attached_images: list[Path], image_counter: int, width: int | None = None) -> str:
    """Format the attached-image badge row for the interactive CLI.

    Narrow terminals such as Termux should get a compact summary that fits on a
    single row, while wider terminals can show the classic per-image badges.
    """
    if not attached_images:
        return ""

    width = width or shutil.get_terminal_size((80, 24)).columns

    def _trunc(name: str, limit: int) -> str:
        return name if len(name) <= limit else name[: max(1, limit - 3)] + "..."

    if width < 52:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 20)}]"
        return f"[📎 {len(attached_images)} images attached]"

    if width < 80:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 32)}]"
        first = _trunc(attached_images[0].name, 20)
        extra = len(attached_images) - 1
        return f"[📎 {first}] [+{extra}]"

    base = image_counter - len(attached_images) + 1
    return " ".join(
        f"[📎 Image #{base + i}]"
        for i in range(len(attached_images))
    )


def _should_auto_attach_clipboard_image_on_paste(pasted_text: str) -> bool:
    """Auto-attach clipboard images only for image-only paste gestures."""
    return not pasted_text.strip()


def _strip_leaked_bracketed_paste_wrappers(text: str) -> str:
    from hermes_cli.input_sanitize import strip_leaked_bracketed_paste_wrappers

    return strip_leaked_bracketed_paste_wrappers(text)


def _hermes_call_output_screen_diff(
    orig_osd,
    app,
    output,
    screen,
    current_pos,
    color_depth,
    previous_screen,
    last_style,
    is_done,
    full_screen,
    attrs_for_style_string,
    style_string_has_style,
    size,
    previous_width,
):
    """Call prompt_toolkit ``_output_screen_diff`` with Hermes resize guards.

    1. Inflate ``previous_screen.height`` when the new screen is taller so pt
       skips the reserve-vertical-space cursor move that stamps chrome into
       scrollback (pt #29 / Hermes #26137).
    2. On AttributeError/TypeError from a corrupt previous paint buffer
       (classic after tmux attach with same width), retry once with
       ``previous_screen=None`` so pt first-paints cleanly instead of crashing
       the event loop with ``'cell' object has no attribute 'char'``.
    """
    try:
        if previous_screen is not None and hasattr(previous_screen, "height") and previous_screen.height < screen.height:
            previous_screen.height = screen.height
    except Exception:
        pass

    try:
        return orig_osd(
            app, output, screen, current_pos, color_depth,
            previous_screen, last_style, is_done, full_screen,
            attrs_for_style_string, style_string_has_style,
            size, previous_width,
        )
    except (AttributeError, TypeError):
        # Corrupt previous_screen / row cells after client reattach.
        return orig_osd(
            app, output, screen, current_pos, color_depth,
            None,  # previous_screen → first-paint erase path
            None,  # last_style
            is_done, full_screen,
            attrs_for_style_string, style_string_has_style,
            size, 0,  # previous_width → treat as changed
        )


def _apply_bracketed_paste_timeout_patch() -> None:
    """Patch prompt_toolkit to recover from torn bracketed-paste sequences.

    prompt_toolkit's ``Vt100Parser.feed()`` buffers all input while waiting
    for the ESC[201~ end mark.  If a terminal drops that end mark (terminal
    race, torn write, SSH glitch, macOS sleep/wake), input appears frozen
    forever — the only recovery used to be killing the tab.

    This patch wraps ``Vt100Parser.feed`` so that bracketed-paste mode
    flushes buffered content as a normal ``BracketedPaste`` event after
    ``_BP_TIMEOUT_S`` seconds without an end marker, then resumes normal
    parsing.  See upstream issue #16263.

    The patch is idempotent — repeated calls are no-ops via the
    ``_hermes_bp_timeout_patched`` sentinel on the module.
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
        from prompt_toolkit.key_binding.key_processor import KeyPress as _PtKeyPress

        if getattr(_vt100_mod, "_hermes_bp_timeout_patched", False):
            return

        _BP_TIMEOUT_S = 2.0  # max time to wait for ESC[201~ before flushing

        def _patched_vt100_feed(self_parser, data: str) -> None:
            if self_parser._in_bracketed_paste:
                self_parser._paste_buffer += data
                end_mark = "\x1b[201~"

                if end_mark in self_parser._paste_buffer:
                    end_index = self_parser._paste_buffer.index(end_mark)
                    paste_content = self_parser._paste_buffer[:end_index]
                    self_parser.feed_key_callback(
                        _PtKeyPress(_PtKeys.BracketedPaste, paste_content)
                    )
                    self_parser._in_bracketed_paste = False
                    remaining = self_parser._paste_buffer[
                        end_index + len(end_mark):
                    ]
                    self_parser._paste_buffer = ""
                    self_parser._hermes_bp_start = None
                    if remaining:
                        _patched_vt100_feed(self_parser, remaining)
                else:
                    bp_start = getattr(self_parser, "_hermes_bp_start", None)
                    now = time.monotonic()
                    if bp_start is None:
                        self_parser._hermes_bp_start = now
                    elif now - bp_start > _BP_TIMEOUT_S:
                        paste_content = self_parser._paste_buffer
                        self_parser._in_bracketed_paste = False
                        self_parser._paste_buffer = ""
                        self_parser._hermes_bp_start = None
                        if paste_content:
                            self_parser.feed_key_callback(
                                _PtKeyPress(_PtKeys.BracketedPaste, paste_content)
                            )
                            logger.warning(
                                "Bracketed-paste timeout (%.1fs) — flushed %d bytes "
                                "without end mark. Terminal may have dropped ESC[201~ "
                                "(see #16263).",
                                now - bp_start,
                                len(paste_content),
                            )
            else:
                # Normal mode — re-inline prompt_toolkit's normal feed path.
                # Calling the original feed here would double-buffer after the
                # bracketed-paste entry transition.
                for i, c in enumerate(data):
                    if self_parser._in_bracketed_paste:
                        _patched_vt100_feed(self_parser, data[i:])
                        break
                    self_parser._input_parser.send(c)

        _vt100_mod.Vt100Parser.feed = _patched_vt100_feed
        _vt100_mod._hermes_bp_timeout_patched = True
        logger.debug("Applied Vt100Parser bracketed-paste timeout patch (#16263)")
    except Exception as exc:  # noqa: BLE001 — defensive: never break startup
        logger.debug("Bracketed-paste timeout patch skipped: %s", exc)


# Cursor Position Report (CPR / DSR) response, format ``ESC[<row>;<col>R``.
# prompt_toolkit's _on_resize() + renderer send ``ESC[6n`` queries to the
# terminal; under resize storms or tab switches the terminal's reply can
# race past the input parser and end up in the input buffer as literal
# text (see issue #14692). Also matches the visible-form ``^[[<row>;<col>R``
# that appears when the ESC byte was stripped by a prior filter.
_DSR_CPR_ESC_RE = re.compile(r"\x1b\[\d+;\d+R")
_DSR_CPR_VISIBLE_RE = re.compile(r"\^\[\[\d+;\d+R")
_SGR_MOUSE_ESC_RE = re.compile(r"\x1b\[<\d+;\d+;\d+[Mm]")
_SGR_MOUSE_VISIBLE_RE = re.compile(r"\^\[\[<\d+;\d+;\d+[Mm]")
# Some terminals/filters can drop ESC and literal "^[[", leaving only
# "<btn;col;rowM" fragments in the buffer. Keep this broad on purpose:
# these fragments are extremely unlikely to be intentional user input, and
# stripping them is better than sending corrupted prompts.
_SGR_MOUSE_BARE_RE = re.compile(r"<\d+;\d+;\d+[Mm]")
_TERMINAL_INPUT_MODE_RESET_SEQ = (
    "\x1b[?1006l"  # disable SGR mouse
    "\x1b[?1003l"  # disable any-motion tracking
    "\x1b[?1002l"  # disable button-motion tracking
    "\x1b[?1000l"  # disable click tracking
    "\x1b[?1004l"  # disable focus events
    "\x1b[?2004l"  # disable bracketed paste
    "\x1b[?1049l"  # leave alt screen (if stuck there)
    "\x1b[<u"      # pop kitty keyboard mode
    "\x1b[>4m"     # reset modifyOtherKeys
    "\x1b[0m"      # reset text attributes
    "\x1b[?25h"    # ensure cursor visible
)
_KITTY_KEYBOARD_PUSH_SEQ = "\x1b[>1u"
_MODIFY_OTHER_KEYS_SEQ = "\x1b[>4;2m"
_EXTENDED_ENTER_KEYS_SEQ = _KITTY_KEYBOARD_PUSH_SEQ + _MODIFY_OTHER_KEYS_SEQ


_BACKSLASH_LINE_CONTINUATION_RE = re.compile(r"\\[ \t]*$")


def _is_ghostty_terminal(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the terminal is Ghostty (either detection path).

    Ghostty must be pushed ONLY modifyOtherKeys, not the Kitty keyboard
    protocol: its Kitty disambiguate-mode implementation strips the Alt
    modifier from the Backspace key, so Option+Backspace arrives as bare
    \\x7f instead of the CSI-u form ``\\x1b[127;3u`` the protocol calls for
    (upstream Ghostty bug), breaking backward-kill-word (#87630
    regression).  Ghostty implements modifyOtherKeys correctly (it then
    emits ``\\x1b[27;3;127~``, which the alias table also maps).

    Matches exactly the two conditions that admit Ghostty through
    ``_terminal_supports_extended_enter_keys``.
    """
    if env is None:
        env = os.environ
    return (
        (env.get("TERM_PROGRAM") or "").strip() == "ghostty"
        or (env.get("TERM") or "").strip().lower() == "xterm-ghostty"
    )


def _terminal_supports_extended_enter_keys(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether it is safe/useful to request modified Enter key reporting.

    The classic CLI already maps Kitty CSI-u / xterm modifyOtherKeys Shift+Enter
    byte sequences to the newline handler. Some terminals (notably iTerm2) only
    emit those distinct sequences after the application asks for extended key
    mode. Keep this allowlist aligned with the Ink TUI, which enables the same
    modes for these terminals.
    """
    if env is None:
        env = os.environ
    term_program = (env.get("TERM_PROGRAM") or "").strip()
    term = (env.get("TERM") or "").strip().lower()
    if env.get("WT_SESSION"):
        return True
    if term_program in {"iTerm.app", "WezTerm", "ghostty", "vscode"}:
        return True
    if env.get("KITTY_WINDOW_ID") or "kitty" in term:
        return True
    if term == "xterm-ghostty":
        return True
    return term.startswith("tmux") or term_program.lower() == "tmux"


def _enable_extended_enter_keys(output=None, env: Optional[Mapping[str, str]] = None) -> bool:
    """Ask allowlisted terminals to report modified keys distinctly.

    Writes the Kitty keyboard protocol push (CSI >1u, disambiguate mode) AND
    xterm modifyOtherKeys level 2 (CSI >4;2m), mirroring the Ink TUI —
    terminals honor whichever protocol they implement (except Ghostty, which
    gets only modifyOtherKeys; see the Ghostty exception below).  Both are
    needed:
    kitty-the-terminal removed modifyOtherKeys support entirely (it only
    speaks its own protocol), while tmux/VS Code only accept modifyOtherKeys.

    Under either protocol the terminal re-encodes modified keys as escape
    sequences — Kitty disambiguate mode as ``ESC[<codepoint>;<mod>u`` (plus
    the Esc key as ``ESC[27u``), modifyOtherKeys=2 as
    ``ESC[27;<mod>;<codepoint>~``.  Stock prompt_toolkit 3.x maps almost
    none of these, which is why the CSI >1u push was temporarily removed in
    #87074 (Ctrl+C arrived as ``ESC[99;5u`` and died, #56684).
    ``install_modify_other_keys_aliases()`` (called at CLI startup from
    ``hermes_cli.pt_input_extras``) now populates ``ANSI_SEQUENCES`` with the
    full Ctrl/Alt/Shift/multi-modifier and functional-key tables under BOTH
    formats, so every existing key binding continues to fire — including
    Ctrl+C, which is handled by prompt_toolkit's ``c-c`` binding (raw mode
    clears ISIG, so the kernel INTR path was never in play for the CLI).

    Ghostty exception: pushes only modifyOtherKeys — see
    ``_is_ghostty_terminal`` for the full rationale (#87630).

    The exit reset sequence pops/resets both modes, so this is safe across
    normal exits, Ctrl+C, and SIGTERM cleanup.
    """
    if not _terminal_supports_extended_enter_keys(env):
        return False
    # Ghostty exception: only modifyOtherKeys — see _is_ghostty_terminal.
    seq = _MODIFY_OTHER_KEYS_SEQ if _is_ghostty_terminal(env) else _EXTENDED_ENTER_KEYS_SEQ
    try:
        target = output
        if target is not None and hasattr(target, "write_raw"):
            target.write_raw(seq)
            target.flush()
            return True
        stream = sys.stdout
        if stream is not None and stream.isatty():
            stream.write(seq)
            stream.flush()
            return True
    except Exception:
        return False
    return False


def _cli_multiline_shortcuts_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return whether classic CLI harness-standard multiline fallbacks are on.

    Default is on to match the norm in adjacent agent harnesses: Ctrl+J is a
    documented no-setup newline shortcut in Claude Code, OpenCode defaults
    ``input_newline`` to include ``ctrl+j``, and Codex exposes Ctrl+J/keymap
    newline behavior. Users on unusual POSIX PTYs that send bare LF for plain
    Enter can set ``display.cli_multiline_shortcuts: false`` to restore the
    legacy c-j submit fallback.
    """
    if config is None:
        config = CLI_CONFIG
    display = config.get("display") if isinstance(config, dict) else None
    value = display.get("cli_multiline_shortcuts", True) if isinstance(display, dict) else True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return True


def _is_backslash_line_continuation(text: str) -> bool:
    """True when Enter should turn a trailing backslash into a newline."""
    return bool(_BACKSLASH_LINE_CONTINUATION_RE.search(text or ""))


def _apply_backslash_line_continuation(text: str) -> str:
    """Replace a trailing ``\\`` marker with an actual newline."""
    return _BACKSLASH_LINE_CONTINUATION_RE.sub("", text or "") + "\n"


def _preserve_ctrl_enter_newline() -> bool:
    """Detect environments where Ctrl+Enter must produce a newline, not submit.

    Windows Terminal, WSL, SSH sessions, Ghostty, and some modern terminals
    deliver Ctrl+Enter/Ctrl+J as bare LF (c-j). On those terminals c-j must
    NOT be bound to submit;
    binding it to submit makes Ctrl+Enter (intended as 'newline like Alt+Enter')
    submit instead. Local POSIX TTYs that deliver Enter as LF (docker exec,
    some thin PTYs without SSH) still need c-j bound to submit when
    display.cli_multiline_shortcuts is disabled, so we keep that legacy opt-out.

    See issue #22379.
    """
    if sys.platform == "win32":
        return True
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return True
    if os.environ.get("WT_SESSION"):
        return True
    if os.environ.get("GHOSTTY_RESOURCES_DIR") or os.environ.get("GHOSTTY_BIN_DIR"):
        return True
    if os.environ.get("TERM", "").lower() == "xterm-ghostty":
        return True
    if os.environ.get("TERM_PROGRAM", "").lower() == "ghostty":
        return True
    if "microsoft" in os.environ.get("WSL_DISTRO_NAME", "").lower():
        return True
    # WSL detection — env vars can be scrubbed under sudo, also peek /proc.
    for p in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if "microsoft" in f.read().lower():
                    return True
        except OSError:
            continue
    return False


def _bind_prompt_submit_keys(
    kb,
    handler,
    *,
    multiline_shortcuts_enabled: Optional[bool] = None,
) -> None:
    """Bind terminal Enter forms to the submit handler.

    Enter is always submit. By default, c-j (Ctrl+J/LF) is left for the
    multiline newline handler because that is the common agent-harness UX.
    Users can set ``display.cli_multiline_shortcuts: false`` to restore the
    legacy POSIX fallback that binds c-j to submit on local thin PTYs whose
    plain Enter arrives as LF instead of CR.

    Even when the setting is disabled, environments where Ctrl+Enter is known
    to arrive as c-j (Windows, WSL, SSH, Windows Terminal, Ghostty) keep c-j
    reserved for newline; otherwise Ctrl+Enter submits instead of composing.
    See _preserve_ctrl_enter_newline() and issue #22379.
    """
    if multiline_shortcuts_enabled is None:
        multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled()
    kb.add("enter")(handler)
    if (
        sys.platform != "win32"
        and not multiline_shortcuts_enabled
        and not _preserve_ctrl_enter_newline()
    ):
        kb.add("c-j")(handler)


def _disable_prompt_toolkit_cpr_warning(app) -> None:
    """Let prompt_toolkit fall back from CPR without printing into the prompt."""
    try:
        app.renderer.cpr_not_supported_callback = None
    except Exception:
        pass


def _terminal_may_leak_cpr() -> bool:
    """Whether classic CLI should suppress prompt_toolkit CPR (ESC[6n) queries.

    Delayed CPR replies (``ESC[<row>;<col>R`` / visible ``^[[<row>;<col>R``)
    leak into the status line and can freeze input when the reply is slow
    (#13870 on SSH/slow PTYs). The same race hits local POSIX TTYs under
    heavy subagent / status-line load — see ``tests/cli/test_cpr_local_leak.py``.

    Policy:
    - ``PROMPT_TOOLKIT_NO_CPR=1`` → always suppress
    - native Windows (``win32``) → keep prompt_toolkit's default for now
      (no native-Windows Application coverage yet); still honor NO_CPR
    - all other platforms → suppress (CPR is only a layout hint; heuristic
      height is enough). SSH env is no longer required to trigger this.
    """
    if os.environ.get("PROMPT_TOOLKIT_NO_CPR", "") == "1":
        return True
    if sys.platform == "win32":
        return False
    return True


def _build_cpr_disabled_output(stdout):
    """Build a Vt100_Output that never sends Cursor Position Report queries.

    prompt_toolkit's renderer sends ``ESC[6n`` (Device Status Report) to learn
    the cursor row before painting in non-fullscreen mode; the terminal replies
    ``ESC[<row>;<col>R``. When that reply is delayed it races into the display
    as raw ``^[[39;1R`` and can stall the renderer's pending-CPR future
    (#13870; also local POSIX under heavy subagent load).

    Constructing the output with ``enable_cpr=False`` marks CPR
    ``NOT_SUPPORTED`` so ``ESC[6n`` is never sent. prompt_toolkit then uses its
    heuristic available-height fallback. Input-side
    ``_strip_leaked_terminal_responses`` remains belt-and-suspenders.

    Note: ``Vt100_Output.from_pty()`` does NOT expose ``enable_cpr`` in
    prompt_toolkit 3.x, so we reproduce its ``get_size`` setup and call the
    constructor directly. Returns ``None`` on any failure so the caller falls back
    to prompt_toolkit's default output (CPR enabled, but input-side scrubbing
    still protects against leaks).
    """
    try:
        import io as _io
        from prompt_toolkit.output.vt100 import Vt100_Output, _get_size
        from prompt_toolkit.data_structures import Size

        def _get_term_size():
            rows = columns = None
            try:
                rows, columns = _get_size(stdout.fileno())
            except (OSError, _io.UnsupportedOperation, AttributeError, ValueError):
                pass
            return Size(rows=rows or 24, columns=columns or 80)

        return Vt100_Output(stdout, _get_term_size, enable_cpr=False)
    except Exception:
        return None


def _select_classic_cli_pt_output(stdout):
    """Select prompt_toolkit Output for classic-CLI Application construction.

    Returns a CPR-disabled ``Vt100_Output`` when ``_terminal_may_leak_cpr()``
    is true, otherwise ``None`` so Application keeps prompt_toolkit's default
    output (Windows preserve-default path).
    """
    if not _terminal_may_leak_cpr():
        return None
    return _build_cpr_disabled_output(stdout)


def _strip_leaked_terminal_responses_with_meta(text: str) -> tuple[str, bool]:
    """Strip leaked terminal control-response sequences from user input.

    Covers Cursor Position Report (CPR / DSR) responses — ``ESC[<row>;<col>R``
    and the visible ``^[[<row>;<col>R`` form. These are replies the terminal
    sends back to queries prompt_toolkit makes during ``_on_resize`` /
    ``_request_absolute_cursor_position``. When the input parser drops one
    (resize storms, multiplexer focus changes, slow PTYs) the response
    lands in the input buffer as literal text and corrupts what the user
    typed.

    Also strips leaked SGR mouse-report fragments (``ESC[<...M/m`` and
    degraded visible forms). Returns ``(cleaned_text, had_mouse_reports)``
    so callers can trigger an in-place terminal mode recovery when needed.
    """
    if not text:
        return text, False

    has_esc = "\x1b[" in text
    has_visible = "^[" in text
    has_bare_mouse = "<" in text and ";" in text and ("M" in text or "m" in text)
    if not (has_esc or has_visible or has_bare_mouse):
        return text, False

    had_mouse_reports = False

    if has_esc:
        text = _DSR_CPR_ESC_RE.sub("", text)
        text, count = _SGR_MOUSE_ESC_RE.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0

    if has_visible:
        text = _DSR_CPR_VISIBLE_RE.sub("", text)
        text, count = _SGR_MOUSE_VISIBLE_RE.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0

    if has_bare_mouse:
        text, count = _SGR_MOUSE_BARE_RE.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0

    return text, had_mouse_reports


def _strip_leaked_terminal_responses(text: str) -> str:
    """Compatibility wrapper returning only cleaned text."""
    cleaned, _ = _strip_leaked_terminal_responses_with_meta(text)
    return cleaned


def _estimate_tui_input_height(
    lines: list[str] | tuple[str, ...],
    prompt_text: str,
    terminal_columns: int,
    *,
    max_height: int = 8,
) -> int:
    """Estimate classic prompt_toolkit input rows using live terminal cells.

    The TextArea prompt is injected with prompt_toolkit's BeforeInput
    processor, which means it consumes cells only on logical line 0. After a
    narrow resize, that first row can leave only one input cell beside an icon
    prompt such as ``⚔ ``, while continuation rows use the full terminal width.
    Never substitute a fake wide fallback here: under- or over-allocating the
    TextArea height leaves stale prompt/input cells visible at the bottom of the
    terminal.
    """
    try:
        from prompt_toolkit.utils import get_cwidth
    except Exception:
        get_cwidth = lambda value: len(value or "")  # type: ignore[assignment]

    try:
        columns = int(terminal_columns or 0)
    except (TypeError, ValueError):
        columns = 0

    columns = max(1, columns)
    prompt_width = max(0, get_cwidth(prompt_text or ""))

    visual_lines = 0
    for index, line in enumerate(lines or [""]):
        # prompt_toolkit's TextArea injects ``prompt`` via BeforeInput, which
        # applies only to logical line 0. Wrapped continuation rows, and later
        # logical lines, use the full terminal width. Count the display cells
        # after that same transformation rather than subtracting the prompt from
        # every wrapped row.
        line_width = get_cwidth(line or "")
        display_width = line_width + (prompt_width if index == 0 else 0)
        if display_width <= 0:
            visual_lines += 1
        else:
            visual_lines += max(1, -(-display_width // columns))

    return min(max(visual_lines, 1), max(1, int(max_height or 1)))


def _status_bar_visible_from_display_config(display_config: object) -> bool:
    """Return the initial classic-CLI status-bar visibility from display config.

    ``display.tui_statusbar`` is the persisted user-facing setting toggled by
    the TUI/statusbar controls. YAML parses bare ``off`` as ``False``, while
    older config snapshots or hand edits may use strings such as ``"off"`` or
    ``"hidden"``. Treat those values consistently so a new CLI process does not
    re-enable a status bar that the user deliberately disabled.
    """
    if not isinstance(display_config, dict):
        display_config = {}
    statusbar_config = display_config.get(
        "statusbar",
        display_config.get("tui_statusbar", "top"),
    )
    if isinstance(statusbar_config, str):
        return statusbar_config.strip().lower() not in {"0", "false", "hidden", "no", "off"}
    return statusbar_config is not False


def _collect_query_images(query: str | None, image_arg: str | None = None) -> tuple[str, list[Path]]:
    """Collect local image attachments for single-query CLI flows."""
    message = query or ""
    images: list[Path] = []

    if isinstance(message, str):
        dropped = _detect_file_drop(message)
        if dropped and dropped.get("is_image"):
            images.append(dropped["path"])
            message = dropped["remainder"] or f"[User attached image: {dropped['path'].name}]"

    if image_arg:
        explicit_path = _resolve_attachment_path(image_arg)
        if explicit_path is None:
            raise ValueError(f"Image file not found: {image_arg}")
        if explicit_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Not a supported image file: {explicit_path}")
        images.append(explicit_path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for img in images:
        key = str(img)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(img)
    return message, deduped


# Strip OSC escape sequences (e.g. OSC-8 hyperlinks) that prompt_toolkit's
# ANSI parser can't handle — it strips \x1b but passes the payload through
# as literal text, garbling the TUI output.
_OSC_ESCAPE_RE = re.compile(r"\x1b\][\s\S]*?(?:\x07|\x1b\\)")


class ChatConsole:
    """Rich Console adapter for prompt_toolkit's patch_stdout context.

    Captures Rich's rendered ANSI output and routes it through _cprint
    so colors and markup render correctly inside the interactive chat loop.
    Drop-in replacement for Rich Console — just pass this to any function
    that expects a console.print() interface.
    """

    def __init__(self):
        from io import StringIO
        self._buffer = StringIO()
        self._inner = Console(
            file=self._buffer,
            force_terminal=True,
            color_system="truecolor",
            highlight=False,
        )

    def print(self, *args, **kwargs):
        self._buffer.seek(0)
        self._buffer.truncate()
        # Read terminal width at render time so panels adapt to current size
        self._inner.width = shutil.get_terminal_size((80, 24)).columns
        self._inner.print(*args, **kwargs)
        output = self._buffer.getvalue()
        # Strip OSC escape sequences (e.g. OSC-8 hyperlinks) before
        # routing through prompt_toolkit's ANSI parser, which only
        # handles CSI/SGR and passes OSC payload through as literal text.
        output = _OSC_ESCAPE_RE.sub("", output)
        for line in output.rstrip("\n").split("\n"):
            _cprint(line)

    @contextmanager
    def status(self, *_args, **_kwargs):
        """Provide a no-op Rich-compatible status context.

        Some slash command helpers use ``console.status(...)`` when running in
        the standalone CLI. Interactive chat routes those helpers through
        ``ChatConsole()``, which historically only implemented ``print()``.
        Returning a silent context manager keeps slash commands compatible
        without duplicating the higher-level busy indicator already shown by
        ``HermesCLI._busy_command()``.
        """
        yield self

# ASCII Art - HERMES-AGENT logo (full width, single line - requires ~95 char terminal)
HERMES_AGENT_LOGO = """[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

# ASCII Art - Hermes Caduceus (compact, fits in left panel)
HERMES_CADUCEUS = """[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""



def _build_compact_banner() -> str:
    """Build a compact banner that fits the current terminal width."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        _skin = get_active_skin()
    except Exception:
        _skin = None

    skin_name = getattr(_skin, "name", "default") if _skin else "default"
    border_color = _skin.get_color("banner_border", "#FFD700") if _skin else "#FFD700"
    title_color = _skin.get_color("banner_title", "#FFBF00") if _skin else "#FFBF00"
    dim_color = _skin.get_color("banner_dim", "#B8860B") if _skin else "#B8860B"

    if skin_name == "default":
        line1 = "⚕ NOUS HERMES - AI Agent Framework"
        tiny_line = "⚕ NOUS HERMES"
    else:
        agent_name = _skin.get_branding("agent_name", "Hermes Agent") if _skin else "Hermes Agent"
        line1 = f"{agent_name} - AI Agent Framework"
        tiny_line = agent_name

    if os.environ.get("HERMES_FAST_STARTUP_BANNER") == "1":
        from hermes_cli import __release_date__ as _release_date
        from hermes_cli import __version__ as _version

        version_line = f"Hermes Agent v{_version} ({_release_date})"
    else:
        version_line = format_banner_version_label()

    w = min(shutil.get_terminal_size().columns - 2, 88)
    if w < 30:
        return f"\n[{title_color}]{tiny_line}[/] [dim {dim_color}]- Nous Research[/]\n"

    inner = w - 2  # inside the box border
    bar = "═" * w
    content_width = inner - 2

    # Truncate and pad to fit
    line1 = line1[:content_width].ljust(content_width)
    line2 = version_line[:content_width].ljust(content_width)

    return (
        f"\n[bold {border_color}]╔{bar}╗[/]\n"
        f"[bold {border_color}]║[/] [{title_color}]{line1}[/] [bold {border_color}]║[/]\n"
        f"[bold {border_color}]║[/] [dim {dim_color}]{line2}[/] [bold {border_color}]║[/]\n"
        f"[bold {border_color}]╚{bar}╝[/]\n"
    )



# ============================================================================
# Slash-command detection helper
# ============================================================================

def _looks_like_slash_command(text: str) -> bool:
    """Return True if *text* looks like a slash command, not a file path.

    Slash commands are ``/help``, ``/model gpt-4``, ``/q``, etc.
    File paths like ``/Users/ironin/file.md:45-46 can you fix this?``
    also start with ``/`` but contain additional ``/`` characters in
    the first whitespace-delimited word.  This helper distinguishes
    the two so that pasted paths are sent to the agent instead of
    triggering "Unknown command".
    """
    if not text or not text.startswith("/"):
        return False
    first_word = text.split()[0]
    # After stripping the leading /, a command name has no slashes.
    # A path like /Users/foo/bar.md always does.
    return "/" not in first_word[1:]


# ============================================================================
# Skill Slash Commands — dynamic commands generated from installed skills
# ============================================================================

_skill_commands = None
_skill_bundles = None


def _slash_args(cmd: str) -> str:
    """Text after the slash-command word, stripped ("" when absent)."""
    parts = cmd.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _ensure_skill_commands() -> dict:
    global _skill_commands
    if _skill_commands is None:
        from agent.skill_commands import scan_skill_commands

        _skill_commands = scan_skill_commands()
    return _skill_commands


def get_skill_commands() -> dict:
    return _ensure_skill_commands()


def build_skill_invocation_message(*args, **kwargs):
    from agent.skill_commands import build_skill_invocation_message as _impl

    return _impl(*args, **kwargs)


def build_preloaded_skills_prompt(*args, **kwargs):
    from agent.skill_commands import build_preloaded_skills_prompt as _impl

    return _impl(*args, **kwargs)


def get_skill_bundles() -> dict:
    global _skill_bundles
    if _skill_bundles is None:
        from agent.skill_bundles import get_skill_bundles as _impl

        _skill_bundles = _impl()
    return _skill_bundles


def build_bundle_invocation_message(*args, **kwargs):
    from agent.skill_bundles import build_bundle_invocation_message as _impl

    return _impl(*args, **kwargs)


def _get_plugin_cmd_handler_names() -> set:
    """Return plugin command names (without slash prefix) for dispatch matching."""
    try:
        from hermes_cli.plugins import get_plugin_commands
        return set(get_plugin_commands().keys())
    except Exception:
        return set()


def _parse_skills_argument(skills: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a CLI skills flag into a deduplicated list of skill identifiers."""
    if not skills:
        return []

    if isinstance(skills, str):
        raw_values = [skills]
    elif isinstance(skills, (list, tuple)):
        raw_values = [str(item) for item in skills if item is not None]
    else:
        raw_values = [str(skills)]

    parsed: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            normalized = part.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            parsed.append(normalized)
    return parsed


def save_config_value(key_path: str, value: any) -> bool:
    """
    Save a value to the active config file at the specified key path.
    
    Respects the same lookup order as load_cli_config():
    1. ~/.hermes/config.yaml (user config - preferred, used if it exists)
    2. ./cli-config.yaml (project config - fallback)
    
    Args:
        key_path: Dot-separated path like "agent.system_prompt"
        value: Value to save
    
    Returns:
        True if successful, False otherwise
    """
    # Runtime persistence ALWAYS targets the user's HERMES_HOME config.yaml,
    # creating it if needed. Resolve HERMES_HOME live (not the import-time
    # _hermes_home constant) so profile switches and test isolation land right.
    #
    # We deliberately do NOT fall back to the repo's project cli-config.yaml:
    # that file is a shipped default/template, and most config readers
    # (load_config → get_hermes_home()/config.yaml, including
    # load_wake_word_config) never read it. Writing a user setting there means
    # the reader never sees it. This was the "wake-word ear reverts to disabled
    # after restart" bug — the toggle's persist wrote to cli-config.yaml (which
    # exists in the checkout) while startup read HERMES_HOME/config.yaml, so the
    # setting silently vanished every restart on any install whose
    # HERMES_HOME/config.yaml didn't exist yet.
    config_path = get_hermes_home() / 'config.yaml'
    
    try:
        # Ensure parent directory exists (for ~/.hermes/config.yaml on first use)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save back atomically while preserving comments, ordering, quotes, and
        # readable Unicode in user-edited config.yaml.
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        
        # Enforce owner-only permissions on config files (contain API keys)
        try:
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass

        # Model/provider changes made through /model and the TUI use this
        # persistence path rather than ``hermes config set``. Surface the same
        # fail-closed cron drift warning for every operator-facing model switch.
        from hermes_cli.config import (
            warn_unpinned_cron_jobs_after_model_config_change,
        )

        warn_unpinned_cron_jobs_after_model_config_change(key_path, value)
        
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False




# ============================================================================
# HermesCLI Class
# ============================================================================


def _normalize_moa_model(model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Map a ``moa:<preset>`` model string to ``(provider, preset)``.

    Returns ``("moa", "<preset>")`` when *model* selects the MoA virtual
    provider, otherwise ``(None, model)`` unchanged. This gives non-interactive
    ``hermes chat -Q -m moa:<preset>`` the same routing the interactive
    ``/moa`` command and the model picker already use: ``resolve_runtime_provider``
    handles ``requested_provider == "moa"`` and ``agent_init`` builds the
    MoAClient off ``provider == "moa"``. Without this the raw ``moa:<preset>``
    string is sent to the real provider and rejected with a 401/400 "model not
    supported" (#56828).
    """
    if isinstance(model, str):
        stripped = model.strip()
        if stripped.lower().startswith("moa:"):
            preset = stripped.split(":", 1)[1].strip()
            if preset:
                return "moa", preset
    return None, model

def _split_model_config_default(raw_default: Any) -> tuple[str, str]:
    # Thin wrapper around the shared helper in config.py — kept for
    # backward compat with existing call sites in this module.
    from hermes_cli.config import split_model_config_default
    return split_model_config_default(raw_default)


class _VoiceInputMessage:
    """Sentinel wrapper for voice-transcribed messages in ``_pending_input``.

    Distinguishes STT output from manually typed text while voice mode is
    active, so the concise-voice-response prefix is applied only to messages
    that actually came from the microphone (#65827).
    """

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _SeededQueryMessage:
    """Sentinel wrapper for a ``-q/--query`` prompt seeded into an
    interactive session.

    When ``hermes chat -q "…"`` runs on a real TTY, the query is submitted as
    the first turn of a normal interactive session instead of the legacy
    answer-and-exit single-query mode. The prompt is arbitrary user text (an
    OS launcher, a desktop integration, a script) — it must be treated
    LITERALLY: no slash-command routing, no ``!`` shell dispatch, no
    file-drop detection. This sentinel marks the seeded first message so
    ``process_loop`` skips those dispatchers for it (and only it).
    """

    __slots__ = ("text", "images")

    def __init__(self, text: str, images=None):
        self.text = text or ""
        self.images = list(images or [])

    def __str__(self) -> str:
        return self.text


def _should_seed_interactive(query, image, quiet: bool, oneshot: bool) -> bool:
    """Whether a ``-q/--image`` invocation should seed an interactive session.

    New default (Aug 2026): on a real TTY, ``chat -q`` submits the prompt as
    the first turn of a normal interactive session (parity with other coding
    agents' seeded launches — e.g. Omarchy's prompted agent terminals).

    The legacy answer-and-exit behavior is preserved for every automation
    surface:
      - ``--oneshot`` on the chat subcommand (explicit legacy opt-in)
      - ``-Q/--quiet`` (machine-readable single-query contract)
      - any non-TTY stdin/stdout (kanban workers, cron, pipes, A2A)
    ``-z/--oneshot`` at the top level never reaches this path at all.
    """
    if not (query or image):
        return False
    if oneshot or quiet:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _panel_box_width(title: str, content_lines: list[str], min_width: int = 46, max_width: int = 76) -> int:
    """Stable TUI panel width wide enough for the title and content (incl. borders)."""
    term_cols = shutil.get_terminal_size((100, 20)).columns
    longest = max([len(title)] + [len(line) for line in content_lines] + [min_width - 4])
    inner = min(max(longest + 4, min_width - 2), max_width - 2, max(24, term_cols - 6))
    return inner + 2  # account for the single leading/trailing spaces inside borders


def _wrap_panel_text(text: str, width: int, subsequent_indent: str = "", *, keep_ws: bool = False) -> list[str]:
    """Wrap panel text; ``keep_ws`` preserves whitespace (command/detail previews)."""
    if keep_ws:
        kw = dict(replace_whitespace=False, drop_whitespace=False)
    else:
        kw = dict(break_long_words=False, break_on_hyphens=False)
    wrapped = textwrap.wrap(text, width=max(8, width), subsequent_indent=subsequent_indent, **kw)
    return wrapped or [""]


_wrap_panel_text_keep_ws = functools.partial(_wrap_panel_text, keep_ws=True)


def _append_panel_line(lines, border_style: str, content_style: str, text: str, box_width: int) -> None:
    inner_width = max(0, box_width - 2)
    lines.append((border_style, "│ "))
    lines.append((content_style, text.ljust(inner_width)))
    lines.append((border_style, " │\n"))


def _append_blank_panel_line(lines, border_style: str, box_width: int) -> None:
    lines.append((border_style, "│" + (" " * box_width) + "│\n"))


@dataclass
class _ChatTurn:
    """Per-turn state shared by the ``chat()`` phases and the agent worker thread.

    ``result`` is written by the worker thread and read by the main thread after the join
    (same object, so late writes from an abandoned thread stay visible exactly as before).
    ``box_opened`` is flipped by the TTS display callback; ``normal_exit`` is set only when
    the TTS worker drained on its own so the ``finally`` never cuts the last sentence.
    """

    result: Optional[dict] = None
    use_streaming_tts: bool = False
    box_opened: bool = False
    thinking_started: bool = False
    text_queue: Optional[queue.Queue] = None
    tts_thread: Optional[threading.Thread] = None
    stream_callback: Optional[Any] = None
    stop_event: Optional[threading.Event] = None
    tts_normal_exit: bool = False
    voice_prefix: str = ""


class HermesCLI(CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, CLITuiMixin, CLIStatusBarMixin, CLIVoiceMixin, CLIModelSwitchMixin, CLISessionMixin, CLIStreamMixin, CLIModalMixin, CLITerminalMixin, CLIInfoMixin, CLILoopsMixin):
    """
    Interactive CLI for the Hermes Agent.
    
    Provides a REPL interface with rich formatting, command history,
    and tool execution capabilities.
    """

    # Seeded -q handoff from main() → run() (see _should_seed_interactive):
    # run() re-creates _pending_input, so the seeded first message rides in
    # on this attribute and is enqueued after the fresh queue exists.
    _seeded_first_message: Optional["_SeededQueryMessage"] = None

    def __init__(
        self,
        model: str = None,
        toolsets: List[str] = None,
        provider: str = None,
        reasoning: str = None,
        api_key: str = None,
        base_url: str = None,
        max_turns: int = None,
        run_budget: float = None,
        verbose: Optional[bool] = None,
        compact: bool = False,
        resume: str = None,
        checkpoints: bool = False,
        pass_session_id: bool = False,
        ignore_rules: bool = False,
    ):
        """
        Initialize the Hermes CLI.

        Args:
            model: Model to use (default: from env or claude-sonnet)
            toolsets: List of toolsets to enable (default: all)
            provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
            reasoning: Reasoning effort override for this run (none|minimal|low|medium|high|xhigh|max|ultra). Wins over config.
            api_key: API key (default: from environment)
            base_url: API base URL (default: OpenRouter)
            max_turns: Maximum tool-calling iterations shared with subagents (default: 500)
            verbose: Enable verbose logging
            compact: Use compact display mode
            resume: Session ID to resume (restores conversation history from SQLite)
            pass_session_id: Include the session ID in the agent's system prompt
        """
        self._init_display_options(verbose, compact)
        
        self._init_model_routing(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                                 checkpoints, pass_session_id, ignore_rules)

        self._init_runtime_state(resume)

    def _init_display_options(self, verbose, compact):
        """Display-related config: compact/tool-progress/focus view, bells, streaming, previews, stream buffers."""
        # Initialize Rich console
        self.console = Console()
        self.config = CLI_CONFIG
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        # tool_progress: "off", "new", "all", "verbose" (from config.yaml display section)
        # YAML 1.1 parses bare `off` as boolean False — normalise to string.
        _raw_tp = CLI_CONFIG["display"].get("tool_progress", "all")
        self.tool_progress_mode = "off" if _raw_tp is False else str(_raw_tp)
        # focus_view: display-only reduced-output mode (/focus). When on, the
        # tool-progress mode is snapped to "off" so the EXISTING suppression
        # path hides per-tool lines, and the pre-focus mode is stashed so
        # /focus off restores it. Purely cosmetic — never changes what is sent
        # to the model. See hermes_cli/focus_view.py.
        self._focus_view_enabled = bool(CLI_CONFIG["display"].get("focus_view", False))
        self._focus_saved_tool_progress = None
        self._focus_hidden_lines = 0
        self._focus_last_counted_tool = None
        if self._focus_view_enabled:
            from hermes_cli.focus_view import (
                FOCUS_TOOL_PROGRESS_MODE,
                normalize_tool_progress_mode,
            )

            self._focus_saved_tool_progress = normalize_tool_progress_mode(
                self.tool_progress_mode
            )
            self.tool_progress_mode = FOCUS_TOOL_PROGRESS_MODE
        # resume_display: "full" (show history) | "minimal" (one-liner only)
        self.resume_display = CLI_CONFIG["display"].get("resume_display", "full")
        # bell_on_complete: play terminal bell (\a) when agent finishes a response
        self.bell_on_complete = CLI_CONFIG["display"].get("bell_on_complete", False)
        # bell_on_prompt: play terminal bell (\a) whenever a blocking prompt
        # modal opens (clarify, approval, sudo password, secret capture)
        self.bell_on_prompt = CLI_CONFIG["display"].get("bell_on_prompt", False)
        # show_reasoning: display model thinking/reasoning before the response
        self.show_reasoning = CLI_CONFIG["display"].get("show_reasoning", True)
        # reasoning_full: when reasoning display is on, print the post-response
        # recap box uncollapsed instead of clamping to the first 10 lines.
        self.reasoning_full = CLI_CONFIG["display"].get("reasoning_full", False)
        _configure_output_history(
            enabled=CLI_CONFIG["display"].get("persistent_output", True),
            max_lines=CLI_CONFIG["display"].get("persistent_output_max_lines", 200),
        )
        # busy_input_mode: "interrupt" (Enter redirects current run),
        # "queue" (Enter queues for next turn), or "steer" (Enter injects
        # mid-run via /steer, arriving after the next tool call).
        _bim = str(CLI_CONFIG["display"].get("busy_input_mode", "interrupt")).strip().lower()
        if _bim == "queue":
            self.busy_input_mode = "queue"
        elif _bim == "steer":
            self.busy_input_mode = "steer"
        else:
            self.busy_input_mode = "interrupt"

        # self.verbose ONLY controls global DEBUG logging (root logger level).
        # display.tool_progress="verbose" controls tool-call rendering (full args,
        # results, think blocks) and is independent — see _apply_logging_levels.
        # Coupling the two (PR #6a1aa420e) caused all module DEBUG logs to spew
        # to console whenever a user set tool_progress: verbose in config.
        self.verbose = bool(verbose) if verbose is not None else False

        # streaming: stream tokens to the terminal as they arrive (display.streaming in config.yaml)
        self.streaming_enabled = CLI_CONFIG["display"].get("streaming", False)
        # show_timestamps: prefix user and assistant labels with timestamps
        self.show_timestamps = CLI_CONFIG["display"].get("timestamps", False)
        self.timestamp_format = CLI_CONFIG["display"].get("timestamp_format", "%H:%M")
        self.final_response_markdown = str(
            CLI_CONFIG["display"].get("final_response_markdown", "strip")
        ).strip().lower() or "strip"
        if self.final_response_markdown not in {"render", "strip", "raw"}:
            self.final_response_markdown = "strip"

        # Inline diff previews for write actions (display.inline_diffs in config.yaml)
        self._inline_diffs_enabled = CLI_CONFIG["display"].get("inline_diffs", True)

        # Per-turn accounting (display.turn_summary / display.spinner_token_flow).
        # Both are CLI-only, display-only chrome. The collector rides the
        # tool-progress feed this class already receives, so no agent-loop
        # bookkeeping is involved.
        self._turn_summary_enabled = bool(CLI_CONFIG["display"].get("turn_summary", True))
        self._spinner_token_flow_enabled = bool(
            CLI_CONFIG["display"].get("spinner_token_flow", True)
        )
        self._turn_summary_collector = None
        self._turn_summary_start = 0.0
        self._turn_token_baseline = 0
        # True only while an interactive (run()-loop) turn is in flight. Single
        # query, -Q, and gateway paths never set it, which is what keeps the
        # summary line out of non-interactive surfaces.
        self._interactive_turn = False

        # Submitted multiline user-message preview (display.user_message_preview in config.yaml)
        _ump = CLI_CONFIG["display"].get("user_message_preview", {})
        if not isinstance(_ump, dict):
            _ump = {}
        try:
            _ump_first_lines = int(_ump.get("first_lines", 2))
        except (TypeError, ValueError):
            _ump_first_lines = 2
        try:
            _ump_last_lines = int(_ump.get("last_lines", 2))
        except (TypeError, ValueError):
            _ump_last_lines = 2
        self.user_message_preview_first_lines = max(1, _ump_first_lines)
        self.user_message_preview_last_lines = max(0, _ump_last_lines)

        # Streaming display state
        self._stream_buf = ""        # Partial line buffer for line-buffered rendering
        self._stream_started = False  # True once first delta arrives
        self._stream_box_opened = False  # True once the response box header is printed
        self._reasoning_preview_buf = ""  # Coalesce tiny reasoning chunks for [thinking] output
        # Table-row buffer.  When a streamed line looks like it could be
        # part of a markdown table, hold it here until the block ends so
        # we can re-pad with wcwidth-aware widths.  Empty by default;
        # populated only while `_in_stream_table` is True.
        self._stream_table_buf: list[str] = []
        self._in_stream_table = False
        self._pending_edit_snapshots = {}
        self._last_input_mode_recovery = 0.0
        self._input_mode_recovery_notice_shown = False
        self._last_termios_drift_check = 0.0
        self._termios_drift_notice_shown = False

    def _init_model_routing(self, model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, checkpoints, pass_session_id, ignore_rules):
        """Resolve model/provider/base_url, turn limits, toolsets, checkpoints, prompt/personality, reasoning + routing config."""
        # Configuration - priority: CLI args > env vars > config file
        # Model comes from: CLI arg or config.yaml (single source of truth).
        # LLM_MODEL/OPENAI_MODEL env vars are NOT checked — config.yaml is
        # authoritative.  This avoids conflicts in multi-agent setups where
        # env vars would stomp each other.
        _model_config = CLI_CONFIG.get("model", {})
        _raw_default = (_model_config.get("default") or _model_config.get("model") or "") if isinstance(_model_config, dict) else (_model_config or "")
        # A dict-valued default (``model.default: {provider: ..., model: ...}``)
        # carries its own provider; flatten it here so the nested provider is
        # available when ``requested_provider`` is constructed below instead of
        # being discarded and replaced by the outer merged ``model.provider``
        # (typically ``"auto"``, which is authoritative at runtime resolution).
        _config_model, _nested_provider = _split_model_config_default(_raw_default)
        _DEFAULT_CONFIG_MODEL = ""
        # Track whether the user passed -m / --model so resume knows not to
        # clobber an explicit override with the session's stored model.
        self._explicit_model_override = bool(model)
        self.model = model or _config_model or _DEFAULT_CONFIG_MODEL
        _startup_provider_override = ""
        _startup_base_url_override = ""
        _startup_api_key_override = ""
        if self.model:
            from hermes_cli.model_switch import resolve_startup_model_route

            _startup_route = resolve_startup_model_route(
                self.model,
                explicit_provider=provider or "",
                current_provider=(
                    provider
                    or _nested_provider
                    or CLI_CONFIG["model"].get("provider")
                    or os.getenv("HERMES_INFERENCE_PROVIDER")
                    or ""
                ),
                user_providers=CLI_CONFIG.get("providers"),
                custom_providers=CLI_CONFIG.get("custom_providers"),
            )
            if _startup_route is not None:
                self.model = _startup_route.model
                _startup_provider_override = _startup_route.provider
                _startup_base_url_override = _startup_route.base_url
                _startup_api_key_override = _startup_route.api_key
        # A ``moa:<preset>`` model string selects the MoA virtual provider in
        # one shot (parity with interactive ``/moa`` and the model picker). Do
        # this before provider resolution so ``-Q -m moa:<preset>`` routes
        # through MoA instead of hitting the real provider with an unknown
        # model (#56828). A ``moa:`` prefix wins over an explicit ``--provider``.
        _moa_provider_override, self.model = _normalize_moa_model(self.model)
        # Read max_tokens from config (env var override: HERMES_MAX_TOKENS)
        _env_mt = os.environ.get("HERMES_MAX_TOKENS")
        if _env_mt:
            try:
                self.max_tokens = int(_env_mt)
            except (ValueError, TypeError):
                self.max_tokens = None
        elif isinstance(_model_config, dict):
            _mt = _model_config.get("max_tokens")
            self.max_tokens = _mt if isinstance(_mt, int) else None
        else:
            self.max_tokens = None
        # Auto-detect model from local server if still on default
        if self.model == _DEFAULT_CONFIG_MODEL:
            _base_url = (_model_config.get("base_url") or "") if isinstance(_model_config, dict) else ""
            if base_url_hostname(_base_url) in ("localhost", "127.0.0.1"):
                from hermes_cli.runtime_provider import _auto_detect_local_model
                _detected = _auto_detect_local_model(_base_url)
                if _detected:
                    self.model = _detected
        # Track whether model was explicitly chosen by the user or fell back
        # to the global default.  Provider-specific normalisation may override
        # the default silently but should warn when overriding an explicit choice.
        # A config model that matches the global fallback is NOT considered an
        # explicit choice — the user just never changed it.  But a config model
        # like "gpt-5.3-codex" IS explicit and must be preserved.
        self._model_is_default = not model and (
            not _config_model or _config_model == _DEFAULT_CONFIG_MODEL
        )

        # An explicit --api-key wins; otherwise a URL-bearing startup alias
        # carries its own credential for the alias host (#28660).
        self._explicit_api_key = api_key or _startup_api_key_override or None
        self._explicit_base_url = base_url

        # Provider selection is resolved lazily at use-time via _ensure_runtime_credentials().
        self.requested_provider = (
            _moa_provider_override
            or provider
            or _startup_provider_override
            or _nested_provider
            or CLI_CONFIG["model"].get("provider")
            or os.getenv("HERMES_INFERENCE_PROVIDER")
            or "auto"
        )
        # `--provider <custom>` without `-m` must use that entry's
        # default_model. Otherwise the global model.default is sent to the
        # custom endpoint and the compressor inherits the wrong context
        # length (#86978). Explicit `-m` still wins.
        if not model and provider:
            try:
                from hermes_cli.runtime_provider import _get_named_custom_provider

                _named_custom = _get_named_custom_provider(provider)
            except Exception as exc:
                logger.warning(
                    "Could not resolve --provider %s default model; "
                    "keeping global model.default (%s)",
                    provider,
                    exc,
                )
                _named_custom = None
            _provider_default = str((_named_custom or {}).get("model") or "").strip()
            if _provider_default:
                self.model = _provider_default
                self._model_is_default = False
        self._provider_source: Optional[str] = None
        self.provider = self.requested_provider
        self.api_mode = "chat_completions"
        self.acp_command: Optional[str] = None
        self.acp_args: list[str] = []
        self.base_url = (
            base_url
            or _startup_base_url_override
            or CLI_CONFIG["model"].get("base_url", "")
            or os.getenv("OPENROUTER_BASE_URL", "")
        ) or None
        # Match key to resolved base_url: OpenRouter URL → prefer OPENROUTER_API_KEY,
        # custom endpoint → prefer OPENAI_API_KEY (issue #560).
        # Note: _ensure_runtime_credentials() re-resolves this before first use.
        if self.base_url and base_url_host_matches(self.base_url, "openrouter.ai"):
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        # Max turns priority: CLI arg > config file > env var > default
        # All paths go through resolve_turn_limit() so that agent.max_turns
        # accepts "none"/"unlimited" (→ sys.maxsize) in addition to ints.
        # See hermes_cli.config.resolve_turn_limit for the full spelling table.
        from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
        if max_turns is not None:  # CLI arg was explicitly set
            self.max_turns = _resolve_turn_limit(max_turns)
        elif CLI_CONFIG["agent"].get("max_turns") is not None:
            self.max_turns = _resolve_turn_limit(CLI_CONFIG["agent"]["max_turns"])
        elif CLI_CONFIG.get("max_turns") is not None:  # Backwards compat: root-level max_turns
            # KEEP (evaluated for the v12 support-floor cleanup, July 2026):
            # no versioned config migration ever rewrote root-level max_turns
            # to agent.max_turns on disk — only load-time normalization
            # (_normalize_max_turns_config) folds it, and configs read through
            # other paths may bypass it. This fallback is therefore the only
            # safety net for configs that still carry the root key.
            self.max_turns = _resolve_turn_limit(CLI_CONFIG["max_turns"])
        else:
            # Env var bridge (set by gateway/run.py from config.yaml, or by the
            # user directly). Empty/unset → default (unlimited).
            self.max_turns = _resolve_turn_limit(os.getenv("HERMES_MAX_ITERATIONS"))

        # Wall-clock run budget: CLI flag wins over config; both optional.
        # None keeps the feature fully off (AIAgent stays dormant).
        if run_budget is not None:
            self.run_budget_seconds = run_budget
        else:
            self.run_budget_seconds = CLI_CONFIG["agent"].get("run_budget_seconds")

        # Parse and validate toolsets
        self.enabled_toolsets = toolsets
        from agent.skill_utils import parse_config_string_list

        self.disabled_toolsets = parse_config_string_list(CLI_CONFIG["agent"].get("disabled_toolsets"))

        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # Validate each toolset — MCP server names are resolved via
            # live registry aliases (registered during discover_mcp_tools),
            # but discovery hasn't run yet at this point, so exclude them.
            mcp_names = set((CLI_CONFIG.get("mcp_servers") or {}).keys())
            invalid = [t for t in toolsets if not validate_toolset(t) and t not in mcp_names]
            if invalid:
                self._console_print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")

        # Filesystem checkpoints: CLI flag > config
        cp_cfg = CLI_CONFIG.get("checkpoints", {})
        if isinstance(cp_cfg, bool):
            cp_cfg = {"enabled": cp_cfg}
        self.checkpoints_enabled = checkpoints or cp_cfg.get("enabled", False)
        self.checkpoint_max_snapshots = cp_cfg.get("max_snapshots", 20)
        self.checkpoint_max_total_size_mb = cp_cfg.get("max_total_size_mb", 500)
        self.checkpoint_max_file_size_mb = cp_cfg.get("max_file_size_mb", 10)
        self.pass_session_id = pass_session_id
        # --ignore-rules: honor either the constructor flag or the env var set
        # by `hermes chat --ignore-rules` in hermes_cli/main.py. When true we
        # pass skip_context_files=True and skip_memory=True to AIAgent so
        # AGENTS.md/SOUL.md/.cursorrules and persistent memory are not loaded.
        self.ignore_rules = ignore_rules or os.environ.get("HERMES_IGNORE_RULES") == "1"

        # Ephemeral system prompt: env var takes precedence, then
        # display.personality / agent.system_prompt from config.
        # hermes_cli.personality is the single owner of overlay resolution.
        from hermes_cli.personality import (
            available_personalities,
            resolve_ephemeral_system_prompt,
        )

        self.system_prompt = (
            os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
            or resolve_ephemeral_system_prompt(CLI_CONFIG)
        )
        self.personalities = available_personalities(CLI_CONFIG)

        # Ephemeral prefill messages (few-shot priming, never persisted)
        self.prefill_messages = _load_prefill_messages(
            _resolve_prefill_messages_file(CLI_CONFIG)
        )

        # Reasoning config (OpenRouter reasoning effort level)
        # Per-model override > global reasoning_effort — resolved through the
        # shared chokepoint in hermes_constants (Closes #21256).
        from hermes_constants import resolve_reasoning_config
        self.reasoning_config = resolve_reasoning_config(CLI_CONFIG, self.model)
        # An explicit --reasoning wins over config for this run only (never
        # persisted). Kanban's dispatcher uses it to pin a task's thinking
        # depth without touching the worker profile's config.yaml. An
        # unparseable level is ignored with a warning rather than silently
        # swapping in the default — same contract as the config path.
        if reasoning is not None and str(reasoning).strip():
            _cli_reasoning = _parse_reasoning_config(reasoning)
            if _cli_reasoning is None:
                logger.warning(
                    "Unknown --reasoning '%s', keeping the configured level",
                    reasoning,
                )
            else:
                self.reasoning_config = _cli_reasoning
        self.service_tier = _parse_service_tier_config(
            CLI_CONFIG["agent"].get("service_tier", "")
        )

        # OpenRouter provider routing preferences
        pr = CLI_CONFIG.get("provider_routing", {}) or {}
        self._provider_sort = pr.get("sort")
        self._providers_only = pr.get("only")
        self._providers_ignore = pr.get("ignore")
        self._providers_order = pr.get("order")
        self._provider_require_params = pr.get("require_parameters", False)
        self._provider_data_collection = pr.get("data_collection")

        # OpenRouter Pareto Code router knob — coding-score floor (0.0-1.0).
        # Only applied when model.model == "openrouter/pareto-code".
        # Empty string / None / out-of-range = unset (let OR pick strongest coder).
        _or_cfg = CLI_CONFIG.get("openrouter", {}) or {}
        _raw_score = _or_cfg.get("min_coding_score")
        self._openrouter_min_coding_score: Optional[float] = None
        if _raw_score not in {None, ""}:
            try:
                _f = float(_raw_score)
                if 0.0 <= _f <= 1.0:
                    self._openrouter_min_coding_score = _f
            except (TypeError, ValueError):
                pass

        # Fallback provider chain — tried in order when primary fails after retries.
        # Merge new ``fallback_providers`` entries with any legacy
        # ``fallback_model`` entries so old configs still participate.
        self._fallback_model = get_fallback_chain(CLI_CONFIG)

    def _init_runtime_state(self, resume):
        """Session store + all per-run mutable state (queues, overlays, pet/voice/status-bar fields)."""
        # Signature of the currently-initialised agent's runtime.  Used to
        # rebuild the agent when provider / model / base_url changes across
        # turns (e.g. after /model or credential rotation).
        self._active_agent_route_signature = None

        # Agent will be initialized on first use
        self.agent: Optional[Any] = None
        self._tool_callbacks_installed = False
        self._tirith_security_checked = False
        self._app = None  # prompt_toolkit Application (set in run())

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self._resumed = False
        # Per-prompt elapsed timer — started at the beginning of each chat turn,
        # frozen when the agent thread completes, displayed in the status bar.
        self._prompt_start_time: Optional[float] = None  # time.time() when turn started
        self._prompt_duration: float = 0.0  # frozen duration of last completed turn
        self._last_turn_finished_at: Optional[float] = None  # time.time() when the last agent loop finished
        # Initialize SQLite session store early so /title works before first message
        self._session_db = None
        self._session_db_unavailable = False
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            # #41386: a failed session store means the transcript is NOT
            # persisted to state.db — the live chat looks healthy but resume
            # later shows a truncated/empty session. A buried log line is not
            # enough; surface it prominently so the user knows persistence is
            # off for this run and can fix the store before relying on resume.
            self._session_db_unavailable = True
            logger.warning("Failed to initialize SessionDB — session will NOT be indexed for search: %s", e)
            try:
                # Console is imported at module scope; do NOT re-import it here.
                # A function-local `import` would make `Console` a local name for
                # the whole __init__ body and break the earlier `self.console =
                # Console()` with UnboundLocalError.
                Console(stderr=True).print(
                    "[bold yellow]⚠ Session store unavailable[/bold yellow] — "
                    "this conversation will [bold]NOT be saved[/bold] to disk and "
                    "cannot be resumed later. Searching past sessions is also disabled.\n"
                    f"  Reason: {e}\n"
                    "  Fix the state.db store (e.g. `hermes update` to rebuild the venv) to restore persistence."
                )
            except Exception:
                # Never let the warning path itself break startup.
                print(
                    "WARNING: Session store unavailable — this conversation will NOT be "
                    f"saved to disk and cannot be resumed later. Reason: {e}"
                )

        # Opportunistic state.db maintenance — runs at most once per
        # min_interval_hours, tracked via state_meta in state.db itself so
        # it's shared across all Hermes processes for this HERMES_HOME.
        # Never blocks startup on failure.
        _run_state_db_auto_maintenance(self._session_db)

        # Opportunistic shadow-repo cleanup — deletes orphan/stale
        # checkpoint repos under ~/.hermes/checkpoints/.  Opt-in via
        # checkpoints.auto_prune, idempotent via .last_prune marker.
        _run_checkpoint_auto_maintenance()

        # Deferred title: stored in memory until the session is created in the DB
        self._pending_title: Optional[str] = None

        # Session ID: reuse existing one when resuming, otherwise generate fresh
        if resume:
            self.session_id = resume
            self._resumed = True
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()

        # History file for persistent input recall across sessions
        self._history_file = _hermes_home / ".hermes_history"
        self._last_invalidate: float = 0.0  # throttle UI repaints
        self._app = None

        # State shared by interactive run() and single-query chat mode.
        # These must exist before any direct chat() call because single-query
        # mode does not go through run().
        self._agent_running = False
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        # Tracks whether the turn that just finished was interrupted via
        # Ctrl+C. Consumed by _maybe_continue_goal_after_turn so /goal loops
        # don't auto-queue another continuation on top of a user-cancelled
        # turn (which would make Ctrl+C feel like it did nothing).
        self._last_turn_interrupted = False
        # When stdout/PTY raises EIO (broken pipe after a stream-stall
        # interrupt), freeze further UI paints so we don't spin the main
        # thread at hundreds of escape-sequence writes/sec (#81521).
        self._terminal_io_broken = False
        self._should_exit = False
        # /exit --delete: when True, the current session's SQLite history and
        # on-disk transcripts are deleted during shutdown. Set by
        # process_command() when the user runs /exit --delete or /quit --delete.
        # Ported from google-gemini/gemini-cli#19332.
        self._delete_session_on_exit = False
        # /update: when set, run() executes relaunch() after prompt_toolkit
        # has fully exited and cleaned up terminal modes.  Set by
        # _handle_update_command() so the relaunch happens on the main thread,
        # not the background process_loop thread.
        self._pending_relaunch: list[str] | None = None
        self._last_ctrl_c_time = 0
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._clarify_multi_base = None
        self._clarify_prefill = ""
        self._sudo_state = None
        self._sudo_deadline = 0
        self._modal_input_snapshot = None
        self._approval_state = None
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._model_picker_state = None
        # Rotating task-oriented composer placeholder (C-09), chosen once per
        # session so it stays stable while the empty input box is on screen.
        try:
            from hermes_cli.tips import get_random_composer_placeholder
            self._composer_placeholder = get_random_composer_placeholder()
        except Exception:
            self._composer_placeholder = ""
        self._command_palette_state = None
        # Armed when a bare `/resume` prints the recent-sessions list so the
        # very next bare numeric input (e.g. `3`) resolves to that session.
        # Holds the exact list used for index resolution; one-shot (cleared on
        # the next submitted input, whether it's the selection or anything
        # else). See #34584.
        self._pending_resume_sessions = None
        # One-shot agent seed set by a slash handler (e.g. /blueprint <name>)
        # that wants its output run as the next agent turn. Consumed and cleared
        # by the interactive loop immediately after process_command() returns.
        self._pending_agent_seed = None
        self._secret_state = None
        self._secret_deadline = 0
        self._spinner_text: str = ""  # thinking spinner text for TUI
        self._tool_start_time: float = 0.0  # monotonic timestamp when current tool started (for live elapsed)
        self._pending_tool_info: dict = {}  # function_name -> list of (preview, args) for stacked scrollback
        self._last_scrollback_tool: str = ""  # last tool name printed to scrollback (for "new" dedup)
        self._command_running = False
        self._command_blocks_input = False
        self._command_status = ""
        # Petdex mascot (opt-in via display.pet). Kitty/Ghostty use Unicode
        # placeholders plus out-of-band image transmission; other terminals
        # use the truecolor half-block fallback.
        self._pet_renderer = None  # agent.pet.render.PetRenderer | None
        self._pet_slug: str = ""
        self._pet_enabled: bool = False
        self._pet_cols: int = 18
        self._pet_scale: float = 0.7
        self._pet_frames_cache: dict = {}  # state -> list[grid]
        self._pet_kitty_cache: dict = {}  # state -> kitty placeholder payload
        self._pet_kitty_image_id: int = 0
        self._pet_kitty_pending: str = ""
        self._pet_frame_idx: int = 0
        self._pet_lock = threading.Lock()
        self._pet_cfg_checked: float = 0.0
        self._pet_anim_running: bool = False
        self._pet_anim_thread = None
        # Transient reaction beats (wave/jump/failed) + steady reasoning flag.
        self._pet_event: str = ""
        self._pet_event_until: float = 0.0
        self._pet_reasoning: bool = False
        self._pet_turn_error: bool = False
        self._attached_images: list[Path] = []
        self._image_counter = 0
        # Ctrl+S prompt stash — park a half-written draft, send something
        # else, bring the draft back.  Session-scoped and in-memory only:
        # drafts routinely contain secrets, so nothing is written to disk.
        from hermes_cli.prompt_stash import PromptStash as _PromptStash
        self._prompt_stash = _PromptStash()
        self.preloaded_skills: list[str] = []
        self._startup_skills_line_shown = False
        # Background --skills preload (started by cmd_chat; joined by
        # finalize_preloaded_skills before any agent is built).
        self._preload_skills_thread: Optional[threading.Thread] = None
        self._preload_skills_result: Optional[tuple] = None
        self._preload_skills_error: Optional[BaseException] = None
        self._preload_skills_requested: list = []
        self._preload_skills_finalized = False
        self._active_session_lease = None

        # Voice mode state (also reinitialized inside run() for interactive TUI).
        self._voice_lock = threading.Lock()
        self._voice_mode = False
        self._voice_tts = False
        self._voice_recorder = None
        self._voice_recording = False
        self._voice_processing = False
        self._voice_continuous = False
        self._voice_tts_done = threading.Event()
        self._voice_tts_done.set()
        self._voice_tts_stop = None  # active streaming pipeline's stop event
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # most recently spoken TTS text (echo guard, #75780)
        self._voice_barge_phase = None  # "generation" or "playback" phase of the last barge trip

        # Status bar visibility (toggled via /statusbar)
        self._status_bar_visible = _status_bar_visible_from_display_config(
            CLI_CONFIG.get("display") if isinstance(CLI_CONFIG, dict) else None
        )
        # Battery read-out in the status bar (toggled via /battery, off by
        # default). Persisted to display.battery so it survives restarts.
        self._battery_visible = bool(CLI_CONFIG["display"].get("battery", False))
        # When True, the input separator rules and the dynamic status bar are
        # hidden until the next user input. Set by _recover_after_resize() so a
        # SIGWINCH cannot stamp a freshly-drawn status bar on top of one that
        # the terminal just reflowed into scrollback — the cause of duplicated
        # bars / "blank line flooding" reports (#19280, #22976).
        self._status_bar_suppressed_after_resize = False
        self._resize_recovery_lock = threading.Lock()
        self._resize_recovery_timer = None
        self._resize_recovery_pending = False
        # Debounced timer that clears the post-resize suppression once the
        # terminal reflow settles, so the status bar returns during idle
        # without waiting for the next submitted input.
        self._status_bar_unsuppress_timer = None
        # Last terminal width seen by the resize handler. Used to distinguish a
        # width change (column reflow → possible ghost chrome, needs a viewport
        # clear) from a rows-only change (no reflow). None until the first
        # resize fires.
        self._last_resize_width = None

        # Background task tracking: {task_id: threading.Thread}
        self._background_tasks: Dict[str, threading.Thread] = {}
        self._background_task_counter = 0

        # Cache-hit ratio baseline — reset on model switch and on
        # context compression so the bar reflects the *current* cache
        # regime, not a lifetime average that survives invalidation.
        self._cache_hit_baseline_prompt = 0
        self._cache_hit_baseline_read = 0
        self._cache_hit_baseline_model: Optional[str] = None
        self._cache_hit_baseline_compressions = 0

    def _claim_active_session(self, surface: str = "cli", *, stderr: bool = False) -> bool:
        """Claim a global active-session slot for this CLI process."""
        if self._active_session_lease is not None:
            return True
        try:
            from hermes_cli.active_sessions import try_acquire_active_session

            lease, message = try_acquire_active_session(
                session_id=self.session_id,
                surface=surface,
                config=self.config,
                # Writer identity for re-entrancy (#94595): a re-claim by this
                # same process for this same session replaces its own entry
                # instead of fencing itself out.
                metadata={"live_session_id": str(self.session_id)},
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return True
        if message:
            if stderr:
                print(message, file=sys.stderr)
            else:
                self._console_print(f"[bold red]{message}[/]")
            return False
        self._active_session_lease = lease
        try:
            atexit.register(self._release_active_session)
        except Exception:
            pass
        return True

    def _release_active_session(self) -> None:
        lease = getattr(self, "_active_session_lease", None)
        if lease is None:
            return
        try:
            lease.release()
        except Exception:
            logger.debug("Failed to release active session slot", exc_info=True)
        finally:
            self._active_session_lease = None

    # ── Per-turn accounting (display.turn_summary / spinner_token_flow) ──
    #
    # Both features are CLI-only chrome. The tally is observed from the
    # tool-progress callback this class already receives on every tool call,
    # so nothing is threaded through the agent loop. Token flow reads the
    # agent's cumulative session counters (bumped per API call in
    # agent/conversation_loop.py) and subtracts a per-turn baseline.

    # ── Petdex mascot (base-CLI pet pane) ───────────────────────────────
    #
    # Parity with the TUI: a sprite in a prompt_toolkit window above the
    # prompt. Kitty/Ghostty use Unicode placeholders — prompt_toolkit owns
    # the measurable grid; image bytes go out-of-band as a virtual placement
    # via after_render + write_raw (cursor untouched). WezTerm/iTerm/sixel
    # stay on half-blocks: they are not placeholder-capable.

    _PET_FRAME_INTERVAL = 0.16
    _PET_CFG_INTERVAL = 2.5

    # ── Streaming display ────────────────────────────────────────────────

    def _install_tool_callbacks(self) -> None:
        """Install tool callbacks that need the live prompt UI."""
        if getattr(self, "_tool_callbacks_installed", False):
            return
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        set_secret_capture_callback(self._secret_capture_callback)
        try:
            from tools.computer_use_tool import set_approval_callback as _set_cu_cb

            _set_cu_cb(self._computer_use_approval_callback)
        except ImportError:
            pass
        self._tool_callbacks_installed = True

    def _ensure_tirith_security(self) -> None:
        """Check tirith availability once before tools can run terminal commands."""
        if getattr(self, "_tirith_security_checked", False):
            return
        self._tirith_security_checked = True
        try:
            from tools.tirith_security import ensure_installed, is_platform_supported

            tirith_path = ensure_installed(log_failures=False)
            if tirith_path is None and is_platform_supported():
                security_cfg = self.config.get("security", {}) or {}
                tirith_enabled = security_cfg.get("tirith_enabled", True)
                if tirith_enabled:
                    _cprint(
                        f"  {_DIM}⚠ tirith security scanner enabled but not available "
                        f"— command scanning will use pattern matching only{_RST}"
                    )
        except Exception:
            pass


    def _show_security_advisories(self):
        """Show a startup banner if any unacked security advisories match.

        Renders a single bold-red box on stderr (so piped stdout remains
        clean) listing the worst hit and pointing at ``hermes doctor``.
        Banner-cache rate-limits this to once per 24h per advisory; full
        remediation lives behind ``hermes doctor`` so the banner stays
        small.
        """
        try:
            from hermes_cli.security_advisories import (
                detect_compromised,
                startup_banner,
            )
            hits = detect_compromised()
            banner = startup_banner(hits)
            if banner:
                # Print to stderr — keeps stdout clean for piped automation,
                # and Rich's banner rendering already wrote to stdout above.
                print(banner, file=sys.stderr, flush=True)
        except Exception:
            # Never let the security banner block startup. Failures are
            # logged at DEBUG by the advisory module.
            pass

    def _show_browser_backend_notice(self):
        """One-time hint when the default Browser Use backend isn't runnable.

        Browser Use mode is the default browser backend, but it silently
        falls back to the built-in browser tools when neither the
        browser-use CLI nor uvx can be found. Surface that downgrade once
        per 24h so users know why browsing behaves differently and how to
        fix it (rate limiting lives in default_downgrade_notice()).
        """
        try:
            from tools.browser_use_cli import default_downgrade_notice

            notice = default_downgrade_notice()
            if notice:
                self._console_print(f"[yellow]⚠ {notice}[/yellow]")
        except Exception:
            # Never let a hint block startup.
            logger.debug("browser backend notice failed", exc_info=True)

    def finalize_preloaded_skills(self) -> None:
        """Join the background --skills preload and fold it into the prompt.

        Idempotent; no-op when no preload was requested. Called from
        ``_init_agent`` (before the agent snapshots ``self.system_prompt``)
        and safe to call from any other consumer of the system prompt.
        Raises ``ValueError`` when EVERY requested skill was unknown —
        the same contract the old synchronous path enforced in cmd_chat.
        """
        if getattr(self, "_preload_skills_finalized", False):
            return
        thread = getattr(self, "_preload_skills_thread", None)
        if thread is None:
            self._preload_skills_finalized = True
            return
        thread.join(timeout=120)
        self._preload_skills_finalized = True
        err = getattr(self, "_preload_skills_error", None)
        if err is not None:
            raise err
        result = getattr(self, "_preload_skills_result", None)
        if not result:
            return
        skills_prompt, loaded_skills, missing_skills = result
        if missing_skills:
            missing_display = ", ".join(missing_skills)
            # If at least one skill loaded, degrade gracefully: skip the
            # unknown ones and continue. A typo'd skill name should not crash
            # the worker (which auto-blocks the Kanban task after retries).
            # Only when EVERY requested skill is missing do we hard-fail, so a
            # fully-misconfigured worker fails loudly instead of running blind.
            if loaded_skills:
                logger.warning(
                    "Unknown skill(s) requested, skipping: %s. "
                    "Continuing with: %s. "
                    "List available skills with `hermes skills list`.",
                    missing_display,
                    ", ".join(loaded_skills),
                )
            else:
                raise ValueError(f"Unknown skill(s): {missing_display}")
        if skills_prompt:
            self.system_prompt = "\n\n".join(
                part for part in (self.system_prompt, skills_prompt) if part
            ).strip()
            self.preloaded_skills = loaded_skills

    def _show_tool_availability_warnings(self):
        """Show warnings about disabled tools due to missing API keys."""
        try:
            from model_tools import check_tool_availability
            
            available, unavailable = check_tool_availability()
            
            # Filter to only those missing API keys (not system deps)
            api_key_missing = [u for u in unavailable if u["missing_vars"]]
            
            if api_key_missing:
                self._console_print()
                self._console_print("[yellow]⚠️  Some tools disabled (missing API keys):[/]")
                for item in api_key_missing:
                    tools_str = ", ".join(item["tools"][:2])  # Show first 2 tools
                    if len(item["tools"]) > 2:
                        tools_str += f", +{len(item['tools'])-2} more"
                    self._console_print(f"   [dim]• {item['name']}[/] [dim italic]({', '.join(item['missing_vars'])})[/]")
                self._console_print("[dim]   Run 'hermes setup' to configure[/]")
        except Exception:
            pass  # Don't crash on import errors
    
    def show_config(self):
        """Display current configuration with kawaii ASCII art."""
        # Get terminal config from environment (which was set from cli-config.yaml)
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        terminal_cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        terminal_timeout = os.getenv("TERMINAL_TIMEOUT", "60")
        
        user_config_path = _hermes_home / 'config.yaml'
        project_config_path = Path(__file__).parent / 'cli-config.yaml'
        if user_config_path.exists():
            config_path = user_config_path
        else:
            config_path = project_config_path
        config_status = "(loaded)" if config_path.exists() else "(not found)"
        
        # ``self.api_key`` may be a callable (Azure Foundry Entra ID bearer
        # provider). Never invoke it; just identify the auth surface.
        from agent.azure_identity_adapter import is_token_provider

        # Prefer the LIVE agent's credential when one exists: HermesCLI's
        # constructor seeds self.api_key from OPENAI/OPENROUTER env vars
        # before provider resolution runs, so on non-OpenAI providers (Nous,
        # Anthropic, ...) the constructor value is a different vendor's key
        # than the one actually authenticating requests. /config displaying
        # an sk-proj-... OpenAI key next to a Nous base URL was the visible
        # symptom (full-surface CLI QA sweep, Aug 2026).
        display_key = self.api_key
        agent = getattr(self, "agent", None)
        if agent is not None and getattr(agent, "api_key", None):
            display_key = agent.api_key
        if is_token_provider(display_key):
            api_key_display = "Microsoft Entra ID"
        elif isinstance(display_key, str) and len(display_key) > 12:
            api_key_display = f"{display_key[:8]}...{display_key[-4:]}"
        else:
            api_key_display = "Not set!"
        
        print()
        title = "(^_^) Configuration"
        width = 50
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        print("  -- Model --")
        print(f"  Model:     {self.model}")
        print(f"  Base URL:  {self.base_url}")
        print(f"  API Key:   {api_key_display}")
        print()
        print("  -- Terminal --")
        print(f"  Environment:  {terminal_env}")
        if terminal_env == "ssh":
            ssh_host = os.getenv("TERMINAL_SSH_HOST", "not set")
            ssh_user = os.getenv("TERMINAL_SSH_USER", "not set")
            ssh_port = os.getenv("TERMINAL_SSH_PORT", "22")
            print(f"  SSH Target:   {ssh_user}@{ssh_host}:{ssh_port}")
        print(f"  Working Dir:  {terminal_cwd}")
        print(f"  Timeout:      {terminal_timeout}s")
        print()
        print("  -- Agent --")
        print(f"  Max Turns:  {self.max_turns}")
        print(f"  Toolsets:   {', '.join(self.enabled_toolsets) if self.enabled_toolsets else 'all'}")
        print(f"  Verbose:    {self.verbose}")
        print()
        print("  -- Session --")
        print(f"  Started:     {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Config File: {config_path} {config_status}")
        print()
    
    @staticmethod
    def _resolve_personality_prompt(value) -> str:
        """Accept string or dict personality value; return system prompt string.

        Delegates to hermes_cli.personality (single owner of rendering).
        """
        from hermes_cli.personality import render_personality_prompt

        return render_personality_prompt(value)


    



    # Slash dispatch: canonical command -> (method name, pass cmd_original?).
    # Commands absent here resolve by convention to ``_handle_<name>_command(cmd)``
    # (dashes -> underscores). Resolved via getattr at dispatch time so
    # instance/class-level monkeypatching still works. Handlers return False
    # to exit the REPL; anything else keeps it alive.
    _SLASH_DISPATCH: dict[str, tuple[str, bool]] = {
        "exit": ("_cmd_exit", True),
        "quit": ("_cmd_exit", True),
        "help": ("_cmd_help", True),
        "palette": ("_open_command_palette", False),
        "whoami": ("_handle_whoami_command", False),
        "profile": ("_handle_profile_command", False),
        "toolsets": ("show_toolsets", False),
        "config": ("show_config", False),
        "redraw": ("_cmd_redraw", True),
        "clear": ("_cmd_clear", True),
        "history": ("show_history", False),
        "title": ("_cmd_title", True),
        "new": ("_cmd_new", True),
        "model": ("_handle_model_switch", True),
        "codex-runtime": ("_handle_codex_runtime", True),
        "retry": ("_cmd_retry", True),
        "prompt": ("_handle_prompt_compose_command", True),
        "undo": ("_cmd_undo", True),
        "save": ("save_conversation", True),
        "skills": ("_cmd_skills", True),
        "platforms": ("_show_gateway_status", False),
        "status": ("_show_session_status", False),
        "context": ("_show_context_breakdown", True),
        "egress": ("_cmd_egress", True),
        "statusbar": ("_cmd_statusbar", True),
        "verbose": ("_toggle_verbose", False),
        "yolo": ("_toggle_yolo", False),
        "compress": ("_manual_compress", True),
        "subscription": ("_show_subscription", False),
        "topup": ("_show_billing", True),
        "insights": ("_show_insights", True),
        "update": ("_cmd_update", True),
        "version": ("_cmd_version", True),
        "paste": ("_handle_paste_command", False),
        "reload": ("_cmd_reload", True),
        "reload-mcp": ("_confirm_and_reload_mcp", True),
        "reload-skills": ("_cmd_reload_skills", True),
        "plugins": ("_cmd_plugins", True),
        "stop": ("_handle_stop_command", False),
        "agents": ("_handle_agents_command", False),
        "bg": ("_handle_background_command", True),
        "queue": ("_cmd_queue", True),
        "steer": ("_cmd_steer", True),
        "moa": ("_cmd_moa", True),
    }

    @classmethod
    def _slash_handler(cls, canonical: str) -> tuple[str, bool] | None:
        """(method name, pass cmd_original?) for a registered command, else None."""
        entry = cls._SLASH_DISPATCH.get(canonical)
        if entry is None:
            name = f"_handle_{canonical.replace('-', '_')}_command"
            if callable(getattr(cls, name, None)):
                entry = (name, True)
        return entry

    def process_command(self, command: str) -> bool:
        """
        Process a slash command.
        
        Args:
            command: The command string (starting with /)
            
        Returns:
            bool: True to continue, False to exit
        """
        # Lowercase only for dispatch matching; preserve original case for arguments
        cmd_lower = command.lower().strip()
        cmd_original = command.strip()

        # Resolve aliases via central registry so adding an alias is a one-line
        # change in hermes_cli/commands.py instead of touching every dispatch site.
        from hermes_cli.commands import resolve_command as _resolve_cmd
        _base_word = cmd_lower.split()[0].lstrip("/")
        _cmd_def = _resolve_cmd(_base_word)
        canonical = _cmd_def.name if _cmd_def else _base_word

        # pre_command observer hook (#64204): fires for every recognized
        # slash command BEFORE its handler runs. Observer-only in v1 —
        # return values are ignored (fire_pre_command_hook logs directives
        # at debug). Never raises, so a broken plugin can't break dispatch.
        if _cmd_def is not None:
            from hermes_cli.plugins import fire_pre_command_hook
            fire_pre_command_hook(
                surface="cli",
                command=canonical,
                alias_used=_base_word,
                args_raw=_slash_args(cmd_original),
                session_key=getattr(self, "session_id", None),
                platform="cli",
            )

        # A bare `/resume` prompt is one-shot: any command other than the
        # resume/sessions handlers (which manage the pending state themselves)
        # disarms it so a later number isn't swallowed as a stale selection.
        # See #34584.
        if canonical not in {"resume", "sessions"}:
            self._pending_resume_sessions = None

        entry = self._slash_handler(canonical)
        if entry is None:
            return self._process_unregistered_slash(cmd_original, cmd_lower)
        method_name, pass_arg = entry
        handler = getattr(self, method_name)
        result = handler(cmd_original) if pass_arg else handler()
        return result is not False

    def _process_unregistered_slash(self, cmd_original: str, cmd_lower: str) -> bool:
        """Fallthrough for slash input with no built-in handler.

        Precedence (order matters): user quick_commands (exec/alias) ->
        plugin commands -> skill bundles -> skill commands -> unique-prefix
        expansion -> unknown-command message. Always returns True unless
        it re-dispatches through process_command.
        """
        # Check for user-defined quick commands (bypass agent loop, no LLM call)
        base_cmd = cmd_lower.split()[0]
        skill_commands = _ensure_skill_commands()
        skill_bundles = get_skill_bundles()
        quick_commands = self.config.get("quick_commands", {})
        if base_cmd.lstrip("/") in quick_commands:
            qcmd = quick_commands[base_cmd.lstrip("/")]
            if qcmd.get("type") == "exec":
                import subprocess
                exec_cmd = qcmd.get("command", "")
                if exec_cmd:
                    try:
                        # shell=True is intentional: quick_commands are user-defined
                        # shell snippets from config.yaml — not agent/LLM controlled.
                        # Sanitize env to prevent credential leakage —
                        # quick commands run in the CLI process which
                        # has all API keys in os.environ.
                        from tools.environments.local import build_subprocess_env
                        sanitized_env = build_subprocess_env()
                        from hermes_cli._subprocess_compat import windows_hide_flags
                        result = subprocess.run(
                            exec_cmd, shell=True, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30, env=sanitized_env,
                            # No console flash on Windows (#56747).
                            creationflags=windows_hide_flags(),
                        )
                        output = result.stdout.strip() or result.stderr.strip()
                        if output:
                            from agent.redact import redact_sensitive_text
                            output = redact_sensitive_text(output)
                            self._console_print(_rich_text_from_ansi(output))
                        else:
                            self._console_print("[dim]Command returned no output[/]")
                    except subprocess.TimeoutExpired:
                        self._console_print("[bold red]Quick command timed out (30s)[/]")
                    except Exception as e:
                        self._console_print(f"[bold red]Quick command error: {e}[/]")
                else:
                    self._console_print(f"[bold red]Quick command '{base_cmd}' has no command defined[/]")
            elif qcmd.get("type") == "alias":
                target = qcmd.get("target", "").strip()
                if target:
                    target = target if target.startswith("/") else f"/{target}"
                    user_args = cmd_original[len(base_cmd):].strip()
                    aliased_command = f"{target} {user_args}".strip()
                    return self.process_command(aliased_command)
                else:
                    self._console_print(f"[bold red]Quick command '{base_cmd}' has no target defined[/]")
            else:
                self._console_print(f"[bold red]Quick command '{base_cmd}' has unsupported type (supported: 'exec', 'alias')[/]")
        # Check for plugin-registered slash commands
        elif base_cmd.lstrip("/") in _get_plugin_cmd_handler_names():
            from hermes_cli.plugins import (
                get_plugin_command_handler,
                resolve_plugin_command_result,
            )
            plugin_handler = get_plugin_command_handler(base_cmd.lstrip("/"))
            if plugin_handler:
                user_args = cmd_original[len(base_cmd):].strip()
                try:
                    result = resolve_plugin_command_result(
                        plugin_handler(user_args)
                    )
                    if result:
                        _cprint(str(result))
                except Exception as e:
                    _cprint(f"\033[1;31mPlugin command error: {e}{_RST}")
        # Skill bundles take precedence over individual skills — /<bundle>
        # loads multiple skills at once. Rescans cheaply when files change.
        elif base_cmd in skill_bundles:
            user_instruction = cmd_original[len(base_cmd):].strip()
            bundle_result = build_bundle_invocation_message(
                base_cmd, user_instruction, task_id=self.session_id
            )
            if bundle_result:
                msg, loaded_names, missing = bundle_result
                bundle_info = skill_bundles[base_cmd]
                print(
                    f"\n⚡ Loading bundle: {bundle_info['name']} "
                    f"({len(loaded_names)} skills)"
                )
                if missing:
                    ChatConsole().print(
                        f"[yellow]Skipped missing skills: {', '.join(missing)}[/]"
                    )
                if hasattr(self, '_pending_input'):
                    self._pending_input.put(msg)
            else:
                ChatConsole().print(
                    f"[bold red]Failed to load bundle for {base_cmd}[/]"
                )
        # Check for skill slash commands (/gif-search, /axolotl, etc.)
        elif base_cmd in skill_commands:
            rest = cmd_original[len(base_cmd):].strip()
            # Stacked slash-skill invocations: `/skill-a /skill-b do XYZ`
            # loads every leading skill (up to 5), not just the first.
            # Inspired by Claude Code v2.1.199.
            from agent.skill_commands import (
                build_stacked_skill_invocation_message,
                split_stacked_skill_commands,
            )
            extra_keys, user_instruction = split_stacked_skill_commands(rest)
            if extra_keys:
                stacked_result = build_stacked_skill_invocation_message(
                    [base_cmd, *extra_keys],
                    user_instruction,
                    task_id=self.session_id,
                )
                if stacked_result:
                    msg, loaded_names, missing = stacked_result
                    print(
                        f"\n⚡ Loading {len(loaded_names)} stacked skills: "
                        f"{', '.join(loaded_names)}"
                    )
                    if missing:
                        ChatConsole().print(
                            f"[yellow]Skipped missing skills: {', '.join(missing)}[/]"
                        )
                    if hasattr(self, '_pending_input'):
                        self._pending_input.put(msg)
                else:
                    ChatConsole().print(
                        f"[bold red]Failed to load stacked skills for {base_cmd}[/]"
                    )
                return True
            user_instruction = rest
            msg = build_skill_invocation_message(
                base_cmd, user_instruction, task_id=self.session_id
            )
            if msg:
                skill_name = skill_commands[base_cmd]["name"]
                print(f"\n⚡ Loading skill: {skill_name}")
                if hasattr(self, '_pending_input'):
                    self._pending_input.put(msg)
            else:
                ChatConsole().print(f"[bold red]Failed to load skill for {base_cmd}[/]")
        else:
            # Prefix matching: if input uniquely identifies one command, execute it.
            # Matches against both built-in COMMANDS and installed skill commands so
            # that execution-time resolution agrees with tab-completion.
            from hermes_cli.commands import COMMANDS
            typed_base = cmd_lower.split()[0]
            all_known = set(COMMANDS) | set(skill_commands) | set(skill_bundles)
            matches = [c for c in all_known if c.startswith(typed_base)]
            if len(matches) > 1:
                # Prefer an exact match (typed the full command name)
                exact = [c for c in matches if c == typed_base]
                if len(exact) == 1:
                    matches = exact
                else:
                    # Prefer the unique shortest match:
                    # /qui → /quit (5) wins over /quint-pipeline (15)
                    min_len = min(len(c) for c in matches)
                    shortest = [c for c in matches if len(c) == min_len]
                    if len(shortest) == 1:
                        matches = shortest
            if len(matches) == 1:
                # Expand the prefix to the full command name, preserving arguments.
                # Guard against redispatching the same token to avoid infinite
                # recursion when the expanded name still doesn't hit an exact branch
                # (e.g. /config with extra args that are not yet handled above).
                full_name = matches[0]
                if full_name == typed_base:
                    # Already an exact token — no expansion possible; fall through
                    _cprint(f"\033[1;31mUnknown command: {cmd_lower}{_RST}")
                    _cprint(f"{_DIM}{_ACCENT}Type /help for available commands{_RST}")
                else:
                    remainder = cmd_original.strip()[len(typed_base):]
                    full_cmd = full_name + remainder
                    return self.process_command(full_cmd)
            elif len(matches) > 1:
                _cprint(f"{_ACCENT}Ambiguous command: {cmd_lower}{_RST}")
                _cprint(f"{_DIM}Did you mean: {', '.join(sorted(matches))}?{_RST}")
            else:
                _cprint(f"\033[1;31mUnknown command: {cmd_lower}{_RST}")
                _cprint(f"{_DIM}{_ACCENT}Type /help for available commands{_RST}")
        return True

    def _owns_process_notification(self, event: dict) -> bool:
        """Return whether this CLI session provably owns a delegation event.

        Delegations dispatched before context compression retain the original
        session key, so resolve that key to its continuation before comparing.
        Missing or foreign keys fail closed and remain queued for their owner.
        """
        event_key = str(event.get("session_key") or "")
        current_key = str(getattr(self, "session_id", "") or "")
        if not event_key or not current_key:
            return False
        if event_key == current_key:
            return True
        try:
            session_db = getattr(self, "_session_db", None)
            resolved_key = (
                session_db.resolve_resume_session_id(event_key)
                if session_db is not None
                else event_key
            ) or event_key
        except Exception:
            resolved_key = event_key
        return str(resolved_key) == current_key

    def _drain_process_notifications(self, consumer: str) -> None:
        """Queue background notifications owned by this visible CLI session.

        ``process_registry`` restores durable delegation completions into every
        process using the same Hermes profile.  Always pass this CLI's stable
        session identity when draining so another window cannot claim and mark
        delivered a completion that belongs to this one.
        """
        from tools.process_registry import process_registry
        from tools.async_delegation import (
            claim_event_delivery,
            complete_event_delivery,
        )

        session_key = getattr(self, "session_id", "") or ""
        for event, synthetic_message in process_registry.drain_notifications(
            session_key=session_key,
            owns_event=self._owns_process_notification,
        ):
            claim = claim_event_delivery(event, consumer)
            if claim is None:
                continue
            self._pending_input.put(synthetic_message)
            complete_event_delivery(event, claim)

    def _drain_interrupt_queue_to_pending_input(self) -> None:
        """Move stray messages from ``_interrupt_queue`` into ``_pending_input``.

        While the agent is running, user input is routed into
        ``_interrupt_queue`` (see the architecture comment near
        ``_route_user_input_when_busy``). The explicit-interrupt path at the
        top of ``process_loop`` only drains that queue when
        ``busy_input_mode == "interrupt"`` AND a ``pending_message`` was
        acknowledged. If the agent's turn finishes naturally (no interrupt),
        any messages typed during the turn stay stuck in ``_interrupt_queue``
        forever. Subsequent ``Enter`` presses re-route to the same blocked
        queue and the CLI appears to hang.

        Called once at the end of every turn from ``process_loop``'s ``finally``
        block. Catches and swallows ``Exception`` because the drain must never
        break the main loop. (#20271)
        """
        try:
            while not self._interrupt_queue.empty():
                stray = self._interrupt_queue.get_nowait()
                if stray:
                    self._pending_input.put(stray)
        except Exception:
            pass  # Non-fatal — never break the main loop

    def _on_reasoning(self, reasoning_text: str):
        """Callback for intermediate reasoning display during tool-call loops."""
        if not reasoning_text:
            return
        self._reasoning_preview_buf = getattr(self, "_reasoning_preview_buf", "") + reasoning_text
        self._flush_reasoning_preview(force=False)

            # NOTE: We deliberately do NOT raise per-logger levels for
            # tools/run_agent/etc. in quiet mode. Setting logger.setLevel
            # above the file handler level filters records before they
            # reach handlers, so agent.log / errors.log lose visibility
            # into stream-retry events, credential rotations, etc.
            # Console quietness is enforced by hermes_logging not
            # installing a console StreamHandler in non-verbose mode.

        # Do NOT join here — process_loop calls this from its idle branch, so a
        # blocking join would freeze input consumption for up to 30s (and a hung
        # MCP server could block far longer). The reload runs purely in the
        # background daemon thread, which reports its own progress/completion
        # status via print() inside _reload_mcp().

    # Inline-skip tokens that bypass the destructive-slash confirmation modal.
    # A general escape hatch for non-interactive use (scripting/automation) and
    # for the degraded path where the modal can't be marshaled onto the app loop
    # — lets users self-serve without flipping approvals.destructive_slash_confirm
    # in config. (Native Windows now drives the modal normally — see #33961.)
    _DESTRUCTIVE_SKIP_TOKENS = frozenset({"now", "--yes", "-y"})

    # ====================================================================
    # Tool-call generation indicator (shown during streaming)
    # ====================================================================

    # ====================================================================
    # Tool progress callback (audio cues for voice mode)
    # ====================================================================

    # ====================================================================
    # Voice mode methods
    # ====================================================================

    # ── Wake word ("Hey Hermes") ─────────────────────────────────────────
    #
    # An always-on hotword listener (tools/wake_word.py) that, on detecting
    # the wake phrase, starts a fresh session and captures one utterance via
    # the existing voice pipeline — the "Hey Siri" pattern, fully on-device.
    #
    # The detector holds the microphone, so it must be paused while a voice
    # turn records (two input streams on one device is unreliable). On wake we
    # pause it and mark the system suspended; a lightweight watchdog resumes it
    # once the turn finishes and the CLI is idle again — covering every exit
    # path (transcript submitted, no speech, or transcription error) without
    # threading resume logic through the voice machinery.

            # Leave _wake_suspended set; the watchdog resumes once idle.

    # --- Batch clarify (multi-question, issue #18450) -----------------------

    def chat(self, message, images: list = None, voice_input: bool = False) -> Optional[str]:
        """
        Send a message to the agent and get a response.
        
        Handles streaming output, interrupt detection (user typing while agent
        is working), and re-queueing of interrupted messages.
        
        Uses a dedicated _interrupt_queue (separate from _pending_input) to avoid
        race conditions between the process_loop and interrupt monitoring. Messages
        typed while the agent is running go to _interrupt_queue; messages typed while
        idle go to _pending_input.
        
        Args:
            message: The user's message (str or multimodal content list)
            images: Optional list of Path objects for attached images
            voice_input: True when the message came from voice transcription
                (gates the concise voice-response prefix, #65827)
            
        Returns:
            The agent's response, or None on error
        """
        # Single-query and direct chat callers do not go through run(), so
        # register secure secret capture here as well.
        set_secret_capture_callback(self._secret_capture_callback)

        # Reset the per-turn interrupt flag. Any subsequent path that
        # discovers an interrupt (below, after run_conversation) will flip
        # this to True. Early returns (credential refresh failure, etc.)
        # leave it False, which is correct — those aren't user interrupts.
        self._last_turn_interrupted = False

        # Refresh provider credentials if needed (handles key rotation transparently)
        if not self._ensure_runtime_credentials():
            return None

        turn_route = self._resolve_turn_agent_config(message)
        if turn_route["signature"] != self._active_agent_route_signature:
            self.agent = None

        # Initialize agent if needed
        if self.agent is None:
            _cprint(f"{_DIM}Initializing agent...{_RST}")
        if not self._init_agent(
            model_override=turn_route["model"],
            runtime_override=turn_route["runtime"],
            request_overrides=turn_route.get("request_overrides"),
        ):
            return None
        agent = self.agent
        if agent is None:
            return None

        # Route image attachments based on the active model's vision capability.
        # "native" → pass pixels as OpenAI-style content parts (adapters
        #            translate for Anthropic/Gemini/Bedrock).
        # "text"   → pre-analyze each image with vision_analyze and prepend the
        #            description as text — works with non-vision models.
        # See agent/image_routing.py for the decision table.
        if images:
            try:
                from agent.image_routing import (
                    build_native_content_parts,
                    decide_image_input_mode,
                )
                from hermes_cli.config import load_config

                _img_model, _img_provider = "", ""
                if isinstance(self.model, dict):
                    _img_model, _ = _split_model_config_default(self.model)
                else:
                    _img_model = str(self.model or "")
                if isinstance(self.provider, dict):
                    _, _img_provider = _split_model_config_default(self.provider)
                else:
                    _img_provider = str(self.provider or "")
                _img_mode = decide_image_input_mode(
                    _img_provider.strip(),
                    _img_model.strip(),
                    load_config(),
                    requested_provider=(self.requested_provider or "").strip(),
                )
            except Exception as _img_exc:
                logging.debug("image_routing decision failed, defaulting to text: %s", _img_exc)
                _img_mode = "text"

            if _img_mode == "native":
                try:
                    _text_for_parts = message if isinstance(message, str) else ""
                    _img_str_paths = [str(p) for p in images]
                    _parts, _skipped = build_native_content_parts(
                        _text_for_parts,
                        _img_str_paths,
                    )
                    if _skipped:
                        _cprint(
                            f"  {_DIM}⚠ skipped {len(_skipped)} unreadable image path(s){_RST}"
                        )
                    if any(p.get("type") == "image_url" for p in _parts):
                        _img_names = ", ".join(Path(p).name for p in _img_str_paths)
                        _cprint(
                            f"  {_DIM}📎 attaching {len(images)} image(s) natively "
                            f"(model supports vision): {_img_names}{_RST}"
                        )
                        message = _parts
                    else:
                        # All images unreadable — fall back to text enrichment.
                        message = self._preprocess_images_with_vision(
                            message if isinstance(message, str) else "", images
                        )
                except Exception as _img_exc:
                    logging.warning("native image attach failed, falling back to text: %s", _img_exc)
                    message = self._preprocess_images_with_vision(
                        message if isinstance(message, str) else "", images
                    )
            else:
                message = self._preprocess_images_with_vision(
                    message if isinstance(message, str) else "", images
                )

        # Expand @ context references (e.g. @file:main.py, @diff, @folder:src/)
        if isinstance(message, str) and "@" in message:
            try:
                from agent.context_references import preprocess_context_references
                from agent.model_metadata import get_model_context_length
                _ctx_len = get_model_context_length(
                    self.model, base_url=self.base_url or "", api_key=self.api_key or "",
                    provider=self.provider or "",
                    config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None)
                _ctx_result = preprocess_context_references(
                    message, cwd=os.getcwd(), context_length=_ctx_len)
                if _ctx_result.expanded or _ctx_result.blocked:
                    if _ctx_result.references:
                        _cprint(
                            f"  {_DIM}[@ context: {len(_ctx_result.references)} ref(s), "
                            f"{_ctx_result.injected_tokens} tokens]{_RST}")
                    for w in _ctx_result.warnings:
                        _cprint(f"  {_DIM}⚠ {w}{_RST}")
                    if _ctx_result.blocked:
                        return "\n".join(_ctx_result.warnings) or "Context injection refused."
                    message = _ctx_result.message
            except Exception as e:
                logging.debug("@ context reference expansion failed: %s", e)

        # Sanitize surrogate characters that can arrive via clipboard paste from
        # rich-text editors (Google Docs, Word, etc.).  Lone surrogates are invalid
        # UTF-8 and crash JSON serialization in the OpenAI SDK.
        if isinstance(message, str):
            from run_agent import _sanitize_surrogates
            message = _sanitize_surrogates(message)

        # Keep the exact CLI input dict available until turn-start persistence.
        # Copy the completed agent transcript before appending: otherwise this
        # UI-only staging step mutates ``agent._session_messages`` and exposes a
        # duplicate-prone intermediate snapshot to terminal-close persistence.
        if self.conversation_history is getattr(agent, "_session_messages", None):
            self.conversation_history = list(self.conversation_history)
        # The prior turn's override applies only to its own user dict. Clear it
        # before exposing the next staged input to close persistence; otherwise
        # a shutdown before the worker prologue can write old API-local text as
        # this new user message (#63766).
        persist_lock = getattr(agent, "_session_persist_lock", None)

        def _stage_user_message() -> None:
            agent._persist_user_message_idx = None
            agent._persist_user_message_override = None
            agent._persist_user_message_timestamp = None
            from agent.message_metadata import stamp_message_timestamp

            staged_user_message = stamp_message_timestamp(
                {"role": "user", "content": message}
            )
            agent._pending_cli_user_message = staged_user_message
            self.conversation_history.append(staged_user_message)

        if persist_lock is None:
            _stage_user_message()
        else:
            with persist_lock:
                _stage_user_message()

        ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
        print(flush=True)
        
        turn = _ChatTurn()
        try:
            # Reset streaming display state for this turn
            self._reset_stream_state()
            # Separate from _reset_stream_state because this must persist
            # across intermediate turn boundaries (tool-calling loops) — only
            # reset at the start of each user turn.
            self._reasoning_shown_this_turn = False

            self._chat_setup_turn_audio(turn, message, voice_input)

            # Start agent in background thread (daemon so it cannot keep the
            # process alive when the user closes the terminal tab — SIGHUP
            # exits the main thread and daemon threads are reaped automatically).
            # Start per-prompt elapsed timer — frozen after the agent thread
            # finishes; reset on the next turn.
            self._prompt_start_time = time.time()
            self._prompt_duration = 0.0
            agent_thread = threading.Thread(
                target=self._chat_run_agent, args=(turn, message), daemon=True
            )
            agent_thread.start()

            interrupt_msg = self._chat_monitor_agent_thread(turn, agent_thread)

            self._chat_settle_turn(turn)

            return self._chat_render_turn(turn, agent_thread, interrupt_msg)

        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            # Stop the ambient thinking sound the moment the turn ends —
            # every exit path (normal, error, interrupt) lands here.
            if turn.thinking_started:
                try:
                    from tools.voice_mode import stop_thinking_sound
                    stop_thinking_sound()
                except Exception:
                    pass
            # Ensure streaming TTS resources are cleaned up even on error.
            # Normal path sends the sentinel at line ~3568; this is a safety
            # net for exception paths that skip it.  Duplicate sentinels are
            # harmless — stream_tts_to_speaker exits on the first None.
            #
            # Only set stop_event on the exception path.  On normal exit
            # (_tts_normal_exit is True) the pipeline has already drained —
            # setting stop_event here would race the playback worker and
            # could cut the final sentence mid-audio.
            if turn.text_queue is not None:
                try:
                    turn.text_queue.put_nowait(None)
                except Exception:
                    pass
            if turn.stop_event is not None and not turn.tts_normal_exit:
                logger.info("TTS CUT: exception finally block setting stop_event")
                turn.stop_event.set()
            if turn.tts_thread is not None and turn.tts_thread.is_alive():
                turn.tts_thread.join(timeout=5)

    def _chat_setup_turn_audio(self, turn, message, voice_input):
        """Arm the full-duplex listener and the streaming-TTS pipeline for this turn (voice mode only)."""
        # Full-duplex agent-turn listener (continuous voice mode): arm
        # the mic NOW — at utterance-submit — not when TTS playback
        # starts. It spans generation (speech interrupts the turn) and
        # playback (speech cuts TTS), and disarms itself when the turn
        # is fully done. See _voice_full_duplex_listener.
        if self._voice_mode and self._voice_continuous:
            self._voice_last_tts_text = ""
            threading.Thread(
                target=self._voice_full_duplex_listener, daemon=True
            ).start()

        # --- Streaming TTS setup ---
        # Any working TTS provider streams sentence-by-sentence as the agent
        # generates tokens: PCM-streaming providers (ElevenLabs, OpenAI) play
        # chunks as they arrive, everything else synthesizes per sentence.

        if self._voice_tts:
            try:
                from tools.tts_tool import (
                    _import_sounddevice,
                    check_tts_requirements,
                    stream_tts_to_speaker,
                )
                _import_sounddevice()
                turn.use_streaming_tts = check_tts_requirements()
            except Exception:
                pass

        if turn.use_streaming_tts:
            turn.text_queue = queue.Queue()
            turn.stop_event = threading.Event()

            # When token streaming is enabled (the common case), the
            # CLI's _stream_delta already renders text token-by-token as
            # the model generates it. Passing a display_callback here too
            # would render every sentence a second time. Only attach the
            # callback when streaming is disabled, so the TTS consumer
            # becomes the sole display path.
            _tts_display_cb = None
            if not self.streaming_enabled:
                def display_callback(sentence: str):
                    """Called by TTS consumer when a sentence is ready to display + speak."""
                    if not turn.box_opened:
                        turn.box_opened = True
                        w = self._scrollback_box_width(getattr(self.console, "width", 80))
                        label = " ⚕ Hermes "
                        if self.show_timestamps:
                            label = f"{label}{datetime.now().strftime(getattr(self, 'timestamp_format', '%H:%M'))} "
                        fill = w - 2 - HermesCLI._status_bar_display_width(label)
                        _cprint(f"\n{_ACCENT}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")
                    _cprint(f"{_STREAM_PAD}{sentence.rstrip()}")
                _tts_display_cb = display_callback

            turn.tts_thread = threading.Thread(
                target=stream_tts_to_speaker,
                args=(turn.text_queue, turn.stop_event, self._voice_tts_done),
                kwargs={"display_callback": _tts_display_cb},
                daemon=True,
            )
            turn.tts_thread.start()
            # Expose the pipeline's stop event so barge-in paths (voice
            # key, full-duplex listener) can cut playback from outside
            # this turn. The full-duplex listener itself was armed at
            # turn start (see above) — it spans generation AND playback.
            self._voice_tts_stop = turn.stop_event

            def stream_callback(delta: str):
                if turn.text_queue is not None:
                    turn.text_queue.put(delta)
                # Track what's actually being spoken so a playback-phase
                # barge capture can be checked against it (echo guard,
                # #75780).
                self._voice_last_tts_text = (self._voice_last_tts_text or "") + delta
            turn.stream_callback = stream_callback

        # When voice mode is active, prepend a brief instruction so the
        # model responds concisely. The prefix is API-call-local only —
        # run_conversation persists the original clean user message.
        if voice_input and isinstance(message, str):
            turn.voice_prefix = (
                "[Voice input — respond concisely and conversationally, "
                "2-3 sentences max. No code blocks or markdown.] "
            )

    def _chat_run_agent(self, turn, message):
        """Agent-thread body: bind per-thread callbacks/approval key, prepend one-shot notes, run the turn."""
        # Set callbacks inside the agent thread so thread-local storage
        # in terminal_tool is populated for this thread.  The main thread
        # registration (run() line ~9046) is invisible here because
        # _callback_tls is threading.local().  Matches the pattern used
        # by acp_adapter/server.py for ACP sessions.
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        try:
            set_secret_capture_callback(self._secret_capture_callback)
        except Exception:
            pass
        # Bind this turn's approval session key into the contextvar so
        # ``tools.approval.is_current_session_yolo_enabled()`` resolves
        # against the same key that ``/yolo`` toggles under (see
        # ``_toggle_yolo`` → ``enable_session_yolo(self.session_id)``).
        # Mirrors ``tui_gateway/server.py`` and ``gateway/run.py`` which
        # bind the same contextvar before invoking the agent.
        try:
            from tools.approval import (
                reset_current_session_key,
                set_current_session_key,
            )
            _approval_session_token = set_current_session_key(
                self.session_id or "default"
            )
        except Exception:
            reset_current_session_key = None  # type: ignore[assignment]
            _approval_session_token = None
        agent_message = turn.voice_prefix + message if turn.voice_prefix else message
        # Prepend pending notes via _prepend_note_to_message, which
        # handles both plain-string and multimodal content-parts list
        # messages. Naive ``note + "\n\n" + agent_message`` crashed with
        # TypeError when an image was attached (agent_message is a list)
        # and a /model or /reload-skills note was queued for the turn.
        _msn = getattr(self, '_pending_model_switch_note', None)
        if _msn:
            agent_message = _prepend_note_to_message(agent_message, _msn)
            self._pending_model_switch_note = None
        # Prepend pending /reload-skills note so the model sees which
        # skills were added/removed before handling this turn. Same
        # one-shot queue pattern as the model-switch note above.
        _srn = getattr(self, '_pending_skills_reload_note', None)
        if _srn:
            agent_message = _prepend_note_to_message(agent_message, _srn)
            self._pending_skills_reload_note = None
        # Barged mid-speech (VAD or record key)? Tell the model it was
        # cut off — same one-shot, API-local note channel as above.
        from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE, take_speech_interrupted
        if take_speech_interrupted():
            agent_message = _prepend_note_to_message(agent_message, SPEECH_INTERRUPTED_NOTE)
        _moa_cfg = getattr(self, "_pending_moa_config", None)
        self._pending_moa_config = None
        if _moa_cfg is None:
            _moa_cfg = None
        # Model/skill notes and voice instructions are API-local. Keep
        # the original staged input as the durable transcript value so a
        # close-path marker follows the same dict into turn setup rather
        # than producing a second noted user row (#63766).
        _persist_clean_user_message = (
            message if (turn.voice_prefix or agent_message != message) else None
        )
        _one_turn_model_restore = getattr(
            self, "_pending_one_turn_model_restore", None
        )
        self._pending_one_turn_model_restore = None
        try:
            turn.result = self.agent.run_conversation(
                user_message=agent_message,
                conversation_history=self.conversation_history[:-1],  # Exclude the message we just added
                stream_callback=turn.stream_callback,
                task_id=self.session_id,
                persist_user_message=_persist_clean_user_message,
                moa_config=_moa_cfg,
            )
            if getattr(self, "_pending_moa_disable_after_turn", False):
                _restore = getattr(self, "_pending_moa_restore_model", None) or {}
                for _key, _value in _restore.items():
                    if _value is not None:
                        setattr(self, _key, _value)
                self.agent = None
                self._pending_moa_restore_model = None
                self._pending_moa_disable_after_turn = False
        except Exception as exc:
            logging.error("run_conversation raised: %s", exc, exc_info=True)
            _summary = getattr(self.agent, '_summarize_api_error', lambda e: str(e)[:300])(exc)
            turn.result = {
                "final_response": f"Error: {_summary}",
                "messages": [],
                "api_calls": 0,
                "completed": False,
                "failed": True,
                "error": _summary,
            }
        finally:
            if _one_turn_model_restore:
                self._restore_model_runtime_snapshot(_one_turn_model_restore)
            # Surface any credit notices queued during the turn (cold-start
            # seed / per-turn capture) now that the response is done — printing
            # at this boundary paints cleanly above the prompt instead of being
            # buried behind the streaming output.
            self._flush_credit_notices()
            # Clear thread-local callbacks so a reused thread doesn't
            # hold stale references to a disposed CLI instance.
            try:
                set_sudo_password_callback(None)
                set_approval_callback(None)
                set_secret_capture_callback(None)
            except Exception:
                pass
            # Release the per-turn approval session key. ``_session_yolo``
            # state itself is preserved across turns (so /yolo persists
            # for the whole CLI run); we just unbind the contextvar so a
            # reused thread doesn't see stale identity on its next run.
            if _approval_session_token is not None and reset_current_session_key is not None:
                try:
                    reset_current_session_key(_approval_session_token)
                except Exception:
                    pass

    def _chat_monitor_agent_thread(self, turn, agent_thread):
        """Poll the interrupt queue while the agent thread runs; returns the interrupting message (or None)."""
        # Ambient "thinking" sound: calm bubble blips while the agent
        # works in voice mode with no audio flowing, so the user knows
        # it's alive during long thinking/tool stretches. Skipped per-blip
        # while TTS speaks, the mic records, or a barge capture is live;
        # stopped outright as soon as the turn ends. voice.thinking_sound
        # gates it (default on); macOS is handled inside (TCC-safe skip).
        if self._voice_mode:
            try:
                from tools.voice_mode import start_thinking_sound

                turn.thinking_started = start_thinking_sound(
                    should_play=lambda: (
                        self._voice_tts_done.is_set()
                        and not self._voice_recording
                        and not self._voice_barge_capture.is_set()
                    )
                )
            except Exception:
                turn.thinking_started = False

        # Monitor the dedicated interrupt queue while the agent runs.
        # _interrupt_queue is separate from _pending_input, so process_loop
        # and chat() never compete for the same queue.
        # When a clarify question is active, user input is handled entirely
        # by the Enter key binding (routed to the clarify response queue),
        # so we skip interrupt processing to avoid stealing that input.
        interrupt_msg = None
        while agent_thread.is_alive():
            if hasattr(self, '_interrupt_queue'):
                try:
                    interrupt_msg = self._interrupt_queue.get(timeout=0.1)
                    if interrupt_msg:
                        # If clarify is active, the Enter handler routes
                        # input directly; this queue shouldn't have anything.
                        # But if it does (race condition), don't interrupt —
                        # and don't drop the message either: park it in
                        # _pending_input so it runs as the next turn.
                        if self._clarify_state or self._clarify_freetext:
                            try:
                                self._pending_input.put(interrupt_msg)
                            except Exception:
                                pass
                            interrupt_msg = None
                            continue
                        print("\n⚡ New message detected, interrupting...")
                        # Signal TTS to stop on interrupt
                        if turn.stop_event is not None:
                            turn.stop_event.set()
                        self.agent.interrupt(interrupt_msg)
                        # Clear any active overlay states the interrupted agent
                        # left behind.  approval/clarify/sudo/secret prompts gate
                        # input (read_only condition + keypress filter) until
                        # explicitly reset — without this the CLI freezes after
                        # an interrupt until the prompt's own timeout expires (#14026).
                        self._clear_active_overlays_for_interrupt()
                        # Debug: log to file (stdout may be devnull from redirect_stdout)
                        try:
                            _dbg = _hermes_home / "interrupt_debug.log"
                            with open(_dbg, "a", encoding="utf-8") as _f:
                                _f.write(f"{time.strftime('%H:%M:%S')} interrupt fired: msg={str(interrupt_msg)[:60]!r}, "
                                         f"children={len(self.agent._active_children)}, "
                                         f"parent._interrupt={self.agent._interrupt_requested}\n")
                                for _ci, _ch in enumerate(self.agent._active_children):
                                    _f.write(f"  child[{_ci}]._interrupt={_ch._interrupt_requested}\n")
                        except Exception:
                            pass
                        break
                except queue.Empty:
                    # Force prompt_toolkit to flush any pending stdout
                    # output from the agent thread.  Without this, the
                    # StdoutProxy buffer only flushes on renderer passes
                    # triggered by input events — on macOS this causes
                    # the CLI to appear frozen until the user types. (#1624)
                    self._invalidate(min_interval=0.15)
            else:
                # Fallback for non-interactive mode (e.g., single-query)
                agent_thread.join(0.1)

        # Wait for the agent thread to finish.  After an interrupt the
        # agent may take a few seconds to clean up (kill subprocess, persist
        # session).  Poll instead of a blocking join so the process_loop
        # stays responsive — if the user sent another interrupt or the
        # agent gets stuck, we can break out instead of freezing forever.
        if interrupt_msg is not None:
            # Interrupt path: poll briefly, then move on.  The agent
            # thread is daemon — it dies on process exit regardless.
            for _wait_tick in range(50):  # 50 * 0.2s = 10s max
                agent_thread.join(timeout=0.2)
                if not agent_thread.is_alive():
                    break
                # Check if user fired ANOTHER interrupt (Ctrl+C sets
                # _should_exit which process_loop checks on next pass).
                if getattr(self, '_should_exit', False):
                    break
            if agent_thread.is_alive():
                logger.warning(
                    "Agent thread still alive after interrupt "
                    "(thread %s). Daemon thread will be cleaned up "
                    "on exit.",
                    agent_thread.ident,
                )
        else:
            # Normal completion: agent thread should be done already,
            # but guard against edge cases.
            agent_thread.join(timeout=30)
        return interrupt_msg

    def _chat_settle_turn(self, turn):
        """After the agent thread ends: freeze timers, flush streams, drain TTS, sync history/session id."""
        # Freeze per-prompt elapsed timer once the agent thread has
        # exited (or been abandoned as a daemon after interrupt).
        if self._prompt_start_time is not None:
            self._prompt_duration = max(0.0, time.time() - self._prompt_start_time)
            self._prompt_start_time = None
        # Record when this agent loop finished so the status bar can show
        # idle time since the last final response.
        self._last_turn_finished_at = time.time()

        # Proactively clean up async clients whose event loop is dead.
        # The agent thread may have created AsyncOpenAI clients bound
        # to a per-thread event loop; if that loop is now closed, those
        # clients' __del__ would crash prompt_toolkit's loop on GC.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass

        # Flush any remaining streamed text and close the box
        self._flush_stream()

        # Signal end-of-text to TTS consumer and wait for it to finish
        if turn.use_streaming_tts and turn.text_queue is not None:
            turn.text_queue.put(None)  # sentinel
            if turn.tts_thread is not None:
                turn.tts_thread.join(timeout=120)
            # Mark normal completion only if the thread actually
            # finished.  If join() timed out and the thread is still
            # alive, leave _tts_normal_exit False so the finally block
            # sets stop_event to kill the runaway worker.
            if turn.tts_thread is not None and not turn.tts_thread.is_alive():
                turn.tts_normal_exit = True

        # Drain any remaining agent output still in the StdoutProxy
        # buffer so tool/status lines render ABOVE our response box.
        # The flush pushes data into the renderer queue; the short
        # sleep lets the renderer actually paint it before we draw.
        sys.stdout.flush()
        time.sleep(0.15)

        # Update history with full conversation
        self.conversation_history = turn.result.get("messages", self.conversation_history) if turn.result else self.conversation_history

        # If auto-compression fired mid-turn, the agent created a new
        # continuation session and mutated self.agent.session_id. Sync
        # the CLI's session_id so /status, /resume, title generation,
        # and the exit summary all target the live child session rather
        # than the ended parent. Mirrors the gateway's post-run sync
        # (gateway/run.py around line 9983).
        if (
            self.agent
            and getattr(self.agent, "session_id", None)
            and self.agent.session_id != self.session_id
        ):
            self._transfer_session_yolo(self.session_id, self.agent.session_id)
            self.session_id = self.agent.session_id
            getattr(self, "_write_terminal_breadcrumb", lambda: None)()
            self._pending_title = None

    def _chat_render_turn(self, turn, agent_thread, interrupt_msg):
        """Post-turn display: error/interrupt handling, reasoning + response panels, bell, re-queues. Returns the response text."""
        # Get the final response
        response = turn.result.get("final_response", "") if turn.result else ""

        # Session titling now runs at TURN START (agent/turn_context.py)
        # from the user's message alone, so it is already done — or in
        # flight — by the time we get here, instead of waiting on a final
        # response that a failed or interrupted turn never produces.

        # Handle failed or partial results (e.g., non-retryable errors, rate limits,
        # truncated output, invalid tool calls). Both "failed" and "partial" with
        # an empty final_response mean the agent couldn't produce a usable answer.
        if turn.result and (turn.result.get("failed") or turn.result.get("partial")) and not response:
            error_detail = turn.result.get("error", "Unknown error")
            response = f"Error: {error_detail}"
            # Stop continuous voice mode on persistent errors (e.g. 429 rate limit)
            # to avoid an infinite error → record → error loop
            if self._voice_continuous:
                self._voice_continuous = False
                _cprint(f"\n{_DIM}Continuous voice mode stopped due to error.{_RST}")

        # Handle interrupt - check if we were interrupted
        pending_message = None
        _show_interrupt_marker = False
        _interrupted_this_turn = bool(turn.result and turn.result.get("interrupted"))
        # Expose the flag for post-turn hooks (e.g. goal continuation)
        # so they can skip themselves when the turn was user-cancelled.
        self._last_turn_interrupted = _interrupted_this_turn
        if _interrupted_this_turn:
            pending_message = turn.result.get("interrupt_message") or interrupt_msg
            # #60920: Don't append the interruption marker to response so it
            # is never recorded in _OUTPUT_HISTORY by the Panel rendering
            # below. The marker is printed separately with _suspend_output_history
            # after the response Panel to preserve the visual while avoiding
            # duplicates on terminal redraw (_recover_terminal_after_interrupt).
            _show_interrupt_marker = bool(response and pending_message)
        elif interrupt_msg:
            # We fired agent.interrupt(interrupt_msg) but the turn result
            # doesn't acknowledge it. Two ways this happens, both racy:
            #   1. The agent thread had already passed its last interrupt
            #      check (or finished) when the interrupt landed — the turn
            #      completed normally and finalize_turn() never saw the flag.
            #   2. The 10s post-interrupt wait above expired and we
            #      abandoned the daemon thread; `result` is still None.
            # In both cases the user's message must NOT be dropped —
            # re-queue it as the next turn (#interrupt-vacuumed-into-void).
            pending_message = interrupt_msg
            # If the interrupt landed after finalize_turn()'s
            # clear_interrupt(), the stale flag would instantly abort the
            # NEXT turn at its first loop check. Clear it now that we've
            # claimed the message — but ONLY if the agent thread actually
            # exited. If it's still alive (abandoned after the 10s wait),
            # the flag is what makes the wedged tool eventually unwind;
            # clearing it would un-signal that thread.
            try:
                if (
                    not agent_thread.is_alive()
                    and self.agent
                    and getattr(self.agent, "_interrupt_requested", False)
                ):
                    self.agent.clear_interrupt()
            except Exception:
                pass

        response_previewed = turn.result.get("response_previewed", False) if turn.result else False

        # Display reasoning (thinking) box if enabled and available.
        # Skip when streaming already showed reasoning live.  Use the
        # turn-persistent flag (_reasoning_shown_this_turn) instead of
        # _reasoning_stream_started — the latter gets reset during
        # intermediate turn boundaries (tool-calling loops), which caused
        # the reasoning box to re-render after the final response.
        _reasoning_already_shown = getattr(self, '_reasoning_shown_this_turn', False)
        if self.show_reasoning and turn.result and not _reasoning_already_shown:
            reasoning = turn.result.get("last_reasoning")
            if reasoning:
                w = self._scrollback_box_width()
                r_label = " Reasoning "
                r_fill = w - 2 - len(r_label)
                r_top = f"{_DIM}┌─{r_label}{'─' * max(r_fill - 1, 0)}┐{_RST}"
                r_bot = f"{_DIM}└{'─' * (w - 2)}┘{_RST}"
                # Collapse long reasoning to the first 10 lines unless the
                # user opted into full display via /reasoning full.
                lines = reasoning.strip().splitlines()
                if len(lines) > 10 and not getattr(self, "reasoning_full", False):
                    display_reasoning = "\n".join(lines[:10])
                    display_reasoning += f"\n{_DIM}  ... ({len(lines) - 10} more lines — /reasoning full to show){_RST}"
                else:
                    display_reasoning = reasoning.strip()
                _cprint(f"\n{r_top}\n{_DIM}{display_reasoning}{_RST}\n{r_bot}")

        if response and not response_previewed:
            # Use skin engine for label/color with fallback
            try:
                from hermes_cli.skin_engine import get_active_skin
                _skin = get_active_skin()
                label = _skin.get_branding("response_label", "⚕ Hermes")
                _resp_color = _maybe_remap_for_light_mode(_skin.get_color("response_border", "#CD7F32"))
                _resp_text = _maybe_remap_for_light_mode(_skin.get_color("banner_text", "#FFF8DC"))
            except Exception:
                label = "⚕ Hermes"
                _resp_color = _maybe_remap_for_light_mode("#CD7F32")
                _resp_text = _maybe_remap_for_light_mode("#FFF8DC")

            is_error_response = turn.result and (turn.result.get("failed") or turn.result.get("partial"))
            already_streamed = self._stream_started and self._stream_box_opened and not is_error_response
            if turn.use_streaming_tts and turn.box_opened and not is_error_response:
                # Text was already printed sentence-by-sentence; just close the box
                w = self._scrollback_box_width()
                _cprint(f"\n{_ACCENT}╰{'─' * (w - 2)}╯{_RST}")
            elif already_streamed:
                # Response was already streamed token-by-token with box framing;
                # _flush_stream() already closed the box. Skip Rich Panel.
                # A transform hook runs after streaming. Show a suffix for
                # append-only changes, or the complete replacement otherwise.
                _post_stream_text = _post_stream_transform_output(response, turn.result)
                if _post_stream_text.strip():
                    _cprint(_post_stream_text)
            else:
                _chat_console = ChatConsole()
                _chat_console.print(Panel(
                    _render_final_assistant_content(response, mode=self.final_response_markdown),
                    title=f"[{_resp_color} bold]{label}[/]",
                    title_align="left",
                    border_style=_resp_color,
                    style=_resp_text,
                    box=rich_box.HORIZONTALS,
                    padding=(1, 0),
                    width=self._scrollback_box_width(),
                ))

            # Durable, provider-agnostic billing CTA below the response. The
            # response panel carries the full guidance; this pins the single
            # action to take (Nous → /topup, other providers → their billing
            # page) so it stays visible instead of scrolling away as prose.
            if turn.result and turn.result.get("failure_reason") == "billing":
                _bb = turn.result.get("billing_block") or {}
                _prov_label = _bb.get("provider_label") or "your provider"
                if _bb.get("is_nous"):
                    _cta_lines = [
                        "Run [bold]/topup[/] to add credits, or "
                        "[bold]/subscription[/] to change plan.",
                    ]
                else:
                    _url = _bb.get("billing_url")
                    _cta_lines = [
                        f"Add credits with {_prov_label}"
                        + (f": [bold]{_url}[/]" if _url else ".")
                    ]
                _cta_lines.append(
                    "Or switch providers with "
                    "[bold]/model <model> --provider <provider>[/]."
                )
                try:
                    ChatConsole().print(Panel(
                        "\n".join(_cta_lines),
                        title="[#CD7F32 bold]⚡ Out of credits[/]",
                        title_align="left",
                        border_style="#CD7F32",
                        box=rich_box.HORIZONTALS,
                        padding=(1, 4),
                        width=self._scrollback_box_width(),
                    ))
                except Exception:
                    pass

        # #60920: Print interruption marker with history suppressed so it
        # is never recorded in _OUTPUT_HISTORY. The marker was previously
        # appended to `response` which caused a duplicate on terminal redraw
        # when _replay_output_history replayed it. Printing it here with
        # _suspend_output_history preserves the user-visible indicator while
        # keeping _OUTPUT_HISTORY clean for replay.
        if _show_interrupt_marker:
            with _suspend_output_history():
                _cprint(f"\n{_DIM}── [Interrupted — processing new message] ──{_RST}")


        # Focus view: dim recovery line reporting what was hidden this turn
        # (and how to reveal it). Printed after the response so the turn
        # reads prompt → answer → "⋯ N tool lines hidden". Display-only;
        # resets the counter for the next turn.
        try:
            self._emit_focus_recovery_line()
        except Exception:
            pass

        # Play terminal bell when agent finishes (if enabled).
        # Works over SSH — the bell propagates to the user's terminal.
        self._ring_bell(context="turn complete")

        # Notify when iteration budget was hit
        if turn.result and not turn.result.get("completed") and not turn.result.get("interrupted"):
            _api_calls = turn.result.get("api_calls", 0)
            if _api_calls >= getattr(self.agent, "max_iterations", 500):
                _max_iter = getattr(self.agent, "max_iterations", 500)
                _cprint(
                    f"\n{_DIM}⚠ Iteration budget reached "
                    f"({_api_calls}/{_max_iter}) — "
                    f"response may be incomplete{_RST}"
                )

        # Speak response aloud if voice TTS is enabled
        # Skip batch TTS when streaming TTS already handled it
        if self._voice_tts and response and not turn.use_streaming_tts:
            self._voice_speak_response_async(response)


        # Re-queue the interrupt message (and any that arrived while we were
        # processing the first) as the next prompt for process_loop.
        # Only reached when busy_input_mode == "interrupt" (the default).
        # In "queue" mode Enter routes directly to _pending_input so this
        # block is never hit.
        if pending_message and hasattr(self, '_pending_input'):
            all_parts = [pending_message]
            while not self._interrupt_queue.empty():
                try:
                    extra = self._interrupt_queue.get_nowait()
                    if extra:
                        all_parts.append(extra)
                except queue.Empty:
                    break
            combined = "\n".join(all_parts)
            n = len(all_parts)
            preview = combined[:50] + ("..." if len(combined) > 50 else "")
            if n > 1:
                print(f"\n⚡ Sending {n} messages after interrupt: '{preview}'")
            else:
                print(f"\n⚡ Sending after interrupt: '{preview}'")
            self._pending_input.put(combined)

        # If a /steer was left over (agent finished before another tool
        # batch could absorb it), deliver it as the next user turn.
        _leftover_steer = turn.result.get("pending_steer") if turn.result else None
        if _leftover_steer and hasattr(self, '_pending_input'):
            preview = _leftover_steer[:60] + ("..." if len(_leftover_steer) > 60 else "")
            print(f"\n⏩ Delivering leftover /steer as next turn: '{preview}'")
            self._pending_input.put(_leftover_steer)

        return response
    
    # --- Protected TUI extension hooks for wrapper CLIs ---

    def _tui_process_loop(self):
        while not self._should_exit:
            try:
                # Check for pending input with timeout
                try:
                    user_input = self._pending_input.get(timeout=0.1)
                except queue.Empty:
                    # Periodic config watcher — auto-reload MCP on mcp_servers change
                    if not self._agent_running:
                        self._check_config_mcp_changes()
                        # Heal cooked-mode termios drift (lost
                        # run_in_terminal restore) before draining
                        # notifications — a drifted tty makes the CLI
                        # look dead even though the loop is healthy.
                        try:
                            self._check_termios_drift()
                        except Exception:
                            pass
                        # Check for background process notifications (completions
                        # and watch pattern matches) while agent is idle.
                        try:
                            self._drain_process_notifications("cli-idle")
                        except Exception:
                            pass
                        # Fire a due /loop wakeup while idle (defers to
                        # queued user input and active /goal loops).
                        try:
                            self._maybe_fire_loop_tick()
                        except Exception:
                            pass
                    continue

                # Voice-transcribed messages arrive wrapped in a sentinel
                # so only genuine STT output gets the voice prefix (#65827).
                is_voice_input = isinstance(user_input, _VoiceInputMessage)
                if is_voice_input:
                    user_input = user_input.text

                # Seeded -q prompts arrive wrapped in _SeededQueryMessage:
                # arbitrary launcher/script text that must be submitted
                # LITERALLY — skip slash routing, ! shell dispatch, and
                # file-drop detection for this one message.
                is_seeded_query = isinstance(user_input, _SeededQueryMessage)
                if is_seeded_query:
                    seeded = user_input
                    user_input = (
                        (seeded.text, seeded.images)
                        if seeded.images
                        else seeded.text
                    )

                if not user_input:
                    continue

                # The user has typed and submitted something, so any
                # post-resize transient suppression should end here.
                self._status_bar_suppressed_after_resize = False

                # Unpack image payload: (text, [Path, ...]) or plain str
                submit_images = []
                if isinstance(user_input, tuple):
                    user_input, submit_images = user_input

                if isinstance(user_input, str):
                    user_input = _strip_leaked_bracketed_paste_wrappers(user_input)
                    user_input, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(user_input)
                    if _had_mouse_reports:
                        self._recover_terminal_input_modes(reason="mouse reports leaked into submitted input")

                # Typed bare stop phrase while a voice chat is active ends
                # the voice chat (same semantics as SAYING "stop") instead
                # of sending the word to the agent. Voice transcripts are
                # already stop-checked at the transcription points, so this
                # only intercepts typed input.
                if not is_voice_input and self._typed_voice_stop(user_input):
                    continue

                # Check for commands — but detect dragged/pasted file paths first.
                # See _detect_file_drop() for details. Seeded -q prompts are
                # literal text: no file-drop detection, no !/slash dispatch.
                _file_drop = (
                    _detect_file_drop(user_input)
                    if isinstance(user_input, str) and not is_seeded_query
                    else None
                )
                if _file_drop:
                    _drop_path = _file_drop["path"]
                    _remainder = _file_drop["remainder"]
                    if _file_drop["is_image"]:
                        submit_images.append(_drop_path)
                        user_input = _remainder or f"[User attached image: {_drop_path.name}]"
                        _cprint(f"  📎 Auto-attached image: {_drop_path.name}")
                    else:
                        _cprint(f"  📄 Detected file: {_drop_path.name}")
                        user_input = (
                            f"[User attached file: {_drop_path}]"
                            + (f"\n{_remainder}" if _remainder else "")
                        )

                # A bare number right after a bare `/resume` prompt selects
                # that session (see #34584). Checked before chat routing so
                # the digit isn't sent to the agent as a message.
                if (
                    not _file_drop
                    and self._pending_resume_sessions
                    and isinstance(user_input, str)
                    and self._consume_pending_resume_selection(user_input)
                ):
                    continue

                # `!<command>` shell mode — run it here and loop back to
                # idle. Checked BEFORE slash routing and before the chat
                # path so nothing enters conversation history and no model
                # turn is spent. See handle_bang_shell().
                if (
                    not _file_drop
                    and not is_seeded_query
                    and isinstance(user_input, str)
                    and self.handle_bang_shell(user_input)
                ):
                    continue

                if (
                    not _file_drop
                    and not is_seeded_query
                    and isinstance(user_input, str)
                    and _looks_like_slash_command(user_input)
                ):
                    _cprint(f"\n⚙️  {user_input}")
                    try:
                        if not self.process_command(user_input):
                            self._should_exit = True
                            # Schedule app exit
                            if self._app.is_running:
                                self._app.exit()
                    except KeyboardInterrupt:
                        # Ctrl+C during a slow slash command (e.g. /skills browse,
                        # /sessions list with a large DB) should interrupt the
                        # command and return to the prompt, NOT exit the entire
                        # session. Without this guard a KeyboardInterrupt unwinds
                        # to the outer prompt_toolkit loop and the session dies.
                        _cprint("\n[dim]Command interrupted.[/dim]")
                        continue
                    # A slash handler may set a one-shot pending seed (e.g.
                    # /blueprint <name>) to be run as the next agent turn.
                    # If present, fall through to the chat path with the seed
                    # as the user message instead of looping back to idle.
                    _seed = getattr(self, "_pending_agent_seed", None)
                    if _seed:
                        self._pending_agent_seed = None
                        user_input = _seed
                    else:
                        continue

                # Expand paste references back to full content
                _paste_ref_re = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')
                paste_refs = list(_paste_ref_re.finditer(user_input)) if isinstance(user_input, str) else []
                if paste_refs:
                    user_input = self._expand_paste_references(user_input)
                print()
                self._print_user_message_preview(user_input)

                # Show image attachment count
                if submit_images:
                    n = len(submit_images)
                    _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

                # Regular chat - run agent
                self._agent_running = True
                self._interactive_turn = True
                self._pet_turn_error = False
                self._pet_reasoning = False
                self._turn_summary_begin()
                self._app.invalidate()  # Refresh status line

                try:
                    self.chat(user_input, images=submit_images or None, voice_input=is_voice_input)
                finally:
                    self._agent_running = False
                    self._spinner_text = ""
                    self._tool_start_time = 0.0
                    self._pending_tool_info.clear()
                    self._last_scrollback_tool = ""
                    self._pet_reasoning = False
                    self._pet_react_turn_end()
                    # Post-turn accounting line (display.turn_summary).
                    # Emitted after the response box, before the prompt
                    # returns, so it reads as a footer for the turn.
                    self._turn_summary_emit()
                    self._interactive_turn = False

                    self._app.invalidate()  # Refresh status line

                    # Post-turn terminal recovery (#33271): after an
                    # interrupt the prompt_toolkit renderer may have
                    # drifted from the physical terminal state — CSI 6n
                    # cursor position reports can leak as literal text
                    # (^[[19;1R), and the VT100 input parser can stall in
                    # a partial-escape state, accepting no further
                    # keystrokes.  Drain stray escape bytes from the OS
                    # input buffer and force a clean renderer redraw.
                    if self._last_turn_interrupted:
                        self._recover_terminal_after_interrupt()

                    # Re-queue any messages that arrived in _interrupt_queue
                    # while the agent was running and were never claimed by
                    # the explicit interrupt path. See
                    # _drain_interrupt_queue_to_pending_input for the full
                    # rationale. Regression of #17666 / #18760 — the drain
                    # block from the original PR #17939 was deferred as
                    # "worth its own review" and never re-landed (#20271).
                    self._drain_interrupt_queue_to_pending_input()

                    # Goal continuation: if a standing goal is active, ask
                    # the judge whether the turn satisfied it. If not, and
                    # there's no real user message already queued, push the
                    # continuation prompt back into _pending_input so the
                    # next loop iteration picks it up naturally (and any
                    # user input that arrives in between still preempts).
                    try:
                        self._maybe_continue_goal_after_turn()
                    except Exception as _goal_exc:
                        logging.debug("goal continuation hook failed: %s", _goal_exc)

                    # /loop tick completion: if the turn that just ended
                    # was a loop wakeup, evaluate it (LOOP_COMPLETE marker,
                    # --until judge, caps) and schedule the next tick.
                    try:
                        self._maybe_complete_loop_tick_after_turn()
                    except Exception as _loop_exc:
                        logging.debug("loop completion hook failed: %s", _loop_exc)

                    # Continuous voice: auto-restart recording after agent responds.
                    # Dispatch to a daemon thread so play_beep (sd.wait) and
                    # AudioRecorder.start (lock acquire) never block process_loop —
                    # otherwise queued user input would stall silently.
                    if self._voice_mode and self._voice_continuous and not self._voice_recording:
                        def _restart_recording():
                            try:
                                if self._voice_tts:
                                    self._voice_tts_done.wait(timeout=60)
                                    time.sleep(0.3)
                                # A barge-in capture already owns the mic and
                                # will submit the interruption itself.
                                if self._voice_barge_capture.is_set():
                                    return
                                self._voice_start_recording()
                                self._app.invalidate()
                            except Exception as e:
                                _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
                        threading.Thread(target=_restart_recording, daemon=True).start()

                    # Drain process notifications (completions + watch matches)
                    # that arrived while the agent was running.
                    try:
                        self._drain_process_notifications("cli-post-turn")
                    except Exception:
                        pass  # Non-fatal — don't break the main loop

            except OSError as e:
                if getattr(e, "errno", None) == errno.EIO:
                    self._mark_terminal_io_broken("process_loop")
                    logger.warning(
                        "process_loop EIO — freezing UI paints (#81521): %s",
                        e,
                    )
                    continue
                logger.warning("process_loop unhandled error (msg may be lost): %s", e)
            except Exception as e:
                if isinstance(e, OSError) and getattr(e, "errno", None) == errno.EIO:
                    self._mark_terminal_io_broken("process_loop")
                    logger.warning(
                        "process_loop EIO — freezing UI paints (#81521): %s",
                        e,
                    )
                    continue
                logger.warning("process_loop unhandled error (msg may be lost): %s", e)

    def _tui_signal_handler(self, signum, frame):
        """Handle SIGHUP/SIGTERM by triggering graceful cleanup.

        Calls ``self.agent.interrupt()`` first so the agent daemon
        thread's poll loop sees the per-thread interrupt and kills the
        tool's subprocess group via ``_kill_process`` (os.killpg).
        Without this, the main thread dies from KeyboardInterrupt and
        the daemon thread is killed with it — before it can run one
        more poll iteration to clean up the subprocess, which was
        spawned with ``os.setsid`` and therefore survives as an orphan
        with PPID=1.

        Grace window (``HERMES_SIGTERM_GRACE``, default 1.5 s) gives
        the daemon time to: detect the interrupt (next 200 ms poll) →
        call _kill_process (SIGTERM + 1 s wait + SIGKILL if needed) →
        return from _wait_for_process.  ``time.sleep`` releases the
        GIL so the daemon actually runs during the window.

        Guarded ``logger.debug``: CPython's ``logging`` module is not
        reentrant-safe.  ``Logger.isEnabledFor`` caches level results
        in ``Logger._cache``; under shutdown races the cache can be
        cleared (``_clear_cache``) or mid-mutation when the signal
        fires, raising ``KeyError: <level_int>`` (e.g. ``KeyError: 10``
        for DEBUG) inside the handler.  That KeyError then escapes
        before ``raise KeyboardInterrupt()`` can fire, which bypasses
        prompt_toolkit's normal interrupt unwind and surfaces as the
        EIO cascade from issue #13710.  Wrap the log in a bare
        ``try/except`` so the handler can never raise through it.
        """
        try:
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
        except Exception:
            pass  # never let logging raise from a signal handler (#13710 regression)
        # Shutdown intent is now unambiguous — arm the exit backstop
        # IMMEDIATELY, before the graceful unwind below.  If any step of
        # that unwind wedges (main thread parked in a syscall, prompt_toolkit
        # teardown never returning), _run_cleanup never runs and would
        # never arm its own watchdog — leaving a "dead" CLI alive for
        # minutes (#65998 class).  Never raises.
        _arm_exit_watchdog_on_shutdown_signal()
        try:
            _signal_agent = getattr(self, "agent", None)
            if _signal_agent is not None and getattr(self, "_agent_running", False):
                request_hard_interrupt(
                    _signal_agent, f"received signal {signum}"
                )
                try:
                    _grace = float(os.getenv("HERMES_SIGTERM_GRACE", "1.5"))
                except (TypeError, ValueError):
                    _grace = 1.5
                if _grace > 0:
                    time.sleep(_grace)
        except Exception:
            pass  # never block signal handling
        # Prefer a clean prompt_toolkit exit over `raise KeyboardInterrupt()`.
        # Raising KBI from a signal handler unwinds into whatever Python
        # frame the interpreter happens to be running — typically an
        # `await asyncio.sleep()` inside prompt_toolkit's
        # `_poll_output_size` coroutine.  The KBI becomes a Task
        # exception, prompt_toolkit's `_handle_exception` prints
        # "Unhandled exception in event loop" + the full traceback, and
        # parks the terminal on "Press ENTER to continue..." (#13710
        # variant — same root cause, different surface).
        #
        # `app.exit()` scheduled via `call_soon_threadsafe` lets the
        # event loop unwind normally; `app.run()` returns and our
        # existing `except (EOFError, KeyboardInterrupt, BrokenPipeError)`
        # block at the bottom of the input loop handles the rest.
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            if _app is not None:
                _loop = getattr(_app, "loop", None)
                if _loop is not None:
                    _loop.call_soon_threadsafe(_app.exit)
                    return  # clean unwind — no traceback, no ENTER pause
        except Exception:
            pass
        raise KeyboardInterrupt()  # fallback for non-prompt_toolkit contexts

    def _tui_print_startup(self):
        """Startup output: light-mode probe, banner, advisories, resume/welcome lines, tips."""
        # Detect light/dark terminal mode now (before pt grabs the tty).
        # Caches the result so subsequent _hex_to_ansi / style calls
        # don't risk re-querying mid-render.
        try:
            _detect_light_mode()
        except Exception:
            pass
        # Push the entire TUI to the bottom of the terminal so the banner,
        # responses, and prompt all appear pinned to the bottom — empty
        # space stays above, not below.  This prints enough blank lines to
        # scroll the cursor to the last row before any content is rendered.
        try:
            _term_lines = shutil.get_terminal_size().lines
            if _term_lines > 2:
                print("\n" * (_term_lines - 1), end="", flush=True)
        except Exception:
            pass

        self.show_banner()
        # Surface any active supply-chain security advisories right after the
        # welcome banner. Quiet/single-query paths call this themselves.
        self._show_security_advisories()
        # Surface a silent browser-backend downgrade (default Browser Use
        # mode with no runnable CLI) — one line, rate-limited to 24h.
        self._show_browser_backend_notice()

        # First-run: a completely unconfigured install must route into
        # provider onboarding, not a chat that cannot work. Previously a
        # keyless `hermes` accepted a message, spun for ~30s, then failed
        # with a provider-specific error the user never chose. Only fires
        # on a real TTY; quiet/single-query paths keep their own handling.
        try:
            if sys.stdin.isatty() and not self._runtime_credentials_ready():
                self._offer_first_run_setup()
        except Exception:
            logger.debug("first-run setup offer failed", exc_info=True)

        # If resuming a session, load history and display it immediately
        # so the user has context before typing their first message.
        if self._resumed and self._preload_resumed_session():
            self._display_resumed_history()

        try:
            from hermes_cli.skin_engine import get_active_skin
            _welcome_skin = get_active_skin()
            _welcome_text = _welcome_skin.get_branding("welcome", "Welcome to Hermes Agent! Type your message or /help for commands.")
            _welcome_color = _welcome_skin.get_color("banner_text", "#FFF8DC")
        except Exception:
            _welcome_text = "Welcome to Hermes Agent! Type your message or /help for commands."
            _welcome_color = "#FFF8DC"
        self._console_print(f"[{_welcome_color}]{_welcome_text}[/]")

        # Warm the /model picker's provider-models cache off-thread during this
        # idle window (banner shown, user about to type). The no-args picker
        # otherwise blocks ~1-2s on serial /v1/models fetches the first time
        # it's opened in a session. Fire-and-forget, guarded once-per-process.
        try:
            from hermes_cli.model_switch import prewarm_picker_cache_async
            prewarm_picker_cache_async()
        except Exception:
            pass

        # Pre-import the agent runtime off-thread during the same idle window.
        # The first turn otherwise pays ~1.5s of module imports on the
        # time-to-first-token critical path: `import run_agent` (~0.9s,
        # deferred by the lazy AIAgent wrapper above) plus the OpenAI SDK
        # (~0.6s, deferred until client construction). Python's import lock
        # makes this safe: if the user submits before the warm finishes, the
        # main thread simply blocks on the remaining import work instead of
        # redoing it. Skipped when agent startup is explicitly deferred
        # (Termux) — that path defers heavy work on purpose.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            def _prewarm_agent_runtime() -> None:
                try:
                    import run_agent  # noqa: F401  (imports model_tools + tool registry)
                    import openai  # noqa: F401
                except Exception:
                    logger.debug("agent runtime pre-import failed", exc_info=True)

            threading.Thread(
                target=_prewarm_agent_runtime,
                name="agent-runtime-prewarm",
                daemon=True,
            ).start()

        # Redaction opt-out warning (#17691): ON by default, loud when off.
        # The redactor snapshots its state at import time so any toggle now
        # won't affect the running process — we just want the operator to
        # see that they're running without the safety net.
        try:
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            if _redact_raw.lower() not in {"1", "true", "yes", "on"}:
                self._console_print(
                    "[bold red]⚠  Secret redaction is DISABLED[/] "
                    f"(HERMES_REDACT_SECRETS={_redact_raw}). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set "
                    "[cyan]security.redact_secrets: true[/] in config.yaml "
                    "to re-enable."
                )
        except Exception:
            pass
        # First-time OpenClaw-residue banner — fires once if ~/.openclaw/ exists
        # after an OpenClaw→Hermes migration (especially migrations done by
        # OpenClaw's own tool, which doesn't archive the source directory).
        try:
            from agent.onboarding import (
                OPENCLAW_RESIDUE_FLAG,
                detect_openclaw_residue,
                is_seen,
                mark_seen,
                openclaw_residue_hint_cli,
            )
            if not is_seen(self.config, OPENCLAW_RESIDUE_FLAG) and detect_openclaw_residue():
                try:
                    _resid_color = _welcome_skin.get_color("banner_dim", "#B8860B")
                except Exception:
                    _resid_color = "#B8860B"
                self._console_print(f"[{_resid_color}]{openclaw_residue_hint_cli()}[/]")
                try:
                    from hermes_cli.config import get_config_path as _get_cfg_path_resid
                    mark_seen(_get_cfg_path_resid(), OPENCLAW_RESIDUE_FLAG)
                except Exception:
                    pass  # best-effort — banner will fire again next session
        except Exception:
            pass  # banner is non-critical — never break startup
        self._print_random_tip()

        # Curator — kick off a background skill-maintenance pass on startup
        # if the schedule says we're due.  Runs in a daemon thread so it
        # never blocks the interactive loop.  Best-effort; any failure is
        # swallowed to avoid breaking session startup.
        try:
            from agent.curator import maybe_run_curator
            maybe_run_curator(
                idle_for_seconds=float("inf"),  # CLI startup = fully idle
                on_summary=lambda msg: self._console_print(
                    f"[dim #6b7684]💾 {msg}[/]"
                ),
            )
        except Exception:
            pass

        # Skill sync — best-effort periodic pull, piggy-backing on the
        # curator tick. Inert unless the access gate is open and a sync base
        # URL is configured; swallows all errors so it never blocks startup.
        try:
            from tools.skills_sync_client import maybe_pull_skills
            maybe_pull_skills()
        except Exception:
            pass

        # Org-shared skills — pull the organisation's approved set into the
        # read-only mirror. Gated on real org membership: resolve_org_identity
        # requires an org role on the token, which is only issued for
        # multi-member organisations, so a solo account never reaches the
        # network here. Fail-quiet, exactly like the personal pull above.
        try:
            from tools.skills_sync_client import maybe_pull_org_skills
            maybe_pull_org_skills()
        except Exception:
            pass
        _skills_for_line = self.preloaded_skills or list(
            getattr(self, "_preload_skills_requested", []) or []
        )
        if _skills_for_line and not self._startup_skills_line_shown:
            # When the background --skills preload hasn't been folded in yet
            # (it joins at agent init), show the REQUESTED names — identical
            # to the loaded set except for typo'd names, which warn later.
            skills_label = ", ".join(_skills_for_line)
            self._console_print(
                f"[bold {_accent_hex()}]Activated skills:[/] {skills_label}"
            )
            self._startup_skills_line_shown = True
        self._console_print()

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        if not self._claim_active_session("cli"):
            return

        self._tui_print_startup()
        self._tui_init_run_state()
        kb = self._tui_build_key_bindings()
        layout, style = self._tui_build_layout(kb)

        # Select CPR-disabled output when _terminal_may_leak_cpr() says so
        # (POSIX local + SSH; Windows keeps PT default — see helper docs).
        # None falls back to prompt_toolkit's default output; input scrubbing
        # in _strip_leaked_terminal_responses still guards residual leaks.
        _cpr_disabled_output = _select_classic_cli_pt_output(sys.stdout)

        # Kitty placeholders encode their image id in exact foreground RGB, so
        # placeholder-capable terminals (kitty/Ghostty) use 24-bit color for
        # the whole prompt_toolkit application — quantizing only that pane
        # is not supported. WezTerm is excluded: it is not placeholder-capable.
        # ColorDepth is imported here (not at module load) so tests that stub
        # ``prompt_toolkit`` as a MagicMock can still import cli.
        color_depth_kw = {}
        if pet_render.supports_kitty_placeholders():
            from prompt_toolkit.output import ColorDepth

            color_depth_kw = {"color_depth": ColorDepth.DEPTH_24_BIT}
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            **({"output": _cpr_disabled_output} if _cpr_disabled_output is not None else {}),
            **color_depth_kw,
            # Read from display.cli_refresh_interval (default 0 = disabled).
            # When non-zero, prompt_toolkit redraws the UI on this cadence
            # during idle, keeping wall-clock status-bar read-outs ticking.
            # Set to 0 to suppress background redraws entirely — avoids
            # fighting terminal auto-scroll in non-fullscreen mode (Xshell,
            # iTerm2, Windows Terminal). See #48309.
            refresh_interval=float(CLI_CONFIG.get("display", {}).get("cli_refresh_interval", 0)),
            # Erase the live bottom chrome (status bar, input box, separator
            # rules) on exit instead of freezing a final copy into scrollback.
            # Without this, prompt_toolkit's render_as_done teardown repaints
            # the chrome one last time and leaves it stranded above the exit
            # summary — so a dead status bar + empty prompt sit between the
            # conversation transcript and the "Resume this session" block, and
            # stack with the next session's UI on resume (#38252). The actual
            # conversation transcript is printed through patch_stdout into
            # normal scrollback and is unaffected; only the managed chrome is
            # erased. Applies to every exit path (/exit, /quit, EOF, Ctrl+C).
            erase_when_done=True,
            **({'cursor': _STEADY_CURSOR} if _STEADY_CURSOR is not None else {}),
        )
        _disable_prompt_toolkit_cpr_warning(app)
        app.after_render += self._pet_flush_kitty_frame
        self._app = app  # Store reference for clarify_callback

        # ── Fix ghost status-bar lines on terminal resize ──────────────
        # Resize handling: monkey-patch prompt_toolkit's _output_screen_diff
        # to suppress the deliberate "reserve vertical space" scroll-up.
        #
        # Background: prompt_toolkit's renderer (renderer.py L232-242)
        # explicitly moves the cursor to the bottom of the canvas after
        # painting "to make sure the terminal scrolls up, even when the
        # lower lines of the canvas just contain whitespace".  In
        # non-fullscreen mode this scrolls chrome content (status bar,
        # input rules) into terminal scrollback on every render.  When
        # the terminal column-shrinks, the emulator reflows the previously
        # rendered full-width rows into multiple narrower rows that get
        # pushed up — leaving ghost duplicates AND polluting scrollback.
        # Same issue as pt #29 (open since 2014), #1675, #1933.
        #
        # Surgical fix: wrap _output_screen_diff so that when its internal
        # `if current_height > previous_screen.height` branch fires (the
        # one that does the bottom-cursor-move), we make it fall through
        # by inflating previous_screen.height first.
        try:
            import prompt_toolkit.renderer as _pt_renderer
            from prompt_toolkit.renderer import _output_screen_diff as _orig_osd

            if not getattr(_pt_renderer, "_hermes_osd_patched", False):
                def _patched_output_screen_diff(
                    app, output, screen, current_pos, color_depth,
                    previous_screen, last_style, is_done, full_screen,
                    attrs_for_style_string, style_string_has_style,
                    size, previous_width,
                ):
                    """Wraps pt's _output_screen_diff to suppress the
                    reserve-vertical-space scroll (renderer.py L232-242).

                    Strategy: ONLY when previous_screen is non-None and
                    its current height is genuinely smaller than the new
                    screen's height, inflate it to match.  This prevents
                    the bottom-cursor-move at L242 without changing any
                    other code path's behavior.

                    Critical: do NOT replace a None previous_screen with
                    a fresh Screen() on the happy path — that would skip
                    the proper reset_attributes()+erase_down() at L178-185
                    which fires when previous_screen is None (first-paint /
                    width-change).  Without that reset, ANSI styles
                    leak between renders.

                    Safety net: if the diff crashes with AttributeError /
                    TypeError (corrupt previous_screen after tmux attach —
                    "'cell' object has no attribute 'char'"), retry once
                    with previous_screen=None so pt takes the first-paint
                    erase path instead of wedging the event loop.
                    """
                    return _hermes_call_output_screen_diff(
                        _orig_osd,
                        app, output, screen, current_pos, color_depth,
                        previous_screen, last_style, is_done, full_screen,
                        attrs_for_style_string, style_string_has_style,
                        size, previous_width,
                    )

                _pt_renderer._output_screen_diff = _patched_output_screen_diff
                _pt_renderer._hermes_osd_patched = True
        except Exception:
            pass

        # Apply bracketed-paste timeout recovery so torn ESC[201~ end marks
        # don't permanently freeze the input (issue #16263). Idempotent.
        _apply_bracketed_paste_timeout_patch()

        self._install_resize_recovery(app)

        spinner_thread = threading.Thread(target=self._tui_spinner_loop, daemon=True)
        spinner_thread.start()
        
        # Background thread to process inputs and run agent
        
        # Start processing thread
        process_thread = threading.Thread(target=self._tui_process_loop, daemon=True)
        process_thread.start()

        # Wake word ("Hey Hermes") — start the always-on hotword listener if
        # enabled. Off-thread so a first-run engine install never blocks the
        # prompt; best-effort, so deps/mic/key gaps are surfaced, never fatal.
        threading.Thread(target=self._tui_wake_startup, daemon=True, name="wake-startup").start()

        # Register atexit cleanup so resources are freed even on unexpected exit
        atexit.register(_run_cleanup)
        
        # Register signal handlers for graceful shutdown on SSH disconnect / SIGTERM
        
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, self._tui_signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, self._tui_signal_handler)

            # Windows: install a SIGINT handler that absorbs the signal
            # instead of letting Python's default handler raise
            # KeyboardInterrupt in MainThread. Windows Terminal / Win32
            # delivers spurious CTRL_C_EVENT to the hermes process when
            # child processes are spawned from background threads (agent
            # subprocess Popen path). The default Python SIGINT handler
            # would then unwind prompt_toolkit's app.run(), trigger
            # _run_cleanup mid-turn, and close browser sessions mid-open
            # — causing "Daemon process exited during startup" errors.
            #
            # The handler is a silent no-op. Real user Ctrl+C still works
            # because prompt_toolkit binds c-c at the TUI layer and never
            # reaches this OS-signal path. This matches how Claude Code
            # handles the same Windows quirk (cancellation is driven by
            # the TUI key handler, not by OS signals).
            #
            # POSIX: leave the default SIGINT handler alone. prompt_toolkit
            # installs its own handler there and it works as expected.
            if sys.platform == "win32":
                def _sigint_absorb(signum, frame):
                    # Absorb silently. Do NOT call agent.interrupt() here:
                    # Windows fires spurious CTRL_C_EVENT whenever a
                    # background thread spawns a .cmd subprocess, and
                    # interrupt() would inject a fake user message each
                    # time. Real user Ctrl+C routes through prompt_toolkit's
                    # own c-c key binding at the TUI layer (same pattern as
                    # Claude Code's Windows handling).
                    return
                _signal.signal(_signal.SIGINT, _sigint_absorb)
        except Exception:
            pass  # Signal handlers may fail in restricted environments
        
        # Install a custom asyncio exception handler that suppresses the
        # "Event loop is closed" RuntimeError from httpx transport cleanup
        # and the "0 is not registered" KeyError from broken stdin (#6393).
        # The RuntimeError fix is defense-in-depth — the primary fix is
        # neuter_async_httpx_del which disables __del__ entirely.  The
        # KeyError fix handles macOS + uv-managed Python environments where
        # fd 0 is not reliably available to the asyncio selector.

        # Validate stdin before launching prompt_toolkit — on macOS with
        # uv-managed Python, fd 0 can be invalid or unregisterable with the
        # asyncio selector, causing "KeyError: '0 is not registered'" (#6393).
        try:
            os.fstat(0)
        except OSError:
            print(
                "Error: stdin (fd 0) is not available.\n"
                "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
            )
            _run_cleanup()
            self._print_exit_summary()
            return

        # On macOS with uv-managed Python, kqueue's selector cannot register
        # fd 0, raising OSError(EINVAL) from kqueue.control() when prompt_toolkit
        # calls loop.add_reader (#6393). Probe kqueue and, if it can't watch
        # stdin, switch to a SelectSelector-backed event loop policy.
        if sys.platform == "darwin":
            try:
                import selectors as _selectors
                if hasattr(_selectors, "KqueueSelector"):
                    _kq = _selectors.KqueueSelector()
                    try:
                        _kq.register(0, _selectors.EVENT_READ)
                        _kq.unregister(0)
                    finally:
                        _kq.close()
            except (OSError, ValueError, KeyError):
                import asyncio as _aio_probe
                import selectors as _selectors

                class _SelectEventLoopPolicy(_aio_probe.DefaultEventLoopPolicy):
                    def new_event_loop(self):
                        return _aio_probe.SelectorEventLoop(_selectors.SelectSelector())

                _aio_probe.set_event_loop_policy(_SelectEventLoopPolicy())

        # Run the application with patch_stdout for proper output handling
        try:
            with patch_stdout():
                # Set the custom handler on prompt_toolkit's event loop
                try:
                    import asyncio as _aio
                    # Use get_running_loop() to avoid DeprecationWarning on
                    # Python 3.10+ when called outside an async context.
                    _loop = _aio.get_running_loop()
                    _loop.set_exception_handler(self._tui_suppress_closed_loop_errors)
                except RuntimeError:
                    pass  # No running loop -- nothing to patch
                except Exception:
                    pass
                # The app enables focus reporting + mouse tracking; record that
                # so _run_cleanup resets them on exit (#36823). When multiline
                # shortcuts are on, also ask supported terminals (e.g. iTerm2)
                # to report modified keys distinctly (kitty protocol +
                # modifyOtherKeys); the cleanup reset pops both modes.
                _mark_tui_input_modes_active()
                if self._tui_multiline_shortcuts:
                    _enable_extended_enter_keys(app.output)
                # Drive the petdex mascot animation (no-op when no pet enabled).
                self._pet_start_anim()
                app.run()
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            pass
        except (KeyError, OSError) as _stdin_err:
            # Catch selector registration failures from broken stdin (#6393)
            # and I/O errors from broken stdout during interrupt (#13710).
            _errno = getattr(_stdin_err, "errno", None) if isinstance(_stdin_err, OSError) else None
            _msg = str(_stdin_err)
            if _errno == errno.EIO:
                pass  # suppress broken-stdout I/O errors on interrupt (#13710)
            elif (
                _errno in {errno.EINVAL, errno.EBADF}
                or "is not registered" in _msg
                or "Bad file descriptor" in _msg
                or "Invalid argument" in _msg
            ):
                print(
                    f"\nError: stdin is not usable ({_stdin_err}).\n"
                    "This can happen with certain Python installations (e.g. uv-managed cPython on macOS)\n"
                    "where kqueue cannot register fd 0.\n"
                    "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
                )
            else:
                raise
        finally:
            self._tui_shutdown()

        # Deferred relaunch: /update sets _pending_relaunch so the exec
        # happens here — after prompt_toolkit has exited and fully restored
        # terminal modes — rather than from the background process_loop
        # thread (which would skip terminal cleanup on POSIX and only exit
        # the worker thread on Windows).
        if getattr(self, '_pending_relaunch', None):
            from hermes_cli.relaunch import relaunch
            relaunch(self._pending_relaunch, preserve_inherited=False)

    def _tui_shutdown(self):
        """Teardown after the prompt_toolkit app exits: interrupt the agent, stop voice/pet, persist + close the session, run cleanup, print the exit summary."""
        self._should_exit = True
        self._pet_stop_anim()
        # Immediate feedback: prompt_toolkit has just torn down the input
        # box + status bar, so without a line here the terminal sits
        # silent for the whole cleanup window (session flush, memory
        # shutdown, MCP/browser/terminal teardown) and the exit looks
        # hung. Print before any potentially-slow step.
        try:
            print(f"{_DIM}Shutting down… (finalizing session){_RST}", flush=True)
        except Exception:
            pass
        # Interrupt the agent immediately so its daemon thread stops making
        # API calls and exits promptly (agent_thread is daemon, so the
        # process will exit once the main thread finishes, but interrupting
        # avoids wasted API calls and lets run_conversation clean up).
        if self.agent and getattr(self, '_agent_running', False):
            try:
                request_hard_interrupt(self.agent)
            except Exception:
                pass
        # Shut down voice recorder (release persistent audio stream)
        if hasattr(self, '_voice_recorder') and self._voice_recorder:
            try:
                self._voice_recorder.shutdown()
            except Exception:
                pass
            self._voice_recorder = None
        # Clean up old temp voice recordings
        try:
            from tools.voice_mode import cleanup_temp_recordings
            cleanup_temp_recordings()
        except Exception:
            pass
        # Unregister callbacks to avoid dangling references
        set_sudo_password_callback(None)
        set_approval_callback(None)
        set_secret_capture_callback(None)
        # Flush any in-memory turn transcript before marking the session
        # closed.  On SIGHUP/SIGTERM/window close the agent thread may not
        # reach its normal run_conversation() persistence path before the
        # daemon thread is reaped.
        self._persist_active_session_before_close()

        # Close session in SQLite
        if hasattr(self, '_session_db') and self._session_db and self.agent:
            try:
                self._session_db.end_session(self.agent.session_id, "cli_close")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Could not close session in DB: %s", e)
            # Started-and-immediately-quit sessions never gained content;
            # drop the empty row so /resume and `hermes sessions list`
            # stay clean (gemini-cli#27770 port). No-op for resumed or
            # titled sessions and anything with messages or children.
            if not getattr(self, '_delete_session_on_exit', False):
                try:
                    self._discard_session_if_empty(self.agent.session_id)
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not prune empty session: %s", e)
            # /exit --delete: also remove the current session's transcripts
            # and SQLite history. Ported from google-gemini/gemini-cli#19332.
            if getattr(self, '_delete_session_on_exit', False):
                try:
                    from hermes_constants import get_hermes_home as _ghh
                    _sessions_dir = _ghh() / "sessions"
                    _sid = self.agent.session_id
                    if self._session_db.delete_session(_sid, sessions_dir=_sessions_dir):
                        _cprint(f"  {_DIM}✓ Session {_escape(_sid)} deleted{_RST}")
                    else:
                        _cprint(f"  {_DIM}✗ Session {_escape(_sid)} not found for deletion{_RST}")
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not delete session on exit: %s", e)
        # Plugin hook: on_session_end — safety net for interrupted exits.
        # run_conversation() already fires this per-turn on normal completion,
        # so only fire here if the agent was mid-turn (_agent_running) when
        # the exit occurred, meaning run_conversation's hook didn't fire.
        if self.agent and getattr(self, '_agent_running', False):
            try:
                from hermes_cli.lifecycle import invoke_hook as _invoke_hook
                _invoke_hook(
                    "on_session_end",
                    session_id=self.agent.session_id,
                    completed=False,
                    interrupted=True,
                    model=getattr(self.agent, 'model', None),
                    platform=getattr(self.agent, 'platform', None) or "cli",
                    reason="shutdown",
                )
            except Exception:
                pass
        _run_cleanup()
        self._print_exit_summary()
        self._release_active_session()


# ============================================================================
# Main Entry Point
# ============================================================================

def _run_kanban_goal_loop_q(cli: "HermesCLI", first_response: str) -> None:
    """Drive a kanban goal_mode worker through the Ralph-style goal loop.

    Called from the quiet single-query path AFTER the worker's first turn,
    only when ``HERMES_KANBAN_GOAL_MODE`` is set (dispatcher-spawned
    goal_mode card). Wires the worker's ``run_conversation`` and the kanban
    DB into ``goals.run_kanban_goal_loop``. All errors are swallowed by the
    caller — a broken goal loop must never wedge a worker, the dispatcher's
    claim TTL / crash detection is the backstop.
    """
    import os as _os

    task_id = (_os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return
    worker_run_id = None
    raw_run_id = (_os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if raw_run_id:
        try:
            worker_run_id = int(raw_run_id)
        except ValueError:
            logger.warning("invalid HERMES_KANBAN_RUN_ID=%r", raw_run_id)

    from hermes_cli import kanban_db as _kb
    from hermes_cli.goals import run_kanban_goal_loop as _run_loop, DEFAULT_MAX_TURNS as _DEF_TURNS

    # Resolve goal text from the card (title + body = the acceptance
    # criteria the judge evaluates against).
    conn = _kb.connect()
    try:
        task = _kb.get_task(conn, task_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if task is None:
        return

    goal_parts = [task.title or ""]
    if task.body:
        goal_parts.append(task.body)
    goal_text = "\n\n".join(p for p in goal_parts if p).strip()
    if not goal_text:
        return

    max_turns = task.goal_max_turns or _DEF_TURNS

    def _run_turn(prompt: str) -> str:
        result = cli.agent.run_conversation(
            user_message=prompt,
            conversation_history=cli.conversation_history,
        )
        # Keep session_id in sync if mid-run compression rotated it.
        if (
            getattr(cli.agent, "session_id", None)
            and cli.agent.session_id != cli.session_id
        ):
            cli.session_id = cli.agent.session_id
        resp = result.get("final_response", "") if isinstance(result, dict) else str(result)
        if resp:
            print(resp)
        return resp or ""

    def _task_status() -> "str | None":
        c = _kb.connect()
        try:
            return _kb.goal_run_status(c, task_id, worker_run_id)
        finally:
            try:
                c.close()
            except Exception:
                pass

    def _block(reason: str) -> None:
        c = _kb.connect()
        try:
            _kb.block_task(
                c,
                task_id,
                reason=reason,
                expected_run_id=worker_run_id,
            )
        finally:
            try:
                c.close()
            except Exception:
                pass

    _run_loop(
        task_id=task_id,
        goal_text=goal_text,
        run_turn=_run_turn,
        task_status_fn=_task_status,
        block_fn=_block,
        max_turns=max_turns,
        first_response=first_response or "",
        log=lambda m: logger.info("%s", m),
    )


def _run_quiet_single_query(cli, effective_query):
    """Quiet (-Q) one-shot turn: run, print the response (stderr for errors/session_id), then sys.exit with the automation exit code."""
    try:
        result = cli.agent.run_conversation(
            user_message=effective_query,
            conversation_history=cli.conversation_history,
        )
    except KeyboardInterrupt:
        _emit_interrupted_session_end(cli, reason="keyboard_interrupt")
        print(f"\nsession_id: {cli.session_id}", file=sys.stderr)
        sys.exit(130)
    # Sync session_id if mid-run compression created a
    # continuation session. The exit line below reports
    # session_id to stderr for automation wrappers; without
    # this sync it would point at the ended parent.
    if (
        getattr(cli.agent, "session_id", None)
        and cli.agent.session_id != cli.session_id
    ):
        cli.session_id = cli.agent.session_id
    response = result.get("final_response", "") if isinstance(result, dict) else str(result)
    # Surface backend errors that produced no visible output
    # (e.g. invalid model slug → provider 4xx). Mirrors the
    # interactive CLI path. Write to stderr so piped stdout
    # stays clean for automation wrappers.
    if (
        not response
        and isinstance(result, dict)
        and result.get("error")
        and (result.get("failed") or result.get("partial"))
    ):
        print(f"Error: {result['error']}", file=sys.stderr)
    elif response:
        print(response)

    # Kanban goal-loop mode: a worker spawned for a
    # goal_mode card keeps working in THIS session until an
    # auxiliary judge agrees the card is done, the worker
    # terminates the task itself, or the turn budget runs
    # out (→ sticky block). Gated on the env vars the
    # dispatcher sets in `_default_spawn`; a no-op for every
    # normal worker and every non-kanban `-q` run.
    if os.environ.get("HERMES_KANBAN_GOAL_MODE") == "1":
        try:
            _run_kanban_goal_loop_q(cli, response)
        except Exception as _goal_exc:
            logger.debug("kanban goal loop failed: %s", _goal_exc)

    # Session ID goes to stderr so piped stdout is clean.
    print(f"\nsession_id: {cli.session_id}", file=sys.stderr)

    # Ensure proper exit code for automation wrappers.
    #
    # Kanban workers get a special case: when the run failed
    # purely because the provider rate-limited / exhausted
    # quota (not because the task itself is broken), exit with
    # the EX_TEMPFAIL sentinel instead of the generic 1. The
    # dispatcher's reap classifier maps that code to a
    # ``rate_limited`` exit and releases the task back to
    # ``ready`` WITHOUT incrementing the failure counter, so a
    # 5-hour quota window can't trip the circuit breaker and
    # permanently block the card. Non-kanban runs keep the
    # plain 0/1 contract automation wrappers expect.
    _exit_code = 0
    if isinstance(result, dict) and result.get("failed"):
        _exit_code = 1
        if os.environ.get("HERMES_KANBAN_TASK") and result.get(
            "failure_reason"
        ) in ("rate_limit", "billing"):
            try:
                from hermes_cli.kanban_db import (
                    KANBAN_RATE_LIMIT_EXIT_CODE as _RL_CODE,
                )
                _exit_code = _RL_CODE
            except Exception:
                _exit_code = 1
    sys.exit(_exit_code)


def _route_single_query_images(cli, query, effective_query, single_query_images, single_query_image_urls):
    """Attach one-shot images natively when the model supports vision, else pre-describe them as text."""
    if single_query_images or single_query_image_urls:
        # Honour the same image-routing decision used by the
        # interactive path. With a vision-capable model (incl.
        # custom-provider models declared via
        # `model.supports_vision: true`), attach images natively
        # as image_url content parts. Otherwise fall back to the
        # text-pipeline (vision_analyze pre-description).
        _img_mode = "text"
        _build_parts = None
        try:
            from agent.image_routing import (
                build_native_content_parts as _build_parts,  # noqa: F811
            )
            from agent.image_routing import decide_image_input_mode
            from hermes_cli.config import load_config

            _img_mode = decide_image_input_mode(
                (cli.provider or "").strip(),
                (cli.model or "").strip(),
                load_config(),
                requested_provider=(
                    cli.requested_provider or ""
                ).strip(),
            )
        except Exception:
            _img_mode = "text"

        if _img_mode == "native" and _build_parts is not None:
            try:
                _parts, _skipped = _build_parts(
                    query if isinstance(query, str) else "",
                    [str(p) for p in single_query_images],
                    image_urls=list(single_query_image_urls) or None,
                )
                if any(p.get("type") == "image_url" for p in _parts):
                    effective_query = _parts
                else:
                    # All images unreadable — text fallback.
                    # ``_preprocess_images_with_vision`` only knows
                    # about local files; URLs would be lost there,
                    # so keep the original query text intact when
                    # only URLs were supplied.
                    if single_query_images:
                        effective_query = cli._preprocess_images_with_vision(
                            query, single_query_images, announce=False,
                        )
            except Exception:
                if single_query_images:
                    effective_query = cli._preprocess_images_with_vision(
                        query, single_query_images, announce=False,
                    )
        elif single_query_images:
            effective_query = cli._preprocess_images_with_vision(
                query,
                single_query_images,
                announce=False,
            )
    return effective_query


def _collect_kanban_task_images(single_query_images):
    """Kanban workers: scan the task body for image paths/URLs and add them to the first turn's attachments."""
    # Kanban workers spawn with ``hermes chat -q "work kanban task <id>"``;
    # the actual task description lives in the task body. Mirror the
    # gateway/CLI behaviour for inbound images by scanning the body for
    # local image paths and http(s) image URLs and attaching them to the
    # worker's first turn. Without this, users who paste a screenshot
    # path or URL into a kanban task body never get it routed to the
    # model's vision input.
    single_query_image_urls: list[str] = []
    _kanban_task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if _kanban_task_id:
        try:
            from hermes_cli import kanban_db as _kb
            from agent.image_routing import extract_image_refs as _extract_refs

            _conn = _kb.connect()
            try:
                _task = _kb.get_task(_conn, _kanban_task_id)
            finally:
                try:
                    _conn.close()
                except Exception:
                    pass
            _body = getattr(_task, "body", "") if _task is not None else ""
            if _body:
                _kb_paths, _kb_urls = _extract_refs(_body)
                if _kb_paths:
                    # Dedupe against any --image the user already passed.
                    _seen = {str(p) for p in single_query_images}
                    for _p in _kb_paths:
                        if _p not in _seen:
                            _seen.add(_p)
                            single_query_images.append(Path(_p))
                if _kb_urls:
                    single_query_image_urls.extend(_kb_urls)
        except Exception as _exc:
            # Best-effort enrichment; never block worker startup on it.
            logger.debug("kanban image-ref extraction failed: %s", _exc)
    return single_query_image_urls


def _install_single_query_signal_handlers(cli):
    """Route SIGINT/SIGTERM/SIGHUP through agent.interrupt() (worker threads see it) before unwinding; kanban workers hard-exit."""
    # Also install signal handlers in single-query / `-q` mode.  Interactive
    # mode registers its own inside HermesCLI.run(), but `-q` runs
    # cli.agent.run_conversation() below and AIAgent spawns worker threads
    # for tools — so when SIGTERM arrives on the main thread, raising
    # KeyboardInterrupt only unwinds the main thread, not the worker
    # running _wait_for_process.  Python then exits, the child subprocess
    # (spawned with os.setsid, its own process group) is reparented to
    # init and keeps running as an orphan.
    #
    # Fix: route SIGTERM/SIGHUP through agent.interrupt() which sets the
    # per-thread interrupt flag the worker's poll loop checks every 200 ms.
    # Give the worker a grace window to call _kill_process (SIGTERM to the
    # process group, then SIGKILL after 1 s), then raise KeyboardInterrupt
    # so main unwinds normally.  HERMES_SIGTERM_GRACE overrides the 1.5 s
    # default for debugging.
    def _signal_handler_q(signum, frame):
        logger.debug("Received signal %s in single-query mode", signum)
        # Arm the exit backstop now that shutdown intent is unambiguous —
        # covers wedges in the unwind below that would otherwise leave the
        # process alive with no watchdog (#65998 class). Never raises.
        _arm_exit_watchdog_on_shutdown_signal()
        try:
            _agent = getattr(cli, "agent", None)
            if _agent is not None:
                request_hard_interrupt(_agent, f"received signal {signum}")
                try:
                    _grace = float(os.getenv("HERMES_SIGTERM_GRACE", "1.5"))
                except (TypeError, ValueError):
                    _grace = 1.5
                if _grace > 0:
                    time.sleep(_grace)
        except Exception:
            pass  # never block signal handling
        # Kanban worker exit path (#28181): SIGTERM hits a dispatcher-spawned
        # worker that's likely in a non-daemon thread waiting on a child
        # subprocess in _wait_for_process. Raising KeyboardInterrupt only
        # unwinds the main thread; the worker thread keeps running, the
        # process gets reparented to init, and the dispatcher's _pid_alive
        # check returns True forever — task stuck in 'running' indefinitely.
        # Skip the controlled-unwind dance and call os._exit(0) so the kernel
        # reclaims the PID immediately and detect_crashed_workers can reclaim
        # the stale claim on the next tick. Flush logging + stdout/stderr
        # first so the final debug trace isn't lost; SIGALRM deadman guards
        # the flush against any rare blocking-I/O case (the reporter measured
        # flush in <1ms; the alarm is a failsafe, not the common path).
        if os.environ.get("HERMES_KANBAN_TASK"):
            try:
                import signal as _sig_mod
                if hasattr(_sig_mod, "SIGALRM"):
                    # Cancel any pre-existing alarm to avoid colliding with
                    # caller-installed timers.
                    _sig_mod.signal(_sig_mod.SIGALRM, lambda *_: os._exit(0))
                    _sig_mod.alarm(5)
            except Exception:
                pass
            # os._exit(0) skips atexit AND SessionDB's token-drain hook, so
            # flush + finalize the session store here or the worker's turn
            # (and its usage deltas) never become durable (#88583 / #50881
            # class). Best-effort under the SIGALRM deadman above.
            try:
                _flush_one_shot_session_store(cli)
            except Exception:
                pass
            try:
                import logging as _lg
                _lg.shutdown()
            except Exception:
                pass
            for _stream in (sys.stdout, sys.stderr):
                try:
                    _stream.flush()
                except Exception:
                    pass
            os._exit(0)
        raise KeyboardInterrupt()
    try:
        import signal as _signal
        _signal.signal(_signal.SIGINT, _signal_handler_q)
        _signal.signal(_signal.SIGTERM, _signal_handler_q)
        if hasattr(_signal, "SIGHUP"):
            _signal.signal(_signal.SIGHUP, _signal_handler_q)
    except Exception:
        pass  # signal handler may fail in restricted environments


def main(
    query: str = None,
    q: str = None,
    oneshot: bool = False,
    image: str = None,
    toolsets: str = None,
    skills: str | list[str] | tuple[str, ...] = None,
    model: str = None,
    provider: str = None,
    reasoning: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = None,
    run_budget: float = None,
    verbose: Optional[bool] = None,
    quiet: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    gateway: bool = False,
    resume: str = None,
    worktree: bool = False,
    w: bool = False,
    checkpoints: bool = False,
    pass_session_id: bool = False,
    ignore_user_config: bool = False,
    ignore_rules: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Query to run. On a real TTY this seeds an interactive session
            (submitted literally as the first turn); with --oneshot/-Q or a
            non-TTY it answers and exits. Alias: -q
        q: Shorthand for --query
        oneshot: With -q: force the legacy answer-and-exit single-query mode
            even on a TTY.
        image: Optional local image path to attach to a single query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        skills: Comma-separated or repeated list of skills to preload for the session
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
        reasoning: Reasoning effort for this run (none|minimal|low|medium|high|xhigh|max|ultra). Overrides agent.reasoning_effort.
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        resume: Resume a previous session by its ID (e.g., 20260225_143052_a1b2c3)
        worktree: Run in an isolated git worktree (for parallel agents). Alias: -w
        w: Shorthand for --worktree
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py --skills hermes-agent-dev,github-auth
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py -q "Describe this" --image ~/storage/shared/Pictures/cat.png
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
        python cli.py -w                         # Start in isolated git worktree
        python cli.py -w -q "Fix issue #123"     # Single query in worktree
    """
    global _active_worktree

    # Force UTF-8 stdio on Windows before any banner/print() runs — the
    # Rich console prints Unicode box-drawing characters that would
    # UnicodeEncodeError on cp1252.  No-op on Linux/macOS.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    # Signal to terminal_tool that we're in interactive mode
    # This enables interactive sudo password prompts with timeout
    os.environ["HERMES_INTERACTIVE"] = "1"
    
    # Handle gateway mode (messaging + cron)
    if gateway:
        import asyncio
        # Startup-liveness watchdog (OOF-298): this legacy entry point must
        # be covered too — arm before importing the gateway graph.
        try:
            from hermes_startup_watchdog import arm_startup_watchdog
            arm_startup_watchdog()
        except Exception:
            pass
        from gateway.run import start_gateway
        print("Starting Hermes Gateway (messaging platforms)...")
        asyncio.run(start_gateway())
        return

    # Skip worktree for list commands (they exit immediately)
    if not list_tools and not list_toolsets:
        # ── Git worktree isolation (#652) ──
        # Create an isolated worktree so this agent instance doesn't collide
        # with other agents working on the same repo.
        use_worktree = worktree or w or CLI_CONFIG.get("worktree", False)
        wt_info = None
        if use_worktree:
            # Overlap tool discovery with the network/subprocess-bound
            # worktree setup (base fetch + parallel `git worktree add`
            # release the GIL for most of their wall time). show_banner()
            # then hits the warm cache instead of paying ~0.4s serially.
            # Only done on the -w path: on plain `hermes` there is no I/O
            # wait to hide and the extra thread just contends for CPU.
            def _prewarm_tools() -> None:
                try:
                    import model_tools as _mt
                    _mt.get_tool_definitions(quiet_mode=True)
                except Exception:
                    logger.debug("tool prewarm failed", exc_info=True)

            threading.Thread(
                target=_prewarm_tools, name="tool-prewarm", daemon=True
            ).start()
            # Worktree creation itself (~0.2-0.6s of git subprocess wall
            # time) runs concurrently with the rest of startup; join right
            # after HermesCLI construction, before anything consumes
            # TERMINAL_CWD / wt_info. Failure semantics preserved: setup
            # failure still aborts the session (checked at join).
            _sync_base = CLI_CONFIG.get("worktree_sync", True)
            _wt_result: dict = {}

            def _create_worktree() -> None:
                try:
                    _wt_result["info"] = _setup_worktree(sync_base=_sync_base)
                except Exception:
                    logger.debug("worktree setup failed", exc_info=True)
                    _wt_result["info"] = None

            _wt_thread = threading.Thread(
                target=_create_worktree, name="worktree-setup", daemon=True
            )
            _wt_thread.start()

            def _join_worktree() -> Optional[Dict[str, str]]:
                _wt_thread.join(timeout=120)
                info = _wt_result.get("info")
                if info:
                    global _active_worktree
                    _active_worktree = info
                    os.environ["TERMINAL_CWD"] = info["path"]
                    atexit.register(_cleanup_worktree, info)
                    # Prune stale worktrees from crashed/killed sessions in
                    # the background — pure GC, nothing downstream depends
                    # on it. Ordered AFTER _setup_worktree so the two never
                    # race on git's worktrees metadata; the new tree itself
                    # is immune to reaping (<24h age gate + live pid lock).
                    _repo = _git_repo_root()
                    if _repo:
                        def _worktree_maintenance(repo: str) -> None:
                            _prune_stale_worktrees(repo)
                            # Same pass: repack when packs sprawl, so object
                            # lookups (and the next `worktree add`) stay fast
                            # on multi-agent boxes. After the pruner so the
                            # repack sees final refs.
                            _maintain_pack_health(repo)

                        threading.Thread(
                            target=_worktree_maintenance,
                            args=(_repo,),
                            name="worktree-prune",
                            daemon=True,
                        ).start()
                return info
        else:
            _join_worktree = None
    else:
        _join_worktree = None
    wt_info = None
    
    # Handle query shorthand
    query = query or q
    
    # Parse toolsets - handle both string and tuple/list inputs
    # Default to hermes-cli toolset which includes cronjob management tools
    toolsets_list = None
    if toolsets:
        if isinstance(toolsets, str):
            toolsets_list = [t.strip() for t in toolsets.split(",")]
        elif isinstance(toolsets, (list, tuple)):
            # Fire may pass multiple --toolsets as a tuple
            toolsets_list = []
            for t in toolsets:
                if isinstance(t, str):
                    toolsets_list.extend([x.strip() for x in t.split(",")])
                else:
                    toolsets_list.append(str(t))
    else:
        # Coding posture (base Hermes): with no explicit --toolsets, collapse
        # to the coding toolset (+ enabled MCP servers) when sitting in a code
        # workspace. See agent/coding_context.py.
        _coding = None
        try:
            from agent.coding_context import coding_selection
            _coding = coding_selection(platform="cli", config=CLI_CONFIG)
        except Exception:
            _coding = None
        if _coding is not None:
            toolsets_list = _coding
        else:
            # Use the shared resolver so MCP servers are included at runtime
            from hermes_cli.tools_config import _get_platform_tools
            toolsets_list = sorted(_get_platform_tools(CLI_CONFIG, "cli"))
    
    parsed_skills = _parse_skills_argument(skills)

    # Create CLI instance
    try:
        cli = HermesCLI(
            model=model,
            toolsets=toolsets_list,
            provider=provider,
            reasoning=reasoning,
            api_key=api_key,
            base_url=base_url,
            max_turns=max_turns,
            run_budget=run_budget,
            verbose=verbose,
            compact=compact,
            resume=resume,
            checkpoints=checkpoints,
            pass_session_id=pass_session_id,
            ignore_rules=ignore_rules,
        )
    except ImportError as e:
        # Direct `python cli.py` / `python -m cli` bypasses cmd_chat's
        # ImportError handler. Same mixed-tree class as #96900.
        from hermes_constants import emit_partial_update_hint

        if emit_partial_update_hint(e):
            sys.exit(1)
        raise

    if parsed_skills:
        # Load the skill payloads in the background: skill_view walks the
        # full skills tree per skill (~0.5s for a large library) and the
        # result is only consumed at agent init (first message / first
        # agent-touching command), not by the banner. cmd_chat joins the
        # thread via cli.finalize_preloaded_skills() before any consumer
        # reads cli.system_prompt — HermesCLI._create_agent calls it too,
        # so no agent can be built with the skills missing.
        def _load_preloaded_skills() -> None:
            try:
                cli._preload_skills_result = build_preloaded_skills_prompt(
                    parsed_skills,
                    task_id=cli.session_id,
                )
            except Exception as exc:  # surfaced by finalize below
                cli._preload_skills_error = exc

        cli._preload_skills_requested = parsed_skills
        cli._preload_skills_thread = threading.Thread(
            target=_load_preloaded_skills, name="skills-preload", daemon=True
        )
        cli._preload_skills_thread.start()

    # Join the background worktree creation (started above) before anything
    # consumes TERMINAL_CWD / wt_info — the HermesCLI construction it
    # overlapped with is done. Setup failure keeps the old abort semantics.
    if _join_worktree is not None:
        wt_info = _join_worktree()
        if not wt_info:
            # Worktree was explicitly requested but setup failed —
            # don't silently run without isolation.
            return

    # Inject worktree context into agent's system prompt
    if wt_info:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{wt_info['path']}. Your branch is `{wt_info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {wt_info['repo_root']}.]"
        )
        cli.system_prompt = (cli.system_prompt or "") + wt_note
    
    # Handle list commands (don't init agent for these)
    if list_tools:
        cli.show_banner()
        cli.show_tools()
        sys.exit(0)
    
    if list_toolsets:
        cli.show_banner()
        cli.show_toolsets()
        sys.exit(0)
    
    # Register cleanup for single-query mode (interactive mode registers in run())
    atexit.register(_run_cleanup)

    _install_single_query_signal_handlers(cli)
    
    # Handle single query mode
    if query or image:
        # NEW DEFAULT (Aug 2026): on a real TTY, a -q/--image invocation
        # seeds a normal interactive session with the prompt as the first
        # turn, submitted LITERALLY (no slash/! dispatch). Legacy
        # answer-and-exit behavior is kept for --oneshot, -Q, and every
        # non-TTY invocation (kanban/cron/pipes) — see
        # _should_seed_interactive().
        if _should_seed_interactive(query, image, quiet, oneshot):
            seeded_query, seeded_images = _collect_query_images(query, image)
            logger.info(
                "Seeding interactive session with -q prompt (%d chars, %d images)",
                len(seeded_query or ""), len(seeded_images),
            )
            cli._seeded_first_message = _SeededQueryMessage(seeded_query, seeded_images)
            cli.run()
            return
        # One-shot mode: no between-turns MCP late-binding refresh, so the
        # agent must wait the full MCP cold-start bound before its first
        # (and only) tool snapshot. See #51316.
        cli._single_query_mode = True
        # Mark single-query for the approval gate. cli.py sets
        # HERMES_INTERACTIVE earlier for interactive sudo prompts, but a -q
        # run has NO user waiting to answer approval prompts. The gate reads
        # this marker (via gateway.session_context.get_session_env, which falls
        # back to os.environ when the session-context layer isn't engaged) and
        # takes the deterministic approvals.single_query_mode path instead of
        # waiting the full timeout. See #86878.
        os.environ["HERMES_SINGLE_QUERY_SESSION"] = "1"
        if not cli._claim_active_session("cli", stderr=bool(quiet)):
            sys.exit(1)
        try:
            query, single_query_images = _collect_query_images(query, image)
            single_query_image_urls = _collect_kanban_task_images(single_query_images)
            if quiet:
                # Quiet mode: suppress banner, spinner, tool previews.
                # Only print the final response and parseable session info.
                cli.tool_progress_mode = "off"
                if cli._ensure_runtime_credentials():
                    effective_query: Any = query
                    effective_query = _route_single_query_images(cli, query, effective_query, single_query_images, single_query_image_urls)
                    turn_route = cli._resolve_turn_agent_config(effective_query)
                    if turn_route["signature"] != cli._active_agent_route_signature:
                        cli.agent = None
                    if cli._init_agent(
                        model_override=turn_route["model"],
                        runtime_override=turn_route["runtime"],
                        request_overrides=turn_route.get("request_overrides"),
                    ):
                        cli.agent.quiet_mode = True
                        cli.agent.suppress_status_output = True
                        # Suppress streaming display callbacks so stdout stays
                        # machine-readable (no styled "Hermes" box, no tool-gen
                        # status lines, no reasoning box).  The response is
                        # printed once below.
                        cli.agent.stream_delta_callback = None
                        cli.agent.tool_gen_callback = None
                        cli.agent.reasoning_callback = None
                        # Inline-diff and progress callbacks print directly to
                        # stdout and are gated by NEITHER quiet_mode nor
                        # tool_progress_mode: _on_tool_complete renders full
                        # file diffs via render_edit_diff_with_delta, and
                        # _on_tool_progress prints MoA reference blocks before
                        # its mode check. Neutralize them too so -Q stdout
                        # carries only the final response (#93220).
                        cli.agent.tool_progress_callback = None
                        cli.agent.tool_start_callback = None
                        cli.agent.tool_complete_callback = None
                        # Belt-and-braces for the executor's direct prints
                        # (they check agent.tool_progress_mode, initialized
                        # from display.tool_progress at construction).
                        cli.agent.tool_progress_mode = "off"
                        _run_quiet_single_query(cli, effective_query)

                # Exit with error code if credentials or agent init fails
                sys.exit(1)
            else:
                # Single-query mode (`hermes chat -q "…"`): skip the welcome
                # banner. Building the banner takes ~420 ms on cold start —
                # ~200 ms of that is the version-update check, the rest is
                # toolset / skill enumeration and Rich panel rendering. None
                # of that is useful for a one-shot query: the user already
                # picked the prompt, doesn't need a toolset reference, and
                # gets the session ID + resume hint from
                # ``_print_exit_summary()`` after the response prints.
                #
                # The fully-quiet ``-Q`` / ``--quiet`` machine-readable path
                # above was already banner-free; this brings the human-
                # facing single-query path in line so all non-interactive
                # invocations are fast.
                _query_label = query or ("[image attached]" if single_query_images else "")
                if _query_label:
                    cli.console.print(f"[bold blue]Query:[/] {_query_label}")
                # Surface security advisories before the agent runs — short
                # banner, doesn't depend on the welcome banner being shown.
                cli._show_security_advisories()
                cli.chat(query, images=single_query_images or None)
                cli._print_exit_summary(clear_screen=False)
        finally:
            _finalize_single_query(cli)
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
