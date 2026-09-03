#!/usr/bin/env python3
"""
Hermes Agent CLI - Interactive Terminal Interface

Usage:
    python cli.py                          # Start interactive mode with all tools
    python cli.py --toolsets web,terminal  # Start with specific toolsets
    python cli.py --skills hermes-agent-dev,github-auth
    python cli.py --list-tools             # List available tools and exit
"""

# hermes_bootstrap must be the very first import — UTF-8 stdio on Windows (no-op on
# POSIX). Missing only during a partial ``hermes update`` (git reset landed, pip
# install didn't); then Windows UTF-8 setup is skipped.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
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
    from hermes_cli import pt_input_extras as _pt_extras

    _pt_extras.install_shift_enter_alias()
    _pt_extras.install_ctrl_enter_alias()
    _pt_extras.install_cmd_backspace_alias()
    _pt_extras.install_modify_other_keys_aliases()
    _pt_extras.install_keypress_data_normalization()
    _pt_extras.install_ignored_terminal_sequences()
    del _pt_extras
except Exception:
    pass
import threading
import queue


def _lazy_shim(module: str, name: str, alias: str | None = None):
    """Module-level function that imports ``module.name`` on first call and delegates to it.

    Keeps heavy imports (agent, tools, toolsets) off the bare-startup path while the
    symbol stays importable/patchable as ``cli.<alias or name>``.
    """
    import importlib

    def shim(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)

    shim.__name__ = shim.__qualname__ = alias or name
    return shim


CanonicalUsage = _lazy_shim("agent.usage_pricing", "CanonicalUsage")
estimate_usage_cost = _lazy_shim("agent.usage_pricing", "estimate_usage_cost")


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


# Reverse map of config.yaml ``model_aliases:`` so the TUI can show friendly names
# instead of long catalog IDs. Process-lifetime cache (config is read once per session).
_REVERSE_ALIAS_CACHE: dict[str, str] | None = None


def _reverse_alias_for_display(model_name: str) -> str:
    """Return the shortest configured alias for ``model_name``, or ``model_name``.

    Looks up both ``model_aliases:`` (dict entries) and ``model.aliases:`` (strings,
    set via ``hermes config set``); the shortest alias wins.
    """
    global _REVERSE_ALIAS_CACHE
    if not model_name:
        return model_name
    if _REVERSE_ALIAS_CACHE is None:
        rmap: dict[str, str] = {}

        def _put(m: str, alias: str) -> None:
            if m and (m not in rmap or len(alias) < len(rmap[m])):
                rmap[m] = alias

        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            ma = cfg.get("model_aliases")
            if isinstance(ma, dict):
                for alias, entry in ma.items():
                    if isinstance(entry, dict):
                        _put(str(entry.get("model", "") or "").strip(), alias)
            mdl = cfg.get("model", {}) or {}
            if isinstance(mdl, dict):
                simple = mdl.get("aliases")
                if isinstance(simple, dict):
                    for alias, val in simple.items():
                        if isinstance(val, str) and val.strip():
                            v = val.strip()
                            _put(v.split("/", 1)[1] if "/" in v else v, alias)
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


is_table_divider = _lazy_shim("agent.markdown_tables", "is_table_divider")
looks_like_table_row = _lazy_shim("agent.markdown_tables", "looks_like_table_row")
realign_markdown_tables = _lazy_shim("agent.markdown_tables", "realign_markdown_tables")
# `agent.account_usage` is deliberately NOT imported at module top — it pulls the
# OpenAI SDK chain (~230 ms cold) and is only needed by `/limits`.
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
_TOOL_CALL_TAGS = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")


def _strip_reasoning_tags(text: str) -> str:
    """Remove reasoning/thinking blocks from displayed text.

    Handles closed pairs, unterminated open tags (truncated generations), and stray
    orphan close tags, case-insensitively. Must stay in sync with
    ``run_agent.py::_strip_think_blocks`` and the stream consumer's
    ``_OPEN_THINK_TAGS`` / ``_CLOSE_THINK_TAGS``. Also strips tool-call XML some open
    models leak into visible content (``<tool_call>``, Gemma-style ``<function name=…>``).
    """
    cleaned = text
    for tag in _REASONING_TAGS:
        cleaned = re.sub(rf"<{tag}>.*?</{tag}>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(rf"<{tag}>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(rf"</{tag}>\s*", "", cleaned, flags=re.IGNORECASE)
    for tc_tag in _TOOL_CALL_TAGS:
        cleaned = re.sub(rf"<{tc_tag}\b[^>]*>.*?</{tc_tag}>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # <function name="..."> — boundary + attribute gated to avoid prose false positives.
    cleaned = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
        r'<function\b[^>]*\bname\s*=[^>]*>'
        r'(?:(?:(?!</function>).)*)</function>\s*',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
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
    """Load ephemeral prefill messages (JSON array of {role, content}) from a file.

    Relative paths resolve from ~/.hermes/. Empty path or missing file -> [].
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
    """Resolve the prefill file path: env, then top-level ``prefill_messages_file``,
    then the legacy ``agent.prefill_messages_file``."""
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
    """Parse a reasoning effort level (string or YAML bool; ``false``/``off`` = disabled)."""
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


_TERMINAL_ENV_MAPPINGS = {
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
    "ssh_host": "TERMINAL_SSH_HOST",
    "ssh_user": "TERMINAL_SSH_USER",
    "ssh_port": "TERMINAL_SSH_PORT",
    "ssh_key": "TERMINAL_SSH_KEY",
    # Container resources (docker, singularity, modal, daytona, vercel_sandbox; ignored for local/ssh)
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
    "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
    "sudo_password": "SUDO_PASSWORD",
}
# Per-task auxiliary endpoint tuples (config key -> env var).
_AUXILIARY_TASK_ENV = {
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
_CWD_PLACEHOLDERS = (".", "auto", "cwd")


def _mirror_config_to_env(defaults, _file_has_terminal_config):
    """Project config.yaml values into the env vars the tool modules read (terminal/browser/auxiliary/security/sessions). Env always wins when already set."""
    terminal_config = defaults.get("terminal", {})

    # "backend" (hermes_cli/config.py + docs) and legacy "env_type" (cli-config.yaml)
    # are both accepted; "backend" wins.
    if "backend" in terminal_config:
        terminal_config["env_type"] = terminal_config["backend"]

    # CWD: local backend is always os.getcwd() (`cd /dir && hermes` controls it);
    # non-local with a placeholder pops it so terminal_tool uses its per-backend
    # default; non-local with an explicit path keeps it.
    effective_backend = terminal_config.get("env_type", "local")
    if effective_backend == "local":
        terminal_config["cwd"] = os.getcwd()
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        terminal_config.pop("cwd", None)

    # TERMINAL_CWD is force-exported (overrides stale .env/inherited values) UNLESS
    # inside a gateway process, where gateway/run.py's config bridge already set it.
    _is_gateway = os.environ.get("_HERMES_GATEWAY") == "1"
    for config_key, env_var in _TERMINAL_ENV_MAPPINGS.items():
        if config_key not in terminal_config:
            continue
        val = terminal_config[config_key]
        if env_var == "TERMINAL_CWD":
            if not _is_gateway:
                os.environ[env_var] = str(val)
        elif _file_has_terminal_config or env_var not in os.environ:
            os.environ[env_var] = json.dumps(val) if isinstance(val, (list, dict)) else str(val)

    browser_config = defaults.get("browser", {})
    if "inactivity_timeout" in browser_config:
        os.environ["BROWSER_INACTIVITY_TIMEOUT"] = str(browser_config["inactivity_timeout"])

    # Auxiliary overrides: only non-empty / non-"auto" values so auto-detection still
    # works. (Compression config is read directly from config.yaml — no bridging.)
    auxiliary_config = defaults.get("auxiliary", {})
    for task_key, env_map in _AUXILIARY_TASK_ENV.items():
        task_cfg = auxiliary_config.get(task_key, {})
        if not isinstance(task_cfg, dict):
            continue
        for field, env_var in env_map.items():
            val = str(task_cfg.get(field, "")).strip()
            if val and not (field == "provider" and val == "auto"):
                os.environ[env_var] = val

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
            os.environ["HERMES_SEARCH_SLOW_MS"] = str(sessions_config["search_slow_ms"])


def _cli_config_defaults():
    """Built-in defaults for every config key the CLI reads (the file overlays these)."""
    return {
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
            "min_tail_user_messages": 1,  # Real user messages guaranteed in the tail
        },
        "agent": {
            "max_turns": 500,  # Default max tool-calling iterations (shared with subagents)
            "verbose": False,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "service_tier": "",
            # Built-in personalities live in hermes_cli.personality (BUILTIN_PERSONALITIES);
            # entries here are user additions/overrides merged on top by name.
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
            # Keep in sync with hermes_cli/config.py DEFAULT_CONFIG (display.show_reasoning).
            "show_reasoning": True,
            "reasoning_full": False,
            "streaming": True,
            "busy_input_mode": "interrupt",
            "persistent_output": True,
            "persistent_output_max_lines": 200,
            # Clear scrollback as well as the viewport on full redraw/resize recovery.
            # Off by default (users prefer history); enable when a terminal/tmux stack
            # stamps stale prompt chrome into scrollback during resizes.
            "cli_rebuild_scrollback_on_redraw": False,
            # One-line summary of resolved modal prompts (approval / clarify) into scrollback.
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
            # First-touch hint flags (see agent/onboarding.py), latched once shown.
            "seen": {},
        },
    }


def _merge_file_config(defaults: Dict[str, Any], file_config: Dict[str, Any]) -> None:
    """Overlay a parsed config file onto *defaults* in place (model normalization, deep merge, legacy keys)."""
    # model: string (new format) or dict (old format with default/base_url)
    if "model" in file_config:
        if isinstance(file_config["model"], str):
            defaults["model"]["default"] = file_config["model"]
        elif isinstance(file_config["model"], dict):
            defaults["model"].update(file_config["model"])
            # Promote model.model to model.default when only the former is set, so a
            # profile config that sets "model:" isn't shadowed by the hardcoded default
            # (HermesCLI.__init__ checks "default" first).
            if "model" in file_config["model"] and "default" not in file_config["model"]:
                defaults["model"]["default"] = file_config["model"]["model"]

    # Deep-merge dict sections, overwrite scalars; a None section keeps the defaults.
    for key in defaults:
        if key == "model" or key not in file_config:
            continue
        if isinstance(defaults[key], dict) and file_config[key] is None:
            continue
        if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
            defaults[key].update(file_config[key])
        else:
            defaults[key] = file_config[key]

    # Carry over keys not in defaults (platform_toolsets, provider_routing, memory, ...)
    for key in file_config:
        if key not in defaults and key != "model":
            defaults[key] = file_config[key]

    # Legacy root-level max_turns -> agent.max_turns whenever the nested key is missing.
    agent_file_config = file_config.get("agent")
    if "max_turns" in file_config and not (
        isinstance(agent_file_config, dict)
        and agent_file_config.get("max_turns") is not None
    ):
        defaults["agent"]["max_turns"] = file_config["max_turns"]


def load_cli_config() -> Dict[str, Any]:
    """Load CLI configuration: ~/.hermes/config.yaml, else ./cli-config.yaml, over built-in defaults.

    Env vars take precedence over file values. ``HERMES_IGNORE_USER_CONFIG=1``
    (``hermes chat --ignore-user-config``) skips the user config entirely — only
    defaults plus the project ``cli-config.yaml`` apply; ``.env`` credentials still load.
    """
    user_config_path = _hermes_home / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'
    ignore_user_config = os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1"

    if user_config_path.exists() and not ignore_user_config:
        config_path = user_config_path
    else:
        config_path = project_config_path

    defaults = _cli_config_defaults()

    # Only a user's config file may overwrite terminal env vars already set by .env;
    # defaults (no file / no terminal section) must not.
    _file_has_terminal_config = False

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                from hermes_cli.config import _normalize_root_model_keys

                file_config = _normalize_root_model_keys(fast_safe_load(f) or {})

            _file_has_terminal_config = "terminal" in file_config
            _merge_file_config(defaults, file_config)
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)

    # Expand ${ENV_VAR} references before bridging to env vars.
    from hermes_cli.config import _expand_env_vars
    defaults = _expand_env_vars(defaults)

    # Managed scope overlays administrator-pinned values LAST. cli.py builds its config
    # independently of hermes_cli.config._load_config_impl, so without this the whole
    # interactive CLI/TUI surface (skin, display prefs) would ignore managed scope
    # while `hermes config`/`doctor` honor it. The shared helper is fail-open.
    from hermes_cli import managed_scope

    defaults = managed_scope.apply_managed_overlay(defaults)

    _mirror_config_to_env(defaults, _file_has_terminal_config)

    return defaults

# Load configuration at module startup
CLI_CONFIG = load_cli_config()


def _init_logging_and_display_from_config() -> None:
    """Best-effort startup side effects: logging, config warnings, skin, display knobs."""
    try:
        from hermes_logging import setup_logging
        setup_logging(mode="cli")
    except Exception:
        pass
    try:
        from hermes_cli.config import print_config_warnings
        print_config_warnings()
    except Exception:
        pass
    try:
        from hermes_cli.skin_engine import init_skin_from_config
        init_skin_from_config(CLI_CONFIG)
    except Exception:
        pass
    try:
        from agent.display import set_tool_preview_max_len
        _tpl = CLI_CONFIG.get("display", {}).get("tool_preview_length", 0)
        set_tool_preview_max_len(int(_tpl) if _tpl else 0)
    except Exception:
        pass
    try:
        from agent.display import set_friendly_tool_labels
        _ftl = CLI_CONFIG.get("display", {}).get("friendly_tool_labels", True)
        set_friendly_tool_labels(bool(_ftl))
    except Exception:
        pass


_init_logging_and_display_from_config()

# Neuter AsyncHttpxClientWrapper.__del__ before any AsyncOpenAI client exists: the
# SDK's __del__ schedules aclose() on the running loop, which during CLI idle time is
# prompt_toolkit's loop, closing transports bound to dead worker loops ("Event loop is
# closed" / "Press ENTER to continue..."). A sys.meta_path finder applies the patch
# when ``openai._base_client`` is first imported — eager import costs ~166ms/30MB per
# cold start, and the import system guarantees the patch lands before instantiation.
try:
    import sys as _httpx_neuter_sys
    import importlib.util as _httpx_neuter_imp_util

    class _AsyncHttpxDelNeuter:
        """Defer ``AsyncHttpxClientWrapper.__del__`` neutering until import.

        See ``agent.auxiliary_client.neuter_async_httpx_del`` for why ``__del__``
        must be a no-op.
        """

        _armed = True

        def find_spec(self, fullname, path=None, target=None):
            if not self._armed or fullname != "openai._base_client":
                return None
            # Disarm before delegating so the recursive find_spec doesn't loop through us.
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

from rich.console import Console
from rich.markup import escape as _escape
from rich.text import Text as _RichText

# Agent and tool systems are imported lazily: bare interactive startup only needs
# the prompt; the full agent/tool registry is initialized on first use.
AIAgent = _lazy_shim("run_agent", "AIAgent")


def get_tool_definitions(*args, **kwargs):
    from hermes_cli.mcp_startup import wait_for_mcp_discovery
    from model_tools import get_tool_definitions as _get_tool_definitions

    wait_for_mcp_discovery()
    return _get_tool_definitions(*args, **kwargs)


get_toolset_for_tool = _lazy_shim("model_tools", "get_toolset_for_tool")

from hermes_cli.banner import build_welcome_banner  # noqa: F401  (CLIInfoMixin imports via cli)

get_all_toolsets = _lazy_shim("toolsets", "get_all_toolsets")
get_toolset_info = _lazy_shim("toolsets", "get_toolset_info")
validate_toolset = _lazy_shim("toolsets", "validate_toolset")


def _sync_process_session_id(session_id: str) -> None:
    """Keep process-local session-id consumers aligned after CLI switches."""
    from gateway.session_context import set_current_session_id

    set_current_session_id(session_id)


# Cron job system for scheduled tasks (execution is handled by the gateway)
get_job = _lazy_shim("cron", "get_job")
_cleanup_all_terminals = _lazy_shim("tools.terminal_tool", "cleanup_all_environments", "_cleanup_all_terminals")
set_sudo_password_callback = _lazy_shim("tools.terminal_tool", "set_sudo_password_callback")
set_approval_callback = _lazy_shim("tools.terminal_tool", "set_approval_callback")
set_secret_capture_callback = _lazy_shim("tools.skills_tool", "set_secret_capture_callback")
_cleanup_all_browsers = _lazy_shim("tools.browser_tool", "_emergency_cleanup_all_sessions", "_cleanup_all_browsers")

# Guard to prevent cleanup from running multiple times on exit
_cleanup_done = False
_cleanup_in_progress = False
_cli_wake_owner = None
# One-shot CLI finalization runs before process cleanup so plugins can observe the
# session boundary while the agent is still attached; atexit cleanup must not emit
# that session's finalization again.
_single_query_finalize_attempted_session_ids: set[str | None] = set()
# Sessions handed off to the gateway via /handoff: the gateway owns their lifecycle,
# so _run_cleanup must NOT finalize them (it would set end_reason on a row the gateway
# just reopened and is writing to, making the handoff leg vanish from history).
_handed_off_session_ids: set[str | None] = set()
# Weak reference to the active AIAgent for memory provider shutdown at exit
_active_agent_ref = None
_deferred_agent_startup_done = False
# True once the TUI's prompt_toolkit app starts (focus reporting + mouse tracking on).
# Gates the on-exit terminal reset so non-TUI one-shot runs — which also register
# _run_cleanup via atexit — don't emit escape codes for modes they never enabled.
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
        logger.warning("plugin discovery failed at deferred CLI startup", exc_info=True)
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(logger=logger, thread_name="termux-cli-mcp-discovery")
    except Exception:
        logger.debug("MCP tool discovery failed at deferred CLI startup", exc_info=True)
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
        logger.debug("shell-hook registration failed at deferred CLI startup", exc_info=True)


def _exit_watchdog_timeout() -> float:
    """``HERMES_EXIT_WATCHDOG_S`` as a float (default 30; ``0`` disables)."""
    try:
        return float(os.getenv("HERMES_EXIT_WATCHDOG_S", "30"))
    except (TypeError, ValueError):
        return 30.0


def _arm_exit_watchdog(timeout_s: float | None = None, *, from_signal: bool = False) -> None:
    """Guarantee the process actually exits once shutdown has begun.

    Backstop for two hang classes: a cleanup step wedged on network I/O (memory
    provider ``on_session_end``, MCP teardown, remote terminal cleanup), and
    interpreter teardown blocked joining non-daemon threads (stdlib
    ``ThreadPoolExecutor`` workers are joined unconditionally by its atexit hook even
    after ``shutdown(wait=False)``). A daemon timer keeps running through
    ``Py_FinalizeEx``'s joins; after ``timeout_s`` it flushes logging/stdio and calls
    ``os._exit(0)``. Tune with ``HERMES_EXIT_WATCHDOG_S`` (seconds); ``0`` disables.
    """
    if timeout_s is None:
        timeout_s = _exit_watchdog_timeout()
    if timeout_s <= 0:
        return
    # Never arm under pytest: tests invoke _run_cleanup() directly and a delayed
    # os._exit(0) would silently kill the test worker.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    def _watchdog():
        time.sleep(timeout_s)
        # The signal-armed outer watchdog yields to the cleanup-owned timer once
        # cleanup is in progress; it only wins when graceful unwind never started.
        if from_signal and _cleanup_in_progress:
            return

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
        threading.Thread(target=_watchdog, daemon=True, name="exit-watchdog").start()
    except Exception:
        pass  # best-effort — never block shutdown on watchdog setup


_signal_watchdog_armed = False


def _arm_exit_watchdog_on_shutdown_signal() -> None:
    """Arm the exit backstop the moment a termination signal arrives.

    The graceful signal -> interrupt -> app.exit -> finally -> ``_run_cleanup`` path
    has several wedge points BEFORE ``_run_cleanup`` arms the normal watchdog (main
    thread parked in a syscall, prompt_toolkit teardown never returning, an agent
    worker blocking the ``finally``); then a "dead" CLI lingers with no backstop.
    The leash is 2x the cleanup timeout so a slow-but-progressing cleanup (which arms
    its own tighter timer) is never cut short. Deliberately NOT armed at chat startup:
    the timer calls ``os._exit(0)`` unconditionally, so arming without shutdown intent
    would hard-kill every session outliving it. Idempotent; never raises.
    """
    global _signal_watchdog_armed
    if _signal_watchdog_armed:
        return
    _signal_watchdog_armed = True
    base = _exit_watchdog_timeout()
    if base <= 0:
        return  # explicitly disabled
    try:
        _arm_exit_watchdog(timeout_s=base * 2, from_signal=True)
    except Exception:
        pass  # never let the backstop break signal handling


def _shutdown_agent_memory_provider(agent) -> None:
    """Memory-provider shutdown (on_session_end + shutdown_all) at the real session boundary."""
    if not (agent and hasattr(agent, 'shutdown_memory_provider')):
        return
    # A /new shortly before exit leaves its end->switch boundary task (old-session
    # extraction, LLM-bound) queued on the memory manager's serialized worker;
    # shutdown_all()'s ~5s drain cancels queued tasks, so give pending work a bounded
    # head start. The exit watchdog remains the hard backstop.
    _mm = getattr(agent, '_memory_manager', None)
    if _mm is not None and hasattr(_mm, 'flush_pending'):
        try:
            _mm.flush_pending(timeout=10)
        except Exception:
            pass
    # Forward the agent's own transcript so on_session_end hooks see the real
    # conversation. ``_session_messages`` is refreshed every turn via ``_persist_session``;
    # fall back to no-arg on test stubs / partially-initialised agents.
    _session_msgs = getattr(agent, '_session_messages', None)
    if isinstance(_session_msgs, list):
        logger.info(
            "CLI cleanup calling memory shutdown for session %s with %d message(s)",
            getattr(agent, "session_id", None) or "<unknown>",
            len(_session_msgs),
        )
        agent.shutdown_memory_provider(_session_msgs)
    else:
        logger.info(
            "CLI cleanup calling memory shutdown for session %s without session message list",
            getattr(agent, "session_id", None) or "<unknown>",
        )
        agent.shutdown_memory_provider()


def _stop_cli_wake_word() -> None:
    from tools.wake_word import stop_listening
    if _cli_wake_owner is not None:
        stop_listening(owner=_cli_wake_owner)


def _interrupt_async_delegations() -> None:
    from tools.async_delegation import interrupt_all
    interrupt_all(reason="CLI shutdown")


def _shutdown_mcp_servers() -> None:
    from tools.mcp_tool import shutdown_mcp_servers
    shutdown_mcp_servers()


def _shutdown_cached_aux_clients() -> None:
    # Close cached auxiliary LLM clients so AsyncHttpxClientWrapper.__del__ doesn't
    # fire on a closed loop and trigger prompt_toolkit's "Press ENTER to continue...".
    from agent.auxiliary_client import shutdown_cached_clients
    shutdown_cached_clients()


# Ordered best-effort teardown steps (module attribute names, resolved at call time so
# tests can patch ``cli._cleanup_all_terminals`` etc.) and the exception each swallows.
_CLEANUP_STEPS = (
    ("_stop_cli_wake_word", Exception),
    ("_cleanup_all_terminals", Exception),
    ("_interrupt_async_delegations", Exception),
    ("_cleanup_all_browsers", Exception),
    ("_shutdown_mcp_servers", BaseException),
    ("_shutdown_cached_aux_clients", Exception),
)


def _run_cleanup(*, notify_session_finalize: bool = True):
    """Run resource cleanup exactly once."""
    global _cleanup_done, _cleanup_in_progress
    if _cleanup_done:
        return
    _cleanup_done = True
    _cleanup_in_progress = True

    try:
        # Bound total shutdown time: a wedged cleanup (or interpreter thread-join
        # teardown) force-exits instead of leaving a zombie CLI holding the terminal.
        _arm_exit_watchdog()

        # Reset terminal input modes FIRST: the slower teardown below can take seconds,
        # and a later step raising must not skip the reset. No-op unless the TUI ran.
        _reset_terminal_input_modes_on_exit()

        for step, swallow in _CLEANUP_STEPS:
            try:
                globals()[step]()
            except swallow:
                pass
        if notify_session_finalize:
            cleanup_session_id = _active_agent_ref.session_id if _active_agent_ref else None
            if _should_emit_cleanup_session_finalize(cleanup_session_id):
                _notify_session_finalize(
                    session_id=cleanup_session_id,
                    platform="cli",
                    reason="shutdown",
                )
        try:
            _shutdown_agent_memory_provider(_active_agent_ref)
        except Exception as e:
            logger.warning("CLI cleanup memory shutdown failed: %s", e, exc_info=True)
    finally:
        _cleanup_in_progress = False


def _should_emit_cleanup_session_finalize(session_id: str | None) -> bool:
    # A handed-off session is owned by the gateway process — never finalize it here.
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
        finalize_session(session_id=session_id, platform=platform, reason=reason)
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
    if session_id in _handed_off_session_ids:  # gateway owns the lifecycle now
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
    if session_id in _handed_off_session_ids:  # gateway owns the lifecycle now
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

    The ``-q`` / ``-Q`` paths get exactly one turn and then exit, so unlike the
    interactive CLI nothing retried a transiently-failed transcript flush (lost turns
    under state.db write-lock contention), the resumed/created titled row was left
    open, and queued token-accounting deltas relied on interpreter-exit hooks the
    kanban ``os._exit(0)`` path skips. Idempotent and best-effort: ``_persist_session``
    dedupes via per-message markers and ``end_session`` no-ops on an ended row.
    Handed-off sessions are left strictly alone.
    """
    agent = getattr(cli, "agent", None)
    if agent is None:
        return
    session_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
    if not session_id or session_id in _handed_off_session_ids:
        return
    if getattr(agent, "_persist_disabled", False):
        return
    # ``cli.conversation_history`` holds the resumed history's live dicts, so passing it
    # keeps restored messages identity-skipped even when the failed flush never stamped them.
    try:
        msgs = getattr(agent, "_session_messages", None)
        if isinstance(msgs, list) and msgs and hasattr(agent, "_persist_session"):
            agent._persist_session(msgs, getattr(cli, "conversation_history", None))
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
    """Bounded linger for notify_on_complete background processes.

    A one-shot run that spawned bounded background work (e.g. a Bot Mode handoff reply
    via ``terminal(background=true, notify_on_complete=true)``) must not exit while it
    runs: the children write to pipes owned by this process. Waits on the whole
    registry (a one-shot process hosts exactly one agent, and task_id filtering would
    skip processes registered before the session id settled). Cheap no-op when idle.
    """
    from tools.process_registry import process_registry

    agent = getattr(cli, "agent", None)
    task_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
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
        # Linger for spawned background work BEFORE any teardown (the parent owns
        # those children's stdout pipes).
        try:
            _wait_for_oneshot_background_completions(cli)
        except Exception:
            logger.debug("one-shot background completion wait failed", exc_info=True)
        # Durable flush FIRST: memory-provider shutdown inside _run_cleanup can issue
        # aux-LLM calls, and nothing after it may fail in a way that loses the turn.
        try:
            _flush_one_shot_session_store(cli)
        except Exception:
            logger.debug("one-shot session store flush failed", exc_info=True)
        _notify_single_query_session_finalize(cli)
        _run_cleanup(notify_session_finalize=False)
    finally:
        cli._release_active_session()


def _reset_terminal_input_modes_on_exit() -> None:
    """Best-effort: disable focus reporting + mouse tracking on TUI exit.

    prompt_toolkit restores these on a clean teardown, but Ctrl+C, SIGTERM/SIGHUP and
    crashes bypass its unwind, leaving raw ``ESC[I``/``ESC[O`` focus events and SGR
    mouse reports as visible text in the next shell sharing the tab. Gated on
    ``_tui_input_modes_active`` so one-shot non-TUI runs never emit these codes. By
    exit prompt_toolkit's output is torn down, so write to ``sys.stdout`` when it is
    the terminal, else ``/dev/tty`` (the TUI may have driven it while stdout was redirected).
    """
    global _tui_input_modes_active
    if not _tui_input_modes_active:
        return
    # Clear first so a re-armed _run_cleanup doesn't re-emit.
    _tui_input_modes_active = False
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
# Git Worktree Isolation
# =============================================================================

from hermes_cli.worktree_ops import (  # noqa: F401  (mixins/tests/worktree_gc import via cli)
    _PACK_SPRAWL_THRESHOLD,
    _WORKTREE_MERGE_CACHE_MAX,
    _classify_prune_candidates,
    _cleanup_failed_worktree_add,
    _copy_worktree_includes,
    _deepen_shallow_repo,
    _ensure_worktrees_gitignored,
    _fetch_remote_branch_heads,
    _git,
    _git_quiet,
    _git_repo_root,
    _load_worktree_merge_cache,
    _maintain_pack_health,
    _normalize_git_bash_path,
    _path_is_within_root,
    _prune_candidates,
    _prune_orphaned_branches,
    _prune_stale_worktrees,
    _reap_prune_verdicts,
    _repo_is_shallow,
    _resolve_worktree_base,
    _save_worktree_merge_cache,
    _setup_worktree,
    _worktree_add,
    _worktree_branch_pr_merged,
    _worktree_branch_pushed_exact,
    _worktree_commits_all_merged_upstream,
    _worktree_current_branch,
    _worktree_has_unpushed_commits,
    _worktree_is_dirty,
    _worktree_lock_is_live,
    _worktree_merge_cache_path,
)

# Tracks the active worktree for cleanup on exit
_active_worktree: Optional[Dict[str, str]] = None


def _cleanup_worktree(info: Dict[str, str] = None) -> None:
    """Remove a worktree and its branch on exit.

    Preserved only when it has unpushed commits (real work). Uncommitted changes alone
    are not enough — agent work lives in commits/PRs, not the working tree.
    """
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    if _worktree_has_unpushed_commits(wt_path, timeout=10):
        if _repo_is_shallow(repo_root):
            # The shallow boundary makes the unpushed verdict unreliable; the startup
            # pruner deepens the clone in the background and reaps it later.
            _cprint(f"\n\033[33m⚠ Shallow clone — cannot verify push state, keeping: {wt_path}\033[0m")
            print("  The next `hermes -w` session deepens the clone and prunes merged worktrees automatically.")
        else:
            _cprint(f"\n\033[33m⚠ Worktree has unpushed commits, keeping: {wt_path}\033[0m")
            print(f"  To clean up manually: git worktree remove --force {wt_path}")
        _active_worktree = None
        return

    # Unlock first so `remove` isn't blocked by the lock placed at creation. Fail-soft.
    _git_quiet(["worktree", "unlock", wt_path], repo_root, log="git worktree unlock failed (non-fatal)")
    _git_quiet(["worktree", "remove", wt_path, "--force"], repo_root, timeout=15, log="Failed to remove worktree")
    _git_quiet(["branch", "-D", branch], repo_root, log=f"Failed to delete branch {branch}")

    _active_worktree = None
    _cprint(f"\033[32m✓ Worktree cleaned up: {wt_path}\033[0m")


def _run_state_db_auto_maintenance(session_db) -> None:
    """Run one-time repairs + ``SessionDB.maybe_auto_prune_and_vacuum`` per the ``sessions:`` config.

    Uses :func:`hermes_cli.config.load_config` (deep-merges DEFAULT_CONFIG so unmigrated
    configs still get defaults). Never raises — maintenance must never block startup.
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

        # One-time finalize of orphaned compression continuations.
        try:
            if not session_db.get_meta("orphaned_compression_finalize_v1"):
                finalized = session_db.finalize_orphaned_compression_sessions()
                session_db.set_meta("orphaned_compression_finalize_v1", "1")
                if finalized:
                    logger.info("Finalized %d orphaned compression sessions", finalized)
        except Exception as _finalize_exc:
            logger.debug("Orphan compression finalize skipped: %s", _finalize_exc)

        cfg = (_load_full_config().get("sessions") or {})

        # Auto-archive is independent of the destructive auto_prune sweep — run it
        # first, before prune's early return.
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
    """Call ``maybe_auto_prune_checkpoints`` per the ``checkpoints:`` config. Never raises."""
    try:
        from hermes_cli.config import load_config as _load_full_config
        cfg = (_load_full_config().get("checkpoints") or {})
        if not cfg.get("auto_prune", False):
            return
        from tools.checkpoint_manager import maybe_auto_prune_checkpoints
        # delete_orphans is never honoured here: a missing workdir at startup is
        # ambiguous (deleted project vs. unmounted volume / VPN not yet up) and this
        # sweep runs unattended. Orphans are only reclaimed by `hermes checkpoints prune`.
        maybe_auto_prune_checkpoints(
            retention_days=int(cfg.get("retention_days", 7)),
            min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            delete_orphans=False,
            max_total_size_mb=int(cfg.get("max_total_size_mb", 500)),
        )
    except Exception as exc:
        logger.debug("checkpoint auto-maintenance skipped: %s", exc)


# ============================================================================
# ASCII Art & Branding
# ============================================================================

# ANSI building blocks for conversation display
_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold — fallback
_BOLD = "\033[1m"
_RST = "\033[0m"
# No indent for streamed response text — leading whitespace pollutes terminal
# copy/paste. Matches the response Panel's flush-left padding.
_STREAM_PAD = ""
# Tail of an unfinished logical line mirrored into the spinner while streaming.
_STREAM_PARTIAL_PREVIEW_LEN = 60


def _hex_to_ansi(hex_color: str, *, bold: bool = False) -> str:
    """Convert '#RRGGBB' to a true-color ANSI escape, remapping dark-tuned colors in light mode."""
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
# Light/dark terminal mode detection (mirrors ui-tui/src/theme.ts detectLightMode()).
# Decides whether near-white skin colors are remapped to darker equivalents readable
# on a light Terminal.app / iTerm2 background. Priority:
#   1. HERMES_LIGHT / HERMES_TUI_LIGHT env (true/false)
#   2. HERMES_TUI_THEME=light|dark
#   3. HERMES_TUI_BACKGROUND=#RRGGBB
#   4. COLORFGBG (xterm/Konsole/urxvt) — bg slot 7/15 = light
#   5. OSC 11 query — ask the terminal directly
#   6. Default: dark
# Cached after first call so the terminal is never queried twice.
_LIGHT_MODE_CACHE: bool | None = None
_TRUE_RE = re.compile(r"^(1|true|on|yes|y)$")
_FALSE_RE = re.compile(r"^(0|false|off|no|n)$")
_LIGHT_DEFAULT_TERM_PROGRAMS = frozenset()  # Apple_Terminal doesn't reliably indicate; require explicit


def _luminance_from_hex(hex_str: str) -> float | None:
    """Rec.709 luma in [0, 1] for '#RGB'/'#RRGGBB', or None when malformed."""
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


_DA1_REPLY_RE = re.compile(rb"\x1b\[\?[0-9;]*c")


def _query_osc11_background() -> str | None:
    """Ask the terminal for its background color via OSC 11; "#RRGGBB" or None.

    The query is fenced with a DA1 sentinel (``ESC[c``), as in the Ink TUI's
    TerminalQuerier: terminals answer in order and virtually all answer DA1, so the
    DA1 reply proves the terminal already processed (or ignored) our OSC 11. Without
    the fence a reply arriving after we stop listening leaks into prompt_toolkit's
    stdin as typed text (the "gibberish ANSI" seen under herdr/WSL bridges/tmux).

    Skipped over SSH: the round-trip routinely exceeds the budget, and a late reply's
    BEL terminator reads as Ctrl+G (open editor). After restoring termios with
    TCSAFLUSH, a 50 ms drain catches late bytes that slipped past the flush (seen on
    loaded VPS/container terminals).
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
            # One write so the OSC 11 query and DA1 fence cannot reorder.
            sys.stdout.write("\x1b]11;?\x1b\\\x1b[c")
            sys.stdout.flush()
        except Exception:
            return None
        # Read until the DA1 fence closes (single-digit ms on real terminals). The 1s
        # deadline is only a safety net for a terminal that ignores DA1; a slow in-order
        # relay delivering OSC 11 at 400ms is handled since we wait for its DA1 reply.
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
        # Reply: \x1b]11;rgb:RRRR/GGGG/BBBB\x1b\\ — components are 1-4 hex digits.
        m = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
        if not m:
            return None

        def norm(h: bytes) -> int:
            v = int(h, 16)
            bits = len(h) * 4
            return (v * 255) // ((1 << bits) - 1) if bits else 0
        r, g, b = norm(m.group(1)), norm(m.group(2)), norm(m.group(3))
        return f"#{r:02X}{g:02X}{b:02X}"
    finally:
        # TCSAFLUSH discards unread input while restoring — scrubs a slow/partial
        # OSC 11 reply before prompt_toolkit can read it as keystrokes.
        try:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        except Exception:
            pass
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
    """Re-apply raw mode on *fd* when termios drifted back to cooked under prompt_toolkit.

    prompt_toolkit's ``run_in_terminal`` wraps every print-above-the-prompt in a
    ``cooked_mode()`` window; Hermes schedules those cross-thread constantly, and if a
    restore is ever lost (coroutine cancelled mid-window, racing chains, an external
    writer on the shared tty) the kernel line-buffers every keystroke and the CLI
    appears to stop taking input while the process is healthy. This is the last line of
    defense: mirrors ``prompt_toolkit.input.vt100.raw_mode`` flag surgery in place.
    Returns True when drift was healed; False when already raw or not inspectable.
    POSIX-only — callers must not invoke this on Windows (no termios).
    """
    try:
        import termios
        attrs = termios.tcgetattr(fd)
    except Exception:
        return False
    lflag = attrs[3]
    if not (lflag & (termios.ICANON | termios.ECHO)):
        return False  # still raw — nothing to do
    # Same surgery as raw_mode._patch_lflag / _patch_iflag on the *current* attrs so
    # user settings are preserved.
    attrs[3] = lflag & ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
    attrs[0] = attrs[0] & ~(
        termios.IXON
        | termios.IXOFF
        | termios.ICRNL
        | termios.INLCR
        | termios.IGNCR
    )
    # VMIN=1 so reads return per-byte (prompt_toolkit sets this in raw_mode.__enter__).
    attrs[6][termios.VMIN] = 1
    try:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        return False
    return True


def _detect_light_mode_uncached() -> bool:
    """The detection ladder documented above; may raise (caller maps errors to dark)."""
    # 1. Explicit env override
    for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT"):
        v = (os.environ.get(var) or "").strip().lower()
        if _TRUE_RE.match(v):
            return True
        if _FALSE_RE.match(v):
            return False
    # 2. Theme hint
    theme = (os.environ.get("HERMES_TUI_THEME") or "").strip().lower()
    if theme == "light":
        return True
    if theme == "dark":
        return False
    # 3. Explicit bg hex
    bg_lum = _luminance_from_hex(os.environ.get("HERMES_TUI_BACKGROUND") or "")
    if bg_lum is not None:
        return bg_lum >= 0.5
    # 4. COLORFGBG (xterm/Konsole/urxvt)
    cfgbg = (os.environ.get("COLORFGBG") or "").strip()
    if cfgbg:
        last = cfgbg.split(";")[-1] if ";" in cfgbg else cfgbg
        if last.isdigit():
            bg = int(last)
            if bg in {7, 15}:
                return True
            if 0 <= bg < 16:
                return False
    # 5. OSC 11 query (best-effort, only when stdin/stdout are TTY)
    bg_color = _query_osc11_background()
    if bg_color:
        lum = _luminance_from_hex(bg_color)
        if lum is not None:
            return lum >= 0.5
    # 6. TERM_PROGRAM allow-list (currently empty)
    tp = (os.environ.get("TERM_PROGRAM") or "").strip()
    return tp in _LIGHT_DEFAULT_TERM_PROGRAMS


def _detect_light_mode() -> bool:
    global _LIGHT_MODE_CACHE
    if _LIGHT_MODE_CACHE is not None:
        return _LIGHT_MODE_CACHE
    try:
        result = _detect_light_mode_uncached()
    except Exception:
        result = False
    _LIGHT_MODE_CACHE = result
    return result


# Light-mode equivalents of skin colors unreadable on cream Terminal.app backgrounds,
# applied by _SkinAwareAnsi at resolution time.
# IMPORTANT: only remap colors used as STANDALONE foregrounds on the terminal
# background. Colors paired with a dark bg (status bar text on bg:#1a1a2e) would
# become invisible the OTHER direction — hence #C0C0C0/#888888/#555555/#8B8682 are skipped.
_LIGHT_MODE_REMAP: dict[str, str] = {
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
}
# Pre-uppercased lookup table for case-insensitive remapping
_LIGHT_MODE_REMAP_UPPER = {k.upper(): v for k, v in _LIGHT_MODE_REMAP.items()}


def _maybe_remap_for_light_mode(hex_color: str) -> str:
    """In light mode, remap a dark-mode-tuned color to a higher-contrast equivalent. No-op in dark mode."""
    if not _detect_light_mode():
        return hex_color
    if not hex_color or not hex_color.startswith("#"):
        return hex_color
    return _LIGHT_MODE_REMAP_UPPER.get(hex_color.upper(), hex_color)


def _install_skin_light_mode_hook() -> None:
    """Wrap SkinConfig.get_color so EVERY skin color read goes through the light-mode remap. Idempotent."""
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


# Prime the light-mode cache at module load when interactive, so OSC 11 happens
# before prompt_toolkit grabs the tty. Skipped for non-tty contexts (subagents, gateway, tests).
try:
    if sys.stdin.isatty() and sys.stdout.isatty():
        _detect_light_mode()
except Exception:
    pass


class _SkinAwareAnsi:
    """Lazy ANSI escape resolved from the skin engine on first use.

    Acts as a string in f-strings and concatenation. ``.reset()`` forces
    re-resolution after a ``/skin`` switch.
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
# ANSI dim+italic attributes instead of a hardcoded hex so dim/thinking text inherits
# the terminal's default foreground and stays readable in light and dark modes.
_DIM = "\x1b[2;3m"


def _tty_wrap(s: str, sgr: str) -> str:
    """Wrap *s* in an SGR attribute if stdout is a real TTY; plain text otherwise (slash-worker safe)."""
    try:
        return f"{sgr}{s}\x1b[0m" if sys.stdout.isatty() else str(s)
    except Exception:
        return str(s)


def _b(s: str) -> str:
    """Bold if stdout is a real TTY; plain text otherwise."""
    return _tty_wrap(s, "\x1b[1m")


def _d(s: str) -> str:
    """Dim-italic if stdout is a real TTY; plain text otherwise."""
    return _tty_wrap(s, "\x1b[2;3m")


def _accent_hex() -> str:
    """Return the active skin accent color for legacy CLI output lines."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color("ui_accent", "#FFBF00")
    except Exception:
        return "#FFBF00"


def _rich_text_from_ansi(text: str) -> _RichText:
    """Render output that may contain ANSI escapes; literal ``[not markup]`` survives."""
    return _RichText.from_ansi(text or "")


def _strip_markdown_syntax(text: str) -> str:
    """Best-effort markdown marker removal for plain-text display."""
    plain = _rich_text_from_ansi(text or "").plain
    # HR markers: "-"/"_" runs of 3+, but "*" only when exactly 3 — Hermes output
    # commonly shows cron schedules like "* * * * *" verbatim.
    plain = re.sub(r"^\s{0,3}(?:[-_]\s*){3,}$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}(?:\*\s*){3}\s*$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", plain, flags=re.MULTILINE)
    # Blockquotes, lists, and checkboxes are preserved because they carry structure.
    plain = re.sub(r"(```+|~~~+)", "", plain)
    plain = re.sub(r"`([^`]*)`", r"\1", plain)
    plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)___([^_]+)___(?!\w)", r"\1", plain)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", plain)
    # `*emphasis*` only when the inner text is non-whitespace (cron expressions again).
    plain = re.sub(r"\*([^\s*][^*]*?[^\s*])\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", plain)
    plain = re.sub(r"~~([^~]+)~~", r"\1", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip("\n")


_WINDOWS_PATH_WITH_DOT_SEGMENT_RE = re.compile(r"(?i)(?:\b[a-z]:\\|\\\\)[^\s`]*\\\.[^\s`]*")


def _preserve_windows_dot_segments_for_markdown(text: str) -> str:
    r"""Double the ``\`` before hidden directories in Windows-path-looking tokens.

    CommonMark treats ``\.`` as an escaped dot, so Rich would render ``D:\repo\.ai`` as
    ``D:\repo.ai``. Ordinary escapes like ``1\. not a list`` are left alone.
    """
    if "\\." not in text:
        return text

    def _protect(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)\\(?=\.)", r"\\\\", match.group(0))

    return _WINDOWS_PATH_WITH_DOT_SEGMENT_RE.sub(_protect, text)


def _terminal_columns() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def _terminal_width_for_streaming() -> int:
    """Display cells available inside the streamed response box (small margin for resize races)."""
    return max(20, _terminal_columns() - len(_STREAM_PAD) - 2)


def _render_final_assistant_content(text: str, mode: str = "render"):
    """Render final assistant content as markdown, stripped text, or raw text."""
    from rich.markdown import Markdown

    # The final-response Panel renders flush-left with 1 border cell each side; small
    # safety margin so resize races don't push a borderline table into soft-wrap.
    panel_width = max(20, _terminal_columns() - 4)

    normalized_mode = str(mode or "render").strip().lower()
    if normalized_mode == "strip":
        # Strip first (inline markdown changes cell width), then re-align padding.
        return _RichText(realign_markdown_tables(_strip_markdown_syntax(text), panel_width))
    if normalized_mode == "raw":
        return _rich_text_from_ansi(text or "")

    # Rich handles CJK width itself, but normalising model-emitted under-padded tables
    # on the way in gives mid-render fallbacks (narrow panels) consistent input.
    plain = _rich_text_from_ansi(text or "").plain
    plain = _preserve_windows_dot_segments_for_markdown(plain)
    plain = realign_markdown_tables(plain, panel_width)
    return Markdown(plain)


def _post_stream_transform_output(response: str, result: dict | None) -> str:
    """Text still needing display after a streamed response transform.

    When the transformed text keeps the streamed response as a prefix, only the suffix
    is printed; a full replacement has no safe suffix, so the complete final response
    is printed rather than silently dropped.
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


def _output_history_recording() -> bool:
    return _OUTPUT_HISTORY_ENABLED and not _OUTPUT_HISTORY_REPLAYING and not _OUTPUT_HISTORY_SUPPRESSED


def _record_output_history_entry(entry) -> None:
    if _output_history_recording():
        _OUTPUT_HISTORY.append(entry)


def _record_output_history(text: str) -> None:
    if not _output_history_recording():
        return
    normalized = str(text).replace("\r", "").rstrip("\n")
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
            # One ANSI payload, not per-line prints: each prompt_toolkit print forces a
            # synchronous terminal I/O + redraw, which reads as a waterfall of old output.
            _pt_print(_PT_ANSI("\n".join(rendered_lines)))
    except Exception:
        pass
    finally:
        _OUTPUT_HISTORY_REPLAYING = False


def _pt_print_ansi(text: str) -> None:
    """``_pt_print(ANSI(text))``, falling back to ``print`` when stdout is not a real console."""
    try:
        _pt_print(_PT_ANSI(text))
    except Exception:
        # prompt_toolkit raises NoConsoleScreenBufferError (Windows) / OSError when
        # stdout is e.g. a subprocess worker's log file.
        try:
            print(text)
        except Exception:
            pass


def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI written via print() is swallowed by patch_stdout's StdoutProxy; routing
    through print_formatted_text(ANSI(...)) renders real colors. From a background
    thread while an Application is running (self-review summaries, curator output), a
    direct print races the input area's redraw and can end up buried behind the
    prompt — those cases go through ``run_in_terminal`` via ``call_soon_threadsafe``.
    """
    _record_output_history(text)

    try:
        from prompt_toolkit.application import get_app_or_none, run_in_terminal
    except Exception:
        _pt_print(_PT_ANSI(text))
        return

    try:
        app = get_app_or_none()
    except Exception:
        app = None

    # No active app: the direct print matches existing behavior (spinner frames,
    # streamed tokens, tool activity prefixes, …).
    if app is None or not getattr(app, "_is_running", False):
        _pt_print_ansi(text)
        return

    import asyncio as _asyncio

    try:
        loop = app.loop  # type: ignore[attr-defined]
    except Exception:
        loop = None
    try:
        # get_running_loop(), not get_event_loop(): the latter warns from threads with
        # no current loop (e.g. the process_loop background thread).
        current_loop = _asyncio.get_running_loop()
    except Exception:
        current_loop = None
    # No loop, or same thread as the app's loop → safe to print directly.
    if loop is None or (current_loop is loop and loop.is_running()):
        _pt_print(_PT_ANSI(text))
        return

    def _schedule():
        # run_in_terminal() returns a coroutine/Future (pt >= 3.0) that must be
        # scheduled or the output is silently dropped, or None (mocks / older PT) when
        # it already ran the callable synchronously. Never fall back to a bare
        # _pt_print when ensure_future raises — the mock path already printed.
        try:
            import asyncio as _aio
            import inspect as _inspect
            coro = run_in_terminal(lambda: _pt_print(_PT_ANSI(text)))
            if coro is not None and (_inspect.isawaitable(coro) or _inspect.iscoroutine(coro)):
                _aio.ensure_future(coro)
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

    ``message`` is a str, or a list of OpenAI-style content parts when an image is
    attached (naive ``note + message`` then raises TypeError — e.g. ``/model`` followed
    by a pasted image). For lists the note is folded into the first text part or
    inserted as a leading text part. Unknown shapes are returned unchanged.
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
        return [{"type": "text", "text": note}, *parts]
    return message


def _pt_app_is_running() -> bool:
    """Whether a prompt_toolkit Application currently owns the live terminal."""
    try:
        from prompt_toolkit.application import get_app_or_none
        app = get_app_or_none()
    except Exception:
        return False
    return app is not None and bool(getattr(app, "_is_running", False))


def _cli_visible_print(text: str = "") -> None:
    """Print normally unless prompt_toolkit owns the live terminal (then route via ``_cprint``).

    Bare ``print()`` is swallowed by ``patch_stdout`` while an Application runs, so
    ``/sessions`` and ``/history`` would render nothing.
    """
    if _pt_app_is_running():
        _cprint(text)
    else:
        print(text)


# ---------------------------------------------------------------------------
# File-drop / local attachment detection — pure helpers.
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
    # Android roots are POSIX paths — literal "/" so the hint is right even on Windows.
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

    if token[0] == token[-1] and token[0] in {'"', "'"}:
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

    # os.stat raises OSError (ENAMETOOLONG) for structurally invalid paths — e.g. a
    # pasted `/goal <long prose>` that passed _detect_file_drop's `/` prefilter. Without
    # this guard the error reaches process_loop and the input is silently lost.
    try:
        if not resolved.exists() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def _file_drop_result(path: Path, remainder: str) -> dict:
    return {"path": path, "is_image": path.suffix.lower() in _IMAGE_EXTENSIONS, "remainder": remainder}


def _detect_file_drop(user_input: str) -> "dict | None":
    """Detect a dragged/pasted local file path at the start of *user_input*.

    Returns ``{"path": Path, "is_image": bool, "remainder": str}`` or None. Catches
    paths before they are mistaken for slash commands (incl. Termux ``~/storage/...``).
    """
    if not isinstance(user_input, str):
        return None

    stripped = user_input.strip()
    if not stripped:
        return None

    # Optionally quoted; then /, ~, ./, ../, a Windows drive prefix, or (unquoted) file://.
    quoted = stripped[:1] in {"'", '"'}
    unquoted = stripped[1:] if quoted else stripped
    starts_like_path = (
        unquoted.startswith(("/", "~", "./", "../"))
        or (not quoted and unquoted.startswith("file://"))
        or (len(unquoted) >= 3 and unquoted[1] == ":" and unquoted[2] in {"\\", "/"} and unquoted[0].isalpha())
    )
    if not starts_like_path:
        return None

    direct_path = _resolve_attachment_path(stripped)
    if direct_path is not None:
        return _file_drop_result(direct_path, "")

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
    return _file_drop_result(drop_path, remainder)


def _format_image_attachment_badges(attached_images: list[Path], image_counter: int, width: int | None = None) -> str:
    """Attached-image badge row: compact summary on narrow terminals (Termux), per-image badges otherwise."""
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
    return " ".join(f"[📎 Image #{base + i}]" for i in range(len(attached_images)))


def _should_auto_attach_clipboard_image_on_paste(pasted_text: str) -> bool:
    """Auto-attach clipboard images only for image-only paste gestures."""
    return not pasted_text.strip()


_strip_leaked_bracketed_paste_wrappers = _lazy_shim(
    "hermes_cli.input_sanitize", "strip_leaked_bracketed_paste_wrappers", "_strip_leaked_bracketed_paste_wrappers"
)


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

    1. Inflate ``previous_screen.height`` when the new screen is taller so pt skips
       the reserve-vertical-space cursor move that stamps chrome into scrollback.
    2. On AttributeError/TypeError from a corrupt previous paint buffer (classic after
       tmux attach at the same width), retry once with ``previous_screen=None`` so pt
       first-paints cleanly instead of crashing the event loop.
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

    ``Vt100Parser.feed()`` buffers all input while waiting for the ESC[201~ end mark;
    if the terminal drops it (race, torn write, SSH glitch, sleep/wake) input appears
    frozen forever. The wrapper flushes the buffer as a normal ``BracketedPaste`` event
    after ``_BP_TIMEOUT_S`` seconds without an end marker. Idempotent via the
    ``_hermes_bp_timeout_patched`` module sentinel.
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
                    remaining = self_parser._paste_buffer[end_index + len(end_mark):]
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
                # Re-inline the normal feed path: calling the original would
                # double-buffer after the bracketed-paste entry transition.
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


# Cursor Position Report (CPR / DSR) replies ``ESC[<row>;<col>R`` to prompt_toolkit's
# ``ESC[6n`` queries can race past the input parser under resize storms / tab switches
# and land in the input buffer as literal text; the visible ``^[[...R`` form appears
# when a prior filter stripped the ESC byte.
_DSR_CPR_ESC_RE = re.compile(r"\x1b\[\d+;\d+R")
_DSR_CPR_VISIBLE_RE = re.compile(r"\^\[\[\d+;\d+R")
_SGR_MOUSE_ESC_RE = re.compile(r"\x1b\[<\d+;\d+;\d+[Mm]")
_SGR_MOUSE_VISIBLE_RE = re.compile(r"\^\[\[<\d+;\d+;\d+[Mm]")
# Bare "<btn;col;rowM" fragments (ESC and "^[[" dropped). Deliberately broad: such
# fragments are almost never intentional input and stripping beats corrupted prompts.
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

    Ghostty must be pushed ONLY modifyOtherKeys, not the Kitty keyboard protocol: its
    Kitty disambiguate mode strips Alt from Backspace (Option+Backspace arrives as bare
    \\x7f, breaking backward-kill-word — upstream bug), while its modifyOtherKeys is
    correct. Matches exactly the two conditions that admit Ghostty through
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

    Writes BOTH the Kitty keyboard protocol push (CSI >1u, disambiguate mode) and xterm
    modifyOtherKeys level 2 (CSI >4;2m), mirroring the Ink TUI: kitty dropped
    modifyOtherKeys entirely while tmux/VS Code only accept modifyOtherKeys. Either
    protocol re-encodes modified keys as escape sequences (``ESC[<cp>;<mod>u`` /
    ``ESC[27;<mod>;<cp>~``) that stock prompt_toolkit barely maps — Ctrl+C once arrived
    as ``ESC[99;5u`` and died — so ``install_modify_other_keys_aliases()`` (run at CLI
    startup) must populate ``ANSI_SEQUENCES`` under both formats first. Ghostty gets only
    modifyOtherKeys (see ``_is_ghostty_terminal``). The exit reset sequence pops both modes.
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
    """Whether classic CLI harness-standard multiline fallbacks (Ctrl+J = newline) are on.

    Default on, matching adjacent agent harnesses. POSIX PTYs that send bare LF for
    plain Enter can set ``display.cli_multiline_shortcuts: false`` to restore the
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

    Windows Terminal, WSL, SSH, Ghostty and some modern terminals deliver Ctrl+Enter as
    bare LF (c-j); binding c-j to submit there makes Ctrl+Enter submit. Local thin POSIX
    PTYs that deliver Enter as LF still need c-j bound to submit when
    display.cli_multiline_shortcuts is disabled, so that legacy opt-out survives.
    """
    env = os.environ
    if sys.platform == "win32":
        return True
    if any(env.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "WT_SESSION",
                                "GHOSTTY_RESOURCES_DIR", "GHOSTTY_BIN_DIR")):
        return True
    if env.get("TERM", "").lower() == "xterm-ghostty" or env.get("TERM_PROGRAM", "").lower() == "ghostty":
        return True
    if "microsoft" in env.get("WSL_DISTRO_NAME", "").lower():
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

    Enter is always submit; c-j (Ctrl+J/LF) is left for the newline handler unless
    ``display.cli_multiline_shortcuts: false`` restores the legacy POSIX submit
    fallback — and even then environments where Ctrl+Enter arrives as c-j
    (``_preserve_ctrl_enter_newline``) keep it reserved for newline.
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

    Delayed CPR replies leak into the status line and can freeze input on SSH/slow
    PTYs and on loaded local TTYs. ``PROMPT_TOOLKIT_NO_CPR=1`` always suppresses;
    native Windows otherwise keeps prompt_toolkit's default (no Application coverage
    yet); every other platform suppresses (CPR is only a layout hint).
    """
    return os.environ.get("PROMPT_TOOLKIT_NO_CPR", "") == "1" or sys.platform != "win32"


def _build_cpr_disabled_output(stdout):
    """Build a Vt100_Output that never sends Cursor Position Report queries.

    ``enable_cpr=False`` marks CPR ``NOT_SUPPORTED`` so ``ESC[6n`` is never sent and
    prompt_toolkit uses its heuristic height; input-side
    ``_strip_leaked_terminal_responses`` stays as belt-and-suspenders.
    ``Vt100_Output.from_pty()`` doesn't expose ``enable_cpr`` in pt 3.x, so its
    ``get_size`` setup is reproduced here. Returns None on failure (caller keeps pt's default).
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
    """Strip leaked CPR/DSR replies and SGR mouse-report fragments from user input.

    Returns ``(cleaned_text, had_mouse_reports)`` so callers can trigger an in-place
    terminal mode recovery when mouse reports leaked.
    """
    if not text:
        return text, False

    has_esc = "\x1b[" in text
    has_visible = "^[" in text
    has_bare_mouse = "<" in text and ";" in text and ("M" in text or "m" in text)
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

    The prompt is injected via BeforeInput, so it consumes cells only on logical line 0;
    continuation rows use the full width. Never substitute a fake wide fallback: mis-
    allocating the TextArea height leaves stale prompt/input cells at the terminal bottom.
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
        line_width = get_cwidth(line or "")
        display_width = line_width + (prompt_width if index == 0 else 0)
        if display_width <= 0:
            visual_lines += 1
        else:
            visual_lines += max(1, -(-display_width // columns))

    return min(max(visual_lines, 1), max(1, int(max_height or 1)))


def _status_bar_visible_from_display_config(display_config: object) -> bool:
    """Initial classic-CLI status-bar visibility from display config.

    YAML parses bare ``off`` as False, while older snapshots or hand edits may hold
    strings like ``"off"``/``"hidden"``; treat both so a new process never re-enables
    a status bar the user disabled.
    """
    if not isinstance(display_config, dict):
        display_config = {}
    statusbar_config = display_config.get("statusbar", display_config.get("tui_statusbar", "top"))
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

    return message, list(dict.fromkeys(images))


# OSC sequences (e.g. OSC-8 hyperlinks): prompt_toolkit's ANSI parser strips the ESC
# but passes the payload through as literal text, garbling TUI output.
_OSC_ESCAPE_RE = re.compile(r"\x1b\][\s\S]*?(?:\x07|\x1b\\)")


class ChatConsole:
    """Rich Console drop-in that routes rendered ANSI through ``_cprint`` so colors
    and markup survive prompt_toolkit's patch_stdout in the interactive loop."""

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
        output = _OSC_ESCAPE_RE.sub("", output)
        for line in output.rstrip("\n").split("\n"):
            _cprint(line)

    @contextmanager
    def status(self, *_args, **_kwargs):
        """No-op Rich-compatible status context.

        Slash-command helpers call ``console.status(...)``; a silent context keeps them
        compatible without duplicating ``HermesCLI._busy_command()``'s busy indicator.
        """
        yield self

# Rich-markup ASCII art (re-exported; canonical copies live in hermes_cli.banner).
from hermes_cli.banner import HERMES_AGENT_LOGO, HERMES_CADUCEUS  # noqa: E402,F401



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
    """True if *text* looks like a slash command (``/help``), not a pasted path
    (``/Users/x/file.md ...``): a command's first word has no further ``/``."""
    if not text or not text.startswith("/"):
        return False
    return "/" not in text.split()[0][1:]


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


build_skill_invocation_message = _lazy_shim("agent.skill_commands", "build_skill_invocation_message")
build_preloaded_skills_prompt = _lazy_shim("agent.skill_commands", "build_preloaded_skills_prompt")


def get_skill_bundles() -> dict:
    global _skill_bundles
    if _skill_bundles is None:
        from agent.skill_bundles import get_skill_bundles as _impl

        _skill_bundles = _impl()
    return _skill_bundles


build_bundle_invocation_message = _lazy_shim("agent.skill_bundles", "build_bundle_invocation_message")


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
    if isinstance(skills, (list, tuple)):
        raw_values = [str(item) for item in skills if item is not None]
    else:
        raw_values = [str(skills)]
    parts = (p.strip() for raw in raw_values for p in raw.split(","))
    return list(dict.fromkeys(p for p in parts if p))


def save_config_value(key_path: str, value: any) -> bool:
    """Persist ``key_path`` (dot-separated, e.g. "agent.system_prompt") = value; True on success.

    ALWAYS targets HERMES_HOME/config.yaml (created if needed), resolved live so profile
    switches and test isolation land right. Never the repo's cli-config.yaml: it is a
    shipped template no config reader loads, so a value written there silently vanishes.
    """
    config_path = get_hermes_home() / 'config.yaml'

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write preserving comments, ordering, quotes and readable Unicode.
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        # Owner-only permissions: config files contain API keys.
        try:
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        # /model and the TUI persist through here rather than `hermes config set`;
        # surface the same fail-closed cron drift warning for every model switch.
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
    """Map ``moa:<preset>`` to ``("moa", "<preset>")``; anything else -> ``(None, model)``.

    Gives ``hermes chat -Q -m moa:<preset>`` the same routing as the interactive
    ``/moa`` command (``resolve_runtime_provider`` / ``agent_init`` key off
    ``provider == "moa"``); the raw string would be rejected by the real provider.
    """
    if isinstance(model, str):
        stripped = model.strip()
        if stripped.lower().startswith("moa:"):
            preset = stripped.split(":", 1)[1].strip()
            if preset:
                return "moa", preset
    return None, model

_split_model_config_default = _lazy_shim("hermes_cli.config", "split_model_config_default", "_split_model_config_default")


class _VoiceInputMessage:
    """Sentinel for voice-transcribed messages in ``_pending_input`` so the concise
    voice-response prefix applies only to microphone input, not typed text."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _SeededQueryMessage:
    """Sentinel wrapper for a ``-q/--query`` prompt seeded into an interactive session.

    The seeded prompt is arbitrary user text (OS launcher, script) and must be treated
    LITERALLY — no slash-command routing, ``!`` shell dispatch, or file-drop detection.
    ``process_loop`` skips those dispatchers for this message only.
    """

    __slots__ = ("text", "images")

    def __init__(self, text: str, images=None):
        self.text = text or ""
        self.images = list(images or [])

    def __str__(self) -> str:
        return self.text


def _should_seed_interactive(query, image, quiet: bool, oneshot: bool) -> bool:
    """Whether a ``-q/--image`` invocation should seed an interactive session.

    On a real TTY ``chat -q`` submits the prompt as the first interactive turn. The
    legacy answer-and-exit behavior is kept for every automation surface: ``--oneshot``,
    ``-Q/--quiet``, and any non-TTY stdin/stdout (kanban, cron, pipes, A2A).
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

    ``result`` is written by the worker thread and read after the join (same object, so
    late writes from an abandoned thread stay visible). ``tts_normal_exit`` is set only
    when the TTS worker drained on its own so the ``finally`` never cuts the last sentence.
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
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


class HermesCLI(CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, CLITuiMixin, CLIStatusBarMixin, CLIVoiceMixin, CLIModelSwitchMixin, CLISessionMixin, CLIStreamMixin, CLIModalMixin, CLITerminalMixin, CLIInfoMixin, CLILoopsMixin, CLIChatTurnMixin):
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
        self.console = Console()
        self.config = CLI_CONFIG
        display = CLI_CONFIG["display"]
        self.compact = compact if compact is not None else display.get("compact", False)
        # tool_progress: "off" | "new" | "all" | "verbose". YAML 1.1 parses bare
        # `off` as False — normalise to the string.
        _raw_tp = display.get("tool_progress", "all")
        self.tool_progress_mode = "off" if _raw_tp is False else str(_raw_tp)
        # focus_view (/focus): display-only. Snaps tool_progress to "off" so the
        # EXISTING suppression path hides per-tool lines and stashes the pre-focus
        # mode for /focus off. Never changes what is sent to the model
        # (hermes_cli/focus_view.py).
        self._focus_view_enabled = bool(display.get("focus_view", False))
        self._focus_saved_tool_progress = self._focus_last_counted_tool = None
        self._focus_hidden_lines = 0
        if self._focus_view_enabled:
            from hermes_cli.focus_view import (
                FOCUS_TOOL_PROGRESS_MODE,
                normalize_tool_progress_mode,
            )

            self._focus_saved_tool_progress = normalize_tool_progress_mode(
                self.tool_progress_mode
            )
            self.tool_progress_mode = FOCUS_TOOL_PROGRESS_MODE
        self.resume_display = display.get("resume_display", "full")  # "full" | "minimal"
        self.bell_on_complete = display.get("bell_on_complete", False)
        # bell_on_prompt: terminal bell whenever a blocking prompt modal opens
        # (clarify, approval, sudo password, secret capture).
        self.bell_on_prompt = display.get("bell_on_prompt", False)
        self.show_reasoning = display.get("show_reasoning", True)
        # reasoning_full: post-response recap box uncollapsed instead of the first 10 lines.
        self.reasoning_full = display.get("reasoning_full", False)
        _configure_output_history(
            enabled=display.get("persistent_output", True),
            max_lines=display.get("persistent_output_max_lines", 200),
        )
        # busy_input_mode: "interrupt" (Enter redirects the current run), "queue"
        # (Enter queues for the next turn) or "steer" (inject mid-run via /steer).
        _bim = str(display.get("busy_input_mode", "interrupt")).strip().lower()
        self.busy_input_mode = _bim if _bim in ("queue", "steer") else "interrupt"

        # self.verbose ONLY controls global DEBUG logging (root logger level).
        # display.tool_progress="verbose" controls tool-call rendering and is
        # independent (see _apply_logging_levels): coupling the two made every
        # module's DEBUG logs spew to the console whenever a user set
        # tool_progress: verbose.
        self.verbose = bool(verbose) if verbose is not None else False

        self.streaming_enabled = display.get("streaming", False)
        self.show_timestamps = display.get("timestamps", False)
        self.timestamp_format = display.get("timestamp_format", "%H:%M")
        self.final_response_markdown = str(
            display.get("final_response_markdown", "strip")
        ).strip().lower() or "strip"
        if self.final_response_markdown not in {"render", "strip", "raw"}:
            self.final_response_markdown = "strip"

        self._inline_diffs_enabled = display.get("inline_diffs", True)  # diff previews for write actions

        # Per-turn accounting (display.turn_summary / display.spinner_token_flow):
        # CLI-only chrome; the collector rides the tool-progress feed this class
        # already receives, so no agent-loop bookkeeping is involved.
        self._turn_summary_enabled = bool(display.get("turn_summary", True))
        self._spinner_token_flow_enabled = bool(
            display.get("spinner_token_flow", True)
        )
        self._turn_summary_collector = None
        self._turn_summary_start = 0.0
        self._turn_token_baseline = 0
        # True only while an interactive (run()-loop) turn is in flight; -Q and
        # gateway paths never set it, which keeps the summary line off them.
        self._interactive_turn = False

        # Submitted multiline user-message preview (display.user_message_preview)
        _ump = display.get("user_message_preview", {})
        if not isinstance(_ump, dict):
            _ump = {}
        self.user_message_preview_first_lines = max(1, _int_or(_ump.get("first_lines", 2), 2))
        self.user_message_preview_last_lines = max(0, _int_or(_ump.get("last_lines", 2), 2))

        # Streaming display state
        self._stream_buf = ""  # partial line buffer for line-buffered rendering
        self._reasoning_preview_buf = ""  # coalesce tiny reasoning chunks for [thinking] output
        self._stream_started = self._stream_box_opened = False  # first delta seen / box header printed
        # Lines that may belong to a markdown table are held here until the block
        # ends so they can be re-padded with wcwidth-aware widths.
        self._stream_table_buf: list[str] = []
        self._in_stream_table = False
        self._pending_edit_snapshots = {}
        self._last_input_mode_recovery = self._last_termios_drift_check = 0.0
        self._input_mode_recovery_notice_shown = self._termios_drift_notice_shown = False

    def _init_model_routing(self, model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, checkpoints, pass_session_id, ignore_rules):
        """Resolve model/provider/base_url, turn limits, toolsets, checkpoints, prompt/personality, reasoning + routing config."""
        _model_config = self._init_model_and_provider(model, provider, api_key, base_url)
        self._init_turn_limits(max_turns, run_budget)
        self._init_toolsets(toolsets)
        self._init_checkpoints_and_rules(checkpoints, pass_session_id, ignore_rules)
        self._init_prompt_and_reasoning(reasoning)

    def _init_model_and_provider(self, model, provider, api_key, base_url):
        """Priority: CLI args > env vars > config file. Returns the raw ``model`` config section."""
        # Model comes from the CLI arg or config.yaml (single source of truth) —
        # LLM_MODEL/OPENAI_MODEL env vars are NOT checked, so multi-agent setups
        # don't stomp each other through the environment.
        _model_config = CLI_CONFIG.get("model", {})
        _raw_default = (_model_config.get("default") or _model_config.get("model") or "") if isinstance(_model_config, dict) else (_model_config or "")
        # A dict-valued default (``model.default: {provider: ..., model: ...}``)
        # carries its own provider; flatten it so that nested provider feeds
        # ``requested_provider`` instead of being replaced by the outer merged
        # ``model.provider`` (typically "auto").
        _config_model, _nested_provider = _split_model_config_default(_raw_default)
        _DEFAULT_CONFIG_MODEL = ""
        # Whether -m/--model was passed: resume must not clobber an explicit
        # override with the session's stored model.
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
        # ``moa:<preset>`` selects the MoA virtual provider in one shot (parity
        # with /moa and the picker). Done before provider resolution so
        # ``-Q -m moa:<preset>`` never hits the real provider with an unknown
        # model (#56828); the prefix wins over an explicit --provider.
        _moa_provider_override, self.model = _normalize_moa_model(self.model)
        # max_tokens: HERMES_MAX_TOKENS env overrides config.
        _env_mt = os.environ.get("HERMES_MAX_TOKENS")
        if _env_mt:
            self.max_tokens = _int_or(_env_mt, None)
        elif isinstance(_model_config, dict):
            _mt = _model_config.get("max_tokens")
            self.max_tokens = _mt if isinstance(_mt, int) else None
        else:
            self.max_tokens = None
        # Auto-detect the model from a local server if still on the default.
        if self.model == _DEFAULT_CONFIG_MODEL:
            _base_url = (_model_config.get("base_url") or "") if isinstance(_model_config, dict) else ""
            if base_url_hostname(_base_url) in ("localhost", "127.0.0.1"):
                from hermes_cli.runtime_provider import _auto_detect_local_model
                _detected = _auto_detect_local_model(_base_url)
                if _detected:
                    self.model = _detected
        # Provider-specific normalisation may silently override the global
        # default but must warn when overriding an explicit choice. A config
        # model equal to the global fallback is NOT explicit; "gpt-5.3-codex" is.
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
        # `--provider <custom>` without `-m` must use that entry's default_model,
        # otherwise the global model.default is sent to the custom endpoint and
        # the compressor inherits the wrong context length (#86978).
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
        # Match the key to the resolved base_url (issue #560); re-resolved by
        # _ensure_runtime_credentials() before first use.
        if self.base_url and base_url_host_matches(self.base_url, "openrouter.ai"):
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        return _model_config

    def _init_turn_limits(self, max_turns, run_budget):
        """max_turns: CLI arg > config > env var > default; run budget: CLI flag > config."""
        # Everything goes through resolve_turn_limit() so "none"/"unlimited"
        # (-> sys.maxsize) are accepted alongside ints.
        from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
        if max_turns is not None:
            self.max_turns = _resolve_turn_limit(max_turns)
        elif CLI_CONFIG["agent"].get("max_turns") is not None:
            self.max_turns = _resolve_turn_limit(CLI_CONFIG["agent"]["max_turns"])
        elif CLI_CONFIG.get("max_turns") is not None:
            # KEEP: root-level max_turns is only folded at load time
            # (_normalize_max_turns_config), never migrated on disk, and configs
            # read through other paths may bypass it — this is the only safety net.
            self.max_turns = _resolve_turn_limit(CLI_CONFIG["max_turns"])
        else:
            # Env bridge (gateway/run.py or the user); empty/unset -> unlimited.
            self.max_turns = _resolve_turn_limit(os.getenv("HERMES_MAX_ITERATIONS"))
        # None keeps the wall-clock budget fully off (AIAgent stays dormant).
        if run_budget is not None:
            self.run_budget_seconds = run_budget
        else:
            self.run_budget_seconds = CLI_CONFIG["agent"].get("run_budget_seconds")

    def _init_toolsets(self, toolsets):
        self.enabled_toolsets = toolsets
        from agent.skill_utils import parse_config_string_list

        self.disabled_toolsets = parse_config_string_list(CLI_CONFIG["agent"].get("disabled_toolsets"))

        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # MCP server names resolve via live registry aliases registered during
            # discover_mcp_tools, which hasn't run yet — exclude them from validation.
            mcp_names = set((CLI_CONFIG.get("mcp_servers") or {}).keys())
            invalid = [t for t in toolsets if not validate_toolset(t) and t not in mcp_names]
            if invalid:
                self._console_print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")

    def _init_checkpoints_and_rules(self, checkpoints, pass_session_id, ignore_rules):
        # Filesystem checkpoints: CLI flag > config
        cp_cfg = CLI_CONFIG.get("checkpoints", {})
        if isinstance(cp_cfg, bool):
            cp_cfg = {"enabled": cp_cfg}
        self.checkpoints_enabled = checkpoints or cp_cfg.get("enabled", False)
        self.checkpoint_max_snapshots = cp_cfg.get("max_snapshots", 20)
        self.checkpoint_max_total_size_mb = cp_cfg.get("max_total_size_mb", 500)
        self.checkpoint_max_file_size_mb = cp_cfg.get("max_file_size_mb", 10)
        self.pass_session_id = pass_session_id
        # --ignore-rules (flag or the env var set by `hermes chat --ignore-rules`):
        # AIAgent gets skip_context_files=True and skip_memory=True so
        # AGENTS.md/SOUL.md/.cursorrules and persistent memory are not loaded.
        self.ignore_rules = ignore_rules or os.environ.get("HERMES_IGNORE_RULES") == "1"

    def _init_prompt_and_reasoning(self, reasoning):
        """Ephemeral system prompt/prefill, reasoning + service tier, OpenRouter routing knobs, fallback chain."""
        # Env var wins, then display.personality / agent.system_prompt via
        # hermes_cli.personality (single owner of overlay resolution).
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

        # Per-model override > global reasoning_effort (shared chokepoint, #21256).
        from hermes_constants import resolve_reasoning_config
        self.reasoning_config = resolve_reasoning_config(CLI_CONFIG, self.model)
        # An explicit --reasoning wins for this run only (never persisted; Kanban
        # pins a task's thinking depth this way). An unparseable level is ignored
        # with a warning, same contract as the config path.
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

        # OpenRouter Pareto Code router: coding-score floor (0.0-1.0), only applied
        # when model.model == "openrouter/pareto-code". Empty/None/out-of-range = unset.
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

        # Fallback provider chain (new ``fallback_providers`` merged with legacy
        # ``fallback_model`` entries so old configs still participate).
        self._fallback_model = get_fallback_chain(CLI_CONFIG)

    def _init_runtime_state(self, resume):
        """Session store + all per-run mutable state (queues, overlays, pet/voice/status-bar fields)."""
        # Runtime signature of the initialised agent; a change across turns
        # (/model, credential rotation) rebuilds the agent.
        self._active_agent_route_signature = None
        self.agent: Optional[Any] = None  # initialized on first use
        self._tool_callbacks_installed = False
        self._tirith_security_checked = False
        self._app = None  # prompt_toolkit Application (set in run())

        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self._resumed = False
        # Per-prompt elapsed timer: started each turn, frozen when the agent thread
        # completes, shown in the status bar.
        self._prompt_start_time: Optional[float] = None
        self._prompt_duration: float = 0.0
        self._last_turn_finished_at: Optional[float] = None
        self._init_session_store()
        # Deferred title: held until the session row exists in the DB.
        self._pending_title: Optional[str] = None
        if resume:
            self.session_id = resume
            self._resumed = True
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()

        self._history_file = _hermes_home / ".hermes_history"  # persistent input recall
        self._last_invalidate: float = 0.0  # throttle UI repaints
        self._app = None
        self._init_ui_state()

    def _init_session_store(self):
        """Open the SQLite session store early (so /title works before the first message) + opportunistic maintenance."""
        self._session_db = None
        self._session_db_unavailable = False
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            # #41386: with no store the transcript is NOT persisted — the live chat
            # looks healthy but resume later shows a truncated/empty session, so
            # surface it prominently rather than only logging.
            self._session_db_unavailable = True
            logger.warning("Failed to initialize SessionDB — session will NOT be indexed for search: %s", e)
            try:
                # Console is the module-scope import; a function-local import would
                # shadow it for the whole method body.
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
        # state.db maintenance: at most once per min_interval_hours, tracked in
        # state_meta so it is shared across every Hermes process for this
        # HERMES_HOME; never blocks startup on failure.
        _run_state_db_auto_maintenance(self._session_db)
        # Orphan/stale checkpoint shadow repos (opt-in via checkpoints.auto_prune).
        _run_checkpoint_auto_maintenance()

    def _init_ui_state(self):
        """Per-run mutable UI state shared by interactive run() and single-query chat().

        Everything here must exist before any direct chat() call because
        single-query mode does not go through run().
        """
        self._agent_running = False
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        # Whether the turn that just finished was Ctrl+C'd; _maybe_continue_goal_after_turn
        # reads it so /goal never auto-queues on top of a user-cancelled turn.
        self._last_turn_interrupted = False
        # Set when stdout/PTY raises EIO (broken pipe after a stream-stall interrupt)
        # to freeze UI paints instead of spinning on escape-sequence writes (#81521).
        self._terminal_io_broken = False
        self._should_exit = False
        # /exit --delete: delete this session's SQLite history + transcripts at shutdown.
        self._delete_session_on_exit = False
        # /update: run() execs relaunch() after prompt_toolkit has fully exited and
        # restored terminal modes — on the main thread, not the process_loop thread.
        self._pending_relaunch: list[str] | None = None
        self._last_ctrl_c_time = 0
        # Blocking-prompt overlays (clarify / sudo / approval / slash-confirm / model picker).
        self._clarify_state = self._clarify_multi_base = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._clarify_prefill = ""
        self._sudo_state = self._modal_input_snapshot = self._approval_state = None
        self._sudo_deadline = self._approval_deadline = 0
        self._approval_lock = threading.Lock()
        self._slash_confirm_state = self._model_picker_state = None
        self._slash_confirm_deadline = 0
        # Composer placeholder chosen once per session so it stays stable on screen.
        try:
            from hermes_cli.tips import get_random_composer_placeholder
            self._composer_placeholder = get_random_composer_placeholder()
        except Exception:
            self._composer_placeholder = ""
        self._command_palette_state = None
        # Armed by a bare `/resume` list so the next bare number selects that
        # session; one-shot, cleared on the next submitted input (#34584).
        self._pending_resume_sessions = None
        # One-shot agent seed set by a slash handler (e.g. /blueprint <name>);
        # consumed by the interactive loop right after process_command() returns.
        self._pending_agent_seed = self._secret_state = None
        self._secret_deadline = 0
        self._spinner_text: str = ""  # thinking spinner text for TUI
        self._tool_start_time: float = 0.0  # monotonic start of the current tool (live elapsed)
        self._pending_tool_info: dict = {}  # function_name -> list of (preview, args) for stacked scrollback
        self._last_scrollback_tool: str = ""  # last tool name printed to scrollback ("new" dedup)
        self._command_running = self._command_blocks_input = False
        self._command_status = ""
        # Petdex mascot (opt-in via display.pet). Kitty/Ghostty: Unicode placeholders
        # + out-of-band image transmission; other terminals: truecolor half-blocks.
        self._pet_renderer = self._pet_anim_thread = None  # agent.pet.render.PetRenderer | None
        self._pet_slug = self._pet_kitty_pending = ""
        self._pet_enabled = self._pet_anim_running = False
        self._pet_cols: int = 18
        self._pet_scale: float = 0.7
        self._pet_frames_cache: dict = {}  # state -> list[grid]
        self._pet_kitty_cache: dict = {}  # state -> kitty placeholder payload
        self._pet_kitty_image_id = self._pet_frame_idx = 0
        self._pet_lock = threading.Lock()
        self._pet_cfg_checked: float = 0.0
        # Transient reaction beats (wave/jump/failed) + steady reasoning flag.
        self._pet_event: str = ""
        self._pet_event_until: float = 0.0
        self._pet_reasoning = self._pet_turn_error = False
        self._attached_images: list[Path] = []
        self._image_counter = 0
        # Ctrl+S prompt stash. In-memory only: drafts routinely contain secrets.
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
        self._voice_mode = self._voice_tts = self._voice_recording = False
        self._voice_processing = self._voice_continuous = False
        self._voice_recorder = None
        self._voice_tts_done = threading.Event()
        self._voice_tts_done.set()
        self._voice_tts_stop = None  # active streaming pipeline's stop event
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # most recently spoken TTS text (echo guard, #75780)
        self._voice_barge_phase = None  # "generation" or "playback" phase of the last barge trip

        self._status_bar_visible = _status_bar_visible_from_display_config(
            CLI_CONFIG.get("display") if isinstance(CLI_CONFIG, dict) else None
        )
        self._battery_visible = bool(CLI_CONFIG["display"].get("battery", False))  # /battery, persisted
        # While True the input rules + status bar stay hidden until the next input:
        # set by _recover_after_resize() so a SIGWINCH cannot stamp a fresh status
        # bar over one the terminal just reflowed into scrollback (#19280, #22976).
        self._status_bar_suppressed_after_resize = self._resize_recovery_pending = False
        self._resize_recovery_lock = threading.Lock()
        self._resize_recovery_timer = None
        # Debounced timer clearing that suppression once the reflow settles.
        self._status_bar_unsuppress_timer = None
        # Last width seen by the resize handler: width change (reflow -> possible
        # ghost chrome, needs a viewport clear) vs rows-only change. None until
        # the first resize.
        self._last_resize_width = None

        self._background_tasks: Dict[str, threading.Thread] = {}  # task_id -> thread
        self._background_task_counter = 0

        # Cache-hit ratio baseline — reset on model switch and on context compression
        # so the bar reflects the current cache regime, not a lifetime average.
        self._cache_hit_baseline_prompt = self._cache_hit_baseline_read = 0
        self._cache_hit_baseline_compressions = 0
        self._cache_hit_baseline_model: Optional[str] = None

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

    # Petdex pet pane cadence (see CLIStatusBarMixin).
    _PET_FRAME_INTERVAL = 0.16
    _PET_CFG_INTERVAL = 2.5

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
        
        config_path = _hermes_home / 'config.yaml'
        if not config_path.exists():
            config_path = Path(__file__).parent / 'cli-config.yaml'
        config_status = "(loaded)" if config_path.exists() else "(not found)"

        # ``self.api_key`` may be a callable (Azure Foundry Entra ID bearer
        # provider). Never invoke it; just identify the auth surface.
        from agent.azure_identity_adapter import is_token_provider

        # Prefer the LIVE agent's credential: the constructor seeds self.api_key
        # from OPENAI/OPENROUTER env vars before provider resolution, so on
        # non-OpenAI providers it is a different vendor's key than the one
        # actually authenticating (an sk-proj-... key next to a Nous base URL).
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
        base_cmd = cmd_lower.split()[0]
        skill_commands = _ensure_skill_commands()
        skill_bundles = get_skill_bundles()
        quick_commands = self.config.get("quick_commands", {})
        user_args = cmd_original[len(base_cmd):].strip()
        if base_cmd.lstrip("/") in quick_commands:
            return self._run_quick_command(base_cmd, quick_commands[base_cmd.lstrip("/")], user_args)
        if base_cmd.lstrip("/") in _get_plugin_cmd_handler_names():
            self._run_plugin_slash_command(base_cmd, user_args)
        elif base_cmd in skill_bundles:
            self._run_skill_bundle_command(base_cmd, skill_bundles[base_cmd], user_args)
        elif base_cmd in skill_commands:
            self._run_skill_slash_command(base_cmd, skill_commands[base_cmd], user_args)
        else:
            return self._expand_slash_prefix(cmd_original, cmd_lower, skill_commands, skill_bundles)
        return True

    def _run_quick_command(self, base_cmd: str, qcmd: dict, user_args: str) -> bool:
        """User-defined quick command (config.yaml): ``exec`` runs a shell snippet, ``alias`` re-dispatches."""
        qtype = qcmd.get("type")
        if qtype == "exec":
            import subprocess
            exec_cmd = qcmd.get("command", "")
            if not exec_cmd:
                self._console_print(f"[bold red]Quick command '{base_cmd}' has no command defined[/]")
                return True
            try:
                # shell=True is intentional: user-defined shell snippets from config.yaml,
                # never agent/LLM controlled. The env is sanitized because the CLI process
                # holds every API key in os.environ.
                from tools.environments.local import build_subprocess_env
                from hermes_cli._subprocess_compat import windows_hide_flags
                result = subprocess.run(
                    exec_cmd, shell=True, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=30, env=build_subprocess_env(),
                    # No console flash on Windows (#56747).
                    creationflags=windows_hide_flags(),
                )
                output = result.stdout.strip() or result.stderr.strip()
                if output:
                    from agent.redact import redact_sensitive_text
                    self._console_print(_rich_text_from_ansi(redact_sensitive_text(output)))
                else:
                    self._console_print("[dim]Command returned no output[/]")
            except subprocess.TimeoutExpired:
                self._console_print("[bold red]Quick command timed out (30s)[/]")
            except Exception as e:
                self._console_print(f"[bold red]Quick command error: {e}[/]")
        elif qtype == "alias":
            target = qcmd.get("target", "").strip()
            if target:
                target = target if target.startswith("/") else f"/{target}"
                return self.process_command(f"{target} {user_args}".strip())
            self._console_print(f"[bold red]Quick command '{base_cmd}' has no target defined[/]")
        else:
            self._console_print(f"[bold red]Quick command '{base_cmd}' has unsupported type (supported: 'exec', 'alias')[/]")
        return True

    def _run_plugin_slash_command(self, base_cmd: str, user_args: str) -> None:
        from hermes_cli.plugins import (
            get_plugin_command_handler,
            resolve_plugin_command_result,
        )
        plugin_handler = get_plugin_command_handler(base_cmd.lstrip("/"))
        if plugin_handler:
            try:
                result = resolve_plugin_command_result(plugin_handler(user_args))
                if result:
                    _cprint(str(result))
            except Exception as e:
                _cprint(f"\033[1;31mPlugin command error: {e}{_RST}")

    def _queue_skill_message(self, msg) -> None:
        if hasattr(self, '_pending_input'):
            self._pending_input.put(msg)

    def _run_skill_bundle_command(self, base_cmd: str, bundle_info: dict, user_instruction: str) -> None:
        """``/<bundle>`` loads several skills at once (bundles win over same-named skills)."""
        bundle_result = build_bundle_invocation_message(
            base_cmd, user_instruction, task_id=self.session_id
        )
        if not bundle_result:
            ChatConsole().print(f"[bold red]Failed to load bundle for {base_cmd}[/]")
            return
        msg, loaded_names, missing = bundle_result
        print(f"\n⚡ Loading bundle: {bundle_info['name']} ({len(loaded_names)} skills)")
        if missing:
            ChatConsole().print(f"[yellow]Skipped missing skills: {', '.join(missing)}[/]")
        self._queue_skill_message(msg)

    def _run_skill_slash_command(self, base_cmd: str, skill_info: dict, rest: str) -> None:
        """``/<skill> ...``; stacked ``/skill-a /skill-b do XYZ`` loads every leading skill (up to 5)."""
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
            if not stacked_result:
                ChatConsole().print(f"[bold red]Failed to load stacked skills for {base_cmd}[/]")
                return
            msg, loaded_names, missing = stacked_result
            print(f"\n⚡ Loading {len(loaded_names)} stacked skills: {', '.join(loaded_names)}")
            if missing:
                ChatConsole().print(f"[yellow]Skipped missing skills: {', '.join(missing)}[/]")
            self._queue_skill_message(msg)
            return
        msg = build_skill_invocation_message(base_cmd, rest, task_id=self.session_id)
        if msg:
            print(f"\n⚡ Loading skill: {skill_info['name']}")
            self._queue_skill_message(msg)
        else:
            ChatConsole().print(f"[bold red]Failed to load skill for {base_cmd}[/]")

    def _expand_slash_prefix(self, cmd_original: str, cmd_lower: str, skill_commands, skill_bundles) -> bool:
        """Unique-prefix expansion against built-in COMMANDS + skill commands/bundles (agrees with tab-completion)."""
        from hermes_cli.commands import COMMANDS
        typed_base = cmd_lower.split()[0]
        all_known = set(COMMANDS) | set(skill_commands) | set(skill_bundles)
        matches = [c for c in all_known if c.startswith(typed_base)]
        if len(matches) > 1:
            exact = [c for c in matches if c == typed_base]
            if len(exact) == 1:
                matches = exact
            else:
                # Unique shortest match wins: /qui → /quit (5) over /quint-pipeline (15)
                min_len = min(len(c) for c in matches)
                shortest = [c for c in matches if len(c) == min_len]
                if len(shortest) == 1:
                    matches = shortest
        if len(matches) == 1 and matches[0] != typed_base:
            # Expand to the full name, preserving arguments.
            return self.process_command(matches[0] + cmd_original.strip()[len(typed_base):])
        if len(matches) > 1:
            _cprint(f"{_ACCENT}Ambiguous command: {cmd_lower}{_RST}")
            _cprint(f"{_DIM}Did you mean: {', '.join(sorted(matches))}?{_RST}")
        else:
            # Exact token with no handler (never re-dispatch the same token: recursion), or no match.
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

    # Inline-skip tokens that bypass the destructive-slash confirmation modal.
    # A general escape hatch for non-interactive use (scripting/automation) and
    # for the degraded path where the modal can't be marshaled onto the app loop
    # — lets users self-serve without flipping approvals.destructive_slash_confirm
    # in config. (Native Windows now drives the modal normally — see #33961.)
    _DESTRUCTIVE_SKIP_TOKENS = frozenset({"now", "--yes", "-y"})

    def _tui_process_loop(self):
        """REPL worker thread: drain ``_pending_input``, run idle housekeeping, dispatch each input."""
        while not self._should_exit:
            try:
                try:
                    user_input = self._pending_input.get(timeout=0.1)
                except queue.Empty:
                    if not self._agent_running:
                        self._tui_idle_tick()
                    continue
                self._tui_process_one_input(user_input)
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
                logger.warning("process_loop unhandled error (msg may be lost): %s", e)

    def _tui_idle_tick(self):
        """Idle housekeeping between inputs (agent not running)."""
        # Auto-reload MCP on mcp_servers change.
        self._check_config_mcp_changes()
        # Heal cooked-mode termios drift (lost run_in_terminal restore) first —
        # a drifted tty makes the CLI look dead even though the loop is healthy.
        for step in (
            self._check_termios_drift,
            lambda: self._drain_process_notifications("cli-idle"),  # background completions / watch matches
            self._maybe_fire_loop_tick,  # due /loop wakeup (defers to queued input + /goal)
        ):
            try:
                step()
            except Exception:
                pass

    def _tui_unwrap_input(self, user_input):
        """Unwrap sentinel-wrapped inputs -> ``(text_or_tuple, is_voice_input, is_seeded_query)``.

        Voice-transcribed messages arrive in ``_VoiceInputMessage`` so only genuine
        STT output gets the voice prefix (#65827). Seeded -q prompts arrive in
        ``_SeededQueryMessage``: launcher/script text submitted LITERALLY — no slash
        routing, ! shell dispatch, or file-drop detection.
        """
        is_voice_input = isinstance(user_input, _VoiceInputMessage)
        if is_voice_input:
            user_input = user_input.text
        is_seeded_query = isinstance(user_input, _SeededQueryMessage)
        if is_seeded_query:
            seeded = user_input
            user_input = (seeded.text, seeded.images) if seeded.images else seeded.text
        return user_input, is_voice_input, is_seeded_query

    def _tui_process_one_input(self, user_input):
        """Route one submitted input: file drop, /resume pick, ! shell, slash command, or a chat turn."""
        user_input, is_voice_input, is_seeded_query = self._tui_unwrap_input(user_input)
        if not user_input:
            return
        # A submitted input ends any post-resize transient suppression.
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

        # A typed bare stop phrase ends an active voice chat (same as SAYING "stop").
        # Voice transcripts are already stop-checked at the transcription points.
        if not is_voice_input and self._typed_voice_stop(user_input):
            return

        # Dragged/pasted file paths are detected before any dispatch (see
        # _detect_file_drop). Seeded -q prompts are literal text: none of it.
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
        elif isinstance(user_input, str):
            # A bare number right after a bare `/resume` prompt selects that
            # session (#34584) — checked before chat routing so the digit is
            # never sent to the agent.
            if self._pending_resume_sessions and self._consume_pending_resume_selection(user_input):
                return
            if not is_seeded_query:
                # `!<command>` shell mode: nothing enters history, no model turn.
                if self.handle_bang_shell(user_input):
                    return
                if _looks_like_slash_command(user_input):
                    user_input = self._tui_run_slash_input(user_input)
                    if user_input is None:
                        return

        # Expand paste references back to full content
        _paste_ref_re = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')
        paste_refs = list(_paste_ref_re.finditer(user_input)) if isinstance(user_input, str) else []
        if paste_refs:
            user_input = self._expand_paste_references(user_input)
        print()
        self._print_user_message_preview(user_input)

        if submit_images:
            n = len(submit_images)
            _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

        self._agent_running = True
        self._interactive_turn = True
        self._pet_turn_error = False
        self._pet_reasoning = False
        self._turn_summary_begin()
        self._app.invalidate()  # Refresh status line
        try:
            self.chat(user_input, images=submit_images or None, voice_input=is_voice_input)
        finally:
            self._tui_after_turn()

    def _tui_run_slash_input(self, user_input: str):
        """Dispatch a slash command. Returns the pending agent seed to run as a chat turn, else None."""
        _cprint(f"\n⚙️  {user_input}")
        try:
            if not self.process_command(user_input):
                self._should_exit = True
                if self._app.is_running:
                    self._app.exit()
        except KeyboardInterrupt:
            # Ctrl+C during a slow slash command (/skills browse, /sessions list on
            # a large DB) returns to the prompt instead of killing the session.
            _cprint("\n[dim]Command interrupted.[/dim]")
            return None
        # A slash handler may set a one-shot seed (e.g. /blueprint <name>) to run
        # as the next agent turn.
        _seed = getattr(self, "_pending_agent_seed", None)
        if _seed:
            self._pending_agent_seed = None
        return _seed or None

    def _tui_after_turn(self):
        """Post-turn bookkeeping after chat() returns (normal, error, or interrupt)."""
        self._agent_running = False
        self._spinner_text = ""
        self._tool_start_time = 0.0
        self._pending_tool_info.clear()
        self._last_scrollback_tool = ""
        self._pet_reasoning = False
        self._pet_react_turn_end()
        # Post-turn accounting line (display.turn_summary) — a footer for the turn.
        self._turn_summary_emit()
        self._interactive_turn = False

        self._app.invalidate()  # Refresh status line

        # After an interrupt the prompt_toolkit renderer may have drifted from
        # the terminal: CSI 6n reports leak as literal text (^[[19;1R) and the
        # VT100 parser can stall in a partial-escape state (#33271). Drain stray
        # escape bytes and force a clean redraw.
        if self._last_turn_interrupted:
            self._recover_terminal_after_interrupt()

        # Re-queue messages that landed in _interrupt_queue during the turn and
        # were never claimed by the explicit interrupt path (#20271).
        self._drain_interrupt_queue_to_pending_input()

        # Goal continuation: if a standing goal is active and unmet, and no real
        # user message is queued, push the continuation prompt into
        # _pending_input (user input arriving in between still preempts).
        try:
            self._maybe_continue_goal_after_turn()
        except Exception as _goal_exc:
            logging.debug("goal continuation hook failed: %s", _goal_exc)

        # /loop tick completion: evaluate LOOP_COMPLETE / --until judge / caps
        # and schedule the next tick.
        try:
            self._maybe_complete_loop_tick_after_turn()
        except Exception as _loop_exc:
            logging.debug("loop completion hook failed: %s", _loop_exc)

        # Continuous voice: auto-restart recording after the agent responds.
        # Off-thread because play_beep (sd.wait) and AudioRecorder.start (lock
        # acquire) must never block process_loop.
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

        # Process notifications (completions + watch matches) that arrived mid-turn.
        try:
            self._drain_process_notifications("cli-post-turn")
        except Exception:
            pass  # Non-fatal — never break the main loop

    def _tui_signal_handler(self, signum, frame):
        """Handle SIGHUP/SIGTERM by triggering graceful cleanup.

        The agent is hard-interrupted first (see _interrupt_agent_for_signal) so
        its daemon thread can kill the tool's setsid subprocess group before the
        main thread unwinds — otherwise the child survives as an orphan (PPID=1).

        The ``logger.debug`` is guarded: CPython's logging is not reentrant-safe;
        ``Logger.isEnabledFor`` caches in ``Logger._cache``, which under shutdown
        races can be cleared or mid-mutation when the signal fires, raising
        ``KeyError: <level_int>`` inside the handler. That escapes before the
        KeyboardInterrupt, bypasses prompt_toolkit's interrupt unwind and surfaces
        as the EIO cascade of #13710.
        """
        try:
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
        except Exception:
            pass  # never let logging raise from a signal handler (#13710 regression)
        # Arm the exit backstop IMMEDIATELY: if the unwind below wedges (main
        # thread parked in a syscall, prompt_toolkit teardown never returning),
        # _run_cleanup never runs and would never arm its own watchdog, leaving
        # a "dead" CLI alive for minutes (#65998 class). Never raises.
        _arm_exit_watchdog_on_shutdown_signal()
        if getattr(self, "_agent_running", False):
            _interrupt_agent_for_signal(getattr(self, "agent", None), signum)
        # Prefer a clean prompt_toolkit exit over `raise KeyboardInterrupt()`: a
        # KBI raised from a signal handler lands in whatever frame is running —
        # typically `await asyncio.sleep()` in pt's `_poll_output_size` — becomes
        # a Task exception, pt prints "Unhandled exception in event loop" and
        # parks the terminal on "Press ENTER to continue..." (#13710 variant).
        # `app.exit()` via `call_soon_threadsafe` lets the loop unwind normally;
        # `app.run()` returns and run()'s except block handles the rest.
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

        _welcome_skin = None  # None when the skin engine failed: residue banner falls back to its default color
        try:
            from hermes_cli.skin_engine import get_active_skin
            _welcome_skin = get_active_skin()
            _welcome_text = _welcome_skin.get_branding("welcome", "Welcome to Hermes Agent! Type your message or /help for commands.")
            _welcome_color = _welcome_skin.get_color("banner_text", "#FFF8DC")
        except Exception:
            _welcome_text = "Welcome to Hermes Agent! Type your message or /help for commands."
            _welcome_color = "#FFF8DC"
        self._console_print(f"[{_welcome_color}]{_welcome_text}[/]")

        self._tui_startup_prewarm_and_warnings(_welcome_skin)
        self._print_random_tip()

        self._tui_startup_background_maintenance()
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

    def _tui_startup_prewarm_and_warnings(self, _welcome_skin):
        """Idle-window prewarms (picker cache, agent runtime imports) plus the redaction-off and OpenClaw-residue banners."""
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

    def _tui_startup_background_maintenance(self):
        """Best-effort startup passes: curator skill maintenance, personal + org skill sync. Never blocks startup."""
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

    def _tui_build_application(self, layout, kb, style):
        """Construct the prompt_toolkit Application for the REPL."""
        # CPR-disabled output when _terminal_may_leak_cpr() says so (POSIX local +
        # SSH; Windows keeps the PT default). None -> prompt_toolkit's default;
        # _strip_leaked_terminal_responses still scrubs residual leaks from input.
        _cpr_disabled_output = _select_classic_cli_pt_output(sys.stdout)

        # Kitty placeholders encode the image id in exact foreground RGB, so
        # placeholder-capable terminals (kitty/Ghostty) run the whole app in
        # 24-bit color — quantizing only that pane is not supported. ColorDepth is
        # imported lazily so tests that stub ``prompt_toolkit`` can still import cli.
        color_depth_kw = {}
        if pet_render.supports_kitty_placeholders():
            from prompt_toolkit.output import ColorDepth

            color_depth_kw = {"color_depth": ColorDepth.DEPTH_24_BIT}
        return Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            **({"output": _cpr_disabled_output} if _cpr_disabled_output is not None else {}),
            **color_depth_kw,
            # display.cli_refresh_interval (default 0 = disabled): non-zero keeps
            # wall-clock status-bar read-outs ticking during idle; 0 avoids fighting
            # terminal auto-scroll in non-fullscreen mode (Xshell, iTerm2, Windows
            # Terminal). See #48309.
            refresh_interval=float(CLI_CONFIG.get("display", {}).get("cli_refresh_interval", 0)),
            # Erase the live bottom chrome (status bar, input box, rules) on exit
            # instead of freezing a final copy into scrollback, where it would sit
            # between the transcript and the exit summary and stack with the next
            # session's UI on resume (#38252). The transcript itself goes through
            # patch_stdout into normal scrollback and is unaffected.
            erase_when_done=True,
            **({'cursor': _STEADY_CURSOR} if _STEADY_CURSOR is not None else {}),
        )

    def _tui_install_signal_handlers(self):
        """SIGTERM/SIGHUP -> graceful shutdown; Windows absorbs SIGINT (see body)."""
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, self._tui_signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, self._tui_signal_handler)

            # Windows: absorb SIGINT. Win32 delivers spurious CTRL_C_EVENT when
            # child processes are spawned from background threads; Python's
            # default handler would unwind app.run() and run _run_cleanup
            # mid-turn ("Daemon process exited during startup"). Real Ctrl+C
            # still works — prompt_toolkit binds c-c at the TUI layer and never
            # reaches this path. POSIX keeps the default handler (prompt_toolkit
            # installs its own). Do NOT call agent.interrupt() here: it would
            # inject a fake user message on every spurious event.
            if sys.platform == "win32":
                def _sigint_absorb(signum, frame):
                    return
                _signal.signal(_signal.SIGINT, _sigint_absorb)
        except Exception:
            pass  # Signal handlers may fail in restricted environments

    def _tui_stdin_usable(self) -> bool:
        """Validate fd 0 before prompt_toolkit starts; on macOS swap in a select()-backed loop if kqueue can't watch it.

        With uv-managed Python on macOS fd 0 can be invalid or unregisterable with
        the asyncio selector ("KeyError: '0 is not registered'", OSError(EINVAL) from
        kqueue.control() in loop.add_reader) — #6393.
        """
        try:
            os.fstat(0)
        except OSError:
            print(
                "Error: stdin (fd 0) is not available.\n"
                "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
            )
            return False
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
        return True

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        if not self._claim_active_session("cli"):
            return

        self._tui_print_startup()
        self._tui_init_run_state()
        kb = self._tui_build_key_bindings()
        layout, style = self._tui_build_layout(kb)

        app = self._tui_build_application(layout, kb, style)
        _disable_prompt_toolkit_cpr_warning(app)
        app.after_render += self._pet_flush_kitty_frame
        self._app = app  # Store reference for clarify_callback

        # ── Fix ghost status-bar lines on terminal resize ──────────────
        # prompt_toolkit's renderer moves the cursor to the canvas bottom after
        # painting so the terminal scrolls up; in non-fullscreen mode that pushes
        # chrome (status bar, input rules) into scrollback on every render, and a
        # column-shrink reflows the old full-width rows into ghost duplicates
        # (pt #29/#1675/#1933). Wrap _output_screen_diff so its
        # `current_height > previous_screen.height` branch never fires, by
        # inflating previous_screen.height first.
        try:
            import prompt_toolkit.renderer as _pt_renderer
            from prompt_toolkit.renderer import _output_screen_diff as _orig_osd

            if not getattr(_pt_renderer, "_hermes_osd_patched", False):
                # Same parameter names as pt's function, so pt's keyword call
                # site binds unchanged; the guards live in the helper's docstring.
                _pt_renderer._output_screen_diff = functools.partial(
                    _hermes_call_output_screen_diff, _orig_osd
                )
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
        
        self._tui_install_signal_handlers()

        if not self._tui_stdin_usable():
            _run_cleanup()
            self._print_exit_summary()
            return

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
        # prompt_toolkit just tore down the input box + status bar; without a
        # line here the terminal sits silent through the whole cleanup window
        # (session flush, memory shutdown, MCP/browser/terminal teardown).
        try:
            print(f"{_DIM}Shutting down… (finalizing session){_RST}", flush=True)
        except Exception:
            pass
        # Interrupt the agent now so its daemon thread stops making API calls
        # and run_conversation gets to clean up.
        if self.agent and getattr(self, '_agent_running', False):
            try:
                request_hard_interrupt(self.agent)
            except Exception:
                pass
        # Release the persistent audio stream.
        if hasattr(self, '_voice_recorder') and self._voice_recorder:
            try:
                self._voice_recorder.shutdown()
            except Exception:
                pass
            self._voice_recorder = None
        try:
            from tools.voice_mode import cleanup_temp_recordings
            cleanup_temp_recordings()
        except Exception:
            pass
        # Unregister callbacks to avoid dangling references
        set_sudo_password_callback(None)
        set_approval_callback(None)
        set_secret_capture_callback(None)
        # On SIGHUP/SIGTERM/window close the agent thread may not reach its normal
        # run_conversation() persistence before the daemon thread is reaped.
        self._persist_active_session_before_close()

        if hasattr(self, '_session_db') and self._session_db and self.agent:
            try:
                self._session_db.end_session(self.agent.session_id, "cli_close")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Could not close session in DB: %s", e)
            if not getattr(self, '_delete_session_on_exit', False):
                # Started-and-immediately-quit sessions never gained content; drop
                # the empty row so /resume and `hermes sessions list` stay clean
                # (gemini-cli#27770 port). No-op for resumed/titled sessions and
                # anything with messages or children.
                try:
                    self._discard_session_if_empty(self.agent.session_id)
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not prune empty session: %s", e)
            else:
                # /exit --delete: remove transcripts + SQLite history (gemini-cli#19332 port).
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
        # on_session_end safety net: run_conversation() fires it per turn on normal
        # completion, so only fire here when the exit happened mid-turn.
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


def _int_or(value, default: int) -> int:
    """``int(value)``, or ``default`` when it does not parse."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _interrupt_agent_for_signal(agent, signum) -> None:
    """Hard-interrupt ``agent`` for a shutdown signal, then sleep the grace window.

    The grace (``HERMES_SIGTERM_GRACE``, default 1.5 s) lets the agent daemon thread
    see the interrupt on its next poll and ``_kill_process`` the tool's setsid
    subprocess group before the main thread unwinds — otherwise the child survives
    as an orphan. ``time.sleep`` releases the GIL so the daemon actually runs.
    Never raises (signal handler context).
    """
    try:
        if agent is not None:
            request_hard_interrupt(agent, f"received signal {signum}")
            try:
                _grace = float(os.getenv("HERMES_SIGTERM_GRACE", "1.5"))
            except (TypeError, ValueError):
                _grace = 1.5
            if _grace > 0:
                time.sleep(_grace)
    except Exception:
        pass  # never block signal handling


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
    with _kb.connect_closing() as conn:
        task = _kb.get_task(conn, task_id)
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
        with _kb.connect_closing() as c:
            return _kb.goal_run_status(c, task_id, worker_run_id)

    def _block(reason: str) -> None:
        with _kb.connect_closing() as c:
            _kb.block_task(c, task_id, reason=reason, expected_run_id=worker_run_id)

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

        def _text_fallback():
            # ``_preprocess_images_with_vision`` only knows local files; when
            # only URLs were supplied keep the original query text intact.
            if single_query_images:
                return cli._preprocess_images_with_vision(query, single_query_images, announce=False)
            return effective_query

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
                    effective_query = _text_fallback()  # all images unreadable
            except Exception:
                effective_query = _text_fallback()
        else:
            effective_query = _text_fallback()
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

            with _kb.connect_closing() as _conn:
                _task = _kb.get_task(_conn, _kanban_task_id)
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
    # Single-query (`-q`) signal handling. Interactive mode registers its own in
    # HermesCLI.run(); here AIAgent's tool worker threads would outlive a plain
    # KeyboardInterrupt (only the main thread unwinds) and orphan the setsid child
    # subprocess. So route SIGTERM/SIGHUP through agent.interrupt() (the worker
    # poll loop checks it every 200 ms), give it a grace window to _kill_process,
    # then raise KeyboardInterrupt. HERMES_SIGTERM_GRACE overrides the 1.5 s default.
    def _signal_handler_q(signum, frame):
        logger.debug("Received signal %s in single-query mode", signum)
        # Arm the exit backstop now that shutdown intent is unambiguous —
        # covers wedges in the unwind below that would otherwise leave the
        # process alive with no watchdog (#65998 class). Never raises.
        _arm_exit_watchdog_on_shutdown_signal()
        _interrupt_agent_for_signal(getattr(cli, "agent", None), signum)
        # Kanban worker (#28181): a non-daemon worker thread blocked in
        # _wait_for_process survives KeyboardInterrupt, so the PID stays alive and
        # the dispatcher's _pid_alive sees 'running' forever. os._exit(0) instead,
        # so detect_crashed_workers reclaims the claim next tick. Flush logging +
        # stdio first (SIGALRM deadman guards a rare blocking flush).
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


def _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills):
    """Resolve the toolset list (explicit / coding posture / platform default), construct HermesCLI, and start the background skills preload."""
    toolsets_list = None
    if isinstance(toolsets, str) and toolsets:
        toolsets_list = [t.strip() for t in toolsets.split(",")]
    elif isinstance(toolsets, (list, tuple)) and toolsets:
        # Fire may pass multiple --toolsets as a tuple
        toolsets_list = []
        for t in toolsets:
            if isinstance(t, str):
                toolsets_list.extend([x.strip() for x in t.split(",")])
            else:
                toolsets_list.append(str(t))
    elif not toolsets:
        # Coding posture: with no explicit --toolsets, collapse to the coding
        # toolset (+ enabled MCP servers) inside a code workspace
        # (agent/coding_context.py); otherwise the shared platform resolver so
        # MCP servers are included at runtime.
        try:
            from agent.coding_context import coding_selection
            toolsets_list = coding_selection(platform="cli", config=CLI_CONFIG)
        except Exception:
            toolsets_list = None
        if toolsets_list is None:
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
    return cli


def _run_legacy_gateway():
    """Legacy `cli.py --gateway` entry: arm the startup watchdog, then run the gateway event loop."""
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


def _start_worktree_setup(list_tools, list_toolsets, worktree, w):
    """Kick off isolated-worktree creation (+ tool prewarm) in the background.

    Returns the ``_join_worktree`` callable that waits for the setup, publishes
    ``_active_worktree``/TERMINAL_CWD and schedules stale-worktree GC — or None when
    no worktree is wanted (list commands exit immediately, or -w not requested).
    """
    if list_tools or list_toolsets:
        return None
    # Git worktree isolation (#652): this agent instance must not collide with
    # other agents working on the same repo.
    if not (worktree or w or CLI_CONFIG.get("worktree", False)):
        return None
    # Overlap tool discovery with the network/subprocess-bound worktree setup
    # (both release the GIL for most of their wall time) so show_banner() hits
    # the warm cache instead of paying ~0.4s serially. Only on the -w path: on
    # plain `hermes` there is no I/O wait to hide and the thread just contends.
    def _prewarm_tools() -> None:
        try:
            import model_tools as _mt
            _mt.get_tool_definitions(quiet_mode=True)
        except Exception:
            logger.debug("tool prewarm failed", exc_info=True)

    threading.Thread(
        target=_prewarm_tools, name="tool-prewarm", daemon=True
    ).start()
    # Worktree creation (~0.2-0.6s of git wall time) runs concurrently with the
    # rest of startup; joined right after HermesCLI construction, before anything
    # consumes TERMINAL_CWD / wt_info. Setup failure still aborts the session.
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
            # Prune stale worktrees from crashed sessions in the background —
            # pure GC. Ordered AFTER _setup_worktree so the two never race on
            # git's worktrees metadata; the new tree is immune to reaping
            # (<24h age gate + live pid lock).
            _repo = _git_repo_root()
            if _repo:
                def _worktree_maintenance(repo: str) -> None:
                    _prune_stale_worktrees(repo)
                    # Repack when packs sprawl so object lookups (and the next
                    # `worktree add`) stay fast on multi-agent boxes; after the
                    # pruner so the repack sees final refs.
                    _maintain_pack_health(repo)

                threading.Thread(
                    target=_worktree_maintenance,
                    args=(_repo,),
                    name="worktree-prune",
                    daemon=True,
                ).start()
        return info

    return _join_worktree


def _configure_quiet_agent(agent) -> None:
    """Neutralize every stdout-writing callback so -Q stdout carries only the final response (#93220)."""
    agent.quiet_mode = True
    agent.suppress_status_output = True
    # No styled "Hermes" box, tool-gen status lines, or reasoning box.
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    agent.reasoning_callback = None
    # Inline-diff and progress callbacks print directly to stdout and are gated
    # by NEITHER quiet_mode nor tool_progress_mode (_on_tool_complete renders
    # full file diffs; _on_tool_progress prints MoA reference blocks before its
    # mode check), so they must go too.
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    # Belt-and-braces for the executor's direct prints (they check
    # agent.tool_progress_mode, initialized from display.tool_progress).
    agent.tool_progress_mode = "off"


def _run_single_query_mode(cli, query, image, quiet, oneshot):
    """``-q``/``--image`` entry: seed an interactive session on a TTY, else run the one-shot turn and exit."""
    # NEW DEFAULT (Aug 2026): on a real TTY, -q/--image seeds a normal interactive
    # session with the prompt as the first turn, submitted LITERALLY (no slash/!
    # dispatch). Answer-and-exit is kept for --oneshot, -Q, and every non-TTY
    # invocation (kanban/cron/pipes) — see _should_seed_interactive().
    if _should_seed_interactive(query, image, quiet, oneshot):
        seeded_query, seeded_images = _collect_query_images(query, image)
        logger.info(
            "Seeding interactive session with -q prompt (%d chars, %d images)",
            len(seeded_query or ""), len(seeded_images),
        )
        cli._seeded_first_message = _SeededQueryMessage(seeded_query, seeded_images)
        cli.run()
        return
    # One-shot: no between-turns MCP late-binding refresh, so the agent waits the
    # full MCP cold-start bound before its only tool snapshot (#51316).
    cli._single_query_mode = True
    # A -q run has NO user to answer approval prompts: the approval gate reads
    # this marker (gateway.session_context.get_session_env, falling back to
    # os.environ) and takes the deterministic approvals.single_query_mode path
    # instead of waiting the full timeout (#86878).
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
                    _configure_quiet_agent(cli.agent)
                    _run_quiet_single_query(cli, effective_query)

            # Exit with error code if credentials or agent init fails
            sys.exit(1)
        # Single-query (`-q`): skip the welcome banner (~420 ms cold — version
        # check + toolset/skill enumeration + Rich render). The session id /
        # resume hint come from _print_exit_summary().
        _query_label = query or ("[image attached]" if single_query_images else "")
        if _query_label:
            cli.console.print(f"[bold blue]Query:[/] {_query_label}")
        cli._show_security_advisories()
        cli.chat(query, images=single_query_images or None)
        cli._print_exit_summary(clear_screen=False)
    finally:
        _finalize_single_query(cli)


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
        _run_legacy_gateway()
        return

    _join_worktree = _start_worktree_setup(list_tools, list_toolsets, worktree, w)
    wt_info = None
    
    # Handle query shorthand
    query = query or q
    
    cli = _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                               verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills)

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
    
    if query or image:
        _run_single_query_mode(cli, query, image, quiet, oneshot)
        return

    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
