"""Interactive setup wizard for Hermes Agent (config lives in ~/.hermes/).

Independently-runnable sections: Model & Provider, Terminal Backend, Agent Settings,
Messaging Platforms, Tools (TTS, web search, image generation, ...). Section bodies live in
sibling modules (setup_tts, setup_terminal, setup_platforms, setup_summary, setup_migration,
setup_quick) and are re-exported here; they resolve shared prompt/config helpers lazily through
this module so test patches on ``hermes_cli.setup.<name>`` keep working.
"""

import importlib.util
import logging
import os
import re
import sys
import copy
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

from hermes_cli.curses_ui import MenuNavigationEvent, MenuNavigationStart
from hermes_cli.nous_subscription import get_nous_subscription_features  # noqa: F401  (re-export; patched by tests)
from tools.tool_backend_helpers import managed_nous_tools_enabled  # noqa: F401  (re-export; patched by tests)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_DOCS_BASE = "https://hermes-agent.nousresearch.com/docs"


# Config helpers (re-exported; tests patch them on this module). display_hermes_home is
# imported lazily at call sites (stale-module safety during hermes update).
from hermes_cli.config import (  # noqa: E402
    cfg_get, DEFAULT_CONFIG, get_hermes_home, get_config_path, get_env_path, load_config, save_config,
    save_env_value, remove_env_value, get_env_value, ensure_hermes_home,
)
from hermes_cli.colors import Colors, color  # noqa: E402


def print_header(title: str):
    """Print a section header."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


from hermes_cli.cli_output import print_error, print_info, print_success, print_warning  # noqa: E402
from hermes_cli.secret_prompt import masked_secret_prompt  # noqa: E402


def _info(*lines: str | None) -> None:
    """print_info each line in order; ``None`` emits a bare blank ``print()``."""
    for line in lines:
        print() if line is None else print_info(line)


def _current_reasoning_effort(config: dict) -> str:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    return ""


def _set_reasoning_effort(config: dict, effort: str) -> None:
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        config["agent"] = agent_cfg
    agent_cfg["reasoning_effort"] = effort


def is_interactive_stdin() -> bool:
    """Return True when stdin looks like a usable interactive TTY."""
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return False
    try:
        return bool(stdin.isatty())
    except Exception:
        return False


def print_noninteractive_setup_guidance(reason: str | None = None) -> None:
    """Print guidance for headless/non-interactive setup flows."""
    print()
    print(color("⚕ Hermes Setup — Non-interactive mode", Colors.CYAN, Colors.BOLD))
    print()
    if reason:
        print_info(reason)
    _info("The interactive wizard cannot be used here.", None,
          "Configure Hermes using environment variables or config commands:",
          "  hermes config set model.provider custom",
          "  hermes config set model.base_url http://localhost:8080/v1",
          "  hermes config set model.default your-model-name", None,
          "Or set OPENROUTER_API_KEY / OPENAI_API_KEY in your environment.",
          "Run 'hermes setup' in an interactive terminal to use the full wizard.", None)


_BRACKETED_PASTE_PATTERN = re.compile(r"\x1b\[\s*200~|\x1b\[\s*201~")


def _sanitize_pasted_input(value: str) -> str:
    """Strip terminal bracketed-paste control markers from pasted text."""
    if not isinstance(value, str) or not value:
        return value
    return _BRACKETED_PASTE_PATTERN.sub("", value)


def prompt(question: str, default: str = None, password: bool = False) -> str:
    """Prompt for input with optional default."""
    display = f"{question} [{default}]: " if default else f"{question}: "

    try:
        if password:
            value = masked_secret_prompt(color(display, Colors.YELLOW))
        else:
            from hermes_cli.cli_output import line_input

            value = line_input(color(display, Colors.YELLOW))

        cleaned = _sanitize_pasted_input(value)
        return cleaned.strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


# =============================================================================
# Setup navigation (Escape cancels, Left arrow goes back) — a ContextVar state
# machine shared with the curses menus.
# =============================================================================


class _SetupControlFlow(BaseException):
    """Bypass provider error handlers that intentionally catch ``Exception``.

    Provider setup has broad compatibility boundaries around network, plugin and credential
    integrations; navigation must cross them unchanged so the outer state machine can replay
    the prior prompt.
    """


class _SetupCancelled(_SetupControlFlow):
    """Internal control flow for cancelling the interactive setup wizard."""


class _SetupGoBack(_SetupControlFlow):
    """Internal control flow for returning to an earlier setup choice."""

    def __init__(self, prompt_index: int):
        super().__init__(prompt_index)
        self.prompt_index = prompt_index


class _SetupNavigationState:
    """Per-invocation navigation state for the synchronous setup wizard."""

    def __init__(self, *, section_index: int = -1, prompt_index: int = 0):
        self.section_index = section_index
        self.prompt_index = prompt_index
        self.active_prompt_index = -1
        self.resolved_choices: list[object] = []
        self.replay_choices: list[object] = []

    def reset(self, section_index: int = -1, replay: list | None = None) -> None:
        """Rewind per-section counters (entering a section, or leaving the wizard)."""
        self.section_index = section_index
        self.prompt_index = 0
        self.active_prompt_index = -1
        self.resolved_choices = []
        self.replay_choices = copy.deepcopy(replay or [])


_SETUP_NAVIGATION: ContextVar[_SetupNavigationState | None] = ContextVar(
    "hermes_setup_navigation", default=None
)


def _handle_setup_menu_navigation(
    event: MenuNavigationEvent, value: object = None
) -> MenuNavigationStart | None:
    """Translate shared curses menu events into setup control flow."""
    state = _SETUP_NAVIGATION.get()
    if state is None:
        return None
    if event is MenuNavigationEvent.BEGIN:
        if state.section_index < 0:
            state.active_prompt_index = -1
            return MenuNavigationStart()
        state.active_prompt_index = state.prompt_index
        state.prompt_index += 1
        allow_back = state.section_index > 0 or state.active_prompt_index > 0
        if state.active_prompt_index < len(state.replay_choices):
            return MenuNavigationStart(
                allow_back=allow_back,
                replay_value=copy.deepcopy(state.replay_choices[state.active_prompt_index]),
            )
        return MenuNavigationStart(allow_back=allow_back)
    if event is MenuNavigationEvent.RESOLVE:
        prompt_index = state.active_prompt_index
        if prompt_index < 0:
            return None
        resolved = copy.deepcopy(value)
        if prompt_index < len(state.resolved_choices):
            state.resolved_choices[prompt_index] = resolved
            del state.resolved_choices[prompt_index + 1 :]
        else:
            state.resolved_choices.append(resolved)
        return None
    if event is MenuNavigationEvent.CANCEL:
        raise _SetupCancelled()
    if event is MenuNavigationEvent.BACK:
        raise _SetupGoBack(state.active_prompt_index)
    return None


@contextmanager
def _setup_navigation_scope():
    """Install and reliably restore the setup menu navigation context."""
    from hermes_cli.curses_ui import reset_menu_navigation_handler, set_menu_navigation_handler
    token = _SETUP_NAVIGATION.set(_SetupNavigationState())
    menu_token = set_menu_navigation_handler(_handle_setup_menu_navigation)
    try:
        yield
    finally:
        reset_menu_navigation_handler(menu_token)
        _SETUP_NAVIGATION.reset(token)


def _run_setup_steps(steps: list[tuple[str, Callable[[], None]]]) -> None:
    """Run setup sections with left-arrow navigation between choices.

    Left arrow at a section's first choice returns to the previous section; from a later, nested
    choice it replays earlier selections invisibly and reopens only the preceding prompt.
    """
    state = _SETUP_NAVIGATION.get()
    section_index = 0
    answers_by_section: dict[int, list[object]] = {}
    replay_by_section: dict[int, list[object]] = {}
    try:
        while section_index < len(steps):
            label, action = steps[section_index]
            if state is not None:
                state.reset(section_index, replay_by_section.pop(section_index, []))
            try:
                action()
            except _SetupGoBack as navigation:
                if state is not None:
                    answers_by_section[section_index] = copy.deepcopy(state.resolved_choices)
                if navigation.prompt_index > 0:
                    previous_index = section_index
                    target_prompt = navigation.prompt_index - 1
                else:
                    previous_index = max(0, section_index - 1)
                    target_prompt = max(0, len(answers_by_section.get(previous_index, [])) - 1)
                replay_by_section[previous_index] = copy.deepcopy(
                    answers_by_section.get(previous_index, [])[:target_prompt]
                )
                print()
                if previous_index == section_index:
                    print_info(f"Returning to the previous choice in {label}...")
                else:
                    print_info(f"Returning to {steps[previous_index][0]}...")
                section_index = previous_index
                continue
            if state is not None:
                answers_by_section[section_index] = copy.deepcopy(state.resolved_choices)
            section_index += 1
    finally:
        if state is not None:
            state.reset()


def run_setup_action_with_navigation(
    label: str, action: Callable[[], None], *, cancelled_message: str = "Setup cancelled."
) -> None:
    """Run a setup-style menu flow with Escape and nested Left navigation.

    Shared commands such as ``hermes model`` use the wizard's pickers outside ``run_setup_wizard``;
    this installs the navigation context for them and reuses the prompt replay state machine.
    """
    with _setup_navigation_scope():
        try:
            _run_setup_steps([(label, action)])
        except _SetupCancelled:
            print()
            print_info(cancelled_message)


# ── Prompt primitives ──


def _curses_prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Single-select menu using curses. Delegates to curses_radiolist."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=-1, description=description)


def prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Prompt for a choice from a list with arrow key navigation.

    Escape cancels an active setup wizard; outside setup it keeps the default. The curses
    component owns its own numbered fallback, so a cancel result must never be mistaken for a
    request to open another prompt. Ctrl+C exits the wizard.
    """
    idx = _curses_prompt_choice(question, choices, default, description=description)
    if idx >= 0:
        if idx == default:
            print_info("  Skipped (keeping current)")
            print()
            return default
        print()
        return idx

    return default


def is_noninteractive() -> bool:
    """True when no human is available to answer a prompt.

    The dashboard/desktop spawn CLI actions with ``stdin=DEVNULL`` and ``HERMES_NONINTERACTIVE=1``
    (see ``hermes_cli/web_server.py``); there ``input()`` raises ``EOFError`` immediately, and a
    prompt that aborts on EOF kills the spawned action (desktop "restart gateway" failed this way
    when the Windows service was not installed yet). Honour the flag so callers fall back to their
    default.
    """
    return os.environ.get("HERMES_NONINTERACTIVE", "").strip().lower() in {"1", "true", "yes", "on"}


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no. Ctrl+C exits, empty input returns default.

    Non-interactive callers (``HERMES_NONINTERACTIVE=1`` or a closed/redirected stdin) have no
    one to answer, so fall back to ``default`` instead of aborting the whole process.
    """
    if is_noninteractive():
        return default

    # Inside setup, route binary selections through the curses menu so ESC and left-arrow work
    # consistently; every other caller keeps the traditional line prompt.
    if _SETUP_NAVIGATION.get() is not None:
        return _curses_prompt_choice(question, ["Yes", "No"], 0 if default else 1) == 0

    default_str = "Y/n" if default else "y/N"

    while True:
        try:
            value = input(color(f"{question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
        except KeyboardInterrupt:
            print()
            sys.exit(1)
        except EOFError:
            # No stdin (closed/redirected, e.g. stdin=DEVNULL): accept the default so the caller
            # proceeds unattended instead of failing the whole command.
            print()
            return default

        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print_error("Please enter 'y' or 'n'")


def prompt_checklist(title: str, items: list, pre_selected: list = None) -> list:
    """Multi-select checklist; returns the sorted indices of selected items.

    ``pre_selected`` indices start checked; Space toggles, Enter on the appended "Continue →"
    confirms, cancel keeps the pre-selection. Numbered fallback when curses is unavailable.
    """
    if pre_selected is None:
        pre_selected = []

    from hermes_cli.curses_ui import curses_checklist

    chosen = curses_checklist(title, items, set(pre_selected), cancel_returns=set(pre_selected))
    return sorted(chosen)


def _prompt_api_key(var: dict):
    """Display a nicely formatted API key input screen for a single env var."""
    tools = var.get("tools", [])
    tools_str = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_str += f", +{len(tools) - 3} more"

    print()
    print(color(f"  ─── {var.get('description', var['name'])} ───", Colors.CYAN))
    print()
    if tools_str:
        print_info(f"  Enables: {tools_str}")
    if var.get("url"):
        print_info(f"  Get your key at: {var['url']}")
    print()

    value = prompt(f"  {var.get('prompt', var['name'])}", password=bool(var.get("password")))

    if value:
        save_env_value(var["name"], value)
        print_success("  ✓ Saved")
    else:
        print_warning("  Skipped (configure later with 'hermes setup')")


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _print_banner(*lines: str) -> None:
    """Print the magenta box banner: top border, the given body lines, bottom border."""
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    for line in lines:
        print(color(line, Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))


# Tool categories and provider config are in tools_config.py (shared
# between `hermes tools` and `hermes setup tools`).


# ── Section 1: Model & Provider Configuration ──


def setup_model_provider(config: dict, *, quick: bool = False):
    """Configure the inference provider and default model.

    Delegates to the ``hermes model`` flow (provider picker, credential prompting, model pick,
    persistence) so there is one code path — any provider added there is available here.
    *quick* skips credential rotation, vision and TTS (first-time quick setup).
    """
    from hermes_cli.config import load_config, save_config
    print_header("Inference Provider")
    print_info("Choose how to connect to your main chat model.")
    print_info(f"   Guide: {_DOCS_BASE}/integrations/providers")
    print()

    from hermes_cli.main import select_provider_and_model
    try:
        select_provider_and_model()
    except (SystemExit, KeyboardInterrupt):
        print()
        print_info("Provider setup skipped.")
    except Exception as exc:
        logger.debug("select_provider_and_model error during setup: %s", exc)
        print_warning(f"Provider setup encountered an error: {exc}")
        print_info("You can try again later with: hermes model")

    # Re-sync from disk in place: cmd_model saved via its own load/save cycle and the wizard's
    # final save_config(config) must not clobber it with stale values (#4172). Rotation, vision
    # and TTS keep safe defaults (configure via `hermes auth add` / `hermes setup tts`).
    config.clear()
    config.update(load_config())
    save_config(config)


# ── Section 3: Agent Settings ──


def _apply_default_agent_settings(config: dict):
    """Apply recommended defaults for all agent settings without prompting."""
    config.setdefault("agent", {})["max_turns"] = 150
    # config.yaml is authoritative for max_turns (the gateway bridges it into HERMES_MAX_ITERATIONS);
    # a stale .env entry silently shadowing it caused the 60-vs-500 bug, so drop it.
    remove_env_value("HERMES_MAX_ITERATIONS")
    config.setdefault("display", {})["tool_progress"] = "all"
    config.setdefault("compression", {})["enabled"] = True
    config["compression"]["threshold"] = 0.50
    # Never auto-reset (the gateway default); written explicitly so it is visible in config.yaml.
    config.setdefault("session_reset", {})["mode"] = "none"

    save_config(config)
    print_success("Applied recommended defaults:")
    _info("  Max iterations: 150", "  Tool progress: all", "  Compression threshold: 0.50",
          "  Session reset: never (use /reset or compression)",
          "  Run `hermes setup agent` later to customize.")


def _prompt_int_setting(section: dict, key: str, label: str, current, accept) -> None:
    """Prompt for an int; store it under *key* only when it parses and *accept* holds."""
    try:
        value = int(prompt(label, str(current)))
        if accept(value):
            section[key] = value
    except ValueError:
        pass


_TOOL_PROGRESS_HELP = (
    "Tool Progress Display",
    "Controls how much tool activity is shown (CLI and messaging).",
    "  off     — Silent, just the final response",
    "  new     — Show tool name only when it changes (less noise)",
    "  all     — Show every tool call with a short preview",
    "  verbose — Full args, results, and debug logs",
    "  log     — Silent in chat; write every tool call to ~/.hermes/logs/tool_calls.log (gateway only)",
)
_SESSION_RESET_HELP = (
    "Messaging sessions (Telegram, Discord, etc.) accumulate context over time.",
    "Each message adds to the conversation history, which means growing API costs.",
    "",
    "To manage this, sessions can automatically reset after a period of inactivity",
    "or at a fixed time each day. When a reset happens, the agent saves important",
    "things to its persistent memory first — but the conversation context is cleared.",
    "",
    "You can also manually reset anytime by typing /reset in chat.",
    "",
)
_SESSION_RESET_CHOICES = [
    "Inactivity + daily reset (reset whichever comes first)",
    "Inactivity only (reset after N minutes of no messages)",
    "Daily only (reset at a fixed hour each day)",
    "Never auto-reset (recommended - context lives until /reset or context compression)",
    "Keep current settings",
]
_SESSION_RESET_MODES = ("both", "idle", "daily", "none")  # index 4 = keep current


def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display, compression, session reset."""

    print_header("Agent Settings")
    print_info(f"   Guide: {_DOCS_BASE}/user-guide/configuration")
    print()

    # ── Max Iterations ── (config.yaml is authoritative; never surface a stale legacy .env value)
    current_max = str(cfg_get(config, "agent", "max_turns", default=90))
    print_info("Maximum tool-calling iterations per conversation.")
    print_info("Higher = more complex tasks, but costs more tokens.")
    print_info(f"Press Enter to keep {current_max}. Use 90 for most tasks or 150+ for open exploration.")

    try:
        max_iter = int(prompt("Max iterations", current_max))
        if max_iter > 0:
            # config.yaml only; gateway/run.py derives HERMES_MAX_ITERATIONS from agent.max_turns.
            config.setdefault("agent", {})["max_turns"] = max_iter
            config.pop("max_turns", None)
            remove_env_value("HERMES_MAX_ITERATIONS")
            print_success(f"Max iterations set to {max_iter}")
    except ValueError:
        print_warning("Invalid number, keeping current value")

    # ── Tool Progress Display ──
    print_info("")
    for line in _TOOL_PROGRESS_HELP:
        print_info(line)

    current_mode = cfg_get(config, "display", "tool_progress", default="all")
    mode = prompt("Tool progress mode", current_mode)
    if mode.lower() in {"off", "new", "all", "verbose", "log"}:
        config.setdefault("display", {})["tool_progress"] = mode.lower()
        save_config(config)
        print_success(f"Tool progress set to: {mode.lower()}")
    else:
        print_warning(f"Unknown mode '{mode}', keeping '{current_mode}'")

    # ── Context Compression ──
    print_header("Context Compression")
    print_info("Automatically summarizes old messages when context gets too long.")
    print_info("Higher threshold = compress later (use more context). Lower = compress sooner.")

    config.setdefault("compression", {})["enabled"] = True

    current_threshold = cfg_get(config, "compression", "threshold", default=0.50)
    try:
        threshold = float(prompt("Compression threshold (0.5-0.95)", str(current_threshold)))
        if 0.5 <= threshold <= 0.95:
            config["compression"]["threshold"] = threshold
    except ValueError:
        pass

    print_success(f"Context compression threshold set to {config['compression'].get('threshold', 0.50)}")

    # ── Session Reset Policy ──
    print_header("Session Reset Policy")
    for line in _SESSION_RESET_HELP:
        print_info(line)

    current_policy = config.get("session_reset", {})
    current_mode = current_policy.get("mode", "none")
    current_idle = current_policy.get("idle_minutes", 1440)
    current_hour = current_policy.get("at_hour", 4)

    default_reset = _SESSION_RESET_MODES.index(current_mode) if current_mode in _SESSION_RESET_MODES else 3
    reset_idx = prompt_choice("Session reset mode:", _SESSION_RESET_CHOICES, default_reset)

    reset_cfg = config.setdefault("session_reset", {})
    mode = _SESSION_RESET_MODES[reset_idx] if 0 <= reset_idx < len(_SESSION_RESET_MODES) else None
    if mode is not None:
        reset_cfg["mode"] = mode
    if mode in ("both", "idle"):
        _prompt_int_setting(reset_cfg, "idle_minutes", "  Inactivity timeout (minutes)", current_idle, lambda v: v > 0)
    if mode in ("both", "daily"):
        _prompt_int_setting(reset_cfg, "at_hour", "  Daily reset hour (0-23, local time)", current_hour, lambda v: 0 <= v <= 23)
    idle_now, hour_now = reset_cfg.get("idle_minutes", 1440), reset_cfg.get("at_hour", 4)
    if mode == "both":
        print_success(f"Sessions reset after {idle_now} min idle or daily at {hour_now}:00")
    elif mode == "idle":
        print_success(f"Sessions reset after {idle_now} min of inactivity")
    elif mode == "daily":
        print_success(f"Sessions reset daily at {hour_now}:00")
    elif mode == "none":
        print_info("Sessions will never auto-reset. Context is managed only by compression.")
        print_warning("Long conversations will grow in cost. Use /reset manually when needed.")

    save_config(config)


# ── Section 5: Tool Configuration (delegates to unified tools_config.py) ──


def setup_tools(config: dict, first_install: bool = False):
    """`hermes setup tools` == `hermes tools`: platform selection → toolset toggles → provider keys.
    ``first_install`` selects the simplified flow (no platform menu, prompts for all missing keys)."""
    from hermes_cli.tools_config import tools_command
    tools_command(first_install=first_install, config=config)


# ── Shared Metrics ──


_SEND_CONSENT_EXPLAINER = (
    "",
    "Sending uploads each daily package to the Nous telemetry",
    "service. Packages carry your profile-scoped install ID, a",
    "stable random UUID that identifies this profile across days",
    "(it contains no personal information and is reset by deleting",
    "the shared-metrics directory). Only packages whose entire",
    "collection period falls inside a recorded consent window are",
    "ever sent — data from before you opt in, or from any gap",
    "while sending was off, stays on this machine. Sending can be",
    "turned off again at any time.",
)


def setup_telemetry(config: dict):
    """Configure the local shared-metrics subscriber and optional sending."""
    print_header("Shared Metrics")
    print_info("Shared metrics contain only bounded counters and histograms.")
    print_info("Collection is local. Sending them to Nous is a separate opt-in.")

    telemetry = config.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
        config["telemetry"] = telemetry
    shared_metrics = telemetry.get("shared_metrics")
    if not isinstance(shared_metrics, dict):
        shared_metrics = {}
        telemetry["shared_metrics"] = shared_metrics

    current = shared_metrics.get("enabled") is True
    shared_metrics["enabled"] = prompt_yes_no("Enable local shared metrics?", default=current)
    if not shared_metrics["enabled"]:
        print_info("Local shared metrics disabled.")
        # Sending cannot outlive collection (send=true would log an error every run, never send).
        if shared_metrics.get("send") is True:
            shared_metrics["send"] = False
            print_info("Sending shared metrics disabled as well.")
        # Turning collection off withdraws send consent too. Recorded unconditionally: the send
        # key may already be false while the consent window is still open, and it must close.
        _record_send_consent_change(enabled=False)
        return

    print_success("Local shared metrics enabled.")
    for line in _SEND_CONSENT_EXPLAINER:
        print_info(line)
    shared_metrics["send"] = prompt_yes_no("Send shared metrics to Nous?", default=shared_metrics.get("send") is True)
    _record_send_consent_change(enabled=shared_metrics["send"])
    if shared_metrics["send"]:
        print_success("Sending shared metrics enabled.")
    else:
        print_info("Sending shared metrics disabled (collection stays local).")


def _record_send_consent_change(*, enabled: bool) -> None:
    """Reconcile consent windows at the moment the user decides.

    Same single writer as the relay and the sender, so wizard, relay and mid-pass callers cannot
    disagree; the relay would reconcile on its next hook anyway — this makes the effect immediate.
    """
    try:
        from hermes_cli.observability.shared_metrics import SharedMetricsStore
        from hermes_cli.observability.shared_metrics_sender import reconcile_send_consent
        from hermes_cli.sqlite_util import write_txn

        store = SharedMetricsStore()
        with store._connection() as connection:
            with write_txn(connection):
                reconcile_send_consent(connection, enabled)
    except Exception:
        # Never block the wizard on telemetry bookkeeping; the relay reconciles on the next hook.
        logger.debug("Unable to record shared-metrics consent change", exc_info=True)


# Extracted sections, re-exported so callers and test patches keep resolving through
# hermes_cli.setup. They import this module lazily inside bodies, so this is cycle-free.

from hermes_cli.setup_tts import (  # noqa: E402,F401
    _run_xai_oauth_login_from_setup, _setup_tts_provider, _xai_oauth_logged_in_for_setup, setup_tts,
)
from hermes_cli.setup_terminal import (  # noqa: E402,F401
    _prompt_vercel_sandbox_settings, _read_nearest_vercel_project, setup_terminal_backend,
)
from hermes_cli.setup_platforms import (  # noqa: E402,F401
    _TELEGRAM_BOT_TOKEN_RE, _profile_name_from_hermes_home, _setup_bluebubbles, _setup_telegram,
    _setup_telegram_auto_result, _setup_webhooks, setup_gateway,
)
from hermes_cli.setup_summary import _print_setup_summary  # noqa: E402,F401
from hermes_cli.setup_migration import (  # noqa: E402,F401
    _OPENCLAW_SCRIPT, _get_section_config_summary, _load_openclaw_migration_module,
    _model_section_has_credentials, _offer_openclaw_migration, _print_migration_preview,
    _skip_configured_section,
)
from hermes_cli.setup_quick import (  # noqa: E402,F401
    _blank_slate_minimal_toolsets, _blank_slate_minimize_config, _blank_slate_walkthrough,
    _print_macos_fda_tip, _run_blank_slate_setup, _run_first_time_quick_setup, _run_portal_one_shot,
    _run_quick_setup,
)


# ── Main Wizard Orchestrator ──

SETUP_SECTIONS = [
    ("model", "Model & Provider", setup_model_provider),
    ("tts", "Text-to-Speech", setup_tts),
    ("terminal", "Terminal Backend", setup_terminal_backend),
    ("gateway", "Messaging Platforms (Gateway)", setup_gateway),
    ("tools", "Tools", setup_tools),
    ("telemetry", "Shared Metrics", setup_telemetry),
    ("agent", "Agent Settings", setup_agent_settings),
]


def run_setup_wizard(args):
    """Run setup with navigation control scoped to this invocation."""
    with _setup_navigation_scope():
        try:
            return _run_setup_wizard_impl(args)
        except _SetupCancelled:
            print()
            print_info("Setup cancelled. Remaining sections were not changed.")
            return None


def _backup_config_file(config_path: Path) -> Path | None:
    """Back up config.yaml before setup modifies it (#3522); None when absent or copy fails."""
    if not config_path.exists():
        return None
    from datetime import datetime as _dt
    backup_path = config_path.with_suffix(f".yaml.bak.{_dt.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        import shutil
        shutil.copy2(config_path, backup_path)
    except Exception:
        return None
    return backup_path


def _run_setup_section(config: dict, section: str) -> None:
    """``hermes setup <section>``: run one SETUP_SECTIONS entry under the banner."""
    for key, label, func in SETUP_SECTIONS:
        if key == section:
            _print_banner(f"│     ⚕ Hermes Setup — {label:<34s} │")
            _run_setup_steps([(label, lambda setup_func=func: setup_func(config))])
            save_config(config)
            print()
            print_success(f"{label} configuration complete!")
            return

    print_error(f"Unknown setup section: {section}")
    print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")


def _run_full_setup(config: dict, hermes_home, *, is_existing: bool, migration_ran: bool) -> None:
    """Full Setup — run all sections, honoring post-migration skips."""
    print_header("Configuration Location")
    print_info(f"Config file:  {get_config_path()}")
    print_info(f"Secrets file: {get_env_path()}")
    print_info(f"Data folder:  {hermes_home}")
    print_info(f"Install dir:  {PROJECT_ROOT}")
    print()
    print_info("You can edit these files directly or use 'hermes config edit'")

    if migration_ran:
        _info(None, "Settings were imported from OpenClaw.",
              "Each section below will show what was imported — press Enter to keep,",
              "or choose to reconfigure if needed.")

    # Agent Settings are not prompted: first installs get defaults, existing keep theirs.
    if not is_existing:
        _apply_default_agent_settings(config)

    def _skip(key: str, label: str) -> bool:
        return migration_ran and _skip_configured_section(config, key, label)

    def _model_step() -> None:
        if not _skip("model", "Model & Provider"):
            setup_model_provider(config)

    def _terminal_step() -> None:
        if not _skip("terminal", "Terminal Backend"):
            setup_terminal_backend(config)

    def _gateway_step() -> None:
        if not _skip("gateway", "Messaging Platforms"):
            setup_gateway(config)
            return

        # A skipped (migrated) gateway section still needs its service so imported platforms
        # and cron jobs become active.
        from hermes_cli.gateway import ensure_gateway_service

        ensure_gateway_service(context="setup")

    def _tools_step() -> None:
        if not _skip("tools", "Tools"):
            setup_tools(config, first_install=not is_existing)

    _run_setup_steps(
        [
            ("Model & Provider", _model_step),
            ("Terminal Backend", _terminal_step),
            ("Messaging Platforms", _gateway_step),
            ("Tools", _tools_step),
        ]
    )


def _run_setup_wizard_impl(args):
    """Run the interactive setup wizard: full/quick (auto-detected), ``--portal``, or one
    ``hermes setup <section>`` from SETUP_SECTIONS."""
    from hermes_cli.config import is_managed, managed_error
    if is_managed():
        managed_error("run setup wizard")
        return
    ensure_hermes_home()

    if bool(getattr(args, "reset", False)):
        save_config(copy.deepcopy(DEFAULT_CONFIG))
        print_success("Configuration reset to defaults.")

    reconfigure_requested = bool(getattr(args, "reconfigure", False))
    quick_requested = bool(getattr(args, "quick", False))

    config = load_config()
    hermes_home = get_hermes_home()

    config_path = get_config_path()
    _backup_path = _backup_config_file(config_path)

    # Detect non-interactive environments (headless SSH, Docker, CI/CD)
    if getattr(args, 'non_interactive', False) or not is_interactive_stdin():
        print_noninteractive_setup_guidance("Running in a non-interactive environment (no TTY detected).")
        return

    # --portal: one-shot Nous Portal setup. Skips the rest of the wizard.
    if bool(getattr(args, "portal", False)):
        _run_portal_one_shot(config)
        return

    section = getattr(args, "section", None)
    if section:
        _run_setup_section(config, section)
        return

    # Check if this is an existing installation with a provider configured
    from hermes_cli.auth import get_active_provider

    is_existing = (
        bool(get_env_value("OPENROUTER_API_KEY"))
        or bool(get_env_value("OPENAI_BASE_URL"))
        or get_active_provider() is not None
    )

    _print_banner(
        "│             ⚕ Hermes Agent Setup Wizard                │",
        "├─────────────────────────────────────────────────────────┤",
        "│  Let's configure your Hermes Agent installation.       │",
        "│  Press Ctrl+C at any time to exit.                     │",
    )

    migration_ran = False

    if is_existing:
        # Existing install — the full reconfigure wizard is the default (Enter keeps each current
        # value); `--quick` narrows it to missing items (partial OpenClaw import, cleared key).
        if quick_requested:
            _run_setup_steps([("Quick Setup", lambda: _run_quick_setup(config, hermes_home))])
            return

        print()
        print_header("Reconfigure")
        print_success("You already have Hermes configured.")
        _info("Running the full wizard — each prompt shows your current value.",
              "Press Enter to keep it, or type a new value to change it.", "",
              "Tip: jump straight to a section with 'hermes setup model|terminal|",
              "     gateway|tools|agent', or fill only missing items with --quick.")
        # --reconfigure is kept for backwards compatibility and is a no-op here.
    else:
        # ── First-Time Setup ── (--reconfigure / --quick are meaningless here; fall through)
        print()
        if reconfigure_requested or quick_requested:
            print_info("No existing configuration found — running first-time setup.")
            print()

        # Offer OpenClaw migration before configuration begins
        migration_ran = _offer_openclaw_migration(hermes_home)
        if migration_ran:
            config = load_config()

        setup_mode = prompt_choice(
            "How would you like to set up Hermes?",
            [
                "Quick Setup (Nous Portal) — free OAuth login, no API keys, model + tools (recommended)",
                "Full setup — configure every provider, tool & option yourself (bring your own keys)",
                "Blank Slate — everything off except the bare minimum; opt in to each capability",
            ],
            0,
        )

        if setup_mode == 0:
            _run_setup_steps(
                [("Quick Setup", lambda: _run_first_time_quick_setup(config, hermes_home, is_existing))]
            )
            return
        if setup_mode == 2:
            _run_setup_steps(
                [("Blank Slate", lambda: _run_blank_slate_setup(config, hermes_home, is_existing))]
            )
            return

    _run_full_setup(config, hermes_home, is_existing=is_existing, migration_ran=migration_ran)

    # Save and show summary
    save_config(config)
    if _backup_path and _backup_path.exists():
        print_info(f"Previous config backed up to: {_backup_path}")
        print_info("If setup changed a value you customized, restore it with:")
        print_info(f"  cp {_backup_path} {config_path}")
    _print_setup_summary(config, hermes_home)
