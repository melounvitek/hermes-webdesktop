"""Slash-command handlers for the interactive CLI, lifted out of ``cli.py``'s ``HermesCLI``.

``HermesCLI`` inherits ``CLICommandsMixin`` so every ``self.<handler>`` resolves via the MRO.
cli.py-internal symbols (``_cprint``/``_ACCENT``/``save_config_value``…) are imported LAZILY
inside each handler via ``from cli import ...`` — the mixin never imports ``cli`` at top level
(import cycle).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

from rich import box as rich_box
from rich.markup import escape as _escape
from rich.panel import Panel

from hermes_constants import display_hermes_home, is_termux as _is_termux_environment
from agent.turn_context import extract_api_content_sidecar
from hermes_cli.browser_connect import (
    DEFAULT_BROWSER_CDP_URL,
    discover_local_cdp_url,
    find_free_debug_port,
    is_browser_debug_ready,
    launch_chrome_debug,
    local_port_in_use,
    manual_chrome_debug_command,
)


def _print_lightpanda_engine_status() -> None:
    """``/browser status`` line(s) about ``browser.engine: lightpanda`` — silent unless set;
    says whether it is in use or which higher-precedence setting shadows it."""
    try:
        from tools.browser_tool import lightpanda_engine_status, _using_lightpanda_engine

        if not _using_lightpanda_engine():
            return
        used, reason = lightpanda_engine_status()
    except Exception:
        return
    if not used:
        print(f"   ⚠ browser.engine is 'lightpanda' but it is NOT in use: {reason}")
        return
    print(f"   Engine: Lightpanda — {reason} (no screenshots)")
    try:
        from tools.browser_lightpanda import LIGHTPANDA_INSTALL_HINT, find_lightpanda_binary

        lightpanda_bin = find_lightpanda_binary()
    except Exception:
        return
    if lightpanda_bin:
        print(f"   Binary: {lightpanda_bin}")
    else:
        print(f"   ⚠ lightpanda binary not found — {LIGHTPANDA_INSTALL_HINT}")


# /cron flag tables: flag -> opts key. Order-sensitive in _parse_flags: bool flags never consume
# a value; --repeat is int-validated separately.
_CRON_BOOL_FLAGS = {"--clear-skills": "clear_skills", "--all": "all"}
_CRON_LIST_FLAGS = {"--skill": "skills", "--add-skill": "add_skills", "--remove-skill": "remove_skills"}
_CRON_VALUE_FLAGS = {"--name": "name", "--deliver": "deliver", "--prompt": "prompt", "--schedule": "schedule"}

_ON_WORDS = {"on", "enable", "true", "1"}
_OFF_WORDS = {"off", "disable", "false", "0"}

# /busy mode -> what Enter does while Hermes is working (status line / post-set explanation).
_BUSY_MODE_SHORT = {
    "queue": "queues for next turn",
    "steer": "steers into current run (after next tool call)",
    "interrupt": "redirects current run immediately",
}
_BUSY_MODE_LONG = {
    "queue": "Enter will queue follow-up input while Hermes is busy.",
    "steer": "Enter will steer your message into the current run (after the next tool call).",
    "interrupt": "Enter will redirect the current run while Hermes is busy; /stop still cancels it.",
}


# /fast argument -> (service_tier value, persisted config value)
_FAST_TIERS = {
    "fast": ("priority", "fast"), "on": ("priority", "fast"),
    "normal": (None, "normal"), "off": (None, "normal"),
    "auto": ("auto", "auto"), "cold": ("cold", "cold"),
}


def _split_scope_flags(raw: str):
    """``(arg, explicit_global)`` for /reasoning + /fast: session scope by default, ``--global``
    persists to config.yaml, ``--session`` is an explicit no-op (parity with /model)."""
    tokens = raw.strip().lower().split()
    return " ".join(t for t in tokens if t not in ("--global", "--session")), "--global" in tokens


def _scope_outcome(explicit_global: bool, saved: bool) -> str:
    """Parenthetical tail for a scoped setting change."""
    if saved:
        return "(saved to config)"
    if explicit_global:
        return "(session only; config save failed)"
    return "(this session — use --global to persist)"


def _toggle_target(arg: str, current: bool):
    """Resolve a ``/x [on|off|status]`` argument: "status" for a status query, a bool for the
    new state (bare arg toggles), or None when the argument is unrecognized."""
    if arg in {"status", "?"}:
        return "status"
    if arg in _ON_WORDS:
        return True
    if arg in _OFF_WORDS:
        return False
    if arg == "":
        return not current
    return None


def _command_arg(cmd: str, *, lower: bool = False) -> str:
    """Everything after the slash-command word, stripped (optionally lowercased)."""
    parts = (cmd or "").strip().split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


def _summarize_paths(paths, limit: int = 5) -> str:
    """``a, b, c (+N more)`` for a list of paths."""
    more = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def _cron_api(**kwargs) -> dict:
    """Call the cronjob model tool and decode its JSON reply."""
    from tools.cronjob_tools import cronjob as cronjob_tool
    return json.loads(cronjob_tool(**kwargs))


def _normalize_skills(values) -> list:
    """Strip, drop empties, and dedupe (order-preserving)."""
    normalized = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _parse_cron_flags(tokens):
    """Parse /cron flags into an opts dict (None after printing an error for a bad --repeat)."""
    opts = {
        "name": None, "deliver": None, "repeat": None, "prompt": None, "schedule": None,
        "skills": [], "add_skills": [], "remove_skills": [],
        "clear_skills": False, "all": False, "positionals": [],
    }
    i = 0
    while i < len(tokens):
        token = tokens[i]
        has_value = i + 1 < len(tokens)
        if token in _CRON_BOOL_FLAGS:
            opts[_CRON_BOOL_FLAGS[token]] = True
            i += 1
        elif token in _CRON_LIST_FLAGS and has_value:
            opts[_CRON_LIST_FLAGS[token]].append(tokens[i + 1])
            i += 2
        elif token == "--repeat" and has_value:
            try:
                opts["repeat"] = int(tokens[i + 1])
            except ValueError:
                print("(._.) --repeat must be an integer")
                return None
            i += 2
        elif token in _CRON_VALUE_FLAGS and has_value:
            opts[_CRON_VALUE_FLAGS[token]] = tokens[i + 1]
            i += 2
        else:
            opts["positionals"].append(token)
            i += 1
    return opts


# /cron subcommand -> CLICommandsMixin handler (bound late, after the class body).
_CRON_SUBCOMMANDS: dict = {}


def _say_block(*lines: str) -> None:
    """print() the lines framed by a blank line above and below (the /browser output style)."""
    print()
    for line in lines:
        print(line)
    print()


def _end_current_session(cli, reason: str) -> None:
    """Flush un-persisted messages, then end the current session row with ``reason``.
    Best-effort on both steps (the switch proceeds even if the DB write fails)."""
    if cli.agent:
        try:
            cli.agent._flush_messages_to_session_db(
                cli.conversation_history, conversation_history=cli.conversation_history,
            )
        except Exception:
            pass
    try:
        cli._session_db.end_session(cli.session_id, reason)
    except Exception:
        pass


def _sync_agent_to_session(cli, session_id: str, *, parent_session_id: str, reason: str) -> None:
    """Point an already-built agent at ``session_id`` after a /resume or /branch switch.

    Resets per-session agent state, re-anchors the DB flush index to the loaded history,
    and notifies memory providers with reset=False — their accumulated state stays valid
    and just targets the new session_id (the parent link keeps the lineage)."""
    if not cli.agent:
        return
    cli.agent.session_id = session_id
    cli.agent.reset_session_state()
    if hasattr(cli.agent, "_last_flushed_db_idx"):
        cli.agent._last_flushed_db_idx = len(cli.conversation_history)
    if hasattr(cli.agent, "_todo_store"):
        try:
            from tools.todo_tool import TodoStore
            cli.agent._todo_store = TodoStore()
        except Exception:
            pass
    if hasattr(cli.agent, "_invalidate_system_prompt"):
        cli.agent._invalidate_system_prompt()
    try:
        _mm = getattr(cli.agent, "_memory_manager", None)
        if _mm is not None:
            _mm.on_session_switch(
                session_id, parent_session_id=parent_session_id or "", reset=False, reason=reason,
            )
    except Exception:
        pass


def _print_side_result_panel(cli, *, header_lines, body, title_suffix, empty_note) -> None:
    """Print a worker-thread result (/bg, /btw) into the scrollback: accent rules around
    ``header_lines``, then ``body`` in a skinned Rich panel (or ``empty_note``).
    Forces a TUI refresh first so the spinner/status bar don't overlap the output."""
    from cli import ChatConsole, _accent_hex, _cprint, _maybe_remap_for_light_mode, _render_final_assistant_content
    if cli._app:
        cli._app.invalidate()
        time.sleep(0.05)  # brief pause for refresh
    print()
    ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
    for line in header_lines:
        _cprint(line)
    ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
    if not body:
        _cprint(empty_note)
        return
    try:
        from hermes_cli.skin_engine import get_active_skin
        _skin = get_active_skin()
        label = _skin.get_branding("response_label", "⚕ Hermes")
        _resp_color = _maybe_remap_for_light_mode(_skin.get_color("response_border", "#CD7F32"))
        _resp_text = _maybe_remap_for_light_mode(_skin.get_color("banner_text", "#FFF8DC"))
    except Exception:
        label, _resp_color, _resp_text = "⚕ Hermes", "#CD7F32", "#FFF8DC"
    ChatConsole().print(Panel(
        _render_final_assistant_content(body, mode=cli.final_response_markdown),
        title=f"[{_resp_color} bold]{label} {title_suffix}[/]",
        title_align="left",
        border_style=_resp_color,
        style=_resp_text,
        box=rich_box.HORIZONTALS,
        padding=(1, 4),
        width=cli._scrollback_box_width(),
    ))


def _browser_use(cli, arg: str) -> None:
    """/browser use [off] — toggle Browser Use mode (browser.backend); resets the session."""
    from hermes_cli.config import load_config, save_config
    from tools.registry import invalidate_check_fn_cache

    if arg not in {"on", "off"}:
        _say_block(
            "Usage: /browser use [off]",
            "   /browser use       — switch to Browser Use mode (browser_exec via CLI 3.0)",
            "   /browser use off   — revert to the built-in browser tools",
        )
        return
    config = load_config()
    if arg == "on":
        config.setdefault("browser", {})["backend"] = "browser-use"
        headline = "🌐 Browser Use mode enabled — browser_exec via the Browser Use CLI 3.0"
    else:
        from tools.browser_use_cli import BACKEND_DISABLED
        config.setdefault("browser", {})["backend"] = BACKEND_DISABLED
        headline = "🌐 Browser Use mode disabled — built-in browser tools restored"
    save_config(config)
    invalidate_check_fn_cache()
    cli.new_session()
    _say_block(headline, "   Session reset. New tool configuration is active.")

def _browser_connect(cli, cdp_url: str) -> None:
    """/browser connect [url] — validate the CDP URL, find or launch a debug browser, then
    point the browser tools at it (BROWSER_CDP_URL) and tell the model."""
    import platform as _plat

    parsed_cdp = urlparse(cdp_url if "://" in cdp_url else f"http://{cdp_url}")
    if parsed_cdp.scheme not in {"http", "https", "ws", "wss"}:
        _say_block(
            f"   ⚠ Unsupported browser url scheme: {parsed_cdp.scheme or '(missing)'} "
            "(expected one of: http, https, ws, wss)"
        )
        return
    try:
        _port = parsed_cdp.port or (443 if parsed_cdp.scheme in {"https", "wss"} else 80)
    except ValueError:
        _say_block(f"   ⚠ Invalid port in browser url: {cdp_url}")
        return
    if not parsed_cdp.hostname:
        _say_block(f"   ⚠ Missing host in browser url: {cdp_url}")
        return
    if parsed_cdp.path.startswith("/devtools/browser/"):
        cdp_url = parsed_cdp.geturl()
    else:
        cdp_url = parsed_cdp._replace(path="", params="", query="", fragment="").geturl()

    # Clear any existing browser sessions so the next tool call uses the new backend
    try:
        from tools.browser_tool import cleanup_all_browsers
        cleanup_all_browsers()
    except Exception:
        pass

    print()

    # Already serving CDP? For the default-local URL probe both loopbacks: a squatter
    # on 127.0.0.1:<port> (e.g. an IDE debugger) can push the browser to bind [::1] only.
    _is_default = cdp_url == DEFAULT_BROWSER_CDP_URL
    if _is_default:
        _found = discover_local_cdp_url(_port, timeout=1.0)
        _already_open = _found is not None
        if _found:
            cdp_url = _found
    else:
        _already_open = is_browser_debug_ready(cdp_url, timeout=1.0)

    if _already_open:
        print(f"   ✓ Chromium-family browser is already listening at {cdp_url}")
    elif _is_default:
        _launch_port = _port
        if local_port_in_use(_port):
            _launch_port = find_free_debug_port(_port)
            print(f"   ⚠ Port {_port} is occupied by another application that isn't a CDP browser")
            print(f"     (an IDE debugger or dev server may be using it) — launching on port {_launch_port} instead...")
        else:
            print("   Chromium-family browser isn't running with remote debugging — attempting to launch...")
        _launch = launch_chrome_debug(_launch_port, _plat.system())
        if _launch.launched:
            for _wait in range(10):  # wait for the DevTools discovery endpoint to come up
                _found = discover_local_cdp_url(_launch_port, timeout=1.0)
                if _found:
                    cdp_url = _found
                    _already_open = True
                    break
                time.sleep(0.5)
            if _already_open:
                print(f"   ✓ Chromium-family browser launched and listening on port {_launch_port}")
            else:
                print(f"   ⚠ Browser launched but port {_launch_port} isn't responding yet")
                print("     Try again in a few seconds — the debug instance may still be starting")
        else:
            print("   ⚠ Could not auto-launch a Chromium-family browser")
            if _launch.hint:
                print(f"     {_launch.hint}")
            chrome_cmd = manual_chrome_debug_command(_launch_port, _plat.system())
            if chrome_cmd:
                print("     Launch a Chromium-family browser manually:")
                print(f"     {chrome_cmd}")
            else:
                print("     No supported Chromium-family browser executable found in this environment")
    else:
        print(f"   ⚠ Port {_port} is not reachable at {cdp_url}")

    if not _already_open:
        _say_block("Browser not connected — start a Chromium-family browser with remote debugging and retry /browser connect")
        return

    os.environ["BROWSER_CDP_URL"] = cdp_url
    # Eagerly start the CDP supervisor so pending_dialogs + frame_tree show up in the next snapshot.
    try:
        from tools.browser_tool import _ensure_cdp_supervisor  # type: ignore[import-not-found]
        _ensure_cdp_supervisor("default")
    except Exception:
        pass
    _say_block("🌐 Browser connected to live Chromium-family browser via CDP", f"   Endpoint: {cdp_url}")

    # Tell the model the CDP browser was made available on purpose.
    if hasattr(cli, '_pending_input'):
        cli._pending_input.put(
            "[System note: The user invoked /browser connect and connected your browser tools to "
            "a Chromium-family dev/debug browser via Chrome DevTools Protocol. "
            "Your browser_navigate, browser_snapshot, browser_click, and other browser tools now "
            "control that CDP browser. The command itself is a signal that using browser tools for "
            "their current browser-related request is expected; do not wait for separate permission "
            "just because CDP is connected. This is typically a Hermes-managed isolated debug "
            "profile, not the user's main everyday browser. It is still user-visible and may contain "
            "pages, logged-in sessions, or cookies in that debug profile, so avoid destructive actions, "
            "closing tabs, or navigating away unless the user's task calls for it.]"
        )

def _browser_disconnect(cli) -> None:
    if not os.environ.get("BROWSER_CDP_URL", "").strip():
        _say_block("Browser is not connected to a live Chromium-family browser (already using default mode)")
        return
    os.environ.pop("BROWSER_CDP_URL", None)
    try:
        from tools.browser_tool import cleanup_all_browsers, _stop_cdp_supervisor
        _stop_cdp_supervisor("default")
        cleanup_all_browsers()
    except Exception:
        pass
    _say_block(
        "🌐 Browser disconnected from live Chromium-family browser",
        "   Browser tools reverted to default mode (local headless or cloud provider)",
    )
    if hasattr(cli, '_pending_input'):
        cli._pending_input.put(
            "[System note: The user has disconnected the browser tools from their live Chromium-family browser. "
            "Browser tools are back to default mode (headless local browser or cloud provider).]"
        )

def _browser_status() -> None:
    current = os.environ.get("BROWSER_CDP_URL", "").strip()
    print()
    try:
        from tools.browser_use_cli import is_browser_use_cli_mode
        _bu_mode = is_browser_use_cli_mode()
    except Exception:
        _bu_mode = False
    if _bu_mode:
        print("🌐 Browser: Browser Use mode (browser_exec via the Browser Use CLI 3.0)")
        print("   Local Chrome via CDP, or Browser Use cloud browsers")
        _print_lightpanda_engine_status()
        _say_block("   /browser use off      — revert to the built-in browser tools")
        return
    if current:
        print("🌐 Browser: connected to live Chromium-family browser via CDP")
        print(f"   Endpoint: {current}")
        _print_lightpanda_engine_status()
        _port = 9222
        try:
            _port = int(current.rsplit(":", 1)[-1].split("/")[0])
        except (ValueError, IndexError):
            pass
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(("127.0.0.1", _port))
            s.close()
            print("   Status: ✓ reachable")
        except (OSError, Exception):
            print("   Status: ⚠ not reachable (browser may not be running)")
    else:
        try:
            from tools.browser_tool import _get_cloud_provider
            provider = _get_cloud_provider()
        except Exception:
            provider = None
        if provider is not None:
            print(f"🌐 Browser: {provider.provider_name()} (cloud)")
            _print_lightpanda_engine_status()
        else:
            try:
                from tools.browser_tool import _get_browser_engine
                engine = _get_browser_engine()
            except Exception:
                engine = "auto"
            if engine == "lightpanda":
                print("🌐 Browser: local Lightpanda (agent-browser --engine lightpanda)")
                print("   ⚡ Lightpanda: faster navigation, no screenshot support")
                print("   Automatic Chromium fallback for screenshots and failed commands")
                _print_lightpanda_engine_status()
            elif engine == "chrome":
                print("🌐 Browser: local headless Chromium (agent-browser --engine chrome)")
            else:
                print("🌐 Browser: local headless Chromium (agent-browser)")
    _say_block(
        "   /browser connect      — connect to your live Chromium-family browser",
        "   /browser disconnect   — revert to default",
    )


class CLICommandsMixin:
    """Mixin holding the interactive-CLI slash-command handlers."""

    def _handle_rollback_command(self, command: str):
        """Handle /rollback — list, diff, or restore filesystem checkpoints.

        Syntax:
            /rollback                 — list checkpoints
            /rollback <N>             — restore checkpoint N, preserving user hand-edits
                                        (also undoes last chat turn)
            /rollback <N> --all       — classic full restore (may overwrite your later edits)
            /rollback diff <N>        — preview changes since checkpoint N
            /rollback <N> <file>      — restore a single file from checkpoint N"""
        from tools.checkpoint_manager import format_checkpoint_list

        if not hasattr(self, 'agent') or not self.agent:
            print("  No active agent session.")
            return

        mgr = self.agent._checkpoint_mgr
        if not mgr.enabled:
            print("  Checkpoints are not enabled.")
            print("  Enable with: hermes --checkpoints")
            print("  Or in config.yaml: checkpoints: { enabled: true }")
            return

        cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        parts = command.split()
        args = parts[1:] if len(parts) > 1 else []

        # --all / --force: classic full restore, overwriting user edits too.
        restore_all = False
        filtered = []
        for a in args:
            if a.lower() in ("--all", "--force"):
                restore_all = True
            else:
                filtered.append(a)
        args = filtered

        if not args:
            # List checkpoints — fall back to the cross-project view when the current
            # directory has none (writes may have landed under the session cwd, not TERMINAL_CWD).
            checkpoints = mgr.list_checkpoints(cwd)
            if not checkpoints:
                all_checkpoints = mgr.list_all_checkpoints()
                if all_checkpoints:
                    print(f"  No checkpoints for {cwd} — showing all directories.")
                    print(format_checkpoint_list(all_checkpoints, "all directories"))
                    return
            print(format_checkpoint_list(checkpoints, cwd))
            return

        # Handle /rollback diff <N>
        if args[0].lower() == "diff":
            if len(args) < 2:
                print("  Usage: /rollback diff <N>")
                return
            checkpoints = mgr.list_checkpoints(cwd)
            if not checkpoints:
                print(f"  No checkpoints found for {cwd}")
                return
            target_hash = self._resolve_checkpoint_ref(args[1], checkpoints)
            if not target_hash:
                return
            result = mgr.diff(cwd, target_hash)
            if result["success"]:
                stat = result.get("stat", "")
                diff = result.get("diff", "")
                if not stat and not diff:
                    print("  No changes since this checkpoint.")
                else:
                    if stat:
                        print(f"\n{stat}")
                    if diff:
                        # Limit diff output to avoid terminal flood
                        diff_lines = diff.splitlines()
                        if len(diff_lines) > 80:
                            print("\n".join(diff_lines[:80]))
                            print(f"\n  ... ({len(diff_lines) - 80} more lines, showing first 80)")
                        else:
                            print(f"\n{diff}")
            else:
                print(f"  ❌ {result['error']}")
            return

        # Resolve checkpoint reference (number or hash)
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            print(f"  No checkpoints found for {cwd}")
            return

        target_hash = self._resolve_checkpoint_ref(args[0], checkpoints)
        if not target_hash:
            return

        # Check for file-level restore: /rollback <N> <file>
        file_path = args[1] if len(args) > 1 else None

        result = mgr.restore(
            cwd, target_hash, file_path=file_path,
            safe=not restore_all and not file_path,
        )
        if result["success"]:
            if file_path:
                print(f"  ✅ Restored {file_path} from checkpoint {result['restored_to']}: {result['reason']}")
            else:
                print(f"  ✅ Restored to checkpoint {result['restored_to']}: {result['reason']}")
            skipped = result.get("skipped_user_edits") or []
            if skipped:
                print(f"  ↷ Kept your hand-edits: {_summarize_paths(skipped)}")
                print("  Use /rollback <N> --all to restore those too.")
            oversize = result.get("skipped_oversize") or []
            if oversize:
                print(f"  ↷ Kept (too large for checkpoints, no stored copy to revert to): {_summarize_paths(oversize)}")
            failed = result.get("failed_deletes") or []
            if failed:
                print(f"  ⚠️ Could not remove (left in place): {_summarize_paths(failed)}")
            print("  A pre-rollback snapshot was saved automatically.")

            # Also undo the last conversation turn so the agent's context
            # matches the restored filesystem state
            if self.conversation_history:
                self.undo_last(prefill=False)
                print("  Chat turn undone to match restored file state.")
        else:
            print(f"  ❌ {result['error']}")

    def _handle_diff_command(self, command: str):
        """Handle /diff — show git changes in the working directory.

        Syntax: ``/diff`` (unstaged + untracked), ``staged``, ``all`` (vs HEAD), ``session``
        (everything Hermes changed since the checkpoint baseline), ``[mode] --stat``, ``[mode] <path...>``."""
        import shlex

        try:
            parts = shlex.split(command)[1:]  # preserves quoted paths
        except ValueError:
            parts = command.split()[1:]

        stat_only = False
        mode = "working"
        paths: list[str] = []
        for arg in parts:
            low = arg.lower()
            if low in ("--stat", "stat"):
                stat_only = True
            elif low in ("staged", "--staged", "cached", "--cached"):
                mode = "staged"
            elif low in ("all", "--all", "head"):
                mode = "all"
            elif low == "session":
                mode = "session"
            else:
                paths.append(arg)

        cwd = os.getenv("TERMINAL_CWD", os.getcwd())

        if mode == "session":
            self._print_session_diff(cwd, stat_only)
            return

        from tools.working_diff import collect_working_diff

        result = collect_working_diff(cwd, mode=mode, paths=paths or None)
        if not result.get("success"):
            print(f"  {result.get('error', 'Could not generate diff')}")
            return

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        untracked = result.get("untracked", [])
        if result.get("empty") or (not stat and not diff and not untracked):
            print("  No changes.")
            return

        label = {"working": "Unstaged", "staged": "Staged", "all": "All (vs HEAD)"}[mode]
        if stat:
            print(f"\n  {label}:")
            self._print_diff_text(stat)
        if untracked and mode in ("working", "all"):
            print("\n  Untracked:")
            for rel in untracked[:20]:
                print(f"    + {rel}")
            if len(untracked) > 20:
                print(f"    ... and {len(untracked) - 20} more")
        if stat_only or not diff:
            return
        self._print_diff_body(diff, "run /diff --stat for a summary")

    def _print_diff_body(self, diff: str, stat_hint: str, limit: int = 400) -> None:
        """Print a diff, capped at ``limit`` lines with a pointer to the --stat form."""
        diff_lines = diff.splitlines()
        print("")
        if len(diff_lines) > limit:
            self._print_diff_text("\n".join(diff_lines[:limit]))
            print(f"\n  ... ({len(diff_lines) - limit} more lines — {stat_hint})")
        else:
            self._print_diff_text(diff)

    def _print_session_diff(self, cwd: str, stat_only: bool):
        """Print the cumulative checkpoint-baseline diff (/diff session)."""
        if not hasattr(self, 'agent') or not self.agent:
            print("  No active agent session.")
            return

        mgr = self.agent._checkpoint_mgr
        if not mgr.enabled:
            print("  Checkpoints are not enabled, so there's no session baseline.")
            print("  Enable with: hermes --checkpoints")
            print("  Or in config.yaml: checkpoints: { enabled: true }")
            print("  (Plain /diff still works — it uses git directly.)")
            return

        result = mgr.session_diff(cwd)
        if not result.get("success"):
            print(f"  {result.get('error', 'Could not generate diff')}")
            return

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        if result.get("empty") or (not stat and not diff):
            print("  No changes — Hermes hasn't edited any files here yet.")
            return

        if stat:
            self._print_diff_text(f"\n{stat}")
        if stat_only or not diff:
            return
        self._print_diff_body(diff, "run /diff session --stat for a summary")

    def _print_diff_text(self, text: str) -> None:
        """Render diff/stat text with color when a rich console is present; plain print otherwise
        (e.g. unit tests instantiating the mixin standalone)."""
        console = getattr(self, "console", None)
        if console is not None:
            try:
                from cli import _rich_text_from_ansi
                console.print(_rich_text_from_ansi(text))
                return
            except Exception:
                pass
        print(text)

    def _handle_snapshot_command(self, command: str):
        """Handle /snapshot — lightweight state snapshots for Hermes config/state.

        Syntax: ``/snapshot`` (list), ``create [label]``, ``restore <id>``, ``prune [N]`` (default 20)."""
        from hermes_cli.backup import (
            create_quick_snapshot, list_quick_snapshots,
            restore_quick_snapshot, prune_quick_snapshots,
        )
        from hermes_constants import display_hermes_home

        parts = command.split()
        subcmd = parts[1].lower() if len(parts) > 1 else "list"

        if subcmd in {"list", "ls"}:
            snaps = list_quick_snapshots()
            if not snaps:
                print("  No state snapshots yet.")
                print("  Create one: /snapshot create [label]")
                return
            print(f"  State snapshots ({display_hermes_home()}/state-snapshots/):\n")
            print(f"  {'#':>3}  {'ID':<35} {'Files':>5} {'Size':>10} {'Label'}")
            print(f"  {'─'*3}  {'─'*35} {'─'*5} {'─'*10} {'─'*20}")
            for i, s in enumerate(snaps, 1):
                size = s.get("total_size", 0)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.0f} KB"
                else:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                label = s.get("label") or ""
                print(f"  {i:3}  {s['id']:<35} {s.get('file_count', 0):>5} {size_str:>10} {label}")

        elif subcmd == "create":
            label = " ".join(parts[2:]) if len(parts) > 2 else None
            snap_id = create_quick_snapshot(label=label)
            if snap_id:
                print(f"  Snapshot created: {snap_id}")
            else:
                print("  No state files found to snapshot.")

        elif subcmd in {"restore", "rewind"}:
            if len(parts) < 3:
                print("  Usage: /snapshot restore <snapshot-id>")
                snaps = list_quick_snapshots(limit=1)
                if snaps:
                    print(f"  Most recent: {snaps[0]['id']}")
                return
            snap_id = parts[2]
            # Allow restore by number (1-indexed)
            try:
                idx = int(snap_id)
                snaps = list_quick_snapshots()
                if 1 <= idx <= len(snaps):
                    snap_id = snaps[idx - 1]["id"]
                else:
                    print(f"  Invalid snapshot number. Use 1-{len(snaps)}.")
                    return
            except ValueError:
                pass

            # Close our SessionDB first so the restore doesn't contend with this process's live connection.
            local_session_db = getattr(self, "_session_db", None)
            if local_session_db is not None:
                try:
                    local_session_db.close()
                    self._session_db = None
                except Exception:
                    pass

            if restore_quick_snapshot(snap_id):
                print(f"  Restored state from: {snap_id}")
                print(
                    "  Restart recommended for gateway/dashboard processes "
                    "to pick up state.db changes."
                )
            else:
                print(f"  Snapshot not found: {snap_id}")

        elif subcmd == "prune":
            keep = 20
            if len(parts) > 2:
                try:
                    keep = int(parts[2])
                except ValueError:
                    print("  Usage: /snapshot prune [keep-count]")
                    return
            deleted = prune_quick_snapshots(keep=keep)
            print(f"  Pruned {deleted} old snapshot(s) (keeping {keep}).")

        else:
            print(f"  Unknown subcommand: {subcmd}")
            print("  Usage: /snapshot [list|create [label]|restore <id>|prune [N]]")

    def _handle_export_command(self, command: str):
        """Handle /export [profile] [-o path] — export a profile to a shareable .tar.gz archive."""
        from hermes_cli.profiles import (
            export_profile,
            get_active_profile_name,
            get_profile_export_path,
        )

        parts = command.split()[1:]
        output = None
        if "-o" in parts:
            idx = parts.index("-o")
            if idx + 1 >= len(parts):
                print("  Usage: /export [profile] [-o output.tar.gz]")
                return
            output = parts[idx + 1]
            parts = parts[:idx] + parts[idx + 2:]

        name = parts[0] if parts else (get_active_profile_name() or "default")

        try:
            if not output:
                output = str(get_profile_export_path(name))
            result = export_profile(name, output)
            print(f"  ✓ Exported '{name}' to {result}")
            print("  Share it: the other user runs /import or `hermes profile import <archive>`.")
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"  Error: {e}")

    def _handle_import_command(self, command: str):
        """Handle /import — import a shared profile archive as a new profile.

        Syntax:
            /import <archive.tar.gz> [--name <name>]
        """
        from hermes_cli.profiles import (
            check_alias_collision, create_wrapper_script, import_profile,
        )

        parts = command.split()[1:]
        name = None
        if "--name" in parts:
            idx = parts.index("--name")
            if idx + 1 >= len(parts):
                print("  Usage: /import <archive.tar.gz> [--name <name>]")
                return
            name = parts[idx + 1]
            parts = parts[:idx] + parts[idx + 2:]

        if not parts:
            print("  Usage: /import <archive.tar.gz> [--name <name>]")
            return

        archive = " ".join(parts)  # paths may contain spaces

        try:
            profile_dir = import_profile(archive, name=name)
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"  Error: {e}")
            return

        imported = profile_dir.name
        print(f"  ✓ Imported profile '{imported}' at {profile_dir}")
        try:
            if not check_alias_collision(imported):
                wrapper_path = create_wrapper_script(imported)
                if wrapper_path:
                    print(f"  Wrapper created: {wrapper_path}")
        except Exception:
            pass
        print(f"  Use it: hermes -p {imported}")

    def _handle_stop_command(self):
        """Handle /stop — kill all running background processes and background (async) delegations.
        Separate from interrupt (stop the current turn), as in Codex."""
        from tools.process_registry import process_registry

        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]

        # Background subagents live in their own registry, not the process registry.
        try:
            from tools.async_delegation import active_count, interrupt_all
            n_async = active_count()
        except Exception:
            n_async = 0
            interrupt_all = None

        if not running and not n_async:
            print("  No running background processes.")
            return

        if running:
            print(f"  Stopping {len(running)} background process(es)...")
            killed = process_registry.kill_all()
            print(f"  ✅ Stopped {killed} process(es).")
        if n_async and interrupt_all is not None:
            stopped = interrupt_all(reason="/stop")
            print(f"  ✅ Interrupted {stopped} background delegation(s).")

    def _handle_agents_command(self):
        """Handle /agents — show background processes and agent status."""
        from cli import _cprint
        from tools.process_registry import format_uptime_short, process_registry

        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]
        finished = [p for p in processes if p.get("status") != "running"]

        _cprint(f"  Running processes: {len(running)}")
        for p in running:
            cmd = p.get("command", "")[:80]
            up = format_uptime_short(p.get("uptime_seconds", 0))
            _cprint(f"    {p.get('session_id', '?')} · {up} · {cmd}")

        if finished:
            _cprint(f"  Recently finished: {len(finished)}")

        # Background (async) delegations — delegate_task(background=true)
        try:
            from tools.async_delegation import list_async_delegations
            delegations = list_async_delegations()
        except Exception:
            delegations = []
        running_d = [
            d for d in delegations
            if d.get("status") in ("running", "stalling")
        ]
        if delegations:
            _cprint(f"  Background delegations: {len(running_d)} running")
            for d in delegations:
                goal = (d.get("goal") or "")[:60]
                status = d.get("status", "?")
                line = (
                    f"    {d.get('delegation_id', '?')} · "
                    f"{status} · {goal}"
                )
                # Live-status detail for in-flight delegations (#51690).
                if status == "stalling":
                    quiet = d.get("stalled_after_quiet_seconds")
                    if quiet is not None:
                        line += (
                            f" · no progress {quiet:.0f}s — interrupting"
                        )
                elif status in ("running",):
                    quiet = d.get("seconds_since_progress")
                    if quiet is not None and quiet >= 60:
                        line += f" · quiet {quiet:.0f}s"
                _cprint(line)
                for i, child in enumerate(d.get("children_activity") or []):
                    if not isinstance(child, dict):
                        continue
                    tool = child.get("current_tool")
                    doing = f"in {tool}" if tool else "between turns"
                    part = (
                        f"      └ child {i + 1}: "
                        f"{child.get('api_calls', '?')} api calls · {doing}"
                    )
                    idle = child.get("seconds_since_activity")
                    if idle is not None:
                        part += f" · last activity {idle:.0f}s ago"
                    _cprint(part)

        agent_running = getattr(self, "_agent_running", False)
        _cprint(f"  Agent: {'running' if agent_running else 'idle'}")

    def _handle_journey_command(self, cmd_original: str) -> None:
        """Handle /journey — the learning timeline (see `hermes journey`). Read-only views render
        Rich color that patch_stdout would swallow, so capture with forced ANSI and re-emit via
        ``_cprint``; ``delete``/``edit`` are interactive and keep real stdio."""
        import argparse
        import io
        import shlex
        from contextlib import redirect_stdout

        from cli import _cprint
        from hermes_cli.journey import register_cli

        parser = argparse.ArgumentParser(prog="/journey", add_help=False)
        register_cli(parser)
        rest = cmd_original.split(None, 1)
        try:
            args = parser.parse_args(shlex.split(rest[1]) if len(rest) > 1 else [])
        except SystemExit:
            return

        interactive = getattr(args, "journey_action", None) in ("delete", "edit")
        try:
            if interactive:
                args.func(args)
                return
            args.force_color = True
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            _cprint(buf.getvalue().rstrip("\n"))
        except Exception as exc:
            _cprint(f"  /journey failed: {exc}")

    def _handle_paste_command(self):
        """Handle /paste — explicitly check the clipboard for an image; the reliable fallback where
        BracketedPaste doesn't fire for image-only clipboards (VSCode terminal, Windows Terminal/WSL2)."""
        from cli import _DIM, _RST, _cprint, _termux_example_image_path
        if _is_termux_environment():
            _cprint(
                f"  {_DIM}Clipboard image paste is not available on Termux — "
                f"use /image <path> or paste a local image path like "
                f"{_termux_example_image_path()}{_RST}"
            )
            return

        from hermes_cli.clipboard import has_clipboard_image
        if has_clipboard_image():
            if self._try_attach_clipboard_image():
                n = len(self._attached_images)
                _cprint(f"  📎 Image #{n} attached from clipboard")
            else:
                _cprint(f"  {_DIM}(>_<) Clipboard has an image but extraction failed{_RST}")
        else:
            _cprint(f"  {_DIM}(._.) No image found in clipboard{_RST}")

    def _handle_copy_command(self, cmd_original: str) -> None:
        """Handle /copy [number] — copy assistant output to clipboard."""
        from cli import _assistant_copy_text, _cprint
        parts = cmd_original.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        assistant = [m for m in self.conversation_history if m.get("role") == "assistant"]
        if not assistant:
            _cprint("  Nothing to copy yet.")
            return

        if arg:
            try:
                idx = int(arg) - 1
            except ValueError:
                _cprint("  Usage: /copy [number]")
                return
            if idx < 0 or idx >= len(assistant):
                _cprint(f"  Invalid response number. Use 1-{len(assistant)}.")
                return
        else:
            idx = len(assistant) - 1
            while idx >= 0 and not _assistant_copy_text(assistant[idx].get("content")):
                idx -= 1
            if idx < 0:
                _cprint("  Nothing to copy in assistant responses yet.")
                return

        text = _assistant_copy_text(assistant[idx].get("content"))
        if not text:
            _cprint("  Nothing to copy in that assistant response.")
            return

        try:
            from hermes_cli.clipboard import (
                is_remote_shell_session,
                write_clipboard_text,
            )
            if is_remote_shell_session():
                # Over SSH native tools write the REMOTE clipboard; OSC 52 reaches the user's terminal.
                self._write_osc52_clipboard(text)
                _cprint(
                    f"  Copied assistant response #{idx + 1} via OSC 52 "
                    "(terminal support required)"
                )
                return
            if write_clipboard_text(text):
                _cprint(f"  Copied assistant response #{idx + 1} to clipboard")
                return
            # Native tools unavailable/failed — OSC 52 fallback (SSH/tmux via the emulator).
            self._write_osc52_clipboard(text)
            _cprint(
                f"  Copied assistant response #{idx + 1} via OSC 52 "
                "(terminal support required)"
            )
        except Exception as e:
            _cprint(f"  Clipboard copy failed: {e}")

    def _handle_image_command(self, cmd_original: str):
        """Handle /image <path> — attach a local image file for the next prompt."""
        from cli import _DIM, _IMAGE_EXTENSIONS, _RST, _cprint, _resolve_attachment_path, _split_path_input, _termux_example_image_path
        raw_args = (cmd_original.split(None, 1)[1].strip() if " " in cmd_original else "")
        if not raw_args:
            hint = _termux_example_image_path() if _is_termux_environment() else "/path/to/image.png"
            _cprint(f"  {_DIM}Usage: /image <path>  e.g. /image {hint}{_RST}")
            return

        path_token, _remainder = _split_path_input(raw_args)
        image_path = _resolve_attachment_path(path_token)
        if image_path is None:
            _cprint(f"  {_DIM}(>_<) File not found: {path_token}{_RST}")
            return
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            _cprint(f"  {_DIM}(._.) Not a supported image file: {image_path.name}{_RST}")
            return

        self._attached_images.append(image_path)
        _cprint(f"  📎 Attached image: {image_path.name}")
        if _remainder:
            _cprint(f"  {_DIM}Now type your prompt (or use --image in single-query mode): {_remainder}{_RST}")
        elif _is_termux_environment():
            _cprint(f"  {_DIM}Tip: type your next message, or run hermes chat -q --image {_termux_example_image_path(image_path.name)} \"What do you see?\"{_RST}")

    def _handle_tools_command(self, cmd: str):
        """Handle /tools [list|disable|enable]. Bare shows the tool list; ``list`` shows per-toolset
        status; disable/enable save to config and reset the session so the new tool set takes
        effect cleanly (no prompt-cache breakage mid-conversation)."""
        from cli import _ACCENT, _DIM, _RST, _cprint
        import shlex
        from argparse import Namespace
        from contextlib import redirect_stdout
        from io import StringIO
        from hermes_cli.tools_config import tools_disable_enable_command

        def _run_capture(ns: Namespace) -> None:
            """Run tools_disable_enable_command, routing its ANSI print() output through _cprint
            inside the interactive TUI so patch_stdout's StdoutProxy doesn't garble the escapes.
            Standalone/tests call straight through so real stdout / pytest capture works."""
            # Standalone/tests, run as usual
            if getattr(self, "_app", None) is None:
                tools_disable_enable_command(ns)
                return

            # isatty()=True so color() in hermes_cli/colors.py still emits ANSI escapes.
            class _TTYBuf(StringIO):
                def isatty(self) -> bool:
                    return True

            buf = _TTYBuf()
            with redirect_stdout(buf):
                tools_disable_enable_command(ns)
            for line in buf.getvalue().splitlines():
                _cprint(line)

        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()

        subcommand = parts[1] if len(parts) > 1 else ""
        if subcommand not in {"list", "disable", "enable"}:
            self.show_tools()
            return

        if subcommand == "list":
            _run_capture(Namespace(tools_action="list", platform="cli"))
            return

        names = parts[2:]
        if not names:
            print(f"(._.) Usage: /tools {subcommand} <name> [name ...]")
            print(f"  Built-in toolset:  /tools {subcommand} web")
            print(f"  MCP tool:          /tools {subcommand} github:create_issue")
            return

        # Typing the command is consent. Do NOT use input() — it hangs in prompt_toolkit's loop.
        verb = "Disabling" if subcommand == "disable" else "Enabling"
        label = ", ".join(names)
        _cprint(f"{_ACCENT}{verb} {label}...{_RST}")

        _run_capture(Namespace(tools_action=subcommand, names=names, platform="cli"))

        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config
        self.enabled_toolsets = _get_platform_tools(load_config(), "cli")
        self.new_session()
        _cprint(f"{_DIM}Session reset. New tool configuration is active.{_RST}")

    def _handle_profile_command(self):
        """Display active profile name and home directory."""
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("profile", CommandContext(surface="cli"))
        profile_name = reply.data["profile"]
        display = reply.data["home"]

        print()
        print(f"  Profile: {profile_name}")
        print(f"  Home:    {display}")
        print()

    def _handle_handoff_command(self, cmd_original: str) -> bool:
        """Handle ``/handoff <platform>`` — transfer this CLI session to a gateway platform.

        Validates the platform + home channel, refuses mid-turn (an in-flight run would race the
        gateway's switch_session), writes ``handoff_state='pending'``, then block-polls state.db:
        60s for the gateway to CLAIM the row, then up to 15 min (with heartbeats) for the claimed
        dispatch to finish. ``completed`` → resume hint + return False (caller exits like /quit).
        ``failed`` / pending-timeout → error + True (session kept). A running-timeout leaves the
        row untouched (the gateway owns it) and returns True."""
        from cli import _cprint
        from hermes_state import format_session_db_unavailable

        parts = cmd_original.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /handoff <platform>")
            _cprint("  Hands the current session off to that platform's home channel.")
            _cprint("  The CLI session ends here; resume it later with /resume.")
            return True

        platform_name = parts[1].strip().lower()

        # Validate platform name + home channel via the live gateway config.
        try:
            from gateway.config import load_gateway_config, Platform
        except Exception as exc:  # pragma: no cover — gateway pkg always shipped
            _cprint(f"  Could not load gateway config: {exc}")
            return True

        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            _cprint(f"  Unknown platform '{platform_name}'.")
            return True

        try:
            gw_config = load_gateway_config()
        except Exception as exc:
            _cprint(f"  Could not load gateway config: {exc}")
            return True

        pcfg = gw_config.platforms.get(platform)
        if not pcfg or not pcfg.enabled:
            # Relay aliasing: a relay-fronted gateway has only a RELAY config block, yet
            # /handoff discord is deliverable when the relay fronts it. UX pre-check only —
            # the gateway watcher re-checks against the authenticated transport before dispatch.
            relay_fronts = False
            try:
                from gateway.relay import relay_platform_identities
                relay_cfg = gw_config.platforms.get(Platform.RELAY)
                if relay_cfg and relay_cfg.enabled:
                    fronted = {p for p, _ in relay_platform_identities()}
                    relay_fronts = platform_name in fronted
            except Exception:
                relay_fronts = False
            if not relay_fronts:
                _cprint(f"  Platform '{platform_name}' is not configured/enabled in the gateway.")
                return True

        home = gw_config.get_home_channel(platform)
        if not home or not home.chat_id:
            _cprint(f"  No home channel configured for {platform_name}.")
            _cprint("  Set one with /sethome on the destination chat first.")
            return True

        # Refuse mid-turn: an in-flight agent run would race with the
        # gateway's switch_session and the synthetic turn dispatch.
        if getattr(self, "_agent_running", False):
            _cprint("  Agent is busy. Wait for the current turn to finish, then retry /handoff.")
            return True

        # Make sure we have a SessionDB handle.
        if not self._session_db:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception:
                pass
        if not self._session_db:
            _cprint(f"  {format_session_db_unavailable()}")
            return True

        # Ensure the session row exists (an empty session has flushed nothing yet): the
        # gateway needs a row to switch_session onto; set_session_title's INSERT OR IGNORE
        # creates it.
        try:
            row = self._session_db.get_session(self.session_id)
            if not row:
                placeholder_title = f"handoff-{self.session_id[:8]}"
                self._session_db.set_session_title(self.session_id, placeholder_title)
        except Exception as exc:
            _cprint(f"  Could not ensure session row in state.db: {exc}")
            return True

        # Display title for messaging.
        session_title = ""
        try:
            row = self._session_db.get_session(self.session_id)
            if row:
                session_title = row.get("title") or ""
        except Exception:
            pass
        if not session_title:
            session_title = self.session_id[:8]

        # Mark pending — gateway watcher will pick this up.
        ok = self._session_db.request_handoff(self.session_id, platform_name)
        if not ok:
            _cprint("  Session is already in flight for handoff. Wait for it to settle, then retry.")
            return True

        _cprint(f"  Queued handoff of '{session_title}' → {platform_name} (home: {home.name}).")
        _cprint("  Waiting for the gateway to pick it up...")

        # Two-phase poll, tick every 0.5s. PENDING (unclaimed): 60s — a timeout means no
        # gateway watcher is looking at this state.db, and the CAS fail can't stomp a claim
        # landing the same instant. RUNNING (claimed): the gateway replays the full
        # transcript via a synthetic turn (routinely >60s), so wait much longer with a
        # heartbeat and on timeout do NOT touch the row — failing it here was the
        # split-brain bug (CLI said "failed" while the gateway completed the switch).
        import time as _time
        _PENDING_TIMEOUT = 60.0
        _RUNNING_TIMEOUT = 900.0  # full synthetic agent turn + delivery
        _HEARTBEAT_EVERY = 30.0
        pending_deadline = _time.time() + _PENDING_TIMEOUT
        running_deadline = None
        next_heartbeat = None
        last_state = "pending"
        while True:
            try:
                state_row = self._session_db.get_handoff_state(self.session_id)
            except Exception:
                state_row = None
            current = (state_row or {}).get("state") or "pending"
            if current != last_state:
                if current == "running":
                    _cprint("  Gateway picked it up; transferring...")
                    running_deadline = _time.time() + _RUNNING_TIMEOUT
                    next_heartbeat = _time.time() + _HEARTBEAT_EVERY
                last_state = current
            if current == "completed":
                _cprint("")
                _cprint(f"  ↻ Handoff complete. The session is now active on {platform_name}.")
                _cprint(f"  Resume it on this CLI later with: /resume {session_title}")
                _cprint("")
                # Mark handed off so _run_cleanup does NOT finalize it on exit: the gateway
                # owns the row now, and setting end_reason under it would drop the handoff
                # leg from session history / session_search.
                from cli import _handed_off_session_ids
                _handed_off_session_ids.add(self.session_id)
                # End the CLI cleanly — same exit semantics as /quit.
                self._should_exit = True
                return False
            if current == "failed":
                err = (state_row or {}).get("error") or "unknown error"
                _cprint(f"  Handoff failed: {err}")
                _cprint("  Your CLI session is intact. Try /handoff again, or /resume on the platform manually.")
                return True
            now = _time.time()
            if current == "pending":
                if now >= pending_deadline:
                    break
            else:  # running
                if next_heartbeat is not None and now >= next_heartbeat:
                    _cprint("  Still transferring (the agent is replaying your session on the destination)...")
                    next_heartbeat = now + _HEARTBEAT_EVERY
                if running_deadline is not None and now >= running_deadline:
                    # Do NOT fail the row: the gateway owns it (split-brain bug otherwise).
                    _cprint("  The gateway is taking unusually long to finish the transfer.")
                    _cprint(f"  Check {platform_name} — the session may still arrive there.")
                    _cprint("  This CLI is no longer waiting. Avoid continuing this session here;")
                    _cprint("  if nothing arrives, retry /handoff once the state settles.")
                    return True
            _time.sleep(0.5)

        # Pending timed out: clear the flag so the user can retry (CAS — a claim racing this
        # instant wins; we lose only the retry convenience, never the handoff).
        try:
            self._session_db.fail_handoff(
                self.session_id,
                "timed out waiting for gateway",
                only_states=("pending",),
            )
        except TypeError:
            # Older SessionDB without only_states (mixed installs): legacy unconditional fail.
            try:
                self._session_db.fail_handoff(self.session_id, "timed out waiting for gateway")
            except Exception:
                pass
        except Exception:
            pass
        _cprint("  Timed out waiting for the gateway. Is `hermes gateway` running?")
        _cprint("  Your CLI session is intact.")
        return True

    def _handle_resume_command(self, cmd_original: str) -> None:
        """Handle /resume <session_id_or_title> — switch to a previous session mid-conversation."""
        from cli import _cprint, _sync_process_session_id
        parts = cmd_original.split(None, 1)
        target = parts[1].strip() if len(parts) > 1 else ""

        # Users copy the help text's placeholder brackets/quotes verbatim (``/resume <abc123>``).
        if len(target) >= 2 and (target[0], target[-1]) in {("<", ">"), ("[", "]"), ('"', '"'), ("'", "'")}:
            target = target[1:-1].strip()

        if not target:
            _cprint("  Usage: /resume <number|session_id_or_title>")
            if self._show_recent_sessions(reason="resume"):
                # Arm a one-shot pending-resume selection so a bare number on the next line
                # works. Must be the same list _show_recent_sessions showed and the numbered
                # branch below resolves — all three use _list_recent_sessions(limit=10).
                self._pending_resume_sessions = self._list_recent_sessions(limit=10)
                return
            _cprint("  Tip:   Use /history or `hermes sessions list` to find sessions.")
            return

        # Any explicit /resume <target> supersedes a previously-armed bare
        # numbered prompt.
        self._pending_resume_sessions = None

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            _cprint(f"  {format_session_db_unavailable()}")
            return

        # Resolve numbered selection, title, or ID
        if target.isdigit():
            sessions = self._list_recent_sessions(limit=10)
            index = int(target)
            if index < 1 or index > len(sessions):
                _cprint(f"  Resume index {index} is out of range.")
                _cprint("  Use /resume with no arguments to see available sessions.")
                return
            selected = sessions[index - 1]
            target_id = selected["id"]
        else:
            from hermes_cli.main import _resolve_session_by_name_or_id
            resolved = _resolve_session_by_name_or_id(target)
            target_id = resolved or target

        session_meta = self._session_db.get_session(target_id)
        if not session_meta:
            _cprint(f"  Session not found: {target}")
            _cprint("  Use /sessions or `hermes sessions list` to see available sessions.")
            return

        # If the target is the empty head of a compression chain, redirect to
        # the descendant that actually holds the transcript. See #15000.
        try:
            resolved_id = self._session_db.resolve_resume_session_id(target_id)
        except Exception:
            resolved_id = target_id
        if resolved_id and resolved_id != target_id:
            _cprint(
                f"  Session {target_id} was compressed into {resolved_id}; "
                f"resuming the descendant with your transcript."
            )
            target_id = resolved_id
            resolved_meta = self._session_db.get_session(target_id)
            if resolved_meta:
                session_meta = resolved_meta

        if target_id == self.session_id:
            _cprint("  Already on that session.")
            return

        old_session_id = self.session_id
        _end_current_session(self, "resumed_other")

        # Switch to the target session
        self.session_id = target_id
        self._resumed = True
        self._pending_title = None
        _sync_process_session_id(target_id)

        # Both projections come from one lineage SELECT: model_history is
        # alternation-repaired for LIVE REPLAY (heals a durable ``user;user`` violation
        # once instead of on every request); display_history is the full lineage
        # verbatim for _display_resumed_history() (matches startup --resume).
        model_history, display_history = self._session_db.get_resume_conversations(target_id)
        self.conversation_history = [m for m in (model_history or []) if m.get("role") != "session_meta"]
        self._resume_display_history = [m for m in (display_history or []) if m.get("role") != "session_meta"]

        # Re-open the target session so it's not marked as ended
        try:
            self._session_db.reopen_session(target_id)
        except Exception:
            pass

        _sync_agent_to_session(self, target_id, parent_session_id=old_session_id, reason="resume")

        title_part = f" \"{session_meta['title']}\"" if session_meta.get("title") else ""
        from agent.context_compressor import is_user_originated_turn

        # Count only user-originated turns: legacy compaction handoffs are durable
        # role=user rows without display_kind.
        msg_count = len([m for m in self._resume_display_history if is_user_originated_turn(m)])
        if self.conversation_history:
            _cprint(
                f"  ↻ Resumed session {target_id}{title_part}"
                f" ({msg_count} user message{'s' if msg_count != 1 else ''},"
                f" {len(self.conversation_history)} total)"
            )
            self._display_resumed_history()
        else:
            _cprint(f"  ↻ Resumed session {target_id}{title_part} — no messages, starting fresh.")

        # Same contract as a startup --resume: retarget the process/tool cwd (else the
        # terminal tools keep operating in the wrong repo), restore the target session's
        # persisted YOLO bypass (the previous session's stops applying — the approval
        # session key changed), and its model/provider (else it reverts to config default).
        self._restore_session_cwd(session_meta)
        self._restore_session_yolo(session_meta)
        self._restore_session_model(session_meta)

    def _handle_sessions_command(self, cmd_original: str) -> None:
        """Handle /sessions [list|<id_or_title>] — browse or resume previous sessions.

        Bare/``list`` prints the same recent-sessions table /resume shows; an explicit target
        delegates to the resume flow so ``/sessions <id>`` and ``/resume <id>`` behave identically.
        (The TUI has a picker overlay; the classic CLI prints inline.)"""
        from cli import _cprint
        parts = cmd_original.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        sub = arg.lower()

        # Bare /sessions or /sessions list — show recent sessions inline.
        if not arg or sub in {"list", "ls", "browse"}:
            if not self._session_db:
                from hermes_state import format_session_db_unavailable
                _cprint(f"  {format_session_db_unavailable()}")
                return
            if not self._show_recent_sessions(reason="sessions"):
                _cprint("  (._.) No previous sessions yet.")
            return

        # /sessions <id_or_title> behaves the same as /resume <id_or_title>.
        self._handle_resume_command(f"/resume {arg}")

    def _handle_worktree_command(self, cmd_original: str) -> None:
        """Handle /worktree — inspect, create, or reclaim isolated git worktrees.

        Syntax:
            /worktree                  — show the active worktree (if any)
            /worktree new [name]       — create a worktree and move this session into it
            /worktree list             — list worktrees under the repo's .worktrees/
            /worktree prune [--dry-run] — reclaim safe trees + merged branches

        ``new`` retargets the terminal/file tools (``TERMINAL_CWD`` + process cwd) at the new
        tree; the launcher's exit cleanup applies (kept only with unpushed commits, as ``hermes -w``).
        ``prune`` is the attended reclaim from hermes_cli/worktree_gc.py: never deletes tracked
        changes, unique unpushed commits, or in-use trees."""
        import subprocess

        import cli as _cli

        parts = cmd_original.split(None, 2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        repo_root = _cli._git_repo_root()

        if not sub or sub in {"status", "show"}:
            active = _cli._active_worktree
            if active:
                print(f"  Active worktree: {active['path']}")
                print(f"  Branch: {active['branch']}")
            else:
                print("  No active worktree for this session.")
            if repo_root:
                print("  /worktree new [name] — create one and move this session into it")
                print("  /worktree prune      — reclaim stale trees and merged branches")
            else:
                print("  (not inside a git repository)")
            return

        if sub in {"prune", "gc", "clean"}:
            if not repo_root:
                print("  Not inside a git repository.")
                return
            rest = parts[2].strip().lower() if len(parts) > 2 else ""
            dry_run = "--dry-run" in rest or "-n" in rest.split()
            from hermes_cli import worktree_gc

            active = _cli._active_worktree
            tree_records = worktree_gc.audit_worktrees(repo_root, with_sizes=False)
            if active:
                # Never reap the tree this session is sitting in, even if judged clean+merged.
                active_path = str(active.get("path") or "")
                tree_records = [
                    record for record in tree_records
                    if record.path != active_path
                ]
            actions = worktree_gc.reclaim_worktrees(
                repo_root, dry_run=dry_run, records=tree_records
            )
            actions += worktree_gc.reclaim_branches(repo_root, dry_run=dry_run)
            if actions:
                for line in actions:
                    print(f"  {line}")
                print(f"  {len(actions)} action(s) {'planned' if dry_run else 'done'}.")
            else:
                print("  Nothing to reclaim — remaining trees/branches carry real work.")
            kept = [
                record for record in tree_records
                if record.verdict == "keep"
                and "kanban" not in record.reason and "in use" not in record.reason
            ]
            if kept:
                print(f"  Preserved {len(kept)} tree(s) with real work:")
                for record in kept:
                    print(f"    {record.name}: {record.reason}")
            return

        if sub in {"list", "ls"}:
            if not repo_root:
                print("  Not inside a git repository.")
                return
            try:
                result = subprocess.run(
                    ["git", "worktree", "list"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=10, cwd=repo_root,
                )
                out = result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                out = ""
            if out:
                for line in out.splitlines():
                    print(f"  {line}")
            else:
                print("  Could not list worktrees.")
            return

        if sub in {"new", "add", "create"}:
            if not repo_root:
                print("  ❌ /worktree new requires being inside a git repository.")
                return
            name = parts[2].strip() if len(parts) > 2 else None
            from hermes_cli.config import load_config
            try:
                sync_base = bool(load_config().get("worktree_sync", True))
            except Exception:
                sync_base = True
            wt_info = _cli._setup_worktree(
                repo_root=repo_root, sync_base=sync_base, name=name,
            )
            if not wt_info:
                return  # _setup_worktree already printed the failure
            # Retarget the session's terminal/file tools at the new tree (as `hermes -w` does).
            try:
                os.chdir(wt_info["path"])
            except OSError as e:
                print(f"  ⚠ Created worktree but could not enter it: {e}")
            os.environ["TERMINAL_CWD"] = wt_info["path"]
            # Same keep-if-unpushed cleanup as `hermes -w`. Only one tree is "active" per
            # process; an earlier one keeps its own atexit registration (explicit info arg).
            import atexit
            _cli._active_worktree = wt_info
            atexit.register(_cli._cleanup_worktree, wt_info)
            print(f"  ✅ Worktree ready: {wt_info['path']}")
            print(f"  Branch: {wt_info['branch']}")
            print("  Terminal and file tools now operate in the worktree.")
            return

        print(f"  Unknown /worktree subcommand: {sub}")
        print("  Usage: /worktree [new [name] | list]")

    def _handle_branch_command(self, cmd_original: str) -> None:
        """Handle /branch [name] — fork the current session into a new independent copy of the
        full history so a different approach can be explored without losing the original."""
        from cli import _cprint, _sync_process_session_id
        if not self.conversation_history:
            _cprint("  No conversation to branch — send a message first.")
            return

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            _cprint(f"  {format_session_db_unavailable()}")
            return

        parts = cmd_original.split(None, 1)
        branch_name = parts[1].strip() if len(parts) > 1 else ""

        # Generate the new session ID
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            # Auto-generate from the current session title
            current_title = None
            if self._session_db:
                current_title = self._session_db.get_session_title(self.session_id)
            base = current_title or "branch"
            branch_title = self._session_db.get_next_title_in_lineage(base)

        # Save the current session's state before branching
        parent_session_id = self.session_id
        _end_current_session(self, "branched")

        # Create the new session with parent link. The stable ``_branched_from`` marker keeps
        # the branch visible in /resume + /sessions even after the parent is re-ended with a
        # different end_reason.
        try:
            self._session_db.create_session(
                session_id=new_session_id,
                source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=self.model,
                model_config={
                    "max_iterations": self.max_turns,
                    "reasoning_config": self.reasoning_config,
                    "_branched_from": parent_session_id,
                },
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            _cprint(f"  Failed to create branch session: {e}")
            return

        # Copy history in bounded-chunk transactions. Best-effort: a failed copy still yields a usable branch.
        try:
            self._session_db.append_messages_batch(
                new_session_id,
                [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content"),
                        "tool_name": msg.get("tool_name") or msg.get("name"),
                        "tool_calls": msg.get("tool_calls"),
                        "tool_call_id": msg.get("tool_call_id"),
                        "reasoning": msg.get("reasoning"),
                        "reasoning_details": msg.get("reasoning_details"),
                        "codex_reasoning_items": msg.get("codex_reasoning_items"),
                        "codex_message_items": msg.get("codex_message_items"),
                        # api_content sidecar: the branch's first turn replays the parent's exact
                        # wire bytes (warm prompt cache) instead of a cold prefill.
                        "api_content": extract_api_content_sidecar(msg),
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in self.conversation_history
                ],
                chunk_rows=500,
            )
        except Exception:
            pass  # Best-effort copy

        # Set title on the branch
        try:
            self._session_db.set_session_title(new_session_id, branch_title)
        except Exception:
            pass

        # Switch to the new session
        self._transfer_session_yolo(self.session_id, new_session_id)
        self.session_id = new_session_id
        self.session_start = now
        self._pending_title = None
        self._resumed = True  # Prevents auto-title generation
        _sync_process_session_id(new_session_id)

        # Sync the agent
        if self.agent:
            self.agent.session_start = now
        _sync_agent_to_session(self, new_session_id, parent_session_id=parent_session_id, reason="branch")

        msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
        _cprint(
            f"  ⑂ Branched session \"{branch_title}\""
            f" ({msg_count} user message{'s' if msg_count != 1 else ''})"
        )
        _cprint(f"  Original session: {parent_session_id}")
        _cprint(f"  Branch session:   {new_session_id}")

    def _handle_personality_command(self, cmd: str):
        """Handle the /personality command to set predefined personalities.

        All resolution/persistence goes through hermes_cli.personality —
        the single owner of personality state on every surface.
        """
        from hermes_cli.personality import (
            describe_personality,
            normalize_personality_name,
            persist_personality,
            prompt_text,
            resolve_personality,
        )
        parts = cmd.split(maxsplit=1)

        if len(parts) > 1:
            # Set personality
            personality_name = parts[1].strip()

            try:
                name, personality_prompt = resolve_personality(
                    personality_name, getattr(self, "config", None)
                )
            except ValueError:
                print(f"(._.) Unknown personality: {personality_name.lower()}")
                print(f"  Available: none, {', '.join(self.personalities.keys())}")
                return

            saved = persist_personality(name)
            if not name:
                # Neutral reset — fall back to the user-owned manual prompt.
                try:
                    from hermes_cli.config import cfg_get, read_raw_config

                    self.system_prompt = prompt_text(
                        cfg_get(read_raw_config(), "agent", "system_prompt", default="")
                    )
                except Exception:
                    self.system_prompt = ""
                self.agent = None  # Force re-init
                if saved:
                    print("(^_^)b Personality cleared (saved to config)")
                else:
                    print("(^_^) Personality cleared (session only)")
                print("  No personality overlay — using base agent behavior.")
            else:
                self.system_prompt = personality_prompt
                self.agent = None  # Force re-init
                if saved:
                    print(f"(^_^)b Personality set to '{name}' (saved to config)")
                else:
                    print(f"(^_^) Personality set to '{name}' (session only)")
                print(f"  \"{personality_prompt[:60]}{'...' if len(personality_prompt) > 60 else ''}\"")
        else:
            # Show available personalities
            try:
                from hermes_cli.config import read_raw_config

                current = normalize_personality_name(
                    (read_raw_config().get("display") or {}).get("personality", "")
                )
            except Exception:
                current = ""
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 12 + "(^o^)/ Personalities" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            marker = " *" if not current else "  "
            print(f" {marker}{'none':<12} - (no personality overlay)")
            for name, prompt in self.personalities.items():
                marker = " *" if name == current else "  "
                print(f" {marker}{name:<12} - {describe_personality(prompt)}")
            print()
            print("  Usage: /personality <name>   (* = active)")
            print()

    def _handle_pet_command(self, cmd: str):
        """Toggle, browse, or adopt a petdex mascot.

        ``/pet`` / ``/pet toggle`` flips ``display.pet.enabled``; ``list`` browses the gallery;
        ``scale <n>`` resizes; ``<slug>`` adopts (installs if needed); ``off`` disables.
        Writes ``display.pet.*`` to config; pet surfaces pick it up on their next poll."""
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import _set_active, _set_enabled, print_pet_gallery, set_pet_scale, toggle_pet_display

        parts = cmd.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        low = arg.lower()

        if not arg or low == "toggle":
            enabled, name, err = toggle_pet_display()
            if err:
                print(f"(x_x) {err}")
                return
            if enabled:
                print(f"(^_^)b {name} is out — it'll pop in shortly.")
            else:
                print(f"(-_-)zzZ {name} put away." if name else "(-_-)zzZ Pet put away.")
            return

        if low in ("list", "gallery", "browse", "all"):
            print_pet_gallery()
            return

        if low == "scale" or low.startswith("scale "):
            value = arg[len("scale"):].strip()
            if not value:
                print("(o_o) Usage: /pet scale <factor>  (e.g. /pet scale 0.5)")
                return
            scale, err = set_pet_scale(value)
            print(f"(x_x) {err}" if err else f"(^_^) Pet scale → {scale:g}.")
            return

        if low == "off":
            _set_enabled(False)
            print("(-_-)zzZ Pet put away.")
            return

        print(f"(o_o) Fetching '{arg}' from petdex…")
        try:
            pet = store.install_pet(arg)
        except (store.PetStoreError, ManifestError) as exc:
            print(f"(x_x) Couldn't adopt '{arg}': {exc}")
            return
        _set_active(arg)
        print(f"(^_^)b {pet.display_name} is out — it'll pop in shortly.")

    def _handle_hatch_command(self, cmd: str):
        """Generate ("hatch") a new petdex pet from a description: base look, one animation row
        per state, spritesheet, then adopt. Progress streams inline (~a minute of image calls).
        The desktop app opens a richer overlay for this command instead."""
        from agent.pet import store
        from agent.pet.generate import orchestrate
        from agent.pet.generate.imagegen import GenerationError
        from hermes_cli.pets import _set_active

        parts = cmd.split(maxsplit=1)
        concept = parts[1].strip() if len(parts) > 1 else ""

        if not concept:
            # Dispatched from the process_loop daemon thread while prompt_toolkit owns
            # stdin — a raw input() here never renders and eats keystrokes. Use the
            # thread-aware prompt helper (run_in_terminal; None when prompting isn't safe).
            prompt_helper = getattr(self, "_prompt_text_input", None)
            if callable(prompt_helper):
                concept = (prompt_helper("(o_o) Describe your pet: ") or "").strip()
            else:
                try:
                    concept = input("(o_o) Describe your pet: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return

        if not concept:
            print("(o_o) Usage: /hatch <description>  (e.g. /hatch a tiny cyber fox)")
            return

        # A short, friendly display name from the first few words of the concept.
        display_name = " ".join(w.capitalize() for w in concept.split()[:3])[:28].strip() or "Pet"
        slug = store.slugify(display_name) or store.slugify(concept) or "pet"

        print(f"(o_o) Designing '{concept}'… (a minute of image-model calls)")
        try:
            drafts = orchestrate.generate_base_drafts(concept, n=1)
        except GenerationError as exc:
            print(f"(x_x) Couldn't generate a base look: {exc}")
            return

        if not drafts:
            print("(x_x) No base draft came back — try again.")
            return

        def _progress(event: str, detail: str) -> None:
            if event == "row":
                # detail is "<state>:<done>:<total>"; show the state name.
                state = detail.split(":", 1)[0]
                print(f"  ┊ drawing {state}…")
            elif event == "compose":
                print("  ┊ composing spritesheet…")
            elif event == "save":
                print("  ┊ saving…")

        try:
            result = orchestrate.hatch_pet(
                base_image=drafts[0],
                slug=slug,
                display_name=display_name,
                concept=concept,
                on_progress=_progress,
            )
        except GenerationError as exc:
            print(f"(x_x) Hatch failed: {exc}")
            return

        _set_active(result.slug)
        print(f"(^_^)b {result.display_name} hatched and adopted — it'll pop in shortly!")

    def _handle_cron_command(self, cmd: str):
        """Handle the /cron command to manage scheduled tasks."""
        import shlex

        tokens = shlex.split(cmd)
        if len(tokens) == 1:
            self._cron_overview()
            return
        subcommand = tokens[1].lower()
        opts = _parse_cron_flags(tokens[2:])
        if opts is None:
            return
        handler = _CRON_SUBCOMMANDS.get(subcommand)
        if handler is None:
            print(f"(._.) Unknown cron command: {subcommand}")
            print("  Available: list, add, edit, pause, resume, run, remove")
            return
        handler(self, subcommand, opts)

    def _cron_overview(self) -> None:
        print()
        print("+" + "-" * 68 + "+")
        print("|" + " " * 22 + "(^_^) Scheduled Tasks" + " " * 23 + "|")
        print("+" + "-" * 68 + "+")
        print()
        print("  Commands:")
        print("    /cron list")
        print('    /cron add "every 2h" "Check server status" [--skill blogwatcher]')
        print('    /cron edit <job_id> --schedule "every 4h" --prompt "New task"')
        print("    /cron edit <job_id> --skill blogwatcher --skill maps")
        print("    /cron edit <job_id> --remove-skill blogwatcher")
        print("    /cron edit <job_id> --clear-skills")
        print("    /cron pause <job_id>")
        print("    /cron resume <job_id>")
        print("    /cron run <job_id>")
        print("    /cron remove <job_id>")
        print()
        result = _cron_api(action="list")
        jobs = result.get("jobs", []) if result.get("success") else []
        if jobs:
            print("  Current Jobs:")
            print("  " + "-" * 63)
            for job in jobs:
                repeat_str = job.get("repeat", "?")
                print(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {repeat_str:<8}")
                if job.get("skills"):
                    print(f"      Skills: {', '.join(job['skills'])}")
                print(f"      {job.get('prompt_preview', '')}")
                if job.get("next_run_at"):
                    print(f"      Next: {job['next_run_at']}")
                print()
        else:
            print("  No scheduled jobs. Use '/cron add' to create one.")
        print()

    def _cron_list(self, subcommand: str, opts: dict) -> None:
        result = _cron_api(action="list", include_disabled=opts["all"])
        jobs = result.get("jobs", []) if result.get("success") else []
        if not jobs:
            print("(._.) No scheduled jobs.")
            return
        print()
        print("Scheduled Jobs:")
        print("-" * 80)
        for job in jobs:
            print(f"  ID: {job['job_id']}")
            print(f"  Name: {job['name']}")
            print(f"  State: {job.get('state', '?')}")
            print(f"  Schedule: {job['schedule']} ({job.get('repeat', '?')})")
            print(f"  Next run: {job.get('next_run_at', 'N/A')}")
            if job.get("skills"):
                print(f"  Skills: {', '.join(job['skills'])}")
            print(f"  Prompt: {job.get('prompt_preview', '')}")
            if job.get("last_run_at"):
                status = job.get("last_status") or "?"
                # delivery_failed: the run succeeded but delivery didn't — the reason lives
                # in last_delivery_error (last_error is None).
                if status == "delivery_failed" and job.get("last_delivery_error"):
                    status = f"delivery_failed: {job['last_delivery_error']}"
                print(f"  Last run: {job['last_run_at']} ({status})")
            print()

    def _cron_add(self, subcommand: str, opts: dict) -> None:
        positionals = opts["positionals"]
        if not positionals:
            print("(._.) Usage: /cron add <schedule> <prompt>")
            return
        schedule = opts["schedule"] or positionals[0]
        prompt = opts["prompt"] or " ".join(positionals[1:])
        skills = _normalize_skills(opts["skills"])
        if not prompt and not skills:
            print("(._.) Please provide a prompt or at least one skill")
            return
        result = _cron_api(
            action="create", schedule=schedule, prompt=prompt or None, name=opts["name"],
            deliver=opts["deliver"], repeat=opts["repeat"], skills=skills or None,
        )
        if result.get("success"):
            print(f"(^_^)b Created job: {result['job_id']}")
            print(f"  Schedule: {result['schedule']}")
            if result.get("skills"):
                print(f"  Skills: {', '.join(result['skills'])}")
            print(f"  Next run: {result['next_run_at']}")
        else:
            print(f"(x_x) Failed to create job: {result.get('error')}")

    def _cron_edit(self, subcommand: str, opts: dict) -> None:
        from cli import get_job

        positionals = opts["positionals"]
        if not positionals:
            print("(._.) Usage: /cron edit <job_id> [--schedule ...] [--prompt ...] [--skill ...]")
            return
        job_id = positionals[0]
        existing = get_job(job_id)
        if not existing:
            print(f"(._.) Job not found: {job_id}")
            return

        # Skill edit precedence: --clear-skills > --skill (replace) > --add/--remove (merge) > untouched.
        final_skills = None
        replacement_skills = _normalize_skills(opts["skills"])
        add_skills = _normalize_skills(opts["add_skills"])
        remove_skills = set(_normalize_skills(opts["remove_skills"]))
        existing_skills = list(existing.get("skills") or ([] if not existing.get("skill") else [existing.get("skill")]))
        if opts["clear_skills"]:
            final_skills = []
        elif replacement_skills:
            final_skills = replacement_skills
        elif add_skills or remove_skills:
            final_skills = [skill for skill in existing_skills if skill not in remove_skills]
            for skill in add_skills:
                if skill not in final_skills:
                    final_skills.append(skill)

        result = _cron_api(
            action="update", job_id=job_id, schedule=opts["schedule"], prompt=opts["prompt"],
            name=opts["name"], deliver=opts["deliver"], repeat=opts["repeat"], skills=final_skills,
        )
        if result.get("success"):
            job = result["job"]
            print(f"(^_^)b Updated job: {job['job_id']}")
            print(f"  Schedule: {job['schedule']}")
            if job.get("skills"):
                print(f"  Skills: {', '.join(job['skills'])}")
            else:
                print("  Skills: none")
        else:
            print(f"(x_x) Failed to update job: {result.get('error')}")

    def _cron_job_action(self, subcommand: str, opts: dict) -> None:
        """pause / resume / run / remove (aliases rm, delete) on one job id."""
        positionals = opts["positionals"]
        if not positionals:
            print(f"(._.) Usage: /cron {subcommand} <job_id>")
            return
        job_id = positionals[0]
        action = "remove" if subcommand in {"remove", "rm", "delete"} else subcommand
        result = _cron_api(action=action, job_id=job_id, reason="paused from /cron" if action == "pause" else None)
        if not result.get("success"):
            print(f"(x_x) Failed to {action} job: {result.get('error')}")
            return
        if action == "pause":
            print(f"(^_^)b Paused job: {result['job']['name']} ({job_id})")
        elif action == "resume":
            print(f"(^_^)b Resumed job: {result['job']['name']} ({job_id})")
            print(f"  Next run: {result['job'].get('next_run_at')}")
        elif action == "run":
            print(f"(^_^)b Triggered job: {result['job']['name']} ({job_id})")
            print("  It will run on the next scheduler tick.")
        else:
            removed = result.get("removed_job", {})
            print(f"(^_^)b Removed job: {removed.get('name', job_id)} ({job_id})")

    def _handle_suggestions_command(self, cmd: str):
        """Handle /suggestions — review/accept/dismiss suggested automations via the shared handler.
        CLI origin is the local platform so an accepted job's "origin" delivery resolves to a home channel."""
        import shlex

        try:
            tokens = shlex.split(cmd)[1:] if cmd else []
        except ValueError:
            tokens = (cmd or "").split()[1:]
        args = " ".join(tokens)
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command
            output = handle_suggestions_command(args)
        except Exception as e:
            output = f"Suggestions command failed: {e}"
        self._console_print(output)

    def _handle_blueprint_command(self, cmd: str):
        """Handle /blueprint — set up an automation from a blueprint template (shared handler).
        Bare lists the catalog; ``<name>`` seeds the agent to ask for each value conversationally
        (``agent_seed``, run as the next turn); ``<name> slot=val …`` creates the job directly."""
        import shlex

        try:
            tokens = shlex.split(cmd)[1:] if cmd else []
        except ValueError:
            tokens = (cmd or "").split()[1:]
        args = " ".join(shlex.quote(t) for t in tokens)
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command
            result = handle_blueprint_command(args)
        except Exception as e:
            self._console_print(f"Cron blueprint command failed: {e}")
            return
        self._console_print(result.text)
        seed = getattr(result, "agent_seed", None)
        if seed:
            # One-shot: the interactive loop picks this up right after the
            # slash command returns and runs it as a normal agent turn.
            self._pending_agent_seed = seed

    def _handle_curator_command(self, cmd: str):
        """Handle /curator slash command.

        Delegates to hermes_cli.curator so the CLI and the `hermes curator`
        subcommand share the same handler set.
        """
        import shlex

        tokens = shlex.split(cmd)[1:] if cmd else []
        if not tokens:
            tokens = ["status"]

        try:
            from hermes_cli.curator import cli_main
            cli_main(tokens)
        except SystemExit:
            # argparse calls sys.exit() on --help or errors; swallow so we
            # don't kill the interactive session.
            pass
        except Exception as exc:
            print(f"(._.) curator: {exc}")

    def _handle_kanban_command(self, cmd: str):
        """Handle /kanban — strip the leading ``/kanban`` and hand the rest to ``kanban.run_slash``."""
        from hermes_cli.kanban import run_slash

        rest = cmd.strip()
        if rest.startswith("/"):
            rest = rest.lstrip("/")
        if rest.startswith("kanban"):
            rest = rest[len("kanban"):].lstrip()
        try:
            output = run_slash(rest)
        except Exception as exc:  # pragma: no cover - defensive
            output = f"(._.) kanban error: {exc}"
        if output:
            print(output)

    def _handle_skills_command(self, cmd: str):
        """Handle /skills slash command — delegates to hermes_cli.skills_hub."""
        from cli import ChatConsole
        # Intercept write-approval review subcommands first (pending/approve/
        # reject/diff/mode); everything else goes to the skills hub.
        parts = cmd.strip().split()
        args = parts[1:] if len(parts) > 1 else []
        if args and args[0].lower() in {"pending", "approve", "apply", "reject",
                                        "deny", "drop", "diff", "approval", "mode"}:
            from hermes_cli.write_approval_commands import handle_pending_subcommand
            from tools import write_approval as wa
            out = handle_pending_subcommand(
                wa.SKILLS, args,
                set_mode_fn=lambda enabled: self._save_write_approval("skills", enabled),
            )
            if out is not None:
                print(out)
                return
        from hermes_cli.skills_hub import handle_skills_slash
        handle_skills_slash(cmd, ChatConsole())

    def _queue_prompt_turn(self, msg: str, command: str) -> None:
        """Inject ``msg`` onto the agent's input queue as the next normal user turn (the
        /learn, /plan, /init pattern: no engine, no model-tool footprint, prompt-cache safe)."""
        if hasattr(self, "_pending_input"):
            self._pending_input.put(msg)
        else:  # pragma: no cover - defensive (no live input loop)
            print(f"  {command} needs an active chat session to run.")

    def _handle_learn_command(self, cmd: str):
        """Handle /learn — distill a reusable skill from anything the user describes.

        Open-ended: the argument is free text describing the source(s) — a directory, a URL,
        "what we just did", pasted notes. The live agent gathers the material with the tools
        it already has and authors the skill via ``skill_manage``."""
        from agent.learn_prompt import build_learn_prompt

        user_request = _command_arg(cmd)
        if user_request:
            print("\n⚡ Learning a skill from what you described...")
        else:
            print("\n⚡ Learning a skill from this conversation...")
        self._queue_prompt_turn(build_learn_prompt(user_request), "/learn")

    def _handle_plan_command(self, cmd: str):
        """Handle /plan — write a markdown implementation plan, no execution. The live agent
        inspects the workspace read-only and saves the plan under ``.hermes/plans/``."""
        from agent.plan_prompt import build_plan_prompt

        task = _command_arg(cmd)  # optional — empty infers the task from conversation context
        if task:
            print(f"\n📋 Planning: {task[:80]}{'...' if len(task) > 80 else ''}")
        else:
            print("\n📋 Planning from this conversation's context...")
        self._queue_prompt_turn(build_plan_prompt(task), "/plan")

    def _handle_init_command(self, cmd: str):
        """Handle /init — generate or update AGENTS.md from a project scan performed by the
        live agent with its own read-only tools."""
        from hermes_cli.init_command import build_init_prompt_for_cwd

        msg = build_init_prompt_for_cwd(extra=_command_arg(cmd))  # optional user emphasis
        if "UPDATE the existing AGENTS.md" in msg:
            print("\n⚡ Updating AGENTS.md from a project scan...")
        else:
            print("\n⚡ Generating AGENTS.md from a project scan...")
        self._queue_prompt_turn(msg, "/init")

    def _handle_memory_command(self, cmd: str):
        """Handle /memory slash command — pending review + approval-gate toggle."""
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        parts = cmd.strip().split()
        args = parts[1:] if len(parts) > 1 else []
        store = getattr(self.agent, "_memory_store", None) if getattr(self, "agent", None) else None
        if store is None:
            # No live agent store (e.g. /memory approve from the Desktop GUI): apply against
            # a freshly loaded on-disk store, mirroring the gateway path. It persists to the
            # same MEMORY/USER.md and honors the user's configured char limits.
            from tools.memory_tool import load_on_disk_store
            store = load_on_disk_store()
        out = handle_pending_subcommand(
            wa.MEMORY, args,
            memory_store=store,
            set_mode_fn=lambda enabled: self._save_write_approval("memory", enabled),
        )
        if out is None:
            out = ("Unknown /memory subcommand. "
                   "Use: pending, approve <id>, reject <id>, approval <on|off>.")
        print(out)

    def _save_write_approval(self, subsystem: str, enabled: bool):
        """Persist <subsystem>.write_approval to config (for /memory|/skills approval)."""
        from cli import save_config_value
        save_config_value(f"{subsystem}.write_approval", bool(enabled))

    def _handle_background_command(self, cmd: str):
        """Handle /bg <prompt> — run a prompt in a separate background session (its own AIAgent
        on a thread); the result prints here without touching the active history."""
        from cli import AIAgent, _cprint, set_approval_callback, set_secret_capture_callback, set_sudo_password_callback
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /bg <prompt>")
            _cprint("  Example: /bg Summarize the top HN stories today")
            _cprint("  (For a side question about this conversation, use /btw <question>.)")
            _cprint("  The task runs in a separate session and results display here when done.")
            return

        prompt = parts[1].strip()
        self._background_task_counter += 1
        task_num = self._background_task_counter
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Make sure we have valid credentials
        if not self._ensure_runtime_credentials():
            _cprint("  (>_<) Cannot start background task: no valid credentials.")
            return

        _cprint(f"  🔄 Background task #{task_num} started: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        _cprint(f"  Task ID: {task_id}")
        _cprint("  You can continue chatting — results will appear when done.\n")

        turn_route = self._resolve_turn_agent_config(prompt)

        def run_background():
            set_sudo_password_callback(self._sudo_password_callback)
            set_approval_callback(self._approval_callback)
            try:
                set_secret_capture_callback(self._secret_capture_callback)
            except Exception:
                pass
            try:
                bg_agent = AIAgent(
                    model=turn_route["model"],
                    api_key=turn_route["runtime"].get("api_key"),
                    base_url=turn_route["runtime"].get("base_url"),
                    provider=turn_route["runtime"].get("provider"),
                    api_mode=turn_route["runtime"].get("api_mode"),
                    acp_command=turn_route["runtime"].get("command"),
                    acp_args=turn_route["runtime"].get("args"),
                    max_tokens=turn_route["runtime"].get("max_tokens"),
                    max_iterations=self.max_turns,
                    enabled_toolsets=self.enabled_toolsets,
                    quiet_mode=True,
                    verbose_logging=False,
                    session_id=task_id,
                    platform="cli",
                    session_db=self._session_db,
                    reasoning_config=self.reasoning_config,
                    service_tier=self.service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=self._providers_only,
                    providers_ignored=self._providers_ignore,
                    providers_order=self._providers_order,
                    provider_sort=self._provider_sort,
                    provider_require_parameters=self._provider_require_params,
                    provider_data_collection=self._provider_data_collection,
                    openrouter_min_coding_score=self._openrouter_min_coding_score,
                    fallback_model=self._fallback_model,
                )
                # Silence raw spinner; route thinking through TUI widget when no foreground agent is active.
                bg_agent._print_fn = lambda *_a, **_kw: None

                def _bg_thinking(text: str) -> None:
                    # Concurrent bg tasks may race on _spinner_text; acceptable for best-effort UI.
                    if not self._agent_running:
                        self._spinner_text = text
                        if self._app:
                            self._app.invalidate()

                bg_agent.thinking_callback = _bg_thinking

                result = bg_agent.run_conversation(
                    user_message=prompt,
                    task_id=task_id,
                )

                response = result.get("final_response", "") if result else ""
                if not response and result and result.get("error"):
                    response = f"Error: {result['error']}"

                _print_side_result_panel(
                    self,
                    header_lines=[
                        f"  ✅ Background task #{task_num} complete",
                        f"  Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"",
                    ],
                    body=response,
                    title_suffix=f"(background #{task_num})",
                    empty_note="  (No response generated)",
                )

                # Play bell if enabled
                if self.bell_on_complete:
                    sys.stdout.write("\a")
                    sys.stdout.flush()

            except Exception as e:
                # Same TUI refresh pattern as success path
                if self._app:
                    self._app.invalidate()
                    time.sleep(0.05)
                print()
                _cprint(f"  ❌ Background task #{task_num} failed: {e}")
            finally:
                try:
                    set_sudo_password_callback(None)
                    set_approval_callback(None)
                    set_secret_capture_callback(None)
                except Exception:
                    pass
                self._background_tasks.pop(task_id, None)
                # Clear spinner only if no foreground agent owns it
                if not self._agent_running:
                    self._spinner_text = ""
                if self._app:
                    self._invalidate(min_interval=0)

        thread = threading.Thread(target=run_background, daemon=True, name=f"bg-task-{task_id}")
        self._background_tasks[task_id] = thread
        thread.start()

    def _handle_btw_command(self, cmd: str):
        """Handle /btw <question> — answer a side question about this conversation from a
        history snapshot via a one-shot auxiliary call. The live session is never touched
        (no history mutation, no role-alternation risk, no cache invalidation)."""
        from cli import _cprint

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /btw <question>")
            _cprint("  Example: /btw which file was that error in?")
            _cprint("  Answers a quick question about this conversation without interrupting it.")
            _cprint("  (For an independent background task, use /bg <prompt>.)")
            return

        question = parts[1].strip()

        if not self._ensure_runtime_credentials():
            _cprint("  (>_<) Cannot answer side question: no valid credentials.")
            return

        # Snapshot NOW, on the UI thread — the foreground turn keeps appending
        # to conversation_history while the worker runs.
        history_snapshot = list(self.conversation_history or [])
        # Live agent → cache-parity fork (full context, warm cache reads).
        parent_agent = self.agent
        turn_route = self._resolve_turn_agent_config(question)
        main_runtime = {
            "model": turn_route["model"],
            "provider": turn_route["runtime"].get("provider"),
            "base_url": turn_route["runtime"].get("base_url"),
            "api_key": turn_route["runtime"].get("api_key"),
            "api_mode": turn_route["runtime"].get("api_mode"),
        }

        preview = question[:60] + ("..." if len(question) > 60 else "")
        _cprint(f"  💬 Side question: \"{preview}\"")
        _cprint("  Answering from a snapshot of this conversation — the current work continues.\n")

        def run_side_question():
            try:
                from agent.side_question import answer_side_question
                answer = answer_side_question(
                    question,
                    history_snapshot,
                    parent_agent=parent_agent,
                    main_runtime=main_runtime,
                )
                _print_side_result_panel(
                    self,
                    header_lines=[f"  💬 /btw: \"{preview}\""],
                    body=answer,
                    title_suffix="(btw)",
                    empty_note="  (No answer generated)",
                )
            except Exception as e:
                if self._app:
                    self._app.invalidate()
                    time.sleep(0.05)
                print()
                _cprint(f"  ❌ /btw failed: {e}")
            finally:
                if self._app:
                    self._invalidate(min_interval=0)

        threading.Thread(target=run_side_question, daemon=True, name="btw-side-question").start()

    def _handle_bundles_command(self, cmd: str) -> None:
        """In-session ``/bundles`` — show installed skill bundles (``hermes bundles list`` rendered
        inside the running CLI). Bundles are loaded via ``/<bundle-name>``."""
        from cli import ChatConsole, _BOLD, _DIM, _RST, _accent_hex, _cprint
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("bundles", CommandContext(surface="cli"))
        if "error" in reply.data:
            _cprint(f"\033[1;31mBundle subsystem unavailable: {reply.data['error']}{_RST}")
            return

        bundles = reply.data["bundles"]
        if not bundles:
            _cprint("  No skill bundles installed.")
            _cprint(
                f"  {_DIM}Create one with: hermes bundles create "
                f"<name> --skill <s1> --skill <s2>{_RST}"
            )
            _cprint(f"  {_DIM}Directory: {reply.data['dir']}{_RST}")
            return

        _cprint(f"\n  ▣ {_BOLD}Skill Bundles{_RST} ({len(bundles)} installed):")
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or f"Load {skill_count} skills"
            ChatConsole().print(
                f"    [bold {_accent_hex()}]/{info['slug']:<20}[/] "
                f"[dim]-[/] {_escape(desc)} [dim]({skill_count} skills)[/]"
            )
            for s in info.get("skills", []):
                ChatConsole().print(f"        [dim]· {_escape(s)}[/]")
        _cprint(
            f"\n  {_DIM}Invoke a bundle with /<slug>. "
            f"Manage with `hermes bundles`.{_RST}"
        )

    def _handle_browser_command(self, cmd: str):
        """Handle /browser connect|disconnect|status|use — manage the live Chromium-family CDP connection."""
        parts = cmd.strip().split(None, 1)
        sub = parts[1].lower().strip() if len(parts) > 1 else "status"
        if sub == "use" or sub.startswith("use "):
            _browser_use(self, sub.split(None, 1)[1].strip() if " " in sub else "on")
        elif sub.startswith("connect"):
            connect_parts = cmd.strip().split(None, 2)  # ["/browser", "connect", "ws://..."]
            _browser_connect(self, connect_parts[2].strip() if len(connect_parts) > 2 else DEFAULT_BROWSER_CDP_URL)
        elif sub == "disconnect":
            _browser_disconnect(self)
        elif sub == "status":
            _browser_status()
        else:
            _say_block(
                "Usage: /browser connect|disconnect|status|use",
                "",
                "   connect      Connect browser tools to your live Chromium-family browser session",
                "   disconnect   Revert to default browser backend",
                "   status       Show current browser mode",
                "   use [off]    Switch to Browser Use mode (CLI 3.0) / back to built-in tools",
            )

    def _handle_heartbeat_command(self, cmd: str) -> None:
        """Dispatch /heartbeat: set / status / pause / resume / clear. ``/heartbeat every 10m <prompt>``
        sets the session's one recurring instruction, injected as a normal user turn when due.
        Session-scoped and in-process — use `hermes cron` for durable schedules."""
        from cli import _DIM, _RST, _cprint
        from hermes_cli.heartbeat import parse_interval, format_interval

        parts = (cmd or "").strip().split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        lower = arg.lower()

        mgr = self._get_heartbeat_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Heartbeats unavailable (no active session).{_RST}")
            return

        if not arg or lower == "status":
            _cprint(f"  {mgr.status_line()}")
            return

        if lower == "pause":
            state = mgr.pause()
            if state is None:
                _cprint(f"  {_DIM}No heartbeat set.{_RST}")
            else:
                _cprint(f"  ⏸ Heartbeat paused: {state.prompt}")
            return

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                _cprint(f"  {_DIM}No heartbeat to resume.{_RST}")
            else:
                self._start_heartbeat_watchdog()
                _cprint(f"  ▶ Heartbeat resumed (every {format_interval(state.interval_seconds)}): {state.prompt}")
            return

        if lower in {"clear", "stop", "off"}:
            if mgr.clear():
                _cprint("  ✓ Heartbeat cleared.")
            else:
                _cprint(f"  {_DIM}No heartbeat set.{_RST}")
            return

        # Set: `/heartbeat every 10m <prompt>` (also accepts `10m <prompt>`).
        tokens = arg.split(None, 2)
        interval = None
        prompt = ""
        if tokens and tokens[0].lower() == "every" and len(tokens) >= 2:
            interval = parse_interval(f"every {tokens[1]}")
            prompt = tokens[2] if len(tokens) > 2 else ""
        elif tokens:
            interval = parse_interval(tokens[0])
            prompt = arg[len(tokens[0]):].strip() if interval and interval > 0 else ""

        if interval is None:
            _cprint("  Usage: /heartbeat every <interval> <prompt>   (e.g. /heartbeat every 10m Check CI)")
            _cprint(f"  {_DIM}Also: /heartbeat status | pause | resume | clear{_RST}")
            return
        if interval < 0:
            from hermes_cli.heartbeat import MIN_INTERVAL_SECONDS
            _cprint(f"  Interval too small — minimum is {MIN_INTERVAL_SECONDS}s.")
            return
        if not prompt.strip():
            _cprint("  Usage: /heartbeat every <interval> <prompt> — the prompt is required.")
            return

        try:
            state = mgr.set(prompt, interval)
        except ValueError as exc:
            _cprint(f"  Invalid heartbeat: {exc}")
            return
        self._start_heartbeat_watchdog()
        _cprint(f"  ♥ Heartbeat set (every {format_interval(state.interval_seconds)}): {state.prompt}")
        _cprint(
            f"  {_DIM}Fires as a normal turn whenever the session is idle and the "
            f"interval has elapsed. /heartbeat pause | resume | clear to manage; "
            f"lives only while this Hermes process runs — use `hermes cron` for "
            f"durable schedules.{_RST}"
        )

    def _handle_refine_command(self, cmd: str) -> None:
        """Dispatch /refine — run the memory/skill review fork on demand (same machinery as the
        automatic post-turn ``_spawn_background_review``), with optional focus text. Background
        fork; the live conversation and prompt cache are never touched."""
        from cli import _DIM, _RST, _cprint

        parts = (cmd or "").strip().split(None, 1)
        focus = parts[1].strip() if len(parts) > 1 else ""

        agent = getattr(self, "agent", None)
        if agent is None:
            _cprint(f"  {_DIM}Nothing to refine yet — send a message first.{_RST}")
            return

        snapshot = list(getattr(self, "conversation_history", None) or [])
        if not snapshot:
            _cprint(f"  {_DIM}Nothing to refine yet — the conversation is empty.{_RST}")
            return

        review_skills = "skill_manage" in getattr(agent, "valid_tool_names", set())
        try:
            agent._spawn_background_review(
                messages_snapshot=snapshot,
                review_memory=True,
                review_skills=review_skills,
                focus=focus or None,
                explicit=True,
            )
        except Exception as exc:
            _cprint(f"  /refine failed to start: {exc}")
            return
        tail = f" (focus: {focus})" if focus else ""
        _cprint(
            f"  ⚗ Reviewing this conversation in the background{tail} — "
            f"any memory/skill updates will be reported when done."
        )

    def _handle_review_command(self, cmd: str) -> None:
        """Dispatch /review — snapshot the last N messages (+ argument text as instructions) and
        spawn an independent reviewer subagent via async delegation; the review re-enters this
        session as a normal delegation completion."""
        from cli import _DIM, _RST, _cprint

        parts = (cmd or "").strip().split(None, 1)
        prompt = parts[1].strip() if len(parts) > 1 else ""

        agent = getattr(self, "agent", None)
        if agent is None:
            _cprint(f"  {_DIM}Nothing to review yet — send a message first.{_RST}")
            return

        snapshot = list(getattr(self, "conversation_history", None) or [])
        try:
            from agent.review_engine import format_dispatch_note, start_review

            result = start_review(agent, snapshot, prompt)
        except ValueError as exc:
            _cprint(f"  {_DIM}{exc}{_RST}")
            return
        except Exception as exc:
            _cprint(f"  /review failed to start: {exc}")
            return
        _cprint(f"  {format_dispatch_note(result, prompt)}")

    def _handle_goal_command(self, cmd: str) -> None:
        """Dispatch /goal subcommands: set / draft / show / gate / wait / status / pause / resume / clear."""
        from cli import _DIM, _RST, _cprint
        arg = _command_arg(cmd)

        mgr = self._get_goal_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Goals unavailable (no active session).{_RST}")
            return

        lower = arg.lower()
        verb, _, rest = arg.partition(" ")
        verb = verb.lower()
        rest = rest.strip()

        if not arg or lower == "status":
            _cprint(f"  {mgr.status_line()}")
        elif lower == "show":
            _cprint(f"  {mgr.status_line()}")
            _cprint(f"  {mgr.render_contract()}")
        elif lower.startswith("draft"):
            # Expand plain text into a structured completion contract so "done" is
            # evidence-based instead of a vibe check.
            objective = arg[len("draft"):].strip()
            if not objective:
                _cprint("  Usage: /goal draft <objective in plain language>")
                return
            self._handle_goal_draft(objective)
        elif lower == "pause":
            state = mgr.pause(reason="user-paused")
            _cprint(f"  ⏸ Goal paused: {state.goal}" if state else f"  {_DIM}No goal set.{_RST}")
        elif lower == "resume":
            self._goal_resume(mgr)
        elif lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            _cprint("  ✓ Goal cleared." if had else f"  {_DIM}No active goal.{_RST}")
        elif verb == "wait":
            self._goal_wait(mgr, rest)
        elif lower == "unwait":
            if mgr.stop_waiting():
                _cprint("  ▶ Wait barrier cleared — goal loop resumes.")
            else:
                _cprint(f"  {_DIM}No wait barrier set.{_RST}")
        elif verb == "gate":
            self._goal_gate(mgr, rest)
        else:
            self._goal_set(mgr, arg)

    def _goal_resume(self, mgr) -> None:
        from cli import _DIM, _RST, _cprint
        state = mgr.resume()
        if state is None:
            _cprint(f"  {_DIM}No goal to resume.{_RST}")
            return
        _cprint(f"  ▶ Goal resumed: {state.goal}")
        # Resume must restart work, not just flip state: queue the continuation prompt
        # the same way /goal <text> queues its kickoff.
        prompt = mgr.next_continuation_prompt()
        queued = False
        if prompt:
            try:
                self._pending_input.put(prompt)
                queued = True
            except Exception:
                pass
        if queued:
            _cprint(f"  {_DIM}Continuing now — taking the next step.{_RST}")
        else:
            _cprint(f"  {_DIM}Send any message to kick off the next step.{_RST}")

    def _goal_wait(self, mgr, wait_arg: str) -> None:
        """/goal wait <pid> [reason] — park the loop on a background process (CI / build);
        the barrier auto-clears when the PID exits."""
        from cli import _cprint
        if not wait_arg:
            _cprint("  Usage: /goal wait <pid> [reason]")
            return
        wtokens = wait_arg.split(None, 1)
        try:
            pid = int(wtokens[0])
        except ValueError:
            _cprint("  /goal wait: <pid> must be an integer process id.")
            return
        reason = wtokens[1].strip() if len(wtokens) > 1 else ""
        try:
            mgr.wait_on(pid, reason=reason)
        except (RuntimeError, ValueError) as exc:
            _cprint(f"  /goal wait: {exc}")
            return
        rtxt = f" ({reason})" if reason else ""
        _cprint(f"  ⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits.")

    def _goal_gate(self, mgr, gate_arg: str) -> None:
        """/goal gate [list | add <command> | remove <N> | clear] — shell commands that must pass
        before the judge may declare the goal done; a failing gate's output becomes the
        continuation prompt."""
        from cli import _cprint
        gate_lower = gate_arg.lower()
        if not gate_arg or gate_lower == "list":
            for line in mgr.render_gates().splitlines():
                _cprint(f"  {line}")
            return
        if gate_lower.startswith("add "):
            command = gate_arg[len("add"):].strip()
            try:
                gate = mgr.add_gate(command)
            except (RuntimeError, ValueError) as exc:
                _cprint(f"  /goal gate add: {exc}")
                return
            _cprint(
                f"  ⚿ Gate added: $ {gate.command} "
                f"({gate.max_retries} retries, {gate.timeout_seconds}s timeout). "
                f"It must pass before the goal can complete."
            )
            return
        if gate_lower.startswith("remove ") or gate_lower.startswith("rm "):
            idx_text = gate_arg.split(None, 1)[1].strip()
            try:
                removed = mgr.remove_gate(int(idx_text))
            except (RuntimeError, ValueError, IndexError) as exc:
                _cprint(f"  /goal gate remove: {exc}")
                return
            _cprint(f"  ✓ Gate removed: $ {removed}")
            return
        if gate_lower == "clear":
            try:
                prev = mgr.clear_gates()
            except RuntimeError as exc:
                _cprint(f"  /goal gate clear: {exc}")
                return
            _cprint(f"  ✓ Cleared {prev} gate{'s' if prev != 1 else ''}.")
            return
        _cprint("  Usage: /goal gate [list | add <command> | remove <N> | clear]")

    def _goal_set(self, mgr, arg: str) -> None:
        """Set the goal from free text; inline `verify:`/`constraints:`/`boundaries:`/`stop when:`
        lines become a completion contract, the remaining prose the headline. Kicks the loop off."""
        from cli import _DIM, _RST, _cprint
        from hermes_cli.goals import parse_contract

        headline, contract = parse_contract(arg)
        goal_text = headline or arg
        try:
            state = mgr.set(goal_text, contract=contract if not contract.is_empty() else None)
        except ValueError as exc:
            _cprint(f"  Invalid goal: {exc}")
            return

        _cprint(f"  ⊙ Goal set ({state.max_turns}-turn budget): {state.goal}")
        if state.has_contract():
            _cprint(f"  {_DIM}Completion contract:{_RST}")
            for line in state.contract.render_block().splitlines():
                _cprint(f"    {line}")
        _cprint(
            f"  {_DIM}After each turn, a judge model checks if the goal is done"
            f"{' against the contract above' if state.has_contract() else ''}. "
            f"Hermes keeps working until it is, you pause/clear it, or the budget is "
            f"exhausted. Use /goal status, /goal show, /goal pause, /goal resume, /goal clear.{_RST}"
        )
        # Kick the loop off immediately so the user doesn't have to send a separate message.
        try:
            self._pending_input.put(state.goal)
        except Exception:
            pass

    def _handle_goal_draft(self, objective: str) -> None:
        """Draft a structured completion contract from a plain objective and
        set it as the active goal. Falls back to a bare goal if the aux model
        can't produce a contract."""
        from cli import _DIM, _RST, _cprint
        from hermes_cli.goals import draft_contract

        mgr = self._get_goal_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Goals unavailable (no active session).{_RST}")
            return

        _cprint(f"  {_DIM}Drafting completion contract…{_RST}")
        try:
            contract = draft_contract(objective)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).debug("goal draft failed: %s", exc)
            contract = None

        try:
            state = mgr.set(objective, contract=contract)
        except ValueError as exc:
            _cprint(f"  Invalid goal: {exc}")
            return

        _cprint(f"  ⊙ Goal set ({state.max_turns}-turn budget): {state.goal}")
        if state.has_contract():
            _cprint(f"  {_DIM}Drafted completion contract:{_RST}")
            for line in state.contract.render_block().splitlines():
                _cprint(f"    {line}")
            _cprint(
                f"  {_DIM}Tighten any field by re-setting the goal with inline "
                f"lines (e.g. verify: <command>), then /goal resume. "
                f"Use /goal show to review.{_RST}"
            )
        else:
            _cprint(
                f"  {_DIM}Couldn't draft a contract (aux model unavailable) — "
                f"running as a free-form goal. The per-turn judge still applies.{_RST}"
            )
        try:
            self._pending_input.put(state.goal)
        except Exception:
            pass

    def _handle_loop_command(self, cmd: str) -> None:
        """Dispatch /loop — recurring in-session wakeups: ``/loop [interval] <prompt> [--times N]
        [--until <cond>]`` starts one; ``status | pause | resume | stop`` control it."""
        from cli import _DIM, _RST, _cprint
        parts = (cmd or "").strip().split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        mgr = self._get_loop_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Loops unavailable (no active session).{_RST}")
            return

        from hermes_cli.loops import dispatch_loop_command

        result = dispatch_loop_command(mgr, arg)
        for line in (result.get("output") or "").splitlines():
            _cprint(f"  {line}")
        if result.get("created"):
            try:
                from hermes_cli.loops import goal_blocks_loop_tick

                if goal_blocks_loop_tick(mgr.session_id):
                    _cprint(
                        f"  {_DIM}Note: an active /goal is driving this session — "
                        f"loop wakeups defer until the goal finishes, pauses, or parks.{_RST}"
                    )
            except Exception:
                pass

    def _handle_subgoal_command(self, cmd: str) -> None:
        """Dispatch /subgoal: bare → show, ``<text>`` → append, ``remove <n>`` (1-based), ``clear``.

        Subgoals are extra criteria added mid-loop; they join both the judge prompt and the
        continuation prompt at the next turn boundary (no special kick)."""
        from cli import _DIM, _RST, _cprint
        parts = (cmd or "").strip().split(None, 2)
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        mgr = self._get_goal_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Goals unavailable (no active session).{_RST}")
            return

        if not mgr.has_goal():
            _cprint(f"  {_DIM}No active goal. Set one with /goal <text>.{_RST}")
            return

        # No args → list current subgoals.
        if not arg:
            _cprint(f"  {mgr.status_line()}")
            _cprint(f"  {mgr.render_subgoals()}")
            return

        tokens = arg.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        if verb == "remove":
            if not rest:
                _cprint("  Usage: /subgoal remove <n>")
                return
            try:
                idx = int(rest.split()[0])
            except ValueError:
                _cprint("  /subgoal remove: <n> must be an integer (1-based index).")
                return
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as exc:
                _cprint(f"  /subgoal remove: {exc}")
                return
            _cprint(f"  ✓ Removed subgoal {idx}: {removed}")
            return

        if verb == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as exc:
                _cprint(f"  /subgoal clear: {exc}")
                return
            if prev:
                _cprint(f"  ✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}.")
            else:
                _cprint(f"  {_DIM}No subgoals to clear.{_RST}")
            return

        # Otherwise — append the whole arg as a new subgoal.
        try:
            text = mgr.add_subgoal(arg)
        except (ValueError, RuntimeError) as exc:
            _cprint(f"  /subgoal: {exc}")
            return
        idx = len(mgr.state.subgoals) if mgr.state else 0
        _cprint(f"  ✓ Added subgoal {idx}: {text}")

    def _handle_skin_command(self, cmd: str):
        """Handle /skin [name] — show or change the display skin."""
        from cli import _ACCENT, save_config_value
        try:
            from hermes_cli.skin_engine import list_skins, set_active_skin, get_active_skin_name
        except ImportError:
            print("Skin engine not available.")
            return

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            # Show current skin and list available
            current = get_active_skin_name()
            skins = list_skins()
            print(f"\n  Current skin: {current}")
            print("  Available skins:")
            for s in skins:
                marker = " ●" if s["name"] == current else "  "
                source = f" ({s['source']})" if s["source"] == "user" else ""
                print(f"   {marker} {s['name']}{source} — {s['description']}")
            print("\n  Usage: /skin <name>")
            print(f"  Custom skins: drop a YAML file in {display_hermes_home()}/skins/\n")
            return

        new_skin = parts[1].strip().lower()
        available = {s["name"] for s in list_skins()}
        if new_skin not in available:
            print(f"  Unknown skin: {new_skin}")
            print(f"  Available: {', '.join(sorted(available))}")
            return

        set_active_skin(new_skin)
        _ACCENT.reset()  # Re-resolve ANSI color for the new skin
        # _DIM is now a fixed dim+italic ANSI escape (terminal-default fg)
        # so it doesn't need re-resolving on skin switch.
        if save_config_value("display.skin", new_skin):
            print(f"  Skin set to: {new_skin} (saved)")
        else:
            print(f"  Skin set to: {new_skin}")
        print("  Note: banner colors will update on next session start.")
        if self._apply_tui_skin_style():
            print("  Prompt + TUI colors updated.")

    def _compose_in_editor(self, initial_text: str = "") -> str:
        """Open ``$VISUAL``/``$EDITOR`` on a temp markdown file and return the saved buffer with
        ``#!`` comment lines stripped; "" if the editor failed or the buffer was left empty.
        Factored out so the read-back/strip logic is unit-testable."""
        import os
        import shlex
        import subprocess
        import tempfile

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            editor = "notepad" if os.name == "nt" else "nano"

        header = (
            "#! Compose your prompt below. Lines starting with '#!' are ignored.\n"
            "#! Save and quit to send; leave empty to cancel.\n\n"
        )
        fd, path = tempfile.mkstemp(suffix=".md", prefix="hermes_prompt_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(header)
                if initial_text:
                    fh.write(initial_text)
            try:
                subprocess.call([*shlex.split(editor), path])
            except Exception:
                # Fall back to a bare invocation (editor value may not be a
                # simple argv-splittable string on some platforms).
                subprocess.call(f"{editor} {shlex.quote(path)}", shell=True)
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        lines = [ln for ln in raw.splitlines() if not ln.startswith("#!")]
        return "\n".join(lines).strip()

    def _handle_prompt_compose_command(self, cmd_original: str) -> None:
        """Handle /prompt — compose the next prompt in $EDITOR (optionally seeded with the argument)
        and queue it as the next agent turn via the one-shot ``_pending_agent_seed`` (as /blueprint)."""
        from cli import _DIM, _RST, _cprint

        initial = ""
        parts = (cmd_original or "").strip().split(None, 1)
        if len(parts) > 1:
            initial = parts[1]

        try:
            composed = self._compose_in_editor(initial)
        except Exception as exc:
            _cprint(f"  {_DIM}(>_<) Could not open editor: {exc}{_RST}")
            return

        if not composed:
            _cprint(f"  {_DIM}(._.) Empty prompt — nothing sent.{_RST}")
            return

        # One-shot seed: the interactive loop runs this as the next agent turn
        # right after process_command() returns (see cli.py main loop).
        self._pending_agent_seed = composed

    def _handle_focus_command(self, cmd_original: str) -> None:
        """Toggle or inspect focus view (``/focus [on|off|status]``) — a DISPLAY-ONLY reduced-output mode.

        Composes with the existing ``/verbose`` machinery instead of adding a second suppression
        path: turning it on stashes the user's tool_progress_mode and snaps it to "off" (the value
        ``agent/tool_executor.py`` gates on); turning it off restores the stash verbatim. Adds a
        per-turn hidden-line count with a recovery hint and a ``focus`` status-bar segment. Never
        touches history, system prompt, or request payloads."""
        from cli import _cprint, save_config_value
        from hermes_cli.colors import Colors as _Colors
        from hermes_cli.focus_view import (
            FOCUS_CONFIG_KEY,
            FOCUS_TOOL_PROGRESS_MODE,
            format_focus_status,
            format_focus_toggle_message,
            normalize_tool_progress_mode,
            resolve_focus_arg,
        )

        arg = _command_arg(cmd_original)
        current = bool(getattr(self, "_focus_view_enabled", False))
        action, target = resolve_focus_arg(arg, current)

        if action == "usage":
            _cprint("  Usage: /focus [on|off|status]")
            return

        # The mode /focus off restores: while focus is ON the live mode is "off", so use the stash.
        restore_mode = normalize_tool_progress_mode(
            getattr(self, "_focus_saved_tool_progress", None)
            if current
            else getattr(self, "tool_progress_mode", "all")
        )

        if action == "status":
            body = format_focus_status(current, restore_mode)
            head, _, tail = body.partition("\n")
            label, _, rest = head.partition(":")
            state_color = _Colors.GREEN if current else _Colors.DIM
            _cprint(
                f"  {_Colors.BOLD}{label}:{_Colors.RESET}"
                f"{state_color}{rest}{_Colors.RESET}"
                + (f"\n{_Colors.DIM}  {tail.strip()}{_Colors.RESET}" if tail else "")
            )
            return

        if target == current:
            # Idempotent explicit set — report without rewriting config.
            _cprint(f"  {format_focus_toggle_message(current, restore_mode)}")
            return

        if target:
            # Stash the user's configured mode, then reuse the EXISTING
            # suppression path by snapping to "off".
            self._focus_saved_tool_progress = restore_mode
            self._set_tool_progress_mode(FOCUS_TOOL_PROGRESS_MODE)
        else:
            self._set_tool_progress_mode(restore_mode)
            self._focus_saved_tool_progress = None

        self._focus_view_enabled = bool(target)
        self._focus_hidden_lines = 0
        save_config_value(FOCUS_CONFIG_KEY, bool(target))

        state = (
            f"{_Colors.GREEN}enabled{_Colors.RESET}" if target
            else f"{_Colors.DIM}disabled{_Colors.RESET}"
        )
        message = format_focus_toggle_message(bool(target), restore_mode)
        # Re-colour just the enabled/disabled word so the line matches siblings.
        for word in ("enabled", "disabled"):
            if word in message:
                message = message.replace(word, state, 1)
                break
        _cprint(f"  {message}")

    def _set_tool_progress_mode(self, mode: str) -> None:
        """Set the live tool-progress mode on both the CLI and the agent (one write path for
        /focus and /verbose — the agent copy is what ``agent/tool_executor.py`` gates on)."""
        from hermes_cli.focus_view import normalize_tool_progress_mode

        normalized = normalize_tool_progress_mode(mode)
        self.tool_progress_mode = normalized
        agent = getattr(self, "agent", None)
        if agent is not None:
            try:
                agent.tool_progress_mode = normalized
            except Exception:
                pass

    def _note_focus_hidden_line(self, function_name: str) -> None:
        """Count one tool line focus view is suppressing this turn — against the mode the user had
        BEFORE focus snapped to "off", so a prior ``/verbose off`` user is never told focus hid lines."""
        if not getattr(self, "_focus_view_enabled", False):
            return
        from hermes_cli.focus_view import would_display_tool_line

        saved = getattr(self, "_focus_saved_tool_progress", None)
        last = getattr(self, "_focus_last_counted_tool", None)
        if not would_display_tool_line(saved, function_name, last):
            return
        self._focus_last_counted_tool = function_name
        self._focus_hidden_lines = int(getattr(self, "_focus_hidden_lines", 0)) + 1

    def _emit_focus_recovery_line(self) -> None:
        """Print the dim post-turn recovery line and reset the counter."""
        count = int(getattr(self, "_focus_hidden_lines", 0) or 0)
        self._focus_hidden_lines = 0
        self._focus_last_counted_tool = None
        if not getattr(self, "_focus_view_enabled", False):
            return
        from hermes_cli.focus_view import format_hidden_line

        line = format_hidden_line(count)
        if not line:
            return
        try:
            from cli import _DIM, _RST, _cprint

            _cprint(f"  {_DIM}{line}{_RST}")
        except Exception:
            pass

    def _handle_approvals_command(self, cmd_original: str) -> None:
        """Show or persist the profile-wide dangerous-command approval mode."""
        from cli import _cprint
        from hermes_cli.approval_mode import run_approval_mode_command

        parts = (cmd_original or "").strip().split(None, 1)
        requested = parts[1] if len(parts) > 1 else None
        result = run_approval_mode_command(requested)
        _cprint(f"  {result.message}")

    def _handle_footer_command(self, cmd_original: str) -> None:
        """Toggle or inspect ``display.runtime_footer.enabled`` (``/footer [on|off|status]``)."""
        from cli import _cprint, save_config_value
        from hermes_cli.config import load_config
        from hermes_cli.colors import Colors as _Colors

        arg = _command_arg(cmd_original, lower=True)
        cfg = load_config() or {}
        footer_cfg = ((cfg.get("display") or {}).get("runtime_footer") or {})
        current = bool(footer_cfg.get("enabled", False))
        fields = footer_cfg.get("fields") or ["model", "context_pct", "cwd"]

        new_state = _toggle_target(arg, current)
        if new_state == "status":
            state = "ON" if current else "OFF"
            _cprint(
                f"  {_Colors.BOLD}Runtime footer:{_Colors.RESET} {state}\n"
                f"  Fields: {', '.join(fields)}"
            )
            return
        if new_state is None:
            _cprint("  Usage: /footer [on|off|status]")
            return

        if save_config_value("display.runtime_footer.enabled", new_state):
            state = (
                f"{_Colors.GREEN}ON{_Colors.RESET}" if new_state
                else f"{_Colors.DIM}OFF{_Colors.RESET}"
            )
            _cprint(f"  Runtime footer: {state}")
        else:
            _cprint("  Failed to save runtime_footer setting to config.yaml")

    def _handle_timestamps_command(self, cmd_original: str) -> None:
        """Toggle or inspect ``display.timestamps`` (``/timestamps [on|off|status]``). When on,
        message labels carry an ``[HH:MM]`` suffix and ``/history`` prefixes stored-timestamp turns."""
        from cli import _cprint, save_config_value
        from hermes_cli.colors import Colors as _Colors

        arg = _command_arg(cmd_original, lower=True)
        current = bool(getattr(self, "show_timestamps", False))

        new_state = _toggle_target(arg, current)
        if new_state == "status":
            state = "ON" if current else "OFF"
            _cprint(f"  {_Colors.BOLD}Message timestamps:{_Colors.RESET} {state}")
            return
        if new_state is None:
            _cprint("  Usage: /timestamps [on|off|status]")
            return

        self.show_timestamps = new_state
        if save_config_value("display.timestamps", new_state):
            state = (
                f"{_Colors.GREEN}ON{_Colors.RESET}" if new_state
                else f"{_Colors.DIM}OFF{_Colors.RESET}"
            )
            _cprint(f"  Message timestamps: {state}")
        else:
            _cprint("  Failed to save timestamps setting to config.yaml")

    def _handle_reasoning_command(self, cmd: str):
        """Handle /reasoning — manage effort level and display toggle.

        Usage:
            /reasoning                    Show effort level and display state
            /reasoning <level> [--global] Set effort for this session (none, minimal, low, medium,
                                          high, xhigh, max, ultra); --global persists to config.yaml
            /reasoning show|hide          Show / hide model thinking in output
            /reasoning full|clamp         Complete thinking vs. first-10-lines clamp"""
        from cli import CLI_CONFIG, _ACCENT, _DIM, _RST, _cprint, _parse_reasoning_config, save_config_value
        parts = cmd.strip().split(maxsplit=1)

        if len(parts) < 2:
            # Show current state
            rc = self.reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "none (disabled)"
            else:
                level = rc.get("effort", "medium")
            display_state = "on ✓" if self.show_reasoning else "off"
            full_state = "full" if getattr(self, "reasoning_full", False) else "clamped to 10 lines"
            _cprint(f"  {_ACCENT}Reasoning effort:  {level}{_RST}")
            _cprint(f"  {_ACCENT}Reasoning display: {display_state} ({full_state}){_RST}")
            _cprint(f"  {_DIM}Usage: /reasoning <none|minimal|low|medium|high|xhigh|max|ultra|show|hide|full|clamp> [--global]{_RST}")
            return

        arg, explicit_global = _split_scope_flags(parts[1])

        # Display toggle
        if arg in {"show", "on"}:
            self.show_reasoning = True
            if self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            save_config_value("display.show_reasoning", True)
            _cprint(f"  {_ACCENT}✓ Reasoning display: ON (saved){_RST}")
            _cprint(f"  {_DIM}  Model thinking will be shown during and after each response.{_RST}")
            return
        if arg in {"hide", "off"}:
            self.show_reasoning = False
            if self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            save_config_value("display.show_reasoning", False)
            _cprint(f"  {_ACCENT}✓ Reasoning display: OFF (saved){_RST}")
            return

        # Full / clamped recap toggle
        if arg in {"full", "all"}:
            self.reasoning_full = True
            save_config_value("display.reasoning_full", True)
            _cprint(f"  {_ACCENT}✓ Reasoning display: FULL (saved){_RST}")
            _cprint(f"  {_DIM}  The post-response recap box will print complete thinking.{_RST}")
            if not self.show_reasoning:
                _cprint(f"  {_DIM}  Note: reasoning display is OFF — run /reasoning show to see it.{_RST}")
            return
        if arg in {"clamp", "collapse", "short"}:
            self.reasoning_full = False
            save_config_value("display.reasoning_full", False)
            _cprint(f"  {_ACCENT}✓ Reasoning display: CLAMPED to 10 lines (saved){_RST}")
            return

        # Effort level change
        parsed = _parse_reasoning_config(arg)
        if parsed is None:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Valid levels: none, minimal, low, medium, high, xhigh, max, ultra{_RST}")
            _cprint(f"  {_DIM}Display:      show, hide{_RST}")
            _cprint(f"  {_DIM}Scope:        session-scoped by default, --global to persist{_RST}")
            return

        self.reasoning_config = parsed
        self.agent = None  # Force agent re-init with new reasoning config

        saved = explicit_global and save_config_value("agent.reasoning_effort", arg)
        if saved:
            agent_cfg = CLI_CONFIG.get("agent")
            if not isinstance(agent_cfg, dict):
                agent_cfg = {}
                CLI_CONFIG["agent"] = agent_cfg
            agent_cfg["reasoning_effort"] = arg
        _cprint(f"  {_ACCENT}✓ Reasoning effort set to '{arg}' {_scope_outcome(explicit_global, saved)}{_RST}")

    def _handle_busy_command(self, cmd: str):
        """Handle /busy [status|queue|steer|interrupt] — what Enter does while Hermes is working."""
        from cli import _ACCENT, _DIM, _RST, _cprint, save_config_value
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or parts[1].strip().lower() == "status":
            _cprint(f"  {_ACCENT}Busy input mode: {self.busy_input_mode}{_RST}")
            _behavior = _BUSY_MODE_SHORT.get(self.busy_input_mode, _BUSY_MODE_SHORT["interrupt"])
            _cprint(f"  {_DIM}Enter while busy: {_behavior}{_RST}")
            _cprint(f"  {_DIM}Usage: /busy [queue|steer|interrupt|status]{_RST}")
            return

        arg = parts[1].strip().lower()
        if arg not in _BUSY_MODE_LONG:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Usage: /busy [queue|steer|interrupt|status]{_RST}")
            return

        self.busy_input_mode = arg
        if save_config_value("display.busy_input_mode", arg):
            _cprint(f"  {_ACCENT}✓ Busy input mode set to '{arg}' (saved to config){_RST}")
            _cprint(f"  {_DIM}{_BUSY_MODE_LONG[arg]}{_RST}")
        else:
            _cprint(f"  {_ACCENT}✓ Busy input mode set to '{arg}' (session only){_RST}")

    def _handle_indicator_command(self, cmd: str):
        """Handle /indicator [status|kaomoji|emoji|unicode|ascii] — pick the TUI busy-indicator style.
        Persists to ``display.tui_status_indicator`` (the key the TUI reads) for its next render."""
        from cli import _ACCENT, _DIM, _RST, _cprint, save_config_value
        from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES
        styles = INDICATOR_STYLES
        current = (
            (self.config.get("display") or {}).get("tui_status_indicator", DEFAULT_INDICATOR_STYLE)
        )

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or parts[1].strip().lower() == "status":
            _cprint(f"  {_ACCENT}Busy-indicator style: {current}{_RST}")
            _cprint(f"  {_DIM}Usage: /indicator [{'|'.join(styles)}]{_RST}")
            return

        arg = parts[1].strip().lower()
        if arg not in styles:
            _cprint(f"  {_DIM}(._.) Unknown indicator style: {arg}{_RST}")
            _cprint(f"  {_DIM}Usage: /indicator [{'|'.join(styles)}]{_RST}")
            return

        self.config.setdefault("display", {})["tui_status_indicator"] = arg
        if save_config_value("display.tui_status_indicator", arg):
            _cprint(f"  {_ACCENT}✓ Busy-indicator style set to '{arg}' (saved to config){_RST}")
            _cprint(f"  {_DIM}The TUI picks up the new style on its next render.{_RST}")
        else:
            _cprint(f"  {_ACCENT}✓ Busy-indicator style set to '{arg}' (session only){_RST}")

    def _handle_fast_command(self, cmd: str):
        """Handle /fast — toggle fast mode (OpenAI Priority Processing / Anthropic Fast Mode).

        Session-scoped by default; ``--global`` persists agent.service_tier
        to config.yaml (parity with /model and /reasoning).
        """
        from cli import _ACCENT, _DIM, _RST, _cprint, save_config_value
        if not self._fast_command_available():
            _cprint("  (._.) /fast is only available for models that support fast mode (OpenAI Priority Processing or Anthropic Fast Mode).")
            return

        # Determine the branding for the current model
        try:
            from hermes_cli.models import _is_anthropic_fast_model
            agent = getattr(self, "agent", None)
            model = getattr(agent, "model", None) or getattr(self, "model", None)
            feature_name = "Anthropic Fast Mode" if _is_anthropic_fast_model(model) else "Priority Processing"
        except Exception:
            feature_name = "Fast mode"

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or parts[1].strip().lower() == "status":
            status = {"priority": "fast", None: "normal"}.get(self.service_tier, self.service_tier)
            _cprint(f"  {_ACCENT}{feature_name}: {status}{_RST}")
            _cprint(f"  {_DIM}Usage: /fast [normal|fast|auto|cold|status] [--global]{_RST}")
            return

        arg, explicit_global = _split_scope_flags(parts[1])
        if arg not in _FAST_TIERS:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Usage: /fast [normal|fast|auto|cold|status] [--global]{_RST}")
            return
        self.service_tier, saved_value = _FAST_TIERS[arg]
        label = saved_value.upper()

        self.agent = None  # Force agent re-init with new service-tier config
        saved = explicit_global and save_config_value("agent.service_tier", saved_value)
        _cprint(f"  {_ACCENT}✓ {feature_name} set to {label} {_scope_outcome(explicit_global, saved)}{_RST}")

    def _handle_debug_command(self, cmd_original: str = ""):
        """Handle /debug [nous|local] — upload debug report + logs and print share URLs.
        Default: public paste service; ``nous``: Nous-internal (staff-only); ``local``: render to
        stdout, no upload. ``local`` wins if both are given (never touches the network)."""
        from hermes_cli.debug import run_debug_share
        from types import SimpleNamespace

        words = {w.lower() for w in cmd_original.split()[1:]}
        local = "local" in words
        nous = "nous" in words and not local
        # Typing /debug is the upload consent (yes=True); input() would hang in prompt_toolkit anyway.
        args = SimpleNamespace(
            lines=200, expire=7, local=local, nous=nous, yes=True
        )
        run_debug_share(args)

    def _handle_update_command(self) -> bool:
        """Handle /update — exit the session and relaunch as ``hermes update``.

        Returns True when confirmed: the caller triggers app exit so the relaunch happens on the
        main thread after prompt_toolkit restores terminal modes. False when cancelled."""
        from hermes_cli.config import is_managed, format_managed_message

        if is_managed():
            print(f"  ✗ {format_managed_message('update Hermes Agent')}")
            return False

        # prompt_toolkit-native modal: renders above the composer, no raw input() races.
        choices = [
            ("once", "Update Now", "exit the current session and update Hermes Agent"),
            ("cancel", "Cancel", "keep the current session"),
        ]
        raw = self._prompt_text_input_modal(
            title="⚕  Update Hermes Agent",
            detail="This will exit the current session and run `hermes update`.",
            choices=choices,
        )
        if raw is None:
            print("  🟡 /update cancelled.")
            return False
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice != "once":
            print("  🟡 /update cancelled.")
            return False

        print()
        print("  ⚕ Launching update...")
        print()

        # run() execs this from the main thread after prompt_toolkit restores terminal
        # modes. Relaunching here (daemon thread) would skip terminal cleanup on POSIX
        # and, on Windows, sys.exit would only end the worker thread.
        self._pending_relaunch = ["update"]
        return True

    def _handle_voice_command(self, command: str):
        """Handle /voice [on|off|tts|status] command."""
        from cli import _cprint
        subcommand = _command_arg(command, lower=True)
        if subcommand == "":  # bare /voice toggles
            subcommand = "off" if self._voice_mode else "on"
        actions = {
            "on": self._enable_voice_mode,
            "off": self._disable_voice_mode,
            "tts": self._toggle_voice_tts,
            "status": self._show_voice_status,
        }
        if subcommand in actions:
            actions[subcommand]()
        else:
            _cprint(f"Unknown voice subcommand: {subcommand}")
            _cprint("Usage: /voice [on|off|tts|status]")

    def _handle_wake_command(self, command: str):
        """Handle /wake [on|off|status] — the 'Hey Hermes' hotword listener. The toggle IS the
        config: on/off also writes ``wake_word.enabled`` so the choice persists; startup
        auto-arm only reads it."""
        from cli import _cprint
        subcommand = _command_arg(command, lower=True)
        if subcommand == "":  # bare /wake toggles
            subcommand = "off" if getattr(self, "_wake_word_active", False) else "on"
        if subcommand == "on":
            if self._start_wake_word_listener(announce=True):
                self._persist_wake_word_enabled(True)
        elif subcommand == "off":
            self._stop_wake_word_listener(announce=True)
            self._persist_wake_word_enabled(False)
        elif subcommand == "status":
            self._show_wake_word_status()
        else:
            _cprint(f"Unknown wake subcommand: {subcommand}")
            _cprint("Usage: /wake [on|off|status]")

    def _persist_wake_word_enabled(self, enabled: bool):
        """Save ``wake_word.enabled`` so the /wake toggle sticks for future sessions."""
        from cli import _cprint, _DIM, _RST, save_config_value

        try:
            from tools.wake_word import load_wake_word_config

            if bool(load_wake_word_config().get("enabled")) == enabled:
                return  # already persisted — don't rewrite config or re-announce
        except Exception:
            pass
        if save_config_value("wake_word.enabled", enabled):
            _cprint(f"{_DIM}Wake word {'enabled' if enabled else 'disabled'} in config "
                    f"(wake_word.enabled: {str(enabled).lower()}).{_RST}")


_CRON_SUBCOMMANDS.update({
    "list": CLICommandsMixin._cron_list,
    "add": CLICommandsMixin._cron_add,
    "create": CLICommandsMixin._cron_add,
    "edit": CLICommandsMixin._cron_edit,
    **{k: CLICommandsMixin._cron_job_action for k in ("pause", "resume", "run", "remove", "rm", "delete")},
})
