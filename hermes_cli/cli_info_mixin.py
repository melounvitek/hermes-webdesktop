"""Informational views and reload flows for the interactive CLI: banner, help, tools, usage, insights, MCP/skills reload, bang shell

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import threading
import time

from hermes_constants import is_termux as _is_termux_environment
from rich.markup import escape as _escape
from utils import base_url_hostname


class CLIInfoMixin:
    """Informational views and reload flows for the interactive CLI: banner, help, tools, usage, insights, MCP/skills reload, bang shell"""

    def show_banner(self):
        """Display the welcome banner in Claude Code style."""
        from cli import _build_compact_banner, build_welcome_banner, get_tool_definitions, logger
        self.console.clear()
        ctx_len = None
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'context_compressor'):
            ctx_len = self.agent.context_compressor.context_length
        
        # Auto-compact for narrow terminals — the full banner with caduceus
        # + tool list needs ~80 columns minimum to render without wrapping.
        term_width = shutil.get_terminal_size().columns
        use_compact = self.compact or term_width < 80
        
        if use_compact:
            self._console_print(_build_compact_banner())
            self._show_status()
        else:
            # Warm-launch fast path: replay last launch's tool panel when the
            # snapshot fingerprint (config.yaml + .env + checkout rev +
            # toolsets) is unchanged, skipping the ~0.5-0.9s cold
            # get_tool_definitions walk. The agent's REAL tool list is still
            # computed fresh at first message; a background refresh below
            # re-verifies the snapshot so any drift self-heals next launch.
            from hermes_cli.banner import (
                compute_toolset_availability,
                load_banner_snapshot,
                save_banner_snapshot,
            )

            snapshot = None
            try:
                snapshot = load_banner_snapshot(self.enabled_toolsets)
            except Exception:
                snapshot = None

            # Get terminal working directory (where commands will execute)
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())

            if snapshot is not None:
                self._defer_tool_warnings = True
                toolset_map = snapshot["toolset_map"]
                build_welcome_banner(
                    console=self.console,
                    model=self.model,
                    cwd=cwd,
                    tools=snapshot["tools"],
                    enabled_toolsets=self.enabled_toolsets,
                    session_id=self.session_id,
                    get_toolset_for_tool=lambda name: toolset_map.get(name),
                    context_length=ctx_len,
                    provider=self.provider,
                    availability=snapshot["availability"],
                    skills_by_category=snapshot.get("skills_by_category"),
                )

                def _refresh_banner_snapshot() -> None:
                    try:
                        from model_tools import get_toolset_for_tool
                        tools = get_tool_definitions(
                            enabled_toolsets=self.enabled_toolsets, quiet_mode=True
                        )
                        availability = compute_toolset_availability(self.enabled_toolsets)
                        tmap = {
                            t["function"]["name"]: get_toolset_for_tool(t["function"]["name"])
                            for t in tools
                        }
                        for item in availability.get("unavailable_toolsets", []):
                            for name in item.get("tools", []):
                                tmap.setdefault(
                                    name, item.get("id", item.get("name", ""))
                                )
                        save_banner_snapshot(
                            tools, self.enabled_toolsets, availability, tmap
                        )
                    except Exception:
                        logger.debug("banner snapshot refresh failed", exc_info=True)

                threading.Thread(
                    target=_refresh_banner_snapshot,
                    name="banner-snapshot-refresh",
                    daemon=True,
                ).start()
            else:
                # Cold path: compute everything live, then persist the snapshot
                # so the next launch replays it.
                from model_tools import get_toolset_for_tool
                tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
                availability = compute_toolset_availability(self.enabled_toolsets)

                build_welcome_banner(
                    console=self.console,
                    model=self.model,
                    cwd=cwd,
                    tools=tools,
                    enabled_toolsets=self.enabled_toolsets,
                    session_id=self.session_id,
                    context_length=ctx_len,
                    provider=self.provider,
                    availability=availability,
                )
                try:
                    tmap = {
                        t["function"]["name"]: get_toolset_for_tool(t["function"]["name"])
                        for t in tools
                    }
                    for item in availability.get("unavailable_toolsets", []):
                        for name in item.get("tools", []):
                            tmap.setdefault(name, item.get("id", item.get("name", "")))
                    save_banner_snapshot(tools, self.enabled_toolsets, availability, tmap)
                except Exception:
                    logger.debug("banner snapshot save failed", exc_info=True)
        
        # Tool discovery is intentionally deferred on the Termux bare prompt
        # path; availability warnings are shown once tools are initialized.
        # On the snapshot fast path (warm launch), the check walks every
        # check_fn (~180ms) — run it in the background refresh thread instead
        # and let its output land above the prompt (patch_stdout-safe).
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            if getattr(self, "_defer_tool_warnings", False):
                threading.Thread(
                    target=self._show_tool_availability_warnings,
                    name="tool-availability-warnings",
                    daemon=True,
                ).start()
            else:
                self._show_tool_availability_warnings()

        # Warn about low context lengths (common with local servers). Keep
        # this tied to the runtime guard so guidance cannot drift again.
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        if ctx_len and ctx_len < MINIMUM_CONTEXT_LENGTH:
            self._console_print()
            self._console_print(
                f"[yellow]⚠️  Context length is only {ctx_len:,} tokens — "
                f"this is likely too low for agent use with tools.[/]"
            )
            self._console_print(
                f"[dim]   Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens. Tool schemas + system prompt use a large fixed prefix.[/]"
            )
            base_url = getattr(self, "base_url", "") or ""
            from urllib.parse import urlparse as _urlparse
            try:
                _parsed = _urlparse(base_url if "://" in base_url else f"//{base_url}")
                _port = _parsed.port
            except ValueError:
                _port = None
            _host = base_url_hostname(base_url)
            if _port == 11434 or "ollama" in _host:
                self._console_print(
                    f"[dim]   Ollama fix: OLLAMA_CONTEXT_LENGTH={MINIMUM_CONTEXT_LENGTH} ollama serve[/]"
                )
            elif _port == 1234:
                self._console_print(
                    "[dim]   LM Studio fix: Set context length in model settings → reload model[/]"
                )
            else:
                self._console_print(
                    "[dim]   Fix: Set model.context_length in config.yaml, or increase your server's context setting[/]"
                )

        # Warn if the configured model is a Nous Hermes LLM (not agentic)
        from hermes_cli.model_switch import is_nous_hermes_non_agentic

        model_name = getattr(self, "model", "") or ""
        if is_nous_hermes_non_agentic(model_name):
            self._console_print()
            self._console_print(
                "[bold yellow]⚠  Nous Research Hermes 3 & 4 models are NOT agentic and are not "
                "designed for use with Hermes Agent.[/]"
            )
            self._console_print(
                "[dim]   They lack tool-calling capabilities required for agent workflows. "
                "Consider using an agentic model (Claude, GPT, Gemini, DeepSeek, etc.).[/]"
            )
            self._console_print(
                "[dim]   Switch with: /model sonnet  or  /model gpt5[/]"
            )

        # Project-local skills: one-line status. Trusted → show count;
        # untrusted-with-skills → point at `hermes skills trust`. Never raises.
        try:
            from agent.skill_utils import (
                get_project_skills_dirs,
                get_untrusted_project_skills_root,
                iter_skill_index_files,
            )
            _proj_dirs = get_project_skills_dirs()
            if _proj_dirs:
                _n = sum(
                    sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
                    for d in _proj_dirs
                )
                if _n:
                    self._console_print(
                        f"[dim]◆ {_n} project skill(s) loaded from this repo[/]"
                    )
            else:
                _untrusted = get_untrusted_project_skills_root()
                if _untrusted is not None:
                    _root, _n = _untrusted
                    self._console_print(
                        f"[yellow]◆ {_n} project skill(s) found in {_root} but not "
                        f"loaded — run `hermes skills trust` to enable them.[/]"
                    )
        except Exception:
            logger.debug("project skills banner notice failed", exc_info=True)

        self._console_print()

    def _fast_command_available(self) -> bool:
        try:
            from hermes_cli.models import model_supports_fast_mode
        except Exception:
            return False
        agent = getattr(self, "agent", None)
        model = getattr(agent, "model", None) or getattr(self, "model", None)
        return model_supports_fast_mode(model)

    def _command_available(self, slash_command: str) -> bool:
        if slash_command == "/fast":
            return self._fast_command_available()
        return True

    def show_help(self, arg: str = ""):
        """Display help. Bare /help shows categorized core commands with the
        skill list collapsed to one line; /help skills lists all skill
        commands; /help <query> filters commands by substring.
        """
        from cli import (
            ChatConsole,
            _BOLD,
            _DIM,
            _RST,
            _accent_hex,
            _cprint,
            _ensure_skill_commands,
            _termux_example_image_path,
            get_skill_bundles,
        )
        from hermes_cli.commands import COMMANDS_BY_CATEGORY, HELP_SESSION_SUBGROUPS

        arg = (arg or "").strip()
        skill_commands = _ensure_skill_commands()

        # /help skills — the full skill-command list (kept out of the default
        # view so core commands don't scroll off screen).
        if arg.lower() in ("skills", "skill"):
            if not skill_commands:
                _cprint("\n  No skill commands installed.\n")
                return
            _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} ({len(skill_commands)} installed):")
            for cmd, info in sorted(skill_commands.items()):
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] {_escape(info['description'])}"
                )
            _cprint("")
            return

        query = arg.lower() if arg else ""

        try:
            from hermes_cli.skin_engine import get_active_help_header
            header = get_active_help_header("(^_^)? Available Commands")
        except Exception:
            header = "(^_^)? Available Commands"
        header = (header or "").strip() or "(^_^)? Available Commands"
        inner_width = 55
        if len(header) > inner_width:
            header = header[:inner_width]
        _cprint(f"\n{_BOLD}+{'-' * inner_width}+{_RST}")
        _cprint(f"{_BOLD}|{header:^{inner_width}}|{_RST}")
        _cprint(f"{_BOLD}+{'-' * inner_width}+{_RST}")

        def _emit(cmd: str, desc: str) -> bool:
            if not self._command_available(cmd):
                return False
            if query and query not in cmd.lower() and query not in desc.lower():
                return False
            ChatConsole().print(
                f"    [bold {_accent_hex()}]{cmd:<15}[/] [dim]-[/] {_escape(desc)}"
            )
            return True

        for category, commands in COMMANDS_BY_CATEGORY.items():
            if category == "Session":
                # Split the oversized Session category into readable sub-groups
                # (Session / Context / Background & Automation) in the renderer.
                sub_of: dict[str, str] = {}
                for _sub, _names in HELP_SESSION_SUBGROUPS.items():
                    for _n in _names:
                        sub_of[f"/{_n}"] = _sub
                buckets: dict[str, list[tuple[str, str]]] = {"Session": []}
                for _sub in HELP_SESSION_SUBGROUPS:
                    buckets[_sub] = []
                for cmd, desc in commands.items():
                    buckets[sub_of.get(cmd, "Session")].append((cmd, desc))
                for _sub in ("Session", *HELP_SESSION_SUBGROUPS.keys()):
                    rows = buckets.get(_sub) or []
                    printed_header = False
                    for cmd, desc in rows:
                        if not self._command_available(cmd):
                            continue
                        if query and query not in cmd.lower() and query not in desc.lower():
                            continue
                        if not printed_header:
                            _cprint(f"\n  {_BOLD}── {_sub} ──{_RST}")
                            printed_header = True
                        _emit(cmd, desc)
                continue

            printed_header = False
            for cmd, desc in commands.items():
                if not self._command_available(cmd):
                    continue
                if query and query not in cmd.lower() and query not in desc.lower():
                    continue
                if not printed_header:
                    _cprint(f"\n  {_BOLD}── {category} ──{_RST}")
                    printed_header = True
                _emit(cmd, desc)

        # Skill commands: collapsed to a one-line pointer by default so the
        # 60+ skill entries don't bury the core command reference (C-04).
        if query:
            # In filter mode, DO include matching skill commands inline.
            matched_skills = [
                (cmd, info) for cmd, info in sorted(skill_commands.items())
                if query in cmd.lower() or query in (info.get("description", "").lower())
            ]
            if matched_skills:
                _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} (matching '{arg}'):")
                for cmd, info in matched_skills:
                    ChatConsole().print(
                        f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] {_escape(info['description'])}"
                    )
        elif skill_commands:
            _cprint(
                f"\n  ⚡ {_BOLD}Skill Commands{_RST}: {len(skill_commands)} installed "
                f"— {_DIM}/help skills{_RST} to list them"
            )

        _bundles_now = get_skill_bundles()
        if _bundles_now and not query:
            _cprint(f"\n  ▣ {_BOLD}Skill Bundles{_RST} ({len(_bundles_now)} installed):")
            for cmd, info in sorted(_bundles_now.items()):
                skill_count = len(info.get("skills", []))
                desc = info.get("description") or f"Load {skill_count} skills"
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] "
                    f"{_escape(desc)} [dim]({skill_count} skills)[/]"
                )

        quick_commands = self.config.get("quick_commands", {})
        if quick_commands and not query:
            _cprint(f"\n  ⚡ {_BOLD}Quick Commands{_RST} ({len(quick_commands)} configured):")
            for name, qcmd in sorted(quick_commands.items()):
                desc = qcmd.get("description", qcmd.get("type", ""))
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{('/' + name):<22}[/] [dim]-[/] {_escape(desc)}"
                )

        if query:
            _cprint(f"\n  {_DIM}Filtered by '{arg}' — run /help for the full list.{_RST}\n")
            return

        _cprint(f"\n  {_DIM}Tip: /help skills lists skill commands · /help <text> filters · Ctrl+P opens the command palette{_RST}")
        _cprint(f"  {_DIM}Multi-line: Ctrl+J, Alt+Enter, or \\\\+Enter for a new line{_RST}")
        _cprint(f"  {_DIM}Draft editor: Ctrl+G (Alt+G in VSCode/Cursor){_RST}")
        if _is_termux_environment():
            _cprint(f"  {_DIM}Attach image: /image {_termux_example_image_path()} or start your prompt with a local image path{_RST}\n")
        else:
            _cprint(f"  {_DIM}Paste image: Alt+V (or /paste){_RST}\n")

    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        from cli import get_tool_definitions, get_toolset_for_tool
        # Pre-assembly list: /tools is a discovery/inspection surface, so it
        # must show the full catalog including tools deferred behind the
        # tool_search bridge (users check this to verify an MCP installed).
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True,
                                     skip_tool_search_assembly=True)
        
        if not tools:
            print("(;_;) No tools available")
            return
        
        # Header
        print()
        title = "(^_^)/ Available Tools"
        width = 78
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        
        # Group tools by toolset
        toolsets = {}
        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            toolset = get_toolset_for_tool(name) or "unknown"
            if toolset not in toolsets:
                toolsets[toolset] = []
            desc = tool["function"].get("description", "")
            # First sentence: split on ". " (period+space) to avoid breaking on "e.g." or "v2.0"
            desc = desc.split("\n")[0]
            if ". " in desc:
                desc = desc[:desc.index(". ") + 1]
            toolsets[toolset].append((name, desc))
        
        # Display by toolset
        for toolset in sorted(toolsets.keys()):
            print(f"  [{toolset}]")
            for name, desc in toolsets[toolset]:
                print(f"    * {name:<20} - {desc}")
            print()
        
        print(f"  Total: {len(tools)} tools  ヽ(^o^)ノ")
        print()

    def show_toolsets(self):
        """Display available toolsets with kawaii ASCII art."""
        from cli import get_all_toolsets, get_toolset_info
        all_toolsets = get_all_toolsets()
        
        # Header
        print()
        title = "(^_^)b Available Toolsets"
        width = 58
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        
        for name in sorted(all_toolsets.keys()):
            info = get_toolset_info(name)
            if info:
                tool_count = info["tool_count"]
                desc = info["description"]
                
                # Mark if currently enabled
                marker = "(*)" if self.enabled_toolsets and name in self.enabled_toolsets else "   "
                print(f"  {marker} {name:<18} [{tool_count:>2} tools] - {desc}")
        
        print()
        print("  (*) = currently enabled")
        print()
        print("  Tip: Use 'all' or '*' to enable all toolsets")
        print("  Example: python cli.py --toolsets web,terminal")
        print()

    def _handle_whoami_command(self):
        """Display slash-command access for the local CLI surface."""
        import getpass

        try:
            user_name = getpass.getuser() or "?"
        except Exception:
            user_name = "?"

        print()
        print("  You:            cli (local terminal)")
        print(f"  User:           {user_name}")
        print("  Tier:           unrestricted")
        print("  Slash commands: all available")
        print()

    def _should_handle_steer_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /steer should be dispatched immediately while the agent is running.

        /steer MUST bypass the normal _pending_input → process_loop path when
        the agent is active, because process_loop is blocked inside
        self.chat() for the duration of the run.  By the time the queued
        command is pulled from _pending_input, _agent_running has already
        flipped back to False, and process_command() takes the idle
        fallback — delivering the steer as a next-turn message instead of
        injecting it mid-run.  Dispatching inline on the UI thread calls
        agent.steer() directly, which is thread-safe (uses _pending_steer_lock).
        """
        from cli import _looks_like_slash_command
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        if not getattr(self, "_agent_running", False):
            return False
        try:
            from hermes_cli.commands import resolve_command
            base = text.split(None, 1)[0].lower().lstrip('/')
            cmd = resolve_command(base)
            return bool(cmd and cmd.name == "steer")
        except Exception:
            return False

    def _should_handle_background_command_inline(
        self, text: str, has_images: bool = False
    ) -> bool:
        """Return True when /bg or /btw should be dispatched while the agent runs.

        Same queue problem /steer had. ``/bg`` exists to start independent
        work *without* waiting for the current turn, and ``/btw`` exists to
        answer a side question about the in-flight conversation, but a slash
        command typed while the agent is busy goes into ``_pending_input``,
        and ``process_loop`` is blocked inside ``self.chat()`` for the whole
        run. The side task would therefore only start once the foreground
        turn has finished, which is the one moment it was not needed.

        Both commands' ``CommandDef`` entries already declare
        ``busy_policy="dispatch"``; the gateway honours that, the classic CLI
        never consulted it. Dispatching inline on the UI thread starts the
        side session immediately and leaves the foreground turn running
        untouched: no interrupt, no steer.
        """
        from cli import _looks_like_slash_command
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        if not getattr(self, "_agent_running", False):
            return False
        try:
            from hermes_cli.commands import resolve_command
            base = text.split(None, 1)[0].lower().lstrip('/')
            cmd = resolve_command(base)
            return bool(cmd and cmd.name in ("bg", "btw"))
        except Exception:
            return False

    def handle_bang_shell(self, text: str) -> bool:
        """Run a ``!<command>`` submission. Returns True when it was handled.

        Dispatched from the input loop BEFORE slash-command routing and before
        anything is queued for the agent, so a bang command never becomes a
        turn: no user message, no assistant message, no tool result touches
        ``self.conversation_history``. That is what makes ``!`` free — zero
        tokens, and role alternation / prompt caching are untouched by
        construction. The invariant is covered by
        tests/cli/test_bang_shell_mode.py.

        Returns False when the text is not a bang command or when bang mode is
        disabled for this context (gateway/cron), letting the caller fall
        through to normal routing.
        """
        from cli import _rich_text_from_ansi
        from hermes_cli.bang_shell import (
            USAGE_HINT,
            bang_shell_enabled,
            check_bang_approval,
            is_bang_command,
            parse_bang_command,
            resolve_bang_cwd,
            run_bang_command,
        )

        if not is_bang_command(text):
            return False
        if not bang_shell_enabled():
            # Gateway / cron / API contexts: no composer, no human at a
            # keyboard, and those users already have their own shells. Let the
            # text route normally rather than becoming remote execution.
            return False

        command = parse_bang_command(text)
        if not command:
            # Bare `!` — show what the feature does instead of running an
            # empty shell or sending "!" to the model.
            self._console_print(f"[dim]{USAGE_HINT}[/]")
            return True

        approval = check_bang_approval(command)
        if not approval.get("approved"):
            message = approval.get("message") or (
                f"Command denied: {approval.get('description', 'flagged as dangerous')}"
            )
            self._console_print(f"[bold red]{_escape(str(message))}[/]")
            return True

        cwd = resolve_bang_cwd(getattr(self, "session_id", None))
        exit_code = run_bang_command(
            command,
            cwd=cwd,
            writer=lambda line: self._console_print(_rich_text_from_ansi(line)),
        )
        if exit_code:
            self._console_print(f"[dim]! exited {exit_code}[/]")
        return True

    def _show_gateway_status(self):
        """Show status of the gateway and connected messaging platforms."""
        from cli import display_hermes_home
        from gateway.config import load_gateway_config, Platform
        
        print()
        print("+" + "-" * 60 + "+")
        print("|" + " " * 15 + "(✿◠‿◠) Gateway Status" + " " * 17 + "|")
        print("+" + "-" * 60 + "+")
        print()
        
        try:
            config = load_gateway_config()
            
            print("  Messaging Platform Configuration:")
            print("  " + "-" * 55)
            
            platform_status = {
                Platform.TELEGRAM: ("Telegram", "TELEGRAM_BOT_TOKEN"),
                Platform.DISCORD: ("Discord", "DISCORD_BOT_TOKEN"),
                Platform.SLACK: ("Slack", "SLACK_BOT_TOKEN"),
                Platform.WHATSAPP: ("WhatsApp", "WHATSAPP_ENABLED"),
            }
            
            for platform, (name, env_var) in platform_status.items():
                pconfig = config.platforms.get(platform)
                if pconfig and pconfig.enabled:
                    home = config.get_home_channel(platform)
                    home_str = f" → {home.name}" if home else ""
                    print(f"    ✓ {name:<12} Enabled{home_str}")
                else:
                    print(f"    ○ {name:<12} Not configured ({env_var})")
            
            print()
            print("  Session Reset Policy:")
            print("  " + "-" * 55)
            policy = config.default_reset_policy
            print(f"    Mode: {policy.mode}")
            print(f"    Daily reset at: {policy.at_hour}:00")
            print(f"    Idle timeout: {policy.idle_minutes} minutes")
            
            print()
            print("  To start the gateway:")
            print("    python cli.py --gateway")
            print()
            print(f"  Configuration file: {display_hermes_home()}/config.yaml")
            print()
            
        except Exception as e:
            print(f"  Error loading gateway config: {e}")
            print()
            print("  To configure the gateway:")
            print("    1. Set environment variables:")
            print("       TELEGRAM_BOT_TOKEN=your_token")
            print("       DISCORD_BOT_TOKEN=your_token")
            print(f"    2. Or configure settings in {display_hermes_home()}/config.yaml")
            print()

    def _print_random_tip(self) -> None:
        """Best-effort discovery tip (startup + /clear); never raises."""
        try:
            from hermes_cli.tips import get_random_tip
            _tip = get_random_tip()
            try:
                from hermes_cli.skin_engine import get_active_skin
                _tip_color = get_active_skin().get_color("banner_dim", "#B8860B")
            except Exception:
                _tip_color = "#B8860B"
            self._console_print(f"[dim {_tip_color}]✦ Tip: {_tip}[/]")
        except Exception:
            pass

    def _toggle_verbose(self):
        """Cycle tool progress mode: off → new → all → verbose → off.

        Tool-progress display (full args / results / think blocks at the
        ``verbose`` step) is INDEPENDENT of global DEBUG logging.  Cycling
        through here does not change ``self.verbose`` or the agent's
        ``verbose_logging`` / ``quiet_mode`` — those remain under the
        explicit ``-v``/``--verbose`` flag and the ``/verbose-logging``
        toggle.  See PR #6a1aa420e for the history that decoupled them.
        """
        from cli import _cprint, save_config_value
        cycle = ["off", "new", "all", "verbose"]
        try:
            idx = cycle.index(self.tool_progress_mode)
        except ValueError:
            idx = 2  # default to "all"
        self.tool_progress_mode = cycle[(idx + 1) % len(cycle)]

        # /verbose is the explicit tool-progress control, so cycling it takes
        # ownership of the mode back from focus view. Leaving _focus_view_enabled
        # set would show a "focus" status-bar badge and hidden-line counts while
        # tool lines were visibly printing. Display-only state change.
        if getattr(self, "_focus_view_enabled", False):
            self._focus_view_enabled = False
            self._focus_saved_tool_progress = None
            self._focus_hidden_lines = 0
            self._focus_last_counted_tool = None
            try:
                from hermes_cli.focus_view import FOCUS_CONFIG_KEY

                save_config_value(FOCUS_CONFIG_KEY, False)
            except Exception:
                pass

        if self.agent:
            self.agent.reasoning_callback = self._current_reasoning_callback()
            # Keep the live agent's tool_progress_mode in sync so the
            # tool_executor rendering path reflects the new mode this turn,
            # without waiting for an agent rebuild.
            self.agent.tool_progress_mode = self.tool_progress_mode

        # Use raw ANSI codes via _cprint so the output is routed through
        # prompt_toolkit's renderer.  self.console.print() with Rich markup
        # writes directly to stdout which patch_stdout's StdoutProxy mangles
        # into garbled sequences like '?[33mTool progress: NEW?[0m' (#2262).
        from hermes_cli.colors import Colors as _Colors
        labels = {
            "off": f"{_Colors.DIM}Tool progress: OFF{_Colors.RESET} — silent mode, just the final response.",
            "new": f"{_Colors.YELLOW}Tool progress: NEW{_Colors.RESET} — show each new tool (skip repeats).",
            "all": f"{_Colors.GREEN}Tool progress: ALL{_Colors.RESET} — show every tool call.",
            "verbose": f"{_Colors.BOLD}{_Colors.GREEN}Tool progress: VERBOSE{_Colors.RESET} — full args, results, and think blocks.",
        }
        _cprint(labels.get(self.tool_progress_mode, ""))

    def _handle_usage_command(self, cmd_original: str):
        """Dispatch `/usage [reset [--force]]`.

        Bare `/usage` keeps the classic display. `/usage reset` redeems one
        banked Codex rate-limit reset credit (guarded: refuses when limits
        aren't exhausted unless --force).
        """
        parts = cmd_original.split()
        args = [p.lower() for p in parts[1:]]
        if args and args[0] == "reset":
            self._usage_reset(force="--force" in args[1:])
            return
        if args:
            print(f"  Unknown /usage subcommand: {' '.join(parts[1:])}. Try /usage or /usage reset [--force].")
            return
        self._show_usage()

    def _usage_reset(self, force: bool = False):
        """`/usage reset [--force]` — redeem one banked Codex reset credit."""
        provider = (
            (getattr(self.agent, "provider", None) if self.agent else None)
            or getattr(self, "provider", None)
        )
        normalized = str(provider or "").strip().lower()
        if normalized != "openai-codex":
            print("  Banked usage resets are only available on the openai-codex provider.")
            print("  Switch with `/model` or `hermes auth` first.")
            return
        base_url = (getattr(self.agent, "base_url", None) if self.agent else None) or getattr(self, "base_url", None)
        api_key = (getattr(self.agent, "api_key", None) if self.agent else None) or getattr(self, "api_key", None)

        from agent.account_usage import redeem_codex_reset_credit

        print("  ⏳ Checking banked reset credits...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            try:
                result = _pool.submit(
                    redeem_codex_reset_credit,
                    base_url=base_url,
                    api_key=api_key,
                    force=force,
                ).result(timeout=45.0)
            except concurrent.futures.TimeoutError:
                print("  ❌ Timed out talking to the Codex backend — try again shortly.")
                return
        print(f"  {result.message}")

    def _show_context_breakdown(self, cmd_original: str = ""):
        """`/context [all]` — visual context-window usage breakdown.

        Renders a 5×20 glyph block grid (each cell ≈ 1% of the model context
        window) plus an estimated per-category table: system prompt, tool
        definitions, rules, skills index, MCP, subagents, memory, and the
        conversation itself — versus free space. `/context all` appends the
        expanded per-skill and per-toolset cost listings.

        Read-only: same chars/4 estimation engine as the desktop context
        popover (agent.context_breakdown) — no provider calls, no prompt-cache
        impact.
        """
        if not self.agent:
            print("  (._.) No active agent -- send a message first.")
            return

        args = cmd_original.split(maxsplit=1)[1].strip().lower() if " " in cmd_original else ""
        expanded = args in {"all", "full", "details"}

        from agent.context_breakdown import (
            compute_context_details,
            compute_session_context_breakdown,
            render_context_breakdown_lines,
        )

        try:
            payload = compute_session_context_breakdown(
                self.agent, self.conversation_history
            )
        except Exception as e:
            print(f"  (._.) Could not compute context breakdown: {e}")
            return

        details = None
        if expanded:
            try:
                details = compute_context_details(self.agent)
            except Exception:
                details = {"skills": [], "toolsets": []}

        model = payload.get("model") or self.model
        print()
        print(f"  🧠 Context Usage — {model}")
        print()
        for line in render_context_breakdown_lines(payload, details=details, grid=True):
            print(f"  {line}")
        print()

    def _show_usage(self):
        """Rate limits + session token usage (when a live agent exists) + Nous credits.

        The Nous credits block is agent-independent (a portal fetch), so it runs even
        with no live agent — important for the TUI, where /usage runs in a slash-worker
        subprocess that resumes the session WITHOUT building an agent (self.agent is None),
        which would otherwise early-return before any credits showed.
        """
        from cli import datetime, format_duration_compact
        if not self.agent:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print("(._.) No active agent -- send a message first.")
            return

        agent = self.agent
        calls = agent.session_api_calls

        if calls == 0:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print("(._.) No API calls made yet in this session.")
            return

        # ── Rate limits (shown first when available) ────────────────
        rl_state = agent.get_rate_limit_state()
        if rl_state and rl_state.has_data:
            from agent.rate_limit_tracker import format_rate_limit_display
            print()
            print(format_rate_limit_display(rl_state))
            print()

        # ── Session token usage ─────────────────────────────────────
        input_tokens = getattr(agent, "session_input_tokens", 0) or 0
        output_tokens = getattr(agent, "session_output_tokens", 0) or 0
        reasoning_tokens = getattr(agent, "session_reasoning_tokens", 0) or 0
        prompt = agent.session_prompt_tokens
        completion = agent.session_completion_tokens
        total = agent.session_total_tokens

        compressor = agent.context_compressor
        last_prompt = compressor.last_prompt_tokens if compressor.last_prompt_tokens > 0 else 0
        ctx_len = compressor.context_length
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        compressions = compressor.compression_count

        msg_count = len(self.conversation_history)
        elapsed = format_duration_compact((datetime.now() - self.session_start).total_seconds())

        print("  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Model:                     {agent.model}")
        print(f"  Input tokens:              {input_tokens:>10,}")
        print(f"  Output tokens:             {output_tokens:>10,}")
        if reasoning_tokens:
            print(f"  ↳ Reasoning (subset):      {reasoning_tokens:>10,}")
        print(f"  Prompt tokens (total):     {prompt:>10,}")
        print(f"  Completion tokens:         {completion:>10,}")
        print(f"  Total tokens:              {total:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  Session duration:          {elapsed:>10}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {msg_count}")
        print(f"  Compressions:     {compressions}")

        # Account limits -- fetched off-thread with a hard timeout so slow
        # provider APIs don't hang the prompt.
        provider = getattr(agent, "provider", None) or getattr(self, "provider", None)
        base_url = getattr(agent, "base_url", None) or getattr(self, "base_url", None)
        api_key = getattr(agent, "api_key", None) or getattr(self, "api_key", None)
        # Lazy import — pulls the OpenAI SDK chain, only needed here.
        from agent.account_usage import fetch_account_usage, render_account_usage_lines
        account_snapshot = None
        if provider:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                try:
                    account_snapshot = _pool.submit(
                        fetch_account_usage, provider,
                        base_url=base_url, api_key=api_key,
                    ).result(timeout=10.0)
                except (concurrent.futures.TimeoutError, Exception):
                    account_snapshot = None
        account_lines = [f"  {line}" for line in render_account_usage_lines(account_snapshot)]
        if account_lines:
            print()
            for line in account_lines:
                print(line)

        # Nous credits magnitudes + monthly-grant gauge (agent-independent — also
        # runs at the no-agent / no-calls early-returns above). See the helper.
        if self._print_nous_credits_block():
            self._print_usage_cta()

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)

    def _show_insights(self, command: str = "/insights"):
        """Show usage insights and analytics from session history."""
        # Parse optional --days flag
        parts = command.split()
        days = 30
        source = None
        i = 1
        while i < len(parts):
            if parts[i] == "--days" and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    print(f"  Invalid --days value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--source" and i + 1 < len(parts):
                source = parts[i + 1]
                i += 2
            elif parts[i].isdigit():
                days = int(parts[i])
                i += 1
            else:
                i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            db = SessionDB()
            try:
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                print(engine.format_terminal(report))
            finally:
                db.close()
        except Exception as e:
            print(f"  Error generating insights: {e}")

    def _check_config_mcp_changes(self) -> None:
        """Detect mcp_servers changes in config.yaml and react.

        Called from process_loop every CONFIG_WATCH_INTERVAL seconds.
        Compares config.yaml mtime + mcp_servers section against the last
        known state.  When a change is detected:

        * By default (``mcp.auto_reload_on_config_change: true``) it
          auto-triggers ``_reload_mcp()`` and informs the user — legacy
          behaviour from #1474.
        * When opted out (``mcp.auto_reload_on_config_change: false``) it
          does NOT reload.  Instead it notifies the user that the config
          changed and that they can apply it with ``/reload-mcp`` — while
          warning that ``/reload-mcp`` rebuilds the tool surface and
          **invalidates the provider prompt cache** (the next message
          re-sends the full input prefix, expensive on long-context /
          high-reasoning models).  This stops silent cache-breaking reloads
          when config.yaml is rewritten frequently by external tooling or
          other Hermes instances.
        """

        import yaml as _yaml

        CONFIG_WATCH_INTERVAL = 5.0  # seconds between config.yaml stat() calls

        now = time.monotonic()
        if now - self._last_config_check < CONFIG_WATCH_INTERVAL:
            return
        self._last_config_check = now

        from hermes_cli.config import get_config_path as _get_config_path
        cfg_path = _get_config_path()
        if not cfg_path.exists():
            return

        try:
            mtime = cfg_path.stat().st_mtime
        except OSError:
            return

        if mtime == self._config_mtime:
            return  # File unchanged — fast path

        # File changed — check whether mcp_servers section changed
        self._config_mtime = mtime
        try:
            with open(cfg_path, encoding="utf-8") as f:
                new_cfg = _yaml.safe_load(f) or {}
        except Exception:
            return

        new_mcp = new_cfg.get("mcp_servers") or {}
        # Expand ${VAR} templates so the comparison is consistent with the
        # init snapshot (self._config_mcp_servers), which was populated from
        # the deep-merged + expanded config.  Without this, any
        # save_config_value() that rewrites config.yaml (even for unrelated
        # keys) triggers a false-positive MCP reload because the raw yaml
        # still has "${POWERMEM_API_KEY}" while the snapshot has the
        # expanded value.
        from hermes_cli.config import _expand_env_vars
        new_mcp = _expand_env_vars(new_mcp)
        if new_mcp == self._config_mcp_servers:
            return  # mcp_servers unchanged (some other section was edited)

        # Detected a change in the mcp_servers section.  By default we
        # auto-reload (legacy behaviour), but if the user has opted out we
        # notify instead of reloading — because every reload rebuilds the
        # agent tool surface and INVALIDATES the provider prompt cache (the
        # next message re-sends the full input prefix, which is expensive on
        # long-context / high-reasoning models).
        #
        # The toggle is the top-level ``mcp.auto_reload_on_config_change``
        # key (see DEFAULT_CONFIG).  Read it from the config we just parsed
        # so the user can flip it in the same edit that changes mcp_servers;
        # missing key means default-on.
        _mcp_cfg = new_cfg.get("mcp")
        _auto = (
            _mcp_cfg.get("auto_reload_on_config_change", True)
            if isinstance(_mcp_cfg, dict)
            else True
        )

        self._config_mcp_servers = new_mcp

        if not _auto:
            # Notify the user that the config changed but do NOT auto-reload.
            # They can apply the new settings on their own terms with
            # /reload-mcp — which we explicitly warn may invalidate the cache.
            print()
            print("🔄 MCP server config changed — reload skipped (auto-reload disabled).")
            print("   New settings are NOT applied yet. To apply them now, run:")
            print("     /reload-mcp")
            print("   ⚠️  Note: /reload-mcp rebuilds the tool set and invalidates the")
            print("   provider prompt cache (next message re-sends full input tokens).")
            return

        # Notify user and reload.  Run in a separate thread with a hard
        # timeout so a hung MCP server cannot block the process_loop
        # indefinitely (which would freeze the entire TUI).
        print()
        print("🔄 MCP server config changed — reloading connections...")
        _reload_thread = threading.Thread(
            target=self._reload_mcp, daemon=True
        )
        _reload_thread.start()

    def _confirm_and_reload_mcp(self, cmd_original: str = "") -> None:
        """Interactive /reload-mcp — confirm with the user, then reload.

        The auto-reload path (config file watcher) calls ``_reload_mcp``
        directly and never goes through this confirmation.

        Reloading MCP tools invalidates the provider prompt cache for the
        active session (tool schemas are baked into the system prompt).
        The next message re-sends full input tokens — can be expensive on
        long-context or high-reasoning models.

        Three options: Approve Once, Always Approve (persists
        ``approvals.mcp_reload_confirm: false`` so future reloads run
        without this prompt), Cancel.  Gated by
        ``approvals.mcp_reload_confirm`` — default on.
        """
        from cli import load_cli_config, save_config_value
        # Gate check — respects prior "Always Approve" clicks.
        try:
            cfg = load_cli_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            confirm_required = True
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("mcp_reload_confirm", True))
        except Exception:
            confirm_required = True

        if not confirm_required:
            with self._busy_command(self._slow_command_status(cmd_original)):
                self._reload_mcp()
            return

        # Render warning + prompt.  Use the same prompt_toolkit-native composer
        # modal as destructive slash confirmations so choices stay visible.
        choices = [
            ("once", "Approve Once", "reload now"),
            ("always", "Always Approve", "reload now and silence this prompt permanently"),
            ("cancel", "Cancel", "leave MCP tools unchanged"),
        ]
        raw = self._prompt_text_input_modal(
            title="⚠️  /reload-mcp — Prompt cache invalidation warning",
            detail=(
                "Reloading MCP servers rebuilds the tool set for this session and\n"
                "invalidates the provider prompt cache. The next message will\n"
                "re-send full input tokens (can be expensive on long-context or\n"
                "high-reasoning models)."
            ),
            choices=choices,
        )
        if raw is None:
            print("🟡 /reload-mcp cancelled (no input).")
            return
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice is None:
            print(f"🟡 Unrecognized choice '{raw}'. /reload-mcp cancelled.")
            return

        if choice == "cancel":
            print("🟡 /reload-mcp cancelled. MCP tools unchanged.")
            return

        if choice == "always":
            if save_config_value("approvals.mcp_reload_confirm", False):
                print("🔒 Future /reload-mcp calls will run without confirmation.")
                print("   Re-enable via `approvals.mcp_reload_confirm: true` in config.yaml.")
            else:
                print("⚠️  Couldn't persist opt-out — reloading once.")

        with self._busy_command(self._slow_command_status(cmd_original)):
            self._reload_mcp()

    def _reload_mcp(self):
        """Reload MCP servers: disconnect all, re-read config.yaml, reconnect.

        After reconnecting, refreshes the agent's tool list so the model
        sees the updated tools on the next turn.
        """
        try:
            from tools.mcp_tool import (
                shutdown_mcp_servers, discover_mcp_tools, reprobe_tool_availability, _servers, _lock,
            )

            # Capture old server names
            with _lock:
                old_servers = set(_servers.keys())

            if not self._command_running:
                print("🔄 Reloading MCP servers...")

            # Shutdown existing connections
            shutdown_mcp_servers()

            # Explicit reload also re-probes tool availability (check_fn).
            reprobe_tool_availability()
            # Reconnect (reads config.yaml fresh)
            new_tools = discover_mcp_tools()

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            if reconnected:
                print(f"  ♻️  Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                print(f"  ➕ Added: {', '.join(sorted(added))}")
            if removed:
                print(f"  ➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                print("  No MCP servers connected.")
            else:
                print(f"  🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Refresh the agent's tool list so the model can call new tools.
            # Route through the shared helper so this CLI /reload-mcp path stays
            # in lockstep with the TUI RPC / gateway reload / late-binding paths
            # (name-diff, thread-safe, and — critically — additive-preserving so
            # memory-provider and context-engine tools survive the rebuild).
            if self.agent is not None:
                from tools.mcp_tool import refresh_agent_mcp_tools
                # Explicit reload: pick up MCP servers the user ENABLED in config
                # this session. self.enabled_toolsets was resolved once at
                # startup; merge in any now-connected server names (unless the
                # user pinned `all`/`*`, which already includes everything) so a
                # freshly-added server isn't filtered out. Mirrors startup, where
                # MCP server names are part of enabled_toolsets (see __init__).
                enabled_override = None
                et = self.enabled_toolsets
                if et and "all" not in et and "*" not in et:
                    merged = list(et)
                    for _name in sorted(connected_servers):
                        if _name not in merged:
                            merged.append(_name)
                    enabled_override = merged
                refresh_agent_mcp_tools(
                    self.agent,
                    enabled_override=enabled_override,
                    quiet_mode=True,
                )
                # Keep the CLI's own list in sync with what the agent now uses.
                if enabled_override is not None:
                    self.enabled_toolsets = enabled_override

            # Inject a message at the END of conversation history so the
            # model knows tools changed.  Appended after all existing
            # messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            self.conversation_history.append({
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            })

            # Persist session immediately so the session log reflects the
            # updated tools list (self.agent.tools was refreshed above).
            if self.agent is not None:
                try:
                    self.agent._persist_session(
                        self.conversation_history,
                        self.conversation_history,
                    )
                except Exception:
                    pass  # Best-effort

            print(f"  ✅ Agent updated — {len(self.agent.tools if self.agent else [])} tool(s) available")

        except Exception as e:
            print(f"  ❌ MCP reload failed: {e}")

    def _reload_skills(self) -> None:
        """Reload skills: rescan ~/.hermes/skills/ and queue a note for the
        next user turn.

        Skills don't need to live in the system prompt for the model to use
        them (they're invoked via ``/skill-name``, ``skills_list``, or
        ``skill_view`` at runtime), so this does NOT clear the prompt cache.
        It rescans the slash-command map, prints the diff for the user, and
        — if any skills were added or removed — queues a one-shot note that
        gets prepended to the next user message. This preserves message
        alternation (no phantom user turn injected out of band) and keeps
        prompt caching intact.
        """
        try:
            from agent.skill_commands import reload_skills, get_skill_commands

            if not self._command_running:
                print("🔄 Reloading skills...")

            result = reload_skills()

            # Sync cli.py's module-level _skill_commands so all consumers
            # (help display, command dispatch, Tab-completion lambda) see the
            # updated dict without needing to restart the session.
            import cli as _cli
            _cli._skill_commands = get_skill_commands()
            added = result.get("added", [])      # [{"name", "description"}, ...]
            removed = result.get("removed", [])  # [{"name", "description"}, ...]
            total = result.get("total", 0)

            if not added and not removed:
                print("  No new skills detected.")
                print(f"  📚 {total} skill(s) available")
                return

            def _fmt_line(item: dict) -> str:
                nm = item.get("name", "")
                desc = item.get("description", "")
                return f"    - {nm}: {desc}" if desc else f"    - {nm}"

            if added:
                print("  ➕ Added Skills:")
                for item in added:
                    print(f"  {_fmt_line(item)}")
            if removed:
                print("  ➖ Removed Skills:")
                for item in removed:
                    print(f"  {_fmt_line(item)}")
            print(f"  📚 {total} skill(s) available")

            # Queue a one-shot note for the NEXT user turn. The CLI's agent
            # loop prepends ``_pending_skills_reload_note`` (if set) to the
            # API-call-local message at ~L8770, then clears it — same
            # pattern as ``_pending_model_switch_note``. Nothing is written
            # to conversation_history here, so message alternation stays
            # intact and no out-of-band user turn is persisted.
            #
            # Format matches how the system prompt renders pre-existing
            # skills (``    - name: description``) so the model reads the
            # diff in the same shape as its original skill catalog.
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            if added:
                sections.append("")
                sections.append("Added Skills:")
                for item in added:
                    sections.append(_fmt_line(item))
            if removed:
                sections.append("")
                sections.append("Removed Skills:")
                for item in removed:
                    sections.append(_fmt_line(item))
            sections.append("")
            sections.append("Use skills_list to see the updated catalog.]")
            self._pending_skills_reload_note = "\n".join(sections)

        except Exception as e:
            print(f"  ❌ Skills reload failed: {e}")
