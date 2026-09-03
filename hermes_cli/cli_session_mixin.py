"""Session lifecycle for the interactive CLI: new/resume/save, undo/retry rewinds, yolo persistence, manual compression, and exit summary

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid

from hermes_constants import get_hermes_home
from pathlib import Path
from rich.console import Console
from rich.markup import escape as _escape
from typing import Any, Dict, List, Optional


class CLISessionMixin:
    """Session lifecycle for the interactive CLI: new/resume/save, undo/retry rewinds, yolo persistence, manual compression, and exit summary"""

    def _restore_session_cwd(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Relaunch a resumed session in the directory it was started from.

        Idempotent and safe to call from every resume path. When the stored
        ``cwd`` differs from the current process directory, we both
        ``os.chdir()`` (so the process and any ``os.getcwd()`` fallback agree)
        and retarget ``TERMINAL_CWD`` (so the terminal tool, code-exec tool,
        and relative-path resolution all land in the same place — the local
        terminal backend snapshots cwd on first use, which happens after this).

        No-ops when: the session recorded no cwd (gateway/remote/older
        sessions), the directory no longer exists, or we're already there.
        A missing directory degrades to a single dim warning rather than a
        crash — repos get moved and deleted.
        """
        recorded = (session_meta or {}).get("cwd")
        if not recorded:
            return
        recorded = os.path.expanduser(str(recorded))
        try:
            current = os.getcwd()
        except OSError:
            current = None
        if current and os.path.realpath(recorded) == os.path.realpath(current):
            return  # Already where the session lived — nothing to announce.

        if not os.path.isdir(recorded):
            msg = f"⚠ Session's working directory is gone: {recorded} — staying in {current or '.'}"
            if quiet:
                print(msg, file=sys.stderr)
            else:
                self._console_print(f"[dim]{_escape(msg)}[/dim]")
            return

        try:
            os.chdir(recorded)
        except OSError as e:
            msg = f"⚠ Could not enter session's working directory {recorded}: {e}"
            if quiet:
                print(msg, file=sys.stderr)
            else:
                self._console_print(f"[dim]{_escape(msg)}[/dim]")
            return

        # Retarget the terminal/code-exec tools to match the process cwd.
        os.environ["TERMINAL_CWD"] = recorded

        msg = f"↻ Working directory: {recorded}"
        if quiet:
            print(msg, file=sys.stderr)
        else:
            self._console_print(f"[dim]{_escape(msg)}[/dim]")

    def _restore_session_yolo(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Re-enable YOLO bypass on resume when the session had it on.

        Companion to ``_restore_session_cwd`` — called from every resume path
        (startup ``--resume``/``-c`` and mid-chat ``/resume``). The persisted
        flag lives in the session row's ``model_config.yolo_mode`` (written by
        ``/yolo`` toggles and ``--yolo`` launches); without this restore the
        in-memory ``tools.approval._session_yolo`` set starts empty in a fresh
        process and the user's bypass silently reverts.

        No-op when the flag is absent/false, when YOLO is already active for
        this session (idempotent across repeated resume paths), or when the
        process was itself launched with ``--yolo`` (frozen bypass already
        covers everything).
        """
        try:
            from hermes_state import SessionDB
            from tools.approval import (
                _YOLO_MODE_FROZEN,
                enable_session_yolo,
                is_session_yolo_enabled,
            )
        except Exception:
            return
        if _YOLO_MODE_FROZEN:
            return
        if not SessionDB.session_yolo_enabled(session_meta):
            return
        session_key = self.session_id or "default"
        if is_session_yolo_enabled(session_key):
            return
        enable_session_yolo(session_key)
        msg = "⚡ YOLO mode restored from session — all commands auto-approved. /yolo to turn off."
        if quiet:
            print(msg, file=sys.stderr)
        else:
            self._console_print(f"[dim]{_escape(msg)}[/dim]")

    def _render_resume_history_panel_lines(self, panel) -> list[str]:
        """Render the resume panel at the current terminal width for resize replay."""
        from cli import _suspend_output_history
        from io import StringIO

        buf = StringIO()
        width = shutil.get_terminal_size((80, 24)).columns
        console = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            highlight=False,
            width=width,
        )
        with _suspend_output_history():
            console.print(panel)
        return buf.getvalue().rstrip("\n").splitlines()

    def _resolve_checkpoint_ref(self, ref: str, checkpoints: list) -> str | None:
        """Resolve a checkpoint number or hash to a full commit hash."""
        try:
            idx = int(ref) - 1  # 1-indexed for user
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]["hash"]
            else:
                print(f"  Invalid checkpoint number. Use 1-{len(checkpoints)}.")
                return None
        except ValueError:
            # Treat as a git hash
            return ref

    def _show_status(self):
        """Show compact startup status line."""
        from cli import get_tool_definitions
        # Avoid pulling the full tool registry into the bare Termux prompt path.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") == "1":
            tool_status = "tools deferred"
        else:
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            tool_count = len(tools) if tools else 0
            tool_status = f"{tool_count} tools"

        # Format model name (shorten if needed)
        model_short = self.model.split("/")[-1] if "/" in self.model else self.model
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."

        # Get API status indicator
        api_indicator = "[green bold]●[/]" if self.api_key else "[red bold]●[/]"

        # Build status line with proper markup — skin-aware colors
        try:
            from hermes_cli.skin_engine import get_active_skin
            skin = get_active_skin()
            separator_color = skin.get_color("banner_dim", "#B8860B")
            accent_color = skin.get_color("ui_accent", "#FFBF00")
            label_color = skin.get_color("ui_label", "#DAA520")
        except Exception:
            separator_color, accent_color, label_color = "#B8860B", "#FFBF00", "cyan"
        toolsets_info = ""
        if self.enabled_toolsets and "all" not in self.enabled_toolsets:
            toolsets_info = f" [dim {separator_color}]·[/] [{label_color}]toolsets: {', '.join(self.enabled_toolsets)}[/]"

        provider_info = f" [dim {separator_color}]·[/] [dim]provider: {self.provider}[/]"
        if self._provider_source:
            provider_info += f" [dim {separator_color}]·[/] [dim]auth: {self._provider_source}[/]"

        self._console_print(
            f"  {api_indicator} [{accent_color}]{model_short}[/] "
            f"[dim {separator_color}]·[/] [bold {label_color}]{tool_status}[/]"
            f"{toolsets_info}{provider_info}"
        )

    def _show_session_status(self):
        """Show gateway-style status for the current CLI session."""
        from cli import datetime, display_hermes_home
        session_meta = {}
        if self._session_db:
            try:
                session_meta = self._session_db.get_session(self.session_id) or {}
            except Exception:
                session_meta = {}

        title = (session_meta.get("title") or "").strip()

        created_at = self.session_start
        started_at = session_meta.get("started_at")
        if started_at:
            try:
                created_at = datetime.fromtimestamp(float(started_at))
            except Exception:
                created_at = self.session_start

        updated_at = created_at
        for field in ("updated_at", "last_updated_at", "last_activity_at"):
            value = session_meta.get(field)
            if not value:
                continue
            try:
                updated_at = datetime.fromtimestamp(float(value))
                break
            except Exception:
                pass

        agent = getattr(self, "agent", None)
        total_tokens = getattr(agent, "session_total_tokens", 0) or 0
        provider = getattr(self, "provider", None) or "unknown"
        model = getattr(self, "model", None) or "(unknown)"
        is_running = bool(getattr(self, "_agent_running", False))

        # Reasoning level (C-02): resolve the effective effort for display.
        reasoning_label = None
        try:
            rc = getattr(agent, "reasoning_config", None) or getattr(self, "reasoning_config", None)
            if isinstance(rc, dict):
                if rc.get("enabled") is False:
                    reasoning_label = "off"
                elif rc.get("effort"):
                    reasoning_label = str(rc.get("effort"))
            show_r = getattr(self, "show_reasoning", None)
            if reasoning_label:
                reasoning_label += f" (display: {'on' if show_r else 'off'})" if show_r is not None else ""
        except Exception:
            reasoning_label = None

        # Approval mode (C-02).
        approval_label = None
        try:
            from tools.approval import _get_approval_mode, is_approval_bypass_active_for_session
            approval_label = _get_approval_mode()
            try:
                if is_approval_bypass_active_for_session(getattr(self, "session_key", "") or ""):
                    approval_label += " (YOLO bypass active)"
            except Exception:
                pass
        except Exception:
            approval_label = None

        # Context window usage (C-02): reuse the status-bar snapshot which
        # already computes tokens / max / percent.
        ctx_label = None
        try:
            snap = self._get_status_bar_snapshot()
            ctx_tokens = snap.get("context_tokens") or 0
            ctx_max = snap.get("context_length")
            ctx_pct = snap.get("context_percent")
            if ctx_max:
                left = ""
                if isinstance(ctx_pct, (int, float)):
                    left = f"{max(0, 100 - int(ctx_pct))}% left · "
                ctx_label = f"{left}{ctx_tokens:,} / {ctx_max:,} tokens used"
        except Exception:
            ctx_label = None

        lines = [
            "Hermes CLI Status",
            "",
            f"Session ID: {self.session_id}",
            f"Path: {display_hermes_home()}",
        ]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Model: {model} ({provider})")
        if reasoning_label:
            lines.append(f"Reasoning: {reasoning_label}")
        if approval_label:
            lines.append(f"Approvals: {approval_label}")
        if ctx_label:
            lines.append(f"Context: {ctx_label}")
        lines.extend([
            f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {total_tokens:,}",
            f"Agent Running: {'Yes' if is_running else 'No'}",
        ])
        self._console_print("\n".join(lines), highlight=False, markup=False)

    def _list_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent CLI sessions for in-chat browsing/resume affordances."""
        if not self._session_db:
            return []
        try:
            from hermes_cli.session_listing import query_session_listing

            return query_session_listing(
                self._session_db,
                source="cli",
                current_session_id=self.session_id,
                include_all_sources=False,
                include_unnamed=True,
                limit=limit,
                exclude_sources=["kanban", "tool"],
            )
        except Exception:
            return []

    def _show_recent_sessions(self, *, reason: str = "history", limit: int = 10) -> bool:
        """Render recent sessions inline from the active chat TUI.

        Returns True when something was shown, False if no session list was available.
        """
        from cli import _cli_visible_print
        sessions = self._list_recent_sessions(limit=limit)
        if not sessions:
            return False

        from hermes_cli.main import _relative_time

        _cli_visible_print()
        if reason == "history":
            _cli_visible_print("(._.) No messages in the current chat yet — here are recent sessions you can resume:")
        else:
            _cli_visible_print("  Recent sessions:")
        _cli_visible_print()
        _cli_visible_print(f"  {'#':<3} {'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}")
        _cli_visible_print(f"  {'─' * 3} {'─' * 32} {'─' * 40} {'─' * 13} {'─' * 24}")
        for idx, session in enumerate(sessions, start=1):
            title = session.get("title") or "—"
            preview = (session.get("preview") or "")[:38]
            last_active = _relative_time(session.get("last_active"))
            _cli_visible_print(f"  {idx:<3} {title:<32} {preview:<40} {last_active:<13} {session['id']}")
        _cli_visible_print()
        _cli_visible_print("  Use /resume <number>, /resume <session id>, or /resume <session title> to continue.")
        _cli_visible_print("  Example: /resume 2")
        _cli_visible_print()
        return True

    def show_history(self):
        """Display conversation history."""
        from cli import _cli_visible_print
        if not self.conversation_history:
            if not self._show_recent_sessions(reason="history"):
                _cli_visible_print("(._.) No conversation history yet.")
            return

        preview_limit = 400
        visible_index = 0
        hidden_tool_messages = 0
        show_ts = bool(getattr(self, "show_timestamps", False))

        def _ts_suffix(message: dict) -> str:
            # Messages restored from SessionDB carry a unix `timestamp`; live
            # unsaved turns may not. Only annotate when both the toggle is on
            # and the turn actually has a stored time — never fabricate one.
            if not show_ts:
                return ""
            ts = message.get("timestamp")
            if not ts:
                return ""
            try:
                from datetime import datetime
                return f"  [{datetime.fromtimestamp(float(ts)).strftime(getattr(self, 'timestamp_format', '%H:%M'))}]"
            except (ValueError, OSError, TypeError):
                return ""

        def flush_tool_summary():
            nonlocal hidden_tool_messages
            if not hidden_tool_messages:
                return

            noun = "message" if hidden_tool_messages == 1 else "messages"
            _cli_visible_print("\n  [Tools]")
            _cli_visible_print(f"    ({hidden_tool_messages} tool {noun} hidden)")
            hidden_tool_messages = 0

        _cli_visible_print()
        _cli_visible_print("+" + "-" * 50 + "+")
        _cli_visible_print("|" + " " * 12 + "(^_^) Conversation History" + " " * 11 + "|")
        _cli_visible_print("+" + "-" * 50 + "+")

        for msg in self.conversation_history:
            role = msg.get("role", "unknown")

            if role == "tool":
                hidden_tool_messages += 1
                continue

            if role not in {"user", "assistant"}:
                continue

            flush_tool_summary()
            visible_index += 1

            content = msg.get("content")
            content_text = "" if content is None else str(content)

            if role == "user":
                _cli_visible_print(f"\n  [You #{visible_index}]{_ts_suffix(msg)}")
                _cli_visible_print(
                    f"    {content_text[:preview_limit]}{'...' if len(content_text) > preview_limit else ''}"
                )
                continue

            _cli_visible_print(f"\n  [Hermes #{visible_index}]{_ts_suffix(msg)}")
            tool_calls = msg.get("tool_calls") or []
            if content_text:
                preview = content_text[:preview_limit]
                suffix = "..." if len(content_text) > preview_limit else ""
            elif tool_calls:
                tool_count = len(tool_calls)
                noun = "call" if tool_count == 1 else "calls"
                preview = f"(requested {tool_count} tool {noun})"
                suffix = ""
            else:
                preview = "(no text response)"
                suffix = ""
            _cli_visible_print(f"    {preview}{suffix}")

        flush_tool_summary()
        _cli_visible_print()

    def _notify_session_boundary(self, event_type: str) -> None:
        """Fire a session-boundary plugin hook (on_session_finalize or on_session_reset).

        Non-blocking — errors are caught and logged.  Safe to call from any
        lifecycle point (shutdown, /new, /reset).
        """
        try:
            from hermes_cli.lifecycle import finalize_session, invoke_hook

            context = {
                "session_id": self.agent.session_id if self.agent else None,
                "platform": getattr(self, "platform", None) or "cli",
                "reason": (
                    "new_session"
                    if event_type == "on_session_reset"
                    else "session_boundary"
                ),
            }
            if event_type == "on_session_finalize":
                finalize_session(**context)
            else:
                invoke_hook(event_type, **context)
        except Exception:
            pass

    def _discard_session_if_empty(self, session_id: Optional[str]) -> bool:
        """Drop a just-ended session row when it never gained content.

        Starting the CLI and immediately quitting (or rotating with /new,
        /clear) used to leave an empty untitled row behind that clutters
        ``/resume`` and ``hermes sessions list``. Delegates the
        check-and-delete to ``SessionDB.delete_session_if_empty``, which
        only removes rows with no messages, no title, and no child
        sessions. Ported from google-gemini/gemini-cli#27770.
        """
        from cli import logger
        if not self._session_db or not session_id:
            return False
        # In-memory transcript is authoritative: if this CLI object holds
        # conversation messages (flushed to the DB or not), the session is
        # not empty. Protects against pruning a real conversation whose DB
        # flush failed or hasn't happened yet.
        if getattr(self, "conversation_history", None):
            return False
        try:
            from hermes_constants import get_hermes_home as _ghh
            return self._session_db.delete_session_if_empty(
                session_id, sessions_dir=_ghh() / "sessions"
            )
        except Exception:
            logger.debug(
                "Could not prune empty session %s", session_id, exc_info=True
            )
            return False

    def _launch_session_boundary_memory_flush(
        self,
        history_snapshot: list,
        *,
        session_id: Optional[str] = None,
    ) -> Optional[list]:
        """Stage old-session memory extraction so /new stays responsive.

        The context-engine ``on_session_end`` boundary is delivered
        synchronously here: it is cheap (local state clear, no LLM call) and
        ordering-sensitive — it must land before ``reset_session_state()``
        rebinds the engine to the new session.

        The memory-provider half (LLM-bound extraction, seconds) is NOT run
        here. The returned snapshot is handed by ``new_session()`` to
        ``MemoryManager.commit_session_boundary_async`` as a single
        end→switch task on the manager's serialized background worker, so
        extraction can never race the provider rebinding (providers key off
        internal ``_session_id`` state — a late ``on_session_end`` after
        ``on_session_switch`` would misattribute the old transcript to the
        new session).

        Returns the history snapshot to queue, or ``None`` when there is
        nothing to extract (no agent / empty history / no memory manager).
        """
        from cli import logger
        agent = getattr(self, "agent", None)
        if not agent or not history_snapshot:
            return None

        engine = getattr(agent, "context_compressor", None)
        if engine is not None and hasattr(engine, "on_session_end"):
            try:
                engine.on_session_end(session_id or "", history_snapshot)
            except Exception:
                logger.debug(
                    "Context engine on_session_end failed at /new boundary",
                    exc_info=True,
                )

        # No provider extraction to queue when no memory manager is
        # configured — new_session() falls back to the inline switch path.
        if getattr(agent, "_memory_manager", None) is None:
            return None
        return history_snapshot

    def new_session(self, silent=False, title=None):
        """Start a fresh session with a new session ID and cleared agent state."""
        from cli import (
            CLI_CONFIG,
            _cprint,
            _parse_reasoning_config,
            _parse_service_tier_config,
            _split_model_config_default,
            _sync_process_session_id,
            datetime,
            logger,
        )
        old_session_id = self.session_id
        _boundary_snapshot = None
        if self.agent and self.conversation_history:
            # Deliver the context-engine boundary synchronously and get back
            # the history snapshot for the deferred provider extraction —
            # queued below (after rotation) so /new never blocks on the
            # LLM-bound extraction call.
            _boundary_snapshot = self._launch_session_boundary_memory_flush(
                list(self.conversation_history),
                session_id=old_session_id,
            )
            self._notify_session_boundary("on_session_finalize")
        elif self.agent:
            # First session or empty history — still finalize the old session
            self._notify_session_boundary("on_session_finalize")

        if self._session_db and old_session_id:
            # Flush any un-persisted messages from the current turn to the
            # old session *before* rotating.  /new can be called mid-turn
            # when _flush_messages_to_session_db() has not yet run — without
            # this, messages generated during the current turn are silently
            # lost on session rotation (#47202).
            if self.agent:
                try:
                    self.agent._flush_messages_to_session_db(
                        self.conversation_history,
                        conversation_history=self.conversation_history,
                    )
                except Exception:
                    pass  # best-effort
            try:
                self._session_db.end_session(old_session_id, "new_session")
            except Exception:
                pass
            # Don't let immediately-rotated empty sessions pile up in
            # /resume and `hermes sessions list` (gemini-cli#27770 port).
            self._discard_session_if_empty(old_session_id)

        self.session_start = datetime.now()
        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        self.session_id = f"{timestamp_str}_{short_uuid}"
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()
        self.conversation_history = []
        self._pending_title = None
        self._resumed = False
        # /new clears the -m / --model override flag: an explicit CLI model
        # was for the previous session only, not for every session spawned
        # afterwards.
        self._explicit_model_override = False
        self.reasoning_config = _parse_reasoning_config(
            CLI_CONFIG["agent"].get("reasoning_effort", "")
        )
        # /new is a full conversation boundary: session-scoped runtime
        # overrides (/model --session, /fast, one-turn restores) do not carry
        # forward.  Re-derive model/provider and service tier from config.yaml
        # so a session-only switch never leaks into the next session (#48055,
        # #23131).
        self._pending_one_turn_model_restore = None
        self.service_tier = _parse_service_tier_config(
            CLI_CONFIG["agent"].get("service_tier", "")
        )
        _model_config = CLI_CONFIG.get("model", {})
        _raw_default2 = (_model_config.get("default") or _model_config.get("model") or "") if isinstance(_model_config, dict) else (_model_config or "")
        _config_model, _ = _split_model_config_default(_raw_default2)
        if _config_model and _config_model != getattr(self, "model", None):
            _config_provider = (
                _model_config.get("provider", "")
                if isinstance(_model_config, dict)
                else ""
            )
            try:
                from hermes_cli.model_switch import switch_model as _switch_model

                _reset_result = _switch_model(
                    raw_input=_config_model,
                    current_provider=self.provider or "",
                    current_model=self.model or "",
                    current_base_url=self.base_url or "",
                    current_api_key=self.api_key or "",
                    is_global=False,
                    explicit_provider=_config_provider or "",
                )
                if _reset_result.success:
                    if self.agent:
                        self.agent.switch_model(
                            new_model=_reset_result.new_model,
                            new_provider=_reset_result.target_provider,
                            api_key=_reset_result.api_key,
                            base_url=_reset_result.base_url,
                            api_mode=_reset_result.api_mode,
                            capabilities=getattr(
                                _reset_result, "runtime_capabilities", None
                            ),
                        )
                    self.model = _reset_result.new_model
                    self.provider = _reset_result.target_provider
                    self.requested_provider = _reset_result.target_provider
                    self._explicit_api_key = _reset_result.api_key
                    self._explicit_base_url = _reset_result.base_url
                    if _reset_result.api_key:
                        self.api_key = _reset_result.api_key
                    if _reset_result.base_url:
                        self.base_url = _reset_result.base_url
                    if _reset_result.api_mode:
                        self.api_mode = _reset_result.api_mode
                    if not silent:
                        _cprint(
                            f"  (model reset to config default: "
                            f"{_reset_result.new_model})"
                        )
            except Exception:
                # Best-effort: an unreachable config default must never block
                # /new. The session keeps the current working model.
                logger.debug("/new model reset to config default failed", exc_info=True)
        _sync_process_session_id(self.session_id)

        if self.agent:
            self.agent.session_id = self.session_id
            self.agent.session_start = self.session_start
            self.agent.reasoning_config = self.reasoning_config
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = 0
            if hasattr(self.agent, "_todo_store"):
                try:
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
                except Exception:
                    pass
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

            if self._session_db:
                try:
                    self.agent._session_db_created = False
                    self._session_db.create_session(
                        session_id=self.session_id,
                        source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                        model=self.model,
                        model_config={
                            "max_iterations": self.max_turns,
                            "reasoning_config": self.reasoning_config,
                        },
                    )
                    self.agent._session_db_created = True
                except Exception:
                    pass
                if title and self._session_db:
                    from hermes_state import SessionDB
                    try:
                        sanitized = SessionDB.sanitize_title(title)
                    except ValueError as e:
                        _cprint(f"  Title rejected: {e}")
                        sanitized = None
                        title = None
                    if sanitized:
                        try:
                            self._session_db.set_session_title(self.session_id, sanitized)
                            self._pending_title = None
                            self._status_bar_title_checked_at = 0.0
                            title = sanitized
                        except ValueError as e:
                            _cprint(f"  {e} — session started untitled.")
                            title = None
                        except Exception:
                            title = None
                    elif title is not None:
                        # sanitize_title returned empty (whitespace-only / unprintable)
                        _cprint("  Title is empty after cleanup — session started untitled.")
                        title = None
            # Notify memory providers that session_id rotated to a fresh
            # conversation. reset=True signals providers to flush accumulated
            # per-session state (_session_turns, _turn_counter, _document_id).
            # Fires BEFORE the plugin on_session_reset hook (shell hooks only
            # see the new id; Python providers see the transition). See #6672.
            #
            # When the old session has history, end-of-session extraction
            # (LLM-bound, seconds) and this switch are queued as ONE task on
            # the memory manager's serialized worker — end strictly before
            # switch, without blocking /new (#16454). With no history there
            # is nothing to extract; switch inline as before.
            try:
                _mm = getattr(self.agent, "_memory_manager", None)
                if _mm is not None:
                    if _boundary_snapshot:
                        _mm.commit_session_boundary_async(
                            _boundary_snapshot,
                            new_session_id=self.session_id,
                            parent_session_id=old_session_id or "",
                            reason="new_session",
                        )
                    else:
                        _mm.on_session_switch(
                            self.session_id,
                            parent_session_id=old_session_id or "",
                            reset=True,
                            reason="new_session",
                        )
            except Exception:
                pass
            self._notify_session_boundary("on_session_reset")

        if not silent:
            if title:
                print(f"(^_^)v New session started: {title}")
            else:
                print("(^_^)v New session started!")

    def _consume_pending_resume_selection(self, text: str) -> bool:
        """Resolve a bare numeric reply that follows a bare ``/resume`` prompt.

        After ``/resume`` (no args) prints the recent-sessions list it arms
        ``self._pending_resume_sessions``. The next submitted input is given
        one chance to be a bare session number (``3``); if so we resume that
        session here. Anything else (another command, free text, blank) simply
        disarms the prompt and is handled normally by the caller.

        Returns True if the input was consumed as a resume selection (caller
        must not treat it as chat); False otherwise. The pending state is
        always one-shot: it is cleared on the first submitted input regardless
        of outcome. See #34584.
        """
        from cli import _cprint
        pending = self._pending_resume_sessions
        if not pending:
            return False
        # One-shot: disarm now so a non-matching input can't leave the prompt
        # armed and hijack a later number the user meant as chat.
        self._pending_resume_sessions = None

        if not isinstance(text, str):
            return False
        stripped = text.strip()
        # Only a pure number selects; let "/resume 3", titles, or any other
        # text fall through to normal handling.
        if not stripped.isdigit():
            return False

        index = int(stripped)
        if index < 1 or index > len(pending):
            _cprint(f"  Resume index {index} is out of range.")
            _cprint("  Use /resume with no arguments to see available sessions.")
            return True

        self._handle_resume_command(f"/resume {index}")
        return True

    def save_conversation(self, cmd: str = "/save"):
        """Handle /save — export the current session to json, md, or html.

        Usage: ``/save [json|md|html] [filename] [redact]``

        The snapshot is a convenience export for sharing or off-line
        inspection; every message is already persisted incrementally to the
        SQLite session DB, so the live session remains resumable via
        ``hermes --resume <id>`` regardless of whether the user ever runs
        ``/save``. ``redact`` runs the export through the force-mode secret
        redaction pass before writing.
        """
        from cli import datetime
        from hermes_cli.session_export import (
            SAVE_USAGE,
            normalize_save_format,
            render_session_for_save,
        )

        parts = cmd.split()[1:]
        if not parts:
            print(SAVE_USAGE)
            return
        redact = False
        if parts[-1].lower() in ("redact", "--redact"):
            redact = True
            parts = parts[:-1]
            if not parts:
                print(SAVE_USAGE)
                return

        try:
            fmt = normalize_save_format(parts[0])
        except ValueError as e:
            print(f"(._.) {e}")
            print(SAVE_USAGE)
            return
        filename = parts[1] if len(parts) > 1 else None

        # Prefer the durable DB row (has metadata + tool calls); fall back to
        # the in-memory history for sessions that never touched the DB.
        # getattr: test doubles (SimpleNamespace / object.__new__) may not
        # carry _session_db or session_id.
        session_data = None
        _db = getattr(self, "_session_db", None)
        _sid = getattr(self, "session_id", None)
        if _db and _sid:
            try:
                session_data = _db.export_session(_sid)
            except Exception:
                session_data = None
        if not session_data:
            if not self.conversation_history:
                print("(;_;) No conversation to save.")
                return
            session_data = {
                "id": self.session_id,
                "model": self.model,
                "started_at": self.session_start.timestamp(),
                "messages": self.conversation_history,
            }

        if redact:
            from hermes_cli.session_export_md import redact_session_data

            session_data = redact_session_data(session_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_dir = get_hermes_home() / "sessions" / "saved"
        try:
            saved_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"(x_x) Failed to create save directory {saved_dir}: {e}")
            return
        if filename:
            path = Path(filename).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
        else:
            path = saved_dir / f"hermes_conversation_{timestamp}.{fmt}"

        try:
            content = render_session_for_save(session_data, fmt)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            label = {"json": "JSON", "md": "Markdown", "html": "HTML"}[fmt]
            print(f"(^_^)v Conversation saved to: {path} ({label})")
            if self.session_id:
                print(f"       Resume the live session with: hermes --resume {self.session_id}")
        except Exception as e:
            print(f"(x_x) Failed to save: {e}")

    def _rewind_persisted_user_turn(
        self,
        *,
        warm_history: List[Dict[str, Any]],
        user_ordinal: int,
        warm_live_view: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Bind one warm user ordinal to a durable row and rewind it atomically."""
        if self._session_db is None or not self.session_id:
            raise RuntimeError("session database is unavailable")

        from agent.context_compressor import (
            history_before_user_originated_turn,
            split_user_originated_turn,
            user_originated_turn_view,
        )
        from agent.memory_manager import sanitize_context
        from agent.tool_dispatch_helpers import (
            _is_multimodal_tool_result,
            _multimodal_text_summary,
        )
        from run_agent import _is_ephemeral_scaffolding

        def _persistence_content(content: Any) -> Any:
            """Project warm content exactly as the session DB flush does."""
            if _is_multimodal_tool_result(content):
                return _multimodal_text_summary(content)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") in {
                        "image",
                        "image_url",
                        "input_image",
                    }:
                        text_parts.append("[screenshot]")
                return "\n".join(text_parts) if text_parts else None
            return content

        def _comparison_content(message: Dict[str, Any]) -> Any:
            content = _persistence_content(message.get("content"))
            if message.get("role") in {"user", "assistant"} and isinstance(
                content, str
            ):
                return sanitize_context(content).strip()
            return content

        expected_active_ids = self._session_db.get_active_message_ids(
            self.session_id
        )
        durable = self._session_db.get_messages_as_conversation(
            self.session_id,
            include_row_ids=True,
        )
        warm_persistence_history = [
            message
            for message in warm_history
            if not _is_ephemeral_scaffolding(message)
        ]
        warm_user_indices = [
            index
            for index, message in enumerate(warm_persistence_history)
            if user_originated_turn_view(message) is not None
        ]
        durable_user_indices = [
            index
            for index, message in enumerate(durable)
            if user_originated_turn_view(message) is not None
        ]
        if len(durable_user_indices) != len(warm_user_indices):
            raise RuntimeError(
                "session history changed before the rewind could be persisted"
            )
        if user_ordinal < 0 or user_ordinal >= len(durable_user_indices):
            raise RuntimeError("persisted rewind target is no longer available")

        warm_prefix, _ = history_before_user_originated_turn(
            warm_persistence_history, warm_user_indices[user_ordinal]
        )
        durable_target_index = durable_user_indices[user_ordinal]
        durable_target = durable[durable_target_index]
        durable_prefix, durable_live_view = history_before_user_originated_turn(
            durable, durable_target_index
        )
        if _comparison_content(durable_live_view) != _comparison_content(
            warm_live_view
        ):
            raise RuntimeError(
                "session history changed before the rewind could be persisted"
            )
        target_row_id = durable_target.get("_row_id")
        if not isinstance(target_row_id, int):
            raise RuntimeError("persisted rewind target has no row identity")
        scaffold, _ = split_user_originated_turn(durable_target)
        result = self._session_db.rewind_to_message(
            self.session_id,
            target_row_id,
            preserve_compaction_handoff=scaffold is not None,
            expected_active_ids=expected_active_ids,
            expected_target_content=durable_live_view.get("content"),
        )
        if scaffold is not None:
            replacement_id = result.get("replacement_message_id")
            if not isinstance(replacement_id, int) or not durable_prefix:
                raise RuntimeError("rewind did not retain its compaction handoff")
            durable_prefix[-1]["_row_id"] = replacement_id
            durable_prefix[-1]["_db_persisted"] = True
            warm_prefix[-1] = durable_prefix[-1]
        return warm_prefix, durable_live_view, result

    def retry_last(self):
        """Retry the last user message by removing the last exchange and re-sending.
        
        Removes the last assistant response (and any tool-call messages) and
        the last user message, then re-sends that user message to the agent.
        Returns the message to re-send, or None if there's nothing to retry.
        """
        if not self.conversation_history:
            print("(._.) No messages to retry.")
            return None
        
        # Walk backwards to the last *real* user message. Timeline bookkeeping
        # rows (display_kind set) are role=user but are not user turns — match
        # CLI resume counting and user_originated_turn_view. Compaction
        # handoffs are excluded too (durable role=user, sometimes without
        # display_kind on legacy sessions; #80622).
        from agent.context_compressor import (
            history_before_user_originated_turn,
            retryable_user_text,
            user_originated_turn_view,
        )
        from agent.memory_manager import sanitize_context
        from run_agent import _is_ephemeral_scaffolding

        warm_history = list(self.conversation_history)

        user_indices = [
            index
            for index, message in enumerate(warm_history)
            if not _is_ephemeral_scaffolding(message)
            and user_originated_turn_view(message) is not None
        ]
        
        if not user_indices:
            print("(._.) No user message found to retry.")
            return None
        last_user_idx = user_indices[-1]
        
        # Resolve a lossless live payload before touching either persistence or
        # memory. A force-user-leading compaction row is one physical carrier:
        # its historical handoff remains in the prefix while only the embedded
        # human ask is retried. Media cannot be replayed by /retry, so fail
        # closed before archiving anything.
        try:
            truncated, live_view = history_before_user_originated_turn(
                warm_history, last_user_idx
            )
            live_content = live_view.get("content")
            if isinstance(live_content, str):
                live_content = sanitize_context(live_content).strip()
            last_message = retryable_user_text(live_content)
        except ValueError as exc:
            print(f"(._.) Cannot retry that message safely: {exc}")
            return None

        # Persist the rewind before publishing the shorter in-memory view.
        # The DB owns the physical carrier split so the archived original and
        # retained scaffold are committed atomically. A plain user row keeps
        # the legacy rewind shape (no replacement scaffold).
        if self._session_db is not None and self.session_id:
            try:
                truncated, _, _ = self._rewind_persisted_user_turn(
                    warm_history=warm_history,
                    user_ordinal=len(user_indices) - 1,
                    warm_live_view=live_view,
                )
            except Exception as exc:
                print(f"(x_x) Retry rewind failed; history was not changed: {exc}")
                return None

        self.conversation_history = truncated
        if self.agent is not None:
            if hasattr(self.agent, "_session_messages"):
                self.agent._session_messages = self.conversation_history
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = len(self.conversation_history)
            if hasattr(self.agent, "_db_flush_scan_prefix"):
                self.agent._db_flush_scan_prefix = self.conversation_history[:]
        
        print(f"(^_^)b Retrying: \"{last_message[:60]}{'...' if len(last_message) > 60 else ''}\"")
        return last_message

    def undo_last(self, n: int = 1, prefill: bool = True):
        """Back up N user turns: truncate history, soft-delete on disk, prefill.

        Walks backwards N user messages and discards everything from the
        Nth-from-last user message onward (its assistant response, tool
        calls, etc.). ``n`` defaults to 1 (the last exchange); ``/undo 3``
        backs up three user turns. If ``n`` exceeds the number of user
        turns, it backs up to the oldest one.

        Beyond the in-memory ``conversation_history`` slice, this also:
          • soft-deletes the truncated rows in SessionDB (``active=0``) so
            they're hidden from re-prompts and search but kept for audit;
          • notifies memory providers via ``on_session_switch(rewound=True)``;
          • mirrors /branch's agent surgery (system-prompt invalidation +
            flush-index reset);
          • when ``prefill`` is set and an input buffer is available,
            pre-fills the composer with the backed-up message text so it
            can be edited and resubmitted.

        ``prefill=False`` is used by callers that drive the undo
        programmatically (e.g. checkpoint rollback) and don't want to
        touch the user's input buffer.
        """
        from cli import logger
        if not self.conversation_history:
            print("(._.) No messages to undo.")
            return

        if n < 1:
            n = 1

        # Walk backwards collecting the indices of the last N *real* user
        # messages (exclude display_kind timeline rows and compaction
        # handoffs — same predicate as user_originated_turn_view, resume
        # turn counting, and /retry; #80622).
        from agent.context_compressor import (
            history_before_user_originated_turn,
            user_originated_turn_view,
        )
        from run_agent import _is_ephemeral_scaffolding

        warm_history = list(self.conversation_history)

        user_indices = [
            index
            for index, message in enumerate(warm_history)
            if not _is_ephemeral_scaffolding(message)
            and user_originated_turn_view(message) is not None
        ]

        if not user_indices:
            print("(._.) No user message found to undo.")
            return

        turns_undone = min(n, len(user_indices))
        target_ordinal = len(user_indices) - turns_undone
        cut_idx = user_indices[target_ordinal]

        removed_count = len(warm_history) - cut_idx
        truncated, live_view = history_before_user_originated_turn(
            warm_history, cut_idx
        )
        removed_text = self._undo_content_to_text(live_view.get("content"))

        # Soft-delete the truncated rows on disk so re-prompts and search
        # see the clean transcript while the rows survive for audit.
        rewound_rows = 0
        if self._session_db is not None and self.session_id:
            try:
                truncated, durable_live_view, result = (
                    self._rewind_persisted_user_turn(
                        warm_history=warm_history,
                        user_ordinal=target_ordinal,
                        warm_live_view=live_view,
                    )
                )
                # Canonicalize the editable prefill before mutation. The raw
                # physical carrier contains the reference summary wrapper.
                durable_text = self._undo_content_to_text(
                    durable_live_view.get("content")
                )
                if durable_text:
                    removed_text = durable_text
                rewound_rows = result.get("rewound_count", 0)
            except Exception as e:
                logger.debug("undo: durable rewind failed: %s", e)
                print(f"(x_x) Undo failed; history was not changed: {e}")
                return

        # Publish only after the durable rewind succeeds (or no store exists).
        self.conversation_history = truncated

        # Agent surgery: invalidate the system-prompt cache and reset the
        # flush index so the next turn re-flushes from the truncated head.
        if self.agent is not None:
            if hasattr(self.agent, "_invalidate_system_prompt"):
                try:
                    self.agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(self.agent, "_last_flushed_db_idx"):
                try:
                    self.agent._last_flushed_db_idx = len(self.conversation_history)
                except Exception:
                    pass
            if hasattr(self.agent, "_session_messages"):
                self.agent._session_messages = self.conversation_history
            if hasattr(self.agent, "_db_flush_scan_prefix"):
                self.agent._db_flush_scan_prefix = self.conversation_history[:]
            # Notify memory providers — same hook /branch fires, with the
            # rewound flag so per-turn document caches invalidate (#6672, #21910).
            try:
                _mm = getattr(self.agent, "_memory_manager", None)
                if _mm is not None and self.session_id:
                    _mm.on_session_switch(
                        self.session_id,
                        parent_session_id="",
                        reset=False,
                        rewound=True,
                    )
            except Exception:
                pass

        turn_word = "turn" if turns_undone == 1 else "turns"
        msg_count = rewound_rows or removed_count
        print(
            f"(^_^)b Undid {turns_undone} {turn_word} ({msg_count} message(s)). "
            f"Backed up to: \"{removed_text[:60]}{'...' if len(removed_text) > 60 else ''}\""
        )
        remaining = len(self.conversation_history)
        print(f"  {remaining} message(s) remaining in history.")

        # Pre-fill the composer with the backed-up message so the user can
        # edit and resubmit (Claude-Code-style). Editable, not auto-sent.
        if prefill and removed_text:
            self._prefill_input_buffer(removed_text)

    @staticmethod
    def _undo_content_to_text(content) -> str:
        """Flatten message content (str or content-part list) to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(t for t in parts if t)
        return ""

    def _write_terminal_breadcrumb(self) -> None:
        """Record this terminal's live session for bare ``hermes -c``.

        Called at session start and whenever ``self.session_id`` is
        reassigned mid-run (/new, /branch, auto-compression rotation) so a
        later bare ``-c`` in THIS terminal resumes THIS conversation's live
        tip. Best-effort — never raises, no-op without a terminal identity
        or when session.terminal_continue is false.
        """
        try:
            from hermes_cli.terminal_breadcrumbs import write_breadcrumb

            write_breadcrumb(self.session_id)
        except Exception:
            pass

    def _transfer_session_yolo(self, old_session_id: str, new_session_id: str) -> None:
        """Move YOLO bypass state from an old session key to a new one.

        Called whenever ``self.session_id`` is reassigned mid-run — ``/branch``
        forks into a new session, and auto-compression rotates the agent's
        session id into a fresh continuation session. Without this transfer
        the user's ``/yolo ON`` toggle would silently revert on the very next
        turn (the same UX failure mode that motivated this entire fix), since
        ``_session_yolo`` is keyed by session id.

        Mirrors ``tui_gateway/server.py`` (~line 1297-1305) which performs the
        same transfer for the TUI's session-rename path. No-op when YOLO
        wasn't enabled or when the ids match.
        """
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return
        try:
            from tools.approval import (
                disable_session_yolo,
                enable_session_yolo,
                is_session_yolo_enabled,
            )
        except Exception:
            return
        if is_session_yolo_enabled(old_session_id):
            enable_session_yolo(new_session_id)
            disable_session_yolo(old_session_id)
            # Carry the persisted flag onto the continuation row so a later
            # `hermes --resume <new_id>` restores the bypass too. getattr
            # guard: tests call this unbound against a minimal stand-in.
            _persist = getattr(self, "_persist_session_yolo", None)
            if _persist:
                _persist(new_session_id, True)

    def _is_session_yolo_active(self) -> bool:
        """Whether YOLO bypass is currently enabled for this CLI session.

        Reads from ``tools.approval._session_yolo`` (the same set that
        ``enable_session_yolo`` / ``disable_session_yolo`` write to) so the
        status bar reflects the actual bypass state instead of a stale env
        var. Also honors the process-start ``--yolo`` flag, which freezes
        ``HERMES_YOLO_MODE`` into ``_YOLO_MODE_FROZEN`` before tool imports
        happen.
        """
        try:
            from tools.approval import (
                _YOLO_MODE_FROZEN,
                is_session_yolo_enabled,
            )
        except Exception:
            return False
        if _YOLO_MODE_FROZEN:
            return True
        # Use ``getattr`` so test fixtures that build a CLI via ``__new__``
        # (skipping ``__init__``) don't trip an AttributeError here; the
        # status-bar builders swallow exceptions silently but lose every
        # field after the failure.
        session_key = getattr(self, "session_id", None) or "default"
        return is_session_yolo_enabled(session_key)

    def _toggle_yolo(self):
        """Toggle YOLO mode — skip all dangerous command approval prompts.

        Per-session toggle that mirrors the gateway and TUI ``/yolo`` handlers
        (see ``gateway/run.py:_handle_yolo_command`` and
        ``tui_gateway/server.py`` key=="yolo"). We deliberately do NOT mutate
        ``HERMES_YOLO_MODE`` here — that env var is read once at module import
        time into ``tools.approval._YOLO_MODE_FROZEN`` to keep prompt-injected
        skills from flipping the bypass mid-session, so setting it after CLI
        startup is a silent no-op. Routing through ``enable_session_yolo`` /
        ``disable_session_yolo`` gives the same auditable, per-session bypass
        the other surfaces have. ``run_conversation`` binds
        ``self.session_id`` as the active approval session key via
        ``set_current_session_key`` so the bypass takes effect on the very
        next dangerous command in this run.
        """
        from cli import _cprint
        from hermes_cli.colors import Colors as _Colors
        from tools.approval import (
            _YOLO_MODE_FROZEN,
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        # Process-level YOLO (--yolo flag / HERMES_YOLO_MODE at startup) is
        # frozen into tools.approval at import time and cannot be disabled by
        # the session toggle. Before this guard, /yolo printed "YOLO mode OFF —
        # dangerous commands will require approval" while every command kept
        # auto-approving (the frozen flag short-circuits the approval gate
        # ahead of the session check) — a false safety claim. Say the truth
        # instead of toggling a bypass that has no effect.
        if _YOLO_MODE_FROZEN:
            _cprint(
                f"  ⚡ YOLO is {_Colors.BOLD}{_Colors.RED}locked ON{_Colors.RESET}"
                " for this process (started with --yolo / HERMES_YOLO_MODE)."
                " /yolo cannot disable it — restart without the flag to"
                " re-enable approvals."
            )
            return

        session_key = self.session_id or "default"
        # ``getattr`` guard: tests exercise this method unbound against a
        # minimal stand-in object (see tests/cli/test_cli_yolo_toggle.py);
        # persistence is best-effort either way.
        _persist = getattr(self, "_persist_session_yolo", None)
        if is_session_yolo_enabled(session_key):
            disable_session_yolo(session_key)
            if _persist:
                _persist(session_key, False)
            _cprint(
                f"  ⚠ YOLO mode {_Colors.BOLD}{_Colors.RED}OFF{_Colors.RESET}"
                " — dangerous commands will require approval."
            )
        else:
            enable_session_yolo(session_key)
            if _persist:
                _persist(session_key, True)
            _cprint(
                f"  ⚡ YOLO mode {_Colors.BOLD}{_Colors.GREEN}ON{_Colors.RESET}"
                " — all commands auto-approved. Use with caution."
            )

    def _persist_session_yolo(self, session_key: str, enabled: bool) -> None:
        """Persist the YOLO flag to the session row so --resume restores it.

        Best-effort: the in-memory toggle is authoritative for this process;
        persistence only affects a future ``hermes --resume``. Skipped when the
        session store is unavailable or the row doesn't exist yet (the row is
        created lazily on the first turn — ``_toggle_yolo`` before any chat
        writes nothing, and the launch-time ``--yolo`` flag is carried into the
        creation-time model_config instead).
        """
        db = getattr(self, "_session_db", None)
        if db is None or not session_key or session_key == "default":
            return
        try:
            db.set_session_yolo(session_key, enabled)
        except Exception:
            pass

    def _manual_compress(self, cmd_original: str = ""):
        """Manually trigger context compression on the current conversation.

        Two modes:

        * ``/compress [<focus>]`` — compress the *whole* history. An
          optional focus topic guides the summariser to preserve
          information related to *focus* while being more aggressive
          about discarding everything else.  Inspired by Claude Code's
          ``/compact <focus>`` feature.
        * ``/compress here [N]`` — boundary-aware compression. Summarize
          everything *except* the most recent ``N`` exchanges (default
          2), which are preserved verbatim. Inspired by Claude Code's
          Rewind "Summarize up to here" action (v2.1.139, May 2026,
          https://code.claude.com/docs/en/whats-new/2026-w20). Lets the
          user pick the compression boundary instead of leaving it to
          the automatic token-budget heuristic.
        """
        if not self.conversation_history or len(self.conversation_history) < 4:
            print("(._.) Not enough conversation to compress (need at least 4 messages).")
            return

        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        # No compression_enabled gate here: the config flag disables
        # *automatic* compaction only. Manual /compress is an explicit user
        # action — the context-overflow error path (conversation_loop.py)
        # directs users here when auto-compaction is off, and the gateway's
        # /compress handler has never gated on the flag.

        from hermes_cli.partial_compress import (
            extract_compress_flags,
            parse_partial_compress_args,
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
            summarize_compress_preview,
        )
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )

        # Args after the command word (e.g. "/compress here 3" -> "here 3").
        raw_args = ""
        if cmd_original:
            _parts = cmd_original.strip().split(None, 1)
            if len(_parts) > 1:
                raw_args = _parts[1].strip()

        # Strip --preview/--dry-run/--aggressive before positional parsing
        # so the flags coexist with 'here [N]' / focus-topic forms.
        raw_args, preview, aggressive = extract_compress_flags(raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(raw_args)
        focus_topic = focus_topic or ""

        if aggressive:
            # LLM-free hard truncation is not supported: it would need its
            # own transcript-persistence path outside the guarded
            # _compress_context rotation machinery. Surface that instead of
            # silently mis-parsing the flag as a focus topic.
            print("(._.) --aggressive is not supported; use '/compress here [N]' "
                  "to keep only recent exchanges, or /undo to drop turns.")
            if not preview:
                return

        if preview:
            from agent.model_metadata import estimate_request_tokens_rough
            _sys_prompt = getattr(self.agent, "_cached_system_prompt", "") or ""
            _tools = getattr(self.agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                self.conversation_history,
                system_prompt=_sys_prompt,
                tools=_tools,
            )
            report = summarize_compress_preview(
                self.conversation_history,
                partial,
                keep_last,
                focus_topic or None,
                approx_tokens,
            )
            for line in report["lines"]:
                print(f"🗜️  {line}")
            return

        original_count = len(self.conversation_history)
        with self._busy_command("Compressing context...", blocks_input=False):
            try:
                from agent.model_metadata import estimate_request_tokens_rough
                from agent.manual_compression_feedback import summarize_manual_compression
                original_history = list(self.conversation_history)

                # Boundary-aware split: only the head is summarized; the
                # most recent `keep_last` exchanges ride along verbatim.
                tail: list = []
                head = original_history
                if partial:
                    head, tail = split_history_for_partial_compress(
                        original_history, keep_last
                    )
                    if not tail:
                        # Split degenerated (everything would be kept, or
                        # no head left to compress). Fall back to full
                        # compression so the user still gets an action.
                        partial = False
                        head = original_history

                # Include system prompt + tool schemas in the estimate —
                # a transcript-only number understates real request pressure
                # and can even appear to grow after compression because a
                # dense handoff summary replaces many short turns (#6217).
                _sys_prompt = getattr(self.agent, "_cached_system_prompt", "") or ""
                _tools = getattr(self.agent, "tools", None) or None
                approx_tokens = estimate_request_tokens_rough(
                    original_history,
                    system_prompt=_sys_prompt,
                    tools=_tools,
                )
                if partial:
                    print(f"🗜️  Summarizing up to here: compressing {len(head)} of "
                          f"{original_count} messages (~{approx_tokens:,} tokens), "
                          f"keeping last {keep_last} exchange(s) verbatim...")
                elif focus_topic:
                    print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens), "
                          f"focus: \"{focus_topic}\"...")
                else:
                    print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens)...")

                # Pass None as system_message so _compress_context rebuilds
                # the system prompt from scratch via _build_system_prompt(None).
                # Passing _cached_system_prompt caused duplication because
                # _build_system_prompt appends system_message to prompt_parts
                # which already contain the agent identity — resulting in the
                # identity block appearing twice (issue #15281).
                compressed, _ = self.agent._compress_context(
                    head,
                    None,
                    approx_tokens=approx_tokens,
                    focus_topic=focus_topic or None,
                    force=True,
                    defer_context_engine_notification=True,
                )

                # If _compress_context returned unchanged because a
                # concurrent compression lock is held, tell the user
                # clearly instead of showing the misleading
                # "No changes from compression" no-op text. The wording
                # distinguishes a confirmed holder from an unconfirmed
                # acquisition failure (describe_compression_lock_skip).
                # Type-pinned check (is True / str): the flag's only real
                # values are None/True/holder-string, and a bare getattr
                # truthiness test is fooled by MagicMock auto-attributes on
                # test-double agents (skill pitfall: MagicMock vs hasattr).
                _lock_skip_signal = getattr(
                    self.agent, "_compression_skipped_due_to_lock", None
                )
                if _lock_skip_signal is True or isinstance(_lock_skip_signal, str):
                    from agent.manual_compression_feedback import (
                        describe_compression_lock_skip,
                    )
                    print(
                        "  "
                        + describe_compression_lock_skip(
                            self.agent._compression_skipped_due_to_lock
                        )
                    )
                    self.agent._compression_skipped_due_to_lock = None
                    # No boundary was committed on a lock-skip; discard the
                    # deferred context-engine notification (exactly-once).
                    finalize_context_engine_compression_notification(
                        self.agent,
                        committed=False,
                    )
                    return

                if partial and tail:
                    compressed = rejoin_compressed_head_and_tail(compressed, tail)
                self.conversation_history = compressed
                # _compress_context ends the old session and creates a new child
                # session on the agent (run_agent.py::_compress_context). Sync the
                # CLI's session_id so /status, /resume, exit summary, and title
                # generation all point at the live continuation session, not the
                # ended parent. Without this, subsequent end_session() calls target
                # the already-closed parent and the child is orphaned.
                if (
                    getattr(self.agent, "session_id", None)
                    and self.agent.session_id != self.session_id
                ):
                    self.session_id = self.agent.session_id
                    getattr(self, "_write_terminal_breadcrumb", lambda: None)()
                    self._pending_title = None
                    # Manual /compress replaces conversation_history with a new
                    # compressed handoff for the child session. Persist it from
                    # offset 0 so resume can recover the continuation after exit.
                    self.agent._flush_messages_to_session_db(self.conversation_history, None)
                finalize_context_engine_compression_notification(
                    self.agent,
                    committed=True,
                )
                new_tokens = estimate_request_tokens_rough(
                    self.conversation_history,
                    system_prompt=_sys_prompt,
                    tools=_tools,
                )
                summary = summarize_manual_compression(
                    original_history,
                    self.conversation_history,
                    approx_tokens,
                    new_tokens,
                    compression_state=getattr(
                        self.agent, "context_compressor", None
                    ),
                )
                if (
                    summary.get("aborted")
                    or summary.get("fallback_used")
                    or summary.get("refused_would_grow")
                ):
                    icon = "⚠️"
                else:
                    icon = "🗜️" if summary["noop"] else "✅"
                print(f"  {icon} {summary['headline']}")
                print(f"     {summary['token_line']}")
                if summary["note"]:
                    print(f"     {summary['note']}")

            except Exception as e:
                finalize_context_engine_compression_notification(
                    self.agent,
                    committed=False,
                )
                print(f"  ❌ Compression failed: {e}")

    def _persist_prompt_summary(self, icon: str, label: str, detail: str, outcome: str) -> None:
        """Print a one-line scrollback summary of a resolved modal prompt.

        Modal panels (approval / clarify) live in the prompt_toolkit layout and
        vanish on the next repaint, so the question and the decision leave no
        trace in the terminal scrollback. When display.persist_prompts is on
        (default), emit a dim single line after the prompt resolves so the
        decision survives in chat history.
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        if not CLI_CONFIG.get("display", {}).get("persist_prompts", True):
            return
        detail = " ".join(detail.split())
        if len(detail) > 120:
            detail = detail[:119] + "…"
        outcome = " ".join(outcome.split())
        if len(outcome) > 120:
            outcome = outcome[:119] + "…"
        _cprint(f"\n{_DIM}{icon} {label}: {detail} → {outcome}{_RST}")

    def _clear_terminal_on_exit(self):
        """Clear screen + scrollback so nothing is stranded above the exit summary.

        Called from ``_print_exit_summary`` after ``app.run()`` has returned and
        prompt_toolkit has torn down its renderer + restored terminal modes —
        so a direct write to the real stdout fd is safe (the StdoutProxy /
        patch_stdout layer is gone by now).

        Sequence: ``ESC[3J`` (erase scrollback) + ``ESC[2J`` (erase visible
        screen) + ``ESC[H`` (cursor home). Modern terminals on Linux, macOS and
        Windows (Terminal / conhost with VT processing, which prompt_toolkit
        already enables) all honor these. Best-effort: skip silently when
        stdout isn't a real console, and fall back to the platform ``clear`` /
        ``cls`` command if the escape write fails.
        """
        try:
            stream = sys.stdout
            if stream is None or not stream.isatty():
                return
        except Exception:
            return
        try:
            stream.write("\033[3J\033[2J\033[H")
            stream.flush()
            return
        except Exception:
            pass
        # Fallback: shell clear command (rarely needed — escapes work on every
        # VT-capable terminal, but this covers exotic stdout wrappers).
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            pass

    def _persist_active_session_before_close(self):
        """Best-effort SQLite/JSON flush before the CLI marks a session closed.

        ``run_conversation()`` normally persists at turn boundaries, but a
        terminal close/SIGHUP/SIGTERM can unwind the prompt_toolkit app while
        the agent thread still holds the current turn only in memory.  Flush the
        agent's live ``_session_messages`` before ``end_session()`` so resume,
        session_search, and state.db do not lose the interrupted turn.
        """
        from cli import logger
        agent = getattr(self, "agent", None)
        if not agent or not hasattr(agent, "_persist_session"):
            return

        persist_lock = getattr(agent, "_session_persist_lock", None)

        def _snapshot_and_persist() -> None:
            # This snapshot must share the staging lock with ``chat()``. Without
            # it, close can retain a mutable history baseline just before chat
            # appends its pending dict; the later flush then mistakes that dict
            # for durable history and stamps it without writing a row (#63766).
            messages = getattr(agent, "_session_messages", None)
            pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
            if not isinstance(messages, list):
                messages = getattr(self, "conversation_history", None)
            if not isinstance(messages, list):
                return
            if isinstance(pending_cli_message, dict) and not any(
                message is pending_cli_message for message in messages
            ):
                # The UI has accepted a new input but the worker still exposes its
                # prior snapshot. Include only that staged dict; the baseline below
                # keeps any durable resumed prefix from being re-appended.
                messages = [*messages, pending_cli_message]
            if not messages:
                return

            # A normal turn builds a new list that reuses the resumed-history dicts.
            # Keep that CLI history as the baseline so a signal between assigning
            # ``_session_messages`` and the turn's DB flush cannot append its durable
            # prefix a second time. Once the CLI takes the turn result, however, both
            # names can point at the same live list; passing that alias would mark an
            # unflushed tail durable without writing it. Marker-only persistence is
            # correct only in that alias case.
            conversation_history = getattr(self, "conversation_history", None)
            pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
            if (
                isinstance(conversation_history, list)
                and conversation_history
                and conversation_history[-1] is pending_cli_message
            ):
                # The UI accepted this user message before the agent finished its
                # early persistence. Its dict can already be in ``messages`` but is
                # not durable yet, so exclude it from the resumed-history baseline.
                conversation_history = conversation_history[:-1]
            elif not isinstance(conversation_history, list) or conversation_history is messages:
                conversation_history = None

            # A first-turn close can arrive before the worker builds its cached
            # prompt. Build or restore it before the DB row is created so the
            # durable transcript never leaves a NULL system_prompt cache entry.
            if getattr(agent, "_cached_system_prompt", None) is None:
                try:
                    from agent.conversation_loop import _restore_or_build_system_prompt

                    _restore_or_build_system_prompt(agent, None, conversation_history)
                except Exception:
                    logger.debug("Could not build system prompt during CLI close", exc_info=True)
                    return
            if getattr(agent, "_cached_system_prompt", None) is None:
                return

            agent._ensure_db_session()
            agent._persist_session(messages, conversation_history)
            if getattr(agent, "session_id", None):
                self.session_id = agent.session_id
                getattr(self, "_write_terminal_breadcrumb", lambda: None)()

        try:
            if persist_lock is None:
                _snapshot_and_persist()
            else:
                with persist_lock:
                    _snapshot_and_persist()
        except (Exception, KeyboardInterrupt) as e:
            logger.debug("Could not persist active CLI session before close: %s", e)

    def _print_exit_summary(self, clear_screen: bool = True):
        """Print session resume info on exit, similar to Claude Code.

        Args:
            clear_screen: When True (default), clear the terminal screen and
                scrollback before printing the summary. This is appropriate for
                interactive TUI teardown (#38252). Single-query (-q) mode should
                pass False to preserve the printed answer (#53009).
        """
        from cli import datetime
        if clear_screen:
            # Clear the screen + scrollback before printing the summary so the
            # live bottom chrome (status bar, input box, separator rules) and the
            # rest of the session transcript don't get stranded above the exit
            # summary (#38252). By this point app.run() has returned and
            # prompt_toolkit has restored terminal modes, so writing raw escapes
            # to stdout is safe. ESC[3J clears scrollback, ESC[2J clears the
            # visible screen, ESC[H homes the cursor — so the summary prints at a
            # clean top-left. Falls back to the platform clear command if stdout
            # isn't a TTY-capable stream. Honors NO_COLOR/dumb terminals by
            # skipping silently when there's no real console.
            self._clear_terminal_on_exit()
        print()
        msg_count = len(self.conversation_history)
        if msg_count > 0:
            user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
            tool_calls = len([m for m in self.conversation_history if m.get("role") == "tool" or m.get("tool_calls")])
            elapsed = datetime.now() - self.session_start
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
            
            # Look up session title for resume-by-name hint
            session_title = None
            if self._session_db:
                try:
                    session_title = self._session_db.get_session_title(self.session_id)
                except Exception:
                    pass

            print("Resume this session with:")
            # Session IDs are profile-constrained, so the resume hint must
            # include `-p <profile>` for non-default profiles. Without this,
            # copying the hint from a non-default profile fails to find the
            # session on the next invocation. The "default" and "custom"
            # profile names use the standard HERMES_HOME, so no -p needed.
            try:
                from hermes_cli.profiles import get_active_profile_name
                _active_profile = get_active_profile_name()
            except Exception:
                _active_profile = "default"
            profile_flag = (
                "" if _active_profile in ("default", "custom") else f" -p {_active_profile}"
            )
            print(f"  hermes --resume {self.session_id}{profile_flag}")
            if session_title:
                print(f"  hermes -c \"{session_title}\"{profile_flag}")
            print()
            print(f"Session:        {self.session_id}")
            if session_title:
                print(f"Title:          {session_title}")
            print(f"Duration:       {duration_str}")
            print(f"Messages:       {msg_count} ({user_msgs} user, {tool_calls} tool calls)")
        else:
            try:
                from hermes_cli.skin_engine import get_active_goodbye
                goodbye = get_active_goodbye("Goodbye! ⚕")
            except Exception:
                goodbye = "Goodbye! ⚕"
            print(goodbye)
