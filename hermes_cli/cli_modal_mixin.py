"""Modal overlays for the interactive CLI: clarify, approval, sudo/secret capture, command palette, slash-confirm, external editor

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import json
import queue
import sys

from hermes_cli.callbacks import prompt_for_secret
from typing import Optional


class CLIModalMixin:
    """Modal overlays for the interactive CLI: clarify, approval, sudo/secret capture, command palette, slash-confirm, external editor"""

    def _open_external_editor(self, buffer=None) -> bool:
        """Open the active input buffer in an external editor."""
        from cli import _DIM, _RST, _cprint
        app = getattr(self, "_app", None)
        if not app:
            _cprint(f"{_DIM}External editor is only available inside the interactive CLI.{_RST}")
            return False
        if self._command_running:
            _cprint(f"{_DIM}Wait for the current command to finish before opening the editor.{_RST}")
            return False
        if self._sudo_state or self._secret_state or self._approval_state or getattr(self, "_slash_confirm_state", None) or self._clarify_state:
            _cprint(f"{_DIM}Finish the active prompt before opening the editor.{_RST}")
            return False
        target_buffer = buffer or getattr(app, "current_buffer", None)
        if target_buffer is None:
            _cprint(f"{_DIM}No active input buffer is available for the external editor.{_RST}")
            return False
        try:
            # Inline pastes so the editor (and the draft it submits) sees real
            # content; skip flag unconditionally so the editor-close text-change
            # doesn't re-collapse it, even when there was nothing to inline.
            self._inline_pastes(target_buffer)
            self._skip_paste_collapse = True
            # Open the editor, then submit the saved draft on a clean exit —
            # matching the TUI's Ctrl+G (openEditor), which sends the buffer
            # instead of requiring a second Enter. Submission in this CLI is
            # driven by the custom `enter` keybinding, NOT the buffer's
            # accept_handler, so validate_and_handle can't route through it;
            # chain a done-callback on the returned Task that re-uses the
            # real submit pipeline via _submit_editor_buffer().
            task = target_buffer.open_in_editor(validate_and_handle=False)
            if task is not None and hasattr(task, "add_done_callback"):
                task.add_done_callback(
                    lambda _t, b=target_buffer: self._submit_editor_buffer(b)
                )
            return True
        except Exception as exc:
            _cprint(f"{_DIM}Failed to open external editor: {exc}{_RST}")
            return False

    def _submit_editor_buffer(self, buffer) -> None:
        """Submit the draft an external editor left in ``buffer``.

        Invoked from the Ctrl+G done-callback so saving the editor sends the
        prompt (TUI parity) instead of leaving it sitting in the input area.
        Mirrors the idle/queue branches of the `enter` keybinding handler:
        an empty save is ignored (never submits a blank turn), a slash command
        is dispatched, otherwise the text is routed through the same input
        queues the normal Enter path uses. Runs on the prompt_toolkit event
        loop via the Task callback, so it must be cheap and non-blocking.
        """
        from cli import _DIM, _RST, _cprint, _looks_like_slash_command
        try:
            text = (getattr(buffer, "text", "") or "").strip()
        except Exception:
            return
        if not text:
            # Editor saved empty / was cleared — match the TUI, which drops
            # an empty draft instead of submitting a blank turn.
            return

        app = getattr(self, "_app", None)

        # `!<command>` shell mode, checked before slash dispatch — matches the
        # Enter path in the input loop so an editor-saved bang command runs
        # locally instead of being sent to the agent.
        try:
            if self.handle_bang_shell(text):
                self._reset_input_buffer(buffer)
                if app is not None:
                    app.invalidate()
                return
        except Exception as exc:
            _cprint(f"  {_DIM}Shell command failed: {exc}{_RST}")
            self._reset_input_buffer(buffer)
            if app is not None:
                app.invalidate()
            return

        # Slash commands: dispatch directly, same as the Enter handler's
        # _looks_like_slash_command branch.
        if _looks_like_slash_command(text):
            try:
                if not self.process_command(text):
                    self._should_exit = True
                    if app is not None and app.is_running:
                        app.exit()
            except Exception as exc:
                _cprint(f"  {_DIM}Command failed: {exc}{_RST}")
            finally:
                self._reset_input_buffer(buffer)
                if app is not None:
                    app.invalidate()
            return

        # Regular prompt: route through the same queues the Enter handler uses.
        if self._agent_running:
            # Agent busy → honour the configured busy-input behaviour by
            # queueing for the next turn (the safe default; interrupt/steer
            # remain reachable via the normal Enter path).
            self._interrupt_queue.put(text) if self.busy_input_mode == "interrupt" else self._pending_input.put(text)
            preview = text[:80] + ("..." if len(text) > 80 else "")
            _cprint(f"  Queued for the next turn: {preview}")
        else:
            self._pending_input.put(text)

        self._reset_input_buffer(buffer)
        if app is not None:
            app.invalidate()

    def _inline_pastes(self, buffer) -> None:
        """Replace collapsed-paste placeholders in ``buffer`` with real content.

        A big paste shows as a compact ``[Pasted text #N -> file]`` placeholder,
        but history recall and the external editor need the actual text — a bare
        reference is useless once the file is gone or on another machine. Inlining
        before ``reset(append_to_history=True)`` also lets prompt_toolkit persist
        the content through its normal path. Sets ``_skip_paste_collapse`` so the
        ensuing text-change doesn't re-collapse it.
        """
        from cli import logger
        try:
            existing = getattr(buffer, "text", "")
            expanded = self._expand_paste_references(existing)
            if expanded != existing and hasattr(buffer, "text"):
                self._skip_paste_collapse = True
                buffer.text = expanded
                if hasattr(buffer, "cursor_position"):
                    buffer.cursor_position = len(expanded)
        except Exception:
            logger.debug("Failed to inline paste placeholders", exc_info=True)

    def _reset_input_buffer(self, buffer) -> None:
        """Clear an input buffer after a programmatic submit (best-effort)."""
        try:
            buffer.reset(append_to_history=True)
        except Exception:
            try:
                buffer.text = ""
            except Exception:
                pass

    def _prefill_input_buffer(self, text: str) -> None:
        """Place ``text`` in the active prompt_toolkit buffer, editable."""
        from cli import logger
        app = getattr(self, "_app", None)
        if app is None:
            return
        try:
            buf = app.current_buffer
            buf.text = text
            if hasattr(buf, "cursor_position"):
                buf.cursor_position = len(text)
            app.invalidate()
        except Exception as e:
            logger.debug("undo: prefill buffer failed: %s", e)

    def _prompt_text_input(self, prompt_text: str) -> str | None:
        """Prompt for free-text input safely inside or outside prompt_toolkit.

        ``run_in_terminal`` returns a coroutine that must be awaited by the prompt_toolkit event loop,
        which only exists on the main thread.  Slash commands are dispatched from
        the ``process_loop`` daemon thread (see issue #23185), so calling
        ``run_in_terminal`` from there orphans the coroutine — ``_ask`` never runs,
        and user keystrokes leak into the composer instead.  Fall back to a direct
        ``input()`` when we're off the main thread.
        """
        import threading
        result = [None]

        def _ask():
            try:
                result[0] = input(prompt_text).strip() or None
            except (KeyboardInterrupt, EOFError):
                pass

        in_main_thread = threading.current_thread() is threading.main_thread()

        # Slash-worker guard (#23185 / billing auto-reload hang): when a
        # prompt_toolkit app is running but we're on a non-main thread (the
        # process_loop / TUI slash-worker daemon thread), stdin is owned by the
        # event loop / JSON-RPC pipe.  A bare input() there blocks forever until
        # the worker's 45s timeout fires.  We cannot safely prompt off the main
        # thread, so cancel cleanly (None) instead of hanging — mirrors the
        # _stdin_fallback discipline in _prompt_text_input_modal.
        if self._app and not in_main_thread:
            self._invalidate()
            return None

        if self._app and in_main_thread:
            from prompt_toolkit.application import run_in_terminal
            was_visible = self._status_bar_visible
            self._status_bar_visible = False
            self._app.invalidate()
            try:
                run_in_terminal(_ask)
            except Exception:
                # WSL / Warp / certain terminal emulators silently drop the
                # scheduled coroutine.  Fall back to a direct input() so the
                # user's keystrokes don't leak into the agent buffer.
                try:
                    _ask()
                except Exception:
                    pass
            finally:
                self._status_bar_visible = was_visible
                self._app.invalidate()
        else:
            _ask()
        return result[0]

    def _prompt_text_input_modal(
        self,
        *,
        title: str,
        detail: str,
        choices: list[tuple[str, str, str]],
        timeout: float = 120,
    ) -> str | None:
        """Prompt through the prompt_toolkit composer instead of raw input().

        This is for CLI slash-command confirmations.  The old raw input() path
        fought prompt_toolkit's active stdin ownership: in some terminals the
        prompt appeared above the TUI, choices were redrawn later, and Enter
        could be interpreted as EOF/exit.  A first-class modal state keeps the
        choices visible and lets the normal Enter key binding submit the typed
        or highlighted choice.

        **Platform note (Windows — issue #33961):**
        Earlier code bypassed the modal on ``sys.platform == "win32"`` and fell
        back to a raw ``input()`` prompt.  When the confirm was triggered from the
        ``process_loop`` daemon thread (the normal case) that ``input()`` ran off
        the main thread and deadlocked against prompt_toolkit's stdin ownership —
        the user saw a frozen cursor and Ctrl-C was swallowed (bare ``/reset``
        froze; ``/reset now`` worked only because it skips the prompt entirely).

        Native Windows now uses the same path as Linux/macOS: the modal is set up
        on ``self._app.loop`` via ``call_soon_threadsafe`` and answered by the
        normal prompt_toolkit key bindings (the same input channel that already
        handles ordinary typing on Windows).  The raw ``input()`` fallback is kept
        only for the genuinely safe cases: no running app (unit tests /
        non-interactive), no resolvable event loop, or a scheduling failure.
        """
        import threading
        import time as _time

        if not choices:
            return None

        # If prompt_toolkit is not running (unit tests / non-interactive calls),
        # keep the simple stdin fallback.
        if not getattr(self, "_app", None):
            return self._prompt_text_input("Choice [1/2/3]: ")

        try:
            app_loop = self._app.loop
        except Exception:
            app_loop = None

        in_main_thread = threading.current_thread() is threading.main_thread()

        def _stdin_fallback() -> str | None:
            # On native Windows a raw input() from a non-main thread deadlocks
            # against prompt_toolkit's stdin ownership (#33961).  With an app
            # running we cannot safely prompt off the main thread, so cancel
            # cleanly (None) rather than hang the terminal.
            if sys.platform == "win32" and not in_main_thread:
                self._invalidate()
                return None
            return self._prompt_text_input("Choice [1/2/3]: ")

        if not in_main_thread and app_loop is None:
            return _stdin_fallback()

        response_queue = queue.Queue()

        def _setup_modal() -> None:
            self._capture_modal_input_snapshot()
            self._slash_confirm_state = {
                "title": title,
                "detail": detail,
                "choices": choices,
                "selected": 0,
                "response_queue": response_queue,
            }
            self._slash_confirm_deadline = _time.monotonic() + timeout
            self._invalidate()

        def _teardown_modal() -> None:
            self._slash_confirm_state = None
            self._slash_confirm_deadline = 0
            self._restore_modal_input_snapshot()
            self._invalidate()

        def _run_on_app_loop(fn) -> bool:
            if in_main_thread or app_loop is None:
                fn()
                return True
            ready = threading.Event()

            def _wrapped() -> None:
                try:
                    fn()
                finally:
                    ready.set()

            try:
                app_loop.call_soon_threadsafe(_wrapped)
            except Exception:
                return False
            return ready.wait(timeout=5)

        if not _run_on_app_loop(_setup_modal):
            return _stdin_fallback()

        _last_countdown_refresh = _time.monotonic()
        try:
            while True:
                try:
                    result = response_queue.get(timeout=1)
                    _run_on_app_loop(_teardown_modal)
                    return result
                except queue.Empty:
                    remaining = self._slash_confirm_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                    now = _time.monotonic()
                    if now - _last_countdown_refresh >= 5.0:
                        _last_countdown_refresh = now
                        self._invalidate()
        finally:
            if self._slash_confirm_state is not None:
                _run_on_app_loop(_teardown_modal)
        return None

    def _submit_slash_confirm_response(self, value: str | None) -> None:
        state = self._slash_confirm_state
        if not state:
            return
        state["response_queue"].put(value)
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._invalidate()

    def _normalize_slash_confirm_choice(
        self,
        raw: str | None,
        choices: list[tuple[str, str, str]],
    ) -> str | None:
        if raw is None:
            return None
        choice_raw = raw.strip().lower()
        if not choice_raw:
            return None
        aliases = {
            "1": "once",
            "once": "once",
            "approve": "once",
            "yes": "once",
            "y": "once",
            "ok": "once",
            "2": "always",
            "always": "always",
            "remember": "always",
            "3": "cancel",
            "cancel": "cancel",
            "nevermind": "cancel",
            "no": "cancel",
            "n": "cancel",
        }
        allowed = {choice[0] for choice in choices}
        normalized = aliases.get(choice_raw)
        if normalized in allowed:
            return normalized
        if choice_raw in allowed:
            return choice_raw
        return None

    def _build_command_palette_entries(self) -> list:
        """Flat list of (command, description) for the Ctrl+P palette.

        Sourced from the same COMMAND_REGISTRY that backs /help, filtered to
        commands available on this surface, plus installed skill commands.
        Selecting an entry inserts the exact command string — never a fuzzy
        resolution.
        """
        from cli import _ensure_skill_commands
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        entries: list[tuple[str, str, str]] = []  # (command, category, desc)
        for category, commands in COMMANDS_BY_CATEGORY.items():
            for cmd, desc in commands.items():
                if not self._command_available(cmd):
                    continue
                entries.append((cmd, category, desc))
        try:
            for cmd, info in sorted(_ensure_skill_commands().items()):
                entries.append((cmd, "Skill", info.get("description", "")))
        except Exception:
            pass
        return entries

    def _open_command_palette(self) -> None:
        """Open the Ctrl+P fuzzy command palette modal."""
        if getattr(self, "_command_palette_state", None):
            return
        # Don't stack over other modals.
        if (self._model_picker_state or self._clarify_state or self._approval_state
                or self._slash_confirm_state or self._sudo_state or self._secret_state):
            return
        self._capture_modal_input_snapshot()
        self._command_palette_state = {
            "entries": self._build_command_palette_entries(),
            "filter": "",
            "selected": 0,
            "_scroll_offset": 0,
        }
        self._invalidate(min_interval=0.0)

    def _close_command_palette(self) -> None:
        self._command_palette_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _command_palette_visible_entries(self) -> list:
        """Return (command, category, desc) rows matching the active filter.

        Ranked, command-name-focused matching (a bare subsequence over the
        whole "cmd category desc" string is uselessly permissive — "steer"
        would match 130+ rows via description text). Priority:
          0 exact command match
          1 command startswith query
          2 query substring in command
          3 query subsequence in command
          4 query substring in description
        Rows that match nowhere are dropped. Ties keep registry order.
        """
        state = self._command_palette_state or {}
        entries = state.get("entries") or []
        q = (state.get("filter", "") or "").strip().lower()
        if not q:
            return list(entries)

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        ranked = []
        for order, row in enumerate(entries):
            cmd, _cat, desc = row
            name = cmd.lower().lstrip("/")
            qn = q.lstrip("/")
            desc_l = (desc or "").lower()
            if name == qn:
                rank = 0
            elif name.startswith(qn):
                rank = 1
            elif qn in name:
                rank = 2
            elif _subseq(qn, name):
                rank = 3
            elif q in desc_l:
                rank = 4
            else:
                continue
            ranked.append((rank, order, row))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return [row for (_r, _o, row) in ranked]

    def _handle_command_palette_selection(self) -> None:
        """Insert the selected command into the composer (does not auto-run)."""
        from cli import logger
        state = self._command_palette_state
        if not state:
            return
        rows = self._command_palette_visible_entries()
        selected = state.get("selected", 0)
        if not (0 <= selected < len(rows)):
            self._close_command_palette()
            return
        cmd = rows[selected][0]  # exact command string, e.g. "/model"
        self._close_command_palette()
        # Prefill the composer so the user can add args / confirm — never
        # auto-execute (a palette pick should be explicit, and many commands
        # take arguments).
        try:
            app = getattr(self, "_app", None)
            if app is not None:
                buf = app.current_buffer
                buf.text = cmd + " "
                buf.cursor_position = len(buf.text)
                self._invalidate(min_interval=0.0)
        except Exception:
            logger.debug("command palette prefill failed", exc_info=True)

    @classmethod
    def _split_destructive_skip(cls, cmd_text: Optional[str]) -> tuple[str, bool]:
        """Split inline-skip tokens out of a destructive slash command.

        Returns ``(remainder, skip)`` where ``remainder`` is the original
        text with the command word and any recognized skip tokens removed,
        and ``skip`` is True iff at least one skip token was found.

        Examples:
            "/reset now"            -> ("", True)
            "/reset --yes My title" -> ("My title", True)
            "/new My title"         -> ("My title", False)
            "/clear"                -> ("", False)
        """
        if not cmd_text:
            return "", False
        tokens = cmd_text.strip().split()
        if not tokens:
            return "", False
        # Drop leading "/cmd" word — callers pass the full command text.
        if tokens[0].startswith("/"):
            tokens = tokens[1:]
        skip = False
        kept: list[str] = []
        for tok in tokens:
            if tok.lower() in cls._DESTRUCTIVE_SKIP_TOKENS:
                skip = True
                continue
            kept.append(tok)
        return " ".join(kept), skip

    def _confirm_destructive_slash(
        self,
        command: str,
        detail: str,
        cmd_original: Optional[str] = None,
    ) -> Optional[str]:
        """Prompt the user to confirm a destructive session slash command.

        Used by ``/clear``, ``/new``/``/reset``, and ``/undo`` before they
        discard conversation state.  Three-option prompt:

          1. Approve Once — proceed this time only
          2. Always Approve — proceed and persist
             ``approvals.destructive_slash_confirm: false`` so future
             destructive commands run without confirmation
          3. Cancel — abort

        Gated by ``approvals.destructive_slash_confirm`` (default on).  If the
        gate is off the function returns ``"once"`` immediately without
        prompting.

        Inline-skip: if ``cmd_original`` contains ``now``, ``--yes``, or
        ``-y`` as an argument (e.g. ``/reset now``, ``/new --yes My title``),
        the modal is bypassed and ``"once"`` is returned immediately. This is
        an escape hatch for non-interactive use and for the degraded path where
        the modal can't be marshaled onto the app loop (native Windows itself now
        drives the modal normally — see #33961). Callers are responsible
        for stripping the skip tokens from any remaining argument parsing
        (see :meth:`_split_destructive_skip`).

        Returns ``"once"``, ``"always"``, or ``None`` (cancelled).  Callers
        proceed with the destructive action when the result is non-None.
        """
        from cli import load_cli_config, save_config_value
        # Inline-skip escape hatch — works regardless of platform/modal state.
        # See class-level _DESTRUCTIVE_SKIP_TOKENS for the accepted tokens.
        if cmd_original:
            _, _skip = self._split_destructive_skip(cmd_original)
            if _skip:
                return "once"

        # Gate check — respects prior "Always Approve" clicks.
        try:
            cfg = load_cli_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            confirm_required = True
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            confirm_required = True

        if not confirm_required:
            return "once"

        # Render a prompt_toolkit-native confirmation panel.  This keeps option
        # labels visible above the composer and avoids raw input()/EOF races with
        # the running TUI.
        choices = [
            ("once", "Approve Once", "proceed this time only"),
            ("always", "Always Approve", "proceed and silence this prompt permanently"),
            ("cancel", "Cancel", "keep current conversation"),
        ]
        raw = self._prompt_text_input_modal(
            title=f"⚠️  /{command} — destroys conversation state",
            detail=detail,
            choices=choices,
        )
        if raw is None:
            print(f"🟡 /{command} cancelled (no input).")
            return None
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice is None:
            print(f"🟡 Unrecognized choice '{raw}'. /{command} cancelled.")
            return None

        if choice == "cancel":
            print(f"🟡 /{command} cancelled. Conversation unchanged.")
            return None

        if choice == "always":
            if save_config_value("approvals.destructive_slash_confirm", False):
                print("🔒 Future /clear, /new, /reset, and /undo will run without confirmation.")
                print("   Re-enable via `approvals.destructive_slash_confirm: true` in config.yaml.")
            else:
                print("⚠️  Couldn't persist opt-out — proceeding once.")

        return choice

    def _ring_bell(self, prompt: bool = False, context: str = "", detail: str = "") -> None:
        """Write a terminal bell (\\a) if the matching display.bell_* flag is on.

        ``prompt=True`` is the blocking-modal variant (clarify / approval /
        sudo / secret capture) gated by ``display.bell_on_prompt``; the default
        is the end-of-turn bell gated by ``display.bell_on_complete``. Works
        over SSH — the BEL propagates to the user's terminal.

        The same flag also emits an OSC 9 desktop notification (Ghostty,
        iTerm2, Kitty, WezTerm) and, inside a supporting Warp build, a
        ``warp://cli-agent`` OSC 777 event — see ``hermes_cli.terminal_notify``.
        ``context`` is the short notification body (e.g. "approval").
        """
        flag = "bell_on_prompt" if prompt else "bell_on_complete"
        if not getattr(self, flag, False):
            return
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass
        try:
            from hermes_cli.terminal_notify import notify as _terminal_notify

            _terminal_notify(
                context or ("input needed" if prompt else "turn complete"),
                prompt=prompt,
                session_id=getattr(self, "session_id", "") or "",
                detail=detail,
            )
        except Exception:
            pass

    def _clarify_callback(self, question, choices, multi_select=False, questions=None):
        """
        Platform callback for the clarify tool. Called from the agent thread.

        Sets up the interactive selection UI (or freetext prompt for open-ended
        questions), then blocks until the user responds via the prompt_toolkit
        key bindings.  If no response arrives within the configured timeout the
        question is dismissed and the agent is told to decide on its own.

        When ``multi_select`` is True, shows checkboxes and the user can
        select multiple options with Space, confirming with Enter.

        When ``questions`` is a non-empty list (batch clarify, issue #18450),
        the panel switches to the A-compact multi-question layout and the
        return value is a dict ``{"answers": {qid: raw_answer}}`` (plus
        ``"timed_out": True`` when the deadline expired with only partial
        answers). The single-question path below is unchanged.
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        import time as _time

        from tools.clarify_gateway import resolve_clarify_timeout

        if questions:
            return self._clarify_callback_batch(questions)

        # Canonical clarify timeout, shared with the gateway/TUI path. `<= 0`
        # means unlimited (never auto-skip mid-think) → a null deadline.
        timeout = resolve_clarify_timeout(CLI_CONFIG)
        response_queue = queue.Queue()
        is_open_ended = not choices
        # multi-select support: only active when multi_select is True and choices exist
        effective_multi = multi_select and not is_open_ended

        self._clarify_state = {
            "question": question,
            "choices": choices if not is_open_ended else [],
            "selected": 0,
            # multi-select support
            "multi_select": effective_multi,
            "selected_indices": set() if effective_multi else None,
            "response_queue": response_queue,
        }
        self._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
        # Open-ended questions skip straight to freetext input
        self._clarify_freetext = is_open_ended
        self._clarify_multi_base = None

        self._ring_bell(prompt=True, context="clarify")
        # Trigger an immediate prompt_toolkit repaint from this (non-main)
        # thread. Modal prompts must paint at once and must not be gated by the
        # _invalidate throttle / resize guard — see _paint_now / _invalidate (#41098).
        self._paint_now()

        # Poll for the user's response. The countdown in the hint line updates
        # on each repaint; refresh it once a second so the timer stays visible
        # while we wait. Selection changes (↑/↓) trigger instant repaints via
        # the key bindings.
        _last_countdown_refresh = _time.monotonic()
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = None
                self._persist_prompt_summary("?", "Clarify", question, str(result))
                return result
            except queue.Empty:
                # None deadline = unlimited: never auto-skip, just keep polling.
                if self._clarify_deadline is not None:
                    remaining = self._clarify_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                now = _time.monotonic()
                if now - _last_countdown_refresh >= 1.0:
                    _last_countdown_refresh = now
                    self._paint_now()

        # Timed out — tear down the UI and let the agent decide
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = None
        self._clarify_multi_base = None
        self._paint_now()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
        return (
            "The user did not provide a response within the time limit. "
            "Use your best judgement to make the choice and proceed."
        )

    def _clarify_batch_set_active(self, state, index) -> None:
        """Point the batch clarify panel at question ``index``.

        Mirrors the active question's data into the flat keys the existing
        single-question keybindings and renderer read (``question``,
        ``choices``, ``selected``, ``multi_select``, ``selected_indices``),
        so ↑/↓/Space/number keys operate on the active question unchanged.
        Open-ended questions drop straight into freetext, matching the
        single-question path. Re-visiting an answered question restores the
        cursor to the earlier selection (choice answers highlight their row,
        an "Other" answer highlights the Other row) so the user can see and
        edit what they picked.
        """
        questions_list = state["questions"]
        index = max(0, min(index, len(questions_list) - 1))
        entry = questions_list[index]
        state["active"] = index
        state["question"] = entry["question"]
        state["choices"] = entry["choices"] or []
        state["selected"] = 0
        state["multi_select"] = bool(entry["multi_select"])
        state["selected_indices"] = set() if entry["multi_select"] else None
        self._clarify_freetext = not entry["choices"]
        self._clarify_multi_base = None
        # Restore the earlier answer's cursor/checkbox position on re-visit.
        meta = (state.get("answer_meta") or {}).get(entry["qid"])
        choices = entry["choices"] or []
        if meta is None:
            return
        if meta.get("kind") == "choice":
            answer = state["answers"].get(entry["qid"])
            if answer in choices:
                state["selected"] = choices.index(answer)
        elif meta.get("kind") == "other":
            state["selected"] = len(choices)
        elif meta.get("kind") == "multi":
            checked = set()
            for label in meta.get("choices") or []:
                if label in choices:
                    checked.add(choices.index(label))
            if meta.get("other_text"):
                checked.add(len(choices))
            state["selected_indices"] = checked

    def _clarify_batch_lock(self, state, answer, meta=None) -> None:
        """Lock ``answer`` for the active batch question and advance.

        Overwrites any earlier answer for the same question (locked answers
        stay editable until the batch completes). ``meta`` records how the
        answer was produced ({"kind": "choice"|"other"|"multi", ...}) so a
        re-visit can restore the cursor and prefill an "Other" edit. Advances
        ``active`` to the next unanswered question; when every question has
        an answer, puts the answers dict on the response queue and tears down
        the panel.
        """
        entry = state["questions"][state["active"]]
        state["answers"][entry["qid"]] = answer
        state.setdefault("answer_meta", {})[entry["qid"]] = meta or {"kind": "choice"}
        self._persist_prompt_summary("?", "Clarify", entry["question"], str(answer))
        total = len(state["questions"])
        for offset in range(1, total + 1):
            candidate = (state["active"] + offset) % total
            if state["questions"][candidate]["qid"] not in state["answers"]:
                self._clarify_batch_set_active(state, candidate)
                return
        # Every question answered — resolve the batch.
        try:
            state["response_queue"].put(dict(state["answers"]))
        except Exception:
            pass
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_multi_base = None

    def _clarify_batch_enter(self, state) -> None:
        """Enter in batch choice mode: lock the active question's selection.

        Multi-select questions lock a JSON array string of the checked
        labels (the tool core parses it via ``_parse_multi_select_response``).
        Selecting "Other" switches to freetext; the freetext submit path
        locks the typed answer. Entering "Other" on a question whose earlier
        answer was typed prefills the composer with that text for editing.
        """
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        entry = state["questions"][state["active"]]
        meta = (state.get("answer_meta") or {}).get(entry["qid"]) or {}
        if state.get("multi_select"):
            indices = state.get("selected_indices") or set()
            sorted_idx = sorted(indices)
            selected_choices = [choices[i] for i in sorted_idx if i < len(choices)]
            other_checked = len(choices) in sorted_idx
            if other_checked:
                # Stash the checked real choices (possibly none) so the
                # freetext submit appends the typed answer to the array.
                self._clarify_multi_base = selected_choices
                self._clarify_freetext = True
                self._clarify_prefill = meta.get("other_text") or ""
                return
            self._clarify_batch_lock(
                state,
                json.dumps(selected_choices, ensure_ascii=False),
                meta={"kind": "multi", "choices": selected_choices, "other_text": ""},
            )
            return
        if selected < len(choices):
            self._clarify_batch_lock(
                state, choices[selected], meta={"kind": "choice"}
            )
            return
        # "Other" highlighted → switch to freetext; prefill an earlier typed
        # answer so Enter on an answered Other edits instead of retyping.
        self._clarify_freetext = True
        self._clarify_prefill = (
            meta.get("other_text") or "" if meta.get("kind") == "other" else ""
        )

    def _clarify_callback_batch(self, questions):
        """Batch clarify panel (A-compact): all questions, one active.

        Blocks on the response queue like the single-question path. Returns
        ``{"answers": {qid: raw_answer}}`` when every question is locked, the
        same dict plus ``"timed_out": True`` when the deadline expires with
        partial (or zero) answers, and passes a cancel string through
        unchanged so the tool core resolves the batch empty.
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        import time as _time

        from tools.clarify_gateway import resolve_clarify_timeout

        timeout = resolve_clarify_timeout(CLI_CONFIG)
        response_queue = queue.Queue()

        state = {
            "questions": list(questions),
            "answers": {},
            "answer_meta": {},
            "active": 0,
            "response_queue": response_queue,
            # Flat keys mirroring the active question — filled by
            # _clarify_batch_set_active below.
            "question": "",
            "choices": [],
            "selected": 0,
            "multi_select": False,
            "selected_indices": None,
        }
        self._clarify_state = state
        self._clarify_batch_set_active(state, 0)
        self._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
        self._ring_bell(prompt=True, context="clarify")
        self._paint_now()

        _last_countdown_refresh = _time.monotonic()
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = None
                if isinstance(result, dict):
                    return {"answers": result}
                # Cancel path (Ctrl+C teardown) posts a plain string — pass
                # it through so the tool core resolves the batch empty.
                return result
            except queue.Empty:
                if self._clarify_deadline is not None:
                    remaining = self._clarify_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                now = _time.monotonic()
                if now - _last_countdown_refresh >= 1.0:
                    _last_countdown_refresh = now
                    self._paint_now()

        # Timed out — keep the answers locked so far and flag the timeout.
        partial = dict(state["answers"])
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = None
        self._clarify_multi_base = None
        self._paint_now()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — locked answers returned){_RST}")
        return {"answers": partial, "timed_out": True}

    def _sudo_password_callback(self) -> str:
        """
        Prompt for sudo password through the prompt_toolkit UI.
        
        Called from the agent thread when a sudo command is encountered.
        Uses the same clarify-style mechanism: sets UI state, waits on a
        queue for the user's response via the Enter key binding.
        """
        from cli import _DIM, _RST, _cprint
        import time as _time

        timeout = 45
        response_queue = queue.Queue()

        self._capture_modal_input_snapshot()
        self._sudo_state = {
            "response_queue": response_queue,
        }
        self._sudo_deadline = _time.monotonic() + timeout
        self._ring_bell(prompt=True, context="sudo password")

        # Modal prompt — paint immediately, bypassing the throttle/resize guard
        # so the prompt can't be dropped and time out unseen (#41098).
        self._paint_now()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._sudo_state = None
                self._sudo_deadline = 0
                self._restore_modal_input_snapshot()
                self._paint_now()
                if result:
                    _cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
                else:
                    _cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
                return result
            except queue.Empty:
                remaining = self._sudo_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                self._paint_now()

        self._sudo_state = None
        self._sudo_deadline = 0
        self._restore_modal_input_snapshot()
        self._paint_now()
        _cprint(f"\n{_DIM}  ⏱ Timeout — continuing without sudo{_RST}")
        return ""

    def _approval_callback(self, command: str, description: str,
                           *, allow_permanent: bool = True,
                           allow_session: bool = True,
                           smart_denied: bool = False) -> str:
        """
        Prompt for dangerous command approval through the prompt_toolkit UI.

        Called from the agent thread. Shows a selection UI similar to clarify
        with choices: once / session / always / deny. Smart DENY owner
        overrides show only once / deny, as do gates that re-ask every time
        (allow_session=False). When allow_permanent is False for another
        reason (for example tirith), only 'always' is hidden.
        Long commands also get a 'view' option so the full command can be
        expanded before deciding.

        Uses _approval_lock to serialize concurrent requests (e.g. from
        parallel delegation subtasks) so each prompt gets its own turn
        and the shared _approval_state / _approval_deadline aren't clobbered.
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        import time as _time

        with self._approval_lock:
            timeout = int(CLI_CONFIG.get("approvals", {}).get("timeout", 300))
            response_queue = queue.Queue()

            self._approval_state = {
                "command": command,
                "description": description,
                "choices": self._approval_choices(
                    command,
                    allow_permanent=allow_permanent,
                    allow_session=allow_session,
                    smart_denied=smart_denied,
                ),
                "selected": 0,
                "response_queue": response_queue,
            }
            self._approval_deadline = _time.monotonic() + timeout

            self._ring_bell(prompt=True, context="approval", detail=command)
            # Modal prompt — paint immediately, bypassing the throttle/resize
            # guard. A throttled paint here can be silently dropped (250ms
            # window collision or in-flight resize), leaving the panel unseen so
            # the command is denied on timeout without the user ever seeing it
            # (#41098). The countdown refreshes below paint the same way.
            self._paint_now()

            _last_countdown_refresh = _time.monotonic()
            while True:
                try:
                    result = response_queue.get(timeout=1)
                    self._approval_state = None
                    self._approval_deadline = 0
                    self._paint_now()
                    _outcome_labels = {
                        "once": "allowed once",
                        "session": "allowed for session",
                        "always": "added to allowlist",
                        "deny": "denied",
                    }
                    self._persist_prompt_summary(
                        "⚠", "Approval", command,
                        _outcome_labels.get(result, str(result)),
                    )
                    return result
                except queue.Empty:
                    remaining = self._approval_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                    now = _time.monotonic()
                    if now - _last_countdown_refresh >= 1.0:
                        _last_countdown_refresh = now
                        self._paint_now()

            self._approval_state = None
            self._approval_deadline = 0
            self._paint_now()
            _cprint(f"\n{_DIM}  ⏱ Timeout — denying command{_RST}")
            self._persist_prompt_summary(
                "⚠", "Approval", command, "timed out (no response)",
            )
            return "timeout"

    def _approval_choices(self, command: str, *, allow_permanent: bool = True,
                          allow_session: bool = True,
                          smart_denied: bool = False) -> list[str]:
        """Return approval choices for a dangerous command prompt."""
        if smart_denied or not allow_session:
            choices = ["once", "deny"]
        else:
            choices = ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]
        if len(command) > 70:
            choices.append("view")
        return choices

    def _computer_use_approval_callback(self, action: str, args: dict, summary: str) -> str:
        """Adapt the generic approval UI for the computer_use tool.

        The computer_use handler expects verdicts of the form
        `approve_once` | `approve_session` | `always_approve` | `deny`.
        The CLI's built-in approval UI returns `once` | `session` | `always`
        | `deny`. Translate between the two.
        """
        # Build a command-ish string so the existing UI renders something
        # meaningful. `summary` is already a one-line human description.
        verdict = self._approval_callback(
            command=f"computer_use: {summary}",
            description=f"Allow computer_use to perform `{action}`?",
        )
        return {
            "once": "approve_once",
            "session": "approve_session",
            "always": "always_approve",
            "deny": "deny",
            "timeout": "timeout",
        }.get(verdict, "deny")

    def _handle_approval_selection(self) -> None:
        """Process the currently selected dangerous-command approval choice."""
        state = self._approval_state
        if not state:
            return

        selected = state.get("selected", 0)
        choices = state.get("choices")
        if not isinstance(choices, list):
            choices = []
        if not (0 <= selected < len(choices)):
            return

        chosen = choices[selected]
        if chosen == "view":
            state["show_full"] = True
            state["choices"] = [choice for choice in choices if choice != "view"]
            if state["selected"] >= len(state["choices"]):
                state["selected"] = max(0, len(state["choices"]) - 1)
            self._invalidate()
            return

        state["response_queue"].put(chosen)
        self._approval_state = None
        self._invalidate()

    def _secret_capture_callback(self, var_name: str, prompt: str, metadata=None) -> dict:
        return prompt_for_secret(self, var_name, prompt, metadata)

    def _capture_modal_input_snapshot(self) -> None:
        """Temporarily clear the input buffer and save the user's in-progress draft."""
        if self._modal_input_snapshot is not None or not getattr(self, "_app", None):
            return
        try:
            buf = self._app.current_buffer
            self._modal_input_snapshot = {
                "text": buf.text,
                "cursor_position": buf.cursor_position,
            }
            buf.reset()
        except Exception:
            self._modal_input_snapshot = None

    def _restore_modal_input_snapshot(self) -> None:
        """Restore any draft text that was present before a modal prompt opened."""
        snapshot = self._modal_input_snapshot
        self._modal_input_snapshot = None
        if not snapshot or not getattr(self, "_app", None):
            return
        try:
            buf = self._app.current_buffer
            buf.text = snapshot.get("text", "")
            buf.cursor_position = min(snapshot.get("cursor_position", 0), len(buf.text))
        except Exception:
            pass

    def _clear_active_overlays_for_interrupt(self) -> None:
        """Drain and clear every input-blocking overlay left by an interrupted agent.

        approval/clarify/sudo/secret prompts each block a worker thread on a
        ``response_queue.get()``.  When the agent is interrupted the worker
        thread is torn down, but the overlay's state dict stays set — leaving
        the CLI input gated (``read_only`` condition + keypress filter) with no
        thread servicing the prompt.  The result is a frozen terminal until the
        prompt's own timeout expires.  Push a terminal value onto each queue so
        any still-blocked thread unblocks cleanly, then nil the state out and
        restore the user's pre-modal draft (#14026).

        Safe default per prompt: approval -> "deny", clarify/sudo/secret ->
        cancel (None / empty).  Each step is wrapped so a dead queue can't
        prevent clearing the others.
        """
        if self._approval_state:
            try:
                self._approval_state["response_queue"].put("deny")
            except Exception:
                pass
            self._approval_state = None
        if self._clarify_state:
            try:
                self._clarify_state["response_queue"].put(
                    "The user cancelled. Use your best judgement to proceed."
                )
            except Exception:
                pass
            self._clarify_state = None
            self._clarify_freetext = False
            self._clarify_multi_base = None
        if self._sudo_state:
            try:
                self._sudo_state["response_queue"].put("")
            except Exception:
                pass
            self._sudo_state = None
            self._sudo_deadline = 0
            self._restore_modal_input_snapshot()
        if self._secret_state:
            try:
                self._cancel_secret_capture()
            except Exception:
                self._secret_state = None

    def _submit_secret_response(self, value: str) -> None:
        if not self._secret_state:
            return
        self._secret_state["response_queue"].put(value)
        self._secret_state = None
        self._secret_deadline = 0
        # Modal teardown — paint directly so the secret panel clears at once and
        # isn't held by the _invalidate throttle/resize guard (#41098).
        self._paint_now()

    def _cancel_secret_capture(self) -> None:
        self._submit_secret_response("")

    def _clear_secret_input_buffer(self) -> None:
        if getattr(self, "_app", None):
            try:
                self._app.current_buffer.reset()
            except Exception:
                pass
