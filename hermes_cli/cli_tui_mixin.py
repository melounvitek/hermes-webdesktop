"""prompt_toolkit TUI construction, key-binding handlers, and overlay display fragments for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import errno
import json
import os
import queue
import shutil
import sys
import threading
import time

from agent.interrupt_compat import request_hard_interrupt
from hermes_cli.commands import SlashCommandAutoSuggest, SlashCommandCompleter
from pathlib import Path
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
    WindowAlign,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import (
    ConditionalProcessor,
    PasswordProcessor,
    Processor,
    Transformation,
)
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea
from typing import Optional


class CLITuiMixin:
    """prompt_toolkit TUI construction, key-binding handlers, and overlay display fragments for the interactive CLI"""

    def _tui_input_rule_height(self, position: str, width: Optional[int] = None) -> int:
        """Return the visible height for the top/bottom input separator rules."""
        if position not in {"top", "bottom"}:
            raise ValueError(f"Unknown input rule position: {position}")
        if getattr(self, "_status_bar_suppressed_after_resize", False):
            return 0
        if position == "top":
            return 1
        return 0 if self._use_minimal_tui_chrome(width=width) else 1

    def _get_slash_confirm_display_fragments(self):
        """Render the /new-/clear-style confirmation panel."""
        from cli import (
            _append_blank_panel_line,
            _append_panel_line,
            _panel_box_width,
            _wrap_panel_text_keep_ws,
        )
        state = self._slash_confirm_state
        if not state:
            return []

        title = state.get("title") or "Confirm action"
        detail = state.get("detail") or ""
        choices = state.get("choices") or []
        selected = state.get("selected", 0)

        _wrap_panel_text = _wrap_panel_text_keep_ws

        preview_lines = []
        for line in detail.splitlines():
            preview_lines.extend(_wrap_panel_text(line, 72))
        for idx, (_value, label, desc) in enumerate(choices):
            marker = "❯" if idx == selected else " "
            preview_lines.extend(_wrap_panel_text(f"{marker} [{idx + 1}] {label} — {desc}", 72, subsequent_indent="    "))
        preview_lines.append("Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels.")

        box_width = _panel_box_width(title, preview_lines, min_width=56, max_width=86)
        inner_text_width = max(8, box_width - 2)
        detail_wrapped = []
        for line in detail.splitlines():
            detail_wrapped.extend(_wrap_panel_text(line, inner_text_width))
        choice_wrapped: list[tuple[int, str]] = []
        for idx, (_value, label, desc) in enumerate(choices):
            marker = "❯" if idx == selected else " "
            for wrapped in _wrap_panel_text(f"{marker} [{idx + 1}] {label} — {desc}", inner_text_width, subsequent_indent="    "):
                choice_wrapped.append((idx, wrapped))

        term_rows = shutil.get_terminal_size((100, 24)).lines
        reserved_below = 6
        chrome_full = 6
        available = max(0, term_rows - reserved_below)
        max_detail_rows = max(1, available - chrome_full - len(choice_wrapped))
        max_detail_rows = min(max_detail_rows, 8)
        if len(detail_wrapped) > max_detail_rows:
            keep = max(1, max_detail_rows - 1)
            detail_wrapped = detail_wrapped[:keep] + ["… (detail truncated)"]

        lines = []
        lines.append(('class:approval-border', '╭' + ('─' * box_width) + '╮\n'))
        _append_panel_line(lines, 'class:approval-border', 'class:approval-title', title, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for wrapped in detail_wrapped:
            _append_panel_line(lines, 'class:approval-border', 'class:approval-desc', wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for idx, wrapped in choice_wrapped:
            style = 'class:approval-selected' if idx == selected else 'class:approval-choice'
            _append_panel_line(lines, 'class:approval-border', style, wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        _append_panel_line(lines, 'class:approval-border', 'class:approval-cmd', 'Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels.', box_width)
        lines.append(('class:approval-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_approval_display_fragments(self):
        """Render the dangerous-command approval panel for the prompt_toolkit UI.

        Layout priority: title + command + choices must always render, even if
        the terminal is short or the description is long. Description is placed
        at the bottom of the panel and gets truncated to fit the remaining row
        budget. This prevents HSplit from clipping approve/deny off-screen when
        tirith findings produce multi-paragraph descriptions or when the user
        runs in a compact terminal pane.
        """
        from cli import (
            _append_blank_panel_line,
            _append_panel_line,
            _panel_box_width,
            _wrap_panel_text_keep_ws,
        )
        state = self._approval_state
        if not state:
            return []

        _wrap_panel_text = _wrap_panel_text_keep_ws

        command = state["command"]
        description = state["description"]
        choices = state["choices"]
        selected = state.get("selected", 0)
        show_full = state.get("show_full", False)

        title = "⚠️  Dangerous Command"
        cmd_display = command
        choice_labels = {
            "once": "Allow once",
            "session": "Allow for this session",
            "always": "Add to permanent allowlist",
            "deny": "Deny",
            "view": "Show full command",
        }

        preview_lines = _wrap_panel_text(description, 60)
        preview_lines.extend(_wrap_panel_text(cmd_display, 60))
        for i, choice in enumerate(choices):
            prefix = '❯ ' if i == selected else '  '
            preview_lines.extend(_wrap_panel_text(
                f"{prefix}{choice_labels.get(choice, choice)}",
                60,
                subsequent_indent="  ",
            ))

        box_width = _panel_box_width(title, preview_lines)
        inner_text_width = max(8, box_width - 2)

        # Pre-wrap the mandatory content — command + choices must always render.
        cmd_wrapped = _wrap_panel_text(cmd_display, inner_text_width)
        if not show_full and "view" in choices and len(cmd_wrapped) > 4:
            cmd_wrapped = cmd_wrapped[:3] + _wrap_panel_text(
                "… (choose Show full command)",
                inner_text_width,
            )

        # (choice_index, wrapped_line) so we can re-apply selected styling below
        choice_wrapped: list[tuple[int, str]] = []
        for i, choice in enumerate(choices):
            label = choice_labels.get(choice, choice)
            # Show number prefix for quick selection (1-9 for items 1-9, 0 for 10th item)
            if i < 9:
                num_prefix = str(i + 1)
            elif i == 9:
                num_prefix = '0'
            else:
                num_prefix = ' '  # No number for items beyond 10th
            prefix = f'❯ {num_prefix}. ' if i == selected else f'  {num_prefix}. '
            for wrapped in _wrap_panel_text(f"{prefix}{label}", inner_text_width, subsequent_indent="    "):
                choice_wrapped.append((i, wrapped))

        # Budget vertical space so HSplit never clips the command or choices.
        # Panel chrome (full layout with separators):
        #   top border + title + blank_after_title
        #   + blank_between_cmd_choices + bottom border = 5 rows.
        # In tight terminals we collapse to:
        #   top border + title + bottom border = 3 rows (no blanks).
        #
        # reserved_below: rows consumed below the approval panel by the
        # spinner/tool-progress line, status bar, input area, separators, and
        # prompt symbol. Measured at ~6 rows during live PTY approval prompts;
        # budget 6 so we don't overestimate the panel's room.
        term_rows = shutil.get_terminal_size((100, 24)).lines
        chrome_full = 5
        chrome_tight = 3
        reserved_below = 6

        available = max(0, term_rows - reserved_below)
        mandatory_full = chrome_full + len(cmd_wrapped) + len(choice_wrapped)

        # If the full-chrome panel doesn't fit, drop the separator blanks.
        # This keeps the command and every choice on-screen in compact terminals.
        use_compact_chrome = mandatory_full > available
        chrome_rows = chrome_tight if use_compact_chrome else chrome_full

        # If the command itself is too long to leave room for choices (e.g. user
        # hit "view" on a multi-hundred-character command), truncate it so the
        # approve/deny buttons still render. Keep at least 1 row of command.
        max_cmd_rows = max(1, available - chrome_rows - len(choice_wrapped))
        if len(cmd_wrapped) > max_cmd_rows:
            keep = max(1, max_cmd_rows - 1) if max_cmd_rows > 1 else 1
            cmd_wrapped = cmd_wrapped[:keep] + _wrap_panel_text(
                "… (command truncated — use /logs or /debug for full text)",
                inner_text_width,
            )

        # Allocate any remaining rows to description. The extra -1 in full mode
        # accounts for the blank separator between choices and description.
        mandatory_no_desc = chrome_rows + len(cmd_wrapped) + len(choice_wrapped)
        desc_sep_cost = 0 if use_compact_chrome else 1
        available_for_desc = available - mandatory_no_desc - desc_sep_cost
        # Even on huge terminals, cap description height so the panel stays compact.
        available_for_desc = max(0, min(available_for_desc, 10))

        desc_wrapped = _wrap_panel_text(description, inner_text_width) if description else []
        if available_for_desc < 1 or not desc_wrapped:
            desc_wrapped = []
        elif len(desc_wrapped) > available_for_desc:
            keep = max(1, available_for_desc - 1)
            desc_wrapped = desc_wrapped[:keep] + ["… (description truncated)"]

        # Render: title → command → choices → description (description last so
        # any remaining overflow clips from the bottom of the least-critical
        # content, never from the command or choices). Use compact chrome (no
        # blank separators) when the terminal is tight.
        lines = []
        lines.append(('class:approval-border', '╭' + ('─' * box_width) + '╮\n'))
        _append_panel_line(lines, 'class:approval-border', 'class:approval-title', title, box_width)
        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:approval-border', box_width)

        for wrapped in cmd_wrapped:
            _append_panel_line(lines, 'class:approval-border', 'class:approval-cmd', wrapped, box_width)
        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:approval-border', box_width)

        for i, wrapped in choice_wrapped:
            style = 'class:approval-selected' if i == selected else 'class:approval-choice'
            _append_panel_line(lines, 'class:approval-border', style, wrapped, box_width)

        if desc_wrapped:
            if not use_compact_chrome:
                _append_blank_panel_line(lines, 'class:approval-border', box_width)
            for wrapped in desc_wrapped:
                _append_panel_line(lines, 'class:approval-border', 'class:approval-desc', wrapped, box_width)

        lines.append(('class:approval-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_tui_prompt_symbols(self) -> tuple[str, str]:
        """Return ``(normal_prompt, state_suffix)`` for the active skin.

        ``normal_prompt`` is the full ``branding.prompt_symbol``.
        ``state_suffix`` is what special states (sudo/secret/approval/agent)
        should render after their leading icon.

        When a profile is active (not "default"), the profile name is
        prepended to the prompt symbol: ``coder ❯`` instead of ``❯``.
        """
        try:
            from hermes_cli.skin_engine import get_active_prompt_symbol
            symbol = get_active_prompt_symbol("❯ ")
        except Exception:
            symbol = "❯ "

        symbol = (symbol or "❯ ").rstrip() + " "

        # Prepend profile name when not default
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile not in {"default", "custom"}:
                symbol = f"{profile} {symbol}"
        except Exception:
            pass
        stripped = symbol.rstrip()
        if not stripped:
            return "❯ ", "❯ "

        parts = stripped.split()
        candidate = parts[-1] if parts else ""
        arrow_chars = ("❯", ">", "$", "#", "›", "»", "→")
        if any(ch in candidate for ch in arrow_chars):
            return symbol, candidate.rstrip() + " "

        # Icon-only custom prompts should still remain visible in special states.
        return symbol, symbol

    def _audio_level_bar(self) -> str:
        """Return a visual audio level indicator based on current RMS."""
        _LEVEL_BARS = " ▁▂▃▄▅▆▇"
        rec = getattr(self, "_voice_recorder", None)
        if rec is None:
            return ""
        rms = rec.current_rms
        # Normalize RMS (0-32767) to 0-7 index, with log-ish scaling
        # Typical speech RMS is 500-5000, we cap display at ~8000
        level = min(rms, 8000) * 7 // 8000
        return _LEVEL_BARS[level]

    def _get_tui_prompt_fragments(self):
        """Return the prompt_toolkit fragments for the current interactive state."""
        symbol, state_suffix = self._get_tui_prompt_symbols()
        compact = self._use_minimal_tui_chrome(width=self._get_tui_terminal_width())

        def _state_fragment(style: str, icon: str, extra: str = ""):
            if compact:
                text = icon
                if extra:
                    text = f"{text} {extra.strip()}".rstrip()
                return [(style, text + " ")]
            if extra:
                return [(style, f"{icon} {extra} {state_suffix}")]
            return [(style, f"{icon} {state_suffix}")]

        if self._voice_recording:
            bar = self._audio_level_bar()
            return _state_fragment("class:voice-recording", "●", bar)
        if self._voice_processing:
            return _state_fragment("class:voice-processing", "◉")
        if self._sudo_state:
            return _state_fragment("class:sudo-prompt", "🔐")
        if self._secret_state:
            return _state_fragment("class:sudo-prompt", "🔑")
        if self._approval_state:
            return _state_fragment("class:prompt-working", "⚠")
        if getattr(self, "_slash_confirm_state", None):
            return _state_fragment("class:prompt-working", "⚠")
        if self._clarify_freetext:
            return _state_fragment("class:clarify-selected", "✎")
        if self._clarify_state:
            return _state_fragment("class:prompt-working", "?")
        if self._command_running:
            return _state_fragment("class:prompt-working", self._command_spinner_frame())
        if self._agent_running:
            return _state_fragment("class:prompt-working", "⚕")
        if self._voice_mode:
            return _state_fragment("class:voice-prompt", "🎤")
        return [("class:prompt", symbol)]

    def _get_tui_prompt_text(self) -> str:
        """Return the visible prompt text for width calculations."""
        return "".join(text for _, text in self._get_tui_prompt_fragments())

    def _build_tui_style_dict(self) -> dict[str, str]:
        """Layer the active skin's prompt_toolkit colors over the base TUI style.

        Also rewrites any hex-color tokens in the resulting style strings
        to their light-mode equivalents (via _LIGHT_MODE_REMAP) when the
        terminal is detected as light.  This makes the chrome readable
        on cream Terminal.app backgrounds without per-skin overrides.
        """
        from cli import _detect_light_mode, _maybe_remap_for_light_mode
        style_dict = dict(getattr(self, "_tui_style_base", {}) or {})
        try:
            from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
            style_dict.update(get_prompt_toolkit_style_overrides())
        except Exception:
            pass
        # Light-mode remap on the style strings.  Each value is a pt
        # style string like "bg:#1a1a2e #C0C0C0 bold" — split on space,
        # rewrite any "#XXX" tokens (including "bg:#XXX") through the
        # light-mode remap, rejoin.
        #
        # CRITICAL: skip the remap entirely when a style string already
        # specifies its own bg (e.g. status-bar / completion-menu styles
        # with `bg:#1a1a2e ...`).  Those colors were tuned for that
        # specific dark bg and remapping the FG to a dark equivalent
        # would produce dark-on-dark (invisible).  The terminal's BG
        # mode is irrelevant — what matters is the bg the style itself
        # paints.
        try:
            if _detect_light_mode():
                def _remap_value(v: str) -> str:
                    if not v:
                        return v
                    tokens = v.split()
                    has_explicit_bg = any(t.startswith("bg:") for t in tokens)
                    if has_explicit_bg:
                        # The style paints its own bg — leave its fg alone.
                        return v
                    return " ".join(
                        _maybe_remap_for_light_mode(t) if t.startswith("#") else t
                        for t in tokens
                    )
                style_dict = {k: _remap_value(v or "") for k, v in style_dict.items()}
        except Exception:
            pass
        return style_dict

    def _apply_tui_skin_style(self) -> bool:
        """Refresh prompt_toolkit styling for a running interactive TUI."""
        if not getattr(self, "_app", None) or not getattr(self, "_tui_style_base", None):
            return False
        self._app.style = PTStyle.from_dict(self._build_tui_style_dict())
        self._invalidate(min_interval=0.0)
        return True

    def _get_extra_tui_widgets(self) -> list:
        """Return extra prompt_toolkit widgets to insert into the TUI layout.

        Wrapper CLIs can override this to inject widgets (e.g. a mini-player,
        overlay menu) into the layout without overriding ``run()``.  Widgets
        are inserted between the spacer and the status bar.
        """
        return []

    def _register_extra_tui_keybindings(self, kb, *, input_area) -> None:
        """Register extra keybindings on the TUI ``KeyBindings`` object.

        Wrapper CLIs can override this to add keybindings (e.g. transport
        controls, modal shortcuts) without overriding ``run()``.

        Parameters
        ----------
        kb : KeyBindings
            The active keybinding registry for the prompt_toolkit application.
        input_area : TextArea
            The main input widget, for wrappers that need to inspect or
            manipulate user input from a keybinding handler.
        """

    def _build_tui_layout_children(
        self,
        *,
        sudo_widget,
        secret_widget,
        approval_widget,
        slash_confirm_widget=None,
        clarify_widget,
        model_picker_widget=None,
        command_palette_widget=None,
        spinner_widget=None,
        spacer,
        status_bar,
        input_rule_top,
        image_bar,
        input_area,
        input_rule_bot,
        voice_status_bar,
        completions_menu,
    ) -> list:
        """Assemble the ordered list of children for the root ``HSplit``.

        Wrapper CLIs typically override ``_get_extra_tui_widgets`` instead of
        this method.  Override this only when you need full control over widget
        ordering.
        """
        return [
            item for item in [
                Window(height=0),
                sudo_widget,
                secret_widget,
                approval_widget,
                slash_confirm_widget,
                clarify_widget,
                model_picker_widget,
                command_palette_widget,
                spinner_widget,
                spacer,
                *self._get_extra_tui_widgets(),
                getattr(self, "_pet_widget", None),
                getattr(self, "_stash_panel_widget", None),
                status_bar,
                input_rule_top,
                image_bar,
                input_area,
                input_rule_bot,
                voice_status_bar,
                completions_menu,
            ] if item is not None
        ]

    def _tui_spinner_loop(self):
        while not self._should_exit:
            if not self._app:
                time.sleep(0.1)
                continue
            if self._command_running:
                self._invalidate(min_interval=0.1)
                time.sleep(0.1)
            else:
                # Do not repaint the idle prompt every second. In non-full-screen
                # prompt_toolkit mode, background redraws can fight tmux/Ghostty/cmux
                # viewport restoration after focus changes and visually move the
                # command input area. Keep idle stable; input/agent events still
                # invalidate explicitly when the UI actually changes.
                time.sleep(0.2)

    def _get_clarify_batch_display_fragments(self, state):
        """Build styled text for the batch (multi-question) clarify panel.

        A-compact layout mirroring the TUI: a "N questions" header, one
        status line per question (✓ answered → answer / ▸ active /
        · pending), and the active question's numbered choices (+ Other)
        expanded directly beneath its status line.
        """
        from cli import _append_panel_line, _panel_box_width, _wrap_panel_text
        questions_list = state.get("questions") or []
        answers = state.get("answers") or {}
        active = state.get("active", 0)
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        multi_select = state.get("multi_select", False)
        selected_indices = state.get("selected_indices", set()) if multi_select else set()

        title = "Hermes needs your input"
        header = f"{len(questions_list)} questions"

        def _status_rows(width):
            """(style, text) rows for the status list + expanded active question."""
            rows = []
            answer_meta = state.get("answer_meta") or {}
            for idx, entry in enumerate(questions_list):
                answered = entry["qid"] in answers
                if answered:
                    marker = "✓"
                elif idx == active:
                    marker = "▸"
                else:
                    marker = "·"
                label = f"{marker} {entry['question']}"
                row_style = 'class:clarify-selected' if idx == active else 'class:clarify-choice'
                for wrapped in _wrap_panel_text(label, width, subsequent_indent="  "):
                    rows.append((row_style, wrapped))
                if answered:
                    # The locked answer on its own line, in its own color,
                    # so the current answer stays readable while walking
                    # the list with Tab/Shift-Tab.
                    for wrapped in _wrap_panel_text(
                        f"    {answers[entry['qid']]}", width, subsequent_indent="    "
                    ):
                        rows.append(('class:clarify-answer', wrapped))
                if idx != active:
                    continue
                # Expanded active question: numbered choices + Other.
                for i, choice in enumerate(choices):
                    num_prefix = str(i + 1) if i < 9 else ('0' if i == 9 else ' ')
                    if multi_select:
                        cb = "[x]" if i in selected_indices else "[ ]"
                        cursor = "❯" if i == selected and not self._clarify_freetext else " "
                        prefix = f"  {cursor} {cb} {num_prefix}. "
                    else:
                        cursor = "❯" if i == selected and not self._clarify_freetext else " "
                        prefix = f"  {cursor} {num_prefix}. "
                    style = 'class:clarify-selected' if i == selected and not self._clarify_freetext else 'class:clarify-choice'
                    for wrapped in _wrap_panel_text(f"{prefix}{choice}", width, subsequent_indent="      "):
                        rows.append((style, wrapped))
                if choices:
                    other_idx = len(choices)
                    other_num = other_idx + 1
                    other_num_prefix = str(other_num) if other_num < 10 else ('0' if other_num == 10 else ' ')
                    if multi_select:
                        cb = "[x]" if other_idx in selected_indices else "[ ]"
                        mid = f"{cb} {other_num_prefix}"
                    else:
                        mid = other_num_prefix
                    # An earlier typed answer stays visible next to Other;
                    # Enter on it edits (the composer is prefilled).
                    meta = answer_meta.get(entry["qid"]) or {}
                    other_text = meta.get("other_text") or ""
                    other_suffix = f"Other: {other_text}" if other_text else None
                    if self._clarify_freetext:
                        other_label = f"  ❯ {mid}. " + (other_suffix or "Other (type below)")
                        other_style = 'class:clarify-active-other'
                    elif selected == other_idx:
                        other_label = f"  ❯ {mid}. " + (other_suffix or "Other (type your answer)")
                        other_style = 'class:clarify-selected'
                    else:
                        other_label = f"    {mid}. " + (other_suffix or "Other (type your answer)")
                        other_style = 'class:clarify-choice'
                    for wrapped in _wrap_panel_text(other_label, width, subsequent_indent="      "):
                        rows.append((other_style, wrapped))
                elif self._clarify_freetext:
                    for wrapped in _wrap_panel_text(
                        "  Type your answer in the prompt below, then press Enter.", width
                    ):
                        rows.append(('class:clarify-active-other', wrapped))
            return rows

        preview_rows = _status_rows(60)
        box_width = _panel_box_width(title, [header] + [text for _, text in preview_rows])
        inner_text_width = max(8, box_width - 2)
        rows = _status_rows(inner_text_width)

        lines = []
        lines.append(('class:clarify-border', '╭─ '))
        lines.append(('class:clarify-title', title))
        lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
        _append_panel_line(lines, 'class:clarify-border', 'class:clarify-question', header, box_width)
        for style, text in rows:
            _append_panel_line(lines, 'class:clarify-border', style, text, box_width)
        lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_clarify_display_fragments(self):
        """Build styled text for the clarify question/choices panel.

        Layout priority: choices + Other option must always render even if
        the question is very long. The question is budgeted to leave enough
        rows for the choices and trailing chrome; anything over the budget
        is truncated with a marker.
        """
        from cli import _append_blank_panel_line, _append_panel_line, _panel_box_width, _wrap_panel_text
        state = self._clarify_state
        if not state:
            return []
        if state.get("questions"):
            return self._get_clarify_batch_display_fragments(state)

        question = state["question"]
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        # multi-select support
        multi_select = state.get("multi_select", False)
        selected_indices = state.get("selected_indices", set()) if multi_select else set()
        preview_lines = _wrap_panel_text(question, 60)
        for i, choice in enumerate(choices):
            # Show number prefix for quick selection (1-9 for items 1-9, 0 for 10th item)
            if i < 9:
                num_prefix = str(i + 1)
            elif i == 9:
                num_prefix = '0'
            else:
                num_prefix = ' '
            if multi_select:
                cb = "[x]" if i in selected_indices else "[ ]"
                if i == selected and not self._clarify_freetext:
                    prefix = f"❯ {cb} {num_prefix}. "
                else:
                    prefix = f"  {cb} {num_prefix}. "
            elif i == selected and not self._clarify_freetext:
                prefix = f"❯ {num_prefix}. "
            else:
                prefix = f"  {num_prefix}. "
            preview_lines.extend(_wrap_panel_text(f"{prefix}{choice}", 60, subsequent_indent="    "))
        # "Other" option in preview
        other_num = len(choices) + 1
        if other_num < 10:
            other_num_prefix = str(other_num)
        elif other_num == 10:
            other_num_prefix = '0'
        else:
            other_num_prefix = ' '
        other_idx_val = len(choices)
        if multi_select:
            cb = "[x]" if other_idx_val in selected_indices else "[ ]"
            other_label = (
                f"❯ {cb} {other_num_prefix}. Other (type below)" if self._clarify_freetext
                else f"❯ {cb} {other_num_prefix}. Other (type your answer)" if selected == other_idx_val
                else f"  {cb} {other_num_prefix}. Other (type your answer)"
            )
        else:
            other_label = (
                f"❯ {other_num_prefix}. Other (type below)" if self._clarify_freetext
                else f"❯ {other_num_prefix}. Other (type your answer)" if selected == len(choices)
                else f"  {other_num_prefix}. Other (type your answer)"
            )
        preview_lines.extend(_wrap_panel_text(other_label, 60, subsequent_indent="    "))
        box_width = _panel_box_width("Hermes needs your input", preview_lines)
        inner_text_width = max(8, box_width - 2)

        # Pre-wrap choices + Other option — these are mandatory.
        choice_wrapped: list[tuple[int, str]] = []
        if choices:
            for i, choice in enumerate(choices):
                # Show number prefix for quick selection (1-9 for items 1-9, 0 for 10th item)
                if i < 9:
                    num_prefix = str(i + 1)
                elif i == 9:
                    num_prefix = '0'
                else:
                    num_prefix = ' '
                # multi-select support: add checkbox after cursor indicator
                if multi_select:
                    cb = "[x]" if i in selected_indices else "[ ]"
                    if i == selected and not self._clarify_freetext:
                        prefix = f'❯ {cb} {num_prefix}. '
                    else:
                        prefix = f'  {cb} {num_prefix}. '
                elif i == selected and not self._clarify_freetext:
                    prefix = f'❯ {num_prefix}. '
                else:
                    prefix = f'  {num_prefix}. '
                for wrapped in _wrap_panel_text(f"{prefix}{choice}", inner_text_width, subsequent_indent="    "):
                    choice_wrapped.append((i, wrapped))
            # Trailing Other row(s)
            other_idx = len(choices)
            other_num = other_idx + 1
            if other_num < 10:
                other_num_prefix = str(other_num)
            elif other_num == 10:
                other_num_prefix = '0'
            else:
                other_num_prefix = ' '
            # multi-select support: add checkbox to Other option
            if multi_select:
                cb = "[x]" if other_idx in selected_indices else "[ ]"
                if selected == other_idx and not self._clarify_freetext:
                    other_label_mand = f'❯ {cb} {other_num_prefix}. Other (type your answer)'
                elif self._clarify_freetext:
                    other_label_mand = f'❯ {cb} {other_num_prefix}. Other (type below)'
                else:
                    other_label_mand = f'  {cb} {other_num_prefix}. Other (type your answer)'
            else:
                if selected == other_idx and not self._clarify_freetext:
                    other_label_mand = f'❯ {other_num_prefix}. Other (type your answer)'
                elif self._clarify_freetext:
                    other_label_mand = f'❯ {other_num_prefix}. Other (type below)'
                else:
                    other_label_mand = f'  {other_num_prefix}. Other (type your answer)'
            other_wrapped = _wrap_panel_text(other_label_mand, inner_text_width, subsequent_indent="    ")
        elif self._clarify_freetext:
            # Freetext-only mode: the guidance line takes the place of choices.
            other_wrapped = _wrap_panel_text(
                "Type your answer in the prompt below, then press Enter.",
                inner_text_width,
            )
        else:
            other_wrapped = []

        # Budget the question so mandatory rows always render.
        # Chrome layouts:
        #   full : top border + blank_after_title + blank_after_question
        #          + blank_before_bottom + bottom border = 5 rows
        #   tight: top border + bottom border = 2 rows (drop all blanks)
        #
        # reserved_below matches the approval-panel budget (~6 rows for
        # spinner/tool-progress + status + input + separators + prompt).
        term_rows = shutil.get_terminal_size((100, 24)).lines
        chrome_full = 5
        chrome_tight = 2
        reserved_below = 6

        available = max(0, term_rows - reserved_below)
        # The compact decision must reserve room for at least one question
        # row on top of the choices, otherwise full chrome (3 blank
        # separators) gets kept when there is no room for it and the panel
        # overflows the viewport — HSplit then clips the panel's tail,
        # silently dropping the choices (the reported bug).
        mandatory_full = chrome_full + 1 + len(choice_wrapped) + len(other_wrapped)

        use_compact_chrome = mandatory_full > available
        chrome_rows = chrome_tight if use_compact_chrome else chrome_full

        max_question_rows = max(1, available - chrome_rows - len(choice_wrapped) - len(other_wrapped))
        max_question_rows = min(max_question_rows, 12)  # soft cap on huge terminals

        # When the choices alone (plus compact chrome) already exceed the
        # viewport, drop the question entirely — the choices are the only
        # thing the user must see to make a selection. Without this the
        # question would still claim its 1-row floor above and push the
        # tail of the choices off-screen (HSplit clips the overflow).
        choices_overflow = chrome_rows + len(choice_wrapped) + len(other_wrapped) >= available
        if choices_overflow:
            max_question_rows = 0

        question_wrapped = _wrap_panel_text(question, inner_text_width)
        if max_question_rows <= 0:
            question_wrapped = []
        elif len(question_wrapped) > max_question_rows:
            # The truncation marker is itself a row, so it must count
            # against the budget. With a 1-row budget there is no room for
            # both a question line and the marker — show the marker alone
            # so the rendered question never exceeds max_question_rows.
            keep = max(0, max_question_rows - 1)
            question_wrapped = question_wrapped[:keep] + ["… (question truncated)"]

        lines = []
        # Box top border
        lines.append(('class:clarify-border', '╭─ '))
        lines.append(('class:clarify-title', 'Hermes needs your input'))
        lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len("Hermes needs your input") - 3)) + '╮\n'))
        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)

        # Question text (bounded)
        for wrapped in question_wrapped:
            _append_panel_line(lines, 'class:clarify-border', 'class:clarify-question', wrapped, box_width)
        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)

        if self._clarify_freetext and not choices:
            for wrapped in other_wrapped:
                _append_panel_line(lines, 'class:clarify-border', 'class:clarify-choice', wrapped, box_width)
            if not use_compact_chrome:
                _append_blank_panel_line(lines, 'class:clarify-border', box_width)

        if choices:
            # Multiple-choice mode: show selectable options
            for i, wrapped in choice_wrapped:
                style = 'class:clarify-selected' if i == selected and not self._clarify_freetext else 'class:clarify-choice'
                _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)

            # "Other" option (trailing row(s), only shown when choices exist)
            other_idx = len(choices)
            # Calculate number prefix for "Other" option
            other_num = other_idx + 1
            if other_num < 10:
                other_num_prefix = str(other_num)
            elif other_num == 10:
                other_num_prefix = '0'
            else:
                other_num_prefix = ' '

            if selected == other_idx and not self._clarify_freetext:
                other_style = 'class:clarify-selected'
            elif self._clarify_freetext:
                other_style = 'class:clarify-active-other'
            else:
                other_style = 'class:clarify-choice'
            for wrapped in other_wrapped:
                _append_panel_line(lines, 'class:clarify-border', other_style, wrapped, box_width)

        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_model_picker_display_fragments(self):
        from cli import (
            HermesCLI,
            _append_blank_panel_line,
            _append_panel_line,
            _panel_box_width,
            _wrap_panel_text,
        )
        state = self._model_picker_state
        if not state:
            return []
        stage = state.get("stage", "provider")
        if stage == "provider":
            title = "⚙ Model Picker — Select Provider"
            choices = []
            _providers = state.get("providers")
            for p in _providers if isinstance(_providers, list) else []:
                count = p.get("total_models", len(p.get("models", [])))
                label = f"{p['name']} ({count} model{'s' if count != 1 else ''})"
                if p.get("is_current"):
                    label += "  ← current"
                choices.append(label)
            choices.append("Cancel")
            hint = f"Current: {state.get('current_model', 'unknown')} on {state.get('current_provider', 'unknown')}"
        else:
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            title = f"⚙ Model Picker — {provider_data.get('name', provider_data.get('slug', 'Provider'))}"
            # Fuzzy filter: narrow the concrete model list by the typed
            # query. Selection still resolves to a real entry (see the
            # filtered_pairs index mapping in the selection handler), so
            # this never introduces an ambiguous model resolution.
            _query = state.get("filter", "") or ""
            filtered_pairs = self._filter_model_picker_entries(model_list, _query)
            state["_filtered_pairs"] = filtered_pairs
            model_labels = [e for (_i, e) in filtered_pairs]
            choices = list(model_labels) + ["← Back", "Cancel"]
            if _query:
                hint = (
                    f"Filter: {_query}▏  ({len(model_labels)}/{len(model_list)} match "
                    "— type to narrow, Backspace to clear)"
                )
            elif model_list:
                hint = f"Select a model ({len(model_list)} available) — type to filter"
            else:
                hint = "No models listed for this provider. Use Back or Cancel."

        box_width = _panel_box_width(title, [hint] + choices, min_width=46, max_width=84)
        inner_text_width = max(8, box_width - 6)
        selected = state.get("selected", 0)

        # Scrolling viewport: the panel renders into a Window with no max
        # height, so without limiting visible items the bottom border and
        # any items past the available terminal rows get clipped on long
        # provider catalogs (e.g. Ollama Cloud's 36+ models).
        try:
            from prompt_toolkit.application import get_app
            term_rows = get_app().output.get_size().rows
        except Exception:
            term_rows = shutil.get_terminal_size((100, 24)).lines
        scroll_offset, visible = HermesCLI._compute_model_picker_viewport(
            selected, state.get("_scroll_offset", 0), len(choices), term_rows,
        )
        state["_scroll_offset"] = scroll_offset

        lines = []
        lines.append(('class:clarify-border', '╭─ '))
        lines.append(('class:clarify-title', title))
        lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
        _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        _append_panel_line(lines, 'class:clarify-border', 'class:clarify-hint', hint, box_width)
        _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        for idx in range(scroll_offset, scroll_offset + visible):
            choice = choices[idx]
            style = 'class:clarify-selected' if idx == selected else 'class:clarify-choice'
            prefix = '❯ ' if idx == selected else '  '
            for wrapped in _wrap_panel_text(prefix + choice, inner_text_width, subsequent_indent='  '):
                _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)
        _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_command_palette_display_fragments(self):
        from cli import (
            HermesCLI,
            _append_blank_panel_line,
            _append_panel_line,
            _panel_box_width,
            _wrap_panel_text,
        )
        state = self._command_palette_state
        if not state:
            return []
        rows = self._command_palette_visible_entries()
        state["_visible_count"] = len(rows)
        _query = state.get("filter", "") or ""
        total = len(state.get("entries") or [])
        title = "⚙ Command Palette"
        if _query:
            hint = f"Filter: {_query}▏  ({len(rows)}/{total} match — Enter inserts, Esc cancels)"
        else:
            hint = f"Type to filter {total} commands — ↑/↓ then Enter inserts, Esc cancels"

        labels = [f"{c}  —  {d}" if d else c for (c, _cat, d) in rows]
        if not labels:
            labels = ["(no matching commands)"]
        box_width = _panel_box_width(title, [hint] + labels, min_width=50, max_width=90)
        inner_text_width = max(8, box_width - 6)
        selected = state.get("selected", 0)
        try:
            from prompt_toolkit.application import get_app
            term_rows = get_app().output.get_size().rows
        except Exception:
            term_rows = shutil.get_terminal_size((100, 24)).lines
        scroll_offset, visible = HermesCLI._compute_model_picker_viewport(
            selected, state.get("_scroll_offset", 0), len(labels), term_rows,
        )
        state["_scroll_offset"] = scroll_offset

        lines = []
        lines.append(('class:clarify-border', '╭─ '))
        lines.append(('class:clarify-title', title))
        lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
        _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        _append_panel_line(lines, 'class:clarify-border', 'class:clarify-hint', hint, box_width)
        _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        for idx in range(scroll_offset, min(scroll_offset + visible, len(labels))):
            label = labels[idx]
            style = 'class:clarify-selected' if idx == selected else 'class:clarify-choice'
            prefix = '❯ ' if idx == selected else '  '
            for wrapped in _wrap_panel_text(prefix + label, inner_text_width, subsequent_indent='    '):
                _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)
        _append_blank_panel_line(lines, 'class:clarify-border', box_width)
        lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_sudo_display_fragments(self):
        from cli import _append_blank_panel_line, _append_panel_line, _panel_box_width
        state = self._sudo_state
        if not state:
            return []
        title = '🔐 Sudo Password Required'
        body = 'Enter password below (hidden), or press Enter to skip'
        box_width = _panel_box_width(title, [body])
        lines = []
        lines.append(('class:sudo-border', '╭─ '))
        lines.append(('class:sudo-title', title))
        lines.append(('class:sudo-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
        _append_blank_panel_line(lines, 'class:sudo-border', box_width)
        _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', body, box_width)
        _append_blank_panel_line(lines, 'class:sudo-border', box_width)
        lines.append(('class:sudo-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _get_secret_display_fragments(self):
        from cli import _append_blank_panel_line, _append_panel_line, _panel_box_width
        state = self._secret_state
        if not state:
            return []

        title = '🔑 Skill Setup Required'
        prompt = state.get("prompt") or f"Enter value for {state.get('var_name', 'secret')}"
        metadata = state.get("metadata") or {}
        help_text = metadata.get("help")
        body = 'Enter secret below (hidden), ESC or Ctrl+C to skip'
        content_lines = [prompt, body]
        if help_text:
            content_lines.insert(1, str(help_text))
        box_width = _panel_box_width(title, content_lines)
        lines = []
        lines.append(('class:sudo-border', '╭─ '))
        lines.append(('class:sudo-title', title))
        lines.append(('class:sudo-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
        _append_blank_panel_line(lines, 'class:sudo-border', box_width)
        _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', prompt, box_width)
        if help_text:
            _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', str(help_text), box_width)
        _append_blank_panel_line(lines, 'class:sudo-border', box_width)
        _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', body, box_width)
        _append_blank_panel_line(lines, 'class:sudo-border', box_width)
        lines.append(('class:sudo-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _tui_hint_text(self):
        if self._sudo_state:
            remaining = max(0, int(self._sudo_deadline - time.monotonic()))
            return [
                ('class:hint', '  password hidden · Enter to skip'),
                ('class:clarify-countdown', f'  ({remaining}s)'),
            ]

        if self._secret_state:
            remaining = max(0, int(self._secret_deadline - time.monotonic()))
            return [
                ('class:hint', '  secret hidden · Enter to skip'),
                ('class:clarify-countdown', f'  ({remaining}s)'),
            ]

        if self._approval_state:
            remaining = max(0, int(self._approval_deadline - time.monotonic()))
            return [
                ('class:hint', '  ↑/↓ to select, Enter to confirm'),
                ('class:clarify-countdown', f'  ({remaining}s)'),
            ]

        if self._slash_confirm_state:
            remaining = max(0, int(self._slash_confirm_deadline - time.monotonic()))
            return [
                ('class:hint', '  type 1/2/3, or ↑/↓ to select, Enter to confirm'),
                ('class:clarify-countdown', f'  ({remaining}s)'),
            ]

        if self._clarify_state:
            # None deadline = unlimited wait → hide the countdown entirely.
            if self._clarify_deadline is None:
                countdown = ''
            else:
                remaining = max(0, int(self._clarify_deadline - time.monotonic()))
                countdown = f'  ({remaining}s)'
            if self._clarify_freetext:
                return [
                    ('class:hint', '  type your answer and press Enter'),
                    ('class:clarify-countdown', countdown),
                ]
            if self._clarify_state.get("questions"):
                return [
                    ('class:hint', '  ↑/↓ to select, Enter to lock, Tab next question'),
                    ('class:clarify-countdown', countdown),
                ]
            return [
                ('class:hint', '  ↑/↓ to select, Enter to confirm'),
                ('class:clarify-countdown', countdown),
            ]

        if self._command_running:
            frame = self._command_spinner_frame()
            detail = "input temporarily disabled" if self._command_blocks_input else "input stays active; Enter queues"
            return [
                ('class:hint', f'  {frame} command in progress · {detail}'),
            ]

        return []

    def _tui_placeholder_text(self):
        if self._voice_recording:
            _label = self._voice_record_key_label()
            return f"recording... {_label} to stop, Ctrl+C to cancel"
        if self._voice_processing:
            return "transcribing..."
        if self._sudo_state:
            return "type password (hidden), Enter to submit · ESC to skip"
        if self._secret_state:
            return "type secret (hidden), Enter to submit · ESC to skip"
        if self._approval_state:
            return ""
        if self._slash_confirm_state:
            return "type 1/2/3, or use ↑/↓ then Enter"
        if self._clarify_freetext:
            return "type your answer here and press Enter"
        if self._clarify_state:
            return ""
        if self._command_running:
            frame = self._command_spinner_frame()
            status = self._command_status or "Processing command..."
            return f"{frame} {status}"
        if self._agent_running:
            return "msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel"
        if self._voice_mode:
            _label = self._voice_record_key_label()
            return f"type or {_label} to record"
        # Advertise a parked draft so the stash can never be silently
        # forgotten — the composer itself tells you how to get it back.
        _stash_hint = ""
        try:
            _stash_hint = self._prompt_stash.placeholder_hint()
        except Exception:
            _stash_hint = ""
        if _stash_hint:
            return _stash_hint
        # Idle + empty composer: show a rotating task-oriented example to
        # nudge the user toward a high-value first action (C-09). Chosen
        # once per session (self._composer_placeholder) so it stays stable
        # while being read, not flickering every render.
        return getattr(self, "_composer_placeholder", "") or ""

    def _get_stash_panel_display_fragments(self):
        try:
            _stash = self._prompt_stash
            return self._render_stash_panel(
                _stash.panel_rows(),
                _stash.panel_cursor,
                self._get_tui_terminal_width(),
            )
        except Exception:
            return []

    def _tui_handle_voice_record(self, event):
        """Toggle voice recording when voice mode is active.

        IMPORTANT: This handler runs in prompt_toolkit's event-loop thread.
        Any blocking call here (locks, sd.wait, disk I/O) freezes the
        entire UI.  All heavy work is dispatched to daemon threads.
        """
        from cli import _DIM, _RST, _cprint, logger
        if not self._voice_mode:
            return
        # Always allow STOPPING a recording (even when agent is running)
        if self._voice_recording:
            # Manual stop via push-to-talk key: stop continuous mode
            with self._voice_lock:
                self._voice_continuous = False
            # Flag clearing is handled atomically inside _voice_stop_and_transcribe
            event.app.invalidate()
            threading.Thread(
                target=self._voice_stop_and_transcribe,
                daemon=True,
            ).start()
        else:
            # Allow disarming continuous mode even when the agent is
            # running or transcribing — otherwise the user is stuck in
            # an auto-restart loop until /voice off (#67545).
            if self._agent_running or self._voice_processing:
                with self._voice_lock:
                    self._voice_continuous = False
                event.app.invalidate()
                return
            # Guard: don't START recording during interactive prompts
            if self._clarify_state or self._sudo_state or self._approval_state or self._slash_confirm_state:
                return

            # Interrupt TTS if playing, so user can start talking.
            # stop_playback() is fast (just terminates a subprocess);
            # the stop event drains the streaming pipeline if one is live.
            if not self._voice_tts_done.is_set():
                try:
                    logger.info("TTS CUT: record key handler cutting TTS")
                    from tools.tts_streaming import mark_speech_interrupted
                    mark_speech_interrupted()
                    if self._voice_tts_stop is not None:
                        self._voice_tts_stop.set()
                    from tools.voice_mode import stop_playback
                    stop_playback()
                    self._voice_tts_done.set()
                except Exception:
                    pass

            with self._voice_lock:
                self._voice_continuous = True

            # Dispatch to a daemon thread so play_beep(sd.wait),
            # AudioRecorder.start(lock acquire), and config I/O
            # never block the prompt_toolkit event loop.
            def _start_recording():
                try:
                    self._voice_start_recording()
                    if hasattr(self, '_app') and self._app:
                        self._app.invalidate()
                except Exception as e:
                    _cprint(f"\n{_DIM}Voice recording failed: {e}{_RST}")

            threading.Thread(target=_start_recording, daemon=True).start()
            event.app.invalidate()

    def _tui_handle_ctrl_c(self, event):
        """Handle Ctrl+C - cancel interactive prompts, interrupt agent, or exit.

        Priority:
        0. Cancel active voice recording
        1. Cancel active sudo/approval/clarify prompt
        2. Interrupt the running agent (first press)
        3. Force exit (second press within 2s, or when idle)
        """
        from cli import _DIM, _RST, _cprint
        now = time.time()

        # Cancel active voice recording.
        # Run cancel() in a background thread to prevent blocking the
        # event loop if AudioRecorder._lock or CoreAudio takes time.
        _should_cancel_voice = False
        _recorder_ref = None
        with self._voice_lock:
            if self._voice_recording and self._voice_recorder:
                _recorder_ref = self._voice_recorder
                self._voice_recording = False
                self._voice_continuous = False
                _should_cancel_voice = True
        if _should_cancel_voice:
            _cprint(f"\n{_DIM}Recording cancelled.{_RST}")
            threading.Thread(
                target=_recorder_ref.cancel, daemon=True
            ).start()
            event.app.invalidate()
            return

        # Cancel slash confirmation prompt (foreground UI, not an
        # agent-blocking overlay — cancel and stop here).
        if self._slash_confirm_state:
            self._submit_slash_confirm_response("cancel")
            event.app.current_buffer.reset()
            event.app.invalidate()
            return

        # Cancel /model picker (foreground UI — cancel and stop here).
        if self._model_picker_state:
            self._close_model_picker()
            event.app.current_buffer.reset()
            event.app.invalidate()
            return

        # Cancel command palette (foreground UI — cancel and stop here).
        if self._command_palette_state:
            self._close_command_palette()
            event.app.current_buffer.reset()
            event.app.invalidate()
            return

        # Clear all agent-blocking overlays (approval/clarify/sudo/secret)
        # in one shot.  We do NOT return after clearing — we fall through so
        # that if the agent is also running we fire the interrupt on the same
        # Ctrl+C press.  This fixes the case where a stale/orphaned overlay
        # (left behind by a previous interrupt) consumes the press without
        # ever reaching the agent-interrupt branch, leaving the chat frozen
        # (#14026).
        _overlay_cleared = bool(
            self._sudo_state
            or self._secret_state
            or self._approval_state
            or self._clarify_state
        )
        if _overlay_cleared:
            self._clear_active_overlays_for_interrupt()
            event.app.current_buffer.reset()
            event.app.invalidate()

        # If we only cleared overlays and the agent is NOT running, stop here
        # (don't fall through to the interrupt/exit path).
        if _overlay_cleared and not (self._agent_running and self.agent):
            return

        if self._agent_running and self.agent:
            if now - self._last_ctrl_c_time < 2.0:
                print("\n⚡ Force exiting...")
                self._should_exit = True
                event.app.exit()
                return

            self._last_ctrl_c_time = now
            print("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
            request_hard_interrupt(self.agent)
        # If there's text or images, clear them (like bash).
        # If everything is already empty, exit.
        elif event.app.current_buffer.text or self._attached_images:
            event.app.current_buffer.reset()
            self._attached_images.clear()
            event.app.invalidate()
        else:
            self._should_exit = True
            event.app.exit()

    def _tui_handle_ctrl_q(self, event):
        """Alternative interrupt/exit shortcut (Ctrl+Q).

        Behaves like Ctrl+C: cancels active prompts, interrupts the
        running agent, or clears the input buffer. Does not support
        the double-press 'force exit' feature of Ctrl+C.
        """
        from cli import _DIM, _RST, _cprint
        # Cancel active voice recording.
        _should_cancel_voice = False
        _recorder_ref = None
        with self._voice_lock:
            if self._voice_recording and self._voice_recorder:
                _recorder_ref = self._voice_recorder
                self._voice_recording = False
                self._voice_continuous = False
                _should_cancel_voice = True
        if _should_cancel_voice:
            _cprint(f"\n{_DIM}Recording cancelled.{_RST}")
            threading.Thread(
                target=_recorder_ref.cancel, daemon=True
            ).start()
            event.app.invalidate()
            return

        # Cancel slash confirmation prompt (foreground UI — cancel and stop).
        if self._slash_confirm_state:
            self._submit_slash_confirm_response("cancel")
            event.app.current_buffer.reset()
            event.app.invalidate()
            return

        # Cancel /model picker (foreground UI — cancel and stop).
        if self._model_picker_state:
            self._close_model_picker()
            event.app.current_buffer.reset()
            event.app.invalidate()
            return

        # Clear all agent-blocking overlays in one shot, then fall through to
        # the agent-interrupt branch so a single Ctrl+Q both clears a stale
        # overlay and interrupts a still-running agent (#14026).
        _overlay_cleared = bool(
            self._sudo_state
            or self._secret_state
            or self._approval_state
            or self._clarify_state
        )
        if _overlay_cleared:
            self._clear_active_overlays_for_interrupt()
            event.app.current_buffer.reset()
            event.app.invalidate()

        if _overlay_cleared and not (self._agent_running and self.agent):
            return

        if self._agent_running and self.agent:
            print("\n⚡ Interrupting agent...")
            request_hard_interrupt(self.agent)
        elif event.app.current_buffer.text or self._attached_images:
            event.app.current_buffer.reset()
            self._attached_images.clear()
            event.app.invalidate()
        else:
            self._should_exit = True
            event.app.exit()

    def _tui_make_clarify_number_handler(self, idx):
        def handler(event):
            if self._clarify_state and not self._clarify_freetext:
                choices = self._clarify_state.get("choices") or []
                # multi-select support: number keys toggle checkboxes instead of submitting
                if self._clarify_state.get("multi_select"):
                    if idx < len(choices):
                        indices = self._clarify_state.get("selected_indices", set())
                        if idx in indices:
                            indices.discard(idx)
                        else:
                            indices.add(idx)
                        event.app.invalidate()
                    elif idx == len(choices):
                        # Toggle "Other" in multi-select mode
                        indices = self._clarify_state.get("selected_indices", set())
                        if idx in indices:
                            indices.discard(idx)
                        else:
                            indices.add(idx)
                        event.app.invalidate()
                    return
                # Original single-select: number keys submit directly
                # Map index to choice (treating "Other" as the last option)
                if idx < len(choices):
                    # Batch mode: lock the numbered choice for the active
                    # question instead of resolving the whole prompt.
                    if self._clarify_state.get("questions"):
                        self._clarify_batch_lock(self._clarify_state, choices[idx])
                        event.app.invalidate()
                        return
                    # Select a numbered choice
                    self._clarify_state["response_queue"].put(choices[idx])
                    self._clarify_state = None
                    self._clarify_freetext = False
                    event.app.invalidate()
                elif idx == len(choices):
                    # Select "Other" option
                    self._clarify_freetext = True
                    event.app.invalidate()
        return handler

    def _tui_restore_stash_payload(self, event, payload) -> None:
        """Put a popped (text, images) payload back into the composer."""
        if not payload:
            return
        text, images = payload
        buf = event.app.current_buffer
        buf.text = text
        buf.cursor_position = len(text)
        if images:
            # Restore attachments the draft was carrying.  Extend rather
            # than replace: the user may have attached something new since
            # the stash was taken and silently dropping it would be data
            # loss.
            for img in images:
                if img not in self._attached_images:
                    self._attached_images.append(img)

    def _tui_handle_stash_panel_up(self, event):
        self._prompt_stash.move_cursor(-1)
        event.app.invalidate()

    def _tui_handle_stash_panel_down(self, event):
        self._prompt_stash.move_cursor(1)
        event.app.invalidate()

    def _tui_handle_stash_panel_delete(self, event):
        """D in the browse panel discards the highlighted draft."""
        self._prompt_stash.delete_at_cursor()
        event.app.invalidate()

    def _tui_handle_stash_panel_close(self, event):
        self._prompt_stash.close_panel()
        event.app.invalidate()

    def _tui_handle_tab(self, event):
        """Tab: accept completion, auto-suggestion, or start completions.

        Priority:
        1. Completion menu open → accept selected completion
        2. Ghost text suggestion available → accept auto-suggestion
        3. Otherwise → start completion menu

        After accepting a provider like 'anthropic:', the completion menu
        closes and complete_while_typing doesn't fire (no keystroke).
        This binding re-triggers completions so stage-2 models appear
        immediately.
        """
        buf = event.current_buffer
        if buf.complete_state:
            # Completion menu is open — accept the selection
            completion = buf.complete_state.current_completion
            if completion is None:
                # Menu open but nothing selected — select first then grab it
                buf.go_to_completion(0)
                completion = buf.complete_state and buf.complete_state.current_completion
            if completion is None:
                return
            # Accept the selected completion
            buf.apply_completion(completion)
        elif buf.suggestion and buf.suggestion.text:
            # No completion menu, but there's a ghost text auto-suggestion — accept it
            buf.insert_text(buf.suggestion.text)
        else:
            # No menu and no suggestion — start completions from scratch
            buf.start_completion()

    def _tui_handle_double_escape(self, event):
        """Double ESC: discard the current draft and any attached images.

        Matches Claude Code / Gemini CLI, where double-Esc is the
        clear-the-composer gesture. It works while the agent is
        streaming, which is the gap Ctrl+C leaves: Ctrl+C interrupts a
        running turn and only clears the draft when idle, so mid-stream
        there was no way to discard a half-typed prompt.

        The draft is appended to history first, so Up recalls it — the
        same undo affordance Claude Code provides, and the reason this
        is safe to bind to a key pressed by reflex.

        Single ESC is the prefix for Alt sequences (escape+enter,
        escape+g, escape+v), so prompt_toolkit's escape-timeout keeps
        those distinct from the double press. Modal prompts bind ESC
        eagerly and are excluded here so cancel still wins.
        """
        buf = event.app.current_buffer
        if not (buf.text or self._attached_images):
            return
        buf.reset(append_to_history=bool(buf.text))
        self._attached_images.clear()
        event.app.invalidate()

    def _tui_handle_ignored_terminal_sequence(self, event):
        """Consume parser-level ignored terminal sequences before self-insert.

        install_ignored_terminal_sequences() in hermes_cli.pt_input_extras
        registers focus reports (CSI I / CSI O) as Keys.Ignore at the
        VT100 parser level. Without this no-op binding the default
        self-insert path would still fire and the bytes would land in
        the buffer.

        Focus-in (CSI I) additionally schedules a rate-limited full
        repaint: while the tab/window was hidden the emulator may have
        coalesced output or repainted the surface, so prompt_toolkit's
        incremental diff would stack a fresh copy of the prompt chrome
        on top of the stale one (#60920 focus-regain variant, #25337).
        """
        try:
            for press in getattr(event, "key_sequence", None) or ():
                if getattr(press, "data", None) == "\x1b[I":
                    self._schedule_focus_regain_redraw()
                    break
        except Exception:
            pass
        return None

    def _tui_handle_escape_modal(self, event):
        """ESC cancels active secret/sudo prompts."""
        if self._secret_state:
            self._cancel_secret_capture()
            event.app.current_buffer.reset()
            event.app.invalidate()
            return
        if self._sudo_state:
            self._sudo_state["response_queue"].put("")
            self._sudo_state = None
            event.app.invalidate()
            return
        if self._slash_confirm_state:
            self._submit_slash_confirm_response("cancel")
            event.app.current_buffer.reset()
            event.app.invalidate()
            return

    def _tui_handle_ctrl_z(self, event):
        """Handle Ctrl+Z - suspend process to background (Unix only)."""
        from cli import _DIM, _RST, _cprint
        if sys.platform == 'win32':
            _cprint(f"\n{_DIM}Suspend (Ctrl+Z) is not supported on Windows.{_RST}")
            event.app.invalidate()
            return
        import signal as _sig
        from prompt_toolkit.application import run_in_terminal
        from hermes_cli.skin_engine import get_active_skin
        agent_name = get_active_skin().get_branding("agent_name", "Hermes Agent")
        msg = f"\n{agent_name} has been suspended. Run `fg` to bring {agent_name} back."
        def _suspend():
            os.write(1, msg.encode())
            os.kill(0, _sig.SIGTSTP)
        run_in_terminal(_suspend)

    def _tui_handle_ctrl_d(self, event):
        """Ctrl+D: delete char under cursor (standard readline behaviour).
        Only exit when the input is empty — same as bash/zsh. Pending
        attached images count as input and block the EOF-exit so the
        user doesn't lose them silently.
        """
        buf = event.app.current_buffer
        if buf.text:
            buf.delete()
        elif self._attached_images:
            # Empty text but pending attachments — no-op, don't exit.
            return
        else:
            self._should_exit = True
            event.app.exit()

    def _tui_recall_without_recollapse(self, buf, move):
        """Run a history-navigation move, suppressing paste-collapse.

        Recalled history can hold the full text of a paste that was
        collapsed to a placeholder at submit time. Loading it back into the
        buffer looks exactly like a fresh large paste to ``_on_text_changed``
        and would be re-collapsed. Set the skip flag around the move; if the
        move didn't change the text (plain cursor movement), clear the flag
        so a later real paste still collapses.
        """
        before = buf.text
        self._skip_paste_collapse = True
        move()
        if buf.text == before:
            self._skip_paste_collapse = False

    def _tui_handle_alt_v(self, event):
        """Alt+V — paste image from clipboard.

        Alt key combos pass through all terminal emulators (sent as
        ESC + key), unlike Ctrl+V which terminals intercept for text
        paste.  This is the reliable way to attach clipboard images
        on WSL2, VSCode, and any terminal over SSH where Ctrl+V
        can't reach the application for image-only clipboard.
        """
        if self._try_attach_clipboard_image():
            event.app.invalidate()
        else:
            # No image found — show a hint
            pass  # silent when no image (avoid noise on accidental press)

    def _tui_handle_ctrl_v(self, event):
        """Fallback image paste for terminals without bracketed paste.

        On Linux terminals (GNOME Terminal, Konsole, etc.), Ctrl+V
        sends raw byte 0x16 instead of triggering a paste.  This
        binding catches that and checks the clipboard for images.
        On terminals that DO intercept Ctrl+V for paste (macOS
        Terminal, iTerm2, VSCode, Windows Terminal), the bracketed
        paste handler fires instead and this binding never triggers.
        """
        if self._try_attach_clipboard_image():
            event.app.invalidate()

    def _tui_handle_ctrl_l(self, event):
        """Ctrl+L: force a clean full-screen repaint.

        Recovers the UI after external terminal buffer drift — tmux /
        cmux tab switches, ``clear`` from a subshell, SSH window
        restores, etc. — that prompt_toolkit can't detect on its own.
        Matches the universal bash/zsh/fish/vim/htop convention.
        """
        self._force_full_redraw()

    def _tui_insert_newline(self, event):
        """Insert a newline for multi-line input (Alt+Enter, and Ctrl+J/Ctrl+Enter
        when multiline shortcuts are on).

        Alt+Enter works on mac/Linux/WSL. On Windows Terminal that keystroke is
        intercepted at the terminal layer (toggles fullscreen) and never reaches
        here — Windows users get newline via Ctrl+Enter, which WT delivers as c-j.
        """
        event.current_buffer.insert_text('\n')

    def _tui_handle_open_in_editor(self, event):
        """Ctrl+G (or Alt+G in VSCode/Cursor) opens the current draft in an external editor."""
        self._open_external_editor(event.current_buffer)

    def _tui_model_picker_down(self, event):
        state = self._model_picker_state
        if not state:
            return
        if state.get("stage") == "provider":
            max_idx = len(state.get("providers") or [])
        else:
            # +1 for "← Back" and Cancel over the filtered visible rows.
            _fp = state.get("_filtered_pairs")
            _visible = len(_fp) if _fp is not None else len(state.get("model_list") or [])
            max_idx = _visible + 1
        state["selected"] = min(max_idx, state.get("selected", 0) + 1)
        event.app.invalidate()

    def _tui_model_picker_up(self, event):
        if self._model_picker_state:
            self._model_picker_state["selected"] = max(0, self._model_picker_state.get("selected", 0) - 1)
            event.app.invalidate()

    def _tui_model_picker_escape(self, event):
        """ESC clears an active filter first, else closes the picker."""
        st = self._model_picker_state
        if st and st.get("stage") == "model" and (st.get("filter") or ""):
            st["filter"] = ""
            st["selected"] = 0
            st["_scroll_offset"] = 0
            event.app.invalidate()
            return
        self._close_model_picker()
        event.app.current_buffer.reset()
        event.app.invalidate()

    def _tui_model_picker_filter_backspace(self, event):
        st = self._model_picker_state
        if not st:
            return
        cur = st.get("filter", "") or ""
        st["filter"] = cur[:-1]
        st["selected"] = 0
        st["_scroll_offset"] = 0
        event.app.invalidate()

    def _tui_make_model_filter_char_handler(self, ch: str):
        def handler(event):
            st = self._model_picker_state
            if not st or st.get("stage") != "model":
                return
            st["filter"] = (st.get("filter", "") or "") + ch
            st["selected"] = 0
            st["_scroll_offset"] = 0
            event.app.invalidate()
        return handler

    def _tui_make_palette_char_handler(self, ch: str):
        def handler(event):
            st = self._command_palette_state
            if not st:
                return
            st["filter"] = (st.get("filter", "") or "") + ch
            st["selected"] = 0
            st["_scroll_offset"] = 0
            event.app.invalidate()
        return handler

    def _tui_make_approval_number_handler(self, idx):
        def handler(event):
            if self._approval_state and idx < len(self._approval_state["choices"]):
                self._approval_state["selected"] = idx
                self._handle_approval_selection()
                event.app.invalidate()
        return handler

    def _tui_make_slash_confirm_number_handler(self, idx):
        def handler(event):
            if self._slash_confirm_state and idx < len(self._slash_confirm_state.get("choices") or []):
                choice = self._slash_confirm_state["choices"][idx][0]
                self._submit_slash_confirm_response(choice)
                event.app.current_buffer.reset()
                event.app.invalidate()
        return handler

    def _tui_clarify_toggle(self, event):
        if self._clarify_state:
            selected = self._clarify_state["selected"]
            indices = self._clarify_state.get("selected_indices", set())
            if selected in indices:
                indices.discard(selected)
            else:
                indices.add(selected)
            event.app.invalidate()

    def _tui_clarify_down(self, event):
        """Move selection down in clarify choices."""
        if self._clarify_state:
            choices = self._clarify_state.get("choices") or []
            max_idx = len(choices)  # last index is the "Other" option
            self._clarify_state["selected"] = min(max_idx, self._clarify_state["selected"] + 1)
            event.app.invalidate()

    def _tui_clarify_up(self, event):
        """Move selection up in clarify choices."""
        if self._clarify_state:
            self._clarify_state["selected"] = max(0, self._clarify_state["selected"] - 1)
            event.app.invalidate()

    def _tui_clarify_batch_tab(self, event):
        state = self._clarify_state
        if state and state.get("questions"):
            self._clarify_batch_set_active(
                state, (state["active"] + 1) % len(state["questions"])
            )
            event.app.invalidate()

    def _tui_clarify_batch_backtab(self, event):
        state = self._clarify_state
        if state and state.get("questions"):
            self._clarify_batch_set_active(
                state, (state["active"] - 1) % len(state["questions"])
            )
            event.app.invalidate()

    def _tui_command_palette_backspace(self, event):
        st = self._command_palette_state
        if st:
            st["filter"] = (st.get("filter", "") or "")[:-1]
            st["selected"] = 0
            st["_scroll_offset"] = 0
            event.app.invalidate()

    def _tui_command_palette_down(self, event):
        st = self._command_palette_state
        if st:
            n = st.get("_visible_count", len(self._command_palette_visible_entries()))
            st["selected"] = min(max(0, n - 1), st.get("selected", 0) + 1)
            event.app.invalidate()

    def _tui_command_palette_up(self, event):
        st = self._command_palette_state
        if st:
            st["selected"] = max(0, st.get("selected", 0) - 1)
            event.app.invalidate()

    def _tui_command_palette_enter(self, event):
        self._handle_command_palette_selection()
        event.app.invalidate()

    def _tui_command_palette_escape(self, event):
        self._close_command_palette()
        event.app.invalidate()

    def _tui_open_command_palette(self, event):
        self._open_command_palette()
        event.app.invalidate()

    def _tui_slash_confirm_down(self, event):
        if self._slash_confirm_state:
            max_idx = len(self._slash_confirm_state.get("choices") or []) - 1
            self._slash_confirm_state["selected"] = min(max_idx, self._slash_confirm_state.get("selected", 0) + 1)
            event.app.invalidate()

    def _tui_slash_confirm_up(self, event):
        if self._slash_confirm_state:
            self._slash_confirm_state["selected"] = max(0, self._slash_confirm_state.get("selected", 0) - 1)
            event.app.invalidate()

    def _tui_approval_down(self, event):
        if self._approval_state:
            max_idx = len(self._approval_state["choices"]) - 1
            self._approval_state["selected"] = min(max_idx, self._approval_state["selected"] + 1)
            event.app.invalidate()

    def _tui_approval_up(self, event):
        if self._approval_state:
            self._approval_state["selected"] = max(0, self._approval_state["selected"] - 1)
            event.app.invalidate()

    def _tui_wake_startup(self):
        from cli import logger
        try:
            self._maybe_start_wake_word()
        except Exception as e:
            logger.debug("wake-word startup skipped: %s", e)

    def _tui_suppress_closed_loop_errors(self, loop, context):
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return  # silently suppress
        if isinstance(exc, KeyError) and "is not registered" in str(exc):
            return  # suppress selector registration failures (#6393)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.EIO:
            return  # suppress I/O errors from broken stdout on interrupt (#13710)
        # Fall back to default handler for everything else
        loop.default_exception_handler(context)

    def _tui_handle_enter(self, event):
        """Handle Enter key - submit input.

        Routes to the correct queue based on active UI state:
        - Sudo password prompt: password goes to sudo response queue
        - Approval selection: selected choice goes to approval response queue
        - Clarify freetext mode: answer goes to the clarify response queue
        - Clarify choice mode: selected choice goes to the clarify response queue
        - Agent running: goes to _interrupt_queue (chat() monitors this)
        - Agent idle: goes to _pending_input (process_loop monitors this)
        Commands (starting with /) always go to _pending_input so they're
        handled as commands, not sent as interrupt text to the agent.
        """
        from cli import (
            CLI_CONFIG,
            _ACCENT,
            _DIM,
            _RST,
            _apply_backslash_line_continuation,
            _cprint,
            _hermes_home,
            _is_backslash_line_continuation,
            _looks_like_slash_command,
        )
        if self._tui_enter_overlay(event):
            return

        # --- Normal input routing ---
        raw_text = event.app.current_buffer.text
        if (
            self._tui_multiline_shortcuts
            and event.app.current_buffer.cursor_position == len(raw_text)
            and _is_backslash_line_continuation(raw_text)
        ):
            continued = _apply_backslash_line_continuation(raw_text)
            event.app.current_buffer.text = continued
            event.app.current_buffer.cursor_position = len(continued)
            event.app.invalidate()
            return
        text = raw_text.strip()
        has_images = bool(self._attached_images)
        if text or has_images:
            # Handle /model directly on the UI thread so interactive pickers
            # can safely use prompt_toolkit terminal handoff helpers.
            if self._should_handle_model_command_inline(text, has_images=has_images):
                if not self.process_command(text):
                    self._should_exit = True
                    if event.app.is_running:
                        event.app.exit()
                event.app.current_buffer.reset(append_to_history=True)
                # Force a repaint: process_command() prints through
                # patch_stdout (scrolls output above the prompt) and never
                # invalidates the app, so the just-cleared input area can
                # keep showing the submitted text until some unrelated
                # redraw fires. Every other early-return branch in this
                # handler invalidates after reset — match them.
                event.app.invalidate()
                return

            # Handle /steer while the agent is running immediately on the
            # UI thread.  Queuing through _pending_input would deadlock the
            # steer until after the agent loop finishes (process_loop is
            # blocked inside self.chat()), which turns /steer into a
            # post-run next-turn message — defeating mid-run injection.
            # agent.steer() is thread-safe (holds _pending_steer_lock).
            if self._should_handle_steer_command_inline(text, has_images=has_images):
                self.process_command(text)
                event.app.current_buffer.reset(append_to_history=True)
                # Force a repaint after clearing the buffer.  /steer is
                # dispatched mid-run while the agent streams output through
                # patch_stdout; process_command() never invalidates the
                # app, so without this the submitted "/steer <text>" can
                # linger in the input area (looking unsent) and invite an
                # accidental re-submit. See issue #34569.
                event.app.invalidate()
                return

            # Same treatment for /bg and /btw while the agent is
            # running.  Queuing them defeats the entire point of the
            # commands: process_loop is blocked inside self.chat(), so the
            # side task would only start once the foreground turn it was
            # meant to run alongside has already finished (#75221).  The
            # foreground turn is left alone: no interrupt, no steer.
            if self._should_handle_background_command_inline(
                text, has_images=has_images
            ):
                self.process_command(text)
                event.app.current_buffer.reset(append_to_history=True)
                # Repaint for the same reason as the /steer branch above:
                # process_command() prints through patch_stdout and never
                # invalidates the app, so the submitted text can linger in
                # the input area looking unsent.
                event.app.invalidate()
                return

            # Snapshot and clear attached images
            images = list(self._attached_images)
            self._attached_images.clear()
            event.app.invalidate()
            # Bundle text + images as a tuple when images are present
            payload = (text, images) if images else text
            # A bang command is treated like a slash command while the
            # agent is busy: it must never be routed into steer/redirect
            # (which would inject `!git status` into the model's context as
            # a prompt). It queues and runs locally once the loop drains.
            _is_local_dispatch = bool(text) and (
                _looks_like_slash_command(text) or text.strip().startswith("!")
            )
            if self._agent_running and not _is_local_dispatch:
                _effective_mode = self.busy_input_mode
                redirected = False
                if _effective_mode == "steer":
                    # Route Enter through /steer — inject mid-run after the
                    # next tool call.  Images can't ride along (steer only
                    # appends text), so fall back to queue when images are
                    # attached.  If the agent lacks steer() or rejects the
                    # payload, also fall back to queue so nothing is lost.
                    if images or not text:
                        _effective_mode = "queue"
                    else:
                        accepted = False
                        try:
                            if self.agent is not None and hasattr(self.agent, "steer"):
                                accepted = bool(self.agent.steer(text))
                        except Exception as exc:
                            _cprint(f"  {_DIM}Steer failed ({exc}) — queued for next turn.{_RST}")
                            accepted = False
                        if accepted:
                            preview = text[:80] + ("..." if len(text) > 80 else "")
                            _cprint(f"  {_ACCENT}⏩ Steered: '{preview}'{_RST}")
                        else:
                            _effective_mode = "queue"
                if _effective_mode == "queue":
                    # Queue for the next turn instead of interrupting
                    self._pending_input.put(payload)
                    preview = text if text else f"[{len(images)} image{'s' if len(images) != 1 else ''} attached]"
                    _cprint(f"  Queued for the next turn: {preview[:80]}{'...' if len(preview) > 80 else ''}")
                elif _effective_mode == "interrupt":
                    if not images and text:
                        try:
                            if (
                                self.agent is not None
                                and getattr(
                                    self.agent,
                                    "_supports_active_turn_redirect",
                                    False,
                                )
                                is True
                                and hasattr(self.agent, "redirect")
                            ):
                                redirected = bool(self.agent.redirect(text))
                        except Exception:
                            redirected = False
                    if redirected:
                        preview = text[:80] + ("..." if len(text) > 80 else "")
                        _cprint(f"  {_ACCENT}↪ Redirected current turn: '{preview}'{_RST}")
                    else:
                        # Compatibility path for older agents, multimodal
                        # follow-ups, or a turn that finished in the race.
                        self._interrupt_queue.put(payload)
                        try:
                            _dbg = _hermes_home / "interrupt_debug.log"
                            with open(_dbg, "a", encoding="utf-8") as _f:
                                _f.write(f"{time.strftime('%H:%M:%S')} ENTER: queued interrupt msg={str(payload)[:60]!r}, "
                                         f"agent_running={self._agent_running}\n")
                        except Exception:
                            pass
                # First-touch onboarding: on the very first busy-while-running
                # event for this install, print a one-line tip explaining the
                # /busy knob.  Flag persists to config.yaml and never fires
                # again.  Guarded for exceptions so onboarding can't break
                # the input loop.
                try:
                    from agent.onboarding import (
                        BUSY_INPUT_FLAG,
                        busy_input_hint_cli,
                        is_seen,
                        mark_seen,
                    )
                    if not is_seen(CLI_CONFIG, BUSY_INPUT_FLAG):
                        _hint_mode = "redirect" if redirected else _effective_mode
                        _cprint(f"  {_DIM}{busy_input_hint_cli(_hint_mode)}{_RST}")
                        mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
                        CLI_CONFIG.setdefault("onboarding", {}).setdefault("seen", {})[BUSY_INPUT_FLAG] = True
                except Exception:
                    pass
            else:
                self._pending_input.put(payload)
            # History stores real pasted content, not the placeholder, so
            # up-arrow recall restores the actual text.
            self._inline_pastes(event.app.current_buffer)
            event.app.current_buffer.reset(append_to_history=True)

    def _tui_enter_overlay(self, event) -> bool:
        """Enter while a modal overlay (sudo/secret/approval/slash-confirm/model picker/clarify) is up: submit it. True when handled."""
        from cli import _cprint
        # --- Sudo password prompt: submit the typed password ---
        if self._sudo_state:
            text = event.app.current_buffer.text
            self._sudo_state["response_queue"].put(text)
            self._sudo_state = None
            event.app.invalidate()
            return True

        # --- Secret prompt: submit the typed secret ---
        if self._secret_state:
            text = event.app.current_buffer.text
            self._submit_secret_response(text)
            event.app.current_buffer.reset()
            event.app.invalidate()
            return True

        # --- Approval selection: confirm the highlighted choice ---
        if self._approval_state:
            self._handle_approval_selection()
            event.app.invalidate()
            return True

        # --- Slash-command confirmation: submit typed or highlighted choice ---
        if self._slash_confirm_state:
            text = event.app.current_buffer.text.strip()
            choices = self._slash_confirm_state.get("choices") or []
            choice = self._normalize_slash_confirm_choice(text, choices) if text else None
            if choice is None:
                selected = self._slash_confirm_state.get("selected", 0)
                if 0 <= selected < len(choices):
                    choice = choices[selected][0]
            self._submit_slash_confirm_response(choice or "cancel")
            event.app.current_buffer.reset()
            event.app.invalidate()
            return True

        # --- /model picker modal ---
        if self._model_picker_state:
            try:
                # Picker selections follow the same session-scoped default
                # as /model <name>; honour model.persist_switch_by_default.
                from hermes_cli.model_switch import resolve_persist_behavior

                self._handle_model_picker_selection(
                    persist_global=resolve_persist_behavior(False, False)
                )
            except Exception as _exc:
                _cprint(f"  ✗ Model selection failed: {_exc}")
                self._close_model_picker()
            event.app.current_buffer.reset()
            event.app.invalidate()
            return True

        # --- Clarify freetext mode: user typed their own answer ---
        if self._clarify_freetext and self._clarify_state:
            text = event.app.current_buffer.text.strip()
            if text:
                state = self._clarify_state
                # Batch mode: lock the typed answer for the active question
                if state.get("questions"):
                    base = getattr(self, '_clarify_multi_base', None)
                    if base is not None:
                        # Multi-select "Other": append the typed answer to
                        # the checked labels as a JSON array string.
                        answer = json.dumps(base + [text], ensure_ascii=False)
                        meta = {"kind": "multi", "choices": list(base), "other_text": text}
                        self._clarify_multi_base = None
                    else:
                        answer = text
                        meta = {"kind": "other", "other_text": text}
                    self._clarify_freetext = False
                    self._clarify_prefill = ""
                    self._clarify_batch_lock(state, answer, meta=meta)
                    event.app.current_buffer.reset()
                    event.app.invalidate()
                    return True
                # multi-select: prepend previously checked real choices
                base = getattr(self, '_clarify_multi_base', None)
                if base:
                    text = ", ".join(base) + ", " + text
                    self._clarify_multi_base = None
                self._clarify_state["response_queue"].put(text)
                self._clarify_state = None
                self._clarify_freetext = False
                event.app.current_buffer.reset()
                event.app.invalidate()
            return True

        # --- Clarify choice mode: confirm the highlighted selection ---
        if self._clarify_state and not self._clarify_freetext:
            state = self._clarify_state
            # Batch mode: Enter locks the active question's answer and
            # advances to the next unanswered question.
            if state.get("questions"):
                self._clarify_batch_enter(state)
                # Editing an earlier "Other" answer: prefill the composer
                # with the previously typed text.
                if self._clarify_freetext and self._clarify_prefill:
                    event.app.current_buffer.text = self._clarify_prefill
                    event.app.current_buffer.cursor_position = len(self._clarify_prefill)
                    self._clarify_prefill = ""
                event.app.invalidate()
                return True
            selected = state["selected"]
            choices = state.get("choices") or []
            # multi-select support: submit comma-joined list of checked choices
            if state.get("multi_select"):
                indices = state.get("selected_indices")
                if not indices:
                    # Nothing checked → submit empty string (parses to [])
                    state["response_queue"].put("")
                    self._clarify_state = None
                    event.app.invalidate()
                    return True
                sorted_idx = sorted(indices)
                selected_choices = [choices[i] for i in sorted_idx if i < len(choices)]
                other_checked = len(choices) in sorted_idx
                if other_checked and selected_choices:
                    # "Other" + real choices: store base choices, switch to freetext
                    # so the user can type a custom answer that gets appended
                    self._clarify_multi_base = selected_choices
                    self._clarify_freetext = True
                    event.app.invalidate()
                    return True
                if selected_choices:
                    state["response_queue"].put(", ".join(selected_choices))
                    self._clarify_state = None
                    event.app.invalidate()
                    return True
                # Only "Other" was checked → switch to freetext
                self._clarify_freetext = True
                event.app.invalidate()
                return True
            # Original single-select behavior: submit the highlighted choice
            if selected < len(choices):
                state["response_queue"].put(choices[selected])
                self._clarify_state = None
                event.app.invalidate()
            else:
                # "Other" selected → switch to freetext
                self._clarify_freetext = True
                event.app.invalidate()
            return True
        return False

    def _tui_handle_paste(self, event):
        """Handle terminal paste — detect clipboard images.

        When the terminal supports bracketed paste, Ctrl+V / Cmd+V
        triggers this with the pasted text. We only auto-attach a
        clipboard image for image-only/empty paste gestures so text
        pastes and dictation do not accidentally attach stale images.

        Large pastes (5+ lines) are collapsed to a file reference
        placeholder while preserving any existing user text in the
        buffer.
        """
        from cli import (
            _hermes_home,
            _should_auto_attach_clipboard_image_on_paste,
            _strip_leaked_bracketed_paste_wrappers,
            _strip_leaked_terminal_responses_with_meta,
            datetime,
            logger,
        )
        # Diagnostic canary: measure how long the paste handler blocks
        # the prompt_toolkit event loop. If this exceeds ~500ms we log
        # it so recurring "CLI freezes on paste" reports (issue #16263,
        # macOS Tahoe 26 + iTerm2/Ghostty) arrive with data attached.
        _paste_handler_start = time.perf_counter()
        _paste_raw_size = len(event.data or "")
        pasted_text = event.data or ""
        # Normalise line endings — Windows \r\n and old Mac \r both become \n
        # so the 5-line collapse threshold and display are consistent.
        pasted_text = pasted_text.replace('\r\n', '\n').replace('\r', '\n')
        pasted_text = _strip_leaked_bracketed_paste_wrappers(pasted_text)
        pasted_text, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(pasted_text)
        if _had_mouse_reports:
            self._recover_terminal_input_modes(reason="mouse reports leaked into bracketed paste payload")
        if _should_auto_attach_clipboard_image_on_paste(pasted_text) and self._try_attach_clipboard_image():
            event.app.invalidate()
        if pasted_text:
            # Sanitize surrogate characters (e.g. from Word/Google Docs paste) before writing
            from run_agent import _sanitize_surrogates
            pasted_text = _sanitize_surrogates(pasted_text)
            line_count = pasted_text.count('\n')
            buf = event.current_buffer
            threshold = self.config.get("paste_collapse_threshold", 5)
            char_threshold = self.config.get("paste_collapse_char_threshold", 2000)
            lines_hit = threshold > 0 and line_count >= threshold
            chars_hit = char_threshold > 0 and len(pasted_text) >= char_threshold
            if (lines_hit or chars_hit) and not buf.text.strip().startswith('/'):
                self._tui_paste_counter[0] += 1
                paste_dir = _hermes_home / "pastes"
                paste_dir.mkdir(parents=True, exist_ok=True)
                paste_file = paste_dir / f"paste_{self._tui_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                paste_file.write_text(pasted_text, encoding="utf-8")
                logger.info("Collapsed paste #%d: %d lines, %d chars -> %s", self._tui_paste_counter[0], line_count + 1, len(pasted_text), paste_file)
                placeholder = f"[Pasted text #{self._tui_paste_counter[0]}: {line_count + 1} lines \u2192 {paste_file}]"
                prefix = ""
                if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n':
                    prefix = "\n"
                self._tui_paste_just_collapsed[0] = True
                buf.insert_text(prefix + placeholder)
            else:
                buf.insert_text(pasted_text)
        _paste_handler_elapsed_ms = (time.perf_counter() - _paste_handler_start) * 1000.0
        if _paste_handler_elapsed_ms > 500.0:
            logger.warning(
                "Slow bracketed-paste handler: %.1fms to process %d bytes "
                "(%d lines) on %s. If the input becomes unresponsive after "
                "this, attach this log line to the bug report.",
                _paste_handler_elapsed_ms,
                _paste_raw_size,
                pasted_text.count('\n') + 1 if pasted_text else 0,
                sys.platform,
            )

    def _tui_on_text_changed(self, buf):
        """Detect large pastes and collapse them to a file reference.

        When bracketed paste is available, handle_paste collapses
        large pastes directly.  This handler is a fallback for
        terminals without bracketed paste support.

        Two heuristics (either triggers collapse):
        1. Many characters added at once (chars_added > 1) — works
           when the terminal delivers the paste in one event-loop tick.
        2. Newline count jumped by 4+ in a single text-change event —
           catches terminals that feed characters individually but
           still batch newlines.  Alt+Enter only adds 1 newline per
           event so it never triggers this.
        """
        from cli import (
            _hermes_home,
            _strip_leaked_bracketed_paste_wrappers,
            _strip_leaked_terminal_responses_with_meta,
            datetime,
            logger,
        )
        text = _strip_leaked_bracketed_paste_wrappers(buf.text)
        text, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(text)
        if _had_mouse_reports:
            self._recover_terminal_input_modes(reason="mouse reports leaked into prompt buffer")
        if text != buf.text:
            cursor = min(buf.cursor_position, len(text))
            self._tui_paste_just_collapsed[0] = True
            buf.text = text
            buf.cursor_position = cursor
            self._tui_prev_text_len[0] = len(text)
            self._tui_prev_newline_count[0] = text.count('\n')
            return
        chars_added = len(text) - self._tui_prev_text_len[0]
        self._tui_prev_text_len[0] = len(text)
        if self._tui_paste_just_collapsed[0] or self._skip_paste_collapse:
            self._tui_paste_just_collapsed[0] = False
            self._skip_paste_collapse = False
            self._tui_prev_newline_count[0] = text.count('\n')
            return
        line_count = text.count('\n')
        newlines_added = line_count - self._tui_prev_newline_count[0]
        self._tui_prev_newline_count[0] = line_count
        is_paste = chars_added > 1 or newlines_added >= 4
        threshold = self.config.get("paste_collapse_threshold_fallback", 5)
        char_threshold = self.config.get("paste_collapse_char_threshold", 2000)
        lines_hit = threshold > 0 and line_count >= threshold
        chars_hit = char_threshold > 0 and len(text) >= char_threshold
        if (lines_hit or chars_hit) and is_paste and not text.startswith('/'):
            self._tui_paste_counter[0] += 1
            paste_dir = _hermes_home / "pastes"
            paste_dir.mkdir(parents=True, exist_ok=True)
            paste_file = paste_dir / f"paste_{self._tui_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
            paste_file.write_text(text, encoding="utf-8")
            logger.info("Collapsed paste #%d: %d lines, %d chars -> %s (fallback)", self._tui_paste_counter[0], line_count + 1, len(text), paste_file)
            self._tui_paste_just_collapsed[0] = True
            buf.text = f"[Pasted text #{self._tui_paste_counter[0]}: {line_count + 1} lines \u2192 {paste_file}]"
            buf.cursor_position = len(buf.text)

    def _tui_handle_prompt_stash(self, event):
        """Ctrl+S: stash the current draft, or restore/browse a stashed one.

        - Composer has content → push it onto the stash and clear the input.
        - Composer empty, one stashed draft → pop it straight back.
        - Composer empty, several stashed → open the browse panel.
        - Browse panel open → close it.

        Pushing onto a stack (rather than a single slot) is what makes
        repeated Ctrl+S safe: a second stash never silently overwrites the
        first, both stay reachable in the panel.
        """
        from hermes_cli.prompt_stash import (
            ACTION_OPEN_PANEL,
            ACTION_RESTORED,
            ACTION_STASHED,
            resolve_ctrl_s,
        )

        buf = event.app.current_buffer
        action, payload = resolve_ctrl_s(
            self._prompt_stash, buf.text, self._attached_images
        )

        if action == ACTION_STASHED:
            # reset() (not `text = ""`) so completion state, selection, and
            # the undo stack are cleared along with the text.
            buf.reset()
            self._attached_images.clear()
        elif action == ACTION_RESTORED:
            self._tui_restore_stash_payload(event, payload)
        elif action == ACTION_OPEN_PANEL:
            pass  # resolve_ctrl_s already flipped panel_open

        event.app.invalidate()

    def _tui_handle_stash_panel_restore(self, event):
        """Enter in the browse panel restores the highlighted draft."""
        payload = self._prompt_stash.restore_at_cursor()
        self._tui_restore_stash_payload(event, payload)
        event.app.invalidate()

    def _tui_history_up(self, event):
        """Up arrow: browse history when on first line, else move cursor up."""
        buf = event.app.current_buffer
        self._tui_recall_without_recollapse(buf, lambda: buf.auto_up(count=event.arg))

    def _tui_history_down(self, event):
        """Down arrow: browse history when on last line, else move cursor down."""
        buf = event.app.current_buffer
        self._tui_recall_without_recollapse(buf, lambda: buf.auto_down(count=event.arg))

    def _tui_image_bar_fragments(self):
        from cli import _format_image_attachment_badges
        if not self._attached_images:
            return []
        badges = _format_image_attachment_badges(
            self._attached_images,
            self._image_counter,
        )
        return [("class:image-badge", f" {badges} ")]

    def _tui_voice_status_fragments(self):
        return self._get_voice_status_fragments()

    def _tui_spinner_text(self):
        spinner_line = self._render_spinner_text()
        if not spinner_line:
            return []
        return [('class:hint', spinner_line)]

    def _tui_spinner_height(self):
        return self._spinner_widget_height()

    def _tui_hint_height(self):
        if self._sudo_state or self._secret_state or self._approval_state or self._slash_confirm_state or self._clarify_state or self._command_running:
            return 1
        # Keep a spacer while the agent runs on roomy terminals, but reclaim
        # the row on narrow/mobile screens where every line matters.
        return self._agent_spacer_height()

    def _tui_init_run_state(self):
        """Reset the per-run REPL state (queues, modal states, voice state, config watcher)."""
        # State for async operation
        self._agent_running = False
        self._pending_input = queue.Queue()     # For normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # For messages typed while agent is running
        # Seeded -q handoff: main() can't put directly into _pending_input
        # (this reinit would discard it), so the seeded first message rides
        # in on an attribute and is enqueued into the fresh queue here.
        _seed_msg = getattr(self, "_seeded_first_message", None)
        if _seed_msg is not None:
            self._seeded_first_message = None
            self._pending_input.put(_seed_msg)
        # See constructor note. Mirrored here for the run() path that skips
        # the earlier __init__ branch.
        self._last_turn_interrupted = False
        self._should_exit = False
        self._last_ctrl_c_time = 0  # Track double Ctrl+C for force exit

        # Give plugin manager a CLI reference so plugins can inject messages
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._cli_ref = self

        # Config file watcher — detect mcp_servers changes and auto-reload
        from hermes_cli.config import get_config_path as _get_config_path
        _cfg_path = _get_config_path()
        self._config_mtime: float = _cfg_path.stat().st_mtime if _cfg_path.exists() else 0.0
        self._config_mcp_servers: dict = self.config.get("mcp_servers") or {}
        self._last_config_check: float = 0.0  # monotonic time of last check

        # Clarify tool state: interactive question/answer with the user.
        # When the agent calls the clarify tool, _clarify_state is set and
        # the prompt_toolkit UI switches to a selection mode.
        self._clarify_state = None      # dict with question, choices, selected, response_queue
        self._clarify_freetext = False  # True when user chose "Other" and is typing
        self._clarify_deadline = 0      # monotonic timestamp when the clarify times out

        # Sudo password prompt state (similar mechanism to clarify)
        self._sudo_state = None         # dict with response_queue when active
        self._sudo_deadline = 0
        self._modal_input_snapshot = None

        # Dangerous command approval state (similar mechanism to clarify)
        self._approval_state = None     # dict with command, description, choices, selected, response_queue
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()  # serialize concurrent approval prompts (delegation race fix)

        # Destructive slash-command confirmation state (/new, /clear, /undo).
        # These prompts are answered through the prompt_toolkit composer, not
        # raw input(), so the option labels stay visible and Enter does not EOF
        # the whole app.
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0

        # Slash command loading state
        self._command_running = False
        self._command_blocks_input = False
        self._command_status = ""

        # Secure secret capture state for skill setup
        self._secret_state = None       # dict with var_name, prompt, metadata, response_queue
        self._secret_deadline = 0

        # Clipboard image attachments (paste images into the CLI)
        self._attached_images: list[Path] = []
        self._image_counter = 0

        # Voice mode state (protected by _voice_lock for cross-thread access)
        self._voice_lock = threading.Lock()
        self._voice_mode = False        # Whether voice mode is enabled
        self._voice_tts = False         # Whether TTS output is enabled
        self._voice_recorder = None     # AudioRecorder instance (lazy init)
        self._voice_recording = False   # Whether currently recording
        self._voice_processing = False  # Whether STT is in progress
        self._voice_continuous = False  # Whether to auto-restart after agent responds
        self._voice_tts_done = threading.Event()  # Signals TTS playback finished
        self._voice_tts_done.set()  # Initially "done" (no TTS pending)
        self._voice_tts_stop = None  # active streaming pipeline's stop event
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # most recently spoken TTS text (echo guard, #75780)
        self._voice_barge_phase = None  # "generation" or "playback" phase of the last barge trip

        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            self._install_tool_callbacks()

        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            self._ensure_tirith_security()

    def _tui_build_key_bindings(self):
        """Build the prompt_toolkit KeyBindings for the REPL input area."""
        from cli import (
            CLI_CONFIG,
            _bind_prompt_submit_keys,
            _cli_multiline_shortcuts_enabled,
            _preserve_ctrl_enter_newline,
            logger,
        )
        # Key bindings for the input area
        kb = KeyBindings()

        _multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled(self.config or CLI_CONFIG)
        self._tui_multiline_shortcuts = _multiline_shortcuts_enabled

        from prompt_toolkit.keys import Keys as _IgnoreKeys

        kb.add(_IgnoreKeys.Ignore, eager=True)(self._tui_handle_ignored_terminal_sequence)

        _bind_prompt_submit_keys(
            kb,
            self._tui_handle_enter,
            multiline_shortcuts_enabled=_multiline_shortcuts_enabled,
        )

        kb.add('escape', 'enter')(self._tui_insert_newline)

        # Ctrl+J inserts a newline (matches Claude Code / Codex / OpenCode).
        # Windows Terminal delivers Ctrl+Enter as the same c-j code, so this
        # covers Ctrl+Enter there. display.cli_multiline_shortcuts: false
        # restores legacy c-j submit on unusual POSIX PTYs where Enter is LF.
        if _multiline_shortcuts_enabled or _preserve_ctrl_enter_newline():
            kb.add('c-j')(self._tui_insert_newline)

        # VSCode/Cursor bind Ctrl+G to "Find Next" at the editor level, so
        # the keystroke never reaches the embedded terminal. Alt+G is unbound
        # in those IDEs and arrives here as ('escape', 'g') — register it as
        # a fallback so the editor handoff works inside Cursor/VSCode too.
        _editor_filter = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._sudo_state and not self._secret_state
        )

        kb.add('c-g', filter=_editor_filter)(kb.add('escape', 'g', filter=_editor_filter)(self._tui_handle_open_in_editor))

        # --- Ctrl+S prompt stash -------------------------------------------
        # Park a half-written draft, send something else, then bring the draft
        # back.  Suppressed while a modal prompt owns the composer (sudo /
        # secret / approval / clarify) so Ctrl+S can't stash a password.
        _stash_filter = Condition(
            lambda: not self._clarify_state
            and not self._approval_state
            and not self._sudo_state
            and not self._secret_state
            and not self._slash_confirm_state
            and not self._model_picker_state
        )
        _stash_panel_filter = Condition(
            lambda: self._prompt_stash.panel_open and bool(len(self._prompt_stash))
        )

        kb.add('c-s', filter=_stash_filter)(self._tui_handle_prompt_stash)

        kb.add('up', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_up)

        kb.add('down', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_down)

        kb.add('enter', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_restore)

        kb.add('d', filter=_stash_panel_filter, eager=True)(kb.add('D', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_delete))

        kb.add('escape', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_close)

        kb.add('tab', eager=True)(self._tui_handle_tab)

        # --- Clarify tool: arrow-key navigation for multiple-choice questions ---

        kb.add('up', filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext))(self._tui_clarify_up)

        kb.add('down', filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext))(self._tui_clarify_down)

        # multi-select support: Space toggles the checkbox at the current cursor position
        kb.add('space', filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext and self._clarify_state.get("multi_select")))(self._tui_clarify_toggle)

        # Batch clarify: Tab cycles the active question (any-order answering;
        # moving onto an answered question lets the user re-answer it before
        # the batch completes). Registered after the generic tab handler so
        # this filtered binding wins while the batch panel is open.
        kb.add('tab', filter=Condition(lambda: bool(self._clarify_state) and bool(self._clarify_state.get("questions")) and not self._clarify_freetext), eager=True)(self._tui_clarify_batch_tab)

        # Shift-Tab walks backwards through the questions.
        kb.add('s-tab', filter=Condition(lambda: bool(self._clarify_state) and bool(self._clarify_state.get("questions")) and not self._clarify_freetext), eager=True)(self._tui_clarify_batch_backtab)

        # Number keys for quick clarify selection (1-9, 0 for 10th item)

        for _num in range(10):
            # 1-9 select items 0-8, 0 selects item 9 (10thitem)
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext))(self._tui_make_clarify_number_handler(_idx))

        # --- Dangerous command approval: arrow-key navigation ---

        kb.add('up', filter=Condition(lambda: bool(self._approval_state)))(self._tui_approval_up)

        kb.add('down', filter=Condition(lambda: bool(self._approval_state)))(self._tui_approval_down)

        # --- Slash-command confirmation: arrow-key navigation ---
        kb.add('up', filter=Condition(lambda: bool(self._slash_confirm_state)))(self._tui_slash_confirm_up)

        kb.add('down', filter=Condition(lambda: bool(self._slash_confirm_state)))(self._tui_slash_confirm_down)

        # --- /model picker: arrow-key navigation ---
        kb.add('up', filter=Condition(lambda: bool(self._model_picker_state)))(self._tui_model_picker_up)

        kb.add('down', filter=Condition(lambda: bool(self._model_picker_state)))(self._tui_model_picker_down)

        def _model_picker_typing_active() -> bool:
            # Type-to-filter is only live on the model stage (concrete list).
            st = self._model_picker_state
            return bool(st) and st.get("stage") == "model"

        # Printable ASCII (space through ~) narrows the model list as you type.
        import string as _string
        for _ch in _string.digits + _string.ascii_letters + "-_.:/ ":
            kb.add(_ch, filter=Condition(_model_picker_typing_active))(
                self._tui_make_model_filter_char_handler(_ch)
            )

        kb.add('backspace', filter=Condition(_model_picker_typing_active))(self._tui_model_picker_filter_backspace)

        kb.add('escape', filter=Condition(lambda: bool(self._model_picker_state)), eager=True)(self._tui_model_picker_escape)

        # --- Ctrl+P command palette keybindings ---
        def _palette_active() -> bool:
            return bool(self._command_palette_state)

        kb.add('c-p', filter=Condition(lambda: not self._command_palette_state and not self._model_picker_state and not self._clarify_state and not self._approval_state and not self._slash_confirm_state and not self._sudo_state and not self._secret_state))(self._tui_open_command_palette)

        kb.add('up', filter=Condition(_palette_active))(self._tui_command_palette_up)

        kb.add('down', filter=Condition(_palette_active))(self._tui_command_palette_down)

        kb.add('enter', filter=Condition(_palette_active))(self._tui_command_palette_enter)

        kb.add('backspace', filter=Condition(_palette_active))(self._tui_command_palette_backspace)

        kb.add('escape', filter=Condition(_palette_active), eager=True)(self._tui_command_palette_escape)

        import string as _pstring
        for _pch in _pstring.digits + _pstring.ascii_letters + "-_.:/ ":
            kb.add(_pch, filter=Condition(_palette_active))(self._tui_make_palette_char_handler(_pch))

        # Number keys for quick approval selection (1-9, 0 for 10th item)

        for _num in range(10):
            # 1-9 select items 0-8, 0 selects item 9 (10th item)
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=Condition(lambda: bool(self._approval_state)))(self._tui_make_approval_number_handler(_idx))

        # Number keys for quick slash-confirm selection (1-9, 0 for 10th item)

        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=Condition(lambda: bool(self._slash_confirm_state)))(self._tui_make_slash_confirm_number_handler(_idx))

        # --- History navigation: up/down browse history in normal input mode ---
        # The TextArea is multiline, so by default up/down only move the cursor.
        # Buffer.auto_up/auto_down handle both: cursor movement when multi-line,
        # history browsing when on the first/last line (or single-line input).
        _normal_input = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._slash_confirm_state and not self._sudo_state and not self._secret_state and not self._model_picker_state and not self._command_palette_state
        )

        kb.add('up', filter=_normal_input)(self._tui_history_up)

        kb.add('down', filter=_normal_input)(self._tui_history_down)

        kb.add('c-l')(self._tui_handle_ctrl_l)

        kb.add('c-c')(self._tui_handle_ctrl_c)

        # Ctrl+Shift+C: no binding needed. Terminal emulators (GNOME Terminal,
        # iTerm2, kitty, Windows Terminal, etc.) intercept Ctrl+Shift+C before
        # the keystroke reaches the application's stdin — prompt_toolkit never
        # sees it, and prompt_toolkit's key spec parser doesn't even recognise
        # 'c-S-c' anyway (the Shift modifier is meaningless on control-sequence
        # keys). #19884 added a handler for this; #19895 patched the resulting
        # startup crash with try/except. Both were based on a misreading of how
        # terminal key events propagate. Deleting the dead handler outright.

        kb.add('c-q')(self._tui_handle_ctrl_q)

        kb.add('c-d')(self._tui_handle_ctrl_d)

        _modal_prompt_active = Condition(
            lambda: bool(self._secret_state or self._sudo_state or self._slash_confirm_state)
        )

        kb.add('escape', filter=_modal_prompt_active, eager=True)(self._tui_handle_escape_modal)

        kb.add('escape', 'escape', filter=~_modal_prompt_active)(self._tui_handle_double_escape)

        kb.add('c-z')(self._tui_handle_ctrl_z)

        # Voice push-to-talk key: configurable via config.yaml (voice.record_key)
        # Default: Ctrl+B (avoids conflict with Ctrl+R readline reverse-search).
        # Config spellings (ctrl/control/alt/option/opt) are normalized to
        # prompt_toolkit's c-x / a-x format via ``normalize_voice_record_key_for_prompt_toolkit``
        # so the same config value binds identically in the TUI and CLI
        # (Copilot round-9 review on #19835). ``super``/``win``/``windows``
        # configs silently fall back to the default here since prompt_toolkit
        # has no super modifier — log a warning so users notice the
        # TUI/CLI split instead of a silent mismatch (round-11).
        _raw_key: object = "ctrl+b"
        try:
            from hermes_cli.config import load_config
            from hermes_cli.voice import (
                normalize_voice_record_key_for_prompt_toolkit,
                pt_key_to_sequence,
                voice_record_key_from_config,
            )
            _raw_key = voice_record_key_from_config(load_config())
            _voice_key = normalize_voice_record_key_for_prompt_toolkit(_raw_key)
            if (
                isinstance(_raw_key, str)
                and _raw_key.strip().lower().split("+", 1)[0].strip() in {"super", "win", "windows"}
                and _voice_key == "c-b"
            ):
                logger.warning(
                    "voice.record_key %r uses a TUI-only modifier (super/win); "
                    "CLI fell back to Ctrl+B. Use ctrl+<key> or alt+<key> for "
                    "cross-runtime parity.",
                    _raw_key,
                )
        except Exception:
            _voice_key = "c-b"

        # Cache the UI label here — same ``_raw_key`` that drives the
        # prompt_toolkit binding below. Every status / placeholder /
        # recording-hint render reads this cached value so display can
        # never drift from the live keybinding even if the user edits
        # voice.record_key mid-session (Copilot round-13 on #19835).
        self.set_voice_record_key_cache(_raw_key)

        kb.add(*pt_key_to_sequence(_voice_key))(self._tui_handle_voice_record)
        from prompt_toolkit.keys import Keys

        kb.add(Keys.BracketedPaste, eager=True)(self._tui_handle_paste)

        kb.add('c-v')(self._tui_handle_ctrl_v)

        kb.add('escape', 'v')(self._tui_handle_alt_v)
        return kb

    def _tui_build_layout(self, kb):
        """Build the TUI widgets, Layout and Style; registers wrapper keybindings on ``kb``."""
        cli_ref = self
        input_area = self._tui_build_input_area()

        # Hint line above input: shown only for interactive prompts that need
        # extra instructions (sudo countdown, approval navigation, clarify).
        # The agent-running interrupt hint is now an inline placeholder above.

        spinner_widget = Window(
            content=FormattedTextControl(self._tui_spinner_text),
            height=self._tui_spinner_height,
            wrap_lines=True,
        )

        # Petdex mascot — right-aligned Kitty placeholder or half-block sprite
        # above the prompt. Collapses to height 0 when no pet is enabled.
        # The animation thread queues virtual Kitty frames; after_render
        # writes them out-of-band while prompt_toolkit owns the placeholder grid.
        self._pet_widget = Window(
            content=FormattedTextControl(self._pet_fragments),
            height=self._pet_widget_height,
            align=WindowAlign.RIGHT,
        )

        spacer = Window(
            content=FormattedTextControl(self._tui_hint_text),
            height=self._tui_hint_height,
        )

        # --- Clarify tool: dynamic display widget for questions + choices ---

        clarify_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_clarify_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._clarify_state is not None),
        )

        # --- Sudo password: display widget ---

        sudo_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_sudo_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._sudo_state is not None),
        )

        secret_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_secret_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._secret_state is not None),
        )

        # --- Dangerous command approval: display widget ---

        approval_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_approval_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._approval_state is not None),
        )

        slash_confirm_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_slash_confirm_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._slash_confirm_state is not None),
        )

        # --- /model picker: display widget ---

        model_picker_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_model_picker_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._model_picker_state is not None),
        )

        # --- Ctrl+P command palette: display widget ---

        command_palette_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_command_palette_display_fragments),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._command_palette_state is not None),
        )

        # Horizontal rules above and below the input.
        # On narrow/mobile terminals we keep the top separator for structure but
        # hide the bottom one to recover a full row for conversation content.
        input_rule_top = Window(
            char='─',
            height=lambda: cli_ref._tui_input_rule_height("top"),
            style='class:input-rule',
        )
        input_rule_bot = Window(
            char='─',
            height=lambda: cli_ref._tui_input_rule_height("bottom"),
            style='class:input-rule',
        )

        # Image attachment indicator — shows badges like [📎 Image #1] above input
        cli_ref = self

        image_bar = Window(
            content=FormattedTextControl(self._tui_image_bar_fragments),
            height=Condition(lambda: bool(cli_ref._attached_images)),
        )

        # Persistent voice mode status bar (visible only when voice mode is on)

        voice_status_bar = ConditionalContainer(
            Window(
                FormattedTextControl(self._tui_voice_status_fragments),
                height=1,
            ),
            filter=Condition(lambda: cli_ref._voice_mode),
        )

        status_bar = ConditionalContainer(
            Window(
                content=FormattedTextControl(lambda: cli_ref._get_status_bar_fragments()),
                height=1,
                # Prevent fragments that overflow the terminal width from
                # wrapping onto a second line, which causes the status bar to
                # appear duplicated (one full + one partial row) during long
                # sessions, especially on SSH where shutil.get_terminal_size
                # may return stale values.  _get_status_bar_fragments now reads
                # width from prompt_toolkit's own output object, so fragments
                # will always fit; wrap_lines=False is the belt-and-suspenders
                # guard against any future width mismatch.
                wrap_lines=False,
            ),
            filter=Condition(
                lambda: cli_ref._status_bar_visible
                and not getattr(cli_ref, "_status_bar_suppressed_after_resize", False)
            ),
        )

        # Stash browse panel — appears just above the status bar when the user
        # presses Ctrl+S on an empty composer with 2+ stashed drafts.

        self._stash_panel_widget = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_stash_panel_display_fragments),
                wrap_lines=False,
            ),
            filter=Condition(
                lambda: cli_ref._prompt_stash.panel_open
                and bool(len(cli_ref._prompt_stash))
            ),
        )

        # Allow wrapper CLIs to register extra keybindings.
        self._register_extra_tui_keybindings(kb, input_area=input_area)

        # Layout: interactive prompt widgets + ruled input at bottom.
        # The sudo, approval, and clarify widgets appear above the input when
        # the corresponding interactive prompt is active.
        completions_menu = CompletionsMenu(max_height=12, scroll_offset=1)

        layout = Layout(
            HSplit(
                self._build_tui_layout_children(
                    sudo_widget=sudo_widget,
                    secret_widget=secret_widget,
                    approval_widget=approval_widget,
                    slash_confirm_widget=slash_confirm_widget,
                    clarify_widget=clarify_widget,
                    model_picker_widget=model_picker_widget,
                    command_palette_widget=command_palette_widget,
                    spinner_widget=spinner_widget,
                    spacer=spacer,
                    status_bar=status_bar,
                    input_rule_top=input_rule_top,
                    image_bar=image_bar,
                    input_area=input_area,
                    input_rule_bot=input_rule_bot,
                    voice_status_bar=voice_status_bar,
                    completions_menu=completions_menu,
                )
            )
        )

        self._tui_set_base_style()
        style = PTStyle.from_dict(self._build_tui_style_dict())
        return (layout, style)

    def _tui_build_input_area(self):
        """Build the multi-line prompt TextArea with slash completion, paste-collapse tracking, and placeholder processors."""
        from cli import _estimate_tui_input_height, get_skill_bundles, get_skill_commands
        # Dynamic prompt: shows Hermes symbol when agent is working,
        # or answer prompt when clarify freetext mode is active.
        cli_ref = self

        def get_prompt():
            return cli_ref._get_tui_prompt_fragments()

        # Create the input area with multiline (Alt+Enter), autocomplete, and paste handling
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import ThreadedCompleter


        _completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            command_filter=cli_ref._command_available,
            skill_bundles_provider=lambda: get_skill_bundles(),
        )
        input_area = TextArea(
            height=Dimension(min=1, max=8, preferred=1),
            prompt=get_prompt,
            style='class:input-area',
            multiline=True,
            wrap_lines=True,
            read_only=Condition(lambda: bool(cli_ref._command_blocks_input)),
            history=FileHistory(str(self._history_file)),
            # complete_while_typing fires the completer on every keystroke. The
            # completer does blocking work — fuzzy @-file indexing shells out to
            # rg/fd (up to a 2s timeout) and path completion hits os.listdir/stat
            # — so running it inline would stall the render loop on each key (very
            # noticeable on WSL2/slow filesystems). ThreadedCompleter moves it off
            # the UI event loop, keeping typing responsive.
            completer=ThreadedCompleter(_completer),
            complete_while_typing=True,
            auto_suggest=SlashCommandAutoSuggest(
                history_suggest=AutoSuggestFromHistory(),
                completer=_completer,
            ),
        )
        # Keep prompt_toolkit on its simple tempfile path. Setting
        # buffer.tempfile = "prompt.md" triggers its complex-tempfile branch,
        # which tries to mkdir() the mkdtemp() directory again and raises
        # EEXIST. The suffix keeps markdown highlighting without that bug.
        input_area.buffer.tempfile_suffix = '.md'

        # Dynamic height: accounts for both explicit newlines AND visual
        # wrapping of long lines so the input area always fits its content.
        def _input_height():
            try:
                from prompt_toolkit.application import get_app

                doc = input_area.buffer.document
                try:
                    terminal_columns = get_app().output.get_size().columns
                except Exception:
                    terminal_columns = shutil.get_terminal_size((80, 24)).columns
                return _estimate_tui_input_height(
                    doc.lines,
                    self._get_tui_prompt_text(),
                    terminal_columns,
                )
            except Exception:
                return 1

        input_area.window.height = _input_height

        # Paste collapsing: detect large pastes and save to temp file
        self._tui_paste_counter = [0]
        self._tui_prev_text_len = [0]
        self._tui_prev_newline_count = [0]
        self._tui_paste_just_collapsed = [False]
        self._skip_paste_collapse = False

        input_area.buffer.on_text_changed += self._tui_on_text_changed

        # --- Input processors for password masking and inline placeholder ---

        # Mask input with '*' when the sudo password prompt is active
        input_area.control.input_processors.append(
            ConditionalProcessor(
                PasswordProcessor(),
                filter=Condition(
                    lambda: bool(cli_ref._sudo_state) or bool(cli_ref._secret_state)
                ),
            )
        )

        class _PlaceholderProcessor(Processor):
            """Render grayed-out placeholder text inside the input when empty."""
            def __init__(self, get_text):
                self._get_text = get_text

            def apply_transformation(self, ti):
                if not ti.document.text and ti.lineno == 0:
                    text = self._get_text()
                    if text:
                        # Append after existing fragments (preserves the ❯ prompt)
                        return Transformation(fragments=ti.fragments + [('class:placeholder', text)])
                return Transformation(fragments=ti.fragments)

        input_area.control.input_processors.append(_PlaceholderProcessor(self._tui_placeholder_text))
        return input_area

    def _tui_set_base_style(self):
        """Populate ``self._tui_style_base`` (skin-aware defaults the style dict is built from)."""
        # Style for the application
        self._tui_style_base = {
            # Input area / prompt: empty style strings inherit the
            # terminal's default foreground/background, so the typed
            # text is readable in both light and dark Terminal.app
            # color schemes.  (Hardcoding a near-white #FFF8DC made
            # input invisible on light backgrounds.)
            'input-area': '',
            'placeholder': '#888888 italic',
            'prompt': '',
            'prompt-working': '#888888 italic',
            'hint': '#888888 italic',
            'status-bar': 'bg:#1a1a2e #C0C0C0',
            'status-bar-strong': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-dim': 'bg:#1a1a2e #8B8682',
            'status-bar-good': 'bg:#1a1a2e #8FBC8F bold',
            'status-bar-warn': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-bad': 'bg:#1a1a2e #FF8C00 bold',
            'status-bar-critical': 'bg:#1a1a2e #FF6B6B bold',
            'status-bar-yolo': 'bg:#1a1a2e #FF4444 bold',
            'status-bar-session-title': 'bg:#FFD700 #1a1a2e bold',
            # Bronze horizontal rules around the input area
            'input-rule': '#CD7F32',
            # Clipboard image attachment badges
            'image-badge': '#87CEEB bold',
            'completion-menu': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion.current': 'bg:#333355 #FFD700',
            'completion-menu.meta.completion': 'bg:#1a1a2e #888888',
            'completion-menu.meta.completion.current': 'bg:#333355 #FFBF00',
            # Clarify question panel
            'clarify-border': '#CD7F32',
            'clarify-title': '#FFD700 bold',
            'clarify-question': '#FFF8DC bold',
            'clarify-choice': '#AAAAAA',
            'clarify-selected': '#FFD700 bold',
            'clarify-active-other': '#FFD700 italic',
            'clarify-answer': '#98FB98',
            'clarify-countdown': '#CD7F32',
            # Sudo password panel
            'sudo-prompt': '#FF6B6B bold',
            'sudo-border': '#CD7F32',
            'sudo-title': '#FF6B6B bold',
            'sudo-text': '#FFF8DC',
            # Dangerous command approval panel
            'approval-border': '#CD7F32',
            'approval-title': '#FF8C00 bold',
            'approval-desc': '#FFF8DC bold',
            'approval-cmd': '#AAAAAA italic',
            'approval-choice': '#AAAAAA',
            'approval-selected': '#FFD700 bold',
            # Voice mode
            'voice-prompt': '#87CEEB',
            'voice-recording': '#FF4444 bold',
            'voice-processing': '#FFA500 italic',
            'voice-status': 'bg:#1a1a2e #87CEEB',
            'voice-status-recording': 'bg:#1a1a2e #FF4444 bold',
        }
