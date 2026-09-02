"""Status bar, spinner, turn-summary, pet pane, and prompt-stash rendering for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import errno
import shutil
import threading
import time

from agent.pet import render as pet_render
from hermes_cli.banner import _format_context_length
from typing import Any, Dict, Optional


class CLIStatusBarMixin:
    """Status bar, spinner, turn-summary, pet pane, and prompt-stash rendering for the interactive CLI"""

    def _status_bar_context_style(self, percent_used: Optional[int]) -> str:
        if percent_used is None:
            return "class:status-bar-dim"
        if percent_used >= 95:
            return "class:status-bar-critical"
        if percent_used > 80:
            return "class:status-bar-bad"
        if percent_used >= 50:
            return "class:status-bar-warn"
        return "class:status-bar-good"

    def _cache_hit_rate(self, snapshot: dict, precision: int = 1) -> "tuple[float, str] | None":
        """Return (cache_pct, formatted_label) or None if no cache data.

        Centralises the cache-hit-rate computation so both the plain-text
        status bar and the prompt-toolkit fragment path share one formula.
        Prefers the baseline-delta percentage computed in
        ``_get_status_bar_snapshot`` (resets on model switch / compression,
        so it reflects the *current* cache regime); falls back to the
        session-lifetime ratio when no delta is available.
        """
        delta_pct = snapshot.get("cache_hit_pct")
        if delta_pct is not None:
            return float(delta_pct), f"◎ {float(delta_pct):.{precision}f}%"
        cache_read = snapshot.get("session_cache_read_tokens", 0)
        prompt_total = snapshot.get("session_prompt_tokens", 0)
        if cache_read > 0 and prompt_total > 0:
            cache_pct = cache_read / prompt_total * 100
            return cache_pct, f"◎ {cache_pct:.{precision}f}%"
        return None

    def _cache_hit_rate_style(self, cache_pct: float) -> str:
        """Style for cache hit rate — higher is better (opposite of context %)."""
        if cache_pct >= 70:
            return "class:status-bar-good"
        if cache_pct >= 40:
            return "class:status-bar-warn"
        return "class:status-bar-bad"

    @staticmethod
    def _battery_status_style(category: str) -> str:
        """Map a battery colour category to a status-bar style class."""
        return {
            "good": "class:status-bar-good",
            "warn": "class:status-bar-warn",
            "bad": "class:status-bar-bad",
            "critical": "class:status-bar-critical",
        }.get(category, "class:status-bar-dim")

    def _handle_battery_command(self, cmd_original: str) -> None:
        """Toggle the status-bar battery read-out.

        ``/battery`` toggles, ``/battery on|off`` sets explicitly, and
        ``/battery status`` reports the current setting plus a live reading.
        The choice is persisted to ``display.battery`` so it survives restarts.
        """
        from cli import save_config_value
        parts = (cmd_original or "").split()
        arg = parts[1].strip().lower() if len(parts) > 1 else ""

        try:
            from agent.battery import format_battery, read_battery
            reading = read_battery(use_cache=False)
        except Exception:
            reading = None

        if arg in ("status", "show"):
            state = "on" if self._battery_visible else "off"
            if reading is not None and reading.available:
                self._console_print(
                    f"  Battery indicator {state} — currently {format_battery(reading)}"
                )
            elif reading is not None:
                self._console_print(
                    f"  Battery indicator {state} — no battery detected on this machine"
                )
            else:
                self._console_print(f"  Battery indicator {state}")
            return

        if arg in ("on", "true", "yes"):
            target = True
        elif arg in ("off", "false", "no"):
            target = False
        elif arg in ("", "toggle"):
            target = not self._battery_visible
        else:
            self._console_print("  Usage: /battery [on|off|status]")
            return

        self._battery_visible = target
        save_config_value("display.battery", target)

        if target:
            if reading is not None and not reading.available:
                self._console_print(
                    "  Battery indicator on — no battery detected, so nothing will show here"
                )
            elif reading is not None and reading.available:
                self._console_print(
                    f"  Battery indicator on — {format_battery(reading)}"
                )
            else:
                self._console_print("  Battery indicator on")
        else:
            self._console_print("  Battery indicator off")

    @staticmethod
    def _compression_count_style(count: int) -> str:
        """Return a style class reflecting context compression pressure."""
        if count >= 10:
            return "class:status-bar-bad"
        if count >= 5:
            return "class:status-bar-warn"
        return "class:status-bar-dim"

    def _build_context_bar(self, percent_used: Optional[int], width: int = 10) -> str:
        safe_percent = max(0, min(100, percent_used or 0))
        filled = round((safe_percent / 100) * width)
        return f"[{('█' * filled) + ('░' * max(0, width - filled))}]"

    @staticmethod
    def _format_prompt_elapsed(prompt_start_time: Optional[float], prompt_duration: float, live: bool = False) -> str:
        """Format per-prompt elapsed time for the status bar.

        Always returns a string — shows 0s on fresh start before first turn.
        Keeps seconds visible at all scales so it increments smoothly:
            59s → 1m → 1m 1s → ... → 1m 59s → 2m → 2m 1s → ...
            59m 59s → 1h → 1h 0m 1s → ...
            23h 59m 59s → 1d → 1d 0h 1m → ...

        Emoji prefix: ⏱ when turn is live, ⏲ when frozen or fresh start.
        Uses width-1 (no variation selector) glyphs so the status bar stays
        aligned in monospace terminals.
        """
        if prompt_start_time is None and prompt_duration == 0.0:
            return "⏲ 0s"
        elapsed = time.time() - prompt_start_time if prompt_start_time is not None else prompt_duration
        elapsed = max(0.0, elapsed)

        days = int(elapsed // 86400)
        remaining = elapsed % 86400
        hours = int(remaining // 3600)
        remaining = remaining % 3600
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)

        if days > 0:
            time_str = f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s" if seconds else f"{hours}h {minutes}m"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
        else:
            time_str = f"{int(elapsed)}s"

        emoji = "⏱" if live else "⏲"
        return f"{emoji} {time_str}"

    @staticmethod
    def _format_idle_since(last_finished_at: Optional[float], turn_live: bool) -> str:
        """Format time since the last final agent response for the status bar.

        Returns an empty string while a turn is live (the per-prompt elapsed
        timer covers that case) or before the first turn has completed.
        Compact read-out: ``✓ 42s`` / ``✓ 3m`` / ``✓ 1h 12m``.
        """
        from cli import format_duration_compact
        if turn_live or last_finished_at is None:
            return ""
        idle = max(0.0, time.time() - last_finished_at)
        return f"✓ {format_duration_compact(idle)}"

    def _get_status_bar_snapshot(self) -> Dict[str, Any]:
        # Prefer the agent's model name — it updates on fallback.
        # self.model reflects the originally configured model and never
        # changes mid-session, so the TUI would show a stale name after
        # _try_activate_fallback() switches provider/model.
        from cli import _reverse_alias_for_display, datetime, format_duration_compact
        agent = getattr(self, "agent", None)
        model_name = (getattr(agent, "model", None) or self.model or "unknown")
        # Friendly display: prefer reverse-alias from config.yaml ``model_aliases:``
        # before slash/length truncation. This turns long Palantir RIDs like
        # ``ri.language-model-service..language-model.anthropic-claude-4-7-opus``
        # into the user's chosen short name (e.g. ``opus-4.7``) in the status bar.
        model_short = _reverse_alias_for_display(model_name)
        if model_short == model_name:
            model_short = model_name.split("/")[-1] if "/" in model_name else model_name
            # Strip Palantir RID prefixes via the shared display formatter so
            # this site and ``ModelSwitchResult`` confirmation can't drift.
            from hermes_cli.model_switch import format_model_for_display
            model_short = format_model_for_display(model_short)
        if model_short.endswith(".gguf"):
            model_short = model_short[:-5]
        if len(model_short) > 26:
            model_short = f"{model_short[:23]}..."

        elapsed_seconds = max(0.0, (datetime.now() - self.session_start).total_seconds())
        snapshot = {
            "model_name": model_name,
            "model_short": model_short,
            "duration": format_duration_compact(elapsed_seconds),
            "session_title": self._get_status_bar_session_title(),
            "prompt_elapsed": self._format_prompt_elapsed(
                getattr(self, "_prompt_start_time", None),
                getattr(self, "_prompt_duration", 0.0),
                live=getattr(self, "_prompt_start_time", None) is not None,
            ),
            "idle_since": self._format_idle_since(
                getattr(self, "_last_turn_finished_at", None),
                turn_live=getattr(self, "_prompt_start_time", None) is not None,
            ),
            "context_tokens": 0,
            "context_length": None,
            "context_percent": None,
            "session_input_tokens": 0,
            "session_output_tokens": 0,
            "session_cache_read_tokens": 0,
            "session_cache_write_tokens": 0,
            "session_prompt_tokens": 0,
            "session_completion_tokens": 0,
            "session_total_tokens": 0,
            "session_api_calls": 0,
            "compressions": 0,
            "active_background_tasks": 0,
            "active_background_processes": 0,
            "active_background_subagents": 0,
            "battery_label": "",
            "battery_category": "dim",
            # Focus view badge (/focus). Persistent indicator so the reduced
            # output mode is never invisible. Display-only.
            "focus_label": "",
        }

        try:
            from hermes_cli.focus_view import focus_statusbar_segment

            snapshot["focus_label"] = focus_statusbar_segment(
                bool(getattr(self, "_focus_view_enabled", False))
            )
        except Exception:
            pass

        # Battery read-out (first status-bar element when enabled). Reads are
        # memoised for a few seconds inside agent.battery, so polling it on
        # every status-bar repaint is cheap.
        if getattr(self, "_battery_visible", False):
            try:
                from agent.battery import (
                    battery_category,
                    format_battery,
                    read_battery,
                )

                _batt = read_battery()
                snapshot["battery_label"] = format_battery(_batt)
                snapshot["battery_category"] = battery_category(_batt)
            except Exception:
                pass

        # Count live /bg tasks. The dict entry is removed in the
        # task thread's finally block, so len() reflects truly-running tasks.
        # len() on a CPython dict is atomic; safe to read without a lock.
        try:
            bg_tasks = getattr(self, "_background_tasks", None)
            if bg_tasks:
                snapshot["active_background_tasks"] = len(bg_tasks)
        except Exception:
            pass

        # Count live background terminal processes (terminal tool background
        # sessions tracked by tools.process_registry). Cheap O(1) read.
        try:
            from tools.process_registry import process_registry
            snapshot["active_background_processes"] = process_registry.count_running()
        except Exception:
            pass

        # Count live background/async subagents (delegate_task batches and
        # background single delegations tracked by tools.async_delegation).
        # active_count() iterates an in-memory records dict under a lock —
        # cheap and only counts records still in the "running" state.
        try:
            from tools.async_delegation import active_count as _async_active_count
            snapshot["active_background_subagents"] = _async_active_count()
        except Exception:
            pass

        # Standing /goal state (Ralph loop). GoalManager is cached on self and
        # keeps its state in memory, so this is a cheap attribute read — no DB
        # hit per repaint. Only an *active* goal earns a segment; paused/done
        # goals stay out of the bar (matching the desktop's active-first row).
        snapshot["goal_active"] = False
        snapshot["goal_turns_used"] = 0
        snapshot["goal_max_turns"] = 0
        try:
            goal_mgr = self._get_goal_manager()
            if goal_mgr is not None and goal_mgr.is_active():
                goal_state = goal_mgr.state
                snapshot["goal_active"] = True
                snapshot["goal_turns_used"] = int(getattr(goal_state, "turns_used", 0) or 0)
                snapshot["goal_max_turns"] = int(getattr(goal_state, "max_turns", 0) or 0)
        except Exception:
            pass


        if not agent:
            return snapshot

        snapshot["session_input_tokens"] = getattr(agent, "session_input_tokens", 0) or 0
        snapshot["session_output_tokens"] = getattr(agent, "session_output_tokens", 0) or 0
        snapshot["session_cache_read_tokens"] = getattr(agent, "session_cache_read_tokens", 0) or 0
        snapshot["session_cache_write_tokens"] = getattr(agent, "session_cache_write_tokens", 0) or 0
        snapshot["session_prompt_tokens"] = getattr(agent, "session_prompt_tokens", 0) or 0
        snapshot["session_completion_tokens"] = getattr(agent, "session_completion_tokens", 0) or 0
        snapshot["session_total_tokens"] = getattr(agent, "session_total_tokens", 0) or 0
        snapshot["session_api_calls"] = getattr(agent, "session_api_calls", 0) or 0

        compressor = getattr(agent, "context_compressor", None)
        if compressor:
            # last_prompt_tokens is parked at the -1 sentinel right after a
            # compression, until the next real API call reports a prompt count
            # (awaiting_real_usage_after_compression). The status bar must not
            # render that sentinel verbatim — it produced "-1/200K" / "-1%".
            # Clamp it to 0 so the one transitional turn reads as empty context.
            context_tokens = getattr(compressor, "last_prompt_tokens", 0) or 0
            if context_tokens < 0:
                context_tokens = 0
            # Durable-transcript view: on reasoning models a long tool loop
            # replays the current turn's thinking + scaffolding on every
            # request, so the LAST request's prompt_tokens can exceed the
            # durable transcript by hundreds of K — all of which evaporates
            # at the turn boundary. Rendering that raw figure makes the bar
            # sawtooth (e.g. 850K mid-turn -> 600K next turn) and reads as a
            # broken compaction. Anchor the display on the turn's FIRST
            # response (minimal replay) plus a delta estimate of messages
            # appended since, excluding stale thinking. Display-only: the
            # compression trigger keeps using real last-request usage.
            try:
                from agent.model_metadata import anchored_context_tokens

                _msgs = getattr(agent, "_session_messages", None)
                _anchored = anchored_context_tokens(
                    _msgs if isinstance(_msgs, list) else [],
                    getattr(agent, "_turn_base_usage_anchor", None),
                    charge_stale_thinking=False,
                )
                if _anchored is not None and _anchored > 0:
                    context_tokens = _anchored
            except Exception:
                pass
            context_length = getattr(compressor, "context_length", 0) or 0
            if context_length < 0:
                context_length = 0
            snapshot["context_tokens"] = context_tokens
            snapshot["context_length"] = context_length or None
            snapshot["compressions"] = getattr(compressor, "compression_count", 0) or 0
            if context_length:
                snapshot["context_percent"] = max(0, min(100, round((context_tokens / context_length) * 100)))

        # -- Cache-hit ratio (delta since last reset) --
        # Reset baseline on model switch and on compression — both invalidate
        # the prompt cache. Formula verified against live logs:
        #   hit = cache_read / prompt_tokens  (prompt = input+cache_read+cache_write)
        #   see agent/conversation_loop.py:4314  cache=read/prompt (87%)
        #   and CanonicalUsage.prompt_tokens = input+read+write
        try:
            base_model = getattr(self, "_cache_hit_baseline_model", None)
            base_prompt = int(getattr(self, "_cache_hit_baseline_prompt", 0) or 0)
            base_read = int(getattr(self, "_cache_hit_baseline_read", 0) or 0)
            base_comps = int(getattr(self, "_cache_hit_baseline_compressions", 0) or 0)
            cur_model = snapshot.get("model_name") or model_name
            cur_comps = int(snapshot.get("compressions", 0) or 0)
            cur_prompt = int(snapshot.get("session_prompt_tokens", 0) or 0)
            cur_read = int(snapshot.get("session_cache_read_tokens", 0) or 0)
            if base_model is None:
                self._cache_hit_baseline_model = cur_model
                self._cache_hit_baseline_compressions = cur_comps
                base_model = cur_model
                base_comps = cur_comps
            if cur_model != base_model:
                self._cache_hit_baseline_model = cur_model
                self._cache_hit_baseline_prompt = cur_prompt
                self._cache_hit_baseline_read = cur_read
                self._cache_hit_baseline_compressions = cur_comps
                base_prompt = cur_prompt
                base_read = cur_read
                base_comps = cur_comps
            if cur_comps != base_comps:
                self._cache_hit_baseline_compressions = cur_comps
                self._cache_hit_baseline_prompt = cur_prompt
                self._cache_hit_baseline_read = cur_read
                base_prompt = cur_prompt
                base_read = cur_read
            delta_prompt = cur_prompt - base_prompt
            delta_read = cur_read - base_read
            # A zero-read regime hides the segment entirely (no cache data
            # is not the same as a 0% hit worth alarming about), and the pct
            # stays a float so renderers control their own precision.
            if delta_prompt > 0 and delta_read > 0:
                pct = max(0.0, min(100.0, (delta_read / delta_prompt) * 100))
                snapshot["cache_hit_pct"] = pct
                snapshot["cache_hit_label"] = f"{pct:.0f}%"
            elif cur_prompt > 0 and cur_read > 0 and base_prompt == 0 and base_read == 0:
                pct = max(0.0, min(100.0, (cur_read / cur_prompt) * 100))
                snapshot["cache_hit_pct"] = pct
                snapshot["cache_hit_label"] = f"{pct:.0f}%"
            else:
                snapshot["cache_hit_pct"] = None
                snapshot["cache_hit_label"] = ""
        except Exception:
            snapshot["cache_hit_pct"] = None
            snapshot["cache_hit_label"] = ""

        # -- Rolling avg latency / velocity (last 10 calls) --
        # Reads the deque maintained in agent/conversation_loop.py (and
        # agent_init). Codex app-server has no latency, so it stays hidden there.
        try:
            agent_obj = getattr(self, "agent", None)
            lhist = list(getattr(agent_obj, "_api_latency_history", []) or []) if agent_obj else []
            ohist = list(getattr(agent_obj, "_api_output_history", []) or []) if agent_obj else []
            # Keep the two histories aligned (they are appended together).
            n = min(len(lhist), len(ohist))
            if n:
                lhist = lhist[-n:]
                ohist = ohist[-n:]
                # Simple mean for latency; sum/sum for velocity (true throughput, not mean of ratios).
                avg_lat = sum(lhist) / len(lhist) if lhist else None
                total_out = sum(ohist)
                total_lat = sum(lhist)
                avg_vel = (total_out / total_lat) if total_lat > 0 else None
                # Guard against NaN / inf from weird provider timings (e.g. -0.8s in logs).
                if avg_lat is not None and (avg_lat != avg_lat or avg_lat < 0 or avg_lat > 1e6):
                    avg_lat = None
                if avg_vel is not None and (avg_vel != avg_vel or avg_vel < 0 or avg_vel > 1e6):
                    avg_vel = None
                snapshot["avg_latency"] = float(avg_lat) if avg_lat is not None else None
                snapshot["avg_latency_label"] = f"{avg_lat:.1f}s" if avg_lat is not None else ""
                snapshot["avg_velocity"] = float(avg_vel) if avg_vel is not None else None
                snapshot["avg_velocity_label"] = f"{avg_vel:.0f} t/s" if avg_vel is not None else ""
            else:
                snapshot["avg_latency"] = None
                snapshot["avg_latency_label"] = ""
                snapshot["avg_velocity"] = None
                snapshot["avg_velocity_label"] = ""
        except Exception:
            snapshot["avg_latency"] = None
            snapshot["avg_latency_label"] = ""
            snapshot["avg_velocity"] = None
            snapshot["avg_velocity_label"] = ""

        return snapshot

    def _get_status_bar_session_title(self) -> str:
        """Return the current title without polling state.db on every repaint."""
        pending = str(getattr(self, "_pending_title", None) or "").strip()
        session_id = str(getattr(self, "session_id", "") or "")
        if pending:
            self._status_bar_title_session_id = session_id
            self._status_bar_title_cache = pending
            self._status_bar_title_checked_at = time.monotonic()
            return pending

        now = time.monotonic()
        cached_session_id = getattr(self, "_status_bar_title_session_id", None)
        checked_at = float(getattr(self, "_status_bar_title_checked_at", 0.0) or 0.0)
        if cached_session_id == session_id and now - checked_at < 1.5:
            return str(getattr(self, "_status_bar_title_cache", "") or "")

        title = ""
        db = getattr(self, "_session_db", None)
        if db is not None and session_id:
            try:
                title = str(db.get_session_title(session_id) or "").strip()
            except Exception:
                title = ""
        self._status_bar_title_session_id = session_id
        self._status_bar_title_cache = title
        self._status_bar_title_checked_at = now
        return title

    @staticmethod
    def _status_bar_display_width(text: str) -> int:
        """Return terminal cell width for status-bar text.

        len() is not enough for prompt_toolkit layout decisions because some
        glyphs can render wider than one Python codepoint. Keeping the status
        bar within the real display width prevents it from wrapping onto a
        second line and leaving behind duplicate rows.
        """
        try:
            from prompt_toolkit.utils import get_cwidth
            return get_cwidth(text or "")
        except Exception:
            return len(text or "")

    @classmethod
    def _trim_status_bar_text(cls, text: str, max_width: int) -> str:
        """Trim status-bar text to a single terminal row."""
        if max_width <= 0:
            return ""
        try:
            from prompt_toolkit.utils import get_cwidth
        except Exception:
            get_cwidth = None

        if cls._status_bar_display_width(text) <= max_width:
            return text

        ellipsis = "..."
        ellipsis_width = cls._status_bar_display_width(ellipsis)
        if max_width <= ellipsis_width:
            return ellipsis[:max_width]

        out = []
        width = 0
        for ch in text:
            ch_width = get_cwidth(ch) if get_cwidth else len(ch)
            if width + ch_width + ellipsis_width > max_width:
                break
            out.append(ch)
            width += ch_width
        return "".join(out).rstrip() + ellipsis

    @classmethod
    def _right_align_status_title(cls, text: str, title: str, width: int) -> str:
        """Pin a bounded session-title badge to the far-right status-bar edge."""
        title = str(title or "").strip()
        if not title or width < 24:
            return cls._trim_status_bar_text(text, width)

        title_width = max(6, min(30, width // 3))
        badge = f" {cls._trim_status_bar_text(title, title_width - 2)} "
        suffix = f" ─{badge}"
        left_width = max(0, width - cls._status_bar_display_width(suffix))
        left = cls._trim_status_bar_text(text.rstrip(), left_width)
        padding = " " * max(0, left_width - cls._status_bar_display_width(left))
        return f"{left}{padding}{suffix}"

    @classmethod
    def _right_align_status_title_fragments(cls, frags, title: str, width: int):
        """Styled counterpart to :meth:`_right_align_status_title`."""
        title = str(title or "").strip()
        if not title or width < 24:
            return frags

        title_width = max(6, min(30, width // 3))
        badge = f" {cls._trim_status_bar_text(title, title_width - 2)} "
        suffix_width = cls._status_bar_display_width(" ─") + cls._status_bar_display_width(badge)
        left_width = max(0, width - suffix_width)
        trimmed = []
        used = 0
        for style, value in frags:
            remaining = left_width - used
            if remaining <= 0:
                break
            value_width = cls._status_bar_display_width(value)
            if value_width <= remaining:
                trimmed.append((style, value))
                used += value_width
                continue
            clipped = cls._trim_status_bar_text(value, remaining)
            if clipped:
                trimmed.append((style, clipped))
                used += cls._status_bar_display_width(clipped)
            break

        if used < left_width:
            trimmed.append(("class:status-bar-dim", " " * (left_width - used)))
        trimmed.extend([
            ("class:status-bar-dim", " ─"),
            ("class:status-bar-session-title", badge),
        ])
        return trimmed

    @staticmethod
    def _get_tui_terminal_width(default: tuple[int, int] = (80, 24)) -> int:
        """Return the live prompt_toolkit width, falling back to ``shutil``.

        The TUI layout can be narrower than ``shutil.get_terminal_size()`` reports,
        especially on Termux/mobile shells, so prefer prompt_toolkit's width whenever
        an app is active.
        """
        try:
            from prompt_toolkit.application import get_app
            return get_app().output.get_size().columns
        except Exception:
            return shutil.get_terminal_size(default).columns

    def _use_minimal_tui_chrome(self, width: Optional[int] = None) -> bool:
        """Hide low-value chrome on narrow/mobile terminals to preserve rows."""
        if width is None:
            width = self._get_tui_terminal_width()
        return width < 64

    @staticmethod
    def _scrollback_box_width(width: Optional[int] = None) -> int:
        """Return the full viewport width for printed scrollback box rules.

        Previously this clamped to ``max(32, min(width, 56))`` as a defense
        against terminal-emulator reflow on column-shrink (#25975, salvaging
        #24403).  That clamp made response/reasoning borders look stubby on
        any modern wide terminal.  We now trust the prompt_toolkit
        ``_output_screen_diff`` monkey-patch landed in #26137 (salvaging
        #25981) to keep chrome out of scrollback in the first place, and
        accept that an aggressive column-shrink may visually reflow already
        printed Panel borders — that's a cosmetic artifact of stamped
        scrollback history, not a live-render bug.

        A small floor (32 cols) is kept so the box still renders on tiny
        terminals without negative ``'─' * (w - 2)`` math.
        """
        if width is None:
            try:
                width = shutil.get_terminal_size((80, 24)).columns
            except Exception:
                width = 80
        return max(32, int(width or 80))

    def _agent_spacer_height(self, width: Optional[int] = None) -> int:
        """Return the spacer height shown above the status bar while the agent runs."""
        if not getattr(self, "_agent_running", False):
            return 0
        return 0 if self._use_minimal_tui_chrome(width=width) else 1

    def _spinner_widget_height(self, width: Optional[int] = None) -> int:
        """Return the visible height for the spinner/status text line above the status bar."""
        spinner_line = self._render_spinner_text()
        if not spinner_line:
            return 0
        if self._use_minimal_tui_chrome(width=width):
            return 0
        width = width or self._get_tui_terminal_width()
        if width and width > 10:
            import math
            text_width = self._status_bar_display_width(spinner_line)
            return max(1, math.ceil(text_width / width))
        return 1

    def _render_spinner_text(self) -> str:
        """Return the live spinner/status text exactly as rendered in the TUI."""
        txt = getattr(self, "_spinner_text", "")
        if not txt:
            return ""
        flow = self._spinner_token_flow()
        t0 = getattr(self, "_tool_start_time", 0) or 0
        if t0 > 0:
            elapsed = time.monotonic() - t0
            if elapsed >= 60:
                _m, _s = int(elapsed // 60), int(elapsed % 60)
                # Fixed-width timer to avoid status-line wrap jitter while
                # scrolling/repainting (e.g. 01m05s, 12m09s).
                elapsed_str = f"{_m:02d}m{_s:02d}s"
            else:
                # Keep width stable before the 60s rollover as well.
                elapsed_str = f"{elapsed:5.1f}s"
            if flow:
                return f"  {txt}  ({elapsed_str} · {flow})"
            return f"  {txt}  ({elapsed_str})"
        if flow:
            return f"  {txt}  ({flow})"
        return f"  {txt}"

    def _spinner_token_flow(self) -> str:
        """Cumulative output tokens for the running turn, for the spinner."""
        if not getattr(self, "_spinner_token_flow_enabled", False):
            return ""
        if not getattr(self, "_agent_running", False):
            return ""
        agent = getattr(self, "agent", None)
        if agent is None:
            return ""
        try:
            from agent.turn_summary import format_token_flow

            produced = (getattr(agent, "session_output_tokens", 0) or 0) - (
                getattr(self, "_turn_token_baseline", 0) or 0
            )
            return format_token_flow(produced)
        except Exception:
            return ""

    def _turn_summary_is_active(self) -> bool:
        """Whether the per-turn summary line should render for this surface.

        Gated off for: the config key, quiet/tool-progress-off mode, and any
        non-interactive path (single query, ``-Q``, gateway/messaging) — those
        surfaces either want machine-readable output or carry their own footer.
        """
        if not getattr(self, "_turn_summary_enabled", False):
            return False
        if getattr(self, "tool_progress_mode", "all") == "off":
            return False
        agent = getattr(self, "agent", None)
        if agent is not None and getattr(agent, "quiet_mode", False):
            return False
        return bool(getattr(self, "_interactive_turn", False))

    def _turn_summary_begin(self) -> None:
        """Start per-turn accounting for the turn that is about to run."""
        try:
            from agent.turn_summary import TurnSummaryCollector

            collector = getattr(self, "_turn_summary_collector", None)
            if collector is None:
                collector = TurnSummaryCollector()
                self._turn_summary_collector = collector
            collector.begin()
            self._turn_summary_start = time.monotonic()
            agent = getattr(self, "agent", None)
            self._turn_token_baseline = (
                getattr(agent, "session_output_tokens", 0) or 0
            ) if agent is not None else 0
        except Exception:
            self._turn_summary_collector = None

    def _turn_summary_record(self, function_name, result, is_error: bool) -> None:
        """Feed one completed tool call into the active tally."""
        collector = getattr(self, "_turn_summary_collector", None)
        if collector is None:
            return
        try:
            collector.record_tool(function_name, result=result, is_error=bool(is_error))
        except Exception:
            pass

    def _turn_summary_emit(self) -> None:
        """Print the post-turn accounting line, when enabled for this surface."""
        from cli import _DIM, _RST, _cprint, logger
        collector = getattr(self, "_turn_summary_collector", None)
        if collector is None or not self._turn_summary_is_active():
            return
        try:
            started = getattr(self, "_turn_summary_start", 0.0) or 0.0
            elapsed = max(0.0, time.monotonic() - started) if started else 0.0
            line = collector.render(elapsed)
            if line:
                _cprint(f"  {_DIM}{line}{_RST}")
        except Exception:
            logger.debug("Turn summary render failed", exc_info=True)

    def _pet_clear_runtime(self) -> None:
        """Drop renderer + queued Kitty state. Caller holds ``_pet_lock``."""
        self._pet_enabled = False
        self._pet_renderer = None
        self._pet_frames_cache.clear()
        self._pet_kitty_cache.clear()
        self._pet_kitty_pending = ""
        self._pet_kitty_image_id = 0

    def _pet_resolve_config(self) -> None:
        """(Re)resolve the active pet from config — picks up live enable/disable/

        switch made via ``/pet`` or ``hermes pets`` without a restart, mirroring
        the TUI's steady poll. Cheap and fail-open: any problem disables the pet.
        """
        try:
            from agent.pet import constants, store
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}

            from utils import is_truthy_value

            enabled = is_truthy_value(pet_cfg.get("enabled"), default=False)
            slug = str(pet_cfg.get("slug", "") or "")
            scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
            cols = constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))
            configured_mode = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            # Placeholders only on kitty/Ghostty. WezTerm speaks kitty APC but
            # not U+10EEEE — detect_terminal_graphics() still returns kitty
            # there, which is why this gate is narrower.
            use_kitty = configured_mode in ("", "auto", "kitty") and pet_render.supports_kitty_placeholders()
            renderer_mode = "kitty" if use_kitty else "unicode"

            if not enabled or configured_mode == "off":
                with self._pet_lock:
                    self._pet_clear_runtime()
                return

            pet = store.resolve_active_pet(slug)
            if pet is None or not pet.exists:
                with self._pet_lock:
                    self._pet_clear_runtime()
                return

            with self._pet_lock:
                # Rebuild only when the resolved pet, mode, or geometry changes.
                if (
                    self._pet_renderer is None
                    or self._pet_slug != pet.slug
                    or self._pet_cols != cols
                    or self._pet_scale != scale
                    or self._pet_renderer.mode != renderer_mode
                ):
                    self._pet_renderer = pet_render.PetRenderer(
                        str(pet.spritesheet), mode=renderer_mode, scale=scale, unicode_cols=cols
                    )
                    self._pet_slug = pet.slug
                    self._pet_cols = cols
                    self._pet_scale = scale
                    self._pet_frames_cache.clear()
                    self._pet_kitty_cache.clear()
                    self._pet_kitty_pending = ""
                    self._pet_kitty_image_id = pet_render.kitty_image_id(pet.slug)
                    self._pet_frame_idx = 0
                self._pet_enabled = True
        except Exception:
            with self._pet_lock:
                self._pet_clear_runtime()

    def _pet_flash(self, state: str, secs: float = 1.6) -> None:
        """Briefly force a transient reaction (wave/jump/failed) before resting."""
        self._pet_event = state
        self._pet_event_until = time.monotonic() + secs

    def _on_reaction(self, kind: str) -> None:
        """User affection (ily / <3 / good bot), core-detected — the pet's share
        of the vibe signal that plays hearts on the TUI/desktop. Flash a celebrate."""
        if kind == "vibe":
            self._pet_flash("jump")

    def _pet_react_turn_end(self) -> None:
        """Flash the end-of-turn beat: failed on error, jump on a finished plan, else wave."""
        if not self._pet_enabled:
            return
        from agent.pet.state import todos_all_done

        if self._pet_turn_error:
            self._pet_flash("failed")
            return
        try:
            store = getattr(self.agent, "_todo_store", None)
            done = todos_all_done(store.read()) if store else False
        except Exception:
            done = False
        self._pet_flash("jump" if done else "wave")

    def _derive_pet_state(self) -> str:
        """Map current CLI activity to a pet animation state.

        A transient reaction beat (wave/jump/failed) wins while it's live;
        otherwise the steady state comes from the shared
        :func:`agent.pet.state.derive_pet_state` so the CLI can't drift from the
        TUI/desktop priority order.
        """
        if self._pet_event and time.monotonic() < self._pet_event_until:
            return self._pet_event
        self._pet_event = ""
        from agent.pet.state import derive_pet_state

        # A live blocking modal (approval / clarify / sudo / secret / slash
        # confirm) means the agent is paused on the user → the `waiting` pose,
        # which outranks the in-flight signals in derive_pet_state.
        awaiting_input = bool(
            self._approval_state
            or self._clarify_state
            or self._sudo_state
            or self._secret_state
            or getattr(self, "_slash_confirm_state", None)
        )

        return derive_pet_state(
            awaiting_input=awaiting_input,
            busy=getattr(self, "_agent_running", False),
            reasoning=self._pet_reasoning,
        ).value

    def _pet_frames_for(self, state: str) -> list:
        """Return (and cache) the half-block grids for one state."""
        cached = self._pet_frames_cache.get(state)
        if cached is not None:
            return cached
        renderer = self._pet_renderer
        if renderer is None:
            return []
        try:
            count = renderer.frame_count(state) or 1
            grids = [renderer.cells(state, i, cols=self._pet_cols) for i in range(count)]
        except Exception:
            grids = []
        self._pet_frames_cache[state] = grids
        return grids

    def _pet_kitty_payload_for(self, state: str) -> dict | None:
        """Return and cache a Kitty virtual-placeholder payload for *state*."""
        with self._pet_lock:
            cached = self._pet_kitty_cache.get(state)
            if cached is not None:
                return cached
            renderer = self._pet_renderer
            image_id = self._pet_kitty_image_id
            if renderer is None or renderer.mode != "kitty":
                return None
        try:
            # PNG encoding is outside _pet_lock: first visit of a state must
            # not stall the prompt under the lock.
            payload = renderer.kitty_payload(state, image_id=image_id)
        except Exception:
            payload = None
        if payload is not None:
            payload = {**payload, "image_id": image_id}
            with self._pet_lock:
                if self._pet_renderer is renderer and self._pet_kitty_image_id == image_id:
                    self._pet_kitty_cache[state] = payload
        return payload

    def _pet_queue_kitty_frame(self, state: str | None = None) -> None:
        """Queue one virtual Kitty frame for the next prompt_toolkit render.

        No-op when the pet pane was never initialized (``__new__`` fixtures
        and ``_force_full_redraw`` / resize recovery on a pet-less CLI).
        """
        if not getattr(self, "_pet_enabled", False):
            return
        if state is None:
            state = self._derive_pet_state()
        payload = self._pet_kitty_payload_for(state)
        if not payload or not payload.get("frames"):
            return
        with self._pet_lock:
            if self._pet_renderer is not None and self._pet_renderer.mode == "kitty":
                self._pet_kitty_pending = payload["frames"][self._pet_frame_idx % len(payload["frames"])]

    def _pet_flush_kitty_frame(self, app) -> None:
        """Write a queued APC after prompt_toolkit has finished its screen diff."""
        with self._pet_lock:
            frame = self._pet_kitty_pending
            self._pet_kitty_pending = ""
        if not frame:
            return
        try:
            # U=1/q=2 leaves the cursor and input stream untouched.
            app.output.write_raw(frame)
            app.output.flush()
        except (OSError, ValueError):
            pass

    def _pet_fragments(self):
        """Return prompt_toolkit FormattedText for the current pet frame, or []."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return []
            state = self._derive_pet_state()
            kitty = self._pet_renderer.mode == "kitty"
        if kitty:
            payload = self._pet_kitty_payload_for(state)
            if not payload:
                return []
            color = pet_render.kitty_color_hex(payload["image_id"])
            frags = []
            for y, row in enumerate(payload["placeholder"]):
                if y:
                    frags.append(("", "\n"))
                frags.append((f"fg:{color}", row))
            return frags
        with self._pet_lock:
            grids = self._pet_frames_for(state)
            if not grids:
                return []
            grid = grids[self._pet_frame_idx % len(grids)]

        frags = []
        for y, row in enumerate(grid):
            if y:
                frags.append(("", "\n"))
            for top, bottom in row:
                tr, tg, tb, ta = top
                br, bg, bb, ba = bottom
                top_op = ta >= 32
                bot_op = ba >= 32
                if not top_op and not bot_op:
                    frags.append(("", " "))
                elif top_op and bot_op:
                    frags.append((f"fg:#{tr:02x}{tg:02x}{tb:02x} bg:#{br:02x}{bg:02x}{bb:02x}", "▀"))
                elif top_op:
                    # Upper half only — leave the lower half the terminal's bg
                    # instead of painting it black (cleaner on light themes).
                    frags.append((f"fg:#{tr:02x}{tg:02x}{tb:02x}", "▀"))
                else:
                    frags.append((f"fg:#{br:02x}{bg:02x}{bb:02x}", "▄"))
        return frags

    def _pet_widget_height(self) -> int:
        """Visible rows for the pet window — 0 collapses it when no pet shows."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return 0
            state = self._derive_pet_state()
            kitty = self._pet_renderer.mode == "kitty"
        if kitty:
            payload = self._pet_kitty_payload_for(state)
            return int(payload.get("rows", 0)) if payload else 0
        with self._pet_lock:
            grids = self._pet_frames_for(state)
            if not grids or not grids[0]:
                return 0
            return len(grids[0])

    def _pet_anim_loop(self) -> None:
        """Advance the frame + invalidate on a timer while a pet is enabled."""
        while self._pet_anim_running:
            time.sleep(self._PET_FRAME_INTERVAL)
            if getattr(self, "_terminal_io_broken", False):
                self._pet_anim_running = False
                break
            now = time.monotonic()
            if now - self._pet_cfg_checked >= self._PET_CFG_INTERVAL:
                self._pet_cfg_checked = now
                self._pet_resolve_config()
            if not self._pet_enabled:
                continue
            with self._pet_lock:
                self._pet_frame_idx += 1
                kitty = self._pet_renderer is not None and self._pet_renderer.mode == "kitty"
            if kitty:
                self._pet_queue_kitty_frame()
            app = getattr(self, "_app", None)
            if app is not None:
                try:
                    app.invalidate()
                except OSError as exc:
                    if getattr(exc, "errno", None) == errno.EIO:
                        self._mark_terminal_io_broken("pet_anim")
                        break
                except Exception:
                    pass

    def _pet_start_anim(self) -> None:
        if self._pet_anim_running:
            return
        self._pet_resolve_config()
        with self._pet_lock:
            kitty = self._pet_enabled and self._pet_renderer is not None and self._pet_renderer.mode == "kitty"
        if kitty:
            self._pet_queue_kitty_frame()
        self._pet_anim_running = True
        self._pet_anim_thread = threading.Thread(target=self._pet_anim_loop, daemon=True)
        self._pet_anim_thread.start()

    def _pet_stop_anim(self) -> None:
        self._pet_anim_running = False
        thread = self._pet_anim_thread
        if thread is not None:
            thread.join(timeout=0.3)
        self._pet_anim_thread = None

    def _voice_record_key_label(self) -> str:
        """Return the configured voice push-to-talk key formatted for UI.

        Shared helper so every voice-facing status line / placeholder /
        recording hint advertises the SAME label as the registered
        prompt_toolkit binding.

        Cached at startup (see ``set_voice_record_key_cache``) rather
        than re-read per render. Two reasons (Copilot round-13 on
        #19835):

        * The prompt_toolkit binding is registered once at session
          start via ``@kb.add(_voice_key)``; re-reading config per
          render meant the status bar could advertise a new shortcut
          after a config edit while the actual binding was still the
          startup chord — exactly the display/binding drift this PR
          is trying to eliminate.
        * The label is on the hot render path (status bar + composer
          placeholder invalidated every 150ms during recording), so
          reading config on every call added avoidable UI overhead.
        """
        return getattr(self, "_voice_record_key_display_cache", None) or "Ctrl+B"

    def set_voice_record_key_cache(self, raw_key: object) -> None:
        """Populate the voice label cache from a raw ``voice.record_key``.

        Called at CLI startup after the prompt_toolkit binding is
        registered so the cached label always matches the live binding.
        """
        try:
            from hermes_cli.voice import format_voice_record_key_for_status
            self._voice_record_key_display_cache = format_voice_record_key_for_status(raw_key)
        except Exception:
            self._voice_record_key_display_cache = "Ctrl+B"

    def _get_voice_status_fragments(self, width: Optional[int] = None):
        """Return the voice status bar fragments for the interactive TUI."""
        width = width or self._get_tui_terminal_width()
        compact = self._use_minimal_tui_chrome(width=width)
        label = self._voice_record_key_label()
        if self._voice_recording:
            if compact:
                return [("class:voice-status-recording", " ● REC ")]
            return [("class:voice-status-recording", f" ● REC  {label} to stop ")]
        if self._voice_processing:
            if compact:
                return [("class:voice-status", " ◉ STT ")]
            return [("class:voice-status", " ◉ Transcribing... ")]
        if compact:
            return [("class:voice-status", f" 🎤 {label} ")]
        tts = " | TTS on" if self._voice_tts else ""
        cont = " | Continuous" if self._voice_continuous else ""
        return [("class:voice-status", f" 🎤 Voice mode{tts}{cont}  —  {label} to record ")]

    @staticmethod
    def _status_bar_goal_segment(snapshot: Dict[str, Any]) -> str:
        """Return the ``⊙ goal 3/20`` segment, or ``""`` when no goal is active.

        Active-goal-only by design: paused/done goals don't occupy status-bar
        real estate (they already print their own glyph lines in the thread).
        """
        if not snapshot.get("goal_active"):
            return ""
        used = snapshot.get("goal_turns_used") or 0
        max_turns = snapshot.get("goal_max_turns") or 0
        if max_turns:
            return f"⊙ goal {used}/{max_turns}"
        return "⊙ goal"

    def _get_status_bar_field_set(self) -> Optional[frozenset]:
        """Return the set of visible status-bar fields from config.

        Reads ``display.status_bar.fields`` from the module-level
        ``CLI_CONFIG`` (no per-render YAML parse — the status bar repaints
        every frame). Returns ``None`` when the user has not customized the
        bar (use built-in defaults, i.e. show everything), or a
        ``frozenset`` of field names when the list is non-empty.

        Available fields: model, context_detail, context_pct, cache_hit,
        latency, tps, compressions, bg_tasks, bg_processes, bg_subagents,
        goal, duration, prompt_elapsed, idle_since, focus, yolo, stash,
        battery, title, total_tokens.
        ``total_tokens`` is opt-in only (never shown by default).
        The field order is fixed; the config controls visibility only.
        """
        from cli import CLI_CONFIG
        if hasattr(self, "_status_bar_field_set_cache"):
            return self._status_bar_field_set_cache
        result = None
        try:
            display = CLI_CONFIG.get("display") if isinstance(CLI_CONFIG, dict) else None
            status_bar = (display or {}).get("status_bar") if isinstance(display, dict) else None
            fields = status_bar.get("fields") if isinstance(status_bar, dict) else None
            if isinstance(fields, list) and fields:
                result = frozenset(str(f) for f in fields)
        except Exception:
            result = None
        self._status_bar_field_set_cache = result
        return result

    def _build_status_bar_text(self, width: Optional[int] = None) -> str:
        """Return a compact one-line session status string for the TUI footer."""
        from cli import format_token_count_compact
        try:
            snapshot = self._get_status_bar_snapshot()
            if width is None:
                width = self._get_tui_terminal_width()
            percent = snapshot["context_percent"]
            percent_label = f"{percent}%" if percent is not None else "--"
            duration_label = snapshot["duration"]
            battery_label = snapshot.get("battery_label") or ""
            battery_prefix = f"{battery_label} │ " if battery_label else ""
            focus_label = snapshot.get("focus_label") or ""
            session_title = snapshot.get("session_title") or ""

            yolo_active = self._is_session_yolo_active()
            goal_segment = self._status_bar_goal_segment(snapshot)
            field_set = self._get_status_bar_field_set()

            def _ok(name: str) -> bool:
                return field_set is None or name in field_set

            if not _ok("title"):
                session_title = ""

            if not _ok("goal"):
                goal_segment = ""
            if not _ok("focus"):
                focus_label = ""
            if width < 52:
                segs = []
                if _ok("model"):
                    segs.append(f"⚕ {snapshot['model_short']}")
                if _ok("duration"):
                    segs.append(duration_label)
                if goal_segment:
                    segs.append(goal_segment)
                if focus_label:
                    segs.append(focus_label)
                if yolo_active and _ok("yolo"):
                    segs.append("⚠ YOLO")
                text = battery_prefix + " · ".join(segs) if segs else f"{battery_prefix}⚕ {snapshot['model_short']}"
                return self._right_align_status_title(text, session_title, width)
            if width < 76:
                parts = []
                if _ok("model"):
                    parts.append(f"⚕ {snapshot['model_short']}")
                if _ok("context_pct"):
                    parts.append(percent_label)
                cache = self._cache_hit_rate(snapshot, precision=0)
                if cache and _ok("cache_hit"):
                    parts.append(cache[1])
                if battery_label:
                    parts.insert(0, battery_label)
                compressions = snapshot.get("compressions", 0)
                if compressions and _ok("compressions"):
                    parts.append(f"🗜️ {compressions}")
                bg_count = snapshot.get("active_background_tasks", 0)
                if bg_count and _ok("bg_tasks"):
                    parts.append(f"▶ {bg_count}")
                bg_proc_count = snapshot.get("active_background_processes", 0)
                if bg_proc_count and _ok("bg_processes"):
                    parts.append(f"⚙ {bg_proc_count}")
                bg_subagent_count = snapshot.get("active_background_subagents", 0)
                if bg_subagent_count and _ok("bg_subagents"):
                    parts.append(f"⛓ {bg_subagent_count}")
                if goal_segment:
                    parts.append(goal_segment)
                if _ok("duration"):
                    parts.append(duration_label)
                if focus_label:
                    parts.append(focus_label)
                if yolo_active and _ok("yolo"):
                    parts.append("⚠ YOLO")
                if not parts:
                    parts = [f"⚕ {snapshot['model_short']}"]
                return self._right_align_status_title(" · ".join(parts), session_title, width)

            parts = []
            if _ok("model"):
                parts.append(f"⚕ {snapshot['model_short']}")
            if _ok("context_detail"):
                if snapshot["context_length"]:
                    ctx_total = _format_context_length(snapshot["context_length"])
                    ctx_used = format_token_count_compact(snapshot["context_tokens"])
                    context_label = f"{ctx_used}/{ctx_total}"
                else:
                    context_label = "ctx --"
                parts.append(context_label)
            if _ok("context_pct"):
                parts.append(percent_label)
            if battery_label:
                parts.insert(0, battery_label)
            compressions = snapshot.get("compressions", 0)
            cache = self._cache_hit_rate(snapshot)
            if cache and _ok("cache_hit"):
                parts.append(cache[1])
            _avg_lat = snapshot.get("avg_latency_label") or ""
            if _avg_lat and _ok("latency"):
                parts.append(f"◷ {_avg_lat}")
            _avg_vel = snapshot.get("avg_velocity_label") or ""
            if _avg_vel and _ok("tps"):
                parts.append(f"↑ {_avg_vel}")
            if compressions and _ok("compressions"):
                parts.append(f"🗜️ {compressions}")
            bg_count = snapshot.get("active_background_tasks", 0)
            if bg_count and _ok("bg_tasks"):
                parts.append(f"▶ {bg_count}")
            bg_proc_count = snapshot.get("active_background_processes", 0)
            if bg_proc_count and _ok("bg_processes"):
                parts.append(f"⚙ {bg_proc_count}")
            bg_subagent_count = snapshot.get("active_background_subagents", 0)
            if bg_subagent_count and _ok("bg_subagents"):
                parts.append(f"⛓ {bg_subagent_count}")
            if goal_segment:
                parts.append(goal_segment)
            if _ok("duration"):
                parts.append(duration_label)
            prompt_elapsed = snapshot.get("prompt_elapsed")
            if prompt_elapsed and _ok("prompt_elapsed"):
                parts.append(prompt_elapsed)
            idle_since = snapshot.get("idle_since")
            if idle_since and _ok("idle_since"):
                parts.append(idle_since)
            if focus_label:
                parts.append(focus_label)
            if yolo_active and _ok("yolo"):
                parts.append("⚠ YOLO")
            # Session token total (Σ) — opt-in only via an explicit fields
            # list, so default bars never widen.
            total_tokens = snapshot.get("session_total_tokens", 0)
            if total_tokens and field_set is not None and "total_tokens" in field_set:
                parts.append(f"Σ{format_token_count_compact(total_tokens)}")
            if not parts:
                parts = [f"⚕ {snapshot['model_short']}"]
            return self._right_align_status_title(" │ ".join(parts), session_title, width)
        except Exception:
            return f"⚕ {self.model if getattr(self, 'model', None) else 'Hermes'}"

    def _get_status_bar_fragments(self):
        from cli import format_token_count_compact
        if not self._status_bar_visible or getattr(self, '_model_picker_state', None) or getattr(self, '_command_palette_state', None):
            return []
        try:
            snapshot = self._get_status_bar_snapshot()
            # Use prompt_toolkit's own terminal width when running inside the
            # TUI — shutil.get_terminal_size() can return stale or fallback
            # values (especially on SSH) that differ from what prompt_toolkit
            # actually renders, causing the fragments to overflow to a second
            # line and produce duplicated status bar rows over long sessions.
            width = self._get_tui_terminal_width()
            duration_label = snapshot["duration"]
            yolo_active = self._is_session_yolo_active()
            goal_segment = self._status_bar_goal_segment(snapshot)
            battery_label = snapshot.get("battery_label") or ""
            battery_style = self._battery_status_style(snapshot.get("battery_category", "dim"))
            focus_label = snapshot.get("focus_label") or ""
            session_title = snapshot.get("session_title") or ""
            field_set = self._get_status_bar_field_set()

            def _ok(name: str) -> bool:
                return field_set is None or name in field_set

            if not _ok("title"):
                session_title = ""

            if not _ok("goal"):
                goal_segment = ""
            if not _ok("focus"):
                focus_label = ""

            def _append(frag_list, sep, *pieces):
                if frag_list:
                    frag_list.append(("class:status-bar-dim", sep))
                frag_list.extend(pieces)

            if width < 52:
                frags = []
                if _ok("model"):
                    frags.append(("class:status-bar", " ⚕ "))
                    frags.append(("class:status-bar-strong", snapshot["model_short"]))
                if _ok("duration"):
                    _append(frags, " · ", ("class:status-bar-dim", duration_label))
                if goal_segment:
                    _append(frags, " · ", ("class:status-bar-strong", goal_segment))
                if focus_label:
                    _append(frags, " · ", ("class:status-bar-strong", focus_label))
                if yolo_active and _ok("yolo"):
                    _append(frags, " · ", ("class:status-bar-yolo", "⚠ YOLO"))
                if not frags:
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                    ]
                frags.append(("class:status-bar", " "))
            else:
                percent = snapshot["context_percent"]
                percent_label = f"{percent}%" if percent is not None else "--"
                if width < 76:
                    compressions = snapshot.get("compressions", 0)
                    bg_count = snapshot.get("active_background_tasks", 0)
                    bg_proc_count = snapshot.get("active_background_processes", 0)
                    bg_subagent_count = snapshot.get("active_background_subagents", 0)
                    frags = []
                    if _ok("model"):
                        frags.append(("class:status-bar", " ⚕ "))
                        frags.append(("class:status-bar-strong", snapshot["model_short"]))
                    if _ok("context_pct"):
                        _append(frags, " · ", (self._status_bar_context_style(percent), percent_label))
                    cache = self._cache_hit_rate(snapshot, precision=0)
                    if cache and _ok("cache_hit"):
                        _append(frags, " · ", (self._cache_hit_rate_style(cache[0]), cache[1]))
                    if compressions and _ok("compressions"):
                        _append(frags, " · ", (self._compression_count_style(compressions), f"🗜️ {compressions}"))
                    if bg_count and _ok("bg_tasks"):
                        _append(frags, " · ", ("class:status-bar-strong", f"▶ {bg_count}"))
                    if bg_proc_count and _ok("bg_processes"):
                        _append(frags, " · ", ("class:status-bar-strong", f"⚙ {bg_proc_count}"))
                    if bg_subagent_count and _ok("bg_subagents"):
                        _append(frags, " · ", ("class:status-bar-strong", f"⛓ {bg_subagent_count}"))
                    if goal_segment:
                        _append(frags, " · ", ("class:status-bar-strong", goal_segment))
                    if _ok("duration"):
                        _append(frags, " · ", ("class:status-bar-dim", duration_label))
                    if focus_label:
                        _append(frags, " · ", ("class:status-bar-strong", focus_label))
                    if yolo_active and _ok("yolo"):
                        _append(frags, " · ", ("class:status-bar-yolo", "⚠ YOLO"))
                    if not frags:
                        frags = [
                            ("class:status-bar", " ⚕ "),
                            ("class:status-bar-strong", snapshot["model_short"]),
                        ]
                    frags.append(("class:status-bar", " "))
                else:
                    bar_style = self._status_bar_context_style(percent)
                    compressions = snapshot.get("compressions", 0)
                    bg_count = snapshot.get("active_background_tasks", 0)
                    bg_proc_count = snapshot.get("active_background_processes", 0)
                    bg_subagent_count = snapshot.get("active_background_subagents", 0)
                    frags = []
                    if _ok("model"):
                        frags.append(("class:status-bar", " ⚕ "))
                        frags.append(("class:status-bar-strong", snapshot["model_short"]))
                    if _ok("context_detail"):
                        if snapshot["context_length"]:
                            ctx_total = _format_context_length(snapshot["context_length"])
                            ctx_used = format_token_count_compact(snapshot["context_tokens"])
                            context_label = f"{ctx_used}/{ctx_total}"
                        else:
                            context_label = "ctx --"
                        _append(frags, " │ ", ("class:status-bar-dim", context_label))
                    if _ok("context_pct"):
                        _append(
                            frags,
                            " │ ",
                            (bar_style, self._build_context_bar(percent)),
                            ("class:status-bar-dim", " "),
                            (bar_style, percent_label),
                        )
                    cache = self._cache_hit_rate(snapshot)
                    if cache and _ok("cache_hit"):
                        _append(frags, " │ ", (self._cache_hit_rate_style(cache[0]), cache[1]))
                    _avg_lat = snapshot.get("avg_latency_label") or ""
                    if _avg_lat and _ok("latency"):
                        _append(frags, " │ ", ("class:status-bar-dim", f"◷ {_avg_lat}"))
                    _avg_vel = snapshot.get("avg_velocity_label") or ""
                    if _avg_vel and _ok("tps"):
                        _append(frags, " │ ", ("class:status-bar-dim", f"↑ {_avg_vel}"))
                    if compressions and _ok("compressions"):
                        _append(frags, " │ ", (self._compression_count_style(compressions), f"🗜️ {compressions}"))
                    if bg_count and _ok("bg_tasks"):
                        _append(frags, " │ ", ("class:status-bar-strong", f"▶ {bg_count}"))
                    if bg_proc_count and _ok("bg_processes"):
                        _append(frags, " │ ", ("class:status-bar-strong", f"⚙ {bg_proc_count}"))
                    if bg_subagent_count and _ok("bg_subagents"):
                        _append(frags, " │ ", ("class:status-bar-strong", f"⛓ {bg_subagent_count}"))
                    if goal_segment:
                        _append(frags, " │ ", ("class:status-bar-strong", goal_segment))
                    if _ok("duration"):
                        _append(frags, " │ ", ("class:status-bar-dim", duration_label))
                    # Position 7: per-prompt elapsed timer (live or frozen)
                    prompt_elapsed = snapshot.get("prompt_elapsed")
                    if prompt_elapsed and _ok("prompt_elapsed"):
                        _append(frags, " │ ", ("class:status-bar-dim", prompt_elapsed))
                    # Position 8: idle time since the last final agent response
                    idle_since = snapshot.get("idle_since")
                    if idle_since and _ok("idle_since"):
                        _append(frags, " │ ", ("class:status-bar-dim", idle_since))
                    # Persistent focus-view badge — so the reduced-output mode
                    # is never invisible (mirrors the YOLO badge convention).
                    if focus_label:
                        _append(frags, " │ ", ("class:status-bar-strong", focus_label))
                    if yolo_active and _ok("yolo"):
                        _append(frags, " │ ", ("class:status-bar-yolo", "⚠ YOLO"))
                    # Session token total (Σ) — opt-in only via an explicit
                    # fields list, so default bars never widen.
                    total_tokens = snapshot.get("session_total_tokens", 0)
                    if total_tokens and field_set is not None and "total_tokens" in field_set:
                        _append(frags, " │ ", ("class:status-bar-dim", f"Σ{format_token_count_compact(total_tokens)}"))
                    if not frags:
                        frags = [
                            ("class:status-bar", " ⚕ "),
                            ("class:status-bar-strong", snapshot["model_short"]),
                        ]
                    frags.append(("class:status-bar", " "))

            # Stash indicator (📌 N) — appended after all width tiers so the
            # user always knows a parked draft exists, even on narrow
            # terminals.  Placed before the battery prepend so it stays at the
            # right edge, and it is the first thing the width trim below drops
            # if the bar genuinely cannot fit.
            try:
                stash_indicator = self._prompt_stash.indicator()
            except Exception:
                stash_indicator = ""
            if stash_indicator and _ok("stash"):
                # Insert before the trailing pad fragment so the bar keeps its
                # one-cell right margin.
                if frags and frags[-1] == ("class:status-bar", " "):
                    frags[-1:-1] = [
                        ("class:status-bar-dim", " · "),
                        ("class:status-bar-strong", stash_indicator),
                    ]
                else:
                    frags.append(("class:status-bar-dim", " · "))
                    frags.append(("class:status-bar-strong", stash_indicator))

            # Battery is the first status-bar element when enabled: prepend it
            # ahead of the leading ⚕ marker in whichever width tier ran above.
            if battery_label and _ok("battery"):
                frags[0:0] = [
                    ("class:status-bar", " "),
                    (battery_style, battery_label),
                    ("class:status-bar-dim", " │"),
                ]

            frags = self._right_align_status_title_fragments(frags, session_title, width)

            total_width = sum(self._status_bar_display_width(text) for _, text in frags)
            if total_width > width:
                plain_text = "".join(text for _, text in frags)
                trimmed = self._trim_status_bar_text(plain_text, width)
                return [("class:status-bar", trimmed)]
            return frags
        except Exception:
            return [("class:status-bar", f" {self._build_status_bar_text()} ")]

    @staticmethod
    def _fmt_stash_age(stashed_at: float) -> str:
        """Return human-readable age string for a stash entry."""
        import time as _t
        secs = int(_t.monotonic() - stashed_at)
        if secs < 10:
            return "just now"
        if secs < 90:
            return f"{secs}s ago"
        mins = secs // 60
        if mins < 60:
            return f"{mins} min ago"
        return f"{mins // 60}h ago"

    def _render_stash_panel(self, stash_list: list, cursor: int, width: int) -> list:
        """Return prompt_toolkit formatted_text fragments for the stash panel box.

        Every horizontal measurement goes through ``_status_bar_display_width``
        (prompt_toolkit's ``get_cwidth``) rather than ``len()``.  The header
        contains 📌, which is one Python codepoint but two terminal cells; the
        original PR chased that off-by-one through three successive
        "subtract 1 from len()" commits.  Measuring in display cells fixes it
        for real and keeps CJK previews from bleeding past the right border.
        """
        cw = self._status_bar_display_width
        W = max(12, min(width - 4, 80))

        n = len(stash_list)
        hdr_prefix_str = f"╭─ 📌 Stash ({n} item{'s' if n != 1 else ''}) "
        HDR_SUFFIX = " Ctrl+S ─╮"
        FTR_PREFIX = "╰"
        FTR_SUFFIX = " ↑↓ Enter=restore  D=delete  Esc ─╯"

        # On narrow terminals the full hint text is wider than the box itself.
        # Drop to compact affordances rather than letting the frame bleed past
        # the right edge (which is what made the panel look broken).
        if cw(hdr_prefix_str) + cw(HDR_SUFFIX) > W:
            hdr_prefix_str = f"╭─ 📌 {n} "
            HDR_SUFFIX = "─╮"
        if cw(FTR_PREFIX) + cw(FTR_SUFFIX) > W:
            FTR_SUFFIX = " ↑↓ ⏎ D Esc ─╯"
        if cw(FTR_PREFIX) + cw(FTR_SUFFIX) > W:
            FTR_SUFFIX = "─╯"

        hdr_dashes = max(0, W - cw(hdr_prefix_str) - cw(HDR_SUFFIX))
        ftr_dashes = max(0, W - cw(FTR_PREFIX) - cw(FTR_SUFFIX))

        # Row inner width: W minus the two '│' border cells.
        INNER = W - 2

        frags: list = []

        def line(text: str, style: str = "") -> None:
            # Final guard: never emit a line wider than the box, whatever the
            # label lengths worked out to.
            frags.append((style, self._trim_status_bar_text(text, W) + "\n"))

        line(f"{hdr_prefix_str}{'─' * hdr_dashes}{HDR_SUFFIX}", "class:subagent-border")

        for i, item in enumerate(stash_list):
            age = self._fmt_stash_age(item["stashed_at"])
            # Row: " ► [N] {age:<10} {preview} "
            prefix = f" {'►' if i == cursor else ' '} [{i + 1}] {age:<10} "
            if cw(prefix) > INNER - 2:
                prefix = f" {'►' if i == cursor else ' '} [{i + 1}] "
            avail = max(0, INNER - cw(prefix) - 1)
            preview = self._trim_status_bar_text(item.get("preview") or "", avail)
            preview = preview + " " * max(0, avail - cw(preview))
            row = self._trim_status_bar_text(f"│{prefix}{preview} │", W)
            if i == cursor:
                frags.append(("class:subagent-selected", row + "\n"))
            else:
                frags.append(("class:subagent-border", "│"))
                frags.append(("class:subagent-sub", f"{prefix}{preview} "))
                frags.append(("class:subagent-border", "│\n"))

        line(f"{FTR_PREFIX}{'─' * ftr_dashes}{FTR_SUFFIX}", "class:subagent-border")
        return frags
