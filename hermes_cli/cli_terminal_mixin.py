"""Terminal repaint/resize recovery, input-mode healing, and clipboard helpers for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import base64
import errno
import os
import shutil
import sys
import threading
import time

from hermes_constants import get_hermes_home


class CLITerminalMixin:
    """Terminal repaint/resize recovery, input-mode healing, and clipboard helpers for the interactive CLI"""

    def _mark_terminal_io_broken(self, reason: str = "") -> None:
        """Stop UI paints after the PTY/stdout becomes unusable (#81521)."""
        from cli import logger
        if getattr(self, "_terminal_io_broken", False):
            return
        self._terminal_io_broken = True
        try:
            self._pet_stop_anim()
        except Exception:
            pass
        logger.warning(
            "Terminal I/O broken%s — freezing UI paints to avoid redraw storm (#81521)",
            f" ({reason})" if reason else "",
        )

    def _invalidate(self, min_interval: float = 0.25) -> None:
        """Throttled UI repaint for high-frequency background updates.

        Use this for spinner frames, streaming token flushes, and other
        repaints that can fire many times per second — the throttle prevents
        terminal blinking on slow/SSH connections, and the resize-recovery
        guard avoids stamping footer/status-bar chrome into scrollback while a
        SIGWINCH reflow is in flight.

        Do NOT use this for user-blocking modal prompts (approval / clarify /
        sudo). Those are rare, one-shot, user-blocking events that must paint
        immediately; route them through ``self._app.invalidate()`` directly, the
        same way the modal key-binding handlers already do. Sending a modal's
        entry paint through this throttle lets an unrelated background repaint
        within the 250ms window — or an in-flight resize — silently drop it, so
        the prompt never renders and times out unseen (#41098).
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        if getattr(self, "_resize_recovery_pending", False):
            return
        now = time.monotonic()
        if hasattr(self, "_app") and self._app and (now - getattr(self, "_last_invalidate", 0.0)) >= min_interval:
            self._last_invalidate = now
            try:
                self._app.invalidate()
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.EIO:
                    self._mark_terminal_io_broken("invalidate")
                    return
                raise

    def _paint_now(self) -> None:
        """Immediate, unthrottled repaint for user-blocking modal prompts.

        Background-thread callbacks (approval / clarify / sudo) set their modal
        state then call this to make the panel visible at once. It deliberately
        bypasses the ``_invalidate`` throttle and resize-recovery guard — a
        modal the user is actively waiting on must never be dropped — mirroring
        the direct ``event.app.invalidate()`` the modal key-binding handlers
        already use. See ``_invalidate`` for why the throttle must not gate
        these paints (#41098).
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        app = getattr(self, "_app", None)
        if app is not None:
            try:
                app.invalidate()
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.EIO:
                    self._mark_terminal_io_broken("paint_now")
                    return
                raise
            except Exception:
                pass

    def _force_full_redraw(self) -> None:
        """Force a clean full-screen repaint of the prompt_toolkit UI.

        Used to recover from terminal buffer drift caused by external
        redraws we can't detect — e.g. macOS cmux / tmux tab switches,
        ``clear`` issued from a subshell, or SSH window restores. These
        wipe or repaint the terminal without firing SIGWINCH, so
        prompt_toolkit's tracked ``_cursor_pos`` no longer matches reality
        and the next incremental redraw stacks on top of stale content
        (ghost status bars, duplicated prompts).

        Bound to Ctrl+L and exposed as the ``/redraw`` slash command,
        matching the standard terminal-UX convention (bash, zsh, fish,
        vim, htop).
        """
        from cli import _replay_output_history
        if getattr(self, "_terminal_io_broken", False):
            return
        app = getattr(self, "_app", None)
        if not app:
            return
        self._clear_prompt_toolkit_screen(
            app,
            rebuild_scrollback=self._redraw_rebuilds_scrollback(),
        )
        if getattr(self, "_terminal_io_broken", False):
            return
        _replay_output_history()
        self._pet_queue_kitty_frame()
        try:
            app.invalidate()
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EIO:
                self._mark_terminal_io_broken("force_full_redraw")
                return
            raise
        except Exception:
            pass

    def _schedule_focus_regain_redraw(self, min_interval: float = 1.0) -> None:
        """Repaint after a terminal focus-in report (``CSI I``), rate-limited.

        Terminals with focus tracking active (Ghostty, iTerm2, xterm builds,
        multiplexers that toggle DECSET 1004 upstream) emit ``\\x1b[I`` when
        the Hermes tab/window becomes visible again. Emulators can coalesce
        or drop hidden-tab output and repaint the surface while we're
        invisible, so on regain prompt_toolkit's incremental diff stacks on
        stale content — a second copy of the composer/prompt chrome next to
        the ghost of the old one (#60920 focus-regain variant, #25337).

        The stock handling maps ``CSI I``/``CSI O`` to ``Keys.Ignore`` so the
        bytes never pollute the input buffer; this hook additionally routes
        focus-in through the same recovery as Ctrl+L / ``/redraw``. It is
        self-gating: terminals that never enable focus tracking never emit
        the sequence, so nothing changes for them. Rate-limited so a burst of
        focus reports (rapid Alt+Tab, mux pane hops) repaints at most once
        per ``min_interval`` seconds.
        """
        now = time.monotonic()
        last = getattr(self, "_last_focus_regain_redraw", 0.0)
        if now - last < min_interval:
            return
        self._last_focus_regain_redraw = now
        self._force_full_redraw()

    @staticmethod
    def _redraw_rebuilds_scrollback() -> bool:
        """Return whether CLI redraw/resize recovery should clear scrollback.

        Some terminal/tmux stacks move prompt_toolkit's non-fullscreen bottom
        chrome into scrollback when the window is maximized/restored. A normal
        CSI 2J viewport clear cannot remove those stale prompt/input-rule rows,
        so users who hit that class of bug need CSI 3J as well, followed by the
        existing bounded output-history replay.
        """
        from cli import CLI_CONFIG
        display_config = CLI_CONFIG.get("display") if isinstance(CLI_CONFIG, dict) else {}
        if not isinstance(display_config, dict):
            display_config = {}
        raw = display_config.get("cli_rebuild_scrollback_on_redraw", False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on", "always"}
        return bool(raw)

    def _recover_terminal_after_interrupt(self) -> None:
        """Recover the terminal after an interrupted agent turn (#33271).

        When the user interrupts a running turn by typing a new message,
        prompt_toolkit may have an in-flight ``CSI 6n`` cursor-position query
        whose reply (``ESC[<row>;<col>R``) arrives on stdin after the input
        parser has torn down. The reply then leaks as literal text
        (``^[[19;1R``) and the VT100 parser can stall in a partial-escape
        state, accepting no further keystrokes — the terminal appears frozen.

        Two steps recover a sane state:
          1. ``flush_stdin()`` drains stray escape bytes from the OS input
             buffer (``termios.tcflush(TCIFLUSH)``; no-op on non-TTY).
          2. ``_force_full_redraw()`` drops prompt_toolkit's cached
             screen/cursor state and forces a clean repaint.

        Both steps are independently safe and self-guard, so a failure of one
        never prevents the other. If the PTY is already dead (EIO), skip the
        redraw entirely — painting a broken fd is the #81521 redraw storm.
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        try:
            from hermes_cli.curses_ui import flush_stdin
            flush_stdin()
        except Exception:
            pass
        # #60920: The interruption marker is now printed with
        # _suspend_output_history in chat(), so _OUTPUT_HISTORY only
        # contains the normal response text (no marker text). Do NOT
        # clear history here — _force_full_redraw → _replay_output_history
        # replays the response correctly without duplicating the marker.
        # The /redraw + Ctrl+L paths also preserve replay for scrollback
        # recovery as intended.
        self._force_full_redraw()

    def _clear_prompt_toolkit_screen(self, app, *, rebuild_scrollback: bool = False) -> None:
        """Clear the terminal and reset prompt_toolkit renderer state."""
        if getattr(self, "_terminal_io_broken", False):
            return
        try:
            renderer = app.renderer
            out = renderer.output
            out.reset_attributes()
            out.erase_screen()
            if rebuild_scrollback:
                try:
                    out.write_raw("\x1b[3J")
                except Exception:
                    pass
            out.cursor_goto(0, 0)
            out.flush()
            # Drop prompt_toolkit's cached screen + cursor state so the
            # next _redraw() starts from a known (0, 0) origin and
            # re-renders every cell rather than diffing against stale.
            renderer.reset(leave_alternate_screen=False)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EIO:
                self._mark_terminal_io_broken("clear_screen")
                return
            pass
        except Exception:
            pass

    def _recover_after_resize(self, app, original_on_resize) -> None:
        """Recover a resized classic CLI without desynchronizing cursor state.

        Unlike _force_full_redraw, we do NOT clear the physical screen or
        scrollback here.  The startup banner and tool summary are printed
        before prompt_toolkit owns the live chrome, so they live in normal
        terminal scrollback.  Erasing the screen on SIGWINCH removes that
        startup UI and ``_replay_output_history`` cannot reconstruct it
        (the banner was never added to ``_OUTPUT_HISTORY``).

        Let prompt_toolkit's own resize path run with its renderer cursor
        cache intact. Its Application._on_resize() starts with
        renderer.erase(leave_alternate_screen=False), which needs the cached
        cursor position to move back to the live prompt origin before
        erase_down(). Resetting the renderer before that erase loses the
        origin and can leave stale prompt glyphs after a narrow resize.

        We also flag ``_status_bar_suppressed_after_resize`` so the dynamic
        status bar and input separator rules stay hidden while the terminal
        reflow settles.  On column shrink the terminal reflows already-rendered
        status bar rows into scrollback before prompt_toolkit can erase them;
        drawing a fresh full-width bar immediately makes the old and new
        versions look duplicated (#19280, #22976).

        Suppression alone is not enough on a WIDTH change.  prompt_toolkit's
        ``renderer.erase()`` does ``cursor_up(_cursor_pos.y)`` + ``erase_down()``
        using the ``_cursor_pos.y`` cached from the LAST render at the OLD
        width (renderer.py).  When the column count shrinks, the terminal
        reflows each already-painted full-width chrome row into 2+ physical
        rows, so the cached ``y`` undershoots: ``cursor_up`` does not climb
        past the reflowed rows and ``erase_down`` leaves the stale bar stranded
        ABOVE the live origin.  The next paint then stacks a fresh bar below it
        — the duplicated-status-bar report (two bars, two elapsed readings).
        Suppression hides the *new* bar but never erases the already-reflowed
        *old* one, so the ghost survives the whole suppression window.

        Fix: on a width change, wipe the visible viewport with ``erase_screen``
        (CSI 2J) BEFORE delegating to prompt_toolkit's resize, then let its
        repaint redraw from a clean origin.  This is banner-safe: 2J clears
        only the visible screen, NOT scrollback history (that is CSI 3J, which
        we do not send here — ``rebuild_scrollback=False``), so the startup
        banner that scrolled into history is preserved and
        ``_replay_output_history`` is not needed.  Row-count-only changes skip
        the clear (no reflow, so no ghost) to avoid an unnecessary repaint.

        The suppression is transient: a short follow-up timer clears it and
        repaints once the reflow has settled, so the bar returns on its own
        during idle.  Previously the flag was only cleared on the next
        *submitted* user input, so a resize/reflow (tmux pane change, SSH
        window restore, font zoom) followed by idle left the status bar hidden
        indefinitely even while the refresh clock kept ticking (the dynamic
        chrome rendered at height 0 on every repaint).  The next-submit clear
        at the input loop remains as a fast path.
        """
        from cli import _replay_output_history
        self._status_bar_suppressed_after_resize = True
        # On a WIDTH change the terminal has already reflowed the old full-width
        # chrome into extra physical rows that prompt_toolkit's stale-cursor
        # erase (cursor_up(_cursor_pos.y) cached at the OLD width) will not
        # reach, leaving a duplicated status bar stranded above the live origin.
        # Ctrl+L / /redraw clears it cleanly, so route the resize path through
        # the SAME recovery: wipe the visible viewport (banner-safe — CSI 2J
        # by default; CSI 3J only when display.cli_rebuild_scrollback_on_redraw
        # is enabled) and replay the transcript so nothing is lost.
        # Same-width SIGWINCH (tmux attach, benign focus/tab signals) is left
        # untouched — no clear, no replay — because a 2J without replay erases
        # the visible transcript and a replay against preserved scrollback
        # duplicates it (#65293). The stale-previous_screen crash tmux attach
        # used to trigger is handled by _hermes_call_output_screen_diff's
        # retry-with-first-paint instead (#83874).
        try:
            new_width = self._get_tui_terminal_width()
        except Exception:
            new_width = None
        prev_width = getattr(self, "_last_resize_width", None)
        # Replay only on an OBSERVED width change.  The first signal of a
        # session must not count as one (#65293): GNOME Terminal and friends
        # deliver benign SIGWINCHes (tab bar appearing, monitor-scale change,
        # focus events), and a 2J+replay against preserved scrollback
        # duplicates everything ``_OUTPUT_HISTORY`` holds — after a resume
        # that is the entire "Previous Conversation" recap plus the first
        # live exchange.  ``_install_resize_recovery`` seeds the baseline at
        # startup, so an initial maximize/restore still differs from it and
        # is still recovered; with no baseline (width probe failed) this
        # signal just records one for the next comparison.
        width_changed = (
            new_width is not None
            and prev_width is not None
            and new_width != prev_width
        )
        if width_changed:
            try:
                self._clear_prompt_toolkit_screen(
                    app,
                    rebuild_scrollback=self._redraw_rebuilds_scrollback(),
                )
                _replay_output_history()
            except Exception:
                pass
        if new_width is not None:
            self._last_resize_width = new_width
        if width_changed:
            self._pet_queue_kitty_frame()
        original_on_resize()
        self._schedule_status_bar_unsuppress(app)

    def _schedule_status_bar_unsuppress(self, app, delay: float = 0.35) -> None:
        """Clear the post-resize status-bar suppression after the reflow settles.

        Debounced: a fresh resize cancels the pending unsuppress and restarts
        the timer, so a resize storm only repaints the bar once it stops.
        """
        try:
            old_timer = getattr(self, "_status_bar_unsuppress_timer", None)
            if old_timer is not None:
                try:
                    old_timer.cancel()
                except Exception:
                    pass

            def _clear():
                self._status_bar_suppressed_after_resize = False
                try:
                    app.invalidate()
                except Exception:
                    pass

            def _fire():
                try:
                    loop = getattr(app, "loop", None)
                except Exception:
                    loop = None
                if loop is not None:
                    try:
                        loop.call_soon_threadsafe(_clear)
                        return
                    except Exception:
                        pass
                _clear()

            timer = threading.Timer(delay, _fire)
            timer.daemon = True
            self._status_bar_unsuppress_timer = timer
            timer.start()
        except Exception:
            # Fail open: never leave the bar stuck hidden.
            self._status_bar_suppressed_after_resize = False

    def _schedule_resize_recovery(self, app, original_on_resize, delay: float = 0.12) -> None:
        """Debounce resize redraws so footer chrome is not stamped into scrollback."""
        try:
            old_timer = getattr(self, "_resize_recovery_timer", None)
            lock = getattr(self, "_resize_recovery_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._resize_recovery_lock = lock

            def _timer_fired(timer_ref):
                def _run_recovery():
                    with lock:
                        if getattr(self, "_resize_recovery_timer", None) is not timer_ref:
                            return
                        self._resize_recovery_timer = None
                        self._resize_recovery_pending = False
                    self._recover_after_resize(app, original_on_resize)

                try:
                    loop = app.loop  # type: ignore[attr-defined]
                except Exception:
                    loop = None
                if loop is not None:
                    try:
                        loop.call_soon_threadsafe(_run_recovery)
                        return
                    except Exception:
                        pass
                _run_recovery()

            with lock:
                if old_timer is not None:
                    try:
                        old_timer.cancel()
                    except Exception:
                        pass
                self._resize_recovery_pending = True
                timer = threading.Timer(delay, lambda: _timer_fired(timer))
                timer.daemon = True
                self._resize_recovery_timer = timer
                timer.start()
        except Exception:
            self._resize_recovery_pending = False
            self._recover_after_resize(app, original_on_resize)

    def _install_resize_recovery(self, app) -> None:
        """Route prompt_toolkit's ``_on_resize`` through the debounced
        ghost-clearing recovery (#5474/#49120) and record the current terminal
        width as the baseline for width-change detection.

        Seeding the baseline here is what keeps the session's FIRST SIGWINCH
        honest (#65293): ``_recover_after_resize`` replays the transcript only
        on an observed width change, and without a startup baseline it could
        not tell a benign signal (GNOME Terminal tab bar, monitor-scale
        change) from a real one.  An initial maximize/restore still differs
        from the seeded width, so it is still recovered.

        The probe reads ``app.output`` directly — NOT
        ``_get_tui_terminal_width`` — because this runs before ``app.run()``,
        when ``get_app()`` still returns prompt_toolkit's DummyApplication
        whose DummyOutput reports a hardcoded 80 columns; seeding that fake
        width would make the first real signal look like a width change and
        resurrect the duplicate-replay bug this exists to fix.
        ``app.output`` is the same object the running app's resize handler
        measures, so install-time and signal-time widths are comparable.
        """
        width = None
        try:
            width = app.output.get_size().columns
        except Exception:
            width = None
        if not width or width <= 0:
            try:
                width = shutil.get_terminal_size((80, 24)).columns
            except Exception:
                width = None
        self._last_resize_width = width
        original_on_resize = app._on_resize

        def _resize_clear_ghosts():
            self._schedule_resize_recovery(app, original_on_resize)

        app._on_resize = _resize_clear_ghosts

    def _try_attach_clipboard_image(self) -> bool:
        """Check clipboard for an image and attach it if found.

        Saves the image to ~/.hermes/images/ and appends the path to
        ``_attached_images``.  Returns True if an image was attached.
        """
        from cli import datetime
        from hermes_cli.clipboard import save_clipboard_image

        img_dir = get_hermes_home() / "images"
        self._image_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = img_dir / f"clip_{ts}_{self._image_counter}.png"

        if save_clipboard_image(img_path):
            self._attached_images.append(img_path)
            return True
        self._image_counter -= 1
        return False

    def _write_osc52_clipboard(self, text: str) -> None:
        """Copy *text* to terminal clipboard via OSC 52.

        Wrapped for tmux/screen passthrough (mirrors the TUI's
        wrapForMultiplexer in ui-tui/src/lib/osc52.ts) — without the DCS
        wrapper the multiplexer consumes the sequence and the copy is
        silently lost.
        """
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        seq = f"\x1b]52;c;{payload}\x07"
        if os.environ.get("TMUX"):
            seq = "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
        elif os.environ.get("STY"):
            seq = "\x1bP" + seq + "\x1b\\"
        out = getattr(self, "_app", None)
        output = getattr(out, "output", None) if out else None
        if output and hasattr(output, "write_raw"):
            output.write_raw(seq)
            output.flush()
            return
        if output and hasattr(output, "write"):
            output.write(seq)
            output.flush()
            return
        sys.stdout.write(seq)
        sys.stdout.flush()

    def _recover_terminal_input_modes(self, *, reason: str) -> None:
        """Best-effort reset when leaked mouse reports indicate mode drift."""
        from cli import (
            CLI_CONFIG,
            _DIM,
            _RST,
            _TERMINAL_INPUT_MODE_RESET_SEQ,
            _cli_multiline_shortcuts_enabled,
            _cprint,
            _enable_extended_enter_keys,
            logger,
        )
        now = time.monotonic()
        # Rate-limit to avoid thrashing if a terminal floods reports.
        if now - self._last_input_mode_recovery < 0.5:
            return
        self._last_input_mode_recovery = now

        out = getattr(self, "_app", None)
        output = getattr(out, "output", None) if out else None
        try:
            if output and hasattr(output, "write_raw"):
                output.write_raw(_TERMINAL_INPUT_MODE_RESET_SEQ)
                output.flush()
            elif output and hasattr(output, "write"):
                output.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
                output.flush()
            else:
                sys.stdout.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
                sys.stdout.flush()
        except Exception:
            return

        # The reset sequence above pops kitty keyboard mode and resets
        # modifyOtherKeys too — re-request extended keys so Shift+Enter /
        # modified-key reporting isn't silently dead for the rest of the
        # session after a recovery (sibling of the startup push).
        try:
            if _cli_multiline_shortcuts_enabled(self.config or CLI_CONFIG):
                _enable_extended_enter_keys(output)
        except Exception:
            pass

        logger.warning("Recovered terminal input modes after leak: %s", reason)
        if not self._input_mode_recovery_notice_shown:
            self._input_mode_recovery_notice_shown = True
            _cprint(
                f"  {_DIM}Recovered terminal input modes after leaked mouse reports. "
                f"If this repeats, run /new or restart this tab.{_RST}"
            )

    def _check_termios_drift(self) -> None:
        """Watchdog: heal the tty if it drifted back to cooked mode.

        See ``_heal_cooked_mode_drift`` for the failure class (a lost
        ``run_in_terminal`` cooked→raw restore leaves the terminal
        line-buffering keystrokes while the prompt_toolkit app believes it
        owns raw mode — the CLI looks dead but the process is healthy).

        Called from ``process_loop``'s idle branch, so a drifted terminal
        self-heals within ~a second of the agent going idle instead of
        requiring an external ``stty`` rescue.  Skipped while a
        ``run_in_terminal`` window is legitimately holding cooked mode
        (``app._running_in_terminal``), while the agent is running (approval
        prompts and sudo prompts legitimately manipulate the tty), and on
        Windows (no termios).
        """
        from cli import _DIM, _RST, _cprint, _heal_cooked_mode_drift, logger
        if os.name == "nt":
            return
        app = getattr(self, "_app", None)
        if app is None or not getattr(app, "_is_running", False):
            return
        # A run_in_terminal window is *supposed* to be cooked — don't fight it.
        if getattr(app, "_running_in_terminal", False):
            return
        now = time.monotonic()
        if now - self._last_termios_drift_check < 1.0:
            return
        self._last_termios_drift_check = now
        try:
            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
        except Exception:
            return
        if _heal_cooked_mode_drift(fd):
            logger.warning(
                "Healed cooked-mode termios drift on stdin — a "
                "run_in_terminal cooked→raw restore was lost."
            )
            # Redraw so the prompt is visibly alive again.
            try:
                self._invalidate()
            except Exception:
                pass
            if not self._termios_drift_notice_shown:
                self._termios_drift_notice_shown = True
                _cprint(
                    f"  {_DIM}Recovered terminal from cooked-mode drift "
                    f"(input should respond normally again).{_RST}"
                )
