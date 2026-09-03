"""Terminal repaint/resize recovery, input-mode healing, and clipboard helpers for the interactive CLI

Mixin bound onto ``HermesCLI`` via the MRO. cli.py-internal symbols are imported LAZILY
inside each method (``from cli import ...``) — never at module load (import cycle).
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


def _is_eio(exc: BaseException) -> bool:
    return getattr(exc, "errno", None) == errno.EIO


def _run_on_app_loop(app, fn) -> None:
    """Run *fn* on the app's asyncio loop when one exists, else inline (fail-open)."""
    try:
        loop = getattr(app, "loop", None)
    except Exception:
        loop = None
    if loop is not None:
        try:
            loop.call_soon_threadsafe(fn)
            return
        except Exception:
            pass
    fn()


def _write_terminal_sequence(app, seq: str) -> None:
    """Write a raw escape *seq* via the app output (write_raw > write) or stdout."""
    output = getattr(app, "output", None) if app else None
    if output and hasattr(output, "write_raw"):
        output.write_raw(seq)
        output.flush()
    elif output and hasattr(output, "write"):
        output.write(seq)
        output.flush()
    else:
        sys.stdout.write(seq)
        sys.stdout.flush()


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

    def _app_invalidate(self, app, where: str, *, swallow: bool) -> None:
        """``app.invalidate()``: EIO freezes paints, other OSErrors re-raise, and any
        other exception re-raises unless *swallow*."""
        try:
            app.invalidate()
        except OSError as exc:
            if _is_eio(exc):
                self._mark_terminal_io_broken(where)
                return
            raise
        except Exception:
            if not swallow:
                raise

    def _invalidate(self, min_interval: float = 0.25) -> None:
        """Throttled UI repaint for high-frequency background updates.

        For spinner frames, streaming token flushes and other repaints that fire many
        times per second: the throttle prevents blinking on slow/SSH links, and the
        resize-recovery guard avoids stamping footer/status-bar chrome into scrollback
        while a SIGWINCH reflow is in flight.

        Do NOT use for user-blocking modal prompts (approval / clarify / sudo): those
        must paint immediately via ``_paint_now``. Sent through this throttle, an
        unrelated repaint within the 250ms window — or an in-flight resize — silently
        drops the modal's entry paint, so it never renders and times out unseen (#41098).
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        if getattr(self, "_resize_recovery_pending", False):
            return
        now = time.monotonic()
        if hasattr(self, "_app") and self._app and (now - getattr(self, "_last_invalidate", 0.0)) >= min_interval:
            self._last_invalidate = now
            self._app_invalidate(self._app, "invalidate", swallow=False)

    def _paint_now(self) -> None:
        """Immediate, unthrottled repaint for user-blocking modal prompts.

        Deliberately bypasses the ``_invalidate`` throttle and resize-recovery guard —
        a modal the user is waiting on must never be dropped (#41098) — mirroring the
        direct ``event.app.invalidate()`` the modal key-binding handlers use.
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        app = getattr(self, "_app", None)
        if app is not None:
            self._app_invalidate(app, "paint_now", swallow=True)

    def _force_full_redraw(self) -> None:
        """Force a clean full-screen repaint of the prompt_toolkit UI (Ctrl+L, ``/redraw``).

        Recovers from terminal buffer drift caused by external redraws we can't detect
        (cmux/tmux tab switches, ``clear`` from a subshell, SSH window restores): they
        repaint without SIGWINCH, so prompt_toolkit's tracked ``_cursor_pos`` is stale
        and the next incremental redraw stacks on old content (ghost status bars).
        """
        from cli import _replay_output_history
        if getattr(self, "_terminal_io_broken", False):
            return
        app = getattr(self, "_app", None)
        if not app:
            return
        self._clear_prompt_toolkit_screen(app, rebuild_scrollback=self._redraw_rebuilds_scrollback())
        if getattr(self, "_terminal_io_broken", False):
            return
        _replay_output_history()
        self._pet_queue_kitty_frame()
        self._app_invalidate(app, "force_full_redraw", swallow=True)

    def _schedule_focus_regain_redraw(self, min_interval: float = 1.0) -> None:
        """Repaint after a terminal focus-in report (``CSI I``), rate-limited.

        Terminals with focus tracking (Ghostty, iTerm2, xterm, muxes toggling DECSET
        1004) emit ``\\x1b[I`` when the Hermes tab becomes visible again; emulators may
        coalesce hidden-tab output, so on regain the incremental diff stacks on stale
        content (#60920 focus-regain variant, #25337). Self-gating — terminals without
        focus tracking never emit it. Rate-limited so a burst (rapid Alt+Tab, pane hops)
        repaints at most once per ``min_interval`` seconds.
        """
        now = time.monotonic()
        if now - getattr(self, "_last_focus_regain_redraw", 0.0) < min_interval:
            return
        self._last_focus_regain_redraw = now
        self._force_full_redraw()

    @staticmethod
    def _redraw_rebuilds_scrollback() -> bool:
        """Whether redraw/resize recovery should also clear scrollback (CSI 3J).

        Some terminal/tmux stacks move prompt_toolkit's bottom chrome into scrollback on
        maximize/restore; CSI 2J cannot remove those rows, so affected users opt in to 3J
        followed by the bounded output-history replay.
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

        An in-flight ``CSI 6n`` cursor query may answer (``ESC[<row>;<col>R``) after the
        input parser tore down: the reply leaks as literal text and the VT100 parser can
        stall mid-escape, so the terminal looks frozen. ``flush_stdin()`` drains stray
        bytes (no-op on non-TTY), then ``_force_full_redraw()`` repaints cleanly; each
        step self-guards so one failing never blocks the other. A dead PTY (EIO) skips
        the redraw — painting a broken fd is the #81521 redraw storm.

        Do NOT clear output history here: the interruption marker is printed under
        ``_suspend_output_history`` in chat(), so the replay reproduces the response
        without duplicating the marker (#60920).
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        try:
            from hermes_cli.curses_ui import flush_stdin
            flush_stdin()
        except Exception:
            pass
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
            # Drop cached screen + cursor state so the next _redraw() starts from a
            # known (0, 0) origin and re-renders every cell instead of diffing stale.
            renderer.reset(leave_alternate_screen=False)
        except OSError as exc:
            if _is_eio(exc):
                self._mark_terminal_io_broken("clear_screen")
        except Exception:
            pass

    def _recover_after_resize(self, app, original_on_resize) -> None:
        """Recover a resized classic CLI without desynchronizing cursor state.

        Unlike ``_force_full_redraw`` this never clears scrollback: the startup banner
        lives there and ``_replay_output_history`` cannot reconstruct it. prompt_toolkit's
        own resize path runs with its renderer cursor cache intact — its
        ``Application._on_resize()`` erases via the cached cursor position, and resetting
        the renderer first loses the origin and strands stale prompt glyphs.

        ``_status_bar_suppressed_after_resize`` hides the dynamic status bar / input rules
        while the reflow settles: on column shrink the terminal reflows already-painted
        rows into scrollback before prompt_toolkit erases them, so a fresh bar looks
        duplicated (#19280, #22976). Suppression alone cannot erase an already-reflowed
        OLD bar (``renderer.erase()`` does ``cursor_up(_cursor_pos.y)`` with the y cached
        at the OLD width, undershooting the reflowed rows), so on an OBSERVED width
        change we wipe the viewport (CSI 2J — banner-safe; 3J only when
        ``display.cli_rebuild_scrollback_on_redraw``) and replay the transcript before
        delegating. Same-width SIGWINCH (tmux attach, GNOME tab bar, focus signals) and
        the first signal without a baseline are left untouched: a 2J+replay against
        preserved scrollback duplicates everything in ``_OUTPUT_HISTORY`` (#65293);
        ``_install_resize_recovery`` seeds the baseline so an initial maximize still
        counts. The stale-previous_screen crash on tmux attach is handled by
        ``_hermes_call_output_screen_diff``'s retry (#83874).

        Suppression is transient: a debounced timer clears it and repaints once the
        reflow settles, so the bar returns during idle (previously it stayed hidden
        until the next submitted input). The next-submit clear remains as a fast path.
        """
        from cli import _replay_output_history
        self._status_bar_suppressed_after_resize = True
        try:
            new_width = self._get_tui_terminal_width()
        except Exception:
            new_width = None
        prev_width = getattr(self, "_last_resize_width", None)
        width_changed = new_width is not None and prev_width is not None and new_width != prev_width
        if width_changed:
            try:
                self._clear_prompt_toolkit_screen(
                    app, rebuild_scrollback=self._redraw_rebuilds_scrollback()
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

        Debounced: a fresh resize cancels the pending timer, so a resize storm repaints
        the bar only once it stops.
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

            timer = threading.Timer(delay, lambda: _run_on_app_loop(app, _clear))
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

                _run_on_app_loop(app, _run_recovery)

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
        """Route ``app._on_resize`` through the debounced ghost-clearing recovery
        (#5474/#49120) and seed the width baseline for width-change detection.

        The seed keeps the session's FIRST SIGWINCH honest (#65293): without it a benign
        signal is indistinguishable from a real resize. It reads ``app.output`` directly,
        NOT ``_get_tui_terminal_width``: before ``app.run()`` ``get_app()`` is the
        DummyApplication whose DummyOutput reports a hardcoded 80 columns, and seeding
        that fake width would make the first real signal look like a change.
        ``app.output`` is what the running resize handler measures, so both are comparable.
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
        """Save a clipboard image to ~/.hermes/images/ and attach it; True if attached."""
        from cli import datetime
        from hermes_cli.clipboard import save_clipboard_image

        self._image_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = get_hermes_home() / "images" / f"clip_{ts}_{self._image_counter}.png"
        if save_clipboard_image(img_path):
            self._attached_images.append(img_path)
            return True
        self._image_counter -= 1
        return False

    def _write_osc52_clipboard(self, text: str) -> None:
        """Copy *text* to the terminal clipboard via OSC 52.

        Wrapped for tmux/screen passthrough (mirrors ui-tui/src/lib/osc52.ts) — without
        the DCS wrapper the multiplexer consumes the sequence and the copy is lost.
        """
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        seq = f"\x1b]52;c;{payload}\x07"
        if os.environ.get("TMUX"):
            seq = "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
        elif os.environ.get("STY"):
            seq = "\x1bP" + seq + "\x1b\\"
        _write_terminal_sequence(getattr(self, "_app", None), seq)

    def _recover_terminal_input_modes(self, *, reason: str) -> None:
        """Best-effort reset when leaked mouse reports indicate mode drift."""
        from cli import (
            CLI_CONFIG, _DIM, _RST, _TERMINAL_INPUT_MODE_RESET_SEQ,
            _cli_multiline_shortcuts_enabled, _cprint, _enable_extended_enter_keys, logger,
        )
        now = time.monotonic()
        # Rate-limit to avoid thrashing if a terminal floods reports.
        if now - self._last_input_mode_recovery < 0.5:
            return
        self._last_input_mode_recovery = now

        app = getattr(self, "_app", None)
        output = getattr(app, "output", None) if app else None
        try:
            _write_terminal_sequence(app, _TERMINAL_INPUT_MODE_RESET_SEQ)
        except Exception:
            return

        # The reset pops kitty keyboard mode and resets modifyOtherKeys too — re-request
        # extended keys so Shift+Enter isn't silently dead for the rest of the session.
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

        See ``_heal_cooked_mode_drift``: a lost ``run_in_terminal`` cooked→raw restore
        leaves the tty line-buffering while prompt_toolkit believes it owns raw mode —
        the CLI looks dead but the process is healthy. Called from ``process_loop``'s
        idle branch so it self-heals within ~1s of idling. Skipped while a
        ``run_in_terminal`` window legitimately holds cooked mode, while the agent is
        running (approval/sudo prompts manipulate the tty), and on Windows (no termios).
        """
        from cli import _DIM, _RST, _cprint, _heal_cooked_mode_drift, logger
        if os.name == "nt":
            return
        app = getattr(self, "_app", None)
        if app is None or not getattr(app, "_is_running", False):
            return
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
            try:
                self._invalidate()  # so the prompt is visibly alive again
            except Exception:
                pass
            if not self._termios_drift_notice_shown:
                self._termios_drift_notice_shown = True
                _cprint(
                    f"  {_DIM}Recovered terminal from cooked-mode drift "
                    f"(input should respond normally again).{_RST}"
                )
