"""Interactive picker for ``hermes sessions browse`` (extracted from sessions_cmd).

Curses UI with live search filtering and ``d`` delete-with-confirmation; a
numbered-list fallback when curses is unavailable (Windows, etc.).
"""

from typing import Optional


def _relative_time(ts) -> str:
    from hermes_cli.timefmt import relative_time

    return relative_time(ts)


def _session_status_tag(status: Optional[str]) -> str:
    """Short fixed-width tag for a session lifecycle status."""
    return {"complete": "done", "interrupted": "intr", "error": "err", "empty": "empty"}.get(status or "", "-")


def _annotate_session_statuses(sessions: list, session_db) -> None:
    """Attach a ``_status`` key to each session row (best-effort, cheap).

    Uses ``SessionDB.session_lifecycle_statuses`` — one indexed last-message
    lookup per listed session, never a transcript scan. On any failure the
    rows simply stay untagged and the picker renders '-' for status.
    """
    if session_db is None or not sessions:
        return
    try:
        statuses = session_db.session_lifecycle_statuses([s.get("id") for s in sessions])
    except Exception:
        return
    for s in sessions:
        s["_status"] = statuses.get(s.get("id"), "")


def _label(s: dict) -> str:
    """Title, else preview, else id."""
    return ((s.get("title") or "").strip() or (s.get("preview") or "").strip() or s["id"])


def _msgs_str(s: dict) -> str:
    msgs = s.get("message_count")
    return str(msgs) if isinstance(msgs, int) else "-"


def _match(s: dict, query: str) -> bool:
    """Case-insensitive substring match over title / preview / id / source."""
    q = query.lower()
    return (
        q in (s.get("title") or "").lower()
        or q in (s.get("preview") or "").lower()
        or q in s.get("id", "").lower()
        or q in (s.get("source") or "").lower()
    )


# Layout: [arrow 3] [title/preview flexible] [status 5] [msgs 5]
#         [active 12] [src 6] [id 18]
_FIXED_COLS = 3 + 5 + 2 + 5 + 2 + 12 + 6 + 18 + 6


def _format_row(s: dict, max_x: int) -> str:
    """Format a session row for display."""
    name_width = max(20, max_x - _FIXED_COLS)
    title = (s.get("title") or "").strip()
    preview = (s.get("preview") or "").strip()
    sid = s["id"][:18]
    name = (title or preview)[:name_width] or sid
    return (
        f"{name:<{name_width}}  {_session_status_tag(s.get('_status')):<5}  "
        f"{_msgs_str(s):>5}  {_relative_time(s.get('last_active')):<10}  "
        f"{s.get('source', '')[:6]:<5} {sid}"
    )


class _CursesBrowser:
    """State + render loop for the curses picker. ``run`` is the wrapper target."""

    def __init__(self, curses, sessions, delete_fn):
        self.curses = curses
        self.sessions = sessions
        self.delete_fn = delete_fn  # None => no delete support
        self.result = None
        self.cursor = 0
        self.scroll = 0
        self.search = ""
        self.confirm_delete = None  # session dict pending y/n confirmation
        self.flash = ""  # one-frame notice (e.g. "Deleted.")
        self.filtered = list(sessions)

    # -- helpers -----------------------------------------------------------
    def _pair(self, n, fallback):
        c = self.curses
        return c.color_pair(n) if c.has_colors() else fallback

    def _status_attr(self, status):
        c = self.curses
        if not c.has_colors():
            return c.A_NORMAL
        return {
            "complete": c.color_pair(1),
            "interrupted": c.color_pair(2),
            "error": c.color_pair(5),
            "empty": c.color_pair(4),
        }.get(status or "", c.A_NORMAL)

    def _put(self, stdscr, y, x, text, n, attr):
        try:
            stdscr.addnstr(y, x, text, n, attr)
        except self.curses.error:
            pass

    def _refilter(self, reset_cursor=True):
        self.filtered = ([s for s in self.sessions if _match(s, self.search)] if self.search else list(self.sessions))
        if reset_cursor:
            self.cursor = 0
            self.scroll = 0

    # -- frame ---------------------------------------------------------------
    def _draw(self, stdscr, max_y, max_x):
        c = self.curses
        if self.search:
            header = f"  Browse sessions — filter: {self.search}█"
            header_attr = c.A_BOLD | self._pair(3, 0)
        else:
            header = ("  Browse sessions — ↑↓ navigate  Enter select  Type to filter  Esc quit")
            header_attr = c.A_BOLD | self._pair(2, 0)
        self._put(stdscr, 0, 0, header, max_x - 1, header_attr)

        name_width = max(20, max_x - _FIXED_COLS)
        col_header = (
            f"   {'Title / Preview':<{name_width}}  {'Stat':<5}  "
            f"{'Msgs':>5}  {'Active':<10}  {'Src':<5} {'ID'}"
        )
        self._put(stdscr, 1, 0, col_header, max_x - 1, self._pair(4, c.A_DIM))

        visible_rows = max(max_y - 4, 1)  # header + col header + blank + footer
        filtered = self.filtered
        if not filtered:
            self._put(stdscr, 3, 0, "  No sessions match the filter.", max_x - 1, c.A_DIM)
        else:
            self.cursor = max(min(self.cursor, len(filtered) - 1), 0)
            if self.cursor < self.scroll:
                self.scroll = self.cursor
            elif self.cursor >= self.scroll + visible_rows:
                self.scroll = self.cursor - visible_rows + 1
            for draw_i, i in enumerate(range(self.scroll, min(len(filtered), self.scroll + visible_rows))):
                y = draw_i + 3
                if y >= max_y - 1:
                    break
                s = filtered[i]
                selected = i == self.cursor
                row = (" → " if selected else "   ") + _format_row(s, max_x - 3)
                attr = c.A_BOLD | self._pair(1, 0) if selected else c.A_NORMAL
                try:
                    stdscr.addnstr(y, 0, row, max_x - 1, attr)
                    if not selected:
                        # Recolor the status tag column in place.
                        status = s.get("_status")
                        tag_x = 3 + max(20, (max_x - 3) - _FIXED_COLS) + 2
                        if tag_x + 5 < max_x - 1:
                            stdscr.addnstr(
                                y, tag_x, f"{_session_status_tag(status):<5}", 5,
                                self._status_attr(status),
                            )
                except c.error:
                    pass

        footer_attr = self._pair(4, c.A_DIM)
        if self.confirm_delete is not None:
            label = _label(self.confirm_delete)
            if len(label) > 40:
                label = label[:37] + "..."
            footer = f"  Delete session '{label}'? [y/N]"
            footer_attr = c.A_BOLD | self._pair(5, 0)
        elif self.flash:
            footer = f"  {self.flash}"
            self.flash = ""
        else:
            if filtered:
                footer = f"  {self.cursor + 1}/{len(filtered)} sessions"
                if len(filtered) < len(self.sessions):
                    footer += f" (filtered from {len(self.sessions)})"
            else:
                footer = f"  0/{len(self.sessions)} sessions"
            if self.delete_fn is not None and not self.search:
                footer += "   d delete"
        self._put(stdscr, max_y - 1, 0, footer, max_x - 1, footer_attr)

    # -- keys ----------------------------------------------------------------
    def _handle_key(self, key) -> bool:
        """Apply one keypress; return True when the picker should exit."""
        c = self.curses
        if self.confirm_delete is not None:
            # y/n confirmation mode — only an explicit 'y' deletes.
            target, self.confirm_delete = self.confirm_delete, None
            if key in {ord("y"), ord("Y")}:
                if self.delete_fn(target["id"]):
                    self.sessions[:] = [s for s in self.sessions if s["id"] != target["id"]]
                    self._refilter(reset_cursor=False)
                    self.flash = "Deleted."
                    if not self.sessions:
                        return True
                else:
                    self.flash = "Delete failed."
            return False

        if key == c.KEY_UP:
            if self.filtered:
                self.cursor = (self.cursor - 1) % len(self.filtered)
        elif key == c.KEY_DOWN:
            if self.filtered:
                self.cursor = (self.cursor + 1) % len(self.filtered)
        elif key in {c.KEY_ENTER, 10, 13}:
            if self.filtered:
                self.result = self.filtered[self.cursor]["id"]
            return True
        elif key == 27:  # Esc: first clears the search, second exits
            if not self.search:
                return True
            self.search = ""
            self._refilter()
        elif key in {c.KEY_BACKSPACE, 127, 8}:
            if self.search:
                self.search = self.search[:-1]
                self._refilter()
        elif key == ord("q") and not self.search:
            return True
        elif (
            key == ord("d")
            and not self.search
            and self.delete_fn is not None
            and self.filtered
        ):
            # 'd' only acts as delete when the filter is empty — while a
            # search is active it types into the query below.
            self.confirm_delete = self.filtered[self.cursor]
        elif 32 <= key <= 126:
            self.search += chr(key)
            self._refilter()
        return False

    def run(self, stdscr):
        c = self.curses
        c.curs_set(0)
        if c.has_colors():
            c.start_color()
            c.use_default_colors()
            c.init_pair(1, c.COLOR_GREEN, -1)  # selected
            c.init_pair(2, c.COLOR_YELLOW, -1)  # header
            c.init_pair(3, c.COLOR_CYAN, -1)  # search
            c.init_pair(4, 8 if c.COLORS > 8 else c.COLOR_WHITE, -1)  # dim
            c.init_pair(5, c.COLOR_RED, -1)  # error/delete
        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 5 or max_x < 40:
                try:
                    stdscr.addstr(0, 0, "Terminal too small")
                except c.error:
                    pass
                stdscr.refresh()
                stdscr.getch()
                return
            self._draw(stdscr, max_y, max_x)
            stdscr.refresh()
            if self._handle_key(stdscr.getch()):
                return


def _fallback_picker(sessions: list) -> Optional[str]:
    """Numbered list (Windows without curses, etc.). Same columns, no delete."""
    print("\n  Browse sessions  (enter number to resume, q to cancel)\n")
    for i, s in enumerate(sessions):
        label = _label(s)
        if len(label) > 50:
            label = label[:47] + "..."
        print(
            f"  {i + 1:>3}. {label:<50}  {_session_status_tag(s.get('_status')):<5}  "
            f"{_msgs_str(s):>5}  {_relative_time(s.get('last_active')):<10}  "
            f"{s.get('source', '')[:6]}"
        )

    while True:
        try:
            val = input(f"\n  Select [1-{len(sessions)}]: ").strip()
            if not val or val.lower() in {"q", "quit", "exit"}:
                return None
            idx = int(val) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]["id"]
            print(f"  Invalid selection. Enter 1-{len(sessions)} or q to cancel.")
        except ValueError:
            print("  Invalid input. Enter a number or q to cancel.")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def _session_browse_picker(sessions: list, session_db=None) -> Optional[str]:
    """Interactive curses-based session browser with live search filtering.

    Shows lifecycle status (done / intr / err / empty) and message count per
    session when *session_db* is provided. With a live *session_db*, pressing
    ``d`` on a row (while the search filter is empty) prompts y/n and deletes
    the session via ``SessionDB.delete_session``.

    Returns the selected session ID, or None if cancelled.
    """
    if not sessions:
        print("No sessions found.")
        return None

    _annotate_session_statuses(sessions, session_db)

    def _delete_session(session_id: str) -> bool:
        try:
            from hermes_cli.sessions_cmd import get_hermes_home

            sessions_dir = get_hermes_home() / "sessions"
        except Exception:
            sessions_dir = None
        try:
            return bool(session_db.delete_session(session_id, sessions_dir=sessions_dir))
        except Exception:
            return False

    # Curses first; any failure (no curses module, odd terminal) falls back.
    try:
        import curses

        browser = _CursesBrowser(curses, sessions, _delete_session if session_db is not None else None)
        curses.wrapper(browser.run)
        return browser.result
    except Exception:
        pass

    return _fallback_picker(sessions)
