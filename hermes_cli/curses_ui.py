"""Shared curses-based UI components for Hermes CLI.

Used by `hermes tools` and `hermes skills` for interactive checklists. Provides a curses multi-
select with keyboard navigation, plus a text-based numbered fallback for terminals without curses
support.
"""
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Protocol, Sequence, Set, Tuple, Union

from hermes_cli.colors import Colors, color

# Rich radiolist rows: (text, style). style is None | "yellow" | "dim".
# Plain ``str`` items remain fully supported.
RadioItem = Union[str, Sequence[Tuple[str, Optional[str]]]]


_NO_REPLAY = object()


@dataclass(frozen=True)
class MenuNavigationStart:
    """Navigation instructions returned when a scoped menu begins."""

    allow_back: bool = False
    replay_value: object = _NO_REPLAY

    @property
    def should_replay(self) -> bool:
        return self.replay_value is not _NO_REPLAY


class MenuNavigationEvent(str, Enum):
    BEGIN = "begin"
    RESOLVE = "resolve"
    CANCEL = "cancel"
    BACK = "back"


class MenuNavigationHandler(Protocol):
    """Typed contract between shared menus and a scoped flow controller."""

    def __call__(self, event: MenuNavigationEvent, value: object = None) -> MenuNavigationStart | None: ...


_MENU_NAVIGATION_HANDLER: ContextVar[MenuNavigationHandler | None] = ContextVar(
    "hermes_menu_navigation_handler", default=None
)
_NUMBERED_BACK_ENABLED: ContextVar[bool] = ContextVar("hermes_numbered_back_enabled", default=False)


def set_menu_navigation_handler(handler: MenuNavigationHandler) -> Token[MenuNavigationHandler | None]:
    """Scope setup-style cancel/back behavior to the current CLI invocation."""
    return _MENU_NAVIGATION_HANDLER.set(handler)


def reset_menu_navigation_handler(token: Token[MenuNavigationHandler | None]) -> None:
    """Restore the menu navigation handler active before ``token``."""
    _MENU_NAVIGATION_HANDLER.reset(token)


def _notify_scoped_navigation(event: MenuNavigationEvent) -> None:
    """Notify an active menu flow that a text fallback was interrupted (CANCEL) or requested BACK."""
    handler = _MENU_NAVIGATION_HANDLER.get()
    if handler is not None:
        handler(event)


class _NumberedNavigation(Enum):
    CANCEL = "cancel"
    BACK = "back"


_NAV_ABORT = object()


def _read_numbered_choice(prompt_text: str) -> int | None | object:
    """Read a numbered fallback choice as a 0-based index.

    Returns ``None`` for empty input and ``_NAV_ABORT`` when the prompt was cancelled, backed out
    of, interrupted, or given a non-integer (scoped navigation is notified for cancel/back).
    """
    try:
        val = _read_numbered_input(prompt_text)
    except (KeyboardInterrupt, EOFError):
        _notify_scoped_navigation(MenuNavigationEvent.CANCEL)
        return _NAV_ABORT
    if isinstance(val, _NumberedNavigation):
        _notify_scoped_navigation(MenuNavigationEvent(val.value))
        return _NAV_ABORT
    if not val.strip():
        return None
    idx = _parse_int(val.strip(), default=None)
    return _NAV_ABORT if idx is None else idx - 1


def _read_numbered_input(prompt_text: str) -> str | _NumberedNavigation:
    """Read a numbered fallback choice with setup navigation key bindings.

    Ordinary numbered menus retain their historical ``input()`` behavior. During setup/model flows,
    prompt_toolkit supplies portable Escape, Ctrl+C, and Left bindings on POSIX and native Windows
    when curses is unavailable.
    """
    if _MENU_NAVIGATION_HANDLER.get() is None:
        return input(prompt_text)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    # Setup can be invoked without importing the classic CLI, which normally
    # installs Ghostty/Kitty CSI-u aliases at process startup.
    from hermes_cli.pt_input_extras import install_modify_other_keys_aliases

    install_modify_other_keys_aliases()
    bindings = KeyBindings()

    @bindings.add(Keys.Escape)
    @bindings.add(Keys.ControlC)
    def _cancel(event) -> None:
        event.app.exit(result=_NumberedNavigation.CANCEL)

    if _NUMBERED_BACK_ENABLED.get():

        @bindings.add(Keys.Left)
        def _back(event) -> None:
            event.app.exit(result=_NumberedNavigation.BACK)

    return PromptSession().prompt(ANSI(prompt_text), key_bindings=bindings)


def radio_item_plain(item: RadioItem) -> str:
    """Flatten a radiolist item to searchable/plain display text."""
    if isinstance(item, str):
        return item
    return "".join(text for text, _style in item)


def _curses_style_attr(curses, style: Optional[str], *, is_cursor: bool):
    """Map a segment style to a curses attribute."""
    has_colors = curses.has_colors()
    if is_cursor:
        return curses.A_BOLD | (curses.color_pair(1) if has_colors else 0)
    if style == "yellow" and has_colors:
        return curses.color_pair(2)
    if style == "dim":
        attr = curses.A_DIM
        if has_colors:
            # Pair 3 is the dim-gray status color (extra_color_pairs).
            try:
                attr |= curses.color_pair(3)
            except curses.error:
                pass
        return attr
    return curses.A_NORMAL


def _addnstr(stdscr, y: int, x: int, text: str, n: int, attr) -> None:
    """``stdscr.addnstr`` that swallows ``curses.error`` (drawing past the screen edge)."""
    import curses

    try:
        stdscr.addnstr(y, x, text, n, attr)
    except curses.error:
        pass


def _draw_title_and_hint(stdscr, title: str, hint: str, max_x: int, *, hint_row: int = 1) -> None:
    """Draw the bold/yellow menu title on row 0 and the dim key hint on ``hint_row``."""
    import curses

    hattr = curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
    _addnstr(stdscr, 0, 0, title, max_x - 1, hattr)
    _addnstr(stdscr, hint_row, 0, hint, max_x - 1, curses.A_DIM)


def _draw_plain_row(stdscr, y: int, line: str, max_x: int, *, is_cursor: bool) -> None:
    """Draw a plain menu row, bold green when it is the cursor row."""
    import curses

    _addnstr(stdscr, y, 0, line, max_x - 1, _curses_style_attr(curses, None, is_cursor=is_cursor))


def _draw_segments(stdscr, y: int, x: int, segments, max_x: int) -> None:
    """Draw ``(text, attr)`` segments left to right from column ``x``, clipped at the screen edge."""
    col = x
    for text, attr in segments:
        remaining = max_x - 1 - col
        if remaining <= 0:
            break
        chunk = text[:remaining]
        _addnstr(stdscr, y, col, chunk, remaining, attr)
        col += len(chunk)


def _draw_description_line(stdscr, y: int, text: str, max_x: int) -> None:
    """Draw a description line, highlighting ★ in yellow when colors exist."""
    import curses

    star_attr = curses.color_pair(2) if curses.has_colors() else curses.A_NORMAL
    segments = []
    for i, part in enumerate(text.split("★")):
        if i:
            segments.append(("★", star_attr))
        if part:
            segments.append((part, curses.A_NORMAL))
    _draw_segments(stdscr, y, 0, segments, max_x)


def _draw_radio_item(stdscr, y: int, x: int, item: RadioItem, max_x: int, *, is_cursor: bool) -> None:
    """Draw a plain or segmented radiolist item starting at column ``x``."""
    import curses

    if isinstance(item, str):
        attr = _curses_style_attr(curses, None, is_cursor=is_cursor)
        _addnstr(stdscr, y, x, item, max(0, max_x - 1 - x), attr)
        return

    _draw_segments(
        stdscr, y, x,
        ((text, _curses_style_attr(curses, style, is_cursor=is_cursor)) for text, style in item),
        max_x,
    )


_WORD_BOUNDARY = frozenset("-_/. ")


def _is_boundary(target: str, index: int) -> bool:
    """True if position ``index`` in ``target`` starts a word.

    Mirrors ``isBoundary`` in the TS scorer: start-of-string, after a separator char, or a
    lower->upper camelCase transition.
    """
    if index == 0:
        return True
    prev = target[index - 1]
    if prev in _WORD_BOUNDARY:
        return True
    # camelCase / lower->upper transition (e.g. the `O` in `gptO`).
    cur = target[index]
    return prev == prev.lower() and cur != cur.lower() and cur == cur.upper()


def _token_score(orig: str, lower: str, token: str) -> float | None:
    """Score one token against a target. None if the token isn't a subsequence.

    Faithful port of ``fuzzyScore`` in ui-tui and web ``fuzzy.ts`` so all three surfaces rank
    model ids identically: contiguous runs, word-boundary/first-char starts, prefixes and exact
    matches outrank scattered hits. Matching runs against ``lower`` while boundary detection
    uses ``orig`` so the camelCase rule works, exactly as in the TS scorer.
    """
    score = 0.0
    prev = -1
    search_from = 0
    positions: list[int] = []

    for ch in token:
        idx = lower.find(ch, search_from)
        if idx < 0:
            return None
        positions.append(idx)
        score += 1
        if prev >= 0 and idx == prev + 1:
            score += 5
        elif prev >= 0:
            score -= min(idx - prev - 1, 3)
        if _is_boundary(orig, idx):
            score += 3
        if idx == 0:
            score += 5
        prev = idx
        search_from = idx + 1

    # Prefix bonus: the token matched a contiguous prefix of the target.
    if positions and positions[0] == 0 and positions[-1] == len(positions) - 1:
        score += 8

    # Exact full match dominates everything else.
    if lower == token:
        score += 20

    # Slightly prefer shorter targets when scores are otherwise close.
    score -= len(lower) * 0.01

    return score


def _fuzzy_score(label: str, query: str) -> float | None:
    """Aggregate score for a multi-token query (AND). None if any token fails.

    Mirrors ``fuzzyScoreMulti`` in the TS scorer: every whitespace-separated token must match; per-
    token scores are summed.
    """
    lower = label.lower()
    total = 0.0
    for token in query.lower().split():
        token_score = _token_score(label, lower, token)
        if token_score is None:
            return None
        total += token_score
    return total


def _filter_indices(items: List[str], query: str) -> List[int]:
    """Return item indices matching *query*, ranked best-first.

    An empty query keeps every item in original order. Otherwise items are filtered to fuzzy matches
    and sorted by score descending, ties broken by original index so equal-scoring rows keep their
    catalog order.
    """
    q = query.strip()
    if not q:
        return list(range(len(items)))
    scored = [(i, score) for i, label in enumerate(items) if (score := _fuzzy_score(label, q)) is not None]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [i for i, _ in scored]


@dataclass
class _SearchState:
    """Mutable search state shared by curses picker loops."""

    active: bool = False
    query: str = ""


def _reconcile_cursor(filtered: List[int], cursor: int) -> tuple[int, int]:
    """Return ``(cursor, cursor_pos)`` inside the filtered index list."""
    if not filtered:
        return cursor, 0
    if cursor not in filtered:
        cursor = filtered[0]
    return cursor, filtered.index(cursor)


def _move_filtered_cursor(filtered: List[int], cursor: int, cursor_pos: int, delta: int) -> int:
    """Move through the filtered index list, wrapping like the legacy menus."""
    return filtered[(cursor_pos + delta) % len(filtered)] if filtered else cursor


def _scroll_for_cursor(scroll_offset: int, cursor_pos: int, visible_rows: int, total_rows: int) -> int:
    """Clamp scroll offset so the cursor remains visible."""
    visible_rows = max(1, visible_rows)
    if cursor_pos < scroll_offset:
        scroll_offset = cursor_pos
    elif cursor_pos >= scroll_offset + visible_rows:
        scroll_offset = cursor_pos - visible_rows + 1
    return max(0, min(scroll_offset, max(0, total_rows - visible_rows)))


def _handle_active_search_key(curses_mod, key: int, search: _SearchState) -> tuple[bool, bool, bool]:
    """Handle a key while the search prompt is active."""
    if not search.active:
        return False, False, False

    if key == 27:
        # Esc stops search AND clears the query, restoring the full list (so a
        # no-match filter can't strand the user on an empty list). Signals
        # `changed` when there was a query so the driver resets scroll/cursor.
        had_query = bool(search.query)
        search.active = False
        search.query = ""
        return True, False, had_query

    if key in (curses_mod.KEY_ENTER, 10, 13):
        return True, True, False

    if key in (curses_mod.KEY_BACKSPACE, 127, 8):
        search.query = search.query[:-1]
    elif key == 21:  # Ctrl+U
        search.query = ""
    elif 32 <= key < 127:  # printable ASCII; avoids Latin-1 mojibake from 128-255
        search.query += chr(key)
    else:
        return False, False, False
    return True, False, True


def flush_stdin() -> None:
    """Flush any stray bytes from the stdin input buffer.

    Must be called after ``curses.wrapper()`` returns, and before the next ``input()`` /
    ``getpass.getpass()`` call. ``curses.endwin()`` restores the terminal but does NOT drain the OS
    input buffer.
    """
    try:
        if sys.stdin.isatty():
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


# Normalized menu actions returned by ``read_menu_key``.  Using sentinels keeps
# every menu's key-handling branch identical and free of raw escape-byte logic.
NAV_UP = "up"
NAV_DOWN = "down"
NAV_BACK = "back"
NAV_SELECT = "select"
NAV_TOGGLE = "toggle"
NAV_CANCEL = "cancel"
NAV_INTERRUPT = "interrupt"
NAV_NONE = "none"


def read_menu_key(stdscr) -> str:
    """Read one keypress and normalize it to a menu action.

    Returns one of the ``NAV_*`` constants. A lone ESC (no continuation byte within a short window)
    is the only thing that maps to ``NAV_CANCEL`` via the escape path; ``q`` also cancels. Unknown
    sequences map to ``NAV_NONE`` so the caller simply ignores them rather than misfiring.
    """
    return _decode_menu_key(stdscr, stdscr.getch())


def _parse_int(value: str, default=0):
    try:
        return int(value)
    except ValueError:
        return default


def _parse_csi_u_key(raw: str) -> tuple[int, int, int] | None:
    """Parse a Kitty/CSI-u key into ``(codepoint, modifier, event_type)``; None without a codepoint."""
    parts = raw.split(";")
    codepoint = _parse_int(parts[0].split(":", 1)[0])
    if not codepoint:
        return None
    mod_fields = parts[1].split(":") if len(parts) > 1 else []
    modifier = _parse_int(mod_fields[0], 1) if mod_fields else 1
    event_type = _parse_int(mod_fields[1], 1) if len(mod_fields) > 1 else 1
    return codepoint, modifier, event_type


def _parse_csi_numbers(raw: str) -> list[int]:
    """Parse semicolon-delimited CSI numbers for modifyOtherKeys."""
    return [_parse_int(part.split(":", 1)[0]) for part in raw.split(";")]


def _enhanced_key_action(codepoint: int, modifier: int = 1) -> str:
    """Map CSI-u/modifyOtherKeys codepoints to setup menu actions."""
    if codepoint in (10, 13):
        return NAV_SELECT
    if codepoint == 27:
        return NAV_CANCEL
    if codepoint == 32:
        return NAV_TOGGLE

    # CSI-u encodes Ctrl+C as codepoint `c` plus the Ctrl modifier. Lock-state
    # bits may be added to the modifier, so inspect the Ctrl bit rather than
    # matching only the canonical value 5.
    has_ctrl = bool((max(1, modifier) - 1) & 4)
    if codepoint == 3 or (codepoint in (ord("c"), ord("C")) and has_ctrl):
        return NAV_INTERRUPT
    return NAV_NONE


def _read_csi_tail(stdscr) -> tuple[str, int | None]:
    """Read CSI/SS3 parameter bytes through the final byte."""
    raw: list[str] = []
    for _ in range(32):
        value = stdscr.getch()
        if 0x40 <= value <= 0x7E:
            return "".join(raw), value
        if not 0x20 <= value <= 0x3F:
            break
        raw.append(chr(value))
    return "".join(raw), None


_CSI_FINAL_NAV = {ord("A"): NAV_UP, ord("k"): NAV_UP, ord("B"): NAV_DOWN, ord("j"): NAV_DOWN, ord("D"): NAV_BACK}


def _decode_menu_key(stdscr, key: int) -> str:
    """Normalize an already-read keypress to a menu action.

    Split out from ``read_menu_key`` so search-aware loops can peek the raw key (e.g. to catch
    ``/``) before falling back to nav decoding.
    """
    import curses

    if key in (curses.KEY_UP, ord("k")):
        return NAV_UP
    if key in (curses.KEY_DOWN, ord("j")):
        return NAV_DOWN
    if key == curses.KEY_LEFT:
        return NAV_BACK
    if key == 3:  # Ctrl+C in curses raw/cbreak mode.
        return NAV_INTERRUPT
    if key in (curses.KEY_ENTER, 10, 13):
        return NAV_SELECT
    if key == ord(" "):
        return NAV_TOGGLE
    if key == ord("q"):
        return NAV_CANCEL

    if key == 27:  # ESC — could be a lone ESC (cancel) or an escape sequence.
        # Wait briefly for a continuation byte.  On slow PTYs (SSH/tmux) the
        # bytes of an arrow key can arrive across separate reads, so a tiny
        # timeout avoids misreading a split sequence as a bare ESC.
        try:
            stdscr.timeout(60)
            nxt = stdscr.getch()
            if nxt == -1:
                return NAV_CANCEL  # genuine lone ESC

            if nxt in (ord("["), ord("O")):  # CSI / SS3 introducer
                raw_params, final = _read_csi_tail(stdscr)
                if final in _CSI_FINAL_NAV:
                    return _CSI_FINAL_NAV[final]
                if final == ord("u"):
                    enhanced = _parse_csi_u_key(raw_params)
                    if enhanced is not None:
                        codepoint, modifier, event_type = enhanced
                        if event_type == 3:  # key release
                            return NAV_NONE
                        return _enhanced_key_action(codepoint, modifier)
                if final == ord("~"):
                    params = _parse_csi_numbers(raw_params)
                    if len(params) >= 3 and params[0] == 27:
                        return _enhanced_key_action(params[2], params[1])
                return NAV_NONE
            # ESC followed by some other byte we don't handle — swallow it.
            return NAV_NONE
        finally:
            stdscr.timeout(-1)  # restore blocking mode

    return NAV_NONE


# Sentinel: an on_action reducer returns this to mean "keep looping" (the
# keypress changed cursor/selection state but didn't resolve the menu).
_KEEP = object()


def _run_curses_menu(
    *,
    initial_cursor,
    item_count,
    draw_header,
    draw_row,
    on_action,
    reserve_bottom=1,
    draw_footer=None,
    extra_color_pairs=False,
    fallback,
    cancel_value,
    searchable=False,
    search_labels=None,
):
    """Shared curses single-/multi-select event loop.

    Owns the non-TTY guard, ``curses.wrapper`` setup, the per-frame clear/refresh cycle, scroll
    math, key dispatch with cursor wrap, and the KeyboardInterrupt / curses-unavailable
    fallback; per-menu behavior comes in as callbacks so rendering stays byte-identical to the
    old hand-rolled loops. ``draw_row`` always receives the ORIGINAL item index (filtering
    doesn't change rendering); ``on_action`` returns ``_KEEP`` to continue or any other value to
    resolve the menu; a ``draw_footer`` row budget must be included in ``reserve_bottom``; with
    ``searchable``, ``/`` filters over ``search_labels`` (length must equal ``item_count``) and
    results are original indices.
    """
    navigation_handler = _MENU_NAVIGATION_HANDLER.get()

    def _notify(event, *value):
        if navigation_handler is not None:
            navigation_handler(event, *value)

    navigation_start = navigation_handler(MenuNavigationEvent.BEGIN) if navigation_handler else None
    if navigation_start is not None and not isinstance(navigation_start, MenuNavigationStart):
        raise TypeError("menu navigation 'begin' must return MenuNavigationStart")
    allow_back = bool(navigation_start and navigation_start.allow_back)
    if navigation_start is not None and navigation_start.should_replay:
        _notify(MenuNavigationEvent.RESOLVE, navigation_start.replay_value)
        return navigation_start.replay_value

    # Non-TTY (piped/redirected stdin): curses and input() both hang or spin,
    # so return the cancel value directly — matching the pre-refactor guard in
    # each menu (the numbered fallback is only for curses errors on a real TTY).
    if not sys.stdin.isatty():
        return cancel_value

    use_search = searchable and search_labels is not None and len(search_labels) == item_count

    def _run_fallback():
        back_token = _NUMBERED_BACK_ENABLED.set(allow_back)
        try:
            result = fallback()
        finally:
            _NUMBERED_BACK_ENABLED.reset(back_token)
        _notify(MenuNavigationEvent.RESOLVE, result)
        return result

    try:
        import curses
    except ImportError:
        return _run_fallback()

    try:
        result_holder = [_KEEP]

        def _resolve(outcome) -> bool:
            """Record a non-``_KEEP`` outcome; True when the menu is done."""
            if outcome is _KEEP:
                return False
            _notify(MenuNavigationEvent.RESOLVE, outcome)
            result_holder[0] = outcome
            return True

        def _draw(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                if extra_color_pairs:
                    curses.init_pair(3, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)
            cursor = initial_cursor
            scroll_offset = 0
            search = _SearchState()

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()

                filtered = _filter_indices(search_labels, search.query) if use_search else list(range(item_count))
                cursor, cursor_pos = _reconcile_cursor(filtered, cursor)

                items_start = draw_header(stdscr, max_y, max_x, search=search, back_enabled=allow_back)

                visible_rows = max(1, max_y - items_start - reserve_bottom)
                scroll_offset = _scroll_for_cursor(scroll_offset, cursor_pos, visible_rows, len(filtered))

                if use_search and search.query and not filtered:
                    _addnstr(stdscr, items_start, 0, "  No matches", max_x - 1, curses.A_DIM)

                for draw_i, i in enumerate(filtered[scroll_offset : scroll_offset + visible_rows]):
                    y = draw_i + items_start
                    if y >= max_y - reserve_bottom:
                        break
                    draw_row(stdscr, y, i, i == cursor, max_x)

                if draw_footer is not None:
                    draw_footer(stdscr, max_y, max_x)

                stdscr.refresh()

                key = stdscr.getch()
                if use_search and search.active and key == 27:
                    # Ghostty/Kitty enhanced keys also begin with ESC.
                    # Decode the full sequence before treating a genuine
                    # Escape as "stop search"; otherwise Enter/Left/Ctrl+C
                    # lose their tail while the search prompt is active.
                    action = _decode_menu_key(stdscr, key)
                    if action == NAV_CANCEL:
                        search.active = False
                        search.query = ""
                        scroll_offset = 0
                        continue
                    if action == NAV_NONE:
                        continue
                elif use_search and search.active:
                    # Active search consumes query-editing keys; nav keys
                    # fall through to be decoded below.
                    handled, confirm, changed = _handle_active_search_key(curses, key, search)
                    if changed:
                        scroll_offset = 0
                        cursor, cursor_pos = _reconcile_cursor(
                            _filter_indices(search_labels, search.query), cursor
                        )
                    if confirm:
                        if filtered and _resolve(on_action(NAV_SELECT, cursor)):
                            return
                        continue
                    if handled:
                        continue
                    action = _decode_menu_key(stdscr, key)
                elif use_search and key == ord("/"):
                    search.active = True
                    continue
                else:
                    action = _decode_menu_key(stdscr, key)

                if action == NAV_UP:
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, -1)
                elif action == NAV_DOWN:
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, 1)
                elif action in (NAV_SELECT, NAV_TOGGLE, NAV_CANCEL, NAV_INTERRUPT) or (
                    action == NAV_BACK and allow_back
                ):
                    if action == NAV_SELECT and use_search and not filtered:
                        continue
                    if action in (NAV_CANCEL, NAV_INTERRUPT):
                        _notify(MenuNavigationEvent.CANCEL)
                    elif action == NAV_BACK:
                        _notify(MenuNavigationEvent.BACK)
                    if _resolve(on_action(action, cursor)):
                        return

        curses.wrapper(_draw)
        flush_stdin()
        return result_holder[0] if result_holder[0] is not _KEEP else cancel_value

    except KeyboardInterrupt:
        _notify(MenuNavigationEvent.CANCEL)
        return cancel_value
    except curses.error:
        return _run_fallback()


def curses_checklist(
    title: str,
    items: List[str],
    selected: Set[int],
    *,
    cancel_returns: Set[int] | None = None,
    status_fn: Optional[Callable[[Set[int]], str]] = None,
) -> Set[int]:
    """Curses multi-select checklist. Returns set of selected indices.

    ``cancel_returns`` (default: the original *selected*) is returned on ESC/q.
    ``status_fn(chosen)`` renders on the bottom row for live aggregate info such as token
    estimates.
    """
    if cancel_returns is None:
        cancel_returns = set(selected)

    chosen = set(selected)

    def _draw_row(stdscr, y, i, is_cursor, max_x):
        check = "✓" if i in chosen else " "
        arrow = "→" if is_cursor else " "
        _draw_plain_row(stdscr, y, f" {arrow} [{check}] {items[i]}", max_x, is_cursor=is_cursor)

    def _draw_footer(stdscr, max_y, max_x):
        import curses
        status_text = status_fn(chosen)
        if status_text:
            # Right-align on the bottom row
            sx = max(0, max_x - len(status_text) - 1)
            sattr = curses.A_DIM | (curses.color_pair(3) if curses.has_colors() else 0)
            _addnstr(stdscr, max_y - 1, sx, status_text, max_x - sx - 1, sattr)

    def _on_action(action, cursor):
        if action == NAV_TOGGLE:
            chosen.symmetric_difference_update({cursor})
            return _KEEP
        if action == NAV_SELECT:
            return set(chosen)
        return cancel_returns  # NAV_CANCEL

    return _run_curses_menu(
        initial_cursor=0,
        item_count=len(items),
        draw_header=_simple_header(title, "SPACE toggle  ENTER confirm", "ESC cancel", False),
        draw_row=_draw_row,
        on_action=_on_action,
        reserve_bottom=(2 if status_fn else 1),
        draw_footer=_draw_footer if status_fn else None,
        extra_color_pairs=bool(status_fn),
        fallback=lambda: _numbered_fallback(title, items, selected, cancel_returns, status_fn),
        cancel_value=cancel_returns,
    )


def _search_hint(search, searchable: bool, confirm: str, cancel: str, back_enabled: bool) -> str:
    """Key-hint row for menus, swapping to the search prompt while ``/`` is active."""
    if searchable and search is not None and search.active:
        hint = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop"
    else:
        hint = f"  \u2191\u2193 navigate  {confirm}  {'/ search  ' if searchable else ''}{cancel}"
    if back_enabled:
        hint += "  \u2190 previous"
    return hint


def _simple_header(title: str, confirm: str, cancel: str, searchable: bool):
    """``draw_header`` callback: title on row 0, key hint on row 1, items start on row 3."""

    def _draw_header(stdscr, max_y, max_x, search=None, back_enabled=False):
        hint = _search_hint(search, searchable, confirm, cancel, back_enabled)
        _draw_title_and_hint(stdscr, title, hint, max_x)
        return 3

    return _draw_header


def curses_radiolist(
    title: str,
    items: List[RadioItem],
    selected: int = 0,
    *,
    cancel_returns: int | None = None,
    description: str | None = None,
    searchable: bool = False,
    search_labels: List[str] | None = None,
) -> int:
    """Curses single-select radio list. Returns the selected index.

    Items are plain strings or ``(text, style)`` segment sequences
    (``None``/``"yellow"``/``"dim"``); the cursor row is forced green, unselected rows honor
    segment styles. ``description`` is shown between title and list so context survives the
    curses screen clear. With ``searchable``, ``/`` filters over ``search_labels`` (default:
    display labels) and the returned value is always the ORIGINAL item index, never a filtered
    row position.
    """
    if cancel_returns is None:
        cancel_returns = selected

    desc_lines = description.splitlines() if description else []

    plain_labels = [radio_item_plain(item) for item in items] if searchable else None

    def _draw_header(stdscr, max_y, max_x, search=None, back_enabled=False):
        # Description lines — paint ★ yellow so the sale legend matches rows.
        row = 1
        for dline in desc_lines[: max(0, max_y - 2)]:
            _draw_description_line(stdscr, row, dline, max_x)
            row += 1

        hint = _search_hint(search, searchable, "ENTER/SPACE select", "ESC cancel", back_enabled)
        _draw_title_and_hint(stdscr, title, hint, max_x, hint_row=row)
        # One blank row between the hint and the item list.
        return row + 2

    def _draw_row(stdscr, y, i, is_cursor, max_x):
        radio = "\u25cf" if i == selected else "\u25cb"
        arrow = "\u2192" if is_cursor else " "
        prefix = f" {arrow} ({radio}) "
        _draw_plain_row(stdscr, y, prefix, max_x, is_cursor=is_cursor)
        _draw_radio_item(
            stdscr, y, len(prefix), items[i], max_x, is_cursor=is_cursor
        )

    def _on_action(action, cursor):
        if action in (NAV_SELECT, NAV_TOGGLE):
            return cursor
        return cancel_returns  # NAV_CANCEL

    return _run_curses_menu(
        initial_cursor=selected,
        item_count=len(items),
        draw_header=_draw_header,
        draw_row=_draw_row,
        on_action=_on_action,
        reserve_bottom=1,
        # Dim gray (pair 3) for unselected "was …" sale chrome.
        extra_color_pairs=True,
        fallback=lambda: _radio_numbered_fallback(title, items, selected, cancel_returns),
        cancel_value=cancel_returns,
        searchable=searchable,
        search_labels=(list(search_labels) if search_labels is not None else plain_labels) if searchable else None,
    )


def format_radio_item_ansi(item: RadioItem) -> str:
    """Apply ANSI colors to a rich radiolist item (numbered fallback / prints)."""
    if isinstance(item, str):
        return item
    return "".join(
        color(text, _ANSI_STYLE[style]) if style in _ANSI_STYLE else text for text, style in item
    )


_ANSI_STYLE = {"yellow": Colors.YELLOW, "dim": Colors.DIM}


def _radio_numbered_fallback(
    title: str,
    items: List[RadioItem],
    selected: int,
    cancel_returns: int,
) -> int:
    """Text-based numbered fallback for radio selection."""
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Select by number, Enter to confirm.\n", Colors.DIM))

    for i, label in enumerate(items):
        marker = color("(\u25cf)", Colors.GREEN) if i == selected else "(\u25cb)"
        print(f"  {marker} {i + 1:>2}. {format_radio_item_ansi(label)}")
    print()
    idx = _read_numbered_choice(color(f"  Choice [default {selected + 1}]: ", Colors.DIM))
    if idx is _NAV_ABORT:
        return cancel_returns
    return idx if idx is not None and 0 <= idx < len(items) else selected


def curses_single_select(
    title: str,
    items: List[str],
    default_index: int = 0,
    *,
    cancel_label: str = "Cancel",
    searchable: bool = False,
) -> int | None:
    """Curses single-select menu. Returns selected index or None on cancel.

    When ``searchable`` is true, ``/`` opens a type-to-filter prompt; the returned value is always
    the original item index (or None for cancel).
    """
    all_items = list(items) + [cancel_label]
    cancel_idx = len(items)

    def _draw_row(stdscr, y, i, is_cursor, max_x):
        arrow = "→" if is_cursor else " "
        _draw_plain_row(stdscr, y, f" {arrow} {all_items[i]}", max_x, is_cursor=is_cursor)

    def _on_action(action, cursor):
        if action == NAV_SELECT:
            # Selecting the synthetic cancel row resolves to None, mirroring
            # the old post-loop ``>= cancel_idx`` guard.
            return None if cursor >= cancel_idx else cursor
        if action in (NAV_CANCEL, NAV_INTERRUPT):
            return None
        return _KEEP  # NAV_TOGGLE — no-op for this menu

    return _run_curses_menu(
        initial_cursor=min(default_index, len(all_items) - 1),
        item_count=len(all_items),
        draw_header=_simple_header(title, "ENTER confirm", "ESC/q cancel", searchable),
        draw_row=_draw_row,
        on_action=_on_action,
        reserve_bottom=1,
        fallback=lambda: _numbered_single_fallback(title, all_items, cancel_idx),
        cancel_value=None,
        searchable=searchable,
        search_labels=list(all_items) if searchable else None,
    )


def _numbered_single_fallback(
    title: str,
    items: List[str],
    cancel_idx: int,
) -> int | None:
    """Text-based numbered fallback for single-select."""
    print(f"\n  {title}\n")
    for i, label in enumerate(items, 1):
        print(f"  {i}. {label}")
    print()
    idx = _read_numbered_choice(f"  Choice [1-{len(items)}]: ")
    return idx if isinstance(idx, int) and 0 <= idx < min(len(items), cancel_idx) else None


def _numbered_fallback(
    title: str,
    items: List[str],
    selected: Set[int],
    cancel_returns: Set[int],
    status_fn: Optional[Callable[[Set[int]], str]] = None,
) -> Set[int]:
    """Text-based toggle fallback for terminals without curses."""
    chosen = set(selected)
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))

    while True:
        for i, label in enumerate(items):
            marker = color("[✓]", Colors.GREEN) if i in chosen else "[ ]"
            print(f"  {marker} {i + 1:>2}. {label}")
        status_text = status_fn(chosen) if status_fn else ""
        if status_text:
            print(color(f"\n  {status_text}", Colors.DIM))
        print()
        idx = _read_numbered_choice(color("  Toggle # (or Enter to confirm): ", Colors.DIM))
        if idx is _NAV_ABORT:
            return cancel_returns
        if idx is None:
            return chosen
        if 0 <= idx < len(items):
            chosen.symmetric_difference_update({idx})
        print()
