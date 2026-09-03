"""Augmentations to prompt_toolkit's input-parsing tables."""

from __future__ import annotations

# kitty CSI-u ORs lock-key state into the modifier parameter of every key event while a lock is
# on: CapsLock=64, NumLock=128, both=192. Every fixed-modifier CSI-u (and legacy CSI-tilde /
# CSI-letter) registration therefore needs lock-offset twins, or those events leak into the prompt
# as literal text. The xterm modifyOtherKeys ``ESC[27;N;CP~`` encoding never carries lock bits.
_LOCK_BIT_OFFSETS = (0, 64, 128, 192)


def _lock_variants(modifier: int) -> tuple[int, ...]:
    """``modifier`` plus its CapsLock/NumLock/both twins."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS)


def _lock_twins(modifier: int) -> tuple[int, ...]:
    """Only the lock twins of ``modifier`` (never the base value)."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS[1:])


def _clear_vt100_prefix_cache() -> None:
    """Drop prompt_toolkit's memoized prefix-match answers after mutating ``ANSI_SEQUENCES``.

    The cache is module-global and lazily filled per prefix, so parsers created before an install
    would keep stale ``False`` answers and misparse newly registered sequences.
    """
    try:
        from prompt_toolkit.input.vt100_parser import _IS_PREFIX_OF_LONGER_MATCH_CACHE
        _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()
    except Exception:
        pass


def _pt_tables():
    """``(ANSI_SEQUENCES, Keys)`` or ``None`` when prompt_toolkit is unavailable."""
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return None
    return ANSI_SEQUENCES, Keys


def _register(table: dict, aliases: dict, *, overwrite: bool) -> int:
    """Install ``aliases`` into ``table``; return the number of entries changed.

    ``overwrite=True`` replaces differing entries; ``overwrite=False`` behaves like ``setdefault``
    so existing/user registrations win. Clears the VT100 prefix cache when anything changed.
    """
    changed = 0
    for seq, key in aliases.items():
        if (table.get(seq) != key) if overwrite else (seq not in table):
            table[seq] = key
            changed += 1
    if changed:
        _clear_vt100_prefix_cache()
    return changed


def install_keypress_data_normalization() -> int:
    """Normalize KeyPress data for extended-key aliases that map to a single plain character
    (Shift+Space → ``' '``, Shift+letter → uppercase, keypad digits/operators).
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
    except Exception:
        return 0

    _orig_call_handler = _vt100_mod.Vt100Parser._call_handler
    if getattr(_orig_call_handler, "_hermes_char_data_normalized", False):
        return 0

    def _patched_call_handler(self, key, insert_text):
        # A single plain character mapped from an extended sequence must carry the mapped
        # character as its data — self-insert inserts event.data and the raw CSI would leak.
        if (
            isinstance(key, str)
            and len(key) == 1
            and not isinstance(key, _PtKeys)
            and isinstance(insert_text, str)
            and insert_text.startswith("\x1b")
        ):
            insert_text = key
        return _orig_call_handler(self, key, insert_text)

    _patched_call_handler._hermes_char_data_normalized = True
    _vt100_mod.Vt100Parser._call_handler = _patched_call_handler
    return 1


def _install_enter_alias(modifier: int) -> int:
    """Map <modifier>+Enter (Kitty CSI-u ``ESC[13;<m>u`` plus lock twins, xterm ``ESC[27;<m>;13~``
    / ``;13u``) to (Escape, ControlM) so the Alt+Enter newline handler fires.

    Stock prompt_toolkit maps the tilde form to plain ControlM (i.e. Shift+Enter == Enter, the very
    bug this fixes), so these keys are overwritten unconditionally.
    """
    tables = _pt_tables()
    if tables is None:
        return 0
    seqs, keys = tables
    alt_enter = (keys.Escape, keys.ControlM)
    aliases = {f"\x1b[13;{m}u": alt_enter for m in _lock_variants(modifier)}
    aliases[f"\x1b[27;{modifier};13~"] = alt_enter
    aliases[f"\x1b[27;{modifier};13u"] = alt_enter
    return _register(seqs, aliases, overwrite=True)


def install_shift_enter_alias() -> int:
    """Map Shift+Enter to (Escape, ControlM). macOS Terminal and stock Windows Terminal send the
    same byte for Enter and Shift+Enter, so nothing can be done for them here.
    """
    return _install_enter_alias(2)


def install_ctrl_enter_alias() -> int:
    """Map Ctrl+Enter to (Escape, ControlM); otherwise Kitty/mintty/xterm over SSH insert raw CSI."""
    return _install_enter_alias(5)


def install_cmd_backspace_alias() -> int:
    """Map Cmd+Backspace -> ControlU and Cmd+ForwardDelete -> ControlK.

    Kitty/modifyOtherKeys report Cmd as the super bit (8), yielding unmapped sequences that insert
    literally. Forward-delete is not a CSI-u codepoint, so it uses the CSI tilde form ``ESC[3;9~``.
    """
    tables = _pt_tables()
    if tables is None:
        return 0
    seqs, keys = tables
    aliases: dict[str, object] = {}
    for base in (9, 10):  # super / super+shift
        for mod in _lock_variants(base):
            aliases[f"\x1b[127;{mod}u"] = keys.ControlU
            aliases[f"\x1b[3;{mod}~"] = keys.ControlK
    aliases["\x1b[27;9;127~"] = keys.ControlU
    return _register(seqs, aliases, overwrite=True)


# Kitty functional keys (Private Use Area codepoints) that have prompt_toolkit equivalents.
# kitty emits these CSI-u encodings even in LEGACY mode, so unmapped they leak as literal text.
_KITTY_FUNCTIONAL_NAMED = {
    57409: ".", 57410: "/", 57411: "*", 57412: "-", 57413: "+", 57414: "ControlM",  # KP ops
    57415: "=", 57416: ",",
    57417: "Left", 57418: "Right", 57419: "Up", 57420: "Down", 57421: "PageUp",  # KP nav
    57422: "PageDown", 57423: "Home", 57424: "End", 57425: "Insert", 57426: "Delete",
}
# No prompt_toolkit equivalent: locks/PrintScreen/Pause/Menu, F25-F35, KP_BEGIN, media keys and
# bare modifier events — consumed as Ignore instead of leaking literal text.
_KITTY_FUNCTIONAL_IGNORED = (*range(57358, 57364), *range(57388, 57399), 57427, *range(57428, 57455))


def _kitty_functional_map(Keys) -> dict[int, object]:
    fm: dict[int, object] = {57399 + d: str(d) for d in range(10)}  # KP_0..KP_9
    fm.update({cp: getattr(Keys, v) if v[0].isupper() else v for cp, v in _KITTY_FUNCTIONAL_NAMED.items()})
    fm.update({57376 + (n - 13): getattr(Keys, f"F{n}") for n in range(13, 25)})  # F13..F24
    for code in _KITTY_FUNCTIONAL_IGNORED:
        fm.setdefault(code, Keys.Ignore)
    return fm


def install_modify_other_keys_aliases() -> int:
    """Map modifyOtherKeys-2 / Kitty CSI-u Ctrl/Alt+key sequences to their raw-byte ``Keys``.

    Once ``modifyOtherKeys=2`` is pushed (to distinguish Shift+Enter) the terminal re-encodes
    EVERY Ctrl combo as ``ESC[27;5;<cp>~``; stock prompt_toolkit maps only Ctrl+Enter, so
    Ctrl+A/C/D/... leak as text. Installs Ctrl/Alt/Shift letters, digits, symbols, multi-modifier
    combos, lock-bit variants, CSI-u Esc, modified Enter/Tab/Backspace/Space and Kitty functional
    keys. ``setdefault`` semantics: existing mappings (incl. the Shift/Ctrl+Enter aliases) win.
    """
    tables = _pt_tables()
    if tables is None:
        return 0
    ANSI_SEQUENCES, Keys = tables

    # Collected first-writer-wins (matching setdefault order), installed once at the end.
    aliases: dict[str, object] = {}
    _put = aliases.setdefault

    def _install_paired(modifier: int, mapping: dict) -> None:
        """Both modifyOtherKeys (ESC[27;N;CP~, never for mod 1) and CSI-u (ESC[CP;Nu + lock twins)."""
        for codepoint, key_val in mapping.items():
            if modifier != 1:
                _put(f"\x1b[27;{modifier};{codepoint}~", key_val)
            for mod in _lock_variants(modifier):
                _put(f"\x1b[{codepoint};{mod}u", key_val)

    # Ctrl+<ch>: the extended sequence maps to whatever Keys value the raw control byte
    # chr(ord(ch) & 0x1f) already maps to, so existing bindings fire identically. Covers a-z and
    # the control-producing symbols @ [ \ ] ^ _ and Space (\x00 -> ControlAt).
    letters = range(ord('a'), ord('z') + 1)
    ctrl_key_map: dict[int, object] = {}
    for codepoint in (*letters, 64, 91, 92, 93, 94, 95, 32):
        existing = ANSI_SEQUENCES.get(chr(codepoint & 0x1F))
        if existing is not None:
            ctrl_key_map[codepoint] = existing
    # Ctrl+digit has no useful raw byte (chr(ord('0') & 0x1F) is ControlP), so map directly.
    for d in range(10):
        ctrl_key_map[ord('0') + d] = getattr(Keys, f"Control{d}")
    _install_paired(5, ctrl_key_map)

    # Letter combos. Alt+a -> (Escape, 'a') like bare Alt. Shift+a -> 'A' (safe on every Latin
    # layout; Shift+digit symbols are layout-specific and deliberately NOT mapped — leaking beats
    # wrong input). Kitty reports the UNSHIFTED codepoint, some modifyOtherKeys emitters the shifted
    # one — map both. Ctrl-bearing combos normalize onto the Ctrl key (Alt adds an Escape prefix),
    # Shift+Alt onto (Escape, UPPER) — the same normalization dte/kakoune apply.
    alt_map: dict[int, tuple] = {}
    shift_map: dict[int, str] = {}
    shift_alt_map: dict[int, tuple] = {}
    ctrl_shift_map: dict[int, object] = {}
    ctrl_alt_map: dict[int, tuple] = {}
    for ch in letters:
        upper_char = chr(ch - 32)
        alt_map[ch] = (Keys.Escape, chr(ch))
        alt_map[ch - 32] = (Keys.Escape, upper_char)
        ctrl_key = ctrl_key_map.get(ch)
        for cp in (ch, ch - 32):
            shift_map[cp] = upper_char
            shift_alt_map[cp] = (Keys.Escape, upper_char)
            if ctrl_key is not None:
                ctrl_shift_map[cp] = ctrl_key
                ctrl_alt_map[cp] = (Keys.Escape, ctrl_key)
    for modifier, mapping in (
        (3, alt_map), (2, shift_map), (4, shift_alt_map), (6, ctrl_shift_map),
        (7, ctrl_alt_map), (8, ctrl_alt_map),  # Ctrl+Alt+Shift — same normalization
    ):
        _install_paired(modifier, mapping)

    # The Esc KEY under Kitty disambiguate mode: ESC[27u (+ modifiers 1-16 incl. super 9+, and
    # lock twins of the modifier-less form, which is how a lone Esc arrives with a lock on).
    _put("\x1b[27u", Keys.Escape)
    for m in range(1, 17):
        for mod in _lock_variants(m):
            _put(f"\x1b[27;{mod}u", Keys.Escape)

    # Modified Enter/Tab/Backspace/Space (Shift/Ctrl+Enter are owned by the enter aliases, which run
    # first and win). Modifier 1 = unmodified keys kitty CSI-u-encodes on their own when a lock bit
    # is set (plain Backspace arrives as ESC[127;129u rather than \x7f).
    alt_enter = (Keys.Escape, Keys.ControlM)
    alt_backspace = (Keys.Escape, Keys.ControlH)  # backward-kill-word, matching Ink TUI + Desktop
    for modifier, mapping in (
        (2, {9: Keys.BackTab, 127: Keys.ControlH, 32: " "}),
        (3, {13: alt_enter, 127: alt_backspace, 32: (Keys.Escape, " ")}),
        (5, {9: Keys.ControlI, 127: alt_backspace}),  # Ctrl+Tab degrades to Tab
        (1, {9: Keys.ControlI, 13: Keys.ControlM, 32: " ", 127: Keys.ControlH}),
    ):
        _install_paired(modifier, mapping)

    # Lock twins for the legacy CSI-letter / CSI-tilde forms kitty keeps using under the
    # disambiguate push (Down with NumLock on = ESC[1;129B; Alt+Left = ESC[1;131D). Derived from
    # whatever the table already maps for the base modifier, stock entries included.
    for m in range(1, 17):
        for trailer in "ABCDFHPQRS":  # Up/Down/Right/Left/End/Home + F1-F4
            base_seq = f"\x1b[1;{m}{trailer}" if m > 1 else f"\x1b[{trailer}"
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None and m == 1:
                key = ANSI_SEQUENCES.get(f"\x1bO{trailer}")  # plain F1-F4 live as SS3 forms
            if key is None:
                continue
            for mod in _lock_twins(m):
                _put(f"\x1b[1;{mod}{trailer}", key)
        for num in range(1, 9):  # Insert/Delete/PageUp/PageDown/Home/End
            base_seq = f"\x1b[{num};{m}~" if m > 1 else f"\x1b[{num}~"
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None:
                continue
            for mod in _lock_twins(m):
                _put(f"\x1b[{num};{mod}~", key)

    for code, key_val in _kitty_functional_map(Keys).items():
        _put(f"\x1b[{code}u", key_val)
        for mod in _lock_twins(1):  # with a lock on these arrive as ESC[<code>;129u etc.
            _put(f"\x1b[{code};{mod}u", key_val)

    return _register(ANSI_SEQUENCES, aliases, overwrite=False)


def install_ignored_terminal_sequences() -> int:
    """Map focus reports ``ESC[I`` / ``ESC[O`` (Ghostty, iTerm2, some xterms) to ``Keys.Ignore``.

    Parser-level handling beats post-hoc regex stripping because the bytes never reach the buffer.
    ``setdefault`` lets user/downstream registrations win.
    """
    tables = _pt_tables()
    if tables is None:
        return 0
    seqs, keys = tables
    return _register(seqs, {"\x1b[I": keys.Ignore, "\x1b[O": keys.Ignore}, overwrite=False)
