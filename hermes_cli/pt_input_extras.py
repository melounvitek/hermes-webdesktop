"""Augmentations to prompt_toolkit's input-parsing tables."""

from __future__ import annotations

# kitty CSI-u ORs lock-key state into the modifier parameter of every key
# event while a lock is on: CapsLock=64, NumLock=128, both=192 (#88221,
# #89651).  Every fixed-modifier CSI-u (and legacy CSI-tilde / CSI-letter)
# registration therefore needs lock-offset twins, or those events leak into
# the prompt as literal text.  The xterm modifyOtherKeys ``ESC[27;N;CP~``
# encoding never carries lock bits, so it never gets the twins.
_LOCK_BIT_OFFSETS = (0, 64, 128, 192)


def _lock_variants(modifier: int) -> tuple[int, ...]:
    """Return ``modifier`` plus its CapsLock/NumLock/both twins."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS)


def _lock_twins(modifier: int) -> tuple[int, ...]:
    """Return only the lock twins of ``modifier`` (never the base value)."""
    return tuple(modifier + off for off in _LOCK_BIT_OFFSETS[1:])


def _clear_vt100_prefix_cache() -> None:
    """Drop prompt_toolkit's memoized prefix-match answers after mutating ``ANSI_SEQUENCES``.

    The cache is module-global and lazily filled per prefix, so parsers created before an install
    would keep stale ``False`` answers and misparse newly registered sequences.
    """
    try:
        from prompt_toolkit.input.vt100_parser import (
            _IS_PREFIX_OF_LONGER_MATCH_CACHE,
        )
        _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()
    except Exception:
        pass


def _pt_tables():
    """Return ``(ANSI_SEQUENCES, Keys)`` or ``None`` when prompt_toolkit is unavailable."""
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return None
    return ANSI_SEQUENCES, Keys


def _register(table: dict, aliases: dict, *, overwrite: bool) -> int:
    """Install ``aliases`` into ``table``; return the number of entries changed.

    ``overwrite=True`` replaces differing entries; ``overwrite=False`` behaves like ``setdefault``
    so existing/user registrations win. Clears the VT100 prefix cache when anything changed, since
    new longer sequences can flip "is this a prefix of a longer match?" answers the parser cached.
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
    (Shift+Space → ``' '``, Shift+letter → the uppercase letter, keypad digits → ``'0'``..``'9'``,
    keypad operators).
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
    except Exception:
        return 0

    if getattr(
        _vt100_mod.Vt100Parser._call_handler, "_hermes_char_data_normalized", False
    ):
        return 0

    _orig_call_handler = _vt100_mod.Vt100Parser._call_handler

    def _patched_call_handler(self, key, insert_text):
        # A single plain character (not a Keys member, not a tuple) mapped
        # from an extended sequence must carry the mapped character as its
        # data — self-insert inserts event.data and the raw CSI would leak.
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
    """Map <modifier>+Enter (Kitty CSI-u ``ESC[13;<m>u`` plus lock-bit twins, xterm
    ``ESC[27;<m>;13~`` / ``;13u``) to (Escape, ControlM) so the Alt+Enter newline handler fires.

    Stock prompt_toolkit maps the tilde form to plain ControlM (i.e. Shift+Enter == Enter, the very
    bug this fixes), so those keys are overwritten unconditionally; other modifier variants are
    untouched.
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
    """Map Shift+Enter sequences to (Escape, ControlM) so the Alt+Enter newline handler fires.

    macOS Terminal and stock Windows Terminal send the same byte for Enter and Shift+Enter, so
    nothing can be done for them here.
    """
    return _install_enter_alias(2)


def install_ctrl_enter_alias() -> int:
    """Map Ctrl+Enter sequences to (Escape, ControlM) so the Alt+Enter newline handler fires.

    Without the alias, Kitty/mintty/xterm users over SSH get a raw CSI sequence inserted as text.
    """
    return _install_enter_alias(5)


def install_cmd_backspace_alias() -> int:
    """Map Cmd+Backspace / Cmd+ForwardDelete to prompt_toolkit's readline kill bindings.

    Terminals that rewrite Cmd+Backspace to Ctrl+U already work; Kitty/modifyOtherKeys report Cmd
    as the super bit (8), yielding unmapped sequences that insert literally. Cmd+Backspace ->
    ControlU (``ESC[127;9u``, ``;10u``, ``ESC[27;9;127~``); Cmd+ForwardDelete -> ControlK via
    the CSI tilde form ``ESC[3;9~`` / ``;10~`` since forward-delete is not a CSI-u codepoint.
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


def install_modify_other_keys_aliases() -> int:
    """Map modifyOtherKeys-2 / Kitty CSI-u Ctrl/Alt+key sequences to their raw-byte ``Keys``.

    Once ``modifyOtherKeys=2`` is pushed (to distinguish Shift+Enter) the terminal re-encodes
    EVERY Ctrl combo as ``ESC[27;5;<cp>~``; stock prompt_toolkit maps only Ctrl+Enter, so
    Ctrl+A/C/D/E/K/R/U/W/Z leak as text. Installs Ctrl/Alt/Shift letters, digits, symbols,
    multi-modifier combos, CapsLock/NumLock lock-bit variants, CSI-u Esc, modified
    Enter/Tab/Backspace/Space, and Kitty functional keys. Uses ``setdefault`` so existing
    mappings (incl. the Shift/Ctrl+Enter aliases) are never overwritten.
    """
    tables = _pt_tables()
    if tables is None:
        return 0
    ANSI_SEQUENCES, Keys = tables

    # Everything below is collected into ``aliases`` (first writer wins, matching setdefault
    # order) and installed once at the end.
    aliases: dict[str, object] = {}

    def _put(seq: str, key_val: object) -> None:
        aliases.setdefault(seq, key_val)

    # -- Ctrl+letter / Ctrl+digit / Ctrl+symbol → Keys.Control* ----
    # codepoint -> Keys value.  The raw control byte for Ctrl+<ch> is
    # chr(ord(ch) & 0x1f) (i.e. ord(ch) - 96 for lowercase).  We map the
    # *extended* sequence to the same Keys value that the raw byte maps to,
    # so prompt_toolkit's existing key bindings fire identically.
    ctrl_key_map: dict[int, object] = {}

    # a-z: Ctrl+A = \x01 = Keys.ControlA, ..., Ctrl+Z = \x1a = Keys.ControlZ
    # Symbols that produce control chars:
    # Ctrl+@   (64)  = \x00 = Keys.ControlAt
    # Ctrl+[   (91)  = \x1b = Keys.Escape
    # Ctrl+\   (92)  = \x1c = Keys.ControlBackslash
    # Ctrl+]   (93)  = \x1d = Keys.ControlSquareClose
    # Ctrl+^   (94)  = \x1e = Keys.ControlCircumflex
    # Ctrl+_   (95)  = \x1f = Keys.ControlUnderscore
    # Ctrl+Space(32) = \x00 = Keys.ControlAt (prompt_toolkit maps \x00 → ControlAt)
    letters = range(ord('a'), ord('z') + 1)
    for codepoint in (*letters, 64, 91, 92, 93, 94, 95, 32):
        existing = ANSI_SEQUENCES.get(chr(codepoint & 0x1F))
        if existing is not None:
            ctrl_key_map[codepoint] = existing

    # 0-9: Ctrl+digit codepoints don't have a useful raw-byte mapping
    # (e.g. chr(ord('0') & 0x1F) = 0x10 = ControlP, not Control0), so map
    # them directly to Keys.Control0..Keys.Control9.
    for d in range(10):
        ctrl_key_map[ord('0') + d] = getattr(Keys, f"Control{d}")

    # Kitty CSI-u encodes CapsLock/NumLock state as extra modifier bits
    # (caps=64, num=128) ORed into the parameter: with NumLock on, Ctrl+C
    # arrives as ESC[99;133u (5 + 128) instead of ESC[99;5u. Terminals
    # that report these bits (kitty, ghostty) break every key combo while
    # a lock is on (#89651) unless the lock variants are mapped too. The
    # xterm modifyOtherKeys encoding never carries the lock bits, so only
    # the CSI-u form needs them.
    def _install_paired(modifier: int, mapping: dict) -> None:
        """Install both modifyOtherKeys (ESC[27;N;CP~) and CSI-u (ESC[CP;Nu) mappings for the given
        modifier and codepoint→key mapping.
        """
        for codepoint, key_val in mapping.items():
            if modifier != 1:
                _put(f"\x1b[27;{modifier};{codepoint}~", key_val)
            for mod in _lock_variants(modifier):
                _put(f"\x1b[{codepoint};{mod}u", key_val)

    # Ctrl+letter / Ctrl+digit / Ctrl+symbol (modifier 5)
    _install_paired(5, ctrl_key_map)

    # -- Alt+letter → (Escape, <letter>) ----
    # Under modifyOtherKeys, Alt+a = ESC[27;3;97~. Without mapping, this
    # leaks as literal text. prompt_toolkit handles bare Alt+letter as
    # (Escape, <letter>), so we map the extended sequences to the same tuple.
    #
    # -- Shift+letter → uppercase letter ----
    # Under modifyOtherKeys=2, some terminals re-encode Shift+a as
    # ESC[27;2;97~. Without mapping, this leaks as literal escape +
    # "[27;2;97~" in the prompt buffer — the "caps locked" / "every key
    # combo is broken" symptom (#87711).
    # Map Shift+letter to the uppercase character so typing works normally.
    # This is safe across all Latin keyboard layouts: Shift always uppercases
    # letters.  Shift+digit symbols are layout-specific (US: '!', AZERTY: '¹',
    # etc.) so they are NOT mapped here — if the terminal sends those under
    # modifyOtherKeys, they will leak, but that's better than wrong input.
    # Map both the lowercase and uppercase codepoints — some terminals send
    # the already-shifted codepoint (65 for 'A') with modifier=2.
    #
    # -- Multi-modifier letters: Shift+Alt (4), Ctrl+Shift (6),
    # Ctrl+Alt (7), Ctrl+Alt+Shift (8) ----
    # The Kitty protocol always reports the UNSHIFTED codepoint; some
    # modifyOtherKeys emitters send the shifted one — map both cases.
    # Ctrl-bearing combos normalize onto the Ctrl key (Alt adds an Escape
    # prefix), Shift+Alt onto (Escape, UPPER) — the same normalization
    # dte/kakoune apply to these protocols. Without these, Ctrl+Shift+R
    # etc. leak as literal text under either protocol.
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
    _install_paired(3, alt_map)
    _install_paired(2, shift_map)
    _install_paired(4, shift_alt_map)
    _install_paired(6, ctrl_shift_map)
    _install_paired(7, ctrl_alt_map)
    _install_paired(8, ctrl_alt_map)  # Ctrl+Alt+Shift — same normalization

    # -- The Esc KEY under Kitty disambiguate mode: ESC[27u (+ modifiers) --
    # Disambiguate mode reports the Esc key as CSI-u so it is
    # distinguishable from the ESC byte that starts escape sequences
    # (#56684 — previously leaked "[27u" as literal text into the prompt).
    # Modifiers run from 1 to 16: kitty reports Cmd as the super bit
    # (mod 9+) — same reason install_cmd_backspace_alias maps 9/10 — and
    # the lock-bit variants of the modifier-less form (1+64/128/192) are
    # how a lone Esc keypress arrives with a lock on. Lock bits (caps/num)
    # get the same variant treatment as _install_paired.
    _put("\x1b[27u", Keys.Escape)
    for m in range(1, 17):
        for mod in _lock_variants(m):
            _put(f"\x1b[27;{mod}u", Keys.Escape)

    # -- Modified Enter / Tab / Backspace / Space ----
    # Shift+Enter / Ctrl+Enter are installed by install_shift_enter_alias /
    # install_ctrl_enter_alias (which run first and win via setdefault).
    _install_paired(2, {
        9: Keys.BackTab,        # Shift+Tab — same as the legacy ESC[Z
        127: Keys.ControlH,     # Shift+Backspace — plain backspace
        32: " ",                # Shift+Space — still a space (#86866)
    })
    _install_paired(3, {
        13: (Keys.Escape, Keys.ControlM),   # Alt+Enter — newline tuple
        127: (Keys.Escape, Keys.ControlH),  # Alt+Backspace — backward-kill-word
        32: (Keys.Escape, " "),             # Alt+Space
    })
    _install_paired(5, {
        9: Keys.ControlI,                   # Ctrl+Tab — degrade to Tab
        127: (Keys.Escape, Keys.ControlH),  # Ctrl+Backspace — backward-kill-word,
                                            # matching Ink TUI + Desktop (#78285)
    })

    # -- Unmodified keys with a lock bit set (kitty modifier 1 = "none") --
    # With a lock on, kitty stamps the lock bit onto keys pressed with NO
    # real modifier too, so plain Backspace arrives as ESC[127;129u
    # (1 + 128) rather than \x7f. _install_paired(1, ...) registers the
    # bare mod-1 spelling and its lock twins. Only keys kitty CSI-u-encodes
    # on their own are listed; plain text characters are still delivered
    # as UTF-8, lock bits or not.
    _install_paired(1, {
        9: Keys.ControlI,     # Tab
        13: Keys.ControlM,    # Enter
        32: " ",              # Space
        127: Keys.ControlH,   # Backspace
    })

    # -- Lock-key modifier bits (NumLock=128, CapsLock=64) on the legacy
    # CSI-letter / CSI-tilde forms kitty keeps using under the disambiguate
    # push: kitty encodes lock state into the modifier parameter, so a
    # plain Down with NumLock on arrives as ESC[1;129B (NumLock), ESC[1;65B
    # (CapsLock) or ESC[1;193B (both) instead of the legacy ESC[B — and a
    # modified one shifts the same way (Alt+Left → ESC[1;131D). Those fall
    # through the parser and leak as literal text ("[1;129B") in the input
    # line. Derive the lock twins from whatever the table already maps for
    # the base modifier (stock prompt_toolkit entries included), so every
    # modifier the terminal can report keeps working under a lock.
    for m in range(1, 17):
        # CSI-letter navigation: Up/Down/Right/Left/End/Home + F1-F4
        for trailer in "ABCDFHPQRS":
            base_seq = f"\x1b[1;{m}{trailer}" if m > 1 else f"\x1b[{trailer}"
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None and m == 1:
                # Plain F1-F4 live in the table as SS3 (ESC O P) forms.
                key = ANSI_SEQUENCES.get(f"\x1bO{trailer}")
            if key is None:
                continue
            for mod in _lock_twins(m):
                _put(f"\x1b[1;{mod}{trailer}", key)
        # CSI-tilde navigation: Insert/Delete/PageUp/PageDown/Home/End
        for num in range(1, 9):
            base_seq = f"\x1b[{num};{m}~" if m > 1 else f"\x1b[{num}~"
            key = ANSI_SEQUENCES.get(base_seq)
            if key is None:
                continue
            for mod in _lock_twins(m):
                _put(f"\x1b[{num};{mod}~", key)

    # -- Kitty functional keys (Private Use Area codepoints) ----
    # kitty emits these CSI-u encodings even in LEGACY mode for keys that
    # have no legacy encoding, so unmapped they leak as literal text in any
    # kitty session regardless of which modes were pushed.
    functional_map: dict[int, object] = {}
    for d in range(10):                       # KP_0..KP_9 → digits
        functional_map[57399 + d] = str(d)
    functional_map.update({                   # KP operators / punctuation
        57409: ".", 57410: "/", 57411: "*", 57412: "-",
        57413: "+", 57414: Keys.ControlM, 57415: "=", 57416: ",",
    })
    functional_map.update({                   # KP navigation → non-keypad keys
        57417: Keys.Left, 57418: Keys.Right, 57419: Keys.Up,
        57420: Keys.Down, 57421: Keys.PageUp, 57422: Keys.PageDown,
        57423: Keys.Home, 57424: Keys.End, 57425: Keys.Insert,
        57426: Keys.Delete,
    })
    for n in range(13, 25):                   # F13..F24
        functional_map[57376 + (n - 13)] = getattr(Keys, f"F{n}")
    # No prompt_toolkit equivalent (lock keys, PrintScreen, Menu, F25-F35,
    # KP_BEGIN, media keys, bare modifier events): consume as Ignore
    # instead of leaking literal text.
    for code in (
        list(range(57358, 57364))       # locks, PrintScreen, Pause, Menu
        + list(range(57388, 57399))     # F25..F35
        + [57427]                       # KP_BEGIN
        + list(range(57428, 57455))     # media keys + modifier key events
    ):
        functional_map.setdefault(code, Keys.Ignore)
    for code, key_val in functional_map.items():
        _put(f"\x1b[{code}u", key_val)
        # Lock twins: with a lock on these arrive as ESC[<code>;129u etc.
        for mod in _lock_twins(1):
            _put(f"\x1b[{code};{mod}u", key_val)

    return _register(ANSI_SEQUENCES, aliases, overwrite=False)


def install_ignored_terminal_sequences() -> int:
    """Map terminal noise sequences to ``Keys.Ignore`` so the VT100 parser consumes them.

    Covers focus reports ``ESC[I`` / ``ESC[O``, which Ghostty, iTerm2 and some xterms emit on
    tab/window switches; unmapped, prompt_toolkit inserts ``[I``/``[O`` into the buffer. Parser-
    level handling beats post-hoc regex stripping because the bytes never reach the buffer.
    ``setdefault`` lets user/downstream registrations win.
    """
    tables = _pt_tables()
    if tables is None:
        return 0
    seqs, keys = tables
    return _register(seqs, {"\x1b[I": keys.Ignore, "\x1b[O": keys.Ignore}, overwrite=False)
