"""Conservative heredoc masking for shell-command scanners.

Guards that scan raw command text (the foreground background-'&' guard in
``tools/terminal_tool.py``, blocked-command checks, ``cron/lifecycle_guard``)
false-positive on heredoc *bodies*, which are usually inline data. Naively
stripping every body is unsafe the other way (fake ``<<`` in quotes can
swallow a real operator; unquoted bodies expand; ``bash <<'EOF'`` executes).

A body is masked ONLY when: every delimiter on the opener is quoted (no
expansion); every heredoc is terminated by an exact delimiter line; the
opener is a single command (no ``;``/``|``/``&`` and no ``$(...)``, backtick
or process substitution); and the consumer is an allowlisted non-shell
interpreter (``_INERT_HEREDOC_CONSUMER_RE``). Otherwise the command is
returned untouched: a false positive is acceptable, hiding real shell syntax
from a guard is not. Masked bodies become an equal number of newlines so
``re.MULTILINE`` scanning keeps its line structure.

Adapted from Wolfram Ravenwolf's security-hardened rework of PR #63788
(commit 69c7663c6de6b6cb05bf99203fa39673efe01ccf).
"""

from __future__ import annotations

import re

# Non-shell interpreters whose quoted heredoc bodies are program text/data for
# THAT interpreter. Optional VAR=... assignments, ``env`` and a path prefix are
# allowed. Deliberately narrow: anything unmatched keeps its body visible.
_INERT_HEREDOC_CONSUMER_RE = re.compile(
    r"^\s*"
    r"(?:[A-Z_][A-Z0-9_]*=\S+\s+)*"
    r"(?:env\s+)?"
    r"(?:[A-Za-z0-9_./-]+/)?"
    r"(?:python(?:3(?:\.\d+)*)?|osascript|cat)(?=\s|$)",
    re.IGNORECASE,
)


def _span_end(command: str, cursor: int, closer: str) -> int:
    """Index just past the backslash-aware span opened at ``cursor``."""
    end = cursor + 1
    while end < len(command):
        if command[end] == "\\" and end + 1 < len(command):
            end += 2
            continue
        if command[end] == closer:
            return end + 1
        end += 1
    return end


def _mask_simple_quotes(command: str) -> str:
    """Blank inert quoted spans; keep ``$(``/backtick-bearing ones visible."""
    result = []
    cursor = 0
    while cursor < len(command):
        char = command[cursor]
        if char == "'":
            closing = command.find("'", cursor + 1)
            if closing == -1:
                result.append(command[cursor:])
                break
            result.append("''")
            cursor = closing + 1
            continue
        if char == '"':
            end = _span_end(command, cursor, '"')
            if not command[cursor:end].endswith('"'):
                result.append(command[cursor:])
                break
            segment = command[cursor:end]
            result.append(segment if "$(" in segment or "`" in segment else '""')
            cursor = end
            continue
        if char == "`":
            end = _span_end(command, cursor, "`")
            result.append(command[cursor:end])
            cursor = end
            continue
        result.append(char)
        cursor += 1
    return "".join(result)


def _parse_heredoc_operator(command: str, index: int):
    """Parse one ``<<`` opener -> ``(end_index, delimiter, strip_tabs, quoted)`` or None."""
    if not command.startswith("<<", index) or command.startswith("<<<", index):
        return None

    cursor = index + 2
    strip_tabs = False
    if cursor < len(command) and command[cursor] == "-":
        strip_tabs = True
        cursor += 1
    while cursor < len(command) and command[cursor] in " \t":
        cursor += 1
    if cursor >= len(command) or command[cursor] in "\r\n":
        return None

    delimiter: list[str] = []
    quoted = False
    while cursor < len(command):
        char = command[cursor]
        if char.isspace() or char in ";&|<>()":
            break
        if char == "\\":
            if cursor + 1 >= len(command) or command[cursor + 1] in "\r\n":
                return None
            quoted = True
            delimiter.append(command[cursor + 1])
            cursor += 2
            continue
        if char in "'\"":
            quoted = True
            quote = char
            cursor += 1
            while cursor < len(command) and command[cursor] != quote:
                if quote == '"' and command[cursor] == "\\":
                    if cursor + 1 >= len(command):
                        return None
                    following = command[cursor + 1]
                    if following in {"$", "`", '"', "\\", "\n"}:
                        delimiter.append(following)
                        cursor += 2
                        continue
                    # In double quotes, backslash is literal before other chars.
                    delimiter.append("\\")
                    cursor += 1
                    continue
                if command[cursor] in "\r\n":
                    return None
                delimiter.append(command[cursor])
                cursor += 1
            if cursor >= len(command):
                return None
            cursor += 1
            continue
        delimiter.append(char)
        cursor += 1

    if not delimiter and not quoted:
        return None
    return cursor, "".join(delimiter), strip_tabs, quoted


def _scan_heredoc_command_unit(command: str, start: int):
    """Scan one logical command -> ``(end, specs, unknown_operator, has_list_operator)``.

    ``unknown_operator``: an unparseable ``<<`` (caller must fail closed).
    ``has_list_operator``: unquoted ``;``/``|``/``&`` on the opener.
    """
    cursor = start
    quote = None
    comment = False
    specs = []
    unknown_operator = False
    has_list_operator = False

    while cursor < len(command):
        char = command[cursor]
        if comment:
            if char == "\n":
                return cursor, specs, unknown_operator, has_list_operator
            cursor += 1
            continue

        if quote is not None:
            if quote in {'"', "`"} and char == "\\" and cursor + 1 < len(command):
                cursor += 2
                continue
            if char == quote:
                quote = None
            cursor += 1
            continue

        if char == "\\" and cursor + 1 < len(command):
            # Includes line continuations: the logical command keeps going.
            cursor += 2
            continue
        if char in "'\"`":
            quote = char
            cursor += 1
            continue
        if char == "#":
            previous = command[cursor - 1] if cursor > start else ""
            if cursor == start or previous.isspace() or previous in ";&|()":
                comment = True
                cursor += 1
                continue
        if char == "\n":
            return cursor, specs, unknown_operator, has_list_operator
        if command.startswith("<<<", cursor):
            cursor += 3
            continue
        if command.startswith("<<", cursor):
            parsed = _parse_heredoc_operator(command, cursor)
            if parsed is None:
                unknown_operator = True
                cursor += 2
                continue
            cursor, delimiter, strip_tabs, quoted = parsed
            specs.append((delimiter, strip_tabs, quoted))
            continue
        if char in ";|&":
            has_list_operator = True
        cursor += 1

    return len(command), specs, unknown_operator, has_list_operator


def _find_heredoc_close(
    command: str,
    body_start: int,
    delimiter: str,
    strip_tabs: bool,
) -> int | None:
    """Return the position after an exact shell heredoc terminator line."""
    cursor = body_start
    while True:
        newline = command.find("\n", cursor)
        if newline == -1:
            line = command[cursor:]
            after = len(command)
        else:
            line = command[cursor:newline]
            after = newline + 1
        if line.endswith("\r"):
            line = line[:-1]
        candidate = line.lstrip("\t") if strip_tabs else line
        if candidate == delimiter:
            return after
        if newline == -1:
            return None
        cursor = after


def strip_inert_heredoc_bodies(command: str) -> str:
    """Mask heredoc bodies that are provably inert data (see module docstring)."""
    ranges: list[tuple[int, int]] = []
    command_start = 0

    # Runs on every terminal call: skip the state machine when no '<<' exists,
    # and stop scanning once past the last '<<'.
    if "<<" not in command:
        return command
    last_opener_index = command.rfind("<<")

    while command_start < len(command):
        if command_start > last_opener_index:
            break
        command_end, specs, unknown_operator, has_list_operator = (
            _scan_heredoc_command_unit(command, command_start)
        )
        if unknown_operator:
            return command
        if not specs:
            if command_end >= len(command):
                break
            command_start = command_end + 1
            continue
        if command_end >= len(command):
            # Opener with no body line: unterminated — leave visible.
            return command

        body_cursor = command_end + 1
        body_ranges: list[tuple[int, int]] = []
        unterminated = False
        for delimiter, strip_tabs, _quoted in specs:
            close_end = _find_heredoc_close(
                command,
                body_cursor,
                delimiter,
                strip_tabs,
            )
            if close_end is None:
                unterminated = True
                break
            body_ranges.append((body_cursor, close_end))
            body_cursor = close_end
        if unterminated:
            return command

        if all(quoted for _delimiter, _strip_tabs, quoted in specs) and not has_list_operator:
            masked_opener = _mask_simple_quotes(command[command_start:command_end])
            nested_scope = any(m in masked_opener for m in ("$(", "`", "<(", ">("))
            if not nested_scope and _INERT_HEREDOC_CONSUMER_RE.search(masked_opener):
                ranges.extend(body_ranges)
        command_start = body_cursor

    if not ranges:
        return command
    # Single-pass rebuild: ranges are sorted and non-overlapping.
    parts: list[str] = []
    previous = 0
    for start, end in ranges:
        parts.append(command[previous:start])
        parts.append("\n" * command.count("\n", start, end))
        previous = end
    parts.append(command[previous:])
    return "".join(parts)
