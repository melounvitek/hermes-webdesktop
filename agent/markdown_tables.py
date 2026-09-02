"""CJK/wide-character-aware re-alignment of model-emitted markdown tables.

Models pad tables assuming one cell per character; CJK glyphs and most emoji
take two, so body rows drift right on real terminals. This rebuilds padding
with ``wcwidth.wcswidth`` while preserving pipes/dashes so the table still reads
as plain text in ``strip``/unrendered modes (Rich already aligns CJK itself).

Deliberately conservative: only contiguous ``| ... |`` blocks with a divider are
rewritten; everything else passes through; single-line/mid-stream fragments are
left alone (callers buffer rows and flush complete blocks).  ``wcwidth`` returns
``-1`` for some emoji+variation-selector sequences (``⚠️``); those clamp to 0 —
a 1-cell drift on that glyph beats widening every table that contains one.
"""

from __future__ import annotations

import re
from typing import List

from wcwidth import wcswidth

__all__ = [
    "is_table_divider",
    "looks_like_table_row",
    "realign_markdown_tables",
    "split_table_row",
]


_DIVIDER_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
_MIN_COL_WIDTH = 3  # matches the divider's minimum dash run.


def _disp_width(s: str) -> int:
    """``wcswidth`` clamped to >= 0 (it returns -1 for control/unknown sequences)."""
    w = wcswidth(s)
    return w if w > 0 else 0


def _pad_to_width(s: str, target: int) -> str:
    return s + " " * max(0, target - _disp_width(s))


def split_table_row(row: str) -> List[str]:
    """Split ``| a | b | c |`` into ``["a", "b", "c"]`` with trims."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_table_divider(row: str) -> bool:
    """True when ``row`` is a markdown table separator line."""
    cells = split_table_row(row)
    return len(cells) > 1 and all(_DIVIDER_CELL_RE.match(c) for c in cells)


def looks_like_table_row(row: str) -> bool:
    """True when ``row`` could plausibly be a markdown table row.

    Intentionally permissive for streaming callers deciding whether to buffer a
    line: the realigner only rewrites divider-backed blocks, so a false positive
    at most delays printing one line.  A leading pipe is the strongest signal;
    without it we accept >= 2 pipes so models that omit the leading pipe still match.
    """
    if "|" not in row:
        return False
    stripped = row.strip()
    if not stripped:
        return False
    return stripped.startswith("|") or stripped.count("|") >= 2


def _render_block(rows: List[List[str]], available_width: int | None = None) -> List[str]:
    """Render ``rows`` (header + body, divider implied) at uniform widths.

    When the horizontal table would exceed ``available_width`` fall back to a
    vertical key-value rendering: terminal soft-wrap mid-cell destroys alignment
    visually even when the bytes are perfectly padded.
    """
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(_MIN_COL_WIDTH, *(_disp_width(r[c]) for r in rows)) for c in range(ncols)]

    # `| ` + cell + ` ` per column, plus the closing `|`.
    horizontal_width = sum(widths) + 3 * ncols + 1
    if available_width is not None and horizontal_width > max(available_width, 20):
        return _render_vertical(rows, ncols, available_width)

    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(_pad_to_width(c, widths[k]) for k, c in enumerate(cells)) + " |"

    out = [_row(rows[0]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(_row(r) for r in rows[1:])
    return out


def _hard_break(word: str, w: int) -> List[str]:
    """Split a single over-wide word into display-width-``w`` chunks."""
    out: List[str] = []
    buf = ""
    bw = 0
    for ch in word:
        cw = _disp_width(ch) or 1
        if bw + cw > w and buf:
            out.append(buf)
            buf = ch
            bw = cw
        else:
            buf += ch
            bw += cw
    if buf:
        out.append(buf)
    return out


def _wrap_to_width(text: str, width: int) -> List[str]:
    """Soft-wrap ``text`` at word boundaries to ``width`` display cells.

    Words wider than ``width`` are hard-broken.  Empty input yields a single
    empty string so the caller's row count stays predictable.
    """
    if width <= 0 or not text:
        return [text]
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = ""
    current_w = 0

    def _start(word: str, ww: int) -> None:
        nonlocal current, current_w
        if ww <= width:
            current, current_w = word, ww
        else:
            pieces = _hard_break(word, width)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""
            current_w = _disp_width(current)

    for word in words:
        ww = _disp_width(word)
        if not current:
            _start(word, ww)
        elif current_w + 1 + ww <= width:
            current += " " + word
            current_w += 1 + ww
        else:
            lines.append(current)
            _start(word, ww)
    if current:
        lines.append(current)
    return lines or [""]


def _render_vertical(rows: List[List[str]], ncols: int, available_width: int) -> List[str]:
    """Render a too-wide table as ``Header: value`` blocks (Claude Code's narrow fallback).

    Each body row becomes one block with continuation lines indented two spaces,
    blocks separated by a thin ``─`` rule; every line stays under ``available_width``.
    """
    if not rows:
        return []
    headers = rows[0] + [""] * (ncols - len(rows[0]))
    labels = [h or f"Column {i + 1}" for i, h in enumerate(headers)]
    sep_width = max(20, min(40, available_width - 2)) if available_width else 30
    separator = "─" * sep_width
    indent = "  "
    cont_budget = max(10, available_width - _disp_width(indent))

    out: List[str] = []
    for ri, row in enumerate(rows[1:]):
        if ri > 0:
            out.append(separator)
        for ci in range(ncols):
            label = labels[ci]
            value = row[ci] if ci < len(row) else ""
            if not value:
                out.append(f"{label}:")
                continue
            wrapped = _wrap_to_width(value, max(10, available_width - _disp_width(label) - 2))
            out.append(f"{label}: {wrapped[0]}")
            if len(wrapped) > 1:
                # Re-flow continuation text at the wider continuation budget.
                for cl in _wrap_to_width(" ".join(wrapped[1:]), cont_budget):
                    if cl.strip():
                        out.append(f"{indent}{cl}")
    return out


def realign_markdown_tables(text: str, available_width: int | None = None) -> str:
    """Rewrite every ``| ... |`` + divider block with wcwidth-aware padding.

    Non-table lines are returned verbatim, so this is safe on arbitrary prose.
    With ``available_width`` (terminal cells), tables wider than that render as
    vertical key-value pairs instead of soft-wrapping mid-cell.
    """
    if "|" not in text:
        return text

    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # A table starts with a header row whose next line is a divider.
        if "|" in line and i + 1 < n and is_table_divider(lines[i + 1]):
            header = split_table_row(line)
            body: List[List[str]] = []
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                if not is_table_divider(lines[j]):
                    body.append(split_table_row(lines[j]))
                j += 1
            if any(c for c in header) or body:
                out.extend(_render_block([header] + body, available_width))
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)
