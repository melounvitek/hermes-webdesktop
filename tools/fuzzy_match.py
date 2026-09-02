#!/usr/bin/env python3
"""Fuzzy find-and-replace for LLM-generated edits.

Tries an ordered chain of increasingly permissive matching strategies (see
:mod:`tools.fuzzy_match_strategies`) so whitespace, indentation, escaping and
Unicode drift in tool-call arguments still land on the intended region.

    new_content, match_count, strategy, error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all=False)
"""

from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from tools.fuzzy_match_strategies import (  # noqa: F401 — re-exported names
    SIMILARITY_STRATEGIES,
    STRATEGIES,
    UNICODE_MAP,
    _build_orig_to_norm_map,
    _calculate_line_positions,
    _invert_norm_map,
    _map_normalized_positions,
    _map_positions_norm_to_orig,
    _norm_end_to_orig,
    _strategy_block_anchor,
    _strategy_context_aware,
    _strategy_escape_normalized,
    _strategy_exact,
    _strategy_indentation_flexible,
    _strategy_line_trimmed,
    _strategy_trimmed_boundary,
    _strategy_unicode_normalized,
    _strategy_whitespace_normalized,
    _unicode_normalize,
)

IDENTICAL_STRINGS_ERROR = (
    "No edit was applied because old_string and new_string are identical. "
    "Provide the existing text to replace in old_string and the changed "
    "replacement text in new_string."
)


def is_already_applied(content: str, old_string: str, new_string: str) -> bool:
    """True when the requested edit is already present (re-sent edit -> success-shaped no-op).

    Conservative: new_string must be non-trivial (>= 8 chars stripped) and
    appear EXACTLY; when it differs from old_string, old_string must be gone.
    """
    if not new_string or len(new_string.strip()) < 8:
        return False
    if new_string not in content:
        return False
    if old_string == new_string:
        return True
    return old_string not in content


def _matched_regions(content: str, matches: List[Tuple[int, int]]) -> str:
    return "".join(content[start:end] for start, end in matches)


def _format_match_locations(content: str, matches: List[Tuple[int, int]],
                            cap: int = 5) -> str:
    """Render up to ``cap`` match positions as 'L<line>: <snippet>' rows."""
    rows = []
    for start, _end in matches[:cap]:
        line_no = content.count("\n", 0, start) + 1
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", line_start)
        if line_end == -1:
            line_end = len(content)
        snippet = content[line_start:line_end].strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        rows.append(f"  L{line_no}: {snippet}")
    extra = len(matches) - cap
    if extra > 0:
        rows.append(f"  ... and {extra} more")
    return "\n".join(rows)


def fuzzy_find_and_replace(content: str, old_string: str, new_string: str,
                           replace_all: bool = False) -> Tuple[str, int, Optional[str], Optional[str]]:
    """Find and replace via the strategy chain.

    Returns ``(new_content, match_count, strategy_name, error)``; on failure
    ``(content, 0, None, error)``.
    """
    if not old_string:
        return content, 0, None, "old_string cannot be empty"
    if not old_string.strip():
        # Whitespace-only anchors match trivially and mass-replace or
        # ambiguity-error; never meaningful.
        return content, 0, None, "old_string is only whitespace — provide non-blank text to match"
    if old_string == new_string:
        return content, 0, None, IDENTICAL_STRINGS_ERROR

    for strategy_name, strategy_fn in STRATEGIES:
        matches = strategy_fn(content, old_string)
        if not matches:
            continue

        if len(matches) > 1 and not replace_all:
            locations = _format_match_locations(content, matches)
            return content, 0, None, (
                f"Found {len(matches)} matches for old_string. "
                f"Provide more context to make it unique, or use replace_all=True. "
                f"Matches:\n{locations}"
            )
        if replace_all and len(matches) > 1 and strategy_name in SIMILARITY_STRATEGIES:
            return content, 0, None, (
                f"Found {len(matches)} approximate matches via the "
                f"'{strategy_name}' strategy; replace_all only applies to exact "
                f"matches. Provide the precise text (whitespace included) so an "
                f"exact/line-trimmed match can be made."
            )

        # Non-exact matches came through some normalization, so new_string may
        # carry serialization drift the file doesn't have.
        if strategy_name != "exact":
            drift_err = _detect_escape_drift(content, matches, old_string, new_string)
            if drift_err:
                return content, 0, None, drift_err

        effective_new = _maybe_unescape_new_string(new_string, content, matches)
        if strategy_name == "unicode_normalized":
            effective_new = _preserve_unicode_in_replacement(
                content, matches, old_string, effective_new,
            )
        new_content = _apply_replacements(
            content, matches, effective_new,
            old_string=old_string if strategy_name != "exact" else None,
        )
        return new_content, len(matches), strategy_name, None

    return content, 0, None, "Could not find a match for old_string in the file"


# ── Escape-drift guards ──────────────────────────────────────────────────

def _detect_escape_drift(content: str, matches: List[Tuple[int, int]],
                         old_string: str, new_string: str) -> Optional[str]:
    """Error string when new_string carries tool-call escape artifacts, else None.

    Fires on ``\\'``/``\\"`` present in both old_string and new_string but
    absent from the matched region (spurious shell-style escaping), and on
    JSON double-escaped backslash runs (see ``_detect_backslash_doubling``).
    """
    has_quote_suspects = "\\'" in new_string or '\\"' in new_string
    if not has_quote_suspects and "\\" not in old_string:
        return None

    matched_regions = _matched_regions(content, matches)
    if has_quote_suspects:
        for suspect in ("\\'", '\\"'):
            if suspect in new_string and suspect in old_string and suspect not in matched_regions:
                plain = suspect[1]
                return (
                    f"Escape-drift detected: old_string and new_string contain "
                    f"the literal sequence {suspect!r} but the matched region of "
                    f"the file does not. This is almost always a tool-call "
                    f"serialization artifact where an apostrophe or quote got "
                    f"prefixed with a spurious backslash. Re-read the file with "
                    f"read_file and pass old_string/new_string without "
                    f"backslash-escaping {plain!r} characters."
                )
    return _detect_backslash_doubling(matched_regions, old_string, new_string)


def _backslash_runs(s: str) -> List[int]:
    """Lengths of maximal backslash runs in ``s``, in order."""
    runs: List[int] = []
    n = 0
    for ch in s:
        if ch == "\\":
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def _detect_backslash_doubling(matched_regions: str, old_string: str,
                               new_string: str) -> Optional[str]:
    """Detect old_string whose every backslash run is exactly 2x the file's.

    That pattern means the arguments were JSON-escaped one extra time; a
    similarity strategy still matches, and writing new_string verbatim would
    double every backslash in the file. Requires the same run count, a
    non-trivial signal (a run >= 2 or 2+ runs), and new_string not already
    matching the file's counts.
    """
    old_runs = _backslash_runs(old_string)
    file_runs = _backslash_runs(matched_regions)
    if not old_runs or not file_runs or len(old_runs) != len(file_runs):
        return None
    if old_runs == file_runs:
        return None
    if any(o != f * 2 for o, f in zip(old_runs, file_runs)):
        return None
    if not (any(f >= 2 for f in file_runs) or len(file_runs) >= 2):
        return None
    if _backslash_runs(new_string) == file_runs:
        return None
    return (
        "Escape-drift detected: every backslash run in old_string is exactly "
        "twice as long as in the matched region of the file (e.g. the file "
        "has `\\\\` where old_string has `\\\\\\\\`). The tool-call arguments "
        "were JSON-escaped one extra time; applying new_string verbatim would "
        "double every backslash in the file. Re-read the file with read_file "
        "and resend old_string/new_string with the backslash counts exactly "
        "as they appear in the file."
    )


def _maybe_unescape_new_string(new_string: str, content: str,
                               matches: List[Tuple[int, int]]) -> str:
    """Convert literal ``\\t``/``\\r`` in new_string to control chars, per sequence,
    only when the matched file region already contains the real control char.

    Files that legitimately contain the two-char string (e.g. ``sep = "\\t"``)
    have a backslash+t in the region, not a tab, so they're left alone.
    ``\\n`` is deliberately excluded: newlines serialize correctly through
    JSON and rewriting them would mangle escape sequences in source literals.
    """
    if "\\t" not in new_string and "\\r" not in new_string:
        return new_string
    matched_regions = _matched_regions(content, matches)
    out = new_string
    if "\\t" in out and "\t" in matched_regions:
        out = out.replace("\\t", "\t")
    if "\\r" in out and "\r" in matched_regions:
        out = out.replace("\\r", "\r")
    return out


# ── Replacement shaping ──────────────────────────────────────────────────

def _leading_whitespace(line: str) -> str:
    return line[:len(line) - len(line.lstrip(" \t"))]


def _first_meaningful_line(text: str) -> Optional[str]:
    for line in text.split("\n"):
        if line.strip():
            return line
    return None


def _reindent_replacement(file_region: str, old_string: str, new_string: str) -> str:
    """Re-anchor ``new_string``'s indentation onto the file's actual base indent.

    After a non-exact match the LLM's base indent (first non-blank line of
    old_string) may differ from the file's. Each non-blank new_string line
    swaps the LLM base prefix for the file's, preserving relative nesting;
    lines shallower than the LLM base are anchored to the file base.
    """
    if not new_string:
        return new_string
    old_first = _first_meaningful_line(old_string)
    file_first = _first_meaningful_line(file_region)
    if old_first is None or file_first is None:
        return new_string
    old_indent = _leading_whitespace(old_first)
    file_indent = _leading_whitespace(file_first)
    if old_indent == file_indent:
        return new_string

    out_lines: List[str] = []
    for line in new_string.split("\n"):
        if not line.strip():
            out_lines.append(line)
        elif _leading_whitespace(line).startswith(old_indent):
            out_lines.append(file_indent + line[len(old_indent):])
        else:
            out_lines.append(file_indent + line.lstrip(" \t"))
    return "\n".join(out_lines)


def _preserve_unicode_in_replacement(
    content: str, matches: List[Tuple[int, int]],
    old_string: str, new_string: str,
) -> str:
    """Apply only the old->new edits onto the file's original (Unicode) text.

    After a unicode_normalized match, writing the LLM's ASCII new_string
    verbatim would flatten the file's em-dashes/smart quotes. Diff the
    normalized old_string against new_string and keep the file's original
    characters for every ``equal`` span.
    """
    file_region = _matched_regions(content, matches)
    norm_old = _unicode_normalize(old_string)
    if norm_old != _unicode_normalize(file_region):
        return new_string  # strategy shouldn't have fired; fall back

    file_orig_to_norm = _build_orig_to_norm_map(file_region)
    file_norm_to_orig = _invert_norm_map(file_orig_to_norm)

    result_parts: List[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, norm_old, new_string).get_opcodes():
        if tag == "equal":
            orig_start = file_norm_to_orig.get(i1, 0)
            orig_end = _norm_end_to_orig(file_orig_to_norm, orig_start, i2)
            result_parts.append(file_region[orig_start:orig_end])
        elif tag != "delete":
            result_parts.append(new_string[j1:j2])
    return "".join(result_parts)


def _apply_replacements(content: str, matches: List[Tuple[int, int]],
                        new_string: str, old_string: Optional[str] = None) -> str:
    """Splice ``new_string`` over each span (end-to-start so offsets stay valid).

    ``old_string`` non-None signals a non-exact match: new_string is
    re-indented per region to the file's actual indentation.
    """
    result = content
    for start, end in sorted(matches, key=lambda x: x[0], reverse=True):
        adjusted = new_string
        if old_string is not None:
            adjusted = _reindent_replacement(content[start:end], old_string, new_string)
        result = result[:start] + adjusted + result[end:]
    return result


# ── "Did you mean?" diagnostics ──────────────────────────────────────────

def _visualize_whitespace(line: str) -> str:
    """Render the leading whitespace run visibly (→ = tab, · = space)."""
    i = 0
    prefix = []
    while i < len(line) and line[i] in (" ", "\t"):
        prefix.append("→" if line[i] == "\t" else "·")
        i += 1
    return "".join(prefix) + line[i:]


def find_closest_lines(old_string: str, content: str, context_lines: int = 2, max_results: int = 3) -> str:
    """Numbered snippets of the lines most similar to old_string's anchor line, or ''."""
    if not old_string or not content:
        return ""
    old_lines = old_string.splitlines()
    content_lines = content.splitlines()
    if not old_lines or not content_lines:
        return ""

    anchor = old_lines[0].strip()
    if not anchor:
        candidates = [l.strip() for l in old_lines if l.strip()]
        if not candidates:
            return ""
        anchor = candidates[0]

    scored = []
    for i, line in enumerate(content_lines):
        stripped = line.strip()
        if not stripped:
            continue
        ratio = SequenceMatcher(None, anchor, stripped).ratio()
        if ratio > 0.3:
            scored.append((ratio, i))
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[0])
    top = scored[:max_results]

    parts = []
    seen_ranges = set()
    for _, line_idx in top:
        start = max(0, line_idx - context_lines)
        end = min(len(content_lines), line_idx + len(old_lines) + context_lines)
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        parts.append("\n".join(
            f"{start + j + 1:4d}| {content_lines[start + j]}"
            for j in range(end - start)
        ))
    if not parts:
        return ""
    result = "\n---\n".join(parts)

    # Whitespace-shaped miss: best line equals the anchor once stripped. Show
    # both with visible leading whitespace so the model copies the file's.
    best_line = content_lines[top[0][1]]
    if best_line.strip() == anchor and best_line != old_lines[0]:
        result += (
            "\n\nWhitespace difference detected (→ = tab, · = space):\n"
            f"  file has: {_visualize_whitespace(best_line)}\n"
            f"  you sent: {_visualize_whitespace(old_lines[0])}\n"
            "Use the exact whitespace shown in 'file has'."
        )
    return result


def format_no_match_hint(error: Optional[str], match_count: int,
                         old_string: str, content: str) -> str:
    """'\\n\\nDid you mean...' snippet for plain no-match errors only, else ''.

    Ambiguous-match, escape-drift and identical-strings errors also have
    ``match_count == 0`` but a hint would mislead there.
    """
    if match_count != 0:
        return ""
    if not error or not error.startswith("Could not find"):
        return ""
    hint = find_closest_lines(old_string, content)
    if not hint:
        return ""
    return "\n\nDid you mean one of these sections?\n" + hint
