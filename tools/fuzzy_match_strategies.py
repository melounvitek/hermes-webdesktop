"""Matching strategies for :mod:`tools.fuzzy_match`.

Each strategy takes ``(content, pattern)`` and returns a list of
``(start, end)`` spans in the ORIGINAL content. ``STRATEGIES`` is the ordered
chain the orchestrator tries; ``SIMILARITY_STRATEGIES`` names the ones whose
matches are approximate and therefore unsafe under ``replace_all``.
"""

import re
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Tuple

UNICODE_MAP = {
    "\u201c": '"', "\u201d": '"',  # smart double quotes
    "\u2018": "'", "\u2019": "'",  # smart single quotes
    "\u2014": "--", "\u2013": "-",  # em/en dashes
    "\u2026": "...", "\u00a0": " ",  # ellipsis and non-breaking space
    "\u2212": "-",  # typographic minus (math/scientific docs)
    # Space-separator family (Zs) beyond NBSP: files with typographic spacing
    # otherwise miss every precise strategy and fall through to the
    # similarity fallback, which can pick the wrong region.
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ",
    "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ",
    "\u2008": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
    "\u205f": " ", "\u3000": " ",
}

Span = Tuple[int, int]


def _unicode_normalize(text: str) -> str:
    """Map typographic Unicode variants to ASCII equivalents."""
    for char, repl in UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text


# ── Position helpers ─────────────────────────────────────────────────────

def _calculate_line_positions(content_lines: List[str], start_line: int,
                              end_line: int, content_length: int) -> Span:
    """Character span covering ``content_lines[start_line:end_line]`` (end exclusive)."""
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    return start_pos, min(content_length, end_pos)


def _match_transformed_lines(content: str, pattern: str,
                             transform: Callable[[str], str]) -> List[Span]:
    """Match ``pattern`` against ``content`` after applying ``transform`` per line."""
    content_lines = content.split('\n')
    norm_lines = [transform(line) for line in content_lines]
    pattern_norm = '\n'.join(transform(line) for line in pattern.split('\n'))
    n = pattern_norm.count('\n') + 1
    matches = []
    for i in range(len(norm_lines) - n + 1):
        if '\n'.join(norm_lines[i:i + n]) == pattern_norm:
            matches.append(_calculate_line_positions(content_lines, i, i + n, len(content)))
    return matches


def _build_orig_to_norm_map(original: str) -> List[int]:
    """Map each original index to its index in ``_unicode_normalize(original)``.

    UNICODE_MAP replacements can expand one char into several, so the map is
    needed to translate normalised spans back. Length is ``len(original)+1``;
    the last entry is a sentinel one past the final character.
    """
    result: List[int] = []
    norm_pos = 0
    for char in original:
        result.append(norm_pos)
        repl = UNICODE_MAP.get(char)
        norm_pos += len(repl) if repl is not None else 1
    result.append(norm_pos)
    return result


def _invert_norm_map(orig_to_norm: List[int]) -> Dict[int, int]:
    """norm_pos -> first original position mapping to it."""
    inverted: Dict[int, int] = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm[:-1]):
        if norm_pos not in inverted:
            inverted[norm_pos] = orig_pos
    return inverted


def _norm_end_to_orig(orig_to_norm: List[int], orig_start: int, norm_end: int) -> int:
    """Walk from ``orig_start`` until the mapped position reaches ``norm_end``."""
    orig_len = len(orig_to_norm) - 1
    orig_end = orig_start
    while orig_end < orig_len and orig_to_norm[orig_end] < norm_end:
        orig_end += 1
    return orig_end


def _map_positions_norm_to_orig(orig_to_norm: List[int],
                                norm_matches: List[Span]) -> List[Span]:
    """Convert spans in the normalised string to original-string spans."""
    norm_to_orig_start = _invert_norm_map(orig_to_norm)
    results: List[Span] = []
    for norm_start, norm_end in norm_matches:
        if norm_start not in norm_to_orig_start:
            continue
        orig_start = norm_to_orig_start[norm_start]
        results.append((orig_start, _norm_end_to_orig(orig_to_norm, orig_start, norm_end)))
    return results


def _map_normalized_positions(original: str, normalized: str,
                              normalized_matches: List[Span]) -> List[Span]:
    """Best-effort span mapping for ``[ \\t]+`` -> ``' '`` whitespace collapsing."""
    orig_to_norm = []  # orig_to_norm[i] = position in normalized
    orig_idx = norm_idx = 0
    while orig_idx < len(original) and norm_idx < len(normalized):
        if original[orig_idx] == normalized[norm_idx]:
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            norm_idx += 1
        elif original[orig_idx] in ' \t' and normalized[norm_idx] == ' ':
            # Collapsed run: advance norm_idx only once the run is consumed.
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            if orig_idx < len(original) and original[orig_idx] not in ' \t':
                norm_idx += 1
        else:
            # Extra whitespace in original, or a mismatch that normalization
            # should never produce — either way, pin to the current norm_idx.
            orig_to_norm.append(norm_idx)
            orig_idx += 1
    while orig_idx < len(original):
        orig_to_norm.append(len(normalized))
        orig_idx += 1

    norm_to_orig_start = {}
    norm_to_orig_end = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos
        norm_to_orig_end[norm_pos] = orig_pos

    original_matches = []
    for norm_start, norm_end in normalized_matches:
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
        else:
            orig_start = min(i for i, n in enumerate(orig_to_norm) if n >= norm_start)
        if norm_end - 1 in norm_to_orig_end:
            orig_end = norm_to_orig_end[norm_end - 1] + 1
        else:
            orig_end = orig_start + (norm_end - norm_start)
        # Absorb trailing collapsed whitespace only when the normalized match
        # itself ended in a space; otherwise the first whitespace after the
        # match is a word boundary that must survive (#52491).
        if norm_end < len(normalized) and normalized[norm_end - 1] == ' ':
            while orig_end < len(original) and original[orig_end] in ' \t':
                orig_end += 1
        original_matches.append((orig_start, min(orig_end, len(original))))
    return original_matches


# ── Strategies ───────────────────────────────────────────────────────────

def _strategy_exact(content: str, pattern: str) -> List[Span]:
    """Strategy 1: exact, non-overlapping occurrences (str.replace semantics)."""
    matches = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        # Advance past the whole match: overlapping spans would corrupt the
        # file under replace_all (reverse-order apply on stale offsets).
        start = pos + len(pattern)
    return matches


def _strategy_line_trimmed(content: str, pattern: str) -> List[Span]:
    """Strategy 2: strip each line before comparing."""
    return _match_transformed_lines(content, pattern, str.strip)


def _strategy_whitespace_normalized(content: str, pattern: str) -> List[Span]:
    """Strategy 3: collapse runs of spaces/tabs to a single space."""
    def normalize(s):
        return re.sub(r'[ \t]+', ' ', s)

    content_normalized = normalize(content)
    matches_in_normalized = _strategy_exact(content_normalized, normalize(pattern))
    if not matches_in_normalized:
        return []
    return _map_normalized_positions(content, content_normalized, matches_in_normalized)


def _strategy_indentation_flexible(content: str, pattern: str) -> List[Span]:
    """Strategy 4: ignore leading indentation entirely."""
    return _match_transformed_lines(content, pattern, str.lstrip)


def _strategy_escape_normalized(content: str, pattern: str) -> List[Span]:
    """Strategy 5: treat literal ``\\n``/``\\t``/``\\r`` in the pattern as control chars."""
    pattern_unescaped = pattern.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    if pattern_unescaped == pattern:
        return []
    return _strategy_exact(content, pattern_unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> List[Span]:
    """Strategy 6: strip whitespace on the first and last lines only."""
    pattern_lines = pattern.split('\n')
    pattern_lines[0] = pattern_lines[0].strip()
    if len(pattern_lines) > 1:
        pattern_lines[-1] = pattern_lines[-1].strip()
    modified_pattern = '\n'.join(pattern_lines)
    n = len(pattern_lines)

    content_lines = content.split('\n')
    matches = []
    for i in range(len(content_lines) - n + 1):
        check_lines = content_lines[i:i + n]
        check_lines[0] = check_lines[0].strip()
        if n > 1:
            check_lines[-1] = check_lines[-1].strip()
        if '\n'.join(check_lines) == modified_pattern:
            matches.append(_calculate_line_positions(content_lines, i, i + n, len(content)))
    return matches


def _strategy_unicode_normalized(content: str, pattern: str) -> List[Span]:
    """Strategy 7: exact/line-trimmed match after Unicode->ASCII normalisation of both sides."""
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    if norm_content == content and norm_pattern == pattern:
        return []
    norm_matches = _strategy_exact(norm_content, norm_pattern)
    if not norm_matches:
        norm_matches = _strategy_line_trimmed(norm_content, norm_pattern)
    if not norm_matches:
        return []
    return _map_positions_norm_to_orig(_build_orig_to_norm_map(content), norm_matches)


def _strategy_block_anchor(content: str, pattern: str) -> List[Span]:
    """Strategy 8: anchor on first+last lines, similarity-score the middle."""
    pattern_lines = _unicode_normalize(pattern).split('\n')
    if len(pattern_lines) < 2:
        return []
    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()
    n = len(pattern_lines)

    # Match on normalized lines; compute offsets from the ORIGINAL lines so
    # multi-char expansions (em-dash -> '--') don't shift positions.
    norm_content_lines = _unicode_normalize(content).split('\n')
    orig_content_lines = content.split('\n')

    potential_matches = [
        i for i in range(len(norm_content_lines) - n + 1)
        if norm_content_lines[i].strip() == first_line
        and norm_content_lines[i + n - 1].strip() == last_line
    ]
    # Looser thresholds (0.10/0.30) matched unrelated blocks; these are the safe floor.
    threshold = 0.50 if len(potential_matches) == 1 else 0.70

    matches = []
    for i in potential_matches:
        if n <= 2:
            similarity = 1.0
        else:
            content_middle = '\n'.join(norm_content_lines[i + 1:i + n - 1])
            pattern_middle = '\n'.join(pattern_lines[1:-1])
            similarity = SequenceMatcher(None, content_middle, pattern_middle).ratio()
        if similarity >= threshold:
            matches.append(_calculate_line_positions(orig_content_lines, i, i + n, len(content)))
    return matches


def _strategy_context_aware(content: str, pattern: str) -> List[Span]:
    """Strategy 9 (last resort): anchored per-line similarity, every non-blank line >= 0.80.

    The first/last-line anchor pre-filter keeps a miss from being an
    O(file x pattern) scan; the all-lines requirement stops one coincidental
    line match from replacing an unrelated block.
    """
    pattern_lines = pattern.split('\n')
    content_lines = content.split('\n')
    n = len(pattern_lines)
    if n > len(content_lines):
        return []

    first_pat = pattern_lines[0].strip()
    last_pat = pattern_lines[-1].strip()
    ANCHOR_THRESHOLD = 0.80

    def _sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    matches = []
    for i in range(len(content_lines) - n + 1):
        block_lines = content_lines[i:i + n]
        if _sim(first_pat, block_lines[0].strip()) < ANCHOR_THRESHOLD:
            continue
        if _sim(last_pat, block_lines[-1].strip()) < ANCHOR_THRESHOLD:
            continue
        all_match = True
        for p_line, c_line in zip(pattern_lines, block_lines):
            p_stripped = p_line.strip()
            if p_stripped and _sim(p_stripped, c_line.strip()) < 0.80:
                all_match = False
                break
        if all_match:
            matches.append(_calculate_line_positions(content_lines, i, i + n, len(content)))
    return matches


# Ordered chain: precise strategies first, similarity-based last.
STRATEGIES: List[Tuple[str, Callable[[str, str], List[Span]]]] = [
    ("exact", _strategy_exact),
    ("line_trimmed", _strategy_line_trimmed),
    ("whitespace_normalized", _strategy_whitespace_normalized),
    ("indentation_flexible", _strategy_indentation_flexible),
    ("escape_normalized", _strategy_escape_normalized),
    ("trimmed_boundary", _strategy_trimmed_boundary),
    ("unicode_normalized", _strategy_unicode_normalized),
    ("block_anchor", _strategy_block_anchor),
    ("context_aware", _strategy_context_aware),
]

# Matches from these only *approximately* resemble old_string — fine for one
# unique replacement, never safe under replace_all.
SIMILARITY_STRATEGIES = frozenset({"block_anchor", "context_aware"})
