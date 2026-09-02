"""Shared helpers for gateway platform adapters: message dedup, markdown
stripping, thread participation tracking, GFM table → bullets, mention-pattern
compilation, and fence-aware markdown chunking."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict

from utils import atomic_json_write

logger = logging.getLogger(__name__)


# ─── Message Deduplication ────────────────────────────────────────────────────


class MessageDeduplicator:
    """TTL-based message deduplication cache (``if dedup.is_duplicate(msg_id): return``)."""

    def __init__(self, max_size: int = 2000, ttl_seconds: float = 300):
        self._seen: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was already seen within the TTL window."""
        if not msg_id:
            return False
        now = time.time()
        if msg_id in self._seen:
            if now - self._seen[msg_id] < self._ttl:
                return True
            # Entry has expired — remove it and treat as new
            del self._seen[msg_id]
        self._seen[msg_id] = now
        if len(self._seen) > self._max_size:
            cutoff = now - self._ttl
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
            if len(self._seen) > self._max_size:
                # All entries still fresh: keep the newest so max_size holds under load.
                self._seen = dict(sorted(self._seen.items(), key=lambda item: item[1])[-self._max_size:])
        return False

    def contains(self, msg_id: str) -> bool:
        """Return whether *msg_id* is live in the cache without inserting it."""
        if not msg_id:
            return False
        seen_at = self._seen.get(msg_id)
        if seen_at is None:
            return False
        if time.time() - seen_at < self._ttl:
            return True
        del self._seen[msg_id]
        return False

    def discard(self, msg_id: str) -> None:
        """Release a claimed message ID after cancelled/failed handoff."""
        self._seen.pop(msg_id, None)

    def clear(self):
        """Clear all tracked messages."""
        self._seen.clear()


# ─── Markdown Stripping ──────────────────────────────────────────────────────

# Pre-compiled regexes for performance
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_RE_ITALIC_STAR = re.compile(r"\*(.+?)\*", re.DOTALL)
_RE_BOLD_UNDER = re.compile(r"\b__(?![\s_])(.+?)(?<![\s_])__\b", re.DOTALL)
_RE_ITALIC_UNDER = re.compile(r"\b_(?![\s_])(.+?)(?<![\s_])_\b", re.DOTALL)
_RE_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n?")
_RE_INLINE_CODE = re.compile(r"`(.+?)`")
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text platforms (SMS, iMessage, etc.)."""
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_ITALIC_UNDER.sub(r"\1", text)
    text = _RE_CODE_BLOCK.sub("", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─── Thread Participation Tracking ───────────────────────────────────────────


class ThreadParticipationTracker:
    """Persistent set of threads the bot has participated in (``<platform>_threads.json``).

    ``thread_id in tracker`` checks membership; ``tracker.mark(thread_id)`` persists.
    """

    _MAX_TRACKED = 500

    def __init__(self, platform_name: str, max_tracked: int = 500):
        self._platform = platform_name
        self._max_tracked = max_tracked
        self._threads: dict[str, None] = dict.fromkeys(str(t) for t in self._load())

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / f"{self._platform}_threads.json"

    def _load(self) -> list[str]:
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [str(thread_id) for thread_id in data]
            except Exception:
                pass
        return []

    def _save(self) -> None:
        path = self._state_path()
        thread_list = list(self._threads)
        if len(thread_list) > self._max_tracked:
            thread_list = thread_list[-self._max_tracked:]
            self._threads = dict.fromkeys(thread_list)
        atomic_json_write(path, thread_list, indent=None)

    def mark(self, thread_id: str) -> None:
        """Mark *thread_id* as participated and persist."""
        if thread_id not in self._threads:
            self._threads[thread_id] = None
            self._save()

    def __contains__(self, thread_id: str) -> bool:
        return thread_id in self._threads

    def clear(self) -> None:
        self._threads.clear()


# ─── Phone Number Redaction ──────────────────────────────────────────────────


def redact_phone(phone: str) -> str:
    """Redact a phone number for logging, preserving country code and last 4."""
    if not phone:
        return "<none>"
    if len(phone) <= 8:
        return phone[:2] + "****" + phone[-2:] if len(phone) > 4 else "****"
    return phone[:4] + "****" + phone[-4:]


# ─── GFM Markdown Table → Bullet Conversion ─────────────────────────────────
# Discord calls convert_table_to_bullets(); Telegram imports the primitives but
# keeps its own MarkdownV2-aware renderer.

# GFM delimiter row: optional outer pipes, dash cells (optional alignment colons).
# Requires at least one internal '|' so a lone '---' rule is NOT matched.
TABLE_SEPARATOR_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$')


def is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    stripped = line.strip()
    return bool(stripped) and '|' in stripped


def split_markdown_table_row(line: str) -> list[str]:
    """Split a GFM table row into stripped cells (delegates to agent.markdown_tables)."""
    from agent.markdown_tables import split_table_row
    return split_table_row(line)


def _render_table_block(table_block: list[str]) -> str:
    """Render a detected GFM table as bold-heading + bullet groups.

    Same alignment logic as Telegram's renderer: without a row-label column the
    full row is the data and the bullet duplicating the heading is skipped.
    """
    if len(table_block) < 3:
        return "\n".join(table_block)
    headers = split_markdown_table_row(table_block[0])
    if len(headers) < 2:
        return "\n".join(table_block)
    first_data_row = split_markdown_table_row(table_block[2])
    has_row_label_col = len(first_data_row) == len(headers) + 1
    rendered_groups: list[str] = []
    for index, row in enumerate(table_block[2:], start=1):
        cells = split_markdown_table_row(row)
        if has_row_label_col:
            heading = cells[0] if cells and cells[0] else f"Row {index}"
            data_cells = cells[1:]
        else:
            heading = next((cell for cell in cells if cell), f"Row {index}")
            data_cells = cells
        if len(data_cells) < len(headers):
            data_cells.extend([""] * (len(headers) - len(data_cells)))
        elif len(data_cells) > len(headers):
            data_cells = data_cells[: len(headers)]
        bullets = [
            f"• {header}: {value}"
            for header, value in zip(headers, data_cells)
            if has_row_label_col or value != heading
        ]
        rendered_groups.append("\n".join([f"**{heading}**", *bullets]))
    return "\n\n".join(rendered_groups)


def convert_table_to_bullets(text: str) -> str:
    """Rewrite GFM pipe tables into bold-heading + bullet groups.

    Tables inside fenced code blocks are left alone.
    """
    if '|' not in text or '-' not in text:
        return text
    lines = text.split('\n')
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith('```'):
            in_fence = not in_fence
        if in_fence or stripped.startswith('```'):
            out.append(line)
            i += 1
            continue
        if '|' in line and i + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[i + 1]):
            table_block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and is_table_row(lines[j]):
                table_block.append(lines[j])
                j += 1
            out.append(_render_table_block(table_block))
            i = j
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


# ─── Mention-pattern compilation ─────────────────────────────────────────────


def compile_mention_patterns(
    raw,
    *,
    log_prefix: str,
    platform_label: str | None = None,
    display_label: str | None = None,
    defaults: 'list[str] | None' = None,
    logger_: 'logging.Logger | None' = None,
) -> 'list[re.Pattern]':
    """Compile regex wake-word/mention patterns from config or env values.

    * **Config-style** (dingtalk, telegram): pass ``platform_label``. ``raw``
      must be a list or string (anything else warns and yields ``[]``);
      non-string entries are skipped; a summary info log is emitted on load.
    * **Wakeword-style** (photon, bluebubbles): pass ``defaults``. ``raw`` may be
      None (defaults), a string (JSON list or comma/newline separated), a list,
      or a scalar; entries are coerced via ``str()``.

    ``log_prefix`` is interpolated into every log line so per-adapter output
    stays byte-identical to the historical inline implementations.
    """
    log = logger_ or logger
    if platform_label is not None:
        display = display_label or platform_label
        patterns = raw
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            log.warning(
                "[%s] %s mention_patterns must be a list or string; got %s",
                log_prefix, platform_label, type(patterns).__name__,
            )
            return []
        compiled: list[re.Pattern] = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                log.warning("[%s] Invalid %s mention pattern %r: %s", log_prefix, display, pattern, exc)
        if compiled:
            log.info("[%s] Loaded %d %s mention pattern(s)", log_prefix, len(compiled), display)
        return compiled
    if raw is None:
        patterns = list(defaults or [])
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            loaded = json.loads(text) if text else []
        except Exception:
            loaded = None
        patterns = loaded if isinstance(loaded, list) else [
            part.strip() for line in text.splitlines() for part in line.split(",")
        ]
    elif isinstance(raw, list):
        patterns = raw
    else:
        patterns = [raw]
    compiled = []
    for pattern in patterns:
        text = str(pattern).strip()
        if not text:
            continue
        try:
            compiled.append(re.compile(text, re.IGNORECASE))
        except re.error as exc:
            log.warning("[%s] Invalid mention pattern %r: %s", log_prefix, text, exc)
    return compiled


# ─── Fence-Aware Markdown Chunking ───────────────────────────────────────────
# Shared core for the chunkers in gateway/stream_consumer.py (newline-preferred
# + fence balancing: ``prefer_paragraphs=False, balance_fences=True``), yuanbao
# (atomic blocks + paragraph splitting, fences kept as atoms:
# ``prefer_paragraphs=True, balance_fences=False``) and weixin (own block
# splitter, reuses ``greedy_pack_blocks``).


def text_has_unclosed_fence(text: str) -> bool:
    """Return True when *text* ends inside an unclosed ``` code fence."""
    in_fence = False
    for line in text.split('\n'):
        if line.startswith('```'):
            in_fence = not in_fence
    return in_fence


def text_ends_with_table_row(text: str) -> bool:
    """True when the last non-empty line starts and ends with ``|``."""
    trimmed = text.rstrip()
    return bool(trimmed) and _is_pipe_row(trimmed.split('\n')[-1])


def is_fence_atom(text: str) -> bool:
    """True when an atomic block is a code block (starts with ```)."""
    return text.lstrip().startswith('```')


def _is_pipe_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|')


def is_table_atom(text: str) -> bool:
    """True when an atomic block is a table (first line is ``|...|``)."""
    return _is_pipe_row(text.split('\n')[0])


_SENTENCE_END_NEWLINE_RE = re.compile(r'[。！？.!?]\n')


def split_at_paragraph_boundary(text, max_chars, len_fn=None):
    """Split at the nearest paragraph boundary within *max_chars*; return (head, tail).

    Priority: blank line → newline after sentence-ending punctuation (CJK/ASCII)
    → last newline → forced split at the window boundary. ``head + tail == text``
    always holds. *len_fn* measures in custom units (e.g. UTF-16 code units).
    """
    _len = len_fn or len
    if _len(text) <= max_chars:
        return text, ''
    if _len is len:
        window = text[:max_chars]
    else:
        from gateway.platforms.base import _custom_unit_to_cp  # heavyweight; lazy
        window = text[:_custom_unit_to_cp(text, max_chars, _len)]
    pos = window.rfind('\n\n')
    if pos > 0:
        return text[:pos + 2], text[pos + 2:]
    best_pos = -1
    for m in _SENTENCE_END_NEWLINE_RE.finditer(window):
        best_pos = m.end()
    if best_pos > 0:
        return text[:best_pos], text[best_pos:]
    pos = window.rfind('\n')
    if pos > 0:
        return text[:pos + 1], text[pos + 1:]
    cut = len(window)
    return text[:cut], text[cut:]


def split_markdown_atoms(text: str) -> "list[str]":
    """Split markdown into indivisible atoms: fenced code blocks, tables
    (consecutive ``|...|`` lines) and paragraphs. Blank lines belong to no atom."""
    lines = text.split('\n')
    atoms: "list[str]" = []
    current_lines: "list[str]" = []
    in_fence = False
    _is_table_line = _is_pipe_row

    def _flush_current() -> None:
        if current_lines:
            atom = '\n'.join(current_lines)
            if atom.strip():
                atoms.append(atom)
            current_lines.clear()
    for line in lines:
        if in_fence:
            current_lines.append(line)
            if line.startswith('```') and len(current_lines) > 1:
                in_fence = False
                _flush_current()
        elif line.startswith('```'):
            _flush_current()
            in_fence = True
            current_lines.append(line)
        elif _is_table_line(line):
            if current_lines and not _is_table_line(current_lines[-1]):
                _flush_current()
            current_lines.append(line)
        elif line.strip() == '':
            _flush_current()
        else:
            if current_lines and _is_table_line(current_lines[-1]):
                _flush_current()
            current_lines.append(line)
    _flush_current()
    return atoms


def infer_block_separator(prev_chunk: str, next_chunk: str) -> str:
    """``'\\n'`` when the boundary sits at a code fence or a continued table, else ``'\\n\\n'``."""
    prev_trimmed = prev_chunk.rstrip()
    next_trimmed = next_chunk.lstrip()
    if prev_trimmed.endswith('```') or next_trimmed.startswith('```'):
        return '\n'
    if text_ends_with_table_row(prev_chunk) and next_trimmed and _is_pipe_row(next_trimmed.split('\n')[0]):
        return '\n'
    return '\n\n'


def merge_streaming_fences(chunks: "list[str]") -> "list[str]":
    """Rejoin chunks truncated mid-fence: while chunk *i* has an unclosed fence
    and a successor exists, merge the successor in via :func:`infer_block_separator`."""
    if not chunks:
        return []
    result: "list[str]" = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        while text_has_unclosed_fence(current) and i + 1 < len(chunks):
            sep = infer_block_separator(current, chunks[i + 1])
            current = current + sep + chunks[i + 1]
            i += 1
        result.append(current)
        i += 1
    return result


def balance_fences_across_chunks(chunks: "list[str]") -> "list[str]":
    """Close orphaned ``` fences at each chunk boundary and reopen (with the
    original language tag) on the next, so every chunk is fence-balanced alone."""
    if len(chunks) <= 1:
        return chunks
    out: "list[str]" = []
    carry_lang = None
    for chunk in chunks:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        in_code, lang = fence_state_after(chunk, carry_lang is not None, carry_lang or "")
        body = prefix + chunk
        if in_code:
            body += "\n```"
            carry_lang = lang
        else:
            carry_lang = None
        out.append(body)
    return out


def fence_state_after(text: str, in_code: bool = False, lang: str = "") -> "tuple[bool, str]":
    """Walk ``text`` line by line toggling on ``` lines; return the final (in_code, lang)."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                in_code, lang = False, ""
            else:
                tag = stripped[3:].strip()
                in_code, lang = True, (tag.split()[0] if tag else "")
    return in_code, lang


def greedy_pack_blocks(blocks, max_length, len_fn=None, sep="\n\n", overflow=None):
    """Greedily pack *blocks* (joined with *sep*) into chunks of at most *max_length*.

    A block that alone exceeds the limit goes through *overflow(block)* (must
    return a list of chunks) when provided, else is emitted as-is.
    """
    _len = len_fn or len
    packed: "list[str]" = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}{sep}{block}"
        if _len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            packed.append(current)
            current = ""
        if _len(block) <= max_length:
            current = block
            continue
        if overflow is not None:
            packed.extend(overflow(block))
        else:
            packed.append(block)
    if current:
        packed.append(current)
    return packed


def split_text_fence_aware(
    text,
    limit,
    len_fn=None,
    *,
    prefer_paragraphs=True,
    balance_fences=False,
):
    """Split markdown into chunks of at most *limit*, respecting fences.

    ``prefer_paragraphs=True`` (yuanbao-derived): atoms (fences, tables,
    paragraphs) are greedily merged, oversized non-atomic chunks split at
    paragraph boundaries, small neighbours re-merged; a single atom larger than
    *limit* is emitted oversize rather than broken.
    ``prefer_paragraphs=False`` (stream_consumer-derived): newline-preferred
    hard splitting with headroom reserved for fence markers.
    ``balance_fences=True`` closes/reopens fences at chunk boundaries so each
    chunk renders standalone.
    """
    _len = len_fn or len
    if not text:
        return []
    if prefer_paragraphs:
        chunks = _chunk_markdown_paragraphs(text, limit, len_fn)
    else:
        chunks = _chunk_newline_preferred(text, limit, _len)
    if balance_fences:
        chunks = balance_fences_across_chunks(chunks)
    return chunks


def _chunk_markdown_paragraphs(text, max_chars, len_fn=None):
    """Yuanbao-derived paragraph/atom chunking pipeline (see module docs)."""
    _len = len_fn or len
    if _len(text) <= max_chars:
        return [text]
    atoms = split_markdown_atoms(text)
    # Phase 2: greedy merge; oversized fence/table atoms stay indivisible.
    chunks: "list[str]" = []
    indivisible_set: "set[int]" = set()
    current_parts: "list[str]" = []
    current_len = 0

    def _flush_parts() -> None:
        if current_parts:
            chunks.append('\n\n'.join(current_parts))
    for atom in atoms:
        atom_len = _len(atom)
        sep_len = 2 if current_parts else 0
        projected_len = current_len + sep_len + atom_len
        if projected_len > max_chars and current_parts:
            _flush_parts()
            current_parts = []
            current_len = 0
            sep_len = 0
        if (not current_parts
                and atom_len > max_chars
                and (is_fence_atom(atom) or is_table_atom(atom))):
            indivisible_set.add(len(chunks))
            chunks.append(atom)
            continue
        current_parts.append(atom)
        current_len += sep_len + atom_len
    _flush_parts()
    # Phase 3: split still-oversized divisible chunks at paragraph boundaries.
    result: "list[str]" = []
    for idx, chunk in enumerate(chunks):
        if _len(chunk) <= max_chars or idx in indivisible_set or text_has_unclosed_fence(chunk):
            result.append(chunk)
            continue
        remaining = chunk
        while _len(remaining) > max_chars:
            head, remaining = split_at_paragraph_boundary(remaining, max_chars, len_fn=len_fn)
            if not head:
                head, remaining = remaining[:max_chars], remaining[max_chars:]
            if head:
                result.append(head)
        if remaining:
            result.append(remaining)
    # Phase 4: merge small chunks with neighbours.
    if len(result) > 1:
        merged: "list[str]" = [result[0]]
        for chunk in result[1:]:
            prev = merged[-1]
            combined = prev + '\n\n' + chunk
            if _len(combined) <= max_chars:
                merged[-1] = combined
            else:
                merged.append(chunk)
        result = merged
    return [c for c in result if c]


def _chunk_newline_preferred(text, limit, len_fn):
    """Stream-consumer-derived newline-preferred splitting (no balancing)."""
    if len_fn(text) <= limit:
        return [text]
    # Reserve headroom for fence markers a balancing pass may add.
    split_limit = limit
    if "```" in text:
        split_limit = max(limit - 16, limit // 2, 1)
    from gateway.platforms.base import _custom_unit_to_cp  # heavyweight; lazy
    chunks: "list[str]" = []
    remaining = text
    while len_fn(remaining) > split_limit:
        _cp_budget = _custom_unit_to_cp(remaining, split_limit, len_fn)
        split_at = remaining.rfind("\n", 0, _cp_budget)
        if split_at < _cp_budget // 2:
            split_at = _cp_budget
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks
