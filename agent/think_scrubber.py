"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

The regex ``run_agent._strip_think_blocks`` is correct for a complete string but,
run per-delta, erases an opening ``<think>`` that arrives alone in one delta, so
downstream state machines never see the open tag and leak reasoning. This class
centralises tag suppression upstream: partial tags at delta boundaries are held
back until resolved, and ``flush()`` releases held-back prose that was not a tag.

Usage::

    scrubber = StreamingThinkScrubber()
    for delta in stream:
        visible = scrubber.feed(delta)
        if visible:
            emit(visible)
    tail = scrubber.flush()  # at end of stream

Call ``reset()`` at the top of each turn so an interrupted block cannot taint
the next turn.  Tags handled (case-insensitive): ``<think>``, ``<thinking>``,
``<reasoning>``, ``<thought>``, ``<REASONING_SCRATCHPAD>``.

Boundary rule: an opening tag only starts a block at a block boundary (stream
start, after a newline, or with only whitespace emitted on the current line), so
prose that *mentions* ``<think>`` is not suppressed.  Closed pairs
(``<think>X</think>``) are always suppressed — a closed pair is intentional.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["StreamingThinkScrubber"]


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State: ``_in_block`` (inside an open block; text discarded), ``_buf``
    (held-back partial-tag tail), ``_last_emitted_ended_newline`` (True iff the
    last emission ended with ``\\n`` or nothing has been emitted yet — decides
    whether an open tag at buffer position 0 sits at a block boundary).
    """

    _OPEN_TAG_NAMES: Tuple[str, ...] = (
        "think",
        "thinking",
        "reasoning",
        "thought",
        "REASONING_SCRATCHPAD",
    )

    # Literal tag strings so the hot path does string ops, not regex per feed().
    _OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name}>" for name in _OPEN_TAG_NAMES)
    _CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name}>" for name in _OPEN_TAG_NAMES)
    _MAX_TAG_LEN: int = max(len(tag) for tag in _OPEN_TAGS + _CLOSE_TAGS)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        if text:
            text = self._strip_orphan_close_tags(text)
            if text:
                out.append(text)
                self._last_emitted_ended_newline = text.endswith("\n")

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion.

        Returns "" when the whole delta is reasoning content or is held back
        pending resolution of a partial tag at the boundary.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_block:
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix, drop the rest.
                    held = self._max_partial_suffix(buf, self._CLOSE_TAGS)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (no boundary gating —
            # even inline pairs are almost certainly leaked reasoning).
            # Priority 2: unterminated open tag at a block boundary (gated so
            # prose that mentions '<think>' isn't over-stripped). Earliest wins.
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            held = max(
                self._max_partial_suffix(buf, self._OPEN_TAGS),
                self._max_partial_suffix(buf, self._CLOSE_TAGS),
            )
            if held:
                self._emit(out, buf[:-held])
                self._buf = buf[-held:]
            else:
                self._emit(out, buf)
            return "".join(out)

        return "".join(out)

    def flush(self) -> str:
        """End-of-stream flush.

        Inside an unterminated block the held-back content is discarded (leaking
        partial reasoning is worse than a truncated answer); otherwise the
        held-back tail is emitted verbatim.  Always resets the boundary flag:
        intra-turn retries flush then stream again without ``reset()``, and a
        stale False flag made the new stream's opening ``<think>`` look mid-line.
        """
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_first_tag(buf: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        best_idx = -1
        best_len = 0
        for tag in tags:
            idx = buf_lower.find(tag.lower())
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_len = len(tag)
        return best_idx, best_len

    def _find_earliest_closed_pair(self, buf: str):
        """Return (start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair, else None.

        Case-insensitive and non-greedy (closest close after the open wins),
        matching ``_strip_think_blocks`` case 1; the earliest open tag wins.
        """
        buf_lower = buf.lower()
        best: "tuple[int, int] | None" = None
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_lower = open_tag.lower()
            close_lower = close_tag.lower()
            open_idx = buf_lower.find(open_lower)
            if open_idx == -1:
                continue
            close_idx = buf_lower.find(close_lower, open_idx + len(open_lower))
            if close_idx == -1:
                continue
            if best is None or open_idx < best[0]:
                best = (open_idx, close_idx + len(close_lower))
        return best

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> Tuple[int, int]:
        """Return the earliest block-boundary open-tag (idx, len), or (-1, 0)."""
        buf_lower = buf.lower()
        best_idx = -1
        best_len = 0
        for tag in self._OPEN_TAGS:
            tag_lower = tag.lower()
            search_start = 0
            while True:
                idx = buf_lower.find(tag_lower, search_start)
                if idx == -1:
                    break
                if self._is_block_boundary(buf, idx, already_emitted):
                    if best_idx == -1 or idx < best_idx:
                        best_idx = idx
                        best_len = len(tag)
                    break  # first boundary hit for this tag is enough
                search_start = idx + 1
        return best_idx, best_len

    def _is_block_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        """True iff position *idx* in *buf* is a block boundary.

        Boundary = position 0 with the prior emission ending in a newline (or
        nothing emitted yet), or any position whose preceding text on the current
        line is whitespace-only (when no newline precedes it in *buf*, the prior
        emission must also have ended with a newline).
        """
        prior_newline = (
            already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        )
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            return prior_newline and preceding.strip() == ""
        return preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
        """Longest buf-suffix that is a strict prefix of any tag (case-insensitive).

        Full-length matches are real tags handled elsewhere, not held-back partials.
        """
        if not buf:
            return 0
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), cls._MAX_TAG_LEN - 1)
        for i in range(max_check, 0, -1):
            suffix = buf_lower[-i:]
            for tag in tags:
                tag_lower = tag.lower()
                if len(tag_lower) > i and tag_lower.startswith(suffix):
                    return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open (always noise) plus trailing whitespace."""
        if "</" not in text:
            return text
        text_lower = text.lower()
        out: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            if text_lower[i:i + 2] == "</":
                for tag in cls._CLOSE_TAGS:
                    tag_lower = tag.lower()
                    tag_len = len(tag_lower)
                    if text_lower[i:i + tag_len] == tag_lower:
                        # Skip the tag and trailing whitespace (matches _strip_think_blocks case 3).
                        j = i + tag_len
                        while j < len(text) and text[j] in " \t\n\r":
                            j += 1
                        i = j
                        matched = True
                        break
            if not matched:
                out.append(text[i])
                i += 1
        return "".join(out)
