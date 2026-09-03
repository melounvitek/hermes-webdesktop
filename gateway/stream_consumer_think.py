"""Think-block filtering for GatewayStreamConsumer.

Some models emit inline <think>...</think> blocks in content.  The agent strips
them from the final response, but intermediate edits go out before that, so this
mirrors the CLI's _stream_delta state machine."""

from __future__ import annotations

import logging

logger = logging.getLogger("gateway.stream_consumer")


class StreamThinkFilterMixin:
    """Progressive <think>-tag suppression over streamed deltas."""

    # Must stay in sync with cli.py _OPEN_TAGS/_CLOSE_TAGS and
    # run_agent.py _strip_think_blocks() tag variants.
    _OPEN_THINK_TAGS = (
        "<REASONING_SCRATCHPAD>", "<think>", "<reasoning>",
        "<THINKING>", "<thinking>", "<thought>",
    )
    _CLOSE_THINK_TAGS = (
        "</REASONING_SCRATCHPAD>", "</think>", "</reasoning>",
        "</THINKING>", "</thinking>", "</thought>",
    )

    def _at_block_boundary(self, buf: str, idx: int) -> bool:
        """Tag at ``idx`` starts a block: start of text, or newline + optional whitespace.

        Prose that merely *mentions* a tag must not trigger (mirrors cli.py).
        """
        acc_boundary = not self._accumulated or self._accumulated.endswith("\n")
        if idx == 0:
            return acc_boundary
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            return acc_boundary and preceding.strip() == ""
        return preceding[last_nl + 1:].strip() == ""

    def _earliest_open_tag(self, buf: str, lower_buf: str) -> "tuple[int, int]":
        """(index, length) of the earliest block-boundary opening tag, or (-1, 0)."""
        best_idx, best_len = -1, 0
        for tag in self._OPEN_THINK_TAGS:
            tag_lower = tag.lower()
            search_start = 0
            while (idx := lower_buf.find(tag_lower, search_start)) != -1:
                if self._at_block_boundary(buf, idx):
                    if best_idx == -1 or idx < best_idx:
                        best_idx, best_len = idx, len(tag)
                    break  # first boundary hit for this tag is enough
                search_start = idx + 1
        return best_idx, best_len

    def _filter_and_accumulate(self, text: str) -> None:
        """Append a delta to the buffer, discarding think blocks.

        Partial tags at buffer boundaries are held in ``_think_buffer`` until
        enough characters arrive to decide.
        """
        buf = self._think_buffer + text
        self._think_buffer = ""

        while buf:
            # Case-insensitive: models emit <Think>, <THINKING>, …
            lower_buf = buf.lower()
            if self._in_think_block:
                best_idx, best_len = -1, 0
                for tag in self._CLOSE_THINK_TAGS:
                    idx = lower_buf.find(tag.lower())
                    if idx != -1 and (best_idx == -1 or idx < best_idx):
                        best_idx, best_len = idx, len(tag)
                if best_len:
                    self._in_think_block = False
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold a tail that could be a partial close tag; discard the rest.
                    max_tag = max(len(t) for t in self._CLOSE_THINK_TAGS)
                    self._think_buffer = buf[-max_tag:] if len(buf) > max_tag else buf
                    return
            else:
                best_idx, best_len = self._earliest_open_tag(buf, lower_buf)
                if best_len:
                    self._append_accumulated(buf[:best_idx])
                    self._in_think_block = True
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold back a partial open tag at the tail.
                    held_back = max(
                        (i for tag in self._OPEN_THINK_TAGS for i in range(1, len(tag))
                         if lower_buf.endswith(tag.lower()[:i])),
                        default=0,
                    )
                    if held_back:
                        self._append_accumulated(buf[:-held_back])
                        self._think_buffer = buf[-held_back:]
                    else:
                        # An orphan </think> (thinking-mode toggle dropped the open, or
                        # incomplete upstream stripping) is noise.
                        self._append_accumulated(self._strip_orphan_close_tags(buf))
                    return

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags (plus trailing whitespace) that have no matching open.

        Mirrors ``agent/think_scrubber.py::StreamingThinkScrubber`` so the
        progressive display matches the post-stream scrubber.
        """
        if "</" not in text:
            return text
        text_lower = text.lower()
        out: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            if text_lower[i:i + 2] == "</":
                for tag in cls._CLOSE_THINK_TAGS:
                    tag_lower = tag.lower()
                    if text_lower.startswith(tag_lower, i):
                        j = i + len(tag_lower)
                        while j < len(text) and text[j] in " \t\n\r":
                            j += 1
                        i = j
                        matched = True
                        break
            if not matched:
                out.append(text[i])
                i += 1
        return "".join(out)

    def _flush_think_buffer(self) -> None:
        """On stream end, flush text held back waiting for a possible open tag."""
        if self._think_buffer and not self._in_think_block:
            self._append_accumulated(self._strip_orphan_close_tags(self._think_buffer))
            self._think_buffer = ""
