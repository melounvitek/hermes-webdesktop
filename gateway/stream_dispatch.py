"""Adapter-driven dispatch of structured stream events to a delivery sink.

``GatewayEventDispatcher`` holds an adapter, the stream consumer (sink) and the
resolved per-channel presentation settings, and routes each typed event
(gateway/stream_events.py) through the adapter's render hooks. Message events
flow into the consumer; tool events are formatted by the adapter — which may
return None to *eat* them on platforms without tool chrome — and enqueued onto
the same tool-progress queue the gateway drains, so the two paths never race.

No platform knowledge and no asyncio: a thin synchronous router callable from
the agent's worker thread, exactly like the callbacks it replaced.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from gateway.stream_events import (
    Commentary, GatewayNotice, LongToolHint, MessageChunk, MessageStop, StreamEvent, ToolCallChunk,
)

logger = logging.getLogger("gateway.stream_events")


class GatewayEventDispatcher:
    """Route typed stream events through an adapter onto a delivery sink.

    adapter: provides ``render_message_event`` / ``format_tool_event``
        (BasePlatformAdapter defaults reproduce legacy behavior).
    sink: the GatewayStreamConsumer; None when streaming is disabled (message
        events are dropped — the final response still goes out normally).
    enqueue_tool_line: puts a rendered tool-progress line on the gateway's
        progress queue; None when tool progress is disabled for the channel.
    tool_mode: "all" / "new" / "verbose" / "off".
    preview_max_len: resolved ``tool_preview_length`` (0 = no cap in verbose).
    on_long_tool / on_notice: optional hooks so the gateway owns the
        "should I surface this here?" decision.
    """

    def __init__(
        self,
        adapter: Any,
        sink: Any = None,
        *,
        enqueue_tool_line: Optional[Callable[[Any], None]] = None,
        tool_mode: str = "all",
        preview_max_len: int = 40,
        on_long_tool: Optional[Callable[[LongToolHint], None]] = None,
        on_notice: Optional[Callable[[GatewayNotice], None]] = None,
    ) -> None:
        self.adapter = adapter
        self.sink = sink
        self._enqueue_tool_line = enqueue_tool_line
        self.tool_mode = tool_mode or "all"
        self.preview_max_len = preview_max_len
        self._on_long_tool = on_long_tool
        self._on_notice = on_notice
        self._last_tool: Optional[str] = None  # "new"-mode dedup

    def dispatch(self, event: StreamEvent) -> None:
        """Route a single event.  Never raises into the agent's worker thread."""
        try:
            self._dispatch(event)
        except Exception:  # presentation must never break the agent loop
            logger.debug("stream-event dispatch error", exc_info=True)

    def _dispatch(self, event: StreamEvent) -> None:
        if isinstance(event, (MessageChunk, MessageStop, Commentary)):
            if self.sink is not None:
                self.adapter.render_message_event(event, self.sink)
        elif isinstance(event, ToolCallChunk):
            self._dispatch_tool_call(event)
        elif isinstance(event, LongToolHint) and self._on_long_tool is not None:
            self._on_long_tool(event)
        elif isinstance(event, GatewayNotice) and self._on_notice is not None:
            self._on_notice(event)
        # ToolCallFinished: no chrome on completion (only "started" is rendered);
        # completion only drives onboarding hints (LongToolHint).

    def _dispatch_tool_call(self, event: ToolCallChunk) -> None:
        if self.tool_mode == "off" or self._enqueue_tool_line is None:
            return
        if self.tool_mode == "new" and event.tool_name == self._last_tool:
            return
        self._last_tool = event.tool_name
        line = self.adapter.format_tool_event(
            event, mode=self.tool_mode, preview_max_len=self.preview_max_len,
        )
        if line:  # None/"" == adapter chose to eat this event
            self._enqueue_tool_line(line)


__all__ = ["GatewayEventDispatcher"]
