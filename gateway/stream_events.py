"""Structured streaming events — the agent→gateway delivery contract.

A small typed vocabulary naming *what happened* without prescribing *how it is
delivered*. The agent emits these from its worker thread; the gateway's stream
consumer (``GatewayStreamConsumer``) is the single sink and the platform adapter
decides rendering (Telegram may stream a native draft; iMessage may drop tool
chrome). This replaced a fan of loosely-typed callbacks whose gateway side
decided both rendering and sending — the cause of tool-progress bubbles racing
the streaming draft.

Plain frozen dataclasses: no behavior, no platform knowledge, no I/O, safe to
hand across the thread/async boundary.

Invariants:
  * Events describe *transport*, never *context*. Nothing here is persisted;
    whatever the gateway chooses to "eat" must never diverge from the bytes in
    the agent's message history, which the agent alone owns.
  * Backward compatible by construction: the gateway adapts existing callbacks
    into events at the boundary; adapters that don't opt in get identical
    behavior via the base-class default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class MessageChunk:
    """A delta of streamed assistant text (reasoning/think-block content is
    filtered upstream and never arrives here); the consumer accumulates chunks
    and renders progressively (native draft on Telegram DMs, edit-in-place
    elsewhere)."""
    text: str


@dataclass(frozen=True)
class MessageStop:
    """The current assistant text segment is complete.

    ``final`` is True only for the terminal stop of the turn; an intermediate
    stop (text → tool call → more text) carries ``final=False`` so the consumer
    finalizes the current bubble and starts a fresh segment below tool chrome.
    """
    final: bool = False


@dataclass(frozen=True)
class Commentary:
    """A complete interim assistant message between tool iterations (not a delta)."""
    text: str


@dataclass(frozen=True)
class ToolCallChunk:
    """A tool invocation started. Raw facts only; the adapter decides presentation."""
    tool_name: str
    preview: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    # Monotonic per-turn index: correlates a finish with its start and lets
    # "new"-mode dedup work without the consumer tracking call order.
    index: int = 0


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool invocation completed. Tool *output* never travels here (it is history).

    The gateway uses it to clear/settle a progress bubble and to drive one-time
    onboarding hints (e.g. suggest /verbose after a long tool run).
    """
    tool_name: str
    duration: float = 0.0  # wall-clock seconds
    ok: bool = True        # returned without raising
    index: int = 0


@dataclass(frozen=True)
class LongToolHint:
    """One-shot onboarding nudge when a tool runs longer than the threshold.

    The gateway (not the agent) gates it on platform capability (the /verbose
    command must be usable) and on the user not having seen it before.
    """
    tool_name: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class GatewayNotice:
    """A gateway-originated control message.

    ``kind`` is a stable string adapters switch on (``"restart"`` / ``"online"``
    / ``"long_run"`` / …); ``text`` is the default rendering.
    """
    kind: str
    text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# Explicit union (not a marker base class) so a missing ``case`` in an
# exhaustive match is a visible type error rather than a silent fall-through.
StreamEvent = Union[
    MessageChunk, MessageStop, Commentary,
    ToolCallChunk, ToolCallFinished, LongToolHint, GatewayNotice,
]

__all__ = [
    "MessageChunk", "MessageStop", "Commentary", "ToolCallChunk",
    "ToolCallFinished", "LongToolHint", "GatewayNotice", "StreamEvent",
]
