"""Structured streaming events — the agent→gateway delivery contract.

A small typed vocabulary naming *what happened* without prescribing *how it is
delivered*: the agent emits these from its worker thread, ``GatewayStreamConsumer``
is the single sink and the platform adapter decides rendering.  Plain frozen
dataclasses: no behavior, no I/O, safe across the thread/async boundary.  Events
describe *transport*, never *context* — whatever the gateway "eats" must never
diverge from the agent-owned message history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class MessageChunk:
    """A delta of streamed assistant text (think-block content is filtered upstream)."""
    text: str


@dataclass(frozen=True)
class MessageStop:
    """The current assistant text segment is complete.

    ``final`` is True only for the terminal stop of the turn; an intermediate stop
    (text → tool call → more text) makes the consumer start a fresh segment below
    tool chrome.
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
    # Monotonic per-turn index: correlates a finish with its start.
    index: int = 0


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool invocation completed. Tool *output* never travels here (it is history).

    Drives progress-bubble settling and one-time onboarding hints (LongToolHint).
    """
    tool_name: str
    duration: float = 0.0  # wall-clock seconds
    ok: bool = True        # returned without raising
    index: int = 0


@dataclass(frozen=True)
class LongToolHint:
    """One-shot onboarding nudge when a tool runs longer than the threshold.

    The gateway gates it on platform capability (/verbose usable) and first-time use.
    """
    tool_name: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class GatewayNotice:
    """A gateway-originated control message.

    ``kind`` is a stable string adapters switch on (``"restart"`` / ``"online"`` /
    ``"long_run"`` / …); ``text`` is the default rendering.
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
