"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models (OpenAI gpt-5.x and anything relaying the Responses
API onto the OpenAI chat wire) emit one ``reasoning_content`` delta per
*completed* summary part, each opening with a bold heading. The Responses API
delimits parts by ``summary_index``; the chat wire carries no such field
(verified live on Nous Portal ``openai/gpt-5.6-sol``), so concatenating deltas
glues ``**One****Two**`` into one half-bold paragraph. We re-derive the
boundary from the one signal the wire keeps — a delta opening a bold heading —
matching the blank-line join Hermes' own Responses adapter already does.
"""

from __future__ import annotations

__all__ = ["separate_glued_reasoning_blocks"]


def separate_glued_reasoning_blocks(previous: str, delta: str) -> str:
    """Return *delta*, prefixed with a paragraph break when it glues onto *previous*.

    A break is inserted when *delta* opens a *closed* bold heading and
    *previous* (the accumulated reasoning; only its tail matters) is mid-line.
    Covers a heading butting a heading (``**One****Two**``) and prose butting a
    heading (``...interaction!**Next**``). Token-streamed reasoning is left
    alone: its deltas carry their own whitespace, and a fragment that merely
    opens emphasis (``**`` alone) is not a part boundary — summary parts always
    carry the whole heading in one delta.
    """
    if not previous or not delta:
        return delta
    if not delta.startswith("**"):
        return delta
    if previous[-1].isspace():
        return delta
    if "**" not in delta[2:]:
        return delta
    return f"\n\n{delta}"
