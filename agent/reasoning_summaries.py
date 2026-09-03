"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models (OpenAI gpt-5.x and Responses-API relays onto the chat wire) emit one
``reasoning_content`` delta per *completed* summary part, each opening with a bold heading.
The chat wire lacks the Responses API's ``summary_index`` delimiter (verified live on Nous
Portal ``openai/gpt-5.6-sol``), so plain concatenation glues ``**One****Two**`` into one
half-bold paragraph. The boundary is re-derived from a delta opening a bold heading, matching
the blank-line join Hermes' own Responses adapter does.
"""

from __future__ import annotations

__all__ = ["separate_glued_reasoning_blocks"]


def separate_glued_reasoning_blocks(previous: str, delta: str) -> str:
    """Return *delta*, prefixed with a paragraph break when it glues onto *previous*.

    A break is inserted when *delta* opens a *closed* bold heading and *previous* is mid-line
    (heading butting heading, or prose butting heading). Token-streamed reasoning is left
    alone: its deltas carry their own whitespace, and a fragment that merely opens emphasis
    (``**`` alone) is not a part boundary — summary parts carry the whole heading in one delta.
    """
    if not previous or not delta or previous[-1].isspace():
        return delta
    if not delta.startswith("**") or "**" not in delta[2:]:
        return delta
    return f"\n\n{delta}"
