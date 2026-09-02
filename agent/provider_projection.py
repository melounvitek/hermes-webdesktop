"""Fold an agent-as-provider's own activity back into Hermes' turn state.

Some providers are *agents* (an ACP CLI behind a client shim; the codex
app-server takes an analogous path in ``agent/codex_runtime.py``): they run
their own tools inside their own session, so by the time Hermes sees the
response that work is done. Those calls must never come back as pending
``tool_calls`` (Hermes would re-run finished work), but two subsystems go blind
if they are merely summarised into ``reasoning``:

* the **self-improvement loop**, which replays ``messages`` to distil memories
  and skills;
* the **skill-review nudge**, whose ``_iters_since_skill`` counter only moves on
  Hermes tool iterations.

So the client hands both back on the completion object —
``hermes_projected_messages`` (completed ``assistant(tool_calls=[…])`` +
``tool(result)`` rows) and ``hermes_provider_tool_iterations`` — and this helper
applies them. Ordinary OpenAI-compatible clients set neither and are unaffected.
The splice is append-only through ``append_message`` so rows carry a timestamp
and persist like any other live-transcript append.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.message_metadata import append_message

logger = logging.getLogger(__name__)

__all__ = ["splice_provider_projection"]


def splice_provider_projection(
    agent: Any, response: Any, messages: list[dict[str, Any]]
) -> int:
    """Append the provider's projected history rows and tick the nudge counter.

    Returns the number of rows spliced. Tolerates absent/garbage attributes so a
    third-party OpenAI-compatible client can't break the turn.
    """
    projected = getattr(response, "hermes_projected_messages", None)
    rows = [m for m in projected if isinstance(m, dict)] if isinstance(projected, list) else []
    for row in rows:
        append_message(messages, row)
    if rows:
        logger.debug(
            "spliced %d provider-projected transcript row(s) from %s",
            len(rows),
            getattr(agent, "provider", "?"),
        )

    try:
        iterations = int(getattr(response, "hermes_provider_tool_iterations", 0) or 0)
    except (TypeError, ValueError):
        iterations = 0
    if iterations > 0:
        agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + iterations

    return len(rows)
