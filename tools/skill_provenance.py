"""Skill write-origin provenance — a ContextVar separating agent-sediment
skill writes from foreground user-directed writes.

The curator only consolidates/prunes skills it autonomously created via the
background self-improvement review fork; skills a user asked a foreground agent
to write belong to the user and must never be auto-curated. run_agent.py binds
the origin before each tool loop (mirroring AIAgent._memory_write_origin, which
is "background_review" for review-fork instances) so tool handlers such as
skill_manage create can check it.

Usage::

    token = set_current_write_origin("background_review")
    try:
        ...  # tool runs here
    finally:
        reset_current_write_origin(token)
"""

import contextvars


_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin",
    default="foreground",
)

# Sentinel used by the background review fork (run_agent._spawn_background_review).
BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind the active write origin; pass the Token to reset_current_write_origin."""
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior write origin context."""
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """Return the active write origin: "foreground" for any regular agent
    (CLI, gateway, cron, subagent); "background_review" for the review fork."""
    return _write_origin.get()


def is_background_review() -> bool:
    """True iff the current write origin is the background review fork."""
    return get_current_write_origin() == BACKGROUND_REVIEW
