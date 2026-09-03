"""Map agent activity → a :class:`PetState`.

The one place the "what is the agent doing?" → "which animation row?" decision
lives. CLI (spinner state + tool outcomes), TUI (gateway tool/message events)
and Desktop (nanostores; re-implemented in TS mirroring this priority order)
all feed it the signals they already track.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent.pet.constants import PetState


def todos_all_done(todos: Iterable[Any] | None) -> bool:
    """True iff there's ≥1 todo and every one is completed/cancelled.

    The "celebrate" beat (``JUMP``) fires when a plan finishes; mirrors the TUI's
    ``isTodoDone``. Accepts dicts (``{"status": ...}``) or objects with ``status``.
    """
    items = list(todos or [])
    return bool(items) and all(
        (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) in ("completed", "cancelled") for t in items
    )


def derive_pet_state(
    *,
    busy: bool = False,
    awaiting_input: bool = False,
    error: bool = False,
    celebrate: bool = False,
    just_completed: bool = False,
    tool_running: bool = False,
    reasoning: bool = False,
) -> PetState:
    """Resolve the animation state from coarse activity signals.

    Priority (highest first) — only one row can show at a time, so the most
    salient signal wins:

    1. ``error``          → ``FAILED``  (a tool/turn just failed)
    2. ``celebrate``      → ``JUMP``    (explicit success beat, e.g. todos done)
    3. ``just_completed`` → ``WAVE``    (turn finished cleanly / greeting)
    4. ``awaiting_input`` → ``WAITING`` (blocked on the user — outranks the in-flight
       signals below because the turn is paused on *you*, even mid tool call)
    5. ``tool_running``   → ``RUN``
    6. ``reasoning``      → ``REVIEW``
    7. ``busy``           → ``RUN``     (turn in flight, unspecified work)
    8. otherwise          → ``IDLE``
    """
    ranked = (
        (error, PetState.FAILED),
        (celebrate, PetState.JUMP),
        (just_completed, PetState.WAVE),
        (awaiting_input, PetState.WAITING),
        (tool_running, PetState.RUN),
        (reasoning, PetState.REVIEW),
        (busy, PetState.RUN),
    )
    return next((state for flag, state in ranked if flag), PetState.IDLE)
