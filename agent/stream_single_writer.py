"""Best-effort accessors for the single-writer stream fence.

The fence lives on ``AIAgent`` (``_claim_stream_writer`` / ``_stream_writer_is_current``)
but is used from other streaming modules. Calling it directly would turn an
*additive* safety net into a fatal AttributeError on a partially-updated checkout,
hot-reloaded gateway, duck-typed agent, or test double (a cron job died this way).
The fence may only drop a *provably* superseded stream, never the sole writer, so
when it is unavailable or raises the correct degradation is "no fence": keep streaming.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def claim_stream_writer(agent: Any) -> int:
    """Claim the delta sink for this stream attempt; ``0`` (never fenced) when the agent lacks the fence or the claim raised."""
    claim = getattr(agent, "_claim_stream_writer", None)
    if callable(claim):
        try:
            return int(claim())
        except Exception:
            logger.debug("stream single-writer: claim failed; proceeding unfenced", exc_info=True)
    return 0


def stream_writer_is_current(agent: Any, token: int) -> bool:
    """True when ``token`` is still the active writer; a falsy token or a fence-less agent cannot prove supersession, so True."""
    if not token:
        return True
    is_current = getattr(agent, "_stream_writer_is_current", None)
    if callable(is_current):
        try:
            return bool(is_current(token))
        except Exception:
            logger.debug("stream single-writer: is_current check failed; treating as current", exc_info=True)
    return True
