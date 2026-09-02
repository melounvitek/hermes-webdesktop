"""Per-model stale-timeout FLOOR for known reasoning models.

Reasoning models (extended thinking before the first content token) routinely
exceed the default chat-model stale detectors (stream ``HERMES_STREAM_STALE_TIMEOUT``
180s, non-stream ``HERMES_API_CALL_STALE_TIMEOUT`` 90s): upstream proxies /
load-balancers idle-kill the stream mid-think, surfacing as
``BrokenPipeError``/``RemoteProtocolError`` on the next read. The existing
stale-detector scaling consults :func:`get_reasoning_stale_timeout_floor` and
applies ``max(default, floor)``. Being a floor it:

* never overrides explicit user config (``providers.<id>.models.<model>.
  stale_timeout_seconds`` / ``request_timeout_seconds`` win — this never runs
  in that branch);
* never lowers an existing threshold;
* has zero effect on non-allowlisted models (resolver returns ``None``).

Matching is start-anchored on the slug after any aggregator prefix
(``openai/``, ``x-ai/``) with an end-or-separator right anchor, so
``qwen3-235b`` matches ``qwen3`` but ``some-other-qwen3`` and a hypothetical
``llama-4-70b-o1-preview`` do not trigger the ``o1`` floor.
"""

from __future__ import annotations

import re
from typing import Optional


# (slug, floor_seconds). Order irrelevant — longest slug wins at match time.
_REASONING_STALE_TIMEOUT_FLOORS: tuple[tuple[str, int], ...] = (
    # NVIDIA Nemotron behind hosted NIM: documented 60-180s upstream idle kill.
    ("nemotron-3-ultra", 600),
    ("nemotron-3-super", 600),
    ("nemotron-3-nano",  300),
    ("nemotron-3.5-lightning", 300),
    # DeepSeek R1 / V4 (reasoning_content streamed before final content).
    ("deepseek-r1", 600),
    ("deepseek-reasoner", 600),
    ("deepseek-v4-flash", 600),
    ("deepseek-v4-pro", 600),
    # Qwen QwQ + the qwen3 family. Instruct variants also match ``qwen3`` —
    # accepted: a slightly longer wait on a hung provider beats a pattern
    # (``qwen3-.*-thinking``) that breaks on the next naming shape.
    ("qwq-32b", 300),
    ("qwen3", 180),
    # OpenAI o-series: each variant enumerated so bare ``o1`` cannot
    # over-match ``olmo-1`` or community derivatives.
    ("o1", 600),
    ("o1-mini", 600),
    ("o1-pro", 600),
    ("o1-preview", 600),
    ("o3", 600),
    ("o3-pro", 600),
    ("o3-mini", 300),
    ("o4-mini", 300),
    # Anthropic Claude 4.x+ thinking variants (anchored so 3.x never matches).
    ("claude-opus-4", 240),
    ("claude-opus-5", 240),
    ("claude-sonnet-5", 180),
    ("claude-sonnet-4.5", 180),
    ("claude-sonnet-4.6", 180),
    # Mythos-class named models (claude-fable-5): 1M ctx + 128K output, a
    # heavier thinking phase than the numbered line — deep-reasoning tier,
    # otherwise the stale detector trips the cross-turn circuit breaker.
    ("claude-fable", 600),
    # xAI Grok: explicit reasoning / non-reasoning pairs only, so bare
    # ``grok-3``/``grok-4`` fast variants don't inherit the 300s floor.
    ("grok-4-fast-reasoning", 300),
    ("grok-4.20-reasoning", 300),
    ("grok-4.5", 300),
    ("grok-4.6", 300),
    ("grok-4-fast-non-reasoning", 180),
    # "Ox Alpha" stealth reasoning model (OpenRouter / OpenCode Zen slugs).
    ("ox-alpha", 300),
    ("x-preview-f-free", 300),
    # Thinking Machines Inkling; covers inkling-small and :free SKUs.
    ("inkling", 300),
)


# Pre-compiled once at import (immutable afterwards — safe under free-threaded
# Python). Right anchor: end-of-string or a slug separator; ``:`` is included
# because OpenRouter routing suffixes (``:free``, ``:nitro``) attach directly
# to the slug. Sorted longest-first so ``o3-mini`` beats ``o3``.
_SORTED_REASONING_FLOORS: list[tuple[str, float, re.Pattern[str]]] = [
    (slug, floor, re.compile(r"^" + re.escape(slug) + r"(?:$|[\-._:])"))
    for slug, floor in sorted(
        _REASONING_STALE_TIMEOUT_FLOORS, key=lambda kv: -len(kv[0])
    )
]


def get_reasoning_stale_timeout_floor(model: object) -> Optional[float]:
    """Return the stale-timeout floor (seconds) for a known reasoning model.

    ``None`` when the model is not allowlisted or the argument is empty / not
    a string. The aggregator prefix (everything up to the last ``/``) is
    stripped so the slug is matched start-anchored. Callers apply this as
    ``max(default, floor)`` and only when no explicit per-model
    ``stale_timeout_seconds`` is configured.

    >>> get_reasoning_stale_timeout_floor("nvidia/nemotron-3-ultra-550b-a55b")
    600.0
    >>> get_reasoning_stale_timeout_floor("openai/o3-mini")
    300.0
    >>> get_reasoning_stale_timeout_floor("deepseek/deepseek-r1")
    600.0
    >>> get_reasoning_stale_timeout_floor("deepseek/deepseek-v4-flash")
    600.0
    >>> get_reasoning_stale_timeout_floor("deepseek/deepseek-v4-pro")
    600.0
    >>> get_reasoning_stale_timeout_floor("qwen/qwen3-235b-a22b-thinking")
    180.0
    >>> get_reasoning_stale_timeout_floor("x-ai/grok-4-fast-reasoning")
    300.0
    >>> get_reasoning_stale_timeout_floor("anthropic/claude-opus-4-6")
    240.0
    >>> get_reasoning_stale_timeout_floor("anthropic/claude-fable-5")
    600.0
    >>> get_reasoning_stale_timeout_floor("gpt-4o") is None
    True
    >>> get_reasoning_stale_timeout_floor("olmo-1") is None
    True
    >>> get_reasoning_stale_timeout_floor(None) is None
    True
    """
    if not model or not isinstance(model, str):
        return None
    name = model.strip().lower()
    if not name:
        return None
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    for _slug, floor, pattern in _SORTED_REASONING_FLOORS:
        if pattern.search(name):
            return float(floor)
    return None
