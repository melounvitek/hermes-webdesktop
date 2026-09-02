"""LM Studio reasoning-effort resolution shared by the chat-completions
transport and run_agent's iteration-limit summary path.

LM Studio publishes per-model ``capabilities.reasoning.allowed_options``
(``["off","on"]`` for toggle models, ``["off","minimal","low"]`` for graduated
ones). We map the user's ``reasoning_config`` onto LM Studio's OpenAI-compatible
vocabulary, then clamp against the model's allowed set so the server doesn't 400.
"""

from __future__ import annotations

from typing import List, Optional

# Top-level reasoning_effort values LM Studio's OpenAI-compatible endpoint accepts.
_LM_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

# Toggle-style models publish allowed_options as ["off","on"]; map onto the
# request vocabulary. Also applied to the published allowed_options themselves.
_LM_EFFORT_ALIASES = {"off": "none", "on": "medium"}

# Hermes' ladder grew past LM Studio's vocabulary ("max", "ultra"). Without this
# ceiling clamp they miss _LM_VALID_EFFORTS, keep the "medium" default and are
# conflated with unparseable input — asking for more yields less than "xhigh".
# Kept separate from _LM_EFFORT_ALIASES, which must not rewrite allowed_options.
_LM_EFFORT_CLAMP = {"max": "xhigh", "ultra": "xhigh"}


def resolve_lmstudio_effort(
    reasoning_config: Optional[dict],
    allowed_options: Optional[List[str]],
) -> Optional[str]:
    """Return the ``reasoning_effort`` to send to LM Studio, or ``None``.

    ``None`` means "omit the field": the user picked a level the model can't
    honor, so LM Studio falls back to the model's declared default rather than
    a silently substituted effort. Falsy ``allowed_options`` (probe failed)
    skips clamping and sends the resolved effort anyway.
    """
    effort = "medium"
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            effort = "none"
        else:
            raw = (reasoning_config.get("effort") or "").strip().lower()
            raw = _LM_EFFORT_ALIASES.get(raw, raw)
            raw = _LM_EFFORT_CLAMP.get(raw, raw)
            if raw in _LM_VALID_EFFORTS:
                effort = raw
    if allowed_options:
        allowed = {_LM_EFFORT_ALIASES.get(opt, opt) for opt in allowed_options}
        if effort not in allowed:
            return None
    return effort
