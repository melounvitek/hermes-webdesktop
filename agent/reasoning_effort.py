"""Canonical reasoning-effort vocabulary and wire clamping.

Hermes' internal effort ladder (``hermes_constants.VALID_REASONING_EFFORTS``
plus ``none``) is wider than any single provider wire accepts. Hand-rolled
per-transport translation maps produced two recurring bugs: a new internal
level (``ultra``) leaking to a wire that 400s on it, and an unknown level
dropped to a weak default so the strongest ask resolved *weaker* than an
explicit ``high`` (ladder inversion). This module is the single source of
truth instead:

- :data:`EFFORT_LADDER` — canonical low→high ordering.
- :func:`clamp_effort` — keep a supported level verbatim, else the **nearest
  weaker** supported level (never silently escalate cost); only when nothing
  weaker exists take the weakest supported level (GLM-5.2's floor is ``high``).
- Named wire-vocabulary constants so call sites declare *data*, not logic.

Rules for call sites:
1. Wire shape (``extra_body.reasoning`` vs top-level ``reasoning_effort`` vs
   a ``thinking`` toggle) stays local; only the vocabulary math lives here.
2. Unset stays unset: ``clamp_effort`` translates an explicit request, never
   invents one — omit the field so the server default applies.
3. Never patch a predicate: when a provider rejects a level, fix its declared
   supported set (data), not the call site.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

#: Matches ``k3`` as a delimited token (``k3``, ``k3-256k``, ``kimi-k3-cot``)
#: without matching K2-era names (``kimi-k2.6``).
_KIMI_K3_SLUG_RE = re.compile(r"(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)")

# Canonical low→high ordering for nearest-level clamping. Includes "none" so an
# explicit disable can be clamped when a provider publishes it as a level.
EFFORT_LADDER: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)

# ``ultra`` is Hermes-internal (the Codex product tier); no wire accepts it, so
# every declared set below stops at ``max`` and ``ultra`` always clamps down.

#: Widest OpenAI-compatible wire vocabulary (OpenRouter, Nous Portal).
OPENAI_COMPAT_WIRE_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
)

#: OpenAI/Codex Responses, per model generation (live-verified): ``minimal``
#: is rejected by both (clamps to low); ``max`` is gpt-5.6-only.
CODEX_GPT56_EFFORTS: tuple[str, ...] = (
    "none", "low", "medium", "high", "xhigh", "max",
)
CODEX_LEGACY_EFFORTS: tuple[str, ...] = (
    "none", "low", "medium", "high", "xhigh",
)


def codex_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for an OpenAI/Codex Responses model."""
    if "gpt-5.6" in (model or "").lower():
        return CODEX_GPT56_EFFORTS
    return CODEX_LEGACY_EFFORTS


#: xAI Responses — Grok 4.6+ accepts xhigh; older Grok tops out at high.
XAI_GROK46_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
XAI_LEGACY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Actual Computer relays (SGLang/vLLM).
ACTUAL_RELAY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")

#: Moonshot/Kimi K3 (server default high) vs K2-era models.
KIMI_K3_EFFORTS: tuple[str, ...] = ("low", "high", "max")
KIMI_K2_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: OpenCode "Ox Alpha" (x-preview-f-free): thinking cannot be disabled and the
#: wire accepts exactly low/high/max (medium/none/xhigh 400); xhigh rounds up.
OX_ALPHA_EFFORTS: tuple[str, ...] = ("low", "high", "max")
OX_ALPHA_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Tencent TokenHub.
TOKENHUB_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Nebius Token Factory (top-level reasoning_effort knob).
NEBIUS_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Kimi K3 vendor-documented quirks: ``high`` is K3's positional middle AND
#: server default, so ``medium`` rounds to it rather than down to ``low``;
#: ``xhigh`` rounds up to ``max`` (K3's top tier).
KIMI_K3_OVERRIDES: dict[str, str] = {"medium": "high", "xhigh": "max"}

#: GLM-5.2 native knob: exactly ``high`` (its minimum thinking level) and
#: ``max``; ``xhigh`` requests the top tier, not the floor.
GLM52_EFFORTS: tuple[str, ...] = ("high", "max")
GLM52_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: GLM-5.3 widens the knob to a graded scale (live-verified, monotonic
#: reasoning-token scaling); ``xhigh`` requests the top tier.
GLM53_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
GLM53_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: DeepSeek V4 OpenAI-compat endpoint; ``xhigh`` requests the top tier.
DEEPSEEK_V4_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
DEEPSEEK_V4_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Ollama Cloud /v1/chat/completions: rejects ``minimal`` with HTTP 400.
OLLAMA_CLOUD_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")
OLLAMA_CLOUD_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Meta Model API (Muse): rejects ``none``.
META_AI_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")

#: Upstage Solar Pro/Open.
SOLAR_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


def kimi_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for a Moonshot/Kimi slug.

    K3 is served as bare ``k3``, plan variants (``k3-256k``) and ``kimi-k3*``
    aliases; everything earlier speaks low/medium/high. Boundary-matched so
    K2-era names (``kimi-k2.6``) never match.
    """
    m = (model or "").strip().lower().split("/")[-1]
    if _KIMI_K3_SLUG_RE.search(m):
        return KIMI_K3_EFFORTS
    return KIMI_K2_EFFORTS


def clamp_effort(
    effort: Optional[str],
    supported: Optional[Sequence[str]],
    overrides: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Clamp a requested reasoning effort onto a wire's supported levels.

    ``overrides`` (a declared vendor mapping, e.g. Kimi K3 ``medium → high``)
    is consulted first. Otherwise the request passes through unchanged when it
    is supported, when the supported set is unknown/empty, or when it isn't a
    recognized ladder level (custom providers may use bespoke names). Else the
    **nearest weaker** supported level is returned so a clamp never escalates
    cost; when nothing weaker exists, the weakest supported level is (the
    provider's floor is the closest honest match). Monotonic: a stronger
    request never resolves weaker than a weaker request would.
    """
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [
        str(level).strip().lower()
        for level in supported
        if str(level).strip().lower() in EFFORT_LADDER
    ]
    if not supported_norm or requested in supported_norm:
        return effort
    if overrides:
        mapped = overrides.get(requested)
        if mapped in supported_norm:
            return mapped
    if requested not in EFFORT_LADDER:
        return effort
    # "none" disables reasoning — never a degradation target for an enabled
    # ask (clamping "minimal" to "none" would silently switch thinking off).
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_LADDER.index(requested)
    below = [
        level for level in candidates
        if EFFORT_LADDER.index(level) < requested_idx
    ]
    if below:
        return max(below, key=EFFORT_LADDER.index)
    return min(candidates, key=EFFORT_LADDER.index)


def requested_effort(reasoning_config: Optional[dict]) -> Optional[str]:
    """Extract the user's explicit effort from a reasoning config, or None.

    None when the config is absent/malformed, carries no effort, or reasoning
    is explicitly disabled — callers then omit the wire field (rule 2 above).
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return None
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort or None
