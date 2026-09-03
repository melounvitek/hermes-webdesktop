"""OpenRouter provider profile."""

import logging
from typing import Any

from agent.portal_tags import get_affinity_scope, get_conversation_context
from agent.transports.codex import _cache_scope_from_session_id
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None

# Legacy allowlist of Anthropic models that still accept an explicit "disable
# thinking" request. Claude 4.6+ and newer named models mandate reasoning and
# 400 on any disable form, so *unknown* Anthropic models default to "cannot
# disable" (mirrors agent/anthropic_adapter._get_anthropic_max_output).
_ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS = (
    "claude-3",          # 3, 3.5, 3.7
    "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
    "claude-sonnet-4-0", "claude-sonnet-4.0",
    "claude-opus-4-2025", "claude-sonnet-4-2025",  # date-stamped 4.0 IDs
    "claude-opus-4-5", "claude-opus-4.5",
    "claude-sonnet-4-5", "claude-sonnet-4.5",
    "claude-haiku-4-5", "claude-haiku-4.5",
)


def _anthropic_reasoning_is_mandatory(model: str | None) -> bool:
    """True for Anthropic models that reject any disable-thinking form (unknown -> True)."""
    m = (model or "").lower()
    if not m.startswith(("anthropic/", "claude")) and "claude" not in m:
        return False
    return not any(sub in m for sub in _ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS)


def _sticky_key(session_id: str | None) -> str | None:
    """Declared routing scope, then ambient conversation, then explicit session_id.
    Aux call sites (compression, titles, vision, MoA…) pass no ``session_id``,
    so the ambient lineage ROOT keeps them pinned to their conversation."""
    return _cache_scope_from_session_id(get_affinity_scope() or get_conversation_context() or session_id)


class OpenRouterProfile(ProviderProfile):
    """OpenRouter aggregator — provider preferences, reasoning config passthrough."""

    @staticmethod
    def _clamp_reasoning_to_catalog(cfg: dict[str, Any], model: str | None) -> dict[str, Any]:
        """Clamp ``cfg["effort"]`` to the nearest LOWER catalog-advertised level. No-op when the
        catalog is unreachable, the model is unlisted, or no supported_efforts list is published."""
        effort = cfg.get("effort")
        if not effort or cfg.get("enabled") is False:
            return cfg
        try:
            from hermes_cli.models import clamp_reasoning_effort_to_supported, openrouter_model_reasoning_capabilities

            caps = openrouter_model_reasoning_capabilities(model)
            if not caps or not caps.get("supports_reasoning"):
                return cfg
            clamped = clamp_reasoning_effort_to_supported(effort, caps.get("supported_efforts"))
        except Exception:
            return cfg
        if clamped and clamped != effort:
            logger.debug(
                "openrouter: clamped reasoning effort %r → %r for %s "
                "(catalog supported_efforts=%s)",
                effort, clamped, model, caps.get("supported_efforts"),
            )
            cfg = {**cfg, "effort": clamped}
        return cfg

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Public OpenRouter catalog (no auth), cached per process. Tool-call
        filtering happens in hermes_cli/models.py, which the picker reaches first."""
        global _CACHE  # noqa: PLW0603
        if _CACHE is not None:
            return _CACHE
        try:
            result = super().fetch_models(api_key=None, base_url=base_url, timeout=timeout)
        except Exception as exc:
            logger.debug("fetch_models(openrouter): %s", exc)
            return None
        if result is not None:
            _CACHE = result
        return result

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        body: dict[str, Any] = {}
        # Top-level session_id is OpenRouter's sticky routing key (used directly,
        # not hashed from the opening messages; active from the first request).
        sticky_key = _sticky_key(session_id)
        if sticky_key:
            body["session_id"] = sticky_key
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs
        # Pareto Code router plugin is only meaningful for openrouter/pareto-code.
        score = context.get("openrouter_min_coding_score")
        if (context.get("model") or "") == "openrouter/pareto-code" and score is not None and score != "":
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = None
            if score_f is not None and 0.0 <= score_f <= 1.0:
                body["plugins"] = [{"id": "pareto-router", "min_coding_score": score_f}]
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pass reasoning_config as extra_body.reasoning; pin Grok's cache via x-grok-conv-id."""
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if supports_reasoning:
            # Reasoning-mandatory Anthropic models use adaptive thinking: any
            # ``reasoning`` field (disable, or an enabled form on a tool-continuation
            # turn without a replayed thinking block) makes OpenRouter emit
            # ``thinking: {type: "disabled"}`` -> 400. Omit it; the user's effort
            # still reaches Anthropic's output_config.effort via top-level ``verbosity``.
            if _anthropic_reasoning_is_mandatory(model):
                cfg = reasoning_config or {}
                effort = cfg.get("effort")
                if cfg.get("enabled", True) is not False and effort and effort != "none":
                    top_level["verbosity"] = effort
            elif reasoning_config is not None:
                extra_body["reasoning"] = self._clamp_reasoning_to_catalog(dict(reasoning_config), model)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        # xAI's prompt cache is pinned per backend server via this header.
        grok_conv_id = _sticky_key(session_id)
        if grok_conv_id and model and model.startswith(("x-ai/grok-", "xai/grok-")):
            top_level["extra_headers"] = {"x-grok-conv-id": grok_conv_id}
        return extra_body, top_level


openrouter = OpenRouterProfile(
    name="openrouter",
    aliases=("or",),
    env_vars=("OPENROUTER_API_KEY",),
    display_name="OpenRouter",
    description="OpenRouter — unified API for 200+ models",
    signup_url="https://openrouter.ai/keys",
    base_url="https://openrouter.ai/api/v1",
    models_url="https://openrouter.ai/api/v1/models",
    fallback_models=(
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "google/gemini-3.8-flash",
        "qwen/qwen3-plus",
    ),
)

register_provider(openrouter)
