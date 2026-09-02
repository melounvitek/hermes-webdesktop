"""CommandCode provider profiles: ``commandcode`` (chat_completions) and
``commandcode-anthropic`` (anthropic_messages, Bearer auth — see
``agent/anthropic_adapter.py``). Same key and base URL for both."""

from __future__ import annotations

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

_COMMANDCODE_BASE = "https://api.commandcode.ai/provider/v1"
_COMMANDCODE_MODELS_URL = f"{_COMMANDCODE_BASE}/models"


def _fetch_commandcode_models(
    timeout: float = 10.0,
    base_url: str | None = None,
) -> list[str] | None:
    """Fetch model IDs from the public (unauthenticated) /models endpoint.

    The picker passes base_url unconditionally, so only a value differing from
    the default counts as a customised endpoint.
    """
    caller_base = (base_url or "").strip().rstrip("/")
    custom = caller_base and caller_base != _COMMANDCODE_BASE
    models_url = caller_base + "/models" if custom else _COMMANDCODE_MODELS_URL
    try:
        req = urllib.request.Request(models_url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
    except Exception as exc:
        logger.debug("fetch_models(commandcode): %s", exc)
        return None


class CommandCodeProfile(ProviderProfile):
    """CommandCode — OpenAI-compatible chat completions endpoint."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        return _fetch_commandcode_models(timeout=timeout, base_url=base_url)


class CommandCodeAnthropicProfile(ProviderProfile):
    """CommandCode — Anthropic Messages API-compatible endpoint."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Public /models endpoint, filtered to Anthropic-family models."""
        all_models = _fetch_commandcode_models(timeout=timeout, base_url=base_url)
        if all_models is None:
            return None
        return [m for m in all_models if m.startswith("claude-")]


commandcode = CommandCodeProfile(
    name="commandcode",
    aliases=("commandcode-chat",),
    api_mode="chat_completions",
    # Same key as the anthropic profile; distinct base-URL override vars so each
    # profile renders its own card on the desktop Keys tab (rows keyed by env var).
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_BASE_URL"),
    display_name="CommandCode",
    description="CommandCode — 20+ models via OpenAI-compatible API",
    signup_url="https://commandcode.ai/",
    base_url=_COMMANDCODE_BASE,
    models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=(
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "Qwen/Qwen3.7-Max",
        "Qwen/Qwen3.6-Plus",
        "moonshotai/Kimi-K2.6",
        "zai-org/GLM-5.1",
        "MiniMaxAI/MiniMax-M2.7",
        "stepfun/Step-3.5-Flash",
        "xiaomi/mimo-v2.5-pro",
        "google/gemini-3.5-flash",
        "gpt-5.5",
    ),
    default_aux_model="deepseek/deepseek-v4-flash",
)

commandcode_anthropic = CommandCodeAnthropicProfile(
    name="commandcode-anthropic",
    aliases=("commandcode-claude",),
    api_mode="anthropic_messages",
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_ANTHROPIC_BASE_URL"),
    display_name="CommandCode (Anthropic)",
    description="CommandCode — Claude models via Anthropic Messages API",
    signup_url="https://commandcode.ai/",
    base_url=_COMMANDCODE_BASE,
    models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=("claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"),
    default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(commandcode)
register_provider(commandcode_anthropic)
