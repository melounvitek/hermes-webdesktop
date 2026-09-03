"""Provider reasoning-parameter policy for ``AIAgent``.

When ``reasoning`` extra_body is safe to send, LM Studio / Ollama / GitHub Models capability probes,
``reasoning_content`` echo families, and strict-API tool-call sanitising.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import time
from typing import Optional

from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.message_sanitization import matches_reasoning_echo_family
from utils import base_url_host_matches

# Static OpenRouter fallback when the live /v1/models capability cache is cold.
_OPENROUTER_REASONING_PREFIXES = (
    "deepseek/", "anthropic/", "openai/", "x-ai/", "google/gemini-2", "google/gemma-4",
    "qwen/qwen3", "tencent/hy", "xiaomi/",
)

# Probe results cache per (model, base_url). Definitive values cache permanently; an
# "unknown" value (empty list / None) caches 60s so a transient failure neither sticks
# for the session nor round-trips every turn.
_PROBE_TTL_S = 60


class ReasoningParamsMixin:
    """Reasoning-parameter gating and echo policy (see module docstring)."""

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body upstream and some routes 400 on ``reasoning``; gate to known
        reasoning-capable families and direct Nous Portal.
        """
        url = self._base_url_lower
        if base_url_host_matches(url, "nousresearch.com") or base_url_host_matches(url, "ai-gateway.vercel.sh"):
            return True
        if base_url_host_matches(url, "models.github.ai") or base_url_host_matches(url, "githubcopilot.com"):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if (self.provider or "").strip().lower() == "lmstudio":
            # "off-only" (or absent) means no real reasoning capability.
            return any(opt and opt != "off" for opt in self._lmstudio_reasoning_options_cached())
        if base_url_host_matches(url, "ollama.com"):
            # Ollama Cloud: /api/show capabilities are authoritative.
            return self._ollama_supports_thinking_cached()
        if not self._is_openrouter_url() or base_url_host_matches(url, "api.mistral.ai"):
            return False
        # Live-catalog metadata first (OpenRouter /v1/models supported_parameters) — the static prefix
        # allowlist repeatedly went stale one vendor at a time. Unknown falls back to the static list.
        try:
            from hermes_cli.models import openrouter_model_reasoning_capabilities, warm_openrouter_reasoning_caps_async
            caps = openrouter_model_reasoning_capabilities(self.model)
            if caps is None:
                warm_openrouter_reasoning_caps_async()  # cache cold — warm in the background, never block
        except Exception:
            caps = None
        if caps is not None:
            return bool(caps.get("supports_reasoning"))
        model = (self.model or "").lower()
        return any(model.startswith(prefix) for prefix in _OPENROUTER_REASONING_PREFIXES)

    def _cached_probe(self, cache_attr: str, probe, unknown, definitive):
        """Run ``probe(model, base_url, api_key)`` once per (model, base_url) with the module TTL policy.

        ``unknown`` is what a raising probe yields; ``definitive(value)`` decides permanent vs 60s TTL.
        """
        cache = getattr(self, cache_attr, None)
        if cache is None:
            cache = {}
            setattr(self, cache_attr, cache)
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            value, ts = cached
            if definitive(value) or (time.monotonic() - ts) < _PROBE_TTL_S:
                return value
        try:
            value = probe(self.model, self.base_url, getattr(self, "api_key", ""))
        except Exception:
            value = unknown
        cache[key] = (value, time.monotonic())
        return value

    def _lmstudio_reasoning_options_cached(self) -> list[str]:
        """LM Studio's published reasoning ``allowed_options`` (gate + clamp so toggle models don't 400 on ``high``)."""
        try:
            from hermes_cli.models import lmstudio_model_reasoning_options
        except Exception:
            return []
        return self._cached_probe("_lm_reasoning_opts_cache", lmstudio_model_reasoning_options, [], bool)

    def _ollama_supports_thinking_cached(self) -> bool:
        """True only if Ollama's ``/api/show`` declares the ``thinking`` capability."""
        try:
            from hermes_cli.models import ollama_model_supports_thinking
        except Exception:
            return False
        return bool(self._cached_probe("_ollama_thinking_cache", ollama_model_supports_thinking, None, lambda v: v is not None))

    def _resolve_lmstudio_summary_reasoning_effort(self) -> Optional[str]:
        """Safe top-level ``reasoning_effort`` for LM Studio; shared with the iteration-limit summary call."""
        from agent.lmstudio_reasoning import resolve_lmstudio_effort
        return resolve_lmstudio_effort(self.reasoning_config, self._lmstudio_reasoning_options_cached())

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported = github_model_reasoning_efforts(self.model)
        if not supported:
            return None

        cfg = self.reasoning_config if isinstance(self.reasoning_config, dict) else {}
        if cfg.get("enabled") is False:
            return None
        effort = str(cfg.get("effort", "medium")).strip().lower()

        if effort == "xhigh" and "xhigh" not in supported and "high" in supported:
            effort = "high"
        elif effort not in supported:
            if effort == "minimal" and "low" in supported:
                effort = "low"
            elif "medium" in supported:
                effort = "medium"
            else:
                effort = supported[0]
        return {"effort": effort}

    _build_assistant_message = _forward("agent.chat_completion_helpers", "build_assistant_message")

    def _needs_thinking_reasoning_pad(self) -> bool:
        """Return True when the active provider enforces ``reasoning_content`` echo-back on tool-call replays.

        DeepSeek thinking, Kimi/Moonshot thinking and Xiaomi MiMo thinking all 400 without it. Cached per
        (provider, model, base_url) and invalidated by ``switch_model()`` / ``_try_activate_fallback()`` —
        the loop calls this ~16× per turn and each miss re-runs several ``urlparse`` host matches.
        """
        key = (self.provider, self.model, getattr(self, "_base_url_lower", self.base_url))
        cached = getattr(self, "_thinking_pad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = (
            self._needs_deepseek_tool_reasoning()
            or self._needs_kimi_tool_reasoning()
            or self._needs_mimo_tool_reasoning()
            or self._reasoning_echo_opt_in()
        )
        self._thinking_pad_cache = (key, result)
        return result

    def _reasoning_echo_opt_in(self) -> bool:
        """User opted in to ``reasoning_content`` echo-back for the *current* provider (``model.reasoning_echo``).

        Covers custom providers/gateways the host-based echo rules miss. Per-active-provider: fallback
        activation swaps the flag and ``restore_primary_runtime()`` restores it, so a strict fallback still strips.
        """
        return bool(getattr(self, "_reasoning_echo_flag", False))

    @staticmethod
    def _read_reasoning_echo_from_config() -> bool:
        """Read ``model.reasoning_echo`` from config; False on any error."""
        try:
            from hermes_cli.config import load_config_readonly
            return bool((load_config_readonly().get("model") or {}).get("reasoning_echo"))
        except Exception:
            return False

    # Echo families are host/provider-driven, not model-name-driven: aggregators re-exporting Kimi reject the
    # echo. Rule table: ``message_sanitization._REASONING_ECHO_RULES``. Kimi deliberately passes the raw
    # provider and no model (its rule matches exact provider ids + hosts only).
    def _needs_kimi_tool_reasoning(self) -> bool:
        """True when the current provider is Kimi / Moonshot thinking mode."""
        return matches_reasoning_echo_family("kimi", self.provider, None, self.base_url)

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """True when the current provider is DeepSeek thinking mode (omitting the echo is an HTTP 400)."""
        return matches_reasoning_echo_family("deepseek", (self.provider or "").lower(), self.model, self.base_url)

    def _needs_mimo_tool_reasoning(self) -> bool:
        """True when the current provider is Xiaomi MiMo thinking mode."""
        return matches_reasoning_echo_family("mimo", (self.provider or "").lower(), self.model, self.base_url)

    _copy_reasoning_content_for_api = _forward("agent.agent_runtime_helpers", "copy_reasoning_content_for_api")

    _reapply_reasoning_echo_for_provider = _forward("agent.agent_runtime_helpers", "reapply_reasoning_echo_for_provider")

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict, model: "str | None" = None) -> dict:
        """Strip Codex Responses fields (call_id, response_item_id, extra_content) from tool_calls.

        Strict Chat Completions APIs (Mistral, Fireworks) 400/422 on unknown fields. ``extra_content`` (Gemini
        thought_signature) is kept only when the outgoing model is Gemini-family (it 400s without it). Builds
        new dicts so the internal history retains the Codex fields for a later fallback.
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        from agent.transports.chat_completions import _model_consumes_thought_signature
        strip = {"call_id", "response_item_id"}
        if not _model_consumes_thought_signature(model):
            strip.add("extra_content")
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in strip} if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    _sanitize_tool_call_arguments = _forward_static("agent.agent_runtime_helpers", "sanitize_tool_call_arguments")

    def _should_sanitize_tool_calls(self) -> bool:
        """True for every non-Codex API: Codex Responses fields are not Chat Completions schema and 400 elsewhere."""
        return self.api_mode != "codex_responses"
