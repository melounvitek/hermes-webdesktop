"""Provider reasoning-parameter policy for ``AIAgent``.

When ``reasoning`` extra_body is safe to send, LM Studio / Ollama / GitHub Models capability probes,
``reasoning_content`` echo families, and strict-API tool-call sanitising.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
from typing import Optional

from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from utils import base_url_host_matches


class ReasoningParamsMixin:
    """Reasoning-parameter gating and echo policy (see module docstring)."""

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body upstream and some routes 400 on ``reasoning``; gate to known
        reasoning-capable families and direct Nous Portal.
        """
        if base_url_host_matches(self._base_url_lower, "nousresearch.com"):
            return True
        if base_url_host_matches(self._base_url_lower, "ai-gateway.vercel.sh"):
            return True
        if (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "githubcopilot.com")
        ):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if (self.provider or "").strip().lower() == "lmstudio":
            opts = self._lmstudio_reasoning_options_cached()
            # "off-only" (or absent) means no real reasoning capability.
            return any(opt and opt != "off" for opt in opts)
        # Ollama Cloud: /api/show capabilities are authoritative — emit reasoning_effort only for models
        # declaring "thinking". Cached per (model, base_url).
        if base_url_host_matches(self._base_url_lower, "ollama.com"):
            return self._ollama_supports_thinking_cached()
        if not self._is_openrouter_url():
            return False
        if base_url_host_matches(self._base_url_lower, "api.mistral.ai"):
            return False

        model = (self.model or "").lower()
        # Live-catalog metadata first (OpenRouter /v1/models supported_parameters) — the static prefix
        # allowlist repeatedly went stale one vendor at a time (#75386). Unknown falls back to the static
        # list.
        try:
            from hermes_cli.models import (
                openrouter_model_reasoning_capabilities,
                warm_openrouter_reasoning_caps_async,
            )
            caps = openrouter_model_reasoning_capabilities(self.model)
            if caps is None:
                # Cache cold — warm in the background; never block this turn on HTTP.
                warm_openrouter_reasoning_caps_async()
        except Exception:
            caps = None
        if caps is not None:
            return bool(caps.get("supports_reasoning"))
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "google/gemma-4",
            "qwen/qwen3",
            "tencent/hy",
            "xiaomi/",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _lmstudio_reasoning_options_cached(self) -> list[str]:
        """Probe LM Studio's published reasoning ``allowed_options`` once per (model, base_url).

        Needed for the supports-reasoning gate and to clamp ``reasoning_effort`` so toggle-style models don't
        400 on ``high``. Non-empty results cache permanently; empty ones (transient failure OR non-reasoning
        model) cache with a 60s TTL to avoid a round-trip per turn while retrying soon.
        """
        import time as _time

        cache = getattr(self, "_lm_reasoning_opts_cache", None)
        if cache is None:
            cache = self._lm_reasoning_opts_cache = {}
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            opts, ts = cached
            # Non-empty → permanent. Empty → 60s TTL.
            if opts or (_time.monotonic() - ts) < 60:
                return opts
        try:
            from hermes_cli.models import lmstudio_model_reasoning_options
            opts = lmstudio_model_reasoning_options(
                self.model, self.base_url, getattr(self, "api_key", ""),
            )
        except Exception:
            opts = []
        cache[key] = (opts, _time.monotonic())
        return opts

    def _ollama_supports_thinking_cached(self) -> bool:
        """Probe Ollama's ``/api/show`` capabilities once per (model, base_url); True only if ``thinking`` is
        declared.

        True/False cache permanently; a probe failure (None) caches 60s so an outage neither suppresses
        reasoning for the session nor round-trips every turn.
        """
        import time as _time

        cache = getattr(self, "_ollama_thinking_cache", None)
        if cache is None:
            cache = self._ollama_thinking_cache = {}
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            supported, ts = cached
            # Definitive True/False → permanent. Unknown (None) → 60s TTL.
            if supported is not None or (_time.monotonic() - ts) < 60:
                return bool(supported)
        try:
            from hermes_cli.models import ollama_model_supports_thinking
            supported = ollama_model_supports_thinking(
                self.model, self.base_url, getattr(self, "api_key", "")
            )
        except Exception:
            supported = None
        cache[key] = (supported, _time.monotonic())
        return bool(supported)

    def _resolve_lmstudio_summary_reasoning_effort(self) -> Optional[str]:
        """Resolve a safe top-level ``reasoning_effort`` for LM Studio.

        The iteration-limit summary calls ``chat.completions.create()`` directly, bypassing the transport;
        share the helper so effort resolution and clamping cannot drift.
        """
        from agent.lmstudio_reasoning import resolve_lmstudio_effort
        return resolve_lmstudio_effort(
            self.reasoning_config,
            self._lmstudio_reasoning_options_cached(),
        )

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "xhigh" not in supported_efforts and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

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
        """True when the user opted in to ``reasoning_content`` echo-back for the *current* provider via
        config.

        Covers custom providers / gateways proxying thinking models that the host-based
        ``_REASONING_ECHO_RULES`` miss. Per-active-provider: primary from ``model.reasoning_echo``, fallback
        from the fallback entry's field, restored by ``restore_primary_runtime()`` — so falling back to a
        strict provider still strips it.
        """
        return bool(getattr(self, "_reasoning_echo_flag", False))

    @staticmethod
    def _read_reasoning_echo_from_config() -> bool:
        """Read ``model.reasoning_echo`` from config; False on any error."""
        try:
            from hermes_cli.config import load_config_readonly
            return bool(
                (load_config_readonly().get("model") or {}).get("reasoning_echo")
            )
        except Exception:
            return False

    def _needs_kimi_tool_reasoning(self) -> bool:
        """Return True when the current provider is Kimi / Moonshot thinking mode (requires
        ``reasoning_content`` echo).

        Host-driven, not model-name-driven: aggregators re-exporting Kimi reject the echo (#17400). Rule
        table: ``message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "kimi", self.provider, None, self.base_url
        )

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """Return True when the current provider is DeepSeek thinking mode (requires ``reasoning_content``
        echo).

        Omitting the echo on replayed assistant tool-call turns is an HTTP 400 (#15250). Rule table:
        ``message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "deepseek", (self.provider or "").lower(), self.model, self.base_url
        )

    def _needs_mimo_tool_reasoning(self) -> bool:
        """Return True when the current provider is Xiaomi MiMo thinking mode (requires ``reasoning_content``
        echo).

        Rule table: ``message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "mimo", (self.provider or "").lower(), self.model, self.base_url
        )

    _copy_reasoning_content_for_api = _forward("agent.agent_runtime_helpers", "copy_reasoning_content_for_api")

    _reapply_reasoning_echo_for_provider = _forward("agent.agent_runtime_helpers", "reapply_reasoning_echo_for_provider")

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict, model: "str | None" = None) -> dict:
        """Strip Codex Responses fields (call_id, response_item_id, extra_content) from tool_calls for strict
        providers.

        Strict Chat Completions APIs (Mistral, Fireworks) 400/422 on unknown fields. ``extra_content`` (Gemini
        thought_signature) is kept only when the outgoing model is Gemini-family (it 400s without it). Builds
        new dicts so the internal history retains the Codex fields for a later fallback.
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        from agent.transports.chat_completions import _model_consumes_thought_signature
        _STRIP_KEYS = {"call_id", "response_item_id"}
        if not _model_consumes_thought_signature(model):
            _STRIP_KEYS = _STRIP_KEYS | {"extra_content"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    _sanitize_tool_call_arguments = _forward_static("agent.agent_runtime_helpers", "sanitize_tool_call_arguments")

    def _should_sanitize_tool_calls(self) -> bool:
        """Determine if tool_calls need sanitization (True for every non-Codex API).

        Codex Responses fields (call_id, response_item_id) are not Chat Completions schema and 400 elsewhere.
        """
        return self.api_mode != "codex_responses"
