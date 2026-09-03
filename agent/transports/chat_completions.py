"""OpenAI Chat Completions transport (default api_mode for OpenAI-compatible providers).

Messages/tools are already OpenAI-shaped, so convert_* are near-identity; the
provider-specific work lives in build_kwargs (max_tokens, reasoning, extra_body).
"""

import json
from typing import Any, Dict

from agent.lmstudio_reasoning import resolve_lmstudio_effort
from agent.reasoning_effort import (
    KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES, OPENAI_COMPAT_WIRE_EFFORTS, TOKENHUB_EFFORTS, clamp_effort,
    kimi_supported_efforts, requested_effort,
)
from agent.moonshot_schema import is_moonshot_model, sanitize_moonshot_tools
from agent.prompt_builder import DEVELOPER_ROLE_MODELS
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

# xAI reserves the function name ``tool_search`` for its server-side tool and
# rejects client declarations of it (HTTP 400, #95003); alias it on the wire and
# map back in normalize_response. Value matches the Codex-side alias (#83122).
_XAI_TOOL_SEARCH_ALIAS = "hermes_tool_search"

# Persistence-only / cross-transport message keys that strict OpenAI-compatible
# providers reject with HTTP 400 ("Extra inputs are not permitted").
_STRIP_MSG_KEYS = (
    "codex_reasoning_items", "codex_message_items", "tool_name", "effect_disposition", "timestamp",
    "platform_message_id", "api_content", "anthropic_content_blocks", "bedrock_content_blocks",
)
_STRIP_TC_KEYS = ("call_id", "response_item_id")
_HIGH_EFFORTS = {"high", "xhigh", "max", "ultra"}


def _rename_tool_search_bridge_for_xai(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Alias the client ``tool_search`` declaration for xAI.

    Returns ``(rewritten_tools, alias_map)`` where alias_map maps each alias
    emitted by THIS request back to ``tool_search``. If a real tool already
    holds ``hermes_tool_search``, the bridge takes a ``_2``/``_3`` suffix.
    """
    from agent.transports.codex import _alias_reserved_tools

    return _alias_reserved_tools(
        tools, ("tool_search",), name_of=lambda t: (t.get("function") or {}).get("name"),
        rename=lambda t, alias: {**t, "function": {**t["function"], "name": alias}},
    )


def _static_prompt_instructions(messages: list[dict[str, Any]]) -> str:
    """Stable leading system/developer prefix used for cache routing (later messages are conversation state)."""
    if not messages or not isinstance(messages[0], dict):
        return ""
    first = messages[0]
    if first.get("role") not in {"system", "developer"}:
        return ""
    content = first.get("content")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(content or "")


def _add_prompt_cache_key(
    api_kwargs: dict[str, Any], *, messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None, supports_prompt_cache_key: bool,
    session_id: str | None = None, cache_scope_id: str | None = None,
) -> None:
    """Add a content-addressed ``prompt_cache_key`` only for a capable endpoint.

    ``cache_scope_id`` (compression-lineage root) beats ``session_id`` so the key
    survives context-compression session rotation (#79017). A caller-supplied key
    is authoritative but is bounded to OpenAI's 64-char wire cap in place.
    """
    # Share the Responses transport's hash + scope normalization so equivalent
    # prefixes hit one bucket across modes without merging unrelated sessions (#78941).
    from agent.transports.codex import (
        _bound_prompt_cache_key_field, _cache_scope_from_session_id, _content_cache_key
    )

    extra_body = api_kwargs.get("extra_body")
    containers = [c for c in (api_kwargs, extra_body) if isinstance(c, dict) and "prompt_cache_key" in c]
    if containers:
        for c in containers:
            _bound_prompt_cache_key_field(c)
        return
    if not supports_prompt_cache_key:
        return
    cache_key = _content_cache_key(
        _static_prompt_instructions(messages), tools,
        _cache_scope_from_session_id(cache_scope_id or session_id),
    )
    if cache_key:
        api_kwargs["prompt_cache_key"] = cache_key


def _reasoning_config_for_model(model: str, reasoning_config: dict | None) -> dict | None:
    """Clamp Hermes' extended effort set (``ultra``) to the OpenAI-compat wire vocabulary (#89503)."""
    if not isinstance(reasoning_config, dict):
        return reasoning_config
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    if not effort:
        return reasoning_config
    clamped = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
    if clamped != effort:
        return {**reasoning_config, "effort": clamped}
    return reasoning_config


def _build_gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """Translate Hermes/OpenRouter-style reasoning config to Gemini thinkingConfig."""
    if reasoning_config is None or not isinstance(reasoning_config, dict):
        return None
    normalized_model = (model or "").strip().lower()
    if normalized_model.startswith("google/"):
        normalized_model = normalized_model.split("/", 1)[1]
    # ``thinking_config`` is Gemini-only; Gemma/PaLM on the same provider reject
    # the field with HTTP 400 even as ``{"includeThoughts": False}`` (#17426).
    if not normalized_model.startswith("gemini"):
        return None
    if reasoning_config.get("enabled") is False:
        return {"includeThoughts": False}
    effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
    if effort == "none":
        return {"includeThoughts": False}
    thinking_config: Dict[str, Any] = {"includeThoughts": True}
    # Gemini 2.5 takes thinkingBudget; don't guess one from coarse effort levels.
    if normalized_model.startswith("gemini-2.5-"):
        return thinking_config
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
        effort = "medium"
    # Gemini 3 Flash documents low/medium/high; Gemini 3 Pro only low/high.
    if normalized_model.startswith(("gemini-3", "gemini-3.1")):
        if "flash" in normalized_model:
            thinking_config["thinkingLevel"] = (
                "low" if effort in {"minimal", "low"} else "high" if effort in _HIGH_EFFORTS else "medium"
            )
        elif "pro" in normalized_model:
            thinking_config["thinkingLevel"] = "high" if effort in _HIGH_EFFORTS else "low"
    return thinking_config


def _snake_case_gemini_thinking_config(config: dict | None) -> dict | None:
    """Convert Gemini thinking config keys to the OpenAI-compat field names."""
    if not isinstance(config, dict) or not config:
        return None
    translated: Dict[str, Any] = {}
    include, level, budget = config.get("includeThoughts"), config.get("thinkingLevel"), config.get("thinkingBudget")
    if isinstance(include, bool):
        translated["include_thoughts"] = include
    if isinstance(level, str) and level.strip():
        translated["thinking_level"] = level.strip().lower()
    if isinstance(budget, (int, float)):
        translated["thinking_budget"] = int(budget)
    return translated or None


def _raise_gemini_thinking_max_tokens(model: str, reasoning_config: dict | None, requested: Any) -> Any:
    """Raise Gemini output caps that thinking tokens (billed against max_tokens) would otherwise exhaust."""
    thinking_config = _build_gemini_thinking_config(model, reasoning_config)
    if not thinking_config:
        return requested
    from agent.gemini_native_adapter import _effective_gemini_max_output_tokens

    return _effective_gemini_max_output_tokens(requested, thinking_config)


def _is_gemini_openai_compat_base_url(base_url: Any) -> bool:
    normalized = str(base_url or "").strip().rstrip("/").lower()
    return bool(normalized) and "generativelanguage.googleapis.com" in normalized and normalized.endswith("/openai")


def _is_openai_api_base_url(base_url: Any) -> bool:
    """True only for the exact api.openai.com host (implies ``prompt_cache_key`` support).

    Not a substring match: Azure / strict OpenAI-compat endpoints may reject the
    field and must stay opt-in via ``supports_prompt_cache_key``.
    """
    try:
        from urllib.parse import urlparse

        return (urlparse(str(base_url or "").strip()).hostname or "").lower() == "api.openai.com"
    except Exception:
        return False


def _model_consumes_thought_signature(model: Any) -> bool:
    """True for Gemini-family targets, which require tool-call ``extra_content`` (thought_signature) replay.

    Every other strict provider rejects a request containing it, so it is kept
    only for Gemini targets and stripped otherwise (incl. stale inherited copies).
    """
    m = str(model or "").lower()
    return "gemini" in m or "gemma" in m


def _thinking_disabled(reasoning_config: Any) -> bool:
    return bool(reasoning_config and isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False)


def _swap_developer_role(sanitized: list, model_lower: str) -> list:
    """GPT-5/Codex models take a ``developer`` role instead of ``system``."""
    if (
        sanitized
        and isinstance(sanitized[0], dict)
        and sanitized[0].get("role") == "system"
        and any(p in model_lower for p in DEVELOPER_ROLE_MODELS)
    ):
        sanitized = list(sanitized)
        sanitized[0] = {**sanitized[0], "role": "developer"}
    return sanitized


def _apply_max_tokens(api_kwargs: dict, model: str, reasoning_config: Any, params: dict, profile_max: Any = None) -> None:
    """Resolve max_tokens — priority: ephemeral > user > profile default > anthropic_max_output."""
    max_tokens_fn = params.get("max_tokens_param_fn")
    for candidate in (params.get("ephemeral_max_output_tokens"), params.get("max_tokens")):
        if candidate is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(_raise_gemini_thinking_max_tokens(model, reasoning_config, candidate)))
            return
    if profile_max and max_tokens_fn:
        api_kwargs.update(max_tokens_fn(_raise_gemini_thinking_max_tokens(model, reasoning_config, profile_max)))
    elif params.get("anthropic_max_output") is not None:
        api_kwargs["max_tokens"] = params["anthropic_max_output"]


def _base_kwargs(model: str, sanitized: list, tools: Any, params: dict) -> dict[str, Any]:
    """Shared ``{model, messages[, timeout][, tools]}`` scaffold for both build paths."""
    api_kwargs: dict[str, Any] = {"model": model, "messages": sanitized}
    timeout = params.get("timeout")
    if timeout is not None:
        api_kwargs["timeout"] = timeout
    if tools:
        # Moonshot/Kimi uses a stricter JSON Schema flavor; rewriting here also covers aggregator routes.
        api_kwargs["tools"] = sanitize_moonshot_tools(tools) if is_moonshot_model(model) else tools
    return api_kwargs


def _finish_kwargs(
    api_kwargs: dict[str, Any], sanitized: list, params: dict, *, supports_prompt_cache_key: bool
) -> dict[str, Any]:
    """Tail shared by both build paths: content-addressed prompt_cache_key, then return."""
    _add_prompt_cache_key(
        api_kwargs, messages=sanitized, tools=api_kwargs.get("tools"),
        supports_prompt_cache_key=supports_prompt_cache_key, session_id=params.get("session_id"),
        cache_scope_id=params.get("cache_scope_id"),
    )
    return api_kwargs


def _msg_strip_keys(msg: dict) -> list:
    """Keys to drop from a message: persistence sidecars plus any ``_``-prefixed Hermes scaffolding marker."""
    return [k for k in msg if k in _STRIP_MSG_KEYS or (isinstance(k, str) and k.startswith("_"))]


def _tc_strip_keys(tc: dict, strip_extra_content: bool) -> list:
    keys = [k for k in _STRIP_TC_KEYS if k in tc]
    if strip_extra_content and "extra_content" in tc:
        keys.append("extra_content")
    return keys


def _invalid_assistant_tool_calls(msg: dict, tool_calls: Any) -> bool:
    """``tool_calls: []`` / ``tool_calls: null`` on an assistant message — strict providers reject both (#58755)."""
    return (
        msg.get("role") == "assistant"
        and "tool_calls" in msg
        and (tool_calls is None or (isinstance(tool_calls, list) and not tool_calls))
    )


class ChatCompletionsTransport(ProviderTransport):
    """Transport for api_mode='chat_completions'."""

    # Wire-alias provenance of the most recent request: ``{alias: original}``.
    # ``None`` = no request recorded (normalize-only call sites) -> fall back to
    # the static alias; ``{}`` = last request emitted no aliases (#95003).
    _last_wire_aliases: dict[str, str] | None = None

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """Strip internal fields that strict chat-completions providers reject (HTTP 400/422).

        Codex sidecars, ``tool_name``, ``_``-prefixed markers, native-transport
        block sidecars, and tool-call ``call_id``/``response_item_id`` are always
        dropped; ``extra_content`` is dropped unless ``model`` is Gemini-family.
        Returns the input list unchanged when nothing needs sanitizing.
        """
        strip_extra_content = not _model_consumes_thought_signature(kwargs.get("model"))

        def sanitize(msg: Any) -> "dict | None":
            """Sanitized copy of ``msg``, or None when nothing needs stripping."""
            if not isinstance(msg, dict):
                return None
            strip_keys = _msg_strip_keys(msg)
            out_msg = dict(msg)
            for key in strip_keys:
                out_msg.pop(key, None)
            tool_calls = msg.get("tool_calls")
            copied_tool_calls = None
            if _invalid_assistant_tool_calls(msg, tool_calls):
                out_msg.pop("tool_calls", None)
                strip_keys.append("tool_calls")
            elif isinstance(tool_calls, list):
                for tc_idx, tc in enumerate(tool_calls):
                    keys = _tc_strip_keys(tc, strip_extra_content) if isinstance(tc, dict) else []
                    if keys:
                        if copied_tool_calls is None:
                            copied_tool_calls = list(tool_calls)
                        copied_tc = dict(tc)
                        for key in keys:
                            copied_tc.pop(key, None)
                        copied_tool_calls[tc_idx] = copied_tc
                if copied_tool_calls is not None:
                    out_msg["tool_calls"] = copied_tool_calls
            return out_msg if strip_keys or copied_tool_calls is not None else None

        sanitized_pairs = [(m, sanitize(m)) for m in messages]
        if all(s is None for _, s in sanitized_pairs):
            return messages
        return [m if s is None else s for m, s in sanitized_pairs]

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tools are already in OpenAI format — identity."""
        return tools

    def build_kwargs(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        **params,
    ) -> dict[str, Any]:
        """Build chat.completions.create() kwargs.

        With ``provider_profile`` every quirk comes from the profile
        (_build_kwargs_from_profile). The legacy flag path below (is_kimi,
        is_openrouter, is_lmstudio, ...) is only reached for unregistered providers.
        """
        sanitized = self.convert_messages(messages, model=model)
        _profile = params.get("provider_profile")
        if _profile:
            return self._build_kwargs_from_profile(_profile, model, sanitized, tools, params)

        sanitized = _swap_developer_role(sanitized, params.get("model_lower", (model or "").lower()))
        api_kwargs = _base_kwargs(model, sanitized, tools, params)

        is_kimi = params.get("is_kimi", False)
        reasoning_config = _reasoning_config_for_model(model, params.get("reasoning_config"))
        _apply_max_tokens(api_kwargs, model, reasoning_config, params)

        # Kimi / TokenHub / LM Studio: top-level reasoning_effort (unless thinking disabled).
        thinking_off = _thinking_disabled(reasoning_config)
        _e = requested_effort(reasoning_config)
        if is_kimi and not thinking_off:
            # K3 = low/high/max (server default high), K2-era = low/medium/high (default medium).
            _supported = kimi_supported_efforts(model)
            is_k3 = _supported is KIMI_K3_EFFORTS
            api_kwargs["reasoning_effort"] = (
                ("high" if is_k3 else "medium") if _e is None
                else clamp_effort(_e, _supported, KIMI_K3_OVERRIDES if is_k3 else None)
            )
        if params.get("is_tokenhub", False) and not thinking_off:
            api_kwargs["reasoning_effort"] = "high" if _e is None else clamp_effort(_e, TOKENHUB_EFFORTS)
        if params.get("is_lmstudio", False) and params.get("supports_reasoning", False):
            _lm_effort = resolve_lmstudio_effort(reasoning_config, params.get("lmstudio_reasoning_options"))
            if _lm_effort is not None:
                api_kwargs["reasoning_effort"] = _lm_effort

        extra_body: dict[str, Any] = {}
        is_openrouter = params.get("is_openrouter", False)
        provider_name = str(params.get("provider_name") or "").strip().lower()
        base_url = params.get("base_url")
        provider_prefs = params.get("provider_preferences")
        if provider_prefs and is_openrouter:
            extra_body["provider"] = provider_prefs
        # Pareto Code router plugin (same shape as the OpenRouter profile path).
        if is_openrouter and model == "openrouter/pareto-code":
            _pareto_score = params.get("openrouter_min_coding_score")
            try:
                _pareto_score_f = float(_pareto_score) if _pareto_score not in (None, "") else None
            except (TypeError, ValueError):
                _pareto_score_f = None
            if _pareto_score_f is not None and 0.0 <= _pareto_score_f <= 1.0:
                extra_body["plugins"] = [{"id": "pareto-router", "min_coding_score": _pareto_score_f}]
        if is_kimi:
            extra_body["thinking"] = {"type": "disabled" if thinking_off else "enabled"}

        # LM Studio is handled above via top-level reasoning_effort.
        if params.get("supports_reasoning", False) and not params.get("is_lmstudio", False):
            if params.get("is_github_models", False):
                gh_reasoning = params.get("github_reasoning_extra")
                if gh_reasoning is not None:
                    extra_body["reasoning"] = gh_reasoning
            else:
                _effort = "medium"
                if reasoning_config and isinstance(reasoning_config, dict):
                    _effort = reasoning_config.get("effort", "medium") or "medium"
                # Honor explicit "thinking off" like the profile path — never re-enable it.
                if thinking_off or _effort == "none":
                    extra_body["reasoning"] = {"enabled": False, "effort": "none"}
                else:
                    extra_body["reasoning"] = {"enabled": True, "effort": _effort}

        if provider_name == "gemini":
            raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
            if _is_gemini_openai_compat_base_url(base_url):
                thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
                if thinking_config:
                    openai_compat_extra = extra_body.get("extra_body", {})
                    google_extra = openai_compat_extra.get("google", {})
                    google_extra["thinking_config"] = thinking_config
                    openai_compat_extra["google"] = google_extra
                    extra_body["extra_body"] = openai_compat_extra
            elif raw_thinking_config:
                extra_body["thinking_config"] = raw_thinking_config

        additions = params.get("extra_body_additions")
        if additions:
            extra_body.update(additions)
        if extra_body:
            api_kwargs["extra_body"] = extra_body
        overrides = params.get("request_overrides")
        if overrides:
            api_kwargs.update(overrides)
        return _finish_kwargs(
            api_kwargs,
            sanitized,
            params,
            supports_prompt_cache_key=bool(params.get("supports_prompt_cache_key"))
            or _is_openai_api_base_url(params.get("base_url")),
        )

    def _build_kwargs_from_profile(self, profile, model, sanitized, tools, params):
        """Build API kwargs from a ProviderProfile — every quirk comes from the profile object."""
        from providers.base import OMIT_TEMPERATURE

        sanitized = _swap_developer_role(profile.prepare_messages(sanitized), (model or "").lower())
        api_kwargs: dict[str, Any] = {"model": model, "messages": sanitized}
        if profile.fixed_temperature is OMIT_TEMPERATURE:
            pass
        elif profile.fixed_temperature is not None:
            api_kwargs["temperature"] = profile.fixed_temperature
        elif params.get("temperature") is not None:
            api_kwargs["temperature"] = params["temperature"]
        api_kwargs.update(_base_kwargs(model, sanitized, tools, params))

        reasoning_config = _reasoning_config_for_model(model, params.get("reasoning_config"))
        # Profiles fronting several backends override get_max_tokens() per model.
        _apply_max_tokens(api_kwargs, model, reasoning_config, params, profile_max=profile.get_max_tokens(model))

        extra_body_from_profile, top_level_from_profile = profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config,
            supports_reasoning=params.get("supports_reasoning", False),
            qwen_session_metadata=params.get("qwen_session_metadata"), model=model,
            base_url=params.get("base_url"), ollama_num_ctx=params.get("ollama_num_ctx"),
            session_id=params.get("session_id"),
        )
        api_kwargs.update(top_level_from_profile)

        extra_body: dict[str, Any] = {}
        profile_body = profile.build_extra_body(
            session_id=params.get("session_id"),
            provider_preferences=params.get("provider_preferences"), model=model,
            base_url=params.get("base_url"), reasoning_config=reasoning_config,
            openrouter_min_coding_score=params.get("openrouter_min_coding_score"),
        )
        for part in (profile_body, extra_body_from_profile, params.get("extra_body_additions")):
            if part:
                extra_body.update(part)
        overrides = params.get("request_overrides")
        if overrides:
            for k, v in overrides.items():
                if k == "extra_body" and isinstance(v, dict):
                    extra_body.update(v)
                else:
                    api_kwargs[k] = v

        if extra_body:
            # Native Gemini speaks Google's REST schema: OpenAI-style extra_body
            # keys (tags, reasoning, provider, ...) are unknown fields -> HTTP 400.
            # The native client only reads thinking_config, so drop everything else.
            try:
                from agent.gemini_native_adapter import is_native_gemini_base_url
                _native_gemini = is_native_gemini_base_url(params.get("base_url"))
            except Exception:
                _native_gemini = False
            if _native_gemini:
                extra_body = {k: v for k, v in extra_body.items() if k in ("thinking_config", "thinkingConfig")}
            if extra_body:
                api_kwargs["extra_body"] = extra_body
        return _finish_kwargs(
            api_kwargs, sanitized, params,
            supports_prompt_cache_key=bool(getattr(profile, "supports_prompt_cache_key", False)),
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize an OpenAI ChatCompletion.

        Gemini ``extra_content`` rides on ToolCall.provider_data; ``reasoning_content``
        (DeepSeek/Moonshot) and ``reasoning_details`` (OpenRouter) are kept apart in
        provider_data because downstream reads them distinctly.
        """
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        _fr = getattr(choice, "finish_reason", None)
        finish_reason = (str(_fr) if isinstance(_fr, int) else _fr) or "stop"  # Poolside returns int finish_reason

        tool_calls = None
        message_tool_calls = getattr(msg, "tool_calls", None)
        if message_tool_calls:
            tool_calls = []
            _alias_map = self._last_wire_aliases
            for tc in message_tool_calls:
                tc_function = getattr(tc, "function", None)
                function_name = getattr(tc_function, "name", None)
                # Match Relay's codec: skip absent function/name, keep an explicit blank name.
                if tc_function is None or function_name is None:
                    continue
                # Reverse only aliases THIS request emitted; a real tool named
                # ``hermes_tool_search`` dispatches as itself when none were.
                if _alias_map is None:
                    if function_name == _XAI_TOOL_SEARCH_ALIAS:
                        function_name = "tool_search"
                elif function_name in _alias_map:
                    function_name = _alias_map[function_name]
                function_arguments = getattr(tc_function, "arguments", None)
                tc_provider_data: dict[str, Any] = {}
                extra = getattr(tc, "extra_content", None)
                if extra is None and hasattr(tc, "model_extra"):
                    extra = (tc.model_extra if isinstance(tc.model_extra, dict) else {}).get("extra_content")
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        for dump_kwargs in ({"warnings": False}, {}):
                            try:
                                extra = extra.model_dump(**dump_kwargs)
                                break
                            except TypeError:
                                continue  # older pydantic: retry without ``warnings``
                            except Exception:
                                break
                    tc_provider_data["extra_content"] = extra
                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", None), name=function_name,
                        arguments=function_arguments if function_arguments is not None else "{}",
                        provider_data=tc_provider_data or None,
                    )
                )

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = Usage.from_openai(response.usage)

        # Fields some SDKs park in pydantic ``model_extra`` rather than as attributes.
        model_extra = getattr(msg, "model_extra", None) or {}
        model_extra = model_extra if isinstance(model_extra, dict) else {}
        reasoning = getattr(msg, "reasoning", None)
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = model_extra.get("reasoning_content")

        provider_data: Dict[str, Any] = {}
        if reasoning_content is not None:
            provider_data["reasoning_content"] = reasoning_content
        rd = getattr(msg, "reasoning_details", None)
        if rd:
            provider_data["reasoning_details"] = rd

        # OpenAI structured refusal: ``message.refusal`` set, ``content`` empty.
        # Proxies fronting Anthropic/Bedrock surface Claude refusals this way; without
        # promotion the loop retries a deterministic refusal as an empty response.
        content = getattr(msg, "content", None)
        refusal = getattr(msg, "refusal", None)
        if refusal is None:
            refusal = model_extra.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            provider_data["refusal"] = refusal
            # Promote to a terminal ``content_filter`` only when the refusal is the
            # sole payload — real text or tool calls alongside it is a usable turn.
            if not (isinstance(content, str) and content.strip()) and not tool_calls:
                content = refusal
                if finish_reason in (None, "stop"):
                    finish_reason = "content_filter"

        return NormalizedResponse(
            content=content, tool_calls=tool_calls, finish_reason=finish_reason,
            reasoning=reasoning, usage=usage, provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check that response has valid choices."""
        return bool(response is not None and getattr(response, "choices", None))

    def extract_cache_stats(self, response: Any) -> dict[str, int] | None:
        """Cache stats from prompt_tokens_details (OpenRouter/OpenAI) or DeepSeek's top-level prompt_cache_hit_tokens."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        written = getattr(details, "cache_write_tokens", 0) or 0 if details else 0
        cached = cached or getattr(usage, "prompt_cache_hit_tokens", 0) or 0  # DeepSeek native (#61871)
        return {"cached_tokens": cached, "creation_tokens": written} if cached or written else None


from agent.transports import register_transport  # noqa: E402

register_transport("chat_completions", ChatCompletionsTransport)
