"""MCP client-side handlers for server-initiated requests: sampling
(sampling/createMessage, text and tool-use results) and elicitation."""

import asyncio
import json
import logging
import time
from typing import Callable, List, Optional
from tools.mcp_tool_common import _MISSING, _exc_str, _safe_numeric, _sanitize_error, mcp_field, _core
from tools.mcp_tool_schema import _normalize_mcp_input_schema

logger = logging.getLogger("tools.mcp_tool")


def _tool_use_id(block):
    """Tool-use id (the discriminator for a tool *result* block), read under both SDK spellings —
    on mcp 2.x a bare ``hasattr(b, "toolUseId")`` is False and would silently drop tool results."""
    return mcp_field(block, "tool_use_id", "toolUseId", _MISSING)


def _is_tool_use(block) -> bool:
    return hasattr(block, "name") and hasattr(block, "input")


def _tool_result_text(block) -> str:
    """Text of a ToolResultContent block ("" when it carries no content)."""
    content = getattr(block, "content", None)
    if content is None:
        return ""
    items = content if isinstance(content, list) else [content]
    return "\n".join(item.text for item in items if hasattr(item, "text"))


def _content_part(block) -> Optional[dict]:
    """One OpenAI content part for a text/image block; None when unsupported."""
    if hasattr(block, "text"):
        return {"type": "text", "text": block.text}
    mime = mcp_field(block, "mime_type", "mimeType", _MISSING)
    if hasattr(block, "data") and mime is not _MISSING:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{block.data}"}}
    logger.warning("Unsupported sampling content block type: %s (skipped)", type(block).__name__)
    return None


def _tool_call_dict(tu, index: int) -> dict:
    args = tu.input
    return {"id": getattr(tu, "id", f"call_{index}"), "type": "function", "function": {
        "name": tu.name, "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)}}


def _convert_sampling_message(msg) -> List[dict]:
    """One MCP SamplingMessage -> OpenAI-format messages (tool results first,
    then either an assistant tool_calls message or plain content)."""
    blocks = msg.content_as_list if hasattr(msg, "content_as_list") else (
        msg.content if isinstance(msg.content, list) else [msg.content])
    tool_results = [b for b in blocks if _tool_use_id(b) is not _MISSING]
    tool_uses = [b for b in blocks if _is_tool_use(b) and _tool_use_id(b) is _MISSING]
    content_blocks = [b for b in blocks if _tool_use_id(b) is _MISSING and not _is_tool_use(b)]

    out = [{"role": "tool", "tool_call_id": _tool_use_id(tr), "content": _tool_result_text(tr)} for tr in tool_results]
    if tool_uses:
        msg_dict: dict = {"role": msg.role, "tool_calls": [_tool_call_dict(tu, i) for i, tu in enumerate(tool_uses)]}
        text_parts = [b.text for b in content_blocks if hasattr(b, "text")]
        if text_parts:
            msg_dict["content"] = "\n".join(text_parts)
        out.append(msg_dict)
    elif content_blocks:
        if len(content_blocks) == 1 and hasattr(content_blocks[0], "text"):
            out.append({"role": msg.role, "content": content_blocks[0].text})
        else:
            parts = [p for p in map(_content_part, content_blocks) if p is not None]
            if parts:
                out.append({"role": msg.role, "content": parts})
    return out


def _parse_tool_call_arguments(server_name: str, args) -> dict:
    """LLM tool_calls arguments -> dict; malformed JSON / non-dict values are
    wrapped as ``{"_raw": ...}`` rather than dropped."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            logger.warning("MCP server '%s': malformed tool_calls arguments from LLM (wrapping as raw): %.100s",
                           server_name, args)
            return {"_raw": args}
    return args if isinstance(args, dict) else {"_raw": str(args)}


def _response_total_tokens(response, default):
    return getattr(getattr(response, "usage", None), "total_tokens", default)


class SamplingHandler:
    """Handles sampling/createMessage requests for one MCP server; passed to ``ClientSession`` as
    ``sampling_callback``. All state (rate-limit timestamps, metrics, tool-loop counter) is per
    instance. Runs on the MCP background loop; the sync LLM call is offloaded via ``asyncio.to_thread``.

    Deprecated upstream (MCP 2026-07-28, SEP-2577, 12-month window): stays fully functional because
    handshake-era servers still issue it, but do NOT grow new capability here — modern servers use
    MRTR, handled by the SDK session layer."""

    _STOP_REASON_MAP = {"stop": "endTurn", "length": "maxTokens", "tool_calls": "toolUse"}
    _LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}

    def __init__(self, server_name: str, config: dict):
        self.server_name = server_name
        self.max_rpm = _safe_numeric(config.get("max_rpm", 10), 10, int)
        self.timeout = _safe_numeric(config.get("timeout", 30), 30, float)
        self.max_tokens_cap = _safe_numeric(config.get("max_tokens_cap", 4096), 4096, int)
        self.max_tool_rounds = _safe_numeric(config.get("max_tool_rounds", 5), 5, int, minimum=0)
        self.model_override = config.get("model")
        self.allowed_models = config.get("allowed_models", [])
        self.audit_level = self._LOG_LEVELS.get(str(config.get("log_level", "info")).lower(), logging.INFO)
        self._rate_timestamps: List[float] = []
        self._tool_loop_count = 0
        self.metrics = {"requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0}

    def _check_rate_limit(self) -> bool:
        """Sliding-window (60s) limiter; True if the request is allowed."""
        now = time.time()
        self._rate_timestamps[:] = [t for t in self._rate_timestamps if t > now - 60]
        if len(self._rate_timestamps) >= self.max_rpm:
            return False
        self._rate_timestamps.append(now)
        return True

    def _resolve_model(self, preferences) -> Optional[str]:
        """Config override > server hint > None (use default)."""
        if self.model_override:
            return self.model_override
        for hint in (getattr(preferences, "hints", None) or []):
            if getattr(hint, "name", None):
                return hint.name
        return None

    def _convert_messages(self, params) -> List[dict]:
        """MCP SamplingMessages -> OpenAI format (per-block duck-typed dispatch)."""
        return [m for msg in params.messages for m in _convert_sampling_message(msg)]

    @staticmethod
    def _error(message: str, code: int = -1):
        """Return ErrorData (MCP spec) or raise as fallback."""
        if _core._MCP_SAMPLING_TYPES:
            return _core.ErrorData(code=code, message=message)
        raise Exception(message)

    def _fail(self, message: str):
        """Count an error and return the ErrorData for it."""
        self.metrics["errors"] += 1
        return self._error(message)

    def _log_response(self, response, suffix: str = "", *args) -> None:
        logger.log(self.audit_level, "MCP server '%s' sampling response: model=%s, tokens=%s" + suffix,
                   self.server_name, response.model, _response_total_tokens(response, "?"), *args)

    def _build_tool_use_result(self, choice, response):
        """CreateMessageResultWithTools from an LLM tool_calls response, subject to tool-loop
        governance (``max_tool_rounds``; 0 disables)."""
        self.metrics["tool_use_count"] += 1
        if self.max_tool_rounds == 0:
            self._tool_loop_count = 0
            return self._error(f"Tool loops disabled for server '{self.server_name}' (max_tool_rounds=0)")
        self._tool_loop_count += 1
        if self._tool_loop_count > self.max_tool_rounds:
            self._tool_loop_count = 0
            return self._error(
                f"Tool loop limit exceeded for server '{self.server_name}' (max {self.max_tool_rounds} rounds)")
        content_blocks = [
            _core.ToolUseContent(type="tool_use", id=tc.id, name=tc.function.name,
                                 input=_parse_tool_call_arguments(self.server_name, tc.function.arguments))
            for tc in choice.message.tool_calls]
        self._log_response(response, ", tool_calls=%d", len(content_blocks))
        return _core.CreateMessageResultWithTools(
            role="assistant", content=content_blocks, model=response.model, stopReason="toolUse")

    def _build_text_result(self, choice, response):
        """CreateMessageResult from a normal text response (resets the tool loop)."""
        self._tool_loop_count = 0
        self._log_response(response)
        return _core.CreateMessageResult(
            role="assistant", model=response.model,
            content=_core.TextContent(type="text", text=_sanitize_error(choice.message.content or "")),
            stopReason=self._STOP_REASON_MAP.get(choice.finish_reason, "endTurn"))

    def session_kwargs(self) -> dict:
        """Kwargs to pass to ClientSession for sampling support."""
        return {"sampling_callback": self,
                "sampling_capabilities": _core.SamplingCapability(tools=_core.SamplingToolsCapability())}

    def _admit(self, params):
        """Rate-limit + allowed_models gate. Returns ``(resolved_model, None)`` or ``(None, ErrorData)``."""
        if not self._check_rate_limit():
            logger.warning("MCP server '%s' sampling rate limit exceeded (%d/min)", self.server_name, self.max_rpm)
            return None, self._fail(
                f"Sampling rate limit exceeded for server '{self.server_name}' ({self.max_rpm} requests/minute)")
        model = self._resolve_model(mcp_field(params, "model_preferences", "modelPreferences"))
        resolved_model = model or self.model_override or ""
        if self.allowed_models and resolved_model and resolved_model not in self.allowed_models:
            logger.warning("MCP server '%s' requested model '%s' not in allowed_models", self.server_name, resolved_model)
            return None, self._fail(f"Model '{resolved_model}' not allowed for server "
                                    f"'{self.server_name}'. Allowed: {', '.join(self.allowed_models)}")
        return resolved_model, None

    def _build_llm_call(self, params, resolved_model: str) -> Callable[[], object]:
        """Translate the sampling params into a zero-arg sync ``call_llm`` thunk (run off-loop so
        the MCP loop is not blocked). Server-provided tools are forwarded."""
        from agent.auxiliary_client import call_llm

        messages = self._convert_messages(params)
        system_prompt = mcp_field(params, "system_prompt", "systemPrompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        max_tokens = min(mcp_field(params, "max_tokens", "maxTokens", self.max_tokens_cap), self.max_tokens_cap)
        temperature = getattr(params, "temperature", None)
        server_tools = getattr(params, "tools", None)
        tools = [{"type": "function", "function": {
            "name": getattr(t, "name", ""), "description": getattr(t, "description", "") or "",
            "parameters": _normalize_mcp_input_schema(mcp_field(t, "input_schema", "inputSchema"))}}
            for t in server_tools] if server_tools else None
        logger.log(self.audit_level, "MCP server '%s' sampling request: model=%s, max_tokens=%d, messages=%d",
                   self.server_name, resolved_model, max_tokens, len(messages))
        return lambda: call_llm(task="mcp", model=resolved_model or None, messages=messages, temperature=temperature,
                                max_tokens=max_tokens, tools=tools, timeout=self.timeout)

    async def __call__(self, context, params):
        """SDK sampling callback (``SamplingFnT``). Returns CreateMessageResult,
        CreateMessageResultWithTools, or ErrorData."""
        resolved_model, err = self._admit(params)
        if err is not None:
            return err
        sync_call = self._build_llm_call(params, resolved_model)
        try:
            response = await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=self.timeout)
        except asyncio.TimeoutError:
            return self._fail(f"Sampling LLM call timed out after {self.timeout}s for server '{self.server_name}'")
        except Exception as exc:
            return self._fail(f"Sampling LLM call failed: {_sanitize_error(_exc_str(exc))}")
        # Empty choices happen on content filtering / provider errors.
        if not getattr(response, "choices", None):
            return self._fail(f"LLM returned empty response (no choices) for server '{self.server_name}'")
        choice = response.choices[0]
        self.metrics["requests"] += 1
        total_tokens = _response_total_tokens(response, 0)
        if isinstance(total_tokens, int):
            self.metrics["tokens_used"] += total_tokens
        if choice.finish_reason == "tool_calls" and getattr(choice.message, "tool_calls", None):
            return self._build_tool_use_result(choice, response)
        return self._build_text_result(choice, response)


def _format_elicitation_schema_summary(schema: dict, server_name: str) -> str:
    """Render a flat-object requested_schema as a human-readable field list (names, types,
    descriptions) so the user knows what they're approving."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or not props:
        return f"Approval requested by MCP server '{server_name}'."
    lines = [f"Fields requested by MCP server '{server_name}':"]
    for field_name, field_spec in props.items():
        spec = field_spec if isinstance(field_spec, dict) else {}
        field_type = str(spec.get("type", "") or "")
        field_desc = str(spec.get("description", "") or "")
        suffix = f" ({field_type})" if field_type else ""
        lines.append(f"  - {field_name}{suffix}: {field_desc}" if field_desc else f"  - {field_name}{suffix}")
    return "\n".join(lines)


class ElicitationHandler:
    """Handles ``elicitation/create`` requests for one MCP server; passed to ``ClientSession`` as
    ``elicitation_callback``. Form-mode requests route through Hermes' approval system (CLI, TUI,
    Telegram, ...); URL-mode is declined as unsupported. Fail-closed: any timeout, exception or
    unexpected state returns decline/cancel, never a silent accept."""

    # asyncio-side safety net over the approval's own input() timeout so the MCP loop never
    # blocks indefinitely if the inner timeout is bypassed.
    _OUTER_TIMEOUT_GRACE_SECONDS = 5
    # consent answer -> (ElicitResult action, metric); anything else declines.
    _ANSWER_RESULTS = {"accept": ("accept", "accepted"), "cancel": ("cancel", "errors")}

    def __init__(self, server_name: str, config: dict, owner: Optional["MCPServerTask"] = None):
        self.server_name = server_name
        # Default 5 min mirrors the gateway approval default so async surfaces (Telegram, Slack)
        # have time to respond.
        self.timeout = _safe_numeric(config.get("timeout", 300), 300, float)
        # Back-reference for the agent's contextvars snapshot; optional so the handler stays
        # unit-testable in isolation.
        self.owner = owner
        self.metrics = {"requests": 0, "accepted": 0, "declined": 0, "errors": 0}

    def session_kwargs(self) -> dict:
        """Kwargs to pass to ClientSession for elicitation support."""
        return {"elicitation_callback": self}

    def _result(self, action: str, metric: str):
        """Count *metric* and return ``ElicitResult(action)`` (accept carries empty content)."""
        self.metrics[metric] += 1
        if action == "accept":
            return _core.ElicitResult(action="accept", content={})
        return _core.ElicitResult(action=action)

    def _consent_thunk(self, message: str, description: str) -> Callable[[], str]:
        """Sync consent call, replaying the agent's contextvars snapshot when the owner captured
        one: the recv-loop task does NOT inherit them, and gateway-platform detection needs them.
        ``Context.run`` executes a context once, so it is copied per elicitation."""
        from tools.approval import request_elicitation_consent

        kwargs = {"timeout_seconds": int(self.timeout), "surface": f"mcp-elicitation/{self.server_name}"}
        captured = getattr(self.owner, "_pending_call_context", None) if self.owner else None
        if captured is None:
            return lambda: request_elicitation_consent(message, description, **kwargs)
        return lambda: captured.copy().run(request_elicitation_consent, message, description, **kwargs)

    async def __call__(self, context, params):
        """SDK elicitation callback (``ElicitationFnT``). Returns ElicitResult or ErrorData."""
        self.metrics["requests"] += 1
        # URL-mode (OAuth, payment) would need a browser + waiting for
        # notifications/elicitation/complete — not implemented; decline cleanly.
        if getattr(params, "mode", "form") == "url":
            logger.info("MCP server '%s' requested URL-mode elicitation; declining (URL-mode elicitation not implemented)",
                        self.server_name)
            return self._result("decline", "declined")

        message = getattr(params, "message", "") or f"MCP server '{self.server_name}' is requesting your approval"
        # ``requestedSchema`` on mcp 1.x, ``requested_schema`` on 2.0 (pydantic aliases don't apply
        # to attribute access) — read both or the user approves without seeing the fields.
        schema = getattr(params, "requestedSchema", None) or getattr(params, "requested_schema", None) or {}
        description = _format_elicitation_schema_summary(schema, self.server_name)
        logger.info("MCP server '%s' elicitation request: %s", self.server_name, _sanitize_error(message)[:200])
        # Lazy import avoids import-order coupling with early-bootstrap tools.approval.
        try:
            invoke_consent = self._consent_thunk(message, description)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.error("MCP server '%s' elicitation: approval system unavailable: %s", self.server_name, exc)
            return self._result("decline", "errors")
        # Offload the sync consent flow to a thread — inline it would freeze the MCP loop and
        # every other RPC on this session.
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(invoke_consent), timeout=self.timeout + self._OUTER_TIMEOUT_GRACE_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("MCP server '%s' elicitation timed out after %ds", self.server_name, int(self.timeout))
            return self._result("cancel", "errors")
        except Exception as exc:
            logger.error("MCP server '%s' elicitation failed: %s", self.server_name, exc, exc_info=True)
            return self._result("decline", "errors")
        return self._result(*self._ANSWER_RESULTS.get(answer, ("decline", "declined")))
