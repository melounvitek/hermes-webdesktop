"""MCP client-side handlers for server-initiated requests: sampling (sampling/createMessage, text and tool-use results) and elicitation. Split from tools/mcp_tool.py."""

import asyncio
import json
import logging
import time
from typing import List, Optional
from tools.mcp_tool_common import _MISSING, _exc_str, _safe_numeric, _sanitize_error, mcp_field, _core
from tools.mcp_tool_schema import _normalize_mcp_input_schema

logger = logging.getLogger("tools.mcp_tool")


class SamplingHandler:
    """Handles sampling/createMessage requests for one MCP server.

    Deprecated upstream (MCP 2026-07-28, SEP-2577, 12-month window): stays fully
    functional because handshake-era servers still issue it, but do NOT grow new
    capability here — modern servers use MRTR, handled by the SDK session layer.

    Callable; passed to ``ClientSession`` as ``sampling_callback``. All state
    (rate-limit timestamps, metrics, tool-loop counter) is per instance. Runs on
    the MCP background loop; the sync LLM call is offloaded via ``asyncio.to_thread``.
    """

    _STOP_REASON_MAP = {"stop": "endTurn", "length": "maxTokens", "tool_calls": "toolUse"}

    def __init__(self, server_name: str, config: dict):
        self.server_name = server_name
        self.max_rpm = _safe_numeric(config.get("max_rpm", 10), 10, int)
        self.timeout = _safe_numeric(config.get("timeout", 30), 30, float)
        self.max_tokens_cap = _safe_numeric(config.get("max_tokens_cap", 4096), 4096, int)
        self.max_tool_rounds = _safe_numeric(
            config.get("max_tool_rounds", 5), 5, int, minimum=0,
        )
        self.model_override = config.get("model")
        self.allowed_models = config.get("allowed_models", [])

        _log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
        self.audit_level = _log_levels.get(
            str(config.get("log_level", "info")).lower(), logging.INFO,
        )

        self._rate_timestamps: List[float] = []
        self._tool_loop_count = 0
        self.metrics = {"requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0}

    def _check_rate_limit(self) -> bool:
        """Sliding-window (60s) limiter; True if the request is allowed."""
        now = time.time()
        window = now - 60
        self._rate_timestamps[:] = [t for t in self._rate_timestamps if t > window]
        if len(self._rate_timestamps) >= self.max_rpm:
            return False
        self._rate_timestamps.append(now)
        return True

    def _resolve_model(self, preferences) -> Optional[str]:
        """Config override > server hint > None (use default)."""
        if self.model_override:
            return self.model_override
        if preferences and hasattr(preferences, "hints") and preferences.hints:
            for hint in preferences.hints:
                if hasattr(hint, "name") and hint.name:
                    return hint.name
        return None

    @staticmethod
    def _extract_tool_result_text(block) -> str:
        """Extract text from a ToolResultContent block."""
        if not hasattr(block, "content") or block.content is None:
            return ""
        items = block.content if isinstance(block.content, list) else [block.content]
        return "\n".join(item.text for item in items if hasattr(item, "text"))

    def _convert_messages(self, params) -> List[dict]:
        """Convert MCP SamplingMessages to OpenAI format.

        Uses ``msg.content_as_list`` when the SDK provides it; dispatches per
        block by duck-typing.
        """
        # A tool-use id is the discriminator for a tool *result* block; it must be
        # read under both spellings (mcp_field) — on mcp 2.x a bare
        # ``hasattr(b, "toolUseId")`` is False, silently dropping tool results.
        def _tool_use_id(block):
            return mcp_field(block, "tool_use_id", "toolUseId", _MISSING)

        def _is_tool_use(block):
            return hasattr(block, "name") and hasattr(block, "input")

        messages: List[dict] = []
        for msg in params.messages:
            blocks = msg.content_as_list if hasattr(msg, "content_as_list") else (
                msg.content if isinstance(msg.content, list) else [msg.content]
            )

            tool_results = [b for b in blocks if _tool_use_id(b) is not _MISSING]
            tool_uses = [
                b for b in blocks
                if _is_tool_use(b) and _tool_use_id(b) is _MISSING
            ]
            content_blocks = [
                b for b in blocks
                if _tool_use_id(b) is _MISSING and not _is_tool_use(b)
            ]

            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": _tool_use_id(tr),
                    "content": self._extract_tool_result_text(tr),
                })

            if tool_uses:
                tc_list = []
                for tu in tool_uses:
                    tc_list.append({
                        "id": getattr(tu, "id", f"call_{len(tc_list)}"),
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.input, ensure_ascii=False) if isinstance(tu.input, dict) else str(tu.input),
                        },
                    })
                msg_dict: dict = {"role": msg.role, "tool_calls": tc_list}
                text_parts = [b.text for b in content_blocks if hasattr(b, "text")]
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                messages.append(msg_dict)
            elif content_blocks:
                # Pure text/image content.
                if len(content_blocks) == 1 and hasattr(content_blocks[0], "text"):
                    messages.append({"role": msg.role, "content": content_blocks[0].text})
                else:
                    parts = []
                    for block in content_blocks:
                        block_mime = mcp_field(
                            block, "mime_type", "mimeType", _MISSING
                        )
                        if hasattr(block, "text"):
                            parts.append({"type": "text", "text": block.text})
                        elif hasattr(block, "data") and block_mime is not _MISSING:
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{block_mime};base64,{block.data}"},
                            })
                        else:
                            logger.warning(
                                "Unsupported sampling content block type: %s (skipped)",
                                type(block).__name__,
                            )
                    if parts:
                        messages.append({"role": msg.role, "content": parts})

        return messages

    @staticmethod
    def _error(message: str, code: int = -1):
        """Return ErrorData (MCP spec) or raise as fallback."""
        if _core._MCP_SAMPLING_TYPES:
            return _core.ErrorData(code=code, message=message)
        raise Exception(message)

    def _build_tool_use_result(self, choice, response):
        """Build a CreateMessageResultWithTools from an LLM tool_calls response."""
        self.metrics["tool_use_count"] += 1

        # Tool-loop governance.
        if self.max_tool_rounds == 0:
            self._tool_loop_count = 0
            return self._error(
                f"Tool loops disabled for server '{self.server_name}' (max_tool_rounds=0)"
            )

        self._tool_loop_count += 1
        if self._tool_loop_count > self.max_tool_rounds:
            self._tool_loop_count = 0
            return self._error(
                f"Tool loop limit exceeded for server '{self.server_name}' "
                f"(max {self.max_tool_rounds} rounds)"
            )

        content_blocks = []
        for tc in choice.message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "MCP server '%s': malformed tool_calls arguments "
                        "from LLM (wrapping as raw): %.100s",
                        self.server_name, args,
                    )
                    parsed = {"_raw": args}
            else:
                parsed = args if isinstance(args, dict) else {"_raw": str(args)}

            content_blocks.append(_core.ToolUseContent(
                type="tool_use",
                id=tc.id,
                name=tc.function.name,
                input=parsed,
            ))

        logger.log(
            self.audit_level,
            "MCP server '%s' sampling response: model=%s, tokens=%s, tool_calls=%d",
            self.server_name, response.model,
            getattr(getattr(response, "usage", None), "total_tokens", "?"),
            len(content_blocks),
        )

        return _core.CreateMessageResultWithTools(
            role="assistant",
            content=content_blocks,
            model=response.model,
            stopReason="toolUse",
        )

    def _build_text_result(self, choice, response):
        """Build a CreateMessageResult from a normal text response (resets the tool loop)."""
        self._tool_loop_count = 0
        response_text = choice.message.content or ""

        logger.log(
            self.audit_level,
            "MCP server '%s' sampling response: model=%s, tokens=%s",
            self.server_name, response.model,
            getattr(getattr(response, "usage", None), "total_tokens", "?"),
        )

        return _core.CreateMessageResult(
            role="assistant",
            content=_core.TextContent(type="text", text=_sanitize_error(response_text)),
            model=response.model,
            stopReason=self._STOP_REASON_MAP.get(choice.finish_reason, "endTurn"),
        )

    def session_kwargs(self) -> dict:
        """Kwargs to pass to ClientSession for sampling support."""
        return {
            "sampling_callback": self,
            "sampling_capabilities": _core.SamplingCapability(
                tools=_core.SamplingToolsCapability(),
            ),
        }

    async def __call__(self, context, params):
        """SDK sampling callback (``SamplingFnT``). Returns CreateMessageResult,
        CreateMessageResultWithTools, or ErrorData."""
        if not self._check_rate_limit():
            logger.warning(
                "MCP server '%s' sampling rate limit exceeded (%d/min)",
                self.server_name, self.max_rpm,
            )
            self.metrics["errors"] += 1
            return self._error(
                f"Sampling rate limit exceeded for server '{self.server_name}' "
                f"({self.max_rpm} requests/minute)"
            )

        model = self._resolve_model(
            mcp_field(params, "model_preferences", "modelPreferences")
        )

        from agent.auxiliary_client import call_llm

        resolved_model = model or self.model_override or ""

        if self.allowed_models and resolved_model and resolved_model not in self.allowed_models:
            logger.warning(
                "MCP server '%s' requested model '%s' not in allowed_models",
                self.server_name, resolved_model,
            )
            self.metrics["errors"] += 1
            return self._error(
                f"Model '{resolved_model}' not allowed for server "
                f"'{self.server_name}'. Allowed: {', '.join(self.allowed_models)}"
            )

        messages = self._convert_messages(params)
        system_prompt = mcp_field(params, "system_prompt", "systemPrompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        max_tokens = min(
            mcp_field(params, "max_tokens", "maxTokens", self.max_tokens_cap),
            self.max_tokens_cap,
        )
        call_temperature = None
        if hasattr(params, "temperature") and params.temperature is not None:
            call_temperature = params.temperature

        # Forward server-provided tools.
        call_tools = None
        server_tools = getattr(params, "tools", None)
        if server_tools:
            call_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", "") or "",
                        "parameters": _normalize_mcp_input_schema(
                            mcp_field(t, "input_schema", "inputSchema")
                        ),
                    },
                }
                for t in server_tools
            ]

        logger.log(
            self.audit_level,
            "MCP server '%s' sampling request: model=%s, max_tokens=%d, messages=%d",
            self.server_name, resolved_model, max_tokens, len(messages),
        )

        # Offload the sync LLM call so the MCP loop is not blocked.
        def _sync_call():
            return call_llm(
                task="mcp",
                model=resolved_model or None,
                messages=messages,
                temperature=call_temperature,
                max_tokens=max_tokens,
                tools=call_tools,
                timeout=self.timeout,
            )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_sync_call), timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            self.metrics["errors"] += 1
            return self._error(
                f"Sampling LLM call timed out after {self.timeout}s "
                f"for server '{self.server_name}'"
            )
        except Exception as exc:
            self.metrics["errors"] += 1
            return self._error(
                f"Sampling LLM call failed: {_sanitize_error(_exc_str(exc))}"
            )

        # Empty choices happen on content filtering / provider errors.
        if not getattr(response, "choices", None):
            self.metrics["errors"] += 1
            return self._error(
                f"LLM returned empty response (no choices) for server "
                f"'{self.server_name}'"
            )

        choice = response.choices[0]
        self.metrics["requests"] += 1
        total_tokens = getattr(getattr(response, "usage", None), "total_tokens", 0)
        if isinstance(total_tokens, int):
            self.metrics["tokens_used"] += total_tokens

        if (
            choice.finish_reason == "tool_calls"
            and hasattr(choice.message, "tool_calls")
            and choice.message.tool_calls
        ):
            return self._build_tool_use_result(choice, response)

        return self._build_text_result(choice, response)


def _format_elicitation_schema_summary(schema: dict, server_name: str) -> str:
    """Render a flat-object requested_schema as a human-readable field list
    (names, types, descriptions) so the user knows what they're approving."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or not props:
        return f"Approval requested by MCP server '{server_name}'."

    lines = [f"Fields requested by MCP server '{server_name}':"]
    for field_name, field_spec in props.items():
        field_type = ""
        field_desc = ""
        if isinstance(field_spec, dict):
            field_type = str(field_spec.get("type", "") or "")
            field_desc = str(field_spec.get("description", "") or "")
        suffix = f" ({field_type})" if field_type else ""
        if field_desc:
            lines.append(f"  - {field_name}{suffix}: {field_desc}")
        else:
            lines.append(f"  - {field_name}{suffix}")
    return "\n".join(lines)


class ElicitationHandler:
    """Handles ``elicitation/create`` requests for one MCP server.

    Callable; passed to ``ClientSession`` as ``elicitation_callback``. Form-mode
    requests route through Hermes' approval system (CLI, TUI, Telegram, ...);
    URL-mode is declined as unsupported. Fail-closed: any timeout, exception,
    or unexpected state returns decline/cancel, never a silent accept.
    """

    # asyncio-side safety net over the approval's own input() timeout so the
    # MCP loop never blocks indefinitely if the inner timeout is bypassed.
    _OUTER_TIMEOUT_GRACE_SECONDS = 5

    def __init__(self, server_name: str, config: dict, owner: Optional["MCPServerTask"] = None):
        self.server_name = server_name
        # Default 5 min mirrors the gateway approval default so async surfaces
        # (Telegram, Slack) have time to respond.
        self.timeout = _safe_numeric(config.get("timeout", 300), 300, float)
        # Back-reference for the agent's contextvars snapshot; optional so the
        # handler stays unit-testable in isolation.
        self.owner = owner
        self.metrics = {
            "requests": 0,
            "accepted": 0,
            "declined": 0,
            "errors": 0,
        }

    def session_kwargs(self) -> dict:
        """Kwargs to pass to ClientSession for elicitation support."""
        return {"elicitation_callback": self}

    async def __call__(self, context, params):
        """SDK elicitation callback (``ElicitationFnT``). Returns ElicitResult or ErrorData."""
        self.metrics["requests"] += 1

        # URL-mode (OAuth, payment) would need a browser + waiting for
        # notifications/elicitation/complete — not implemented; decline cleanly.
        mode = getattr(params, "mode", "form")
        if mode == "url":
            logger.info(
                "MCP server '%s' requested URL-mode elicitation; "
                "declining (URL-mode elicitation not implemented)",
                self.server_name,
            )
            self.metrics["declined"] += 1
            return _core.ElicitResult(action="decline")

        message = getattr(params, "message", "") or (
            f"MCP server '{self.server_name}' is requesting your approval"
        )
        # ``requestedSchema`` on mcp 1.x, ``requested_schema`` on 2.0 (pydantic
        # aliases don't apply to attribute access) — read both or the user is
        # asked to approve without seeing which fields the server wants.
        schema = (
            getattr(params, "requestedSchema", None)
            or getattr(params, "requested_schema", None)
            or {}
        )
        description = _format_elicitation_schema_summary(schema, self.server_name)

        logger.info(
            "MCP server '%s' elicitation request: %s",
            self.server_name, _sanitize_error(message)[:200],
        )

        # Lazy import avoids import-order coupling with early-bootstrap tools.approval.
        try:
            from tools.approval import request_elicitation_consent
        except Exception as exc:  # pragma: no cover -- defensive
            logger.error(
                "MCP server '%s' elicitation: approval system unavailable: %s",
                self.server_name, exc,
            )
            self.metrics["errors"] += 1
            return _core.ElicitResult(action="decline")

        # Offload the sync consent flow to a thread — inline it would freeze the
        # MCP loop and every other RPC on this session. The recv-loop task does
        # NOT inherit the agent's contextvars, so replay the snapshot captured on
        # owner._pending_call_context for gateway-platform detection.
        captured = getattr(self.owner, "_pending_call_context", None) if self.owner else None

        def _invoke_consent() -> str:
            if captured is None:
                return request_elicitation_consent(
                    message,
                    description,
                    timeout_seconds=int(self.timeout),
                    surface=f"mcp-elicitation/{self.server_name}",
                )
            # Context.run executes a context once — copy so multiple
            # elicitations within one tool call work.
            return captured.copy().run(
                request_elicitation_consent,
                message,
                description,
                timeout_seconds=int(self.timeout),
                surface=f"mcp-elicitation/{self.server_name}",
            )

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(_invoke_consent),
                timeout=self.timeout + self._OUTER_TIMEOUT_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "MCP server '%s' elicitation timed out after %ds",
                self.server_name, int(self.timeout),
            )
            self.metrics["errors"] += 1
            return _core.ElicitResult(action="cancel")
        except Exception as exc:
            logger.error(
                "MCP server '%s' elicitation failed: %s",
                self.server_name, exc, exc_info=True,
            )
            self.metrics["errors"] += 1
            return _core.ElicitResult(action="decline")

        if answer == "accept":
            self.metrics["accepted"] += 1
            return _core.ElicitResult(action="accept", content={})
        if answer == "cancel":
            self.metrics["errors"] += 1
            return _core.ElicitResult(action="cancel")
        self.metrics["declined"] += 1
        return _core.ElicitResult(action="decline")
