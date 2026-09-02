"""Anthropic Messages API transport.

Delegates format conversion to agent/anthropic_adapter.py; owns normalization,
not client lifecycle.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall

_MCP_PREFIX = "mcp__"


def _unprefix_oauth_tool_name(name: str) -> str:
    """Reverse the OAuth-wire ``mcp__`` prefix back to the registered tool name.

    Two originals map onto one wire name (``mcp__read_file`` <- ``read_file``;
    ``mcp__linear_get_issue`` <- ``mcp_linear_get_issue``), so resolve by registry
    lookup, never rewriting a name that already resolves natively (GH-25255).
    OAuth wire aliases (e.g. chat_history_lookup -> session_search) are checked
    LAST so a real tool registered under the wire name still wins.
    """
    from agent.anthropic_adapter import _OAUTH_TOOL_NAME_REVERSE_ALIASES
    from tools.registry import registry as _tool_registry

    bare = name[len(_MCP_PREFIX):]
    for candidate in (name, "mcp_" + bare, bare):
        if _tool_registry.get_entry(candidate):
            return candidate
    return _OAUTH_TOOL_NAME_REVERSE_ALIASES.get(bare, name)


class AnthropicTransport(ProviderTransport):
    """Transport for api_mode='anthropic_messages'."""

    _STOP_REASON_MAP = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "refusal": "content_filter",
        "model_context_window_exceeded": "length",
    }

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to an Anthropic (system, messages) tuple; ``base_url`` affects thinking-signature handling."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        return convert_messages_to_anthropic(messages, base_url=kwargs.get("base_url"))

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Anthropic input_schema format."""
        from agent.anthropic_adapter import convert_tools_to_anthropic

        return convert_tools_to_anthropic(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Anthropic messages.create() kwargs (converts messages and tools internally)."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        return build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=params.get("max_tokens", 16384),
            reasoning_config=params.get("reasoning_config"),
            tool_choice=params.get("tool_choice"),
            is_oauth=params.get("is_oauth", False),
            preserve_dots=params.get("preserve_dots", False),
            context_length=params.get("context_length"),
            base_url=params.get("base_url"),
            fast_mode=params.get("fast_mode", False),
            drop_context_1m_beta=params.get("drop_context_1m_beta", False),
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Parse content blocks (text/thinking/tool_use), map stop_reason, collect reasoning_details."""
        import json
        from agent.anthropic_adapter import _sanitize_replay_block, _to_plain_data

        strip_tool_prefix = kwargs.get("strip_tool_prefix", False)
        text_parts, reasoning_parts, reasoning_details, tool_calls = [], [], [], []
        # Anthropic signs each thinking block against the blocks that PRECEDE it.
        # When thinking interleaves with tool_use, the parallel reasoning_details +
        # tool_calls lists lose that ordering and replay -> HTTP 400 "thinking ...
        # blocks cannot be modified". Keep the exact sequence for the adapter.
        ordered_blocks = []

        for block in response.content:
            block_dict = _to_plain_data(block)
            clean_block = None
            if isinstance(block_dict, dict):
                # Sanitize at capture so output-only SDK fields never persist to
                # state.db and leak back as request input on replay (HTTP 400).
                clean_block = _sanitize_replay_block(block_dict)
                if clean_block is not None:
                    ordered_blocks.append(clean_block)
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type in ("thinking", "redacted_thinking"):
                if block.type == "thinking":
                    reasoning_parts.append(block.thinking)
                # Prefer the sanitized block (replayed on the non-ordered path); raw only if sanitize dropped it.
                if isinstance(clean_block, dict):
                    reasoning_details.append(clean_block)
                elif isinstance(block_dict, dict):
                    reasoning_details.append(block_dict)
            elif block.type == "tool_use":
                name = block.name
                if strip_tool_prefix and name.startswith(_MCP_PREFIX):
                    name = _unprefix_oauth_tool_name(name)
                tool_calls.append(ToolCall(id=block.id, name=name, arguments=json.dumps(block.input)))

        provider_data = {}
        if reasoning_details:
            provider_data["reasoning_details"] = reasoning_details
        # Carry the ordered channel only for the one shape the parallel lists
        # reconstruct wrongly: signed thinking interleaved with tool_use.
        _has_signed_thinking = any(
            isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") and (b.get("signature") or b.get("data"))
            for b in ordered_blocks
        )
        if _has_signed_thinking and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in ordered_blocks):
            provider_data["anthropic_content_blocks"] = ordered_blocks

        return NormalizedResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            finish_reason=self.map_finish_reason(response.stop_reason),
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            usage=None,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Structural check. An empty content list is legitimate for ``end_turn`` (nothing to add
        after a tool turn) and ``refusal`` (Claude 4.5+ declines with empty content); treating
        either as invalid would retry a completed/deterministic response forever."""
        content_blocks = getattr(response, "content", None) if response is not None else None
        if not isinstance(content_blocks, list):
            return False
        return bool(content_blocks) or getattr(response, "stop_reason", None) in {"end_turn", "refusal"}

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Anthropic cache_read / cache_creation token counts."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return {"cached_tokens": cached, "creation_tokens": written} if cached or written else None


from agent.transports import register_transport  # noqa: E402

register_transport("anthropic_messages", AnthropicTransport)
