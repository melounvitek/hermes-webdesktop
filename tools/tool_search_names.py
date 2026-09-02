"""Reserved bridge tool names shared by tool_search and its catalog module."""

# Reserved: a user/plugin/MCP tool may not take these names — the registry's
# override protection rejects such registrations.
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"

BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})
