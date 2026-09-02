"""MCP tool schema conversion and naming: JSON-schema normalisation for provider
compatibility, mcp__server__tool naming, utility-tool schemas, include/exclude
filters and description injection scanning."""

import logging
import fnmatch
import re
from typing import Any, List
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import mcp_field

logger = logging.getLogger("tools.mcp_tool")


# Prompt-injection indicators in MCP tool descriptions. WARNING-level only:
# log but never block, since false positives would break legitimate servers.
_MCP_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
     "prompt override attempt ('ignore previous instructions')"),
    (re.compile(r"you\s+are\s+now\s+a", re.I),
     "identity override attempt ('you are now a...')"),
    (re.compile(r"your\s+new\s+(task|role|instructions?)\s+(is|are)", re.I),
     "task override attempt"),
    (re.compile(r"system\s*:\s*", re.I),
     "system prompt injection attempt"),
    (re.compile(r"<\s*(system|human|assistant)\s*>", re.I),
     "role tag injection attempt"),
    (re.compile(r"do\s+not\s+(tell|inform|mention|reveal)", re.I),
     "concealment instruction"),
    (re.compile(r"(curl|wget|fetch)\s+https?://", re.I),
     "network command in description"),
    (re.compile(r"base64\.(b64decode|decodebytes)", re.I),
     "base64 decode reference"),
    (re.compile(r"exec\s*\(|eval\s*\(", re.I),
     "code execution reference"),
    (re.compile(r"import\s+(subprocess|os|shutil|socket)", re.I),
     "dangerous import reference"),
]


def _scan_mcp_description(server_name: str, tool_name: str, description: str) -> List[str]:
    """Scan a tool description for injection patterns; returns finding strings
    (empty = clean) and logs a warning when any match."""
    findings = []
    if not description:
        return findings
    for pattern, reason in _MCP_INJECTION_PATTERNS:
        if pattern.search(description):
            findings.append(reason)
    if findings:
        logger.warning(
            "MCP server '%s' tool '%s': suspicious description content — %s. "
            "Description: %.200s",
            server_name, tool_name, "; ".join(findings),
            description,
        )
    return findings


def _normalize_mcp_input_schema(schema: dict | None) -> dict:
    """Normalize MCP input schemas so one form is valid on OpenAI, Anthropic,
    Gemini and Moonshot.

    Repairs, applied recursively: ``definitions``/``#/definitions/`` refs ->
    ``$defs`` (Moonshot rejects the draft-07 form); missing/null ``type`` on an
    object-shaped node -> ``"object"``; an object without ``properties`` gets an
    empty one so ``required`` can't dangle; ``required`` pruned to names present
    in ``properties`` (Gemini 400s otherwise); nullable ``anyOf`` unions
    collapsed to the non-null branch (Anthropic rejects nullable branches),
    optionality living solely in the parent's ``required``.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    def _rewrite_local_refs(node):
        """Promote legacy ``definitions`` to ``$defs`` — but ONLY where it is a
        JSON Schema meta-keyword, never as a property NAME inside ``properties``/
        ``patternProperties``. A tool parameter legitimately named ``definitions``
        rewritten to ``$defs`` would 400 the whole tool array (Anthropic/OpenAI
        forbid ``$`` in property names). Property names are kept verbatim and
        recursion resumes ordinary semantics inside each property's schema."""
        if isinstance(node, dict):
            normalized = {}
            for key, value in node.items():
                if key in ("properties", "patternProperties") and isinstance(value, dict):
                    normalized[key] = {
                        prop_name: _rewrite_local_refs(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                else:
                    out_key = "$defs" if key == "definitions" else key
                    normalized[out_key] = _rewrite_local_refs(value)
            ref = normalized.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                normalized["$ref"] = "#/$defs/" + ref[len("#/definitions/"):]
            return normalized
        if isinstance(node, list):
            return [_rewrite_local_refs(item) for item in node]
        return node

    def _strip_nullable_union(node):
        """Shared implementation with the Anthropic guard and global sanitizer.
        Keeps the ``nullable: true`` hint so runtime coercion can still map a
        model-emitted ``"null"`` string to ``None``."""
        from tools.schema_sanitizer import strip_nullable_unions

        return strip_nullable_unions(node, keep_nullable_hint=True)

    def _collapse_const_unions(node):
        """Collapse anyOf/oneOf unions of same-typed consts to enums. Must run
        AFTER the nullable strip: consts -> enum, null branch -> ``nullable`` hint."""
        from tools.schema_sanitizer import collapse_const_unions

        return collapse_const_unions(node)

    def _repair_object_shape(node):
        """Recursively fill missing object ``type``, ensure ``properties``, prune ``required``."""
        if isinstance(node, list):
            return [_repair_object_shape(item) for item in node]
        if not isinstance(node, dict):
            return node

        repaired = {k: _repair_object_shape(v) for k, v in node.items()}

        if not repaired.get("type") and (
            "properties" in repaired or "required" in repaired
        ):
            repaired["type"] = "object"

        if repaired.get("type") == "object":
            if not isinstance(repaired.get("properties"), dict):
                repaired["properties"] = {}

            required = repaired.get("required")
            if isinstance(required, list):
                props = repaired.get("properties") or {}
                valid = [r for r in required if isinstance(r, str) and r in props]
                if len(valid) != len(required):
                    if valid:
                        repaired["required"] = valid
                    else:
                        repaired.pop("required", None)

        return repaired

    normalized = _rewrite_local_refs(schema)
    normalized = _strip_nullable_union(normalized)
    normalized = _collapse_const_unions(normalized)
    normalized = _repair_object_shape(normalized)

    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized = {**normalized, "properties": {}}

    return normalized


def sanitize_mcp_name_component(value: str) -> str:
    """Replace every char outside ``[A-Za-z0-9_]`` with ``_`` (hyphens included,
    the historical behavior) so generated names pass provider validation."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


# ``mcp__<server>__<tool>``: the convention shared by Claude Code, Codex and
# OpenCode. The double underscore disambiguates the server/tool boundary even
# when either contains underscores, and matches the Anthropic-OAuth wire form.
MCP_TOOL_NAME_PREFIX = "mcp__"


_MCP_NAME_DELIM = "__"


def mcp_prefixed_tool_name(server_name: str, tool_name: str) -> str:
    """Registry/wire name: ``mcp__<sanitizedServer>__<sanitizedTool>``."""
    safe_server = sanitize_mcp_name_component(server_name)
    safe_tool = sanitize_mcp_name_component(tool_name)
    return f"{MCP_TOOL_NAME_PREFIX}{safe_server}{_MCP_NAME_DELIM}{safe_tool}"


def _convert_mcp_schema(server_name: str, mcp_tool) -> dict:
    """Convert an MCP ``Tool`` (``.input_schema``, or ``.inputSchema`` before
    mcp 2.0) to a ``registry.register(schema=...)`` dict."""
    return {
        "name": mcp_prefixed_tool_name(server_name, mcp_tool.name),
        "description": strip_unicode_tags(
            mcp_tool.description or f"MCP tool {mcp_tool.name} from {server_name}"
        ),
        "parameters": _normalize_mcp_input_schema(
            mcp_field(mcp_tool, "input_schema", "inputSchema")
        ),
    }


def _build_utility_schemas(server_name: str) -> List[dict]:
    """Schemas for the resource/prompt utility tools as ``{schema, handler_key}`` dicts."""
    return [
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "list_resources"),
                "description": f"List available resources from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            "handler_key": "list_resources",
        },
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "read_resource"),
                "description": f"Read a resource by URI from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "URI of the resource to read",
                        },
                    },
                    "required": ["uri"],
                },
            },
            "handler_key": "read_resource",
        },
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "list_prompts"),
                "description": f"List available prompts from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            "handler_key": "list_prompts",
        },
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "get_prompt"),
                "description": f"Get a prompt by name from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the prompt to retrieve",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Optional arguments to pass to the prompt",
                            "properties": {},
                            "additionalProperties": True,
                        },
                    },
                    "required": ["name"],
                },
            },
            "handler_key": "get_prompt",
        },
    ]


def _normalize_name_filter(value: Any, label: str) -> set[str]:
    """Normalize include/exclude config to a set of exact names or fnmatch globs."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    logger.warning("MCP config %s must be a string or list of strings; ignoring %r", label, value)
    return set()


def matches_name_filter(tool_name: str, patterns: set[str]) -> bool:
    """True if ``tool_name`` matches any entry: exact names literally, entries
    with ``*``/``?``/``[`` as case-sensitive globs (same semantics as
    ``approvals.deny``). Exact membership is checked first so big lists stay O(1)."""
    if not patterns:
        return False
    if tool_name in patterns:
        return True
    return any(
        fnmatch.fnmatchcase(tool_name, p)
        for p in patterns
        if "*" in p or "?" in p or "[" in p
    )


_UTILITY_CAPABILITY_METHODS = {
    "list_resources": "list_resources",
    "read_resource": "read_resource",
    "list_prompts": "list_prompts",
    "get_prompt": "get_prompt",
}


# Utility handler -> capability key that must be non-None on the server's
# ``initialize`` response for the handler to be registered. Without this gate a
# tools-only server got all four stubs and every call returned JSON-RPC -32601,
# making the model conclude the server was broken.
_UTILITY_CAPABILITY_ATTRS = {
    "list_resources": "resources",
    "read_resource": "resources",
    "list_prompts": "prompts",
    "get_prompt": "prompts",
}
