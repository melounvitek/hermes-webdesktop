"""Helpers for translating OpenAI-style tool schemas to Moonshot's schema subset.

Moonshot (Kimi) accepts a stricter subset of JSON Schema than OpenAI tool
calling; violations fail with HTTP 400 "tools.function.parameters is not a
valid moonshot flavored json schema". Rules applied here:

1. Every property schema must carry a ``type`` (JSON Schema allows omitting it).
2. With ``anyOf``, ``type`` belongs on the children, never the parent.
3. Enum arrays under scalar types may not contain null / empty string.
4. Every object schema must carry a ``required`` array, even an empty one.

The ``#/definitions/...`` → ``#/$defs/...`` rewrite for draft-07 refs lives in
``tools/mcp_tool._normalize_mcp_input_schema`` so it applies to all providers.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

# Values are maps of name → schema: recurse into the values, but the map itself
# is not a schema and gets no repairs.
_SCHEMA_MAP_KEYS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
# Values are lists of schemas.
_SCHEMA_LIST_KEYS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
# Values are a single nested schema (additionalProperties may also be a bool).
_SCHEMA_NODE_KEYS = frozenset({"items", "contains", "not", "additionalProperties", "propertyNames"})


def _empty_object_schema() -> Dict[str, Any]:
    return {"type": "object", "properties": {}, "required": []}


def _repair_schema(node: Any) -> Any:
    """Recursively apply the Moonshot repairs to a schema node."""
    if isinstance(node, list):
        return [_repair_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    repaired: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            repaired[key] = {sub_key: _repair_schema(sub_val) for sub_key, sub_val in value.items()}
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            repaired[key] = [_repair_schema(v) for v in value]
        elif key in _SCHEMA_NODE_KEYS and isinstance(value, dict):
            repaired[key] = _repair_schema(value)
        else:
            repaired[key] = value

    # Rule 2, plus: Moonshot rejects null-type branches inside anyOf. Drop
    # them; a single surviving branch is promoted into this node and falls
    # through to rules 1/3/4, otherwise the pruned anyOf is returned as-is.
    if "anyOf" in repaired and isinstance(repaired["anyOf"], list):
        repaired.pop("type", None)
        non_null = [b for b in repaired["anyOf"] if isinstance(b, dict) and b.get("type") != "null"]
        if not non_null or len(non_null) == len(repaired["anyOf"]):
            return repaired
        if len(non_null) > 1:
            repaired["anyOf"] = non_null
            return repaired
        merge = {k: v for k, v in repaired.items() if k != "anyOf"}
        merge.update(non_null[0])
        repaired = merge

    # Moonshot also rejects the non-standard ``nullable`` keyword.
    repaired.pop("nullable", None)

    # Rule 1 ($ref nodes take their type from the referenced definition).
    # Runs before rule 3 so enum cleanup can see the type.
    if "$ref" not in repaired:
        repaired = _fill_missing_type(repaired)

    # Rule 3: drop null/"" enum values under scalar types; drop an emptied enum.
    if isinstance(repaired.get("enum"), list) and repaired.get("type") in {"string", "integer", "number", "boolean"}:
        cleaned = [v for v in repaired["enum"] if v is not None and v != ""]
        if cleaned:
            repaired["enum"] = cleaned
        else:
            repaired.pop("enum")

    # Rule 4.
    if repaired.get("type") == "object":
        repaired = _ensure_required_array(repaired)

    return repaired


def _ensure_required_array(node: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee an object schema carries a ``required`` list (Moonshot 400s
    otherwise), pruning names that don't exist in ``properties`` — Moonshot
    also rejects dangling names. Mutates and returns ``node``."""
    props = node.get("properties")
    req = node.get("required")
    if isinstance(req, list):
        if isinstance(props, dict):
            node["required"] = [r for r in req if r in props]
    else:
        node["required"] = []
    return node


def _fill_missing_type(node: Dict[str, Any]) -> Dict[str, Any]:
    """Infer a reasonable ``type`` if this schema node has none.

    A type list collapses to its first concrete member; otherwise
    ``properties``/``required``/``additionalProperties`` → object,
    ``items``/``prefixItems`` → array, ``enum`` → type of its first value,
    else ``string`` (safest scalar).
    """
    node_type = node.get("type")
    if isinstance(node_type, list):
        concrete = next(
            (t for t in node_type if isinstance(t, str) and t not in {"", "null"}),
            "string",
        )
        return {**node, "type": concrete}
    if "type" in node and node_type not in {None, ""}:
        return node

    if "properties" in node or "required" in node or "additionalProperties" in node:
        inferred = "object"
    elif "items" in node or "prefixItems" in node:
        inferred = "array"
    elif isinstance(node.get("enum"), list) and node["enum"]:
        sample = node["enum"][0]  # bool before int: bool is an int subclass
        scalar_types = ((bool, "boolean"), (int, "integer"), (float, "number"))
        inferred = next((t for cls, t in scalar_types if isinstance(sample, cls)), "string")
    else:
        inferred = "string"

    return {**node, "type": inferred}


def sanitize_moonshot_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Return a deep-copied, Moonshot-compatible object schema; input is not mutated."""
    if not isinstance(parameters, dict):
        return _empty_object_schema()

    repaired = _repair_schema(copy.deepcopy(parameters))
    if not isinstance(repaired, dict):
        return _empty_object_schema()

    # Top-level must be an object schema.
    repaired["type"] = "object"
    repaired.setdefault("properties", {})
    return _ensure_required_array(repaired)


def sanitize_moonshot_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply ``sanitize_moonshot_tool_parameters`` to every tool's parameters.

    Returns the input list object itself when nothing needed repairing.
    """
    if not tools:
        return tools

    sanitized: List[Dict[str, Any]] = []
    any_change = False
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            sanitized.append(tool)
            continue
        params = fn.get("parameters")
        repaired = sanitize_moonshot_tool_parameters(params)
        if repaired is not params:
            any_change = True
            sanitized.append({**tool, "function": {**fn, "parameters": repaired}})
        else:
            sanitized.append(tool)

    return sanitized if any_change else tools


def is_moonshot_model(model: str | None) -> bool:
    """True for any Kimi / Moonshot model slug, regardless of aggregator prefix.

    Matches bare names (``kimi-k2.6``, ``moonshotai/Kimi-K2.6``) and
    aggregator-prefixed slugs (``nous/moonshotai/kimi-k2.6``), since aggregators
    route to Moonshot inference under their own base URL.
    """
    if not model:
        return False
    bare = model.strip().lower()
    tail = bare.rsplit("/", 1)[-1]
    if tail.startswith("kimi-") or tail == "kimi":
        return True
    # Kimi Coding Plan serves K3 under the bare slug ``k3`` (plus ``k3.1`` / ``k3-turbo``).
    if tail == "k3" or tail.startswith(("k3.", "k3-")):
        return True
    return "moonshot" in bare or "/kimi" in bare or bare.startswith("kimi")
