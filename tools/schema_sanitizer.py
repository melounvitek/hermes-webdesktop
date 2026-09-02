"""Sanitize tool JSON schemas for broad LLM-backend compatibility.

Some backends are strict about JSON Schema shapes that OpenAI/Anthropic/most
cloud providers silently accept — llama.cpp's ``json-schema-to-grammar`` fails
the whole request (``Unrecognized schema: "object"``), Anthropic rejects
nullable ``anyOf`` at the top of ``input_schema``, Fireworks rejects ``default``
beside ``$ref``, OpenAI's Codex backend rejects top-level combinators. Known
hostile constructs:

* ``{"type": "object"}`` with no ``properties``.
* A bare string (``"object"``) where a schema dict belongs (malformed MCP output).
* ``"type": ["string", "null"]`` array types.
* ``anyOf``/``oneOf`` unions whose only purpose is to permit ``null``.
* ``default`` (etc.) alongside ``$ref`` — e.g. ``{"$ref": "#/$defs/Foo", "default": null}``.

This module walks the final tool schema tree (after MCP normalization and any
per-tool dynamic rebuilds) and fixes those in place on a deep copy. It is
deliberately conservative: it only modifies shapes the backend couldn't use.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Anthropic (and Bedrock/Vertex/Azure fronting it) reject tool input schemas
# whose property keys don't match this pattern; one bad key anywhere in the
# tools array 400s the entire request (Cloudflare's MCP ships 61 such keys).
_PROP_KEY_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
_PROP_KEY_BAD_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")

_UNION_KEYS = ("anyOf", "oneOf")
# Outer-node metadata carried onto a union's replacement node.
_UNION_META_KEYS = ("title", "description", "default", "examples")


def _empty_object() -> dict:
    return {"type": "object", "properties": {}}


def sanitize_property_key(key: str) -> str:
    """Deterministically map an arbitrary property key to a conforming one."""
    return _PROP_KEY_BAD_CHARS.sub("_", key)[:64] or "param"


def _rename_property_keys(props: dict, path: str) -> dict[str, str]:
    """Return {original_key: conforming_key} for one properties dict.

    Identity entries are omitted. Deterministic (insertion order, numeric
    suffixes on collision) so the model-visible schema and the dispatch-time
    reverse map computed from the registry's original schema always agree.
    """
    renames: dict[str, str] = {}
    taken = {k for k in props if _PROP_KEY_RE.match(k)}
    for key in props:
        if _PROP_KEY_RE.match(key):
            continue
        base = sanitize_property_key(key)
        candidate, i = base, 2
        while candidate in taken:
            suffix = f"_{i}"
            candidate = base[: 64 - len(suffix)] + suffix
            i += 1
        taken.add(candidate)
        renames[key] = candidate
        logger.debug(
            "schema_sanitizer[%s]: renamed property key %r -> %r "
            "(provider key-pattern compat)", path, key, candidate,
        )
    return renames


def unrename_tool_args(params_schema: Any, args: Any) -> Any:
    """Map sanitized property keys in model-emitted args back to wire names.

    ``params_schema`` is the ORIGINAL (unsanitized) registry schema. Recurses
    into object values and array items; unknown keys pass through untouched.
    """
    if not isinstance(params_schema, dict) or not isinstance(args, dict):
        return args
    props = params_schema.get("properties")
    if not isinstance(props, dict):
        return args
    reverse = {v: k for k, v in _rename_property_keys(props, "<unrename>").items()}
    out = {}
    for key, value in args.items():
        orig = reverse.get(key, key)
        subschema = props.get(orig)
        if isinstance(subschema, dict):
            if isinstance(value, dict):
                value = unrename_tool_args(subschema, value)
            elif isinstance(value, list) and isinstance(subschema.get("items"), dict):
                value = [
                    unrename_tool_args(subschema["items"], item)
                    if isinstance(item, dict) else item
                    for item in value
                ]
        out[orig] = value
    return out


def sanitize_tool_schemas(tools: list[dict]) -> list[dict]:
    """Return a deep-copied ``tools`` list (OpenAI format) with each tool's
    parameter schema sanitized; callers may mutate the result freely."""
    if not tools:
        return tools
    return [_sanitize_single_tool(tool) for tool in tools]


def _sanitize_single_tool(tool: dict) -> dict:
    """Deep-copy and sanitize a single OpenAI-format tool entry."""
    out = copy.deepcopy(tool)
    fn = out.get("function") if isinstance(out, dict) else None
    if not isinstance(fn, dict):
        return out

    params = fn.get("parameters")
    if not isinstance(params, dict):  # missing / non-dict → minimal valid shape
        fn["parameters"] = _empty_object()
        return out

    name = fn.get("name", "<tool>")
    top = _sanitize_node(params, path=name)
    # Guarantee the top level is an object with properties.
    if not isinstance(top, dict):
        top = _empty_object()
    else:
        if top.get("type") != "object":
            top["type"] = "object"
        if not isinstance(top.get("properties"), dict):
            top["properties"] = {}
    # Collapse nullable unions the recursive pass leaves intact (it only
    # handles the array-form ``type: [X, "null"]``); keep ``nullable: true`` so
    # runtime coercion (``model_tools._schema_allows_null``) still maps a
    # model-emitted ``"null"`` string to Python ``None``.
    top = strip_nullable_unions(top, keep_nullable_hint=True)
    top = _strip_top_level_combinators(top, path=name)
    fn["parameters"] = _strip_ref_siblings(top)
    return out


# Sibling keywords strict JSON Schema validators reject alongside ``$ref``.
_REF_FORBIDDEN_SIBLINGS = frozenset({"default"})


def _strip_ref_siblings(node: Any) -> Any:
    """Recursively drop forbidden sibling keywords from nodes carrying ``$ref``
    (Fireworks: ``keyword(s) ['default'] not allowed at the same level as $ref``)."""
    if isinstance(node, list):
        return [_strip_ref_siblings(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _strip_ref_siblings(value) for key, value in node.items()}
    if "$ref" in out:
        for key in _REF_FORBIDDEN_SIBLINGS:
            out.pop(key, None)
    return out


_TOP_LEVEL_FORBIDDEN_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")


def _strip_top_level_combinators(params: dict, *, path: str = "<tool>") -> dict:
    """Drop combinator keywords from the TOP level of a parameters schema only.

    OpenAI's Codex backend rejects ``oneOf/anyOf/allOf/enum/not`` at the top
    level. They are usually conditional-required hints; dropping them does not
    change which argument values are valid (handlers re-validate). Nested
    combinators are preserved.
    """
    if not isinstance(params, dict):
        return params
    out = dict(params)
    for key in _TOP_LEVEL_FORBIDDEN_KEYS:
        if key in out:
            logger.debug(
                "schema_sanitizer[%s]: stripped top-level %r combinator "
                "from tool parameters (strict-backend compat)",
                path, key,
            )
            out.pop(key, None)
    return out


def _is_null_branch(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "null"


def _carry_union_meta(outer: dict, replacement: dict, *, skip_default_on_ref: bool) -> None:
    """Copy outer-union metadata onto *replacement* where absent."""
    for meta_key in _UNION_META_KEYS:
        if meta_key in outer and meta_key not in replacement:
            # ``default`` is illegal alongside ``$ref`` on strict backends.
            if skip_default_on_ref and meta_key == "default" and "$ref" in replacement:
                continue
            replacement[meta_key] = outer[meta_key]


def strip_nullable_unions(
    schema: Any,
    *,
    keep_nullable_hint: bool = True,
) -> Any:
    """Collapse ``anyOf``/``oneOf`` nullable unions to the single non-null branch.

    MCP/Pydantic optional fields arrive as
    ``{"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}``;
    Anthropic rejects the null branch, and optionality is already expressed by
    the parent's ``required``. Only collapses when a null branch was dropped
    AND exactly one non-null branch survives. Outer metadata is carried over.
    ``keep_nullable_hint`` sets ``nullable: true`` on the replacement for
    downstream consumers (runtime ``"null"`` → ``None`` coercion).
    """
    if isinstance(schema, list):
        return [strip_nullable_unions(item, keep_nullable_hint=keep_nullable_hint) for item in schema]
    if not isinstance(schema, dict):
        return schema

    stripped = {
        k: strip_nullable_unions(v, keep_nullable_hint=keep_nullable_hint)
        for k, v in schema.items()
    }
    for key in _UNION_KEYS:
        variants = stripped.get(key)
        if not isinstance(variants, list):
            continue
        non_null = [item for item in variants if not _is_null_branch(item)]
        if len(non_null) == 1 and len(non_null) != len(variants):
            replacement = dict(non_null[0]) if isinstance(non_null[0], dict) else {}
            if keep_nullable_hint:
                replacement.setdefault("nullable", True)
            _carry_union_meta(stripped, replacement, skip_default_on_ref=True)
            return strip_nullable_unions(replacement, keep_nullable_hint=keep_nullable_hint)
    return stripped


_CONST_PRIMITIVE_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


def _const_branch_type(branch: Any) -> str | None:
    """JSON-Schema primitive type of a pure ``const`` branch, else None.

    Qualifies when the dict carries a primitive ``const`` and any declared
    ``type`` matches it; ``title``/``description`` are allowed, any other
    constraining keyword disqualifies.
    """
    if not isinstance(branch, dict) or "const" not in branch:
        return None
    if set(branch) - {"const", "type", "title", "description"}:
        return None
    value = branch["const"]
    # ``type(value) is`` (not isinstance): bool is a subclass of int.
    json_type = _CONST_PRIMITIVE_TYPES.get(type(value))
    if json_type is None:
        return None
    declared = branch.get("type")
    if declared is not None and declared != json_type:
        return None
    return json_type


def collapse_const_unions(schema: Any) -> Any:
    """Collapse ``anyOf``/``oneOf`` unions of same-typed consts to ``enum``.

    Ported from block/goose ``tool_schema_normalize.rs`` (Apache-2.0). MCP
    servers generated from Rust/TS union types emit
    ``{"anyOf": [{"const": "red"}, {"const": "green"}]}``; strict backends
    mishandle these while ``{"type": "string", "enum": [...]}`` is universal.

    Applies only when EVERY non-null branch is a pure ``const`` of one
    primitive type (``bool`` never merges with ``integer``). One
    ``{"type": "null"}`` branch is tolerated and recorded as ``nullable: true``
    (``strip_nullable_unions`` only handles single-non-null unions, so
    null+multi-const unions land here). Enum order preserves branch order;
    outer metadata is carried over; input is never mutated.
    """
    if isinstance(schema, list):
        return [collapse_const_unions(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out = {k: collapse_const_unions(v) for k, v in schema.items()}
    for key in _UNION_KEYS:
        variants = out.get(key)
        if not isinstance(variants, list) or not variants:
            continue
        null_branches = [
            item for item in variants if _is_null_branch(item) and "const" not in item
        ]
        const_branches = [item for item in variants if item not in null_branches]
        if len(null_branches) > 1 or not const_branches:
            continue
        branch_types = {_const_branch_type(item) for item in const_branches}
        if len(branch_types) != 1 or None in branch_types:
            continue
        replacement: dict = {
            "type": branch_types.pop(),
            "enum": [item["const"] for item in const_branches],
        }
        if null_branches:
            replacement["nullable"] = True
        _carry_union_meta(out, replacement, skip_default_on_ref=False)
        return replacement
    return out


_BARE_TYPE_NAMES = frozenset({"object", "string", "number", "integer", "boolean", "array", "null"})
# Sibling keywords whose values are NOT schemas: recursing would mistake literal
# strings like "path" for bare-string schemas. Passed through unchanged
# (``required`` remapped through property renames).
_NON_SCHEMA_LIST_KEYS = frozenset({"required", "enum", "examples", "dependentRequired"})


def _normalize_type_array(value: list, out: dict) -> None:
    """Normalize a ``type: [...]`` array into *out*.

    Several backends reject array types (llama.cpp's grammar generator; Gemini
    via OpenAI-compatible transports 400s). Per the AI-SDK behavior: one
    non-null type → ``type: X`` (+ ``nullable`` if ``null`` present); several →
    ``anyOf`` of single-type schemas so EVERY branch survives; none → ``null``
    or the object fallback. Ported from anomalyco/opencode#31877.
    """
    has_null = "null" in value
    non_null = [t for t in value if isinstance(t, str) and t != "null"]
    if len(non_null) == 1:
        out["type"] = non_null[0]
    elif len(non_null) >= 2:
        out["anyOf"] = [{"type": t} for t in non_null]
    else:
        out["type"] = "null" if has_null else "object"
        return
    if has_null:
        out.setdefault("nullable", True)


def _sanitize_node(node: Any, path: str) -> Any:
    """Recursively sanitize a JSON-Schema fragment.

    - Bare-string schema values become ``{"type": <value>}`` (unknown strings
      become a permissive object schema rather than something backends reject).
    - Object-typed nodes gain ``properties: {}`` (llama.cpp can't constrain a
      free-form object).
    - ``type`` arrays are normalized (see ``_normalize_type_array``).
    - Recurses into ``properties``, ``items``, ``additionalProperties``,
      ``anyOf``/``oneOf``/``allOf`` and ``$defs``/``definitions``; property
      keys are renamed to the provider-safe pattern and ``required`` follows.
    - ``required`` entries that don't exist in ``properties`` are pruned
      (malformed MCP schemas; built-in/plugin tools skip the MCP-level check).
    """
    if isinstance(node, str):
        if node in _BARE_TYPE_NAMES:
            logger.debug(
                "schema_sanitizer[%s]: replacing bare-string schema %r "
                "with {'type': %r}",
                path, node, node,
            )
            return _empty_object() if node == "object" else {"type": node}
        logger.debug(
            "schema_sanitizer[%s]: replacing non-schema string %r "
            "with empty object schema", path, node,
        )
        return _empty_object()

    if isinstance(node, list):
        return [_sanitize_node(item, f"{path}[{i}]") for i, item in enumerate(node)]

    if not isinstance(node, dict):
        return node

    # Renames are computed up front so ``required`` can be remapped even when
    # it precedes ``properties`` in the source dict.
    prop_renames: dict[str, str] = {}
    if isinstance(node.get("properties"), dict):
        prop_renames = _rename_property_keys(node["properties"], f"{path}.properties")

    out: dict = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, list):
            _normalize_type_array(value, out)
        elif key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            renames = prop_renames if key == "properties" else {}
            out[key] = {
                renames.get(sub_k, sub_k): _sanitize_node(sub_v, f"{path}.{key}.{renames.get(sub_k, sub_k)}")
                for sub_k, sub_v in value.items()
            }
        elif key in {"items", "additionalProperties"}:
            # Bool ``additionalProperties`` is valid and widely accepted;
            # ``items: true/false`` is non-standard but preserved rather than dropped.
            out[key] = value if isinstance(value, bool) else _sanitize_node(value, f"{path}.{key}")
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
            out[key] = [_sanitize_node(item, f"{path}.{key}[{i}]") for i, item in enumerate(value)]
        elif key in _NON_SCHEMA_LIST_KEYS:
            if key == "required" and prop_renames and isinstance(value, list):
                out[key] = [prop_renames.get(r, r) if isinstance(r, str) else r for r in value]
            else:
                out[key] = copy.deepcopy(value) if isinstance(value, (list, dict)) else value
        else:
            out[key] = _sanitize_node(value, f"{path}.{key}") if isinstance(value, (dict, list)) else value

    if out.get("type") == "object":
        if not isinstance(out.get("properties"), dict):
            out["properties"] = {}
        if isinstance(out.get("required"), list):
            props = out.get("properties") or {}
            valid = [r for r in out["required"] if isinstance(r, str) and r in props]
            if not valid:
                out.pop("required", None)
            elif len(valid) != len(out["required"]):
                out["required"] = valid
    return out


# =============================================================================
# Reactive strips — only invoked after a backend rejects a schema
# =============================================================================

_STRIP_ON_RECOVERY_KEYS = frozenset({"pattern", "format"})


def _reactive_strip(tools: list[dict], strip_node: Callable[[dict], int], log_msg: str) -> tuple[list[dict], int]:
    """Walk every tool's parameters in place, applying *strip_node* to each dict
    node (it returns how many keywords it removed). Handles OpenAI format
    (``{"function": {"parameters": ...}}``) and Responses format
    (``{"name": ..., "parameters": ...}`` — codex_responses mode, xAI, etc.).
    Returns ``(tools, stripped_count)`` — the same list reference."""
    if not tools:
        return tools, 0
    stripped = 0

    def _walk(node: Any) -> None:
        nonlocal stripped
        if isinstance(node, dict):
            stripped += strip_node(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
            _walk(fn["parameters"])
            continue
        if isinstance(tool.get("parameters"), dict):
            _walk(tool["parameters"])

    if stripped:
        logger.info(log_msg, stripped)
    return tools, stripped


def strip_pattern_and_format(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``pattern``/``format`` keywords from tool schemas, in place.

    Reactive: invoked only after llama.cpp's grammar converter rejected a
    schema with HTTP 400. Its regex engine supports a small ECMAScript subset
    (no ``\\d``/``\\w``/``\\s``) and most ``format`` values; cloud providers rely
    on these as prompting hints, so they stay in the default schema.

    Only strips as a sibling of ``type``/combinators (i.e. on schema nodes), so
    a property literally *named* ``pattern`` (``search_files``) is untouched —
    property names live inside ``properties``, not beside ``type``.
    """
    def _strip(node: dict) -> int:
        if not ("type" in node or "anyOf" in node or "oneOf" in node or "allOf" in node):
            return 0
        hits = [k for k in node if k in _STRIP_ON_RECOVERY_KEYS]
        for k in hits:
            node.pop(k, None)
        return len(hits)

    return _reactive_strip(
        tools, _strip,
        "schema_sanitizer: stripped %d pattern/format keyword(s) from "
        "tool schemas (llama.cpp grammar-parse recovery)",
    )


def strip_slash_enum(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``enum`` keywords whose string values contain ``/``, in place.

    xAI's ``/v1/responses`` and ``/v1/chat/completions`` compile schemas to a
    grammar that rejects ``/`` in enum values (HTTP 400 before any token) —
    typically MCP enums of HuggingFace model IDs or owner/name env IDs. The
    constraint is a prompting hint only; the model still sees the description.
    """
    def _strip(node: dict) -> int:
        enum_val = node.get("enum")
        if isinstance(enum_val, list) and any(isinstance(v, str) and "/" in v for v in enum_val):
            node.pop("enum", None)
            return 1
        return 0

    return _reactive_strip(
        tools, _strip,
        "schema_sanitizer: stripped %d enum keyword(s) containing '/' "
        "from tool schemas (xAI Responses grammar-compile recovery)",
    )
