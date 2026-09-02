"""Tool-argument type coercion: repair string-typed values the model emitted against a tool's JSON Schema.

Models emit "42" for integers, "true" for booleans, JSON-encoded strings for
arrays/objects (also nested inside containers), and bare scalars where an array
is expected. Coercion is schema-guided and conservative: originals are kept
whenever a repair is not unambiguous.
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry

# Logger name kept as "model_tools": these messages were always emitted under
# that name and log-based tooling filters on it.
logger = logging.getLogger("model_tools")


def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce string-typed args to their JSON-Schema types; originals kept on failure.

    Models emit "42" for integers, "true" for booleans, JSON-encoded strings for
    arrays/objects (also nested inside containers), and bare scalars where an
    array is expected (wrapped in a one-element list).
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    # The model saw the SANITIZED schema (provider-illegal property keys were
    # renamed); map those keys back to the registry's wire names first.
    try:
        from tools.schema_sanitizer import unrename_tool_args
        args = unrename_tool_args(schema.get("parameters"), args)
    except Exception:  # pragma: no cover — never break dispatch
        pass

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")

        # Bare non-list value for an array schema. Strings go through
        # _coerce_value first so a JSON-encoded array is parsed and a nullable
        # "null" becomes None (not ["null"]). None itself is preserved: the tool's
        # own default handling decides between "omit" and "empty list".
        if expected == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    args[key] = coerced
                    continue
                if value.strip().startswith("["):
                    logger.warning(
                        "coerce_tool_args: %s.%s looks like a JSON array string "
                        "but could not be parsed — model may have emitted a "
                        "JSON-encoded string instead of a native array. "
                        "Falling back to single-element list.",
                        tool_name, key,
                    )
                args[key] = [value]
                logger.info(
                    "coerce_tool_args: wrapped bare string in list for %s.%s",
                    tool_name, key,
                )
                continue
            args[key] = [value]
            logger.info(
                "coerce_tool_args: wrapped bare %s in list for %s.%s",
                type(value).__name__, tool_name, key,
            )
            continue

        if not isinstance(value, str):
            # Native container: still normalize JSON-encoded elements/sub-fields.
            if (expected == "array" and isinstance(value, (list, tuple))) or (
                expected == "object" and isinstance(value, dict)
            ):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced
            if isinstance(coerced, (list, tuple, dict)):
                args[key] = _normalize_json_strings_for_schema(coerced, prop_schema)

    return args


def _schema_accepts_kind(schema: Any, kind: str) -> bool:
    """True when *schema* permits JSON type *kind* via ``type`` or any anyOf/oneOf/allOf branch."""
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and any(
            _schema_accepts_kind(b, kind) for b in branches
        ):
            return True
    return False


def _normalize_json_strings_for_schema(value: Any, schema: Any) -> Any:
    """Recursively parse JSON-encoded strings where the schema expects array/object.

    Schema-guided: a string is only parsed when its schema position expects a
    container, so legitimate JSON-looking ``type: string`` fields survive.
    Returns the same object when nothing changed (identity = cheap no-op check).
    """
    if not isinstance(schema, dict):
        return value

    if isinstance(value, str):
        trimmed = value.strip()
        expects_array = _schema_accepts_kind(schema, "array")
        expects_object = _schema_accepts_kind(schema, "object")
        if (expects_array and trimmed.startswith("[")) or (
            expects_object and trimmed.startswith("{")
        ):
            try:
                parsed = json.loads(trimmed)
            except (ValueError, TypeError):
                return value
            if (isinstance(parsed, list) and expects_array) or (isinstance(parsed, dict) and expects_object):
                value = parsed
            else:
                return value
        else:
            return value

    if isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        changed = False
        out = []
        for item in value:
            nxt = _normalize_json_strings_for_schema(item, items_schema)
            changed = changed or (nxt is not item)
            out.append(nxt)
        return out if changed else value

    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        changed = False
        out = dict(value)
        for k, prop_schema in props.items():
            if k not in value or not isinstance(prop_schema, dict):
                continue
            nxt = _normalize_json_strings_for_schema(value[k], prop_schema)
            if nxt is not value[k]:
                out[k] = nxt
                changed = True
        return out if changed else value

    return value


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Coerce string *value* to *expected_type* (str or union list); original on failure."""
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        for t in expected_type:
            result = _coerce_value(value, t, schema=schema)
            if result is not value:
                return result
        return value

    if expected_type in {"integer", "number"}:
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type == "array":
        return _coerce_json(value, list)
    if expected_type == "object":
        return _coerce_json(value, dict)
    if expected_type == "null" and value.strip().lower() == "null":
        return None
    return value


def _schema_allows_null(schema: dict | None) -> bool:
    """True when a JSON Schema fragment explicitly permits null."""
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    if schema.get("nullable") is True:
        return True
    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if isinstance(variants, list) and any(
            isinstance(v, dict) and v.get("type") == "null" for v in variants
        ):
            return True
    return False


def _coerce_json(value: str, expected_python_type: type):
    """json.loads *value* when the schema expects array/object; original string on mismatch."""
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "coerce_tool_args: failed to parse string as JSON for expected type %s: %s",
            expected_python_type.__name__,
            exc,
        )
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug(
            "coerce_tool_args: coerced string to %s via json.loads",
            expected_python_type.__name__,
        )
        return parsed
    logger.warning(
        "coerce_tool_args: JSON-parsed value is %s, expected %s — skipping coercion",
        type(parsed).__name__,
        expected_python_type.__name__,
    )
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Parse *value* as a number; original string on failure, inf/nan, or decimals when integer_only."""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    if f != f or f == float("inf") or f == float("-inf"):
        return value  # not JSON-serializable
    if f == int(f):
        return int(f)
    if integer_only:
        return value
    return f


def _coerce_boolean(value: str):
    """Parse "true"/"false" (case-insensitive); original string otherwise."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value
