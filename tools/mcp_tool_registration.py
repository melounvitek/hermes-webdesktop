"""Registering a connected (or schema-cached) MCP server's tools into the tool
registry: include/exclude filtering, trust-tier metadata capture, utility-tool
selection, name-collision resolution and the schema-cache write-through."""

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List
from tools.mcp_tool_common import _parse_boolish, _core
from tools.mcp_tool_handlers import _make_check_fn, _make_get_prompt_handler, _make_list_prompts_handler, _make_list_resources_handler, _make_read_resource_handler
from tools.mcp_tool_common import _resolve_tool_timeout
from tools.mcp_tool_schema import _UTILITY_CAPABILITY_ATTRS, _UTILITY_CAPABILITY_METHODS, _build_utility_schemas, _normalize_name_filter, matches_name_filter

if TYPE_CHECKING:  # pragma: no cover
    from tools.mcp_tool import MCPServerTask

logger = logging.getLogger("tools.mcp_tool")


def _normalize_server_trust(value: Any) -> str:
    """Normalize a config ``trust`` value: None -> ``full`` (backward-compatible
    default); any unrecognized string -> ``untrusted``, so a misspelled tier
    fails closed rather than silently disabling gating."""
    if value is None:
        return _core._TRUST_FULL
    text = str(value).strip().lower()
    if text == _core._TRUST_FULL:
        return _core._TRUST_FULL
    if text == _core._TRUST_UNTRUSTED:
        return _core._TRUST_UNTRUSTED
    logger.warning(
        "MCP trust: unrecognized trust value %r — treating as 'untrusted' "
        "(valid values: full, untrusted)", value,
    )
    return _core._TRUST_UNTRUSTED


def _annotation_read_only_hint(mcp_tool: Any) -> bool:
    """True only when annotations (SDK object or schema-cache dict) carry
    ``readOnlyHint is True``; anything else is False — unknown metadata means
    the tool must be treated as write-capable."""
    annotations = getattr(mcp_tool, "annotations", None)
    if annotations is None:
        return False
    if isinstance(annotations, dict):
        hint = annotations.get("readOnlyHint")
    else:
        hint = getattr(annotations, "readOnlyHint", None)
    return hint is True


def _record_tool_trust_metadata(
    server_name: str, config: dict, tools: List[Any]
) -> None:
    """Capture per-server trust and per-tool readOnlyHint at discovery."""
    with _core._lock:
        _core._server_trust_levels[server_name] = _normalize_server_trust(
            (config or {}).get("trust")
        )
        hints = _core._tool_read_only_hints.setdefault(server_name, {})
        for tool in tools:
            name = getattr(tool, "name", None)
            if name:
                hints[name] = _annotation_read_only_hint(tool)


def _track_mcp_tool_server(tool_name: str, server_name: str) -> None:
    """Remember the exact raw MCP server that registered *tool_name*."""
    with _core._lock:
        _core._mcp_tool_server_names[tool_name] = server_name


def _forget_mcp_tool_server(tool_name: str) -> None:
    """Forget MCP server provenance for a deregistered tool."""
    with _core._lock:
        _core._mcp_tool_server_names.pop(tool_name, None)


def _select_utility_schemas(server_name: str, server: "MCPServerTask", config: dict) -> List[dict]:
    """Select utility schemas based on config and server capabilities."""
    tools_filter = config.get("tools") or {}
    resources_enabled = _parse_boolish(tools_filter.get("resources"), default=True)
    prompts_enabled = _parse_boolish(tools_filter.get("prompts"), default=True)

    # ``initialize_result.capabilities`` is the source of truth: its sub-objects
    # are non-None iff the server advertises that request family. The old
    # ``hasattr(server.session, ...)`` gate never filtered anything because
    # ClientSession defines all four methods on the class.
    advertised_caps = None
    init_result = getattr(server, "initialize_result", None)
    if init_result is not None:
        advertised_caps = getattr(init_result, "capabilities", None)

    selected: List[dict] = []
    for entry in _build_utility_schemas(server_name):
        handler_key = entry["handler_key"]
        if handler_key in {"list_resources", "read_resource"} and not resources_enabled:
            logger.debug("MCP server '%s': skipping utility '%s' (resources disabled)", server_name, handler_key)
            continue
        if handler_key in {"list_prompts", "get_prompt"} and not prompts_enabled:
            logger.debug("MCP server '%s': skipping utility '%s' (prompts disabled)", server_name, handler_key)
            continue

        if advertised_caps is not None:
            cap_attr = _UTILITY_CAPABILITY_ATTRS[handler_key]
            if getattr(advertised_caps, cap_attr, None) is None:
                logger.debug(
                    "MCP server '%s': skipping utility '%s' "
                    "(server does not advertise '%s' capability)",
                    server_name,
                    handler_key,
                    cap_attr,
                )
                continue
        else:
            # Legacy fallback when initialize_result wasn't captured (test
            # fixtures, older paths): register every stub, as before.
            required_method = _UTILITY_CAPABILITY_METHODS[handler_key]
            if not hasattr(server.session, required_method):
                logger.debug(
                    "MCP server '%s': skipping utility '%s' (session lacks %s)",
                    server_name,
                    handler_key,
                    required_method,
                )
                continue
        selected.append(entry)
    return selected


def _existing_tool_names() -> List[str]:
    """Return tool names for all currently connected servers."""
    names: List[str] = []
    for _sname, server in _core._servers.items():
        if hasattr(server, "_registered_tool_names"):
            names.extend(server._registered_tool_names)
            continue
        for mcp_tool in server._tools:
            schema = _core._convert_mcp_schema(server.name, mcp_tool)
            names.append(schema["name"])
    # Lazy servers registered from the schema cache have no MCPServerTask yet —
    # their tools live only in the registry.
    with _core._lock:
        lazy_names = [
            n
            for sname, tool_names in _core._lazy_server_tool_names.items()
            if sname not in _core._servers
            for n in tool_names
        ]
    names.extend(lazy_names)
    return names


# Utility tool key -> handler factory; each takes (server_name, tool_timeout).
_UTILITY_HANDLER_FACTORIES = {
    "list_resources": _make_list_resources_handler,
    "read_resource": _make_read_resource_handler,
    "list_prompts": _make_list_prompts_handler,
    "get_prompt": _make_get_prompt_handler,
}


def _make_tool_filter(name: str, config: dict) -> Callable[[str], bool]:
    """Build the include/exclude predicate for a server's tool names.

    Rules: ``tools.include`` is a whitelist, ``tools.exclude`` a blacklist;
    entries may be exact names or fnmatch globs; include wins over exclude;
    ``include: []`` is an explicit empty whitelist (register nothing); neither
    set registers everything.
    """
    tools_filter = config.get("tools") or {}
    include_raw = tools_filter.get("include")
    include_set = _normalize_name_filter(include_raw, f"mcp_servers.{name}.tools.include")
    include_active = isinstance(include_raw, (str, list, tuple, set))
    exclude_set = _normalize_name_filter(
        tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude"
    )

    def _should_register(tool_name: str) -> bool:
        if include_active:
            return matches_name_filter(tool_name, include_set)
        if exclude_set:
            return not matches_name_filter(tool_name, exclude_set)
        return True

    return _should_register


def _resolve_name_collisions(name: str, candidates: List[dict]):
    """Preflight registry-name collisions among one server's candidates.

    Returns ``(unique_candidates, ambiguous_names, shadowed_utilities)``. Exact
    duplicate rows (same name + origin) are dropped silently; a generated
    utility that normalizes onto a server-native tool's name is shadowed (the
    native tool wins); any other multi-origin collision is ambiguous and every
    colliding entry is skipped (fail closed).
    """
    unique_candidates: List[dict] = []
    seen_candidates: set[tuple[str, str]] = set()
    origins_by_name: Dict[str, set[str]] = {}
    for candidate in candidates:
        key = (candidate["registry_name"], candidate["origin"])
        if key in seen_candidates:
            logger.debug(
                "MCP server '%s': duplicate registration candidate %s for '%s'; "
                "keeping one",
                name,
                candidate["origin"],
                candidate["registry_name"],
            )
            continue
        seen_candidates.add(key)
        unique_candidates.append(candidate)
        origins_by_name.setdefault(candidate["registry_name"], set()).add(
            candidate["origin"]
        )

    ambiguous_names: Dict[str, List[str]] = {}
    shadowed_utilities: set[tuple[str, str]] = set()
    for registry_name, origins in origins_by_name.items():
        if len(origins) <= 1:
            continue
        utility_origins = sorted(
            o for o in origins if o.startswith("generated utility ")
        )
        native_origins = sorted(origins - set(utility_origins))
        if len(native_origins) == 1 and utility_origins:
            for util_origin in utility_origins:
                shadowed_utilities.add((registry_name, util_origin))
            logger.info(
                "MCP server '%s': generated utility %s normalizes onto "
                "server-native %s — keeping the native tool and dropping the "
                "utility (the utility only applies when the server has no such "
                "tool of its own)",
                name,
                ", ".join(utility_origins),
                native_origins[0],
            )
            continue
        ambiguous_names[registry_name] = sorted(origins)

    for registry_name, origins in sorted(ambiguous_names.items()):
        logger.error(
            "MCP server '%s': name normalization collision for '%s' from %s; "
            "skipping every colliding entry instead of choosing an arbitrary "
            "handler",
            name,
            registry_name,
            ", ".join(origins),
        )
    return unique_candidates, ambiguous_names, shadowed_utilities


def _write_schema_cache(name: str, server: "MCPServerTask", config: dict, should_register) -> None:
    """Write-through: persist the manifest so the next startup can register this
    server lazily without spawning it. Never raises."""
    try:
        from tools.mcp_schema_cache import config_fingerprint, write_cache_entry

        tools_payload: List[dict] = []
        for mcp_tool in server._tools:
            if not should_register(mcp_tool.name):
                continue
            schema_obj = getattr(mcp_tool, "inputSchema", None)
            tools_payload.append({
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "inputSchema": schema_obj if isinstance(schema_obj, dict) else {},
                # Persisted so the lazy path trust-gates identically next startup.
                "annotations": {
                    "readOnlyHint": _annotation_read_only_hint(mcp_tool),
                },
            })
        utility_payload = [
            {"schema": entry["schema"], "handler_key": entry["handler_key"]}
            for entry in _select_utility_schemas(name, server, config)
        ]
        cache_meta = getattr(server, "_list_cache_meta", None) or {}
        write_cache_entry(
            name,
            config_fingerprint(config),
            tools=tools_payload,
            utility_tools=utility_payload,
            ttl_ms=cache_meta.get("ttl_ms"),
            cache_scope=cache_meta.get("cache_scope"),
        )
    except Exception as exc:
        logger.debug("MCP schema cache write failed for '%s': %s", name, exc)


def _register_server_tools(name: str, server: "MCPServerTask", config: dict) -> List[str]:
    """Register an already-connected server's tools (plus utility tools) into
    the registry; used by initial discovery and list_changed refresh.

    Toolset resolution for ``mcp-{server}`` / raw-name aliases derives from the
    live registry rather than mutating ``toolsets.TOOLSETS``. Lossy name
    normalization can map distinct raw names (``read-file``/``read_file``) to
    one registry name; such collisions fail closed — every ambiguous entry is
    skipped. Returns the registered prefixed names.
    """
    from tools.registry import registry

    registered_names: List[str] = []
    toolset_name = f"mcp-{name}"

    _should_register = _make_tool_filter(name, config)
    check_fn = _make_check_fn(name)
    candidates: List[dict] = []

    # Security boundary: capture trust tier and readOnlyHint NOW, at discovery,
    # so the call-time gate classifies from data we control, not re-read
    # server-supplied state.
    _record_tool_trust_metadata(name, config, server._tools)

    for mcp_tool in server._tools:
        if not _should_register(mcp_tool.name):
            logger.debug(
                "MCP server '%s': skipping tool '%s' (filtered by config)",
                name,
                mcp_tool.name,
            )
            continue

        _core._scan_mcp_description(name, mcp_tool.name, mcp_tool.description or "")
        schema = _core._convert_mcp_schema(name, mcp_tool)
        candidates.append(
            {
                "registry_name": schema["name"],
                "origin": f"tool {mcp_tool.name!r}",
                "schema": schema,
                "handler": _core._make_tool_handler(
                    name, mcp_tool.name, server.tool_timeout
                ),
                "check_fn": check_fn,
            }
        )

    # Generated resource/prompt utility tools share the same namespace as raw
    # MCP tools, so they must participate in the same collision preflight.
    for entry in _select_utility_schemas(name, server, config):
        schema = entry["schema"]
        handler_key = entry["handler_key"]
        candidates.append(
            {
                "registry_name": schema["name"],
                "origin": f"generated utility {handler_key!r}",
                "schema": schema,
                "handler": _UTILITY_HANDLER_FACTORIES[handler_key](
                    name, server.tool_timeout
                ),
                "check_fn": check_fn,
            }
        )

    unique_candidates, ambiguous_names, shadowed_utilities = _resolve_name_collisions(name, candidates)

    for candidate in unique_candidates:
        registry_name = candidate["registry_name"]
        if registry_name in ambiguous_names:
            continue
        if (registry_name, candidate["origin"]) in shadowed_utilities:
            continue

        existing_toolset = registry.get_toolset_for_tool(registry_name)
        if existing_toolset and existing_toolset != toolset_name:
            if existing_toolset.startswith("mcp-"):
                logger.error(
                    "MCP server '%s': %s normalizes to '%s', already owned by "
                    "MCP toolset '%s' — skipping to preserve the existing owner",
                    name,
                    candidate["origin"],
                    registry_name,
                    existing_toolset,
                )
            else:
                logger.warning(
                    "MCP server '%s': %s (→ '%s') collides with built-in tool "
                    "in toolset '%s' — skipping to preserve built-in",
                    name,
                    candidate["origin"],
                    registry_name,
                    existing_toolset,
                )
            continue

        registry.register(
            name=registry_name,
            toolset=toolset_name,
            schema=candidate["schema"],
            handler=candidate["handler"],
            check_fn=candidate["check_fn"],
            is_async=False,
            description=candidate["schema"]["description"],
            scope=_core._server_registry_scope(name),
        )

        # The pre-check above is advisory only. Multiple servers connect in
        # parallel, so ToolRegistry.register() is the atomic ownership gate.
        if registry.get_toolset_for_tool(registry_name) != toolset_name:
            logger.error(
                "MCP server '%s': registration of %s as '%s' was rejected by "
                "the registry; skipping provenance/count updates",
                name,
                candidate["origin"],
                registry_name,
            )
            continue

        _core._track_mcp_tool_server(registry_name, name)
        registered_names.append(registry_name)

    if registered_names:
        registry.register_toolset_alias(name, toolset_name)
        _write_schema_cache(name, server, config, _should_register)

    return registered_names


class _CachedMCPTool:
    """Minimal stand-in for MCP Tool objects loaded from the schema cache."""

    __slots__ = ("name", "description", "inputSchema")

    def __init__(self, name: str, description: str, inputSchema: dict):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema or {}


def _register_from_cache_sync(name: str, config: dict, entry: dict) -> List[str]:
    """Lazy startup: register a server's tools from a cached manifest with no
    child process. The first real call routes through
    ``_get_connected_server_for_call`` -> ``_ensure_lazy_server_connected``."""
    from tools.registry import registry
    from tools.mcp_schema_cache import (
        config_fingerprint,
        tools_from_cache_entry,
        utility_tools_from_cache_entry,
    )

    registered_names: List[str] = []
    toolset_name = f"mcp-{name}"
    fingerprint = config_fingerprint(config)
    tool_timeout = _resolve_tool_timeout(config)
    _should_register = _make_tool_filter(name, config)
    check_fn = _make_check_fn(name)
    # Record trust metadata before registration so the call-time gate is
    # identical whether the server was spawned live or registered from cache.
    # Missing "annotations" in older cache files fails closed to write-capable.
    cached_tool_objs = [
        SimpleNamespace(
            name=raw.get("name"),
            annotations=raw.get("annotations")
            if isinstance(raw.get("annotations"), dict) else None,
        )
        for raw in tools_from_cache_entry(entry)
        if isinstance(raw, dict) and raw.get("name")
    ]
    _record_tool_trust_metadata(name, config, cached_tool_objs)
    for raw in tools_from_cache_entry(entry):
        if not isinstance(raw, dict):
            continue
        raw_name = raw.get("name")
        if not raw_name or not _should_register(raw_name):
            continue
        raw_schema = raw.get("inputSchema")
        mcp_tool = _CachedMCPTool(
            raw_name,
            raw.get("description") or "",
            raw_schema if isinstance(raw_schema, dict) else {},
        )
        # Defense-in-depth: the cache file is user-writable JSON, so apply the
        # same injection scan as eager discovery.
        _core._scan_mcp_description(name, mcp_tool.name, mcp_tool.description or "")
        schema = _core._convert_mcp_schema(name, mcp_tool)
        registry_name = schema["name"]
        existing_toolset = registry.get_toolset_for_tool(registry_name)
        if existing_toolset and existing_toolset != toolset_name:
            logger.warning(
                "MCP server '%s' (lazy): cached tool '%s' collides with "
                "toolset '%s' — skipping",
                name, registry_name, existing_toolset,
            )
            continue
        registry.register(
            name=registry_name,
            toolset=toolset_name,
            schema=schema,
            handler=_core._make_tool_handler(name, raw_name, tool_timeout),
            check_fn=check_fn,
            is_async=False,
            description=schema["description"],
            scope=_core._mcp_registry_scope(),
        )
        if registry.get_toolset_for_tool(registry_name) != toolset_name:
            continue
        _core._track_mcp_tool_server(registry_name, name)
        registered_names.append(registry_name)

    for raw in utility_tools_from_cache_entry(entry):
        if not isinstance(raw, dict):
            continue
        schema = raw.get("schema")
        handler_key = raw.get("handler_key")
        if not isinstance(schema, dict) or handler_key not in _UTILITY_HANDLER_FACTORIES:
            continue
        util_name = schema.get("name") or ""
        if not util_name:
            continue
        existing_toolset = registry.get_toolset_for_tool(util_name)
        if existing_toolset and existing_toolset != toolset_name:
            continue
        registry.register(
            name=util_name,
            toolset=toolset_name,
            schema=schema,
            handler=_UTILITY_HANDLER_FACTORIES[handler_key](name, tool_timeout),
            check_fn=check_fn,
            is_async=False,
            description=schema.get("description") or "",
            scope=_core._mcp_registry_scope(),
        )
        if registry.get_toolset_for_tool(util_name) != toolset_name:
            continue
        _core._track_mcp_tool_server(util_name, name)
        registered_names.append(util_name)

    if registered_names:
        registry.register_toolset_alias(name, toolset_name)
        with _core._lock:
            _core._lazy_server_configs[name] = dict(config)
            _core._lazy_server_fingerprints[name] = fingerprint
            _core._lazy_server_tool_names[name] = list(registered_names)
        logger.info(
            "MCP server '%s' (lazy): registered %d tool(s) from schema cache",
            name, len(registered_names),
        )
    return registered_names
