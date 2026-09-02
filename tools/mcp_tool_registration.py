"""Registering a connected (or schema-cached) MCP server's tools into the tool
registry: include/exclude filtering, trust-tier metadata capture, utility-tool
selection, name-collision resolution and the schema-cache write-through.

Both entry points (``_register_server_tools`` for a live server,
``_register_from_cache_sync`` for a lazy cached manifest) build a list of
``_Candidate`` records and hand them to the single ``_register_candidates`` loop."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional
from tools.mcp_tool_common import _parse_boolish, _core, _resolve_tool_timeout
from tools.mcp_tool_handlers import _make_check_fn, _make_get_prompt_handler, _make_list_prompts_handler, _make_list_resources_handler, _make_read_resource_handler
from tools.mcp_tool_schema import _UTILITY_CAPABILITY_ATTRS, _UTILITY_CAPABILITY_METHODS, _build_utility_schemas, _normalize_name_filter, matches_name_filter

if TYPE_CHECKING:  # pragma: no cover
    from tools.mcp_tool import MCPServerTask

logger = logging.getLogger("tools.mcp_tool")

_UTILITY_ORIGIN_PREFIX = "generated utility "

# Utility tool key -> handler factory; each takes (server_name, tool_timeout).
_UTILITY_HANDLER_FACTORIES = {
    "list_resources": _make_list_resources_handler,
    "read_resource": _make_read_resource_handler,
    "list_prompts": _make_list_prompts_handler,
    "get_prompt": _make_get_prompt_handler,
}


def _normalize_server_trust(value: Any) -> str:
    """Normalize a config ``trust`` value: None -> ``full`` (backward-compatible
    default); any unrecognized string -> ``untrusted``, so a misspelled tier
    fails closed rather than silently disabling gating."""
    if value is None:
        return _core._TRUST_FULL
    text = str(value).strip().lower()
    if text in (_core._TRUST_FULL, _core._TRUST_UNTRUSTED):
        return text
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
        return annotations.get("readOnlyHint") is True
    return getattr(annotations, "readOnlyHint", None) is True


def _record_tool_trust_metadata(server_name: str, config: dict, tools: List[Any]) -> None:
    """Capture per-server trust and per-tool readOnlyHint at discovery (the
    security boundary: the call-time gate classifies from data we control,
    never re-read server-supplied state)."""
    with _core._lock:
        _core._server_trust_levels[server_name] = _normalize_server_trust((config or {}).get("trust"))
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
    family_enabled = {
        family: _parse_boolish(tools_filter.get(family), default=True)
        for family in ("resources", "prompts")
    }
    # ``initialize_result.capabilities`` is the source of truth: its sub-objects
    # are non-None iff the server advertises that request family. The old
    # ``hasattr(server.session, ...)`` gate never filtered anything because
    # ClientSession defines all four methods on the class.
    init_result = getattr(server, "initialize_result", None)
    advertised_caps = getattr(init_result, "capabilities", None) if init_result is not None else None

    selected: List[dict] = []
    for entry in _build_utility_schemas(server_name):
        handler_key = entry["handler_key"]
        family = _UTILITY_CAPABILITY_ATTRS[handler_key]
        if not family_enabled[family]:
            logger.debug("MCP server '%s': skipping utility '%s' (%s disabled)", server_name, handler_key, family)
            continue
        if advertised_caps is not None:
            if getattr(advertised_caps, family, None) is None:
                logger.debug(
                    "MCP server '%s': skipping utility '%s' (server does not advertise '%s' capability)",
                    server_name, handler_key, family,
                )
                continue
        # Legacy fallback when initialize_result wasn't captured (test
        # fixtures, older paths): register every stub the session can serve.
        elif not hasattr(server.session, _UTILITY_CAPABILITY_METHODS[handler_key]):
            logger.debug(
                "MCP server '%s': skipping utility '%s' (session lacks %s)",
                server_name, handler_key, _UTILITY_CAPABILITY_METHODS[handler_key],
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
            names.append(_core._convert_mcp_schema(server.name, mcp_tool)["name"])
    # Lazy servers registered from the schema cache have no MCPServerTask yet —
    # their tools live only in the registry.
    with _core._lock:
        names.extend(
            n
            for sname, tool_names in _core._lazy_server_tool_names.items()
            if sname not in _core._servers
            for n in tool_names
        )
    return names


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
    exclude_set = _normalize_name_filter(tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude")

    def _should_register(tool_name: str) -> bool:
        if include_active:
            return matches_name_filter(tool_name, include_set)
        if exclude_set:
            return not matches_name_filter(tool_name, exclude_set)
        return True

    return _should_register


class _CachedMCPTool:
    """Minimal stand-in for MCP Tool objects loaded from the schema cache.
    Missing/non-dict ``annotations`` (older cache files) fail closed to
    write-capable via ``_annotation_read_only_hint``."""

    __slots__ = ("name", "description", "inputSchema", "annotations")

    def __init__(self, name: str, description: str, inputSchema: dict, annotations: Optional[dict] = None):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema or {}
        self.annotations = annotations if isinstance(annotations, dict) else None

    @classmethod
    def from_cache_dicts(cls, raws: Iterable[Any]) -> List["_CachedMCPTool"]:
        """Cached tool rows -> stand-ins; rows that are not dicts or lack a name are dropped."""
        tools = []
        for raw in raws:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            schema = raw.get("inputSchema")
            tools.append(cls(
                raw["name"],
                raw.get("description") or "",
                schema if isinstance(schema, dict) else {},
                raw.get("annotations"),
            ))
        return tools


@dataclass
class _Candidate:
    """One registry registration attempt for a server: a native tool or a
    generated utility. ``origin`` is the human-readable provenance used in
    collision diagnostics."""

    registry_name: str
    origin: str
    schema: dict
    handler: Callable

    @property
    def is_utility(self) -> bool:
        return self.origin.startswith(_UTILITY_ORIGIN_PREFIX)

    @property
    def description(self) -> str:
        return self.schema.get("description") or ""


def _tool_candidates(name: str, tools: Iterable[Any], should_register: Callable[[str], bool], tool_timeout) -> List[_Candidate]:
    """Native tools (live SDK objects or ``_CachedMCPTool``) -> candidates.
    The description scan runs on BOTH paths: the cache file is user-writable JSON."""
    candidates: List[_Candidate] = []
    for mcp_tool in tools:
        if not should_register(mcp_tool.name):
            logger.debug("MCP server '%s': skipping tool '%s' (filtered by config)", name, mcp_tool.name)
            continue
        _core._scan_mcp_description(name, mcp_tool.name, mcp_tool.description or "")
        schema = _core._convert_mcp_schema(name, mcp_tool)
        candidates.append(_Candidate(
            schema["name"], f"tool {mcp_tool.name!r}", schema,
            _core._make_tool_handler(name, mcp_tool.name, tool_timeout),
        ))
    return candidates


def _utility_candidates(name: str, entries: Iterable[Any], tool_timeout) -> List[_Candidate]:
    """``{schema, handler_key}`` rows (live selection or cache) -> candidates;
    malformed rows are dropped."""
    candidates: List[_Candidate] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        schema, handler_key = raw.get("schema"), raw.get("handler_key")
        if not isinstance(schema, dict) or handler_key not in _UTILITY_HANDLER_FACTORIES or not schema.get("name"):
            continue
        candidates.append(_Candidate(
            schema["name"], f"{_UTILITY_ORIGIN_PREFIX}{handler_key!r}", schema,
            _UTILITY_HANDLER_FACTORIES[handler_key](name, tool_timeout),
        ))
    return candidates


def _resolve_name_collisions(name: str, candidates: List[_Candidate]) -> List[_Candidate]:
    """Preflight registry-name collisions among one server's candidates.

    Exact duplicate rows (same name + origin) are dropped silently; a generated
    utility that normalizes onto a server-native tool's name is shadowed (the
    native tool wins); any other multi-origin collision is ambiguous and every
    colliding entry is skipped (fail closed). Returns the survivors in order.
    """
    unique: List[_Candidate] = []
    seen: set[tuple[str, str]] = set()
    origins_by_name: Dict[str, set[str]] = {}
    for c in candidates:
        if (c.registry_name, c.origin) in seen:
            logger.debug(
                "MCP server '%s': duplicate registration candidate %s for '%s'; keeping one",
                name, c.origin, c.registry_name,
            )
            continue
        seen.add((c.registry_name, c.origin))
        unique.append(c)
        origins_by_name.setdefault(c.registry_name, set()).add(c.origin)

    ambiguous: Dict[str, List[str]] = {}
    shadowed: set[tuple[str, str]] = set()
    for registry_name, origins in origins_by_name.items():
        if len(origins) <= 1:
            continue
        utility_origins = sorted(o for o in origins if o.startswith(_UTILITY_ORIGIN_PREFIX))
        native_origins = sorted(origins - set(utility_origins))
        if len(native_origins) == 1 and utility_origins:
            shadowed.update((registry_name, o) for o in utility_origins)
            logger.info(
                "MCP server '%s': generated utility %s normalizes onto "
                "server-native %s — keeping the native tool and dropping the "
                "utility (the utility only applies when the server has no such "
                "tool of its own)",
                name, ", ".join(utility_origins), native_origins[0],
            )
            continue
        ambiguous[registry_name] = sorted(origins)

    for registry_name, origins in sorted(ambiguous.items()):
        logger.error(
            "MCP server '%s': name normalization collision for '%s' from %s; "
            "skipping every colliding entry instead of choosing an arbitrary handler",
            name, registry_name, ", ".join(origins),
        )
    return [
        c for c in unique
        if c.registry_name not in ambiguous and (c.registry_name, c.origin) not in shadowed
    ]


def _log_foreign_owner(name: str, c: _Candidate, existing_toolset: str, lazy: bool) -> None:
    """Diagnostics for a candidate whose registry name is already owned by
    another toolset (skipped to preserve the existing owner)."""
    if lazy:
        if not c.is_utility:
            logger.warning(
                "MCP server '%s' (lazy): cached tool '%s' collides with toolset '%s' — skipping",
                name, c.registry_name, existing_toolset,
            )
    elif existing_toolset.startswith("mcp-"):
        logger.error(
            "MCP server '%s': %s normalizes to '%s', already owned by MCP toolset '%s' "
            "— skipping to preserve the existing owner",
            name, c.origin, c.registry_name, existing_toolset,
        )
    else:
        logger.warning(
            "MCP server '%s': %s (→ '%s') collides with built-in tool in toolset '%s' "
            "— skipping to preserve built-in",
            name, c.origin, c.registry_name, existing_toolset,
        )


def _register_candidates(
    name: str, candidates: List[_Candidate], *, check_fn: Callable, scope: Callable[[], Optional[str]], lazy: bool,
) -> List[str]:
    """Register candidates under toolset ``mcp-{name}``; returns the names that
    actually landed. The ownership pre-check is advisory only — multiple
    servers connect in parallel, so ``ToolRegistry.register()`` is the atomic
    ownership gate and its verdict is re-read after every call."""
    from tools.registry import registry

    toolset_name = f"mcp-{name}"
    registered: List[str] = []
    for c in candidates:
        existing_toolset = registry.get_toolset_for_tool(c.registry_name)
        if existing_toolset and existing_toolset != toolset_name:
            _log_foreign_owner(name, c, existing_toolset, lazy)
            continue
        registry.register(
            name=c.registry_name,
            toolset=toolset_name,
            schema=c.schema,
            handler=c.handler,
            check_fn=check_fn,
            is_async=False,
            description=c.description,
            scope=scope(),
        )
        if registry.get_toolset_for_tool(c.registry_name) != toolset_name:
            if not lazy:
                logger.error(
                    "MCP server '%s': registration of %s as '%s' was rejected by "
                    "the registry; skipping provenance/count updates",
                    name, c.origin, c.registry_name,
                )
            continue
        _core._track_mcp_tool_server(c.registry_name, name)
        registered.append(c.registry_name)

    if registered:
        registry.register_toolset_alias(name, toolset_name)
    return registered


def _write_schema_cache(name: str, server: "MCPServerTask", config: dict, should_register) -> None:
    """Write-through: persist the manifest so the next startup can register this
    server lazily without spawning it. Never raises."""
    try:
        from tools.mcp_schema_cache import config_fingerprint, write_cache_entry

        tools_payload = []
        for mcp_tool in server._tools:
            if not should_register(mcp_tool.name):
                continue
            schema_obj = getattr(mcp_tool, "inputSchema", None)
            tools_payload.append({
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "inputSchema": schema_obj if isinstance(schema_obj, dict) else {},
                # Persisted so the lazy path trust-gates identically next startup.
                "annotations": {"readOnlyHint": _annotation_read_only_hint(mcp_tool)},
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
    skipped. Generated utilities share the namespace and join the same
    preflight. Returns the registered prefixed names.
    """
    should_register = _make_tool_filter(name, config)
    check_fn = _make_check_fn(name)
    _record_tool_trust_metadata(name, config, server._tools)
    candidates = _tool_candidates(name, server._tools, should_register, server.tool_timeout)
    candidates += _utility_candidates(name, _select_utility_schemas(name, server, config), server.tool_timeout)
    registered = _register_candidates(
        name, _resolve_name_collisions(name, candidates),
        check_fn=check_fn, scope=lambda: _core._server_registry_scope(name), lazy=False,
    )
    if registered:
        _write_schema_cache(name, server, config, should_register)
    return registered


def _register_from_cache_sync(name: str, config: dict, entry: dict) -> List[str]:
    """Lazy startup: register a server's tools from a cached manifest with no
    child process. The first real call routes through
    ``_get_connected_server_for_call`` -> ``_ensure_lazy_server_connected``.
    Trust metadata is recorded first so the call-time gate is identical whether
    the server was spawned live or registered from cache."""
    from tools.mcp_schema_cache import config_fingerprint, tools_from_cache_entry, utility_tools_from_cache_entry

    tool_timeout = _resolve_tool_timeout(config)
    check_fn = _make_check_fn(name)
    cached_tools = _CachedMCPTool.from_cache_dicts(tools_from_cache_entry(entry))
    _record_tool_trust_metadata(name, config, cached_tools)
    candidates = _tool_candidates(name, cached_tools, _make_tool_filter(name, config), tool_timeout)
    candidates += _utility_candidates(name, utility_tools_from_cache_entry(entry), tool_timeout)
    registered = _register_candidates(
        name, candidates, check_fn=check_fn, scope=_core._mcp_registry_scope, lazy=True,
    )
    if registered:
        with _core._lock:
            _core._lazy_server_configs[name] = dict(config)
            _core._lazy_server_fingerprints[name] = config_fingerprint(config)
            _core._lazy_server_tool_names[name] = list(registered)
        logger.info("MCP server '%s' (lazy): registered %d tool(s) from schema cache", name, len(registered))
    return registered
