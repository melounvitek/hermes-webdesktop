"""Registering a connected (or schema-cached) MCP server's tools into the tool
registry: include/exclude filtering, trust-tier metadata capture, utility-tool
selection, name-collision resolution and the schema-cache write-through.

Both entry points (``_register_server_tools`` for a live server,
``_register_from_cache_sync`` for a lazy cached manifest) build ``_Candidate``
records and feed the single ``_register_candidates`` loop."""

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
    """Config ``trust`` -> tier. None -> ``full`` (backward-compatible default);
    an unrecognized string -> ``untrusted`` so a misspelled tier fails closed."""
    if value is None:
        return _core._TRUST_FULL
    text = str(value).strip().lower()
    if text in (_core._TRUST_FULL, _core._TRUST_UNTRUSTED):
        return text
    logger.warning(
        "MCP trust: unrecognized trust value %r — treating as 'untrusted' (valid values: full, untrusted)", value,
    )
    return _core._TRUST_UNTRUSTED


def _annotation_read_only_hint(mcp_tool: Any) -> bool:
    """True only when annotations (SDK object or schema-cache dict) carry
    ``readOnlyHint is True``; unknown metadata means write-capable."""
    annotations = getattr(mcp_tool, "annotations", None)
    if isinstance(annotations, dict):
        return annotations.get("readOnlyHint") is True
    return getattr(annotations, "readOnlyHint", None) is True


def _record_tool_trust_metadata(server_name: str, config: dict, tools: List[Any]) -> None:
    """Capture per-server trust and per-tool readOnlyHint at discovery — the
    security boundary: the call-time gate classifies from data we control,
    never re-read server-supplied state."""
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
    """Utility schemas allowed by config (``tools.resources``/``tools.prompts``)
    and by the server's advertised capabilities.

    ``initialize_result.capabilities`` is the source of truth: its sub-objects are
    non-None iff the server advertises that request family (the old
    ``hasattr(server.session, ...)`` gate never filtered anything — ClientSession
    defines all four methods). When no initialize_result was captured (test
    fixtures, older paths) fall back to that legacy session-method check."""
    tools_filter = config.get("tools") or {}
    enabled = {f: _parse_boolish(tools_filter.get(f), default=True) for f in ("resources", "prompts")}
    init_result = getattr(server, "initialize_result", None)
    advertised = getattr(init_result, "capabilities", None) if init_result is not None else None

    def _skip_reason(handler_key: str) -> Optional[str]:
        family = _UTILITY_CAPABILITY_ATTRS[handler_key]
        if not enabled[family]:
            return f"{family} disabled"
        if advertised is not None:
            return None if getattr(advertised, family, None) is not None else f"server does not advertise '{family}' capability"
        method = _UTILITY_CAPABILITY_METHODS[handler_key]
        return None if hasattr(server.session, method) else f"session lacks {method}"

    selected: List[dict] = []
    for entry in _build_utility_schemas(server_name):
        reason = _skip_reason(entry["handler_key"])
        if reason:
            logger.debug("MCP server '%s': skipping utility '%s' (%s)", server_name, entry["handler_key"], reason)
            continue
        selected.append(entry)
    return selected


def _existing_tool_names() -> List[str]:
    """Tool names for all currently connected servers plus lazy (cache-registered)
    servers, whose tools live only in the registry."""
    names: List[str] = []
    for _sname, server in _core._servers.items():
        if hasattr(server, "_registered_tool_names"):
            names.extend(server._registered_tool_names)
        else:
            names.extend(_core._convert_mcp_schema(server.name, t)["name"] for t in server._tools)
    with _core._lock:
        names.extend(
            n for sname, tool_names in _core._lazy_server_tool_names.items()
            if sname not in _core._servers for n in tool_names
        )
    return names


def _make_tool_filter(name: str, config: dict) -> Callable[[str], bool]:
    """Include/exclude predicate for a server's tool names: ``tools.include`` is a
    whitelist (``[]`` = register nothing), ``tools.exclude`` a blacklist; entries
    are exact names or fnmatch globs; include wins over exclude."""
    tools_filter = config.get("tools") or {}
    include_raw = tools_filter.get("include")
    include_set = _normalize_name_filter(include_raw, f"mcp_servers.{name}.tools.include")
    include_active = isinstance(include_raw, (str, list, tuple, set))
    exclude_set = _normalize_name_filter(tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude")

    def _should_register(tool_name: str) -> bool:
        if include_active:
            return matches_name_filter(tool_name, include_set)
        return not (exclude_set and matches_name_filter(tool_name, exclude_set))

    return _should_register


class _CachedMCPTool:
    """Stand-in for MCP Tool objects loaded from the schema cache. Missing or
    non-dict ``annotations`` (older cache files) fail closed to write-capable."""

    __slots__ = ("name", "description", "inputSchema", "annotations")

    def __init__(self, name: str, description: str, inputSchema: dict, annotations: Optional[dict] = None):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema or {}
        self.annotations = annotations if isinstance(annotations, dict) else None

    @classmethod
    def from_cache_dicts(cls, raws: Iterable[Any]) -> List["_CachedMCPTool"]:
        """Cached rows -> stand-ins; rows that are not dicts or lack a name are dropped."""
        out = []
        for raw in raws:
            if isinstance(raw, dict) and raw.get("name"):
                schema = raw.get("inputSchema")
                out.append(cls(raw["name"], raw.get("description") or "", schema if isinstance(schema, dict) else {}, raw.get("annotations")))
        return out


@dataclass
class _Candidate:
    """One registration attempt: a native tool or a generated utility.
    ``origin`` is the provenance text used in collision diagnostics."""

    registry_name: str
    origin: str
    schema: dict
    handler: Callable

    @property
    def is_utility(self) -> bool:
        return self.origin.startswith(_UTILITY_ORIGIN_PREFIX)


def _tool_candidates(name: str, tools: Iterable[Any], should_register: Callable[[str], bool], tool_timeout) -> List[_Candidate]:
    """Native tools (live SDK objects or ``_CachedMCPTool``) -> candidates. The
    injection scan runs on BOTH paths: the cache file is user-writable JSON."""
    out: List[_Candidate] = []
    for t in tools:
        if not should_register(t.name):
            logger.debug("MCP server '%s': skipping tool '%s' (filtered by config)", name, t.name)
            continue
        _core._scan_mcp_description(name, t.name, t.description or "")
        schema = _core._convert_mcp_schema(name, t)
        out.append(_Candidate(schema["name"], f"tool {t.name!r}", schema, _core._make_tool_handler(name, t.name, tool_timeout)))
    return out


def _utility_candidates(name: str, entries: Iterable[Any], tool_timeout) -> List[_Candidate]:
    """``{schema, handler_key}`` rows (live selection or cache) -> candidates; malformed rows dropped."""
    out: List[_Candidate] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        schema, key = raw.get("schema"), raw.get("handler_key")
        if isinstance(schema, dict) and key in _UTILITY_HANDLER_FACTORIES and schema.get("name"):
            out.append(_Candidate(schema["name"], f"{_UTILITY_ORIGIN_PREFIX}{key!r}", schema, _UTILITY_HANDLER_FACTORIES[key](name, tool_timeout)))
    return out


def _resolve_name_collisions(name: str, candidates: List[_Candidate]) -> List[_Candidate]:
    """Preflight registry-name collisions among one server's candidates.

    Exact duplicates (same name + origin) are dropped silently; a generated
    utility that normalizes onto a server-native tool's name is shadowed (the
    native tool wins); any other multi-origin collision is ambiguous and every
    colliding entry is skipped (fail closed). Returns the survivors in order."""
    unique: List[_Candidate] = []
    seen: set[tuple[str, str]] = set()
    origins_by_name: Dict[str, set[str]] = {}
    for c in candidates:
        if (c.registry_name, c.origin) in seen:
            logger.debug("MCP server '%s': duplicate registration candidate %s for '%s'; keeping one", name, c.origin, c.registry_name)
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
                "MCP server '%s': generated utility %s normalizes onto server-native %s — keeping the "
                "native tool and dropping the utility (the utility only applies when the server has no "
                "such tool of its own)",
                name, ", ".join(utility_origins), native_origins[0],
            )
            continue
        ambiguous[registry_name] = sorted(origins)
    for registry_name, origins in sorted(ambiguous.items()):
        logger.error(
            "MCP server '%s': name normalization collision for '%s' from %s; skipping every colliding "
            "entry instead of choosing an arbitrary handler",
            name, registry_name, ", ".join(origins),
        )
    return [c for c in unique if c.registry_name not in ambiguous and (c.registry_name, c.origin) not in shadowed]


def _log_foreign_owner(name: str, c: _Candidate, existing_toolset: str, lazy: bool) -> None:
    """Diagnostics for a name already owned by another toolset (skipped to preserve the owner)."""
    if lazy:
        if not c.is_utility:
            logger.warning("MCP server '%s' (lazy): cached tool '%s' collides with toolset '%s' — skipping", name, c.registry_name, existing_toolset)
        return
    log, fmt = (
        (logger.error, "MCP server '%s': %s normalizes to '%s', already owned by MCP toolset '%s' — skipping to preserve the existing owner")
        if existing_toolset.startswith("mcp-") else
        (logger.warning, "MCP server '%s': %s (→ '%s') collides with built-in tool in toolset '%s' — skipping to preserve built-in")
    )
    log(fmt, name, c.origin, c.registry_name, existing_toolset)


def _register_candidates(name: str, candidates: List[_Candidate], *, check_fn: Callable, scope: Callable[[], Optional[str]], lazy: bool) -> List[str]:
    """Register candidates under toolset ``mcp-{name}``; returns the names that
    landed. The ownership pre-check is advisory only — servers connect in
    parallel, so ``ToolRegistry.register()`` is the atomic ownership gate and
    its verdict is re-read after every call."""
    from tools.registry import registry

    toolset_name = f"mcp-{name}"
    registered: List[str] = []
    for c in candidates:
        existing_toolset = registry.get_toolset_for_tool(c.registry_name)
        if existing_toolset and existing_toolset != toolset_name:
            _log_foreign_owner(name, c, existing_toolset, lazy)
            continue
        registry.register(
            name=c.registry_name, toolset=toolset_name, schema=c.schema, handler=c.handler, check_fn=check_fn,
            is_async=False, description=c.schema.get("description") or "", scope=scope(),
        )
        if registry.get_toolset_for_tool(c.registry_name) != toolset_name:
            if not lazy:
                logger.error(
                    "MCP server '%s': registration of %s as '%s' was rejected by the registry; skipping provenance/count updates",
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
        for t in server._tools:
            if should_register(t.name):
                schema_obj = getattr(t, "inputSchema", None)
                tools_payload.append({
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": schema_obj if isinstance(schema_obj, dict) else {},
                    # Persisted so the lazy path trust-gates identically next startup.
                    "annotations": {"readOnlyHint": _annotation_read_only_hint(t)},
                })
        utility_payload = [{"schema": e["schema"], "handler_key": e["handler_key"]} for e in _select_utility_schemas(name, server, config)]
        cache_meta = getattr(server, "_list_cache_meta", None) or {}
        write_cache_entry(
            name, config_fingerprint(config), tools=tools_payload, utility_tools=utility_payload,
            ttl_ms=cache_meta.get("ttl_ms"), cache_scope=cache_meta.get("cache_scope"),
        )
    except Exception as exc:
        logger.debug("MCP schema cache write failed for '%s': %s", name, exc)


def _register_server_tools(name: str, server: "MCPServerTask", config: dict) -> List[str]:
    """Register an already-connected server's tools (plus utility tools); used by
    initial discovery and list_changed refresh. Returns the registered names.

    Toolset resolution for ``mcp-{server}`` / raw-name aliases derives from the
    live registry rather than mutating ``toolsets.TOOLSETS``. Lossy name
    normalization can map distinct raw names (``read-file``/``read_file``) to
    one registry name; such collisions fail closed. Generated utilities share
    the namespace and join the same preflight."""
    should_register = _make_tool_filter(name, config)
    _record_tool_trust_metadata(name, config, server._tools)
    candidates = _tool_candidates(name, server._tools, should_register, server.tool_timeout)
    candidates += _utility_candidates(name, _select_utility_schemas(name, server, config), server.tool_timeout)
    registered = _register_candidates(
        name, _resolve_name_collisions(name, candidates),
        check_fn=_make_check_fn(name), scope=lambda: _core._server_registry_scope(name), lazy=False,
    )
    if registered:
        _write_schema_cache(name, server, config, should_register)
    return registered


def _register_from_cache_sync(name: str, config: dict, entry: dict) -> List[str]:
    """Lazy startup: register a server's tools from a cached manifest with no
    child process; the first real call goes through
    ``_get_connected_server_for_call`` -> ``_ensure_lazy_server_connected``.
    Trust metadata is recorded first so the call-time gate is identical whether
    the server was spawned live or registered from cache."""
    from tools.mcp_schema_cache import config_fingerprint, tools_from_cache_entry, utility_tools_from_cache_entry

    tool_timeout = _resolve_tool_timeout(config)
    cached_tools = _CachedMCPTool.from_cache_dicts(tools_from_cache_entry(entry))
    _record_tool_trust_metadata(name, config, cached_tools)
    candidates = _tool_candidates(name, cached_tools, _make_tool_filter(name, config), tool_timeout)
    candidates += _utility_candidates(name, utility_tools_from_cache_entry(entry), tool_timeout)
    registered = _register_candidates(name, candidates, check_fn=_make_check_fn(name), scope=_core._mcp_registry_scope, lazy=True)
    if registered:
        with _core._lock:
            _core._lazy_server_configs[name] = dict(config)
            _core._lazy_server_fingerprints[name] = config_fingerprint(config)
            _core._lazy_server_tool_names[name] = list(registered)
        logger.info("MCP server '%s' (lazy): registered %d tool(s) from schema cache", name, len(registered))
    return registered
