"""Live-agent tool-list maintenance after MCP (re)discovery: refreshing an
AIAgent's tools/tool names, preserving the cached tools[] prefix across rebuilds,
and re-injecting post-build tools."""

import logging
import json
import threading
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")


# Serializes in-place swaps of ``agent.tools`` / ``agent.valid_tool_names`` by
# the reload RPC, gateway reload and late-binding refresh thread; the run loop
# reads them during tool iteration and must never see a half-updated pair.
_agent_tools_lock = threading.Lock()


def _def_name(tool_def: dict) -> str:
    return (tool_def.get("function") or {}).get("name", "")


def _agent_tool_defs(agent) -> list:
    return list(getattr(agent, "tools", None) or [])


def _resolve_refresh_toolsets(agent, enabled_override, disabled_override):
    """Explicit reloads pass freshly-resolved toolsets (so a server just ENABLED
    in config is picked up) and the agent's selection is updated to match;
    automatic paths pass nothing and reuse the build-time selection."""
    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    if enabled_override is not None or disabled_override is not None:
        enabled = enabled_override if enabled_override is not None else enabled
        disabled = disabled_override if disabled_override is not None else disabled
        agent.enabled_toolsets = enabled
        agent.disabled_toolsets = disabled
    return enabled, disabled


def _tool_defs_content_changed(agent, new_defs: list) -> bool:
    """Byte-level diff of the serialized tool arrays (dynamic schemas change
    CONTENT under stable names); False if either side fails to serialize."""
    try:
        dump = lambda defs: json.dumps(defs, sort_keys=True, separators=(",", ":"), default=str)  # noqa: E731
        return dump(_agent_tool_defs(agent)) != dump(new_defs)
    except Exception:  # noqa: BLE001
        return False


def refresh_agent_mcp_tools(
    agent,
    *,
    enabled_override=None,
    disabled_override=None,
    quiet_mode: bool = True,
    content_aware: bool = False,
    preserve_prefix: bool = False,
) -> set:
    """Re-derive an already-built agent's tool snapshot from the live registry.

    The agent snapshots ``agent.tools`` once at build time; servers that connect
    later (slow OAuth server, ``/reload-mcp``) are invisible until rebuilt. This
    is the single shared rebuild for the TUI RPC, gateway reload, late-binding
    thread and between-turns refresh. It respects the agent's own toolset
    filter, diffs by tool NAME (a count compare misses an equal-size swap), and
    is additive-preserving: memory-provider and context-engine (``lcm_*``) tools
    that ``agent_init`` appends after ``get_tool_definitions`` are re-injected,
    since a naive rebuild would silently delete them. ``(tools, valid_tool_names)``
    are published together under ``_agent_tools_lock``.

    ``preserve_prefix`` is for rebuilds inside a live conversation, where the
    tool array is a cached request prefix and any moved byte re-prefills the
    whole history: existing tools keep their slot (schemas still refresh), a
    still-registered tool whose ``check_fn`` merely flapped is carried forward,
    a tool that left the registry is dropped, and new tools append at the tail.
    Carrying an unavailable tool forward is safe: ``check_fn`` gates exposure,
    never invocation, and every handler owns its own unavailability error.

    Returns the newly-added tool names (empty when unchanged). The caller owns
    the prompt-cache contract (turn-boundary policy differs per caller).
    """
    from model_tools import get_tool_definitions
    from tools.registry import registry

    enabled, disabled = _resolve_refresh_toolsets(agent, enabled_override, disabled_override)

    # Capture the registry generation BEFORE the slow get_tool_definitions call;
    # at publish time a slower caller holding an OLDER set must not clobber a
    # newer set another caller already published.
    snapshot_generation = registry._generation

    # Computed OUTSIDE the lock (can be slow); diff + publish happen together in
    # one critical section so concurrent callers can't torn-publish.
    new_defs = list(get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=quiet_mode) or [])
    new_names = {_def_name(t) for t in new_defs}

    # Re-append the post-build families on LOCALS only; live agent attributes
    # are untouched until the single atomic publish below.
    staged_engine_names = _core._reinject_post_build_tools(agent, new_defs, new_names)

    # Registry membership is read OUTSIDE ``_agent_tools_lock``: taking
    # ``registry._lock`` under the tools lock would be the first nesting of the two.
    registered_names: set = set()
    if preserve_prefix:
        try:
            registered_names = {entry.name for entry in registry.get_all_entries()}
        except Exception:  # noqa: BLE001
            preserve_prefix = False  # fail open to the plain rebuild

    # Single atomic read-diff-publish so ``added`` matches what was published
    # and a stale (older-generation) rebuild can't overwrite a newer one.
    with _agent_tools_lock:
        # Tolerate an agent that never set the generation (or a non-int mock)
        # rather than failing the whole refresh on the comparison.
        published_gen_raw = getattr(agent, "_tool_snapshot_generation", -1)
        published_gen = published_gen_raw if isinstance(published_gen_raw, int) else -1
        if snapshot_generation < published_gen:
            return set()  # a newer snapshot already won
        current_defs = _agent_tool_defs(agent)
        current = {_def_name(t) for t in current_defs}
        if preserve_prefix:
            new_defs, new_names = _merge_preserving_prefix(current_defs, new_defs, registered_names)
        # Same NAME set: no change for MCP-reload callers. Content-aware callers
        # (compaction boundary) also diff serialized bytes.
        if new_names == current and not (content_aware and _tool_defs_content_changed(agent, new_defs)):
            # Record the generation so an in-flight older caller can't clobber.
            agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
            return set()
        agent.tools = new_defs
        agent.valid_tool_names = new_names
        # Publish context-engine routing names atomically with the snapshot.
        engine_names = getattr(agent, "_context_engine_tool_names", None)
        if isinstance(engine_names, set):
            engine_names.clear()
            engine_names.update(staged_engine_names)
        agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
        added = new_names - current
    # Re-pin the session's tool order so a rebuild-for-existing-session
    # (gateway agent-cache eviction) restores exactly these names.
    persist_agent_tool_names(agent)
    return added


def reprobe_tool_availability() -> None:
    """Explicit ``/reload-mcp`` hatch out of the tools[] freeze: drop the
    ``check_fn`` verdict cache AND the ``get_tool_definitions`` memo (keyed on
    registry generation, so it would otherwise replay the stale verdicts)."""
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def persist_agent_tool_names(agent) -> None:
    """Best-effort: write ``agent.tools`` names to the session row (freeze pin)."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not db or not session_id:
        return
    try:
        db.update_session_tool_names(session_id, [_def_name(t) for t in _agent_tool_defs(agent)])
    except Exception:  # noqa: BLE001
        logger.debug("tool_names persist skipped", exc_info=True)


def restore_agent_tool_prefix(agent, saved_names: list) -> bool:
    """Fold a freshly built agent's ``tools`` onto the session's saved order.

    The gateway rebuilds a NEW AIAgent for an existing session after agent-cache
    eviction, with no predecessor to preserve; the saved name list stands in:
    a saved tool still registered but failing its probe is carried forward from
    the registry schema, a deregistered one is dropped, new tools append at the
    tail (same rule as ``_merge_preserving_prefix``). Returns True if changed.
    """
    if not saved_names:
        return False
    from tools.registry import registry

    fresh_defs = _agent_tool_defs(agent)
    fresh = {_def_name(t): t for t in fresh_defs}
    saved_defs = []
    for name in saved_names:
        entry_def = fresh.get(name)
        if entry_def is None:
            entry = registry.get_entry(name)
            if entry is None:
                continue
            entry_def = {"type": "function", "function": {**entry.schema, "name": entry.name}}
        saved_defs.append(entry_def)
    registered_names = {entry.name for entry in registry.get_all_entries()}
    merged, merged_names = _merge_preserving_prefix(saved_defs, fresh_defs, registered_names)
    with _agent_tools_lock:
        if merged == fresh_defs:
            return False
        agent.tools = merged
        agent.valid_tool_names = merged_names
    if [_def_name(t) for t in merged] != list(saved_names):
        persist_agent_tool_names(agent)
    return True


def _merge_preserving_prefix(current_defs: list, new_defs: list, registered_names: set) -> tuple[list, set]:
    """Fold a fresh tool snapshot into a live one without moving existing bytes.

    Ordered by ``current_defs`` (the cached request prefix): a name in both
    keeps its slot but takes the fresh schema; a name only in the live list is
    kept if still registered (``check_fn`` flapped) and dropped if not; a name
    only in the fresh list is appended at the tail.
    """
    fresh = {_def_name(entry): entry for entry in new_defs if _def_name(entry)}
    merged = []
    for entry in current_defs:
        name = _def_name(entry)
        replacement = fresh.pop(name, None)
        if replacement is not None:
            merged.append(replacement)
        elif name and name in registered_names:
            merged.append(entry)
    merged.extend(fresh.values())
    return merged, {_def_name(t) for t in merged}


def _reinject_post_build_tools(agent, tools_list: list, name_set: set) -> set:
    """Append memory-provider and context-engine tools onto the caller's staged
    ``tools_list`` / ``name_set`` (never the live agent attributes), mirroring
    ``agent_init``'s post-build injection. Idempotent and fail-soft.

    Returns the context-engine routing names THIS rebuild appended: a name
    already owned by a registry/plugin tool is not claimed, matching agent_init.
    """
    def _add(schema) -> bool:
        name = schema.get("name", "") if isinstance(schema, dict) else ""
        if not name or name in name_set:
            return False
        tools_list.append({"type": "function", "function": schema})
        name_set.add(name)
        return True

    enabled = getattr(agent, "enabled_toolsets", None)
    try:
        memory_manager = getattr(agent, "_memory_manager", None)
        get_mem_schemas = getattr(memory_manager, "get_all_tool_schemas", None) if memory_manager else None
        if callable(get_mem_schemas):
            # Same toolset gate inject_memory_provider_tools uses.
            from agent.memory_manager import memory_provider_tools_enabled
            if memory_provider_tools_enabled(
                enabled, getattr(agent, "disabled_toolsets", None), memory_tool_present="memory" in name_set,
            ):
                for schema in get_mem_schemas():
                    _add(schema)
    except Exception:
        logger.debug("Memory-provider tool re-injection skipped", exc_info=True)

    # The `context_engine` toolset is intentionally empty, so lcm_* tools exist
    # only via this append. Honor the enabled_toolsets gate agent_init uses, or a
    # restricted-toolset platform would re-leak tools the build excluded.
    staged_engine_names: set = set()
    try:
        compressor = getattr(agent, "context_compressor", None)
        get_schemas = getattr(compressor, "get_tool_schemas", None) if compressor else None
        if (enabled is None or "context_engine" in enabled) and callable(get_schemas):
            for schema in get_schemas():
                # Claim the routing name only when WE appended the schema.
                if _add(schema):
                    staged_engine_names.add(schema["name"])
    except Exception:
        logger.debug("Context-engine tool re-injection skipped", exc_info=True)

    return staged_engine_names
