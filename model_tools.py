"""
Model Tools Module

Thin orchestration layer over the tool registry: importing this module runs tool
discovery (each tools/*.py self-registers via tools.registry.register()), then
exposes get_tool_definitions() (schemas sent to the model, toolset-filtered) and
handle_function_call() (dispatch with hooks/middleware) plus registry wrappers.
"""

import os
import json
import re
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import (
    CHECK_FN_CACHE_BYPASS,
    check_fn_cache_scope,
    discover_builtin_tools,
    registry,
    tool_error,
)
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)

_post_tool_call_hook_suppressed: ContextVar[bool] = ContextVar(
    "post_tool_call_hook_suppressed", default=False
)


@contextmanager
def suppress_post_tool_call_hook():
    """Let an outer executor own the terminal post-tool event."""
    token = _post_tool_call_hook_suppressed.set(True)
    try:
        yield
    finally:
        _post_tool_call_hook_suppressed.reset(token)

# Platform-bundle names already flagged in disabled_toolsets (advisory logged once per name).
_WARNED_DISABLED_BUNDLES: set = set()


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _is_dispatcher_owned_worker() -> bool:
    """False when HERMES_KANBAN_* is present but this execution does not own it
    (delegate_task child, or a cron job fired in-process from a worker)."""
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context

        return is_dispatcher_owned_worker_context()
    except Exception:
        return True


# =============================================================================
# Async Bridging  (single source of truth -- used by registry.dispatch too)
# =============================================================================
# Loops are persistent (never asyncio.run per call): cached httpx/AsyncOpenAI
# clients stay bound to a live loop, so their GC cleanup can't hit
# "Event loop is closed". Main thread shares one loop; worker threads
# (parallel tool execution) each own a thread-local loop to avoid contention.

_tool_loop = None          # persistent loop for the main (CLI) thread
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()  # per-worker-thread persistent loops


def _get_tool_loop():
    """Long-lived event loop for async tool handlers on the main thread."""
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Persistent event loop for the current worker thread (thread-local)."""
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run a coroutine from sync code; safe under a running loop (gateway/RL env)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop: run in a fresh thread whose loop we
        # hold a reference to, so on timeout we can cancel the task inside it
        # (ThreadPoolExecutor.cancel() is a no-op on a running worker and
        # would leak the thread on every 300 s timeout).
        import concurrent.futures

        worker_loop: Optional[asyncio.AbstractEventLoop] = None
        loop_ready = threading.Event()

        def _run_in_worker():
            nonlocal worker_loop
            worker_loop = asyncio.new_event_loop()
            loop_ready.set()
            try:
                asyncio.set_event_loop(worker_loop)
                return worker_loop.run_until_complete(coro)
            finally:
                try:
                    # Drain tasks still pending after an external cancel.
                    pending = asyncio.all_tasks(worker_loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                worker_loop.close()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Carry profile + approval/sudo context so get_hermes_home() resolves correctly.
        from tools.thread_context import propagate_context_to_thread

        future = pool.submit(propagate_context_to_thread(_run_in_worker))
        try:
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            # Cancel inside the worker's own loop so the thread can wind down.
            if loop_ready.wait(timeout=1.0) and worker_loop is not None:
                try:
                    for t in asyncio.all_tasks(worker_loop):
                        worker_loop.call_soon_threadsafe(t.cancel)
                except RuntimeError:
                    pass  # loop already closed
            raise
        finally:
            # wait=False: never block the caller on a stuck coroutine.
            pool.shutdown(wait=False)

    if threading.current_thread() is not threading.main_thread():
        return _get_worker_loop().run_until_complete(coro)
    return _get_tool_loop().run_until_complete(coro)


# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

discover_builtin_tools()

# MCP discovery is deliberately NOT run here: it blocks up to 120 s and the
# gateway lazy-imports this module inside its event loop. Each entry point
# (gateway/run.py, cli.py, tui_gateway, acp_adapter) runs it at its own startup.

# Plugin tool discovery (user/project/pip plugins)
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


# =============================================================================
# Backward-compat constants  (built once after discovery)
# =============================================================================

TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# Tool names from the last get_tool_definitions() call (execute_code sandbox fallback).
_last_resolved_tool_names: List[str] = []


# =============================================================================
# Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
# =============================================================================

_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "cronjob_tools": ["cronjob_manage"],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions  (the main schema provider)
# =============================================================================

# Memo for get_tool_definitions(), active only with quiet_mode=True (the
# non-quiet path prints). Hot callers (gateway runner, AIAgent.__init__) hit it
# every turn; a miss costs ~7 ms of registry walk + check_fn probing. The key
# includes registry._generation (bumped on register/deregister/alias) so
# invalidation is transparent; check_fn drift is handled by registry.py's 30 s TTL.
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}
_tool_defs_cache_lock = threading.Lock()

# FIFO cap: a long-lived gateway sees many toolset/config fingerprints; 8
# covers the warm working set of platform/toolset combos it actually serves.
_TOOL_DEFS_CACHE_MAX = 8


def _clear_tool_defs_cache() -> None:
    """Drop memoized results when a dynamic-schema dependency changes (discord caps, sandbox mode)."""
    with _tool_defs_cache_lock:
        _tool_defs_cache.clear()


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """Tool definitions for model API calls, filtered by toolset.

    Args:
        enabled_toolsets: Only include tools from these toolsets (None = all).
        disabled_toolsets: Toolsets subtracted after enabling.
        quiet_mode: Suppress status prints (and enable memoization).
        skip_tool_search_assembly: Return the pre-assembly list (raw schemas for
            every enabled tool). Only the tool_search bridge should use this so
            it reads the real catalog rather than the collapsed one.
    """
    # Memo key covers every argument plus everything that changes the result
    # without an argument changing: registry generation, config.yaml mtime/size
    # (dynamic schemas: execute_code mode, discord allowlist), kanban context,
    # profile scope. check_fn results are TTL-cached inside registry.get_definitions.
    cache_key = None
    if quiet_mode:
        try:
            from hermes_cli.config import get_config_path
            cfg_stat = get_config_path().stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        profile_scope = check_fn_cache_scope()
        if profile_scope != CHECK_FN_CACHE_BYPASS:
            cache_key = (
                registry.current_scope_key(),
                frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
                frozenset(disabled_toolsets) if disabled_toolsets else None,
                registry._generation,
                cfg_fp,
                bool(os.environ.get("HERMES_KANBAN_TASK")),
                bool(skip_tool_search_assembly),
                _is_delegated_child_context(),
                _is_dispatcher_owned_worker(),
                profile_scope,
            )
        with _tool_defs_cache_lock:
            cached = _tool_defs_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            global _last_resolved_tool_names
            _last_resolved_tool_names = [t["function"]["name"] for t in cached]
            return list(cached)

    result = _compute_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode,
                                       skip_tool_search_assembly=skip_tool_search_assembly)
    if quiet_mode and cache_key is not None:
        with _tool_defs_cache_lock:
            # Another thread may have filled this key meanwhile; reuse it.
            cached = _tool_defs_cache.get(cache_key)
            if cached is None:
                if len(_tool_defs_cache) >= _TOOL_DEFS_CACHE_MAX:
                    _tool_defs_cache.pop(next(iter(_tool_defs_cache)))
                _tool_defs_cache[cache_key] = result
                cached = result
        return list(cached)
    # Quiet callers always get a shallow copy: run_agent appends memory/LCM
    # schemas to its list, and a shared list would accumulate duplicate tool
    # names across agent inits (rejected with HTTP 400 by DeepSeek/Kimi/MiMo).
    if quiet_mode:
        return list(result)
    return result


def _find_tool(tool_defs: List[Dict[str, Any]], name: str) -> int:
    """Index of the tool named *name* in *tool_defs*, or -1."""
    for i, td in enumerate(tool_defs):
        if td.get("function", {}).get("name") == name:
            return i
    return -1


def _drop_tool(tool_defs: List[Dict[str, Any]], available: set, name: str) -> List[Dict[str, Any]]:
    available.discard(name)
    return [td for td in tool_defs if td.get("function", {}).get("name") != name]


def _compute_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """Uncached implementation of :func:`get_tool_definitions`."""
    tools_to_include: set = set()

    if enabled_toolsets is not None:
        effective_enabled_toolsets = list(enabled_toolsets)
        # Dispatcher-spawned kanban workers always get the lifecycle handoff
        # tools, even when the assignee profile restricts its chat toolsets.
        if (
            os.environ.get("HERMES_KANBAN_TASK")
            and not _is_delegated_child_context()
            and _is_dispatcher_owned_worker()
            and "kanban" not in effective_enabled_toolsets
        ):
            effective_enabled_toolsets.append("kanban")
        for toolset_name in effective_enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # Disabled toolsets are always subtracted LAST, so a tool in a disabled
    # toolset is stripped even when a composite (hermes-cli) re-enables it.
    if disabled_toolsets:
        from toolsets import bundle_non_core_tools, get_toolset
        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                if toolset_name.startswith("hermes-") or (get_toolset(toolset_name) or {}).get("posture"):
                    # Platform bundles and posture toolsets re-list the core tools
                    # without owning them; subtracting the whole set would empty
                    # the tool list. Remove only the non-core delta.
                    to_remove = bundle_non_core_tools(toolset_name)
                    tools_to_include.difference_update(to_remove)
                    resolved = sorted(to_remove)
                    if (not quiet_mode and toolset_name.startswith("hermes-")
                            and toolset_name not in _WARNED_DISABLED_BUNDLES):
                        _WARNED_DISABLED_BUNDLES.add(toolset_name)
                        logger.info(
                            "agent.disabled_toolsets contains platform-bundle "
                            "name '%s'; core tools are preserved and only its "
                            "platform-specific tools (%s) are removed. Bundle "
                            "names usually belong in `toolsets:`, not "
                            "`disabled_toolsets` (#33924).",
                            toolset_name,
                            ", ".join(resolved) if resolved else "none",
                        )
                else:
                    resolved = resolve_toolset(toolset_name)
                    tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")

    # Registry returns only tools whose check_fn passes. Every cross-reference
    # below must use available_tool_names (not tools_to_include) so the model
    # is never told about a tool that isn't actually in the list.
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)
    available_tool_names = {t["function"]["name"] for t in filtered_tools}

    # execute_code: list only sandbox tools that are actually available.
    if "execute_code" in available_tool_names:
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema, _get_execution_mode
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
        dynamic_schema = build_execute_code_schema(sandbox_enabled, mode=_get_execution_mode())
        i = _find_tool(filtered_tools, "execute_code")
        if i != -1:
            filtered_tools[i] = {"type": "function", "function": dynamic_schema}

    # discord / discord_admin: schema depends on the bot's privileged intents
    # and the config action allowlist; a None schema drops the tool entirely.
    _discord_schema_fns = {
        "discord": "get_dynamic_schema_core",
        "discord_admin": "get_dynamic_schema_admin",
    }
    for discord_tool_name, schema_fn_name in _discord_schema_fns.items():
        if discord_tool_name in available_tool_names:
            try:
                from tools import discord_tool as _dt
                dynamic = getattr(_dt, schema_fn_name)()
            except Exception:
                dynamic = None
            if dynamic is None:
                filtered_tools = _drop_tool(filtered_tools, available_tool_names, discord_tool_name)
            else:
                i = _find_tool(filtered_tools, discord_tool_name)
                if i != -1:
                    filtered_tools[i] = {"type": "function", "function": dynamic}

    # browser_navigate: drop the "prefer web_search or web_extract" hint when
    # neither web tool is present (otherwise the model hallucinates them).
    if "browser_navigate" in available_tool_names and not ({"web_search", "web_extract"} & available_tool_names):
        i = _find_tool(filtered_tools, "browser_navigate")
        if i != -1:
            td = filtered_tools[i]
            desc = td["function"].get("description", "").replace(
                " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
                "",
            )
            filtered_tools[i] = {
                "type": "function",
                "function": {**td["function"], "description": desc},
            }

    # browser_exec runs arbitrary host Python; a session without the terminal
    # surface must not regain host execution through the browser toolset.
    # Session-level gate (not a check_fn: those are TTL-cached process-wide
    # while one gateway serves sessions with different toolsets).
    if "browser_exec" in available_tool_names and "terminal" not in available_tool_names:
        filtered_tools = _drop_tool(filtered_tools, available_tool_names, "browser_exec")

    # delegate_task's child-restrictions line names sibling tools (clarify,
    # memory, cronjob). Trim it to tools actually present, or drop the line
    # when none apply, so the model never learns ghost vocabulary. Two source
    # variants exist (depth-off also names delegate_task itself); test the
    # longer one first because the sibling list is a substring of it.
    if "delegate_task" in available_tool_names:
        blocked_present = [
            t for t in ("clarify", "memory", "cronjob_manage") if t in available_tool_names
        ]
        i = _find_tool(filtered_tools, "delegate_task")
        if len(blocked_present) < 3 and i != -1:
            td = filtered_tools[i]
            fn = td.get("function", {})
            desc = fn.get("description", "")
            full_offvariant = "delegate_task, clarify, memory, or cronjob"
            full_onvariant = "clarify, memory, or cronjob"
            if full_offvariant in desc:
                full, keep_self = full_offvariant, True
            elif full_onvariant in desc:
                full, keep_self = full_onvariant, False
            else:
                full = None
            if full is not None:
                names = (["delegate_task"] if keep_self else []) + blocked_present
                if blocked_present:
                    if len(names) == 1:
                        replacement = names[0]
                    elif len(names) == 2:
                        replacement = f"{names[0]} or {names[1]}"
                    else:
                        replacement = ", ".join(names[:-1]) + ", or " + names[-1]
                    desc = desc.replace(full, replacement)
                else:
                    # Both variants end at the following newline.
                    start = desc.find("- Children cannot call " + full)
                    if start != -1:
                        end = desc.index("\n", start) + 1
                        desc = desc[:start] + desc[end:]
                filtered_tools[i] = {
                    **td,
                    "function": {**fn, "description": desc},
                }

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    # Normalize schema shapes llama.cpp's grammar converter rejects (bare
    # "type": "object", string-valued nodes from malformed MCP servers).
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    # Tool Search (progressive disclosure): replace MCP/plugin tools with the
    # tool_search/describe/call bridge when the deferrable surface exceeds the
    # configured share of the context window. Core tools are never deferred.
    # Must be the LAST step (after sanitization); idempotent if called twice.
    try:
        from tools.tool_search import assemble_tool_defs, load_config as _load_ts_config
        ts_cfg = _load_ts_config()
        if not skip_tool_search_assembly and ts_cfg.enabled != "off":
            assembly = assemble_tool_defs(
                filtered_tools,
                context_length=_resolve_active_context_length(),
                config=ts_cfg,
            )
            if assembly.activated and not quiet_mode:
                _forms = {"full": "catalog listing embedded",
                          "names": "names-only listing embedded",
                          "mixed": "listing embedded (oversized servers summarized)",
                          "groups": "server summary embedded (search-only discovery)",
                          "none": "no listing (search-only)"}
                print(
                    f"🔎 Tool Search (tier {assembly.tier}): {assembly.deferred_count} "
                    f"MCP/plugin tools deferred (~{assembly.deferred_tokens} tokens) behind "
                    f"tool_search/describe/call — {_forms.get(assembly.listing_form, assembly.listing_form)}."
                )
            filtered_tools = assembly.tool_defs
    except Exception as e:  # pragma: no cover — never break tool loading
        logger.warning("Tool search assembly skipped: %s", e)

    return filtered_tools


def _resolve_active_context_length() -> int:
    """Active model's context length for the tool-search gate (0 if unresolvable).

    Order: explicit `model.context_length` in config.yaml; provider-aware
    resolution (Codex OAuth enforces a smaller window than the direct API for
    the same slug); the on-disk metadata cache (a slightly stale window is fine
    for picking a disclosure tier and avoids a ~200 ms /models probe per CLI
    startup); then the full live resolver.
    """
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        _raw_model_id = model_cfg.get("model") or model_cfg.get("default") or ""
        if isinstance(_raw_model_id, dict):
            from hermes_cli.config import split_model_config_default
            _raw_model_id, _ = split_model_config_default(_raw_model_id)
        model_id = str(_raw_model_id).strip()
        if not model_id:
            return 0
        from agent.model_metadata import get_model_context_length
        raw_ctx = model_cfg.get("context_length")
        config_ctx = raw_ctx if isinstance(raw_ctx, int) and raw_ctx > 0 else None
        provider = str(model_cfg.get("provider") or "").strip()
        base_url = str(model_cfg.get("base_url") or "").strip()
        api_key = ""
        if provider:
            # Credential resolution failing (offline, no keys) degrades to a
            # provider+base_url-only lookup so static fallbacks still apply.
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                rt = resolve_runtime_provider(
                    requested=provider, target_model=model_id
                ) or {}
                base_url = str(rt.get("base_url") or base_url or "").strip()
                api_key = str(rt.get("api_key") or "").strip()
            except Exception as rt_exc:
                logger.debug(
                    "Runtime credential resolution failed for tool-search "
                    "context gate (provider=%s): %s — using config values only",
                    provider, rt_exc,
                )
        if config_ctx is None and base_url:
            try:
                from agent.model_metadata import get_cached_context_length
                cached_ctx = get_cached_context_length(model_id, base_url)
                if isinstance(cached_ctx, int) and cached_ctx > 0:
                    return cached_ctx
            except Exception:
                pass
        return int(get_model_context_length(
            model_id,
            base_url=base_url,
            api_key=api_key,
            config_context_length=config_ctx,
            provider=provider,
        ) or 0)
    except Exception as e:
        logger.debug("Could not resolve active context length: %s", e)
        return 0


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Tools the agent loop (run_agent.py) intercepts because they need agent-level
# state. The registry still holds their schemas; dispatch returns a stub error.
_AGENT_LOOP_TOOLS = {"todo_list", "memory", "session_search", "delegate_task"}

# Legacy tool-name aliases (2026-08 renames), accepted at every dispatch seam so
# old sessions and saved prompts keep working; schemas advertise only new names.
_LEGACY_TOOL_ALIASES = {
    "todo": "todo_list",
    "cronjob": "cronjob_manage",
    "process": "process_manage",
    "tour": "gui_tour",
    "tip": "show_tip",
}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# =========================================================================
# Tool error sanitization
# =========================================================================
# Defense-in-depth: json.dumps already prevents framing escape, but the model
# still reads the text, so strip role tags / CDATA / code fences from exception
# messages and cap length. The cap is shared with tools/registry.py so text never
# passes two different caps with two different markers.
_TOOL_ERROR_ROLE_TAG_RE = re.compile(
    r'</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>',
    re.IGNORECASE,
)
_TOOL_ERROR_FENCE_OPEN_RE = re.compile(r'^\s*```(?:json|xml|html|markdown)?\s*', re.MULTILINE)
_TOOL_ERROR_FENCE_CLOSE_RE = re.compile(r'\s*```\s*$', re.MULTILINE)
_TOOL_ERROR_CDATA_RE = re.compile(r'<!\[CDATA\[.*?\]\]>', re.DOTALL)
from tools.registry import _MAX_TOOL_ERROR_CHARS as _TOOL_ERROR_MAX_LEN


def _sanitize_tool_error(error_msg: str) -> str:
    """Strip structural framing tokens from a tool error before the model sees it."""
    if not error_msg:
        return "[TOOL_ERROR] "
    sanitized = _TOOL_ERROR_ROLE_TAG_RE.sub("", error_msg)
    sanitized = _TOOL_ERROR_FENCE_OPEN_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_FENCE_CLOSE_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_CDATA_RE.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


# =========================================================================
# Tool argument type coercion
# =========================================================================

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


def _tool_result_observer_fields(
    tool_name: str,
    result: Any,
) -> tuple[str, Optional[str], Optional[str]]:
    """Derive (status, error_type, error_message) from a tool result for observer hooks."""
    try:
        parsed_result = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed_result, dict) and parsed_result.get("error"):
            return "error", "tool_error", str(parsed_result.get("error"))
    except Exception:
        pass
    try:
        from agent.display import _detect_tool_failure

        failed, suffix = _detect_tool_failure(tool_name, result)
        if failed:
            return "error", "tool_error", suffix.strip().strip("[]") or None
    except Exception:
        pass
    return "ok", None, None


def _emit_post_tool_call_hook(
    *,
    function_name: str,
    function_args: Dict[str, Any],
    result: Any,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    duration_ms: int = 0,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit the ``post_tool_call`` observer hook.

    Gated on has_hook so the no-listener path costs one dict lookup; when
    ``status`` is None the ok/error fields are derived from the result only
    after that gate.
    """
    if _post_tool_call_hook_suppressed.get():
        return
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not has_hook("post_tool_call"):
            return
        if status is None:
            status, error_type, error_message = _tool_result_observer_fields(
                function_name,
                result,
            )
        invoke_hook(
            "post_tool_call",
            tool_name=function_name,
            args=function_args,
            result=result,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception as _hook_err:
        logger.debug("post_tool_call hook error: %s", _hook_err)


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False,
    skip_tool_request_middleware: bool = False,
    skip_tool_execution_middleware: bool = False,
    tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    """Route a tool call through hooks/middleware to the registry; returns a JSON string.

    Args:
        task_id: Terminal/browser session isolation key.
        user_task: The user's original task (browser_snapshot context).
        enabled_tools: Session tool names; execute_code uses them to pick sandbox
            tools (falls back to the process-global ``_last_resolved_tool_names``).
        skip_pre_tool_call_hook: Caller already fired pre_tool_call (single-fire contract).
        enabled_toolsets / disabled_toolsets: The session's toolset selection,
            used to scope the Tool Search bridge catalog so tool_search /
            tool_describe / tool_call only see tools this session was granted.
            None = no restriction, matching get_tool_definitions semantics.
    """
    function_args = coerce_tool_args(function_name, function_args)
    if not isinstance(function_args, dict):
        function_args = {}
    _tool_middleware_trace = list(tool_request_middleware_trace or [])
    function_name = _LEGACY_TOOL_ALIASES.get(function_name, function_name)
    _dispatch_start = time.monotonic()

    def _emit(result: Any, **extra: Any) -> Any:
        """Emit post_tool_call with this call's identity fields; returns *result*."""
        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            middleware_trace=list(_tool_middleware_trace),
            **extra,
        )
        return result

    # Tool Search bridge: tool_search / tool_describe are catalog reads handled
    # inline; tool_call is unwrapped so every downstream hook (pre/post, edit
    # approval, guardrails) sees the real tool name, never the bridge.
    try:
        from tools import tool_search as _ts_mod
    except Exception:
        _ts_mod = None

    if _ts_mod is not None and _ts_mod.is_bridge_tool(function_name):
        # Read the un-collapsed catalog, scoped to the session's toolsets so a
        # restricted session (subagent, kanban worker) cannot see or invoke the
        # whole process registry through the bridge.
        try:
            current_defs = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True, skip_tool_search_assembly=True,
            ) or []
        except Exception:
            current_defs = []

        def _elapsed() -> int:
            return int((time.monotonic() - _dispatch_start) * 1000)

        if function_name == _ts_mod.TOOL_SEARCH_NAME:
            return _emit(_ts_mod.dispatch_tool_search(function_args or {}, current_tool_defs=current_defs),
                         duration_ms=_elapsed())
        if function_name == _ts_mod.TOOL_DESCRIBE_NAME:
            return _emit(_ts_mod.dispatch_tool_describe(function_args or {}, current_tool_defs=current_defs),
                         duration_ms=_elapsed())
        if function_name == _ts_mod.TOOL_CALL_NAME:
            underlying_name, underlying_args, err = _ts_mod.resolve_underlying_call(function_args or {})
            if err or not underlying_name:
                return _emit(tool_error(err or "tool_call could not be resolved"), duration_ms=_elapsed())
            # Defense in depth: resolve_underlying_call only checks the global
            # registry; also require membership in the session-scoped catalog.
            if underlying_name not in _ts_mod.scoped_deferrable_names(current_defs):
                return _emit(
                    tool_error(
                        f"'{underlying_name}' is not available in this session. "
                        "Use tool_search to find tools you can call."
                    ),
                    duration_ms=_elapsed(),
                )
            # Validate against the deferred tool's concrete schema — the generic
            # ``arguments: object`` bridge schema can't enforce it.
            _probe_err = _ts_mod.validate_deferred_call_args(underlying_name, underlying_args)
            if _probe_err is not None:
                return _emit(_probe_err, duration_ms=_elapsed())
            return handle_function_call(
                function_name=underlying_name,
                function_args=underlying_args,
                task_id=task_id,
                tool_call_id=tool_call_id,
                session_id=session_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                user_task=user_task,
                enabled_tools=enabled_tools,
                skip_pre_tool_call_hook=skip_pre_tool_call_hook,
                skip_tool_request_middleware=skip_tool_request_middleware,
                skip_tool_execution_middleware=skip_tool_execution_middleware,
                tool_request_middleware_trace=list(_tool_middleware_trace),
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
            )

    _tool_original_args = dict(function_args)
    if not skip_tool_request_middleware:
        try:
            from hermes_cli.middleware import apply_tool_request_middleware

            _tool_request_mw = apply_tool_request_middleware(
                function_name,
                function_args,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                turn_id=turn_id or "",
                api_request_id=api_request_id or "",
            )
            function_args = _tool_request_mw.payload
            _tool_original_args = _tool_request_mw.original_payload
            _tool_middleware_trace = _tool_request_mw.trace
        except Exception as _mw_err:
            logger.debug("tool_request middleware error: %s", _mw_err)

    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return tool_error(f"{function_name} must be handled by the agent loop")

        # pre_tool_call fires exactly once per execution: _dispatch_pre_tool_call_hooks
        # returns the block message (block/approve) and modified args (modify) from a
        # single invoke_hook pass. skip=True means the caller already fired it.
        if not skip_pre_tool_call_hook:
            block_message: Optional[str] = None
            try:
                from hermes_cli.plugins import _dispatch_pre_tool_call_hooks
                block_message, modified_args = _dispatch_pre_tool_call_hooks(
                    function_name,
                    function_args,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                    middleware_trace=list(_tool_middleware_trace),
                )
                if modified_args is not None:
                    function_args = modified_args
            except Exception as _hook_err:
                logger.debug("pre_tool_call hook error: %s", _hook_err)

            if block_message is not None:
                return _emit(tool_error(block_message), status="blocked",
                             error_type="plugin_block", error_message=block_message)

        # ACP/Zed edit approval before any file mutation. The requester is bound
        # via ContextVar only for ACP sessions, so CLI/gateway paths are unaffected.
        try:
            from acp_adapter.edit_approval import maybe_require_edit_approval

            edit_block_message = maybe_require_edit_approval(function_name, function_args)
            if edit_block_message is not None:
                return _emit(edit_block_message, status="blocked", error_type="edit_approval_denied")
        except Exception as _edit_approval_err:
            logger.debug("ACP edit approval guard error: %s", _edit_approval_err)
            if function_name in {"write_file", "patch"}:
                return _emit(tool_error("Edit approval denied: approval guard failed"),
                             status="blocked", error_type="edit_approval_error")

        # Any non-read/search tool resets the consecutive-read-loop counter.
        if function_name not in _READ_SEARCH_TOOLS:
            try:
                from tools.file_tools import notify_other_tool_call
                notify_other_tool_call(task_id or "default")
            except Exception:
                pass  # file_tools may not be loaded yet

        # duration_ms (monotonic) is exposed to post_tool_call / transform_tool_result.
        _dispatch_start = time.monotonic()
        _approval_tokens = None
        _reset_obs = None
        try:
            from tools.approval import (
                reset_current_observability_context as _reset_obs,
                set_current_observability_context,
            )
            _approval_tokens = set_current_observability_context(
                turn_id=turn_id or "",
                tool_call_id=tool_call_id or "",
                session_id=session_id or "",
            )
        except Exception:
            _reset_obs = None
        try:
            dispatch_kwargs: Dict[str, Any] = {"task_id": task_id, "session_id": session_id}
            if function_name == "execute_code":
                # Prefer the caller's list so subagents can't overwrite the
                # parent's tool set via the process-global.
                dispatch_kwargs["enabled_tools"] = (
                    enabled_tools if enabled_tools is not None else _last_resolved_tool_names
                )
            else:
                dispatch_kwargs["user_task"] = user_task

            def _dispatch(next_args: Dict[str, Any]) -> Any:
                return registry.dispatch(function_name, next_args, **dispatch_kwargs)

            if skip_tool_execution_middleware:
                result = _dispatch(function_args)
            else:
                from hermes_cli.middleware import run_tool_execution_middleware

                result = run_tool_execution_middleware(
                    function_name,
                    function_args,
                    _dispatch,
                    original_args=_tool_original_args,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                )
        finally:
            if _approval_tokens is not None and _reset_obs is not None:
                try:
                    _reset_obs(_approval_tokens)
                except Exception:
                    pass
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)

        _emit(result, duration_ms=duration_ms)

        # transform_tool_result: plugins may replace the final result string.
        # Runs after post_tool_call (observational) and before the result enters
        # context. Fail-open; first valid string return wins; non-strings ignored.
        try:
            from hermes_cli.lifecycle import has_hook, invoke_hook
            if has_hook("transform_tool_result"):
                status, error_type, error_message = _tool_result_observer_fields(
                    function_name,
                    result,
                )
                hook_results = invoke_hook(
                    "transform_tool_result",
                    tool_name=function_name,
                    args=function_args,
                    result=result,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                    duration_ms=duration_ms,
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        result = hook_result
                        break
        except Exception as _hook_err:
            logger.debug("transform_tool_result hook error: %s", _hook_err)

        return result

    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.exception(error_msg)
        return _emit(
            tool_error(_sanitize_tool_error(error_msg)),
            duration_ms=int((time.monotonic() - _dispatch_start) * 1000),
            status="error",
            error_type=type(e).__name__,
            error_message=str(e),
        )


# =============================================================================
# Backward-compat wrapper functions
# =============================================================================

def get_all_tool_names() -> List[str]:
    """Return all registered tool names."""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """Return the toolset a tool belongs to."""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Return toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """Return {toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """Return (available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
