#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with a fresh conversation, their own task_id
(terminal session, file-ops cache), the parent's toolsets minus child-blocked
tools, and a focused system prompt built from goal + context. Single-task and
batch (parallel) modes; top-level model calls run in the background while
orchestrator children wait for their own workers. The parent only ever sees
the delegation call and the summary result, never the child's intermediate
tool calls or reasoning.
"""

import json
import logging
import re
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from toolsets import TOOLSETS

from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb  # noqa: F401  (used via _await_child)
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# The delegate_tool_* siblings hold the pieces split out of this module; every
# moved name is re-imported so ``tools.delegate_tool.<name>`` keeps resolving for
# callers and patching tests. Mutable flag globals live only in their owning module.
from tools.delegate_tool_child_run import (  # noqa: F401
    _WorktreeReporter,
    _append_missed_steer,
    _append_sibling_write_reminder,
    _attach_child,
    _await_child,
    _build_result_entry,
    _cleanup_child_run,
    _dump_subagent_timeout_diagnostic,
    _emit_child_complete,
    _fabricated_entry,
    _lease_child_credential,
    _make_text_relay,
    _merge_late_steer,
    _register_child,
    _seed_child_workspace,
    _start_heartbeat,
    _validate_child_output_schema,
)
from tools.delegate_tool_config import (  # noqa: F401
    _DEFAULT_MAX_CONCURRENT_CHILDREN,
    _get_child_timeout,
    _get_inherit_mcp_toolsets,
    _get_max_async_children,
    _get_max_concurrent_children,
    _get_max_spawn_depth,
    _get_orchestrator_enabled,
    _get_subagent_approval_callback,
    _get_worktree_isolation,
    _inherit_parent_base_url,
    _inherit_parent_capabilities,
    _load_config,
    _merge_request_overrides,
    _resolve_child_credential_pool,
    _require_pinned_command,
    _resolve_delegation_credentials,
    _subagent_auto_approve,
    _subagent_auto_deny,
)
from tools.delegate_tool_dispatch import (  # noqa: F401
    _dispatch_background,
    _run_children_parallel,
)
from tools.delegate_tool_progress import (  # noqa: F401
    DelegateEvent,
    SUBAGENT_FAILURE_STATUSES,
    _batch_prefix,
    _build_child_progress_callback,
    _build_child_system_prompt,
    _clean_error_text,
    _emit_parent_console,
    _resolve_workspace_hint,
    _safe_progress,
    format_batch_tag,
    format_subagent_failure_line,
)
from tools.delegate_tool_registry import (  # noqa: F401
    _CONTROL_ACTIONS,
    _active_subagents,
    _active_subagents_lock,
    _capture_gateway_steer_authority,
    _close_subagent_steering,
    _handle_control_action,
    _is_descendant_of,
    _owns_subagent_record,
    _register_subagent,
    _unregister_subagent,
    get_subagent_attribution,
    interrupt_subagent,
    is_spawn_paused,
    list_active_subagents,
    set_spawn_paused,
    steer_subagent,
)
from tools.delegate_tool_results import (  # noqa: F401
    _apply_summary_budget,
    _build_child_preserving_parent_tools,
    _finalize_child_results,
    _run_child_lifecycle,
    _summarize_tool_arguments,
)


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "cronjob_manage",  # no scheduling more work in the parent's name
    ]
)


# Nested delegation is granted by depth/role in _build_child_agent, never by the
# model naming toolsets (there is no model-facing toolsets argument).
def _normalize_role(r: Optional[str]) -> str:
    """'leaf' | 'orchestrator'; None/empty/unknown -> 'leaf' (unknown warns)."""
    if r is None or not r:
        return "leaf"
    r_norm = str(r).strip().lower()
    if r_norm in {"leaf", "orchestrator"}:
        return r_norm
    logger.warning("Unknown delegate_task role=%r, coercing to 'leaf'", r)
    return "leaf"


def _is_mcp_toolset_name(name: str) -> bool:
    """Return True for canonical MCP toolsets and their registered aliases."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    try:
        from tools.registry import registry

        target = registry.get_toolset_alias_target(str(name))
    except Exception:
        target = None
    return bool(target and str(target).startswith("mcp-"))


def _expand_parent_toolsets(parent_toolsets: set) -> set:
    """Add every toolset whose tools are a subset of the parent's tools.

    A parent on a composite like ``hermes-cli`` must still let a child request
    ``web``/``terminal``; bare name intersection would reject them.
    """
    parent_tool_names: set = set()
    for ts_name in parent_toolsets:
        ts_def = TOOLSETS.get(ts_name)
        if ts_def:
            parent_tool_names.update(ts_def.get("tools", []))

    if not parent_tool_names:
        return set(parent_toolsets)

    expanded = set(parent_toolsets)
    for ts_name, ts_def in TOOLSETS.items():
        if ts_name in expanded:
            continue
        ts_tools = ts_def.get("tools", [])
        if ts_tools and set(ts_tools).issubset(parent_tool_names):
            expanded.add(ts_name)
    return expanded


def _preserve_parent_mcp_toolsets(child_toolsets: List[str], parent_toolsets: set[str]) -> List[str]:
    """Append any parent MCP toolsets that are missing from a narrowed child."""
    preserved = list(child_toolsets)
    for toolset_name in sorted(parent_toolsets):
        if _is_mcp_toolset_name(toolset_name) and toolset_name not in preserved:
            preserved.append(toolset_name)
    return preserved


DEFAULT_MAX_ITERATIONS = 250
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
# Stale-heartbeat thresholds (cycles of _HEARTBEAT_INTERVAL with no progress).
# Progress = iteration, current_tool OR last_activity_ts advancing; an in-flight
# model wait refreshes last_activity_ts, so slow models are not "idle". Idle
# stays tight so a truly wedged child doesn't mask the gateway timeout; in-tool
# is much higher so legitimately long tools can finish.
_HEARTBEAT_STALE_CYCLES_IDLE = 15  # 450s idle between turns → stale
_HEARTBEAT_STALE_CYCLES_IN_TOOL = 40  # 1200s stuck on same tool → stale
DEFAULT_TOOLSETS = ["terminal", "file", "web"]


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets whose tools are ALL blocked (derived from DELEGATE_BLOCKED_TOOLS
    so the two can't drift) plus composite toolsets children must never get."""
    _COMPOSITE_BLOCKED_TOOLSETS = frozenset({"delegation"})
    blocked_toolset_names = {
        name
        for name, defn in TOOLSETS.items()
        if name in _COMPOSITE_BLOCKED_TOOLSETS
        or all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
    }
    blocked_toolset_names.add("kanban")
    return [t for t in toolsets if t not in blocked_toolset_names]


def _blocked_toolsets_for_role(role: str) -> List[str]:
    """One-tool deny toolsets for the role; passed as ``disabled_toolsets`` so
    blocked names inside mixed bundles are subtracted AFTER composite expansion."""
    blocked_names = set(DELEGATE_BLOCKED_TOOLS)
    if role == "orchestrator":
        blocked_names.discard("delegate_task")
    return sorted(
        name
        for name, defn in TOOLSETS.items()
        if defn.get("tools")
        and set(defn.get("tools", ())).issubset(blocked_names)
    )


def _resolve_child_toolsets(
    parent_agent, toolsets: Optional[List[str]], effective_role: str
) -> tuple[List[str], List[str]]:
    """Return ``(enabled_toolsets, disabled_toolsets)`` for a child.

    Children never gain tools the parent lacks: explicit ``toolsets`` are
    intersected with the parent's (composite-expanded) set, else the parent's
    enabled set is inherited. Blocked tools are stripped twice — whole blocked
    toolsets here, and exact one-tool deny toolsets via ``disabled_toolsets`` so
    blocked names inside mixed bundles (hermes-cli) are subtracted AFTER
    composite expansion and survive registry refreshes. Orchestrators get
    ``delegation`` re-added unconditionally (role-granted, not inherited).
    """
    # enabled_toolsets=None means "all tools", so derive from loaded tool names.
    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        import model_tools

        parent_toolsets = {
            ts
            for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    if toolsets:
        expanded_parent = _expand_parent_toolsets(parent_toolsets)
        child_toolsets = [t for t in toolsets if t in expanded_parent]
        if _get_inherit_mcp_toolsets():
            child_toolsets = _preserve_parent_mcp_toolsets(child_toolsets, parent_toolsets)
        child_toolsets = _strip_blocked_tools(child_toolsets)
    elif parent_agent and parent_enabled is not None:
        child_toolsets = _strip_blocked_tools(parent_enabled)
    elif parent_toolsets:
        child_toolsets = _strip_blocked_tools(sorted(parent_toolsets))
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    raw_parent_disabled = getattr(parent_agent, "disabled_toolsets", None)
    inherited_disabled = (
        [str(name) for name in raw_parent_disabled] if isinstance(raw_parent_disabled, (list, tuple, set)) else []
    )
    if effective_role == "orchestrator":
        inherited_disabled = [name for name in inherited_disabled if name != "delegation"]
        if "delegation" not in child_toolsets:
            child_toolsets.append("delegation")
    child_disabled_toolsets = list(
        dict.fromkeys(inherited_disabled + _blocked_toolsets_for_role(effective_role) + ["kanban"])
    )
    return child_toolsets, child_disabled_toolsets


# OpenRouter routing filters: inherited from the parent, but reset to these
# defaults under a pinned provider — parent filters (e.g. only=["Anthropic"])
# would silently force the child back onto the parent's provider.
# openrouter_min_coding_score stays inherited: model-gated, no-op elsewhere.
_ROUTING_FILTER_DEFAULTS = (
    ("providers_allowed", None),
    ("providers_ignored", None),
    ("providers_order", None),
    ("provider_sort", None),
    ("provider_require_parameters", False),
    ("provider_data_collection", ""),
)
_NOUS_PROVIDERS = frozenset({"nous", "nous-portal", "nousresearch"})


def _resolve_child_runtime(
    parent_agent,
    delegation_cfg: dict,
    parent_api_key: Any,
    *,
    model: Optional[str],
    override_provider: Optional[str],
    override_base_url: Optional[str],
    override_api_key: Optional[str],
    override_api_mode: Optional[str],
    override_max_tokens: Optional[int],
    override_acp_command: Optional[str],
    override_acp_args: Optional[List[str]],
) -> Dict[str, Any]:
    """Resolve the child's credentials, transport and routing (config override >
    parent inherit) as ``AIAgent`` keyword arguments.

    Rules that are easy to break: api_mode is re-derived (not inherited) when
    the child's provider differs from the parent's or is Nous Portal (dual-wire);
    a pinned ``delegation.command`` must exist on PATH or the spawn fails loudly;
    ``override_provider`` clears the parent's ACP transport, fallback chain and
    OpenRouter routing filters so the pinned provider is actually honoured.
    """
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or _inherit_parent_base_url(parent_agent, parent_agent.base_url)
    # api_mode: each provider has its own wire, so a different provider re-derives
    # (None) instead of inheriting (404s otherwise). Nous Portal is dual-wire
    # within one provider (anthropic/* → Messages, else chat_completions), so
    # same-provider inheritance would pin the child on the wrong wire — re-derive.
    _parent_provider = getattr(parent_agent, "provider", None) or ""
    if override_api_mode is not None:
        effective_api_mode = override_api_mode
    elif (effective_provider or "").strip().lower() in _NOUS_PROVIDERS:
        from hermes_cli.providers import nous_api_mode

        effective_api_mode = nous_api_mode(effective_model)
    elif effective_provider != _parent_provider:
        effective_api_mode = None  # force re-derivation from provider's defaults
    else:
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    # A pinned transport that cannot run must fail the spawn loudly, never fall
    # back silently (delegate_task pre-validates; this covers direct callers).
    _require_pinned_command(
        override_acp_command,
        f"Pinned delegation command '{override_acp_command}' was not "
        f"found on PATH. Install it or remove delegation.command from "
        f"config.yaml.",
    )
    effective_acp_command = override_acp_command or getattr(parent_agent, "acp_command", None)
    effective_acp_args = list(
        override_acp_args if override_acp_args is not None else (getattr(parent_agent, "acp_args", []) or [])
    )
    # A pinned provider must use direct API calls; inheriting the parent's ACP
    # transport would bypass the override credentials entirely.
    if override_provider and not override_acp_command:
        effective_acp_command, effective_acp_args = None, []
    if override_acp_command:
        # Forced ACP transport requires provider copilot-acp for run_agent to init the client.
        effective_provider, effective_api_mode = "copilot-acp", "chat_completions"

    # Reasoning: delegation.reasoning_effort > parent. Keep the raw value — a
    # YAML ``false`` must disable thinking, not coerce to "" and inherit.
    child_reasoning = getattr(parent_agent, "reasoning_config", None)
    try:
        delegation_effort = delegation_cfg.get("reasoning_effort")
        if delegation_effort or delegation_effort is False:
            from hermes_constants import parse_reasoning_effort

            parsed = parse_reasoning_effort(delegation_effort)
            if parsed is not None:
                child_reasoning = parsed
            else:
                logger.warning("Unknown delegation.reasoning_effort '%s', inheriting parent level", delegation_effort)
    except Exception as exc:
        logger.debug("Could not load delegation reasoning_effort: %s", exc)

    kwargs: Dict[str, Any] = {
        "base_url": effective_base_url,
        "api_key": override_api_key or parent_api_key,
        "model": effective_model,
        "provider": effective_provider,
        "capabilities": _inherit_parent_capabilities(parent_agent, override_provider, override_base_url),
        "api_mode": effective_api_mode,
        "acp_command": effective_acp_command,
        "acp_args": effective_acp_args,
        "reasoning_config": child_reasoning,
        # Inherit the parent's fallback chain EXCEPT under a pinned provider: a
        # mid-run 429/auth failure must not silently reroute the quiet child onto
        # the parent's fallbacks. Predictability > liveness for explicit pins.
        "fallback_model": None if override_provider else (getattr(parent_agent, "_fallback_chain", None) or None),
        "openrouter_min_coding_score": getattr(parent_agent, "openrouter_min_coding_score", None),
    }
    for attr, pinned_default in _ROUTING_FILTER_DEFAULTS:
        kwargs[attr] = pinned_default if override_provider else getattr(parent_agent, attr, pinned_default)
    if not override_provider:
        kwargs["provider_data_collection"] = kwargs["provider_data_collection"] or ""
    child_max_tokens = (
        override_max_tokens if override_max_tokens is not None else getattr(parent_agent, "max_tokens", None)
    )
    if isinstance(child_max_tokens, int):
        kwargs["max_tokens"] = child_max_tokens
    return kwargs


def _open_child_session_db(parent_agent) -> Any:
    """DEDICATED SessionDB handle for the child, or None.

    The parent's handle can be closed by its own lifecycle while a background
    child still flushes (transcript silently dropped). It MUST open the same db
    FILE as the parent's handle (non-launch profiles), else lineage /
    session_search break; released by the child's close() via _owns_session_db.
    """
    parent_session_db = getattr(parent_agent, "_session_db", None)
    if parent_session_db is None:
        return None
    try:
        from hermes_state import get_shared_session_db

        _parent_db_path = getattr(parent_session_db, "db_path", None)
        return get_shared_session_db(_parent_db_path) if _parent_db_path is not None else get_shared_session_db()
    except Exception:
        logger.debug("subagent: failed to open dedicated SessionDB; child persistence disabled", exc_info=True)
        return None


def _construct_child_agent(
    rt: Dict[str, Any],
    *,
    task_index: int,
    max_iterations: int,
    parent_agent,
    child_toolsets: List[str],
    child_disabled_toolsets: List[str],
    child_prompt: str,
    child_progress_cb: Any,
    child_session_db: Any,
    override_provider: Optional[str],
    override_request_overrides: Optional[Dict[str, Any]],
):
    """Instantiate the child AIAgent; releases the dedicated SessionDB handle on
    a construction failure (no child close() will ever run)."""
    from run_agent import AIAgent
    from agent.delegation_context import delegated_child_context

    child_thinking_cb = None
    if child_progress_cb:

        def _child_thinking(text: str) -> None:
            if text:
                _safe_progress(child_progress_cb, "_thinking", text)

        child_thinking_cb = _child_thinking

    with delegated_child_context():
        try:
            return AIAgent(
                **rt,
                max_iterations=max_iterations,
                prefill_messages=getattr(parent_agent, "prefill_messages", None),
                enabled_toolsets=child_toolsets,
                disabled_toolsets=child_disabled_toolsets,
                quiet_mode=True,
                ephemeral_system_prompt=child_prompt,
                log_prefix=f"[subagent-{task_index}]",
                platform="subagent",
                skip_context_files=True,
                skip_memory=True,
                clarify_callback=None,
                thinking_callback=child_thinking_cb,
                session_db=child_session_db,
                parent_session_id=getattr(parent_agent, "session_id", None),
                request_overrides=(
                    # honored whenever set, incl. the inherit branch where
                    # _resolve_delegation_credentials already merged OVER the parent's
                    dict(override_request_overrides)
                    if override_request_overrides is not None
                    else ({} if override_provider else dict(getattr(parent_agent, "request_overrides", {}) or {}))
                ),
                tool_progress_callback=child_progress_cb,
                iteration_budget=None,  # fresh budget per subagent
            )
        except BaseException:
            if child_session_db is not None:
                try:
                    from hermes_state import release_or_close
                    release_or_close(child_session_db)
                except Exception:
                    pass
            raise


def _announce_child_spawn(child, parent_agent, child_progress_cb, *, goal, subagent_id, parent_subagent_id, role) -> None:
    """spawn_requested event (now — the child may queue for seconds when the
    pool is saturated) plus the subagent_start lifecycle hook."""
    _safe_progress(child_progress_cb, "subagent.spawn_requested", preview=goal)
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "subagent_start",
            parent_session_id=getattr(parent_agent, "session_id", None),
            parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
            parent_subagent_id=parent_subagent_id,
            child_session_id=getattr(child, "session_id", None),
            child_subagent_id=subagent_id,
            child_role=role,
            child_goal=goal,
        )
    except Exception:
        logger.debug("subagent_start hook invocation failed", exc_info=True)


def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    task_count: int,
    parent_agent,
    # Credential overrides from delegation config
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    override_request_overrides: Optional[Dict[str, Any]] = None,
    override_max_tokens: Optional[int] = None,
    # ACP transport overrides from trusted delegation config.
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    # Legacy; accepted for wire compat but ignored (capability is depth-derived).
    role: str = "leaf",
):
    """Build (don't run) a child AIAgent on the main thread.

    override_* (from delegation config) replace parent inheritance so children
    can run on a different provider:model pair.
    """
    import uuid as _uuid

    # Role is depth-derived: a child may delegate iff the kill switch is on and
    # depth budget remains below max_spawn_depth. The `role` arg is ignored.
    child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
    max_spawn = _get_max_spawn_depth()
    effective_role = "orchestrator" if _get_orchestrator_enabled() and child_depth < max_spawn else "leaf"

    # One subagent_id shared by the progress callback, spawn_requested event and
    # the live registry; parent_id is set when THIS parent is itself a subagent.
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)

    delegation_cfg = _load_config()
    child_toolsets, child_disabled_toolsets = _resolve_child_toolsets(parent_agent, toolsets, effective_role)
    child_prompt = _build_child_system_prompt(
        goal,
        context,
        workspace_path=_resolve_workspace_hint(parent_agent),
        role=effective_role,
        max_spawn_depth=max_spawn,
        child_depth=child_depth,
    )
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    child_session_ref: Dict[str, Any] = {}
    child_progress_cb = _build_child_progress_callback(
        task_index,
        goal,
        parent_agent,
        task_count,
        subagent_id=subagent_id,
        parent_id=parent_subagent_id,
        depth=max(0, child_depth - 1),  # 0 = first-level child for the UI
        model=model or getattr(parent_agent, "model", None),
        toolsets=child_toolsets,
        session_ref=child_session_ref,
    )
    rt = _resolve_child_runtime(
        parent_agent,
        delegation_cfg,
        parent_api_key,
        model=model,
        override_provider=override_provider,
        override_base_url=override_base_url,
        override_api_key=override_api_key,
        override_api_mode=override_api_mode,
        override_max_tokens=override_max_tokens,
        override_acp_command=override_acp_command,
        override_acp_args=override_acp_args,
    )
    child_session_db = _open_child_session_db(parent_agent)
    child = _construct_child_agent(
        rt,
        task_index=task_index,
        max_iterations=max_iterations,
        parent_agent=parent_agent,
        child_toolsets=child_toolsets,
        child_disabled_toolsets=child_disabled_toolsets,
        child_prompt=child_prompt,
        child_progress_cb=child_progress_cb,
        child_session_db=child_session_db,
        override_provider=override_provider,
        override_request_overrides=override_request_overrides,
    )
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    if child_session_db is not None:
        child._owns_session_db = True  # released by the child's close(), never by the parent
    # Shared ref: session_id now, delegation_id once delegate_task stamps it —
    # both ride on every relayed event (first emit is spawn_requested below).
    child_session_ref["session_id"] = getattr(child, "session_id", "") or ""
    child._progress_identity_ref = child_session_ref
    child._delegate_depth = child_depth
    child._delegate_role = effective_role  # post-degrade role
    child._subagent_id = subagent_id
    child._parent_subagent_id = parent_subagent_id
    # Ownership chain for action=list/steer/stop; weakref so a finished parent
    # can be collected while a detached child record lingers in the registry.
    try:
        child._delegate_parent_ref = weakref.ref(parent_agent)
    except TypeError:
        child._delegate_parent_ref = None  # non-weakref-able test doubles
    # Sidebar marker: subagent sessions stay out of session pickers even when a
    # parent delete orphans them (mirrors /branch's ``_branched_from``).
    parent_sid = getattr(parent_agent, "session_id", None)
    if parent_sid and getattr(child, "_session_init_model_config", None) is not None:
        child._session_init_model_config["_delegate_from"] = parent_sid

    # Shared pool lets children rotate credentials on rate limits.
    child_pool = _resolve_child_credential_pool(rt["provider"], parent_agent, rt["base_url"])
    if child_pool is not None:
        child._credential_pool = child_pool

    _attach_child(parent_agent, child)  # interrupt propagation
    _announce_child_spawn(
        child, parent_agent, child_progress_cb,
        goal=goal, subagent_id=subagent_id, parent_subagent_id=parent_subagent_id, role=effective_role,
    )
    return child


def _run_single_child(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    *,
    owner_session_id: Optional[str] = None,
    owner_transport: Any = None,
    owner_session_record: Any = None,
    **_kwargs,
) -> Dict[str, Any]:
    """Run a pre-built child agent (called from a worker thread) and return its result entry.

    Contract, derived from the child's structured completion fields:
      status      ∈ {completed, interrupted, failed} — a structured failure
                    (failed=True / non-empty error) or an invalid terminal state
                    is "failed" even when a summary exists.
      exit_reason ∈ {completed, max_iterations, interrupted, error} —
                    "max_iterations" only for genuine budget exhaustion
                    (completed=False with no failure fields), never for errors.
      truncated   == (exit_reason == "max_iterations").
    """
    child_start = time.monotonic()
    # Set when a timed-out Future still owns the child: closing it from this
    # thread before the worker settles races the conversation's finally path.
    _child_close_deferred = False
    child_progress_cb = getattr(child, "tool_progress_callback", None)
    child_pool, leased_cred_id = _lease_child_credential(child)

    # Heartbeat keeps the parent's _last_activity_ts moving so the gateway
    # inactivity timeout doesn't fire while the child works; it stops itself
    # once the child looks stale (see _HEARTBEAT_STALE_CYCLES_*).
    heartbeat = _start_heartbeat(child, parent_agent, task_index)
    # TUI/RPC registry entry (kill/pause/status by subagent_id); None for test
    # doubles without a stable id. Unregistered in the finally block.
    _subagent_id = _register_child(
        child,
        parent_agent,
        goal,
        owner_session_id=owner_session_id,
        owner_transport=owner_transport,
        owner_session_record=owner_session_record,
    )
    worktree = _WorktreeReporter()

    try:
        heartbeat[1].start()
        _safe_progress(child_progress_cb, "subagent.start", preview=goal)

        ws = _seed_child_workspace(child, parent_agent, goal, task_index, _subagent_id, worktree)
        goal = ws.goal
        _relay_child_text = _make_text_relay(child_progress_cb)
        result, failure = _await_child(
            child,
            goal,
            ws,
            _relay_child_text,
            task_index=task_index,
            subagent_id=_subagent_id,
            child_start=child_start,
            child_progress_cb=child_progress_cb,
            worktree=worktree,
        )
        if failure is not None:
            _child_close_deferred = failure.close_deferred
            return failure.entry

        schema = _validate_child_output_schema(child, result, task_index, ws.child_task_id, _relay_child_text)
        _merge_late_steer(result, _subagent_id, child)

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, "_flush"):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - child_start, 2)
        entry = _build_result_entry(child, result, task_index, duration, schema)
        _append_sibling_write_reminder(entry, ws)
        _emit_child_complete(child, result, entry, ws, duration, child_progress_cb)
        worktree.attach(entry)
        return entry

    except Exception as exc:
        _late_pending_steer = (_close_subagent_steering(_subagent_id, child) if _subagent_id else None)
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        _safe_progress(
            child_progress_cb,
            "subagent.complete",
            preview=str(exc),
            status="failed",
            duration_seconds=duration,
            summary=str(exc),
        )
        _error_entry = _fabricated_entry(task_index, "error", str(exc), child, duration)
        _append_missed_steer(_error_entry, _late_pending_steer)
        worktree.attach(_error_entry)  # no-op when isolation never engaged
        return _error_entry

    finally:
        _cleanup_child_run(
            child,
            parent_agent,
            subagent_id=_subagent_id,
            heartbeat=heartbeat,
            child_pool=child_pool,
            leased_cred_id=leased_cred_id,
            close_deferred=_child_close_deferred,
        )


def _recover_tasks_from_json_string(tasks: Any) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not isinstance(tasks, str):
        return None, None
    raw = tasks.strip()
    if not raw:
        return None, "Provide either 'goal' (single task) or 'tasks' (batch)."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, (
            "tasks must be a JSON array of task objects; received a string "
            f"that could not be parsed as JSON ({exc.msg})."
        )
    if not isinstance(parsed, list):
        return None, (f"tasks must be a JSON array of task objects; parsed " f"{type(parsed).__name__} instead.")
    return parsed, None


# Placeholder shapes for batch goal validation: bare 'TODO' / 'task N' labels,
# or unexpanded template markers. The marker regex is deliberately NARROW —
# only snake_case / space-separated placeholder identifiers (`<feature_name>`,
# `{file path}`, `<FEATURE-NAME>`), the shape LLM templates leave behind. Bare
# single-word brackets must never be rejected: legitimate goals are full of
# generics (`Vec<T>`), HTML tags (`<div>`), dict snippets (`{"key": 1}`), glob
# braces (`{a,b}`) and f-string style (`{i}`).
_PLACEHOLDER_GOAL_RE = re.compile(r"^(todo|task\s*\d+)$", re.IGNORECASE)
_TEMPLATE_MARKER_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+>"
    r"|\{[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+\}"
)
_MIN_BATCH_GOAL_LEN = 10


def _validate_batch_tasks(task_list: List[Dict[str, Any]]) -> Optional[str]:
    """Validate a tasks=[...] batch beyond per-task goal presence; actionable
    error string or None.

    No minimum count: a one-entry array is the canonical single-task shape
    (legacy top-level `goal` is wrapped into one). Duplicate goals are
    deliberately NOT rejected — identical-goal fan-outs (best-of-N / ensemble
    sampling) are legitimate and blocking them broke real workflows.
    """
    for i, task in enumerate(task_list):
        goal = str(task.get("goal", "")).strip()
        normalized = " ".join(goal.lower().split())

        if _PLACEHOLDER_GOAL_RE.match(normalized):
            return (
                f"Task {i} has a placeholder goal ({goal!r}). Replace it "
                "with a specific, self-contained description of what the "
                "subagent should accomplish."
            )
        marker = _TEMPLATE_MARKER_RE.search(goal)
        if marker:
            return (
                f"Task {i} goal contains an unexpanded template marker "
                f"({marker.group(0)!r}). Substitute the real value before "
                "calling delegate_task — subagents cannot resolve "
                "placeholders."
            )
        if len(goal) < _MIN_BATCH_GOAL_LEN and len(task_list) >= 2:
            # Multi-task fan-outs with terse goals are usually unexpanded
            # templates; a SINGLE task legitimately uses short goals
            # ("Fix the tests"), so one-entry arrays keep the historical
            # single-`goal` exemption.
            return (
                f"Task {i} goal is too short ({goal!r}). Write a specific, "
                "self-contained goal of at least "
                f"{_MIN_BATCH_GOAL_LEN} characters so the subagent knows "
                "exactly what to do."
            )
    return None


@dataclass
class _Batch:
    """One delegate_task call's built children plus everything needed to run
    them and assemble the combined result (shared by the sync path and the
    background runner)."""

    task_list: List[Dict[str, Any]]
    children: List[tuple]
    parent_agent: Any
    max_children: int
    live_deleg_id: Optional[str]
    live_writers: list
    live_paths: list
    origin_ui_session_id: str
    origin_owner_transport: Any
    origin_owner_session_record: Any
    overall_start: float


def _normalize_task_list(
    goal, context, tasks, output_schema, top_role: str, max_children: int
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """``(task_list, None)`` from ``tasks=[...]`` or the legacy single ``goal``, else ``(None, error)``."""
    recovered_tasks, tasks_error = _recover_tasks_from_json_string(tasks)
    if tasks_error:
        return None, tasks_error
    if recovered_tasks is not None:
        tasks = recovered_tasks
    # Small models emit tasks=[] alongside a single goal: treat as "no batch".
    if isinstance(tasks, list) and not tasks:
        tasks = None

    if tasks and isinstance(tasks, list):
        if len(tasks) > max_children:
            return None, (
                f"Too many tasks: {len(tasks)} provided, but "
                f"max_concurrent_children is {max_children}. "
                f"Either reduce the task count, split into multiple "
                f"delegate_task calls, or increase "
                f"delegation.max_concurrent_children in config.yaml."
            )
        task_list = tasks
    elif goal and isinstance(goal, str) and goal.strip():
        single_task: Dict[str, Any] = {"goal": goal, "context": context, "role": top_role}
        if output_schema is not None:
            single_task["output_schema"] = output_schema
        task_list = [single_task]
    else:
        return None, (
            "No tasks provided. Pass tasks=[{goal: '...', context: '...'}, "
            "...] — one entry per subagent (a single task is a one-entry "
            "array)."
        )

    for i, task in enumerate(task_list):
        if not isinstance(task, dict):
            return None, f"Task {i} must be an object, got {type(task).__name__}."
        if not task.get("goal", "").strip():
            return None, f"Task {i} is missing a 'goal'."

    # Batch-only quality gate (placeholders, template markers); the single-goal
    # form is exempt because short goals are valid there.
    if tasks is not None and isinstance(tasks, list):
        batch_error = _validate_batch_tasks(task_list)
        if batch_error:
            return None, batch_error
    return task_list, None


def _coerce_task_schemas(
    task_list: List[Dict[str, Any]], output_schema: Optional[Dict[str, Any]]
) -> tuple[List[Optional[Dict[str, Any]]], Optional[str]]:
    """Per-task coerced output schemas. A malformed output_schema fails the whole
    call before any child spawns; schema-less tasks resolve to None and take no
    new code paths downstream."""
    from tools.delegation_output_schema import coerce_output_schema

    task_schemas: List[Optional[Dict[str, Any]]] = []
    for i, task in enumerate(task_list):
        raw_schema = task.get("output_schema")
        if raw_schema is None and len(task_list) == 1 and output_schema is not None:
            raw_schema = output_schema
        coerced_schema, schema_err = coerce_output_schema(raw_schema)
        if schema_err:
            return [], f"Task {i} output_schema invalid: {schema_err}"
        task_schemas.append(coerced_schema)
    return task_schemas, None


def _announce_batch(parent_agent, n_tasks: int, live_deleg_id: Optional[str]) -> None:
    """Announce the batch tag once so interleaved ``[tag n/N]`` lines are attributable."""
    if n_tasks <= 1 or not live_deleg_id:
        return
    _hdr = f"  🔀 [{format_batch_tag(live_deleg_id)}] delegating {n_tasks} tasks"
    _hdr_spinner = getattr(parent_agent, "_delegate_spinner", None)
    if _hdr_spinner:
        try:
            _hdr_spinner.print_above(_hdr)
            return
        except Exception:
            pass
    _emit_parent_console(parent_agent, _hdr)


def _capture_origin() -> tuple[str, str, Any, Any]:
    """``(wake_sid, ui_session_id, owner_transport, owner_session_record)`` of the
    ORIGINATING session, captured BEFORE building any child: AIAgent construction
    clobbers the HERMES_SESSION_ID ContextVar/os.environ with the subagent's id."""
    from tools.async_delegation import _current_origin_session_id

    _origin_wake_sid = _current_origin_session_id()
    try:
        from gateway.session_context import get_session_env

        _origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
    except Exception:
        _origin_ui_session_id = ""
    transport, record = _capture_gateway_steer_authority(_origin_ui_session_id)
    return _origin_wake_sid, _origin_ui_session_id, transport, record


def _build_children(
    task_list: List[Dict[str, Any]],
    task_schemas: List[Optional[Dict[str, Any]]],
    creds: Dict[str, Any],
    *,
    top_role: str,
    max_iterations: int,
    parent_agent,
    live_deleg_id: Optional[str],
    live_writers: list,
) -> tuple[List[tuple], Optional[str]]:
    """Build every child on the main thread (construction is not thread-safe);
    ``(children, None)`` or ``([], error)`` on an explicit-pin preflight failure."""
    from tools.delegation_live_log import wrap_progress_callback

    children = []
    for i, t in enumerate(task_list):
        effective_role = _normalize_role(t.get("role") or top_role)
        _task_schema = task_schemas[i] if i < len(task_schemas) else None
        _child_context = t.get("context")
        if _task_schema is not None:
            from tools.delegation_output_schema import append_output_contract

            _child_context = append_output_contract(_child_context, _task_schema)
        try:
            child = _build_child_preserving_parent_tools(
                task_index=i,
                goal=t["goal"],
                context=_child_context,
                toolsets=None,  # always inherit the parent's toolsets
                model=creds["model"],
                max_iterations=max_iterations,
                task_count=len(task_list),
                parent_agent=parent_agent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_request_overrides=creds.get("request_overrides"),
                override_max_tokens=creds.get("max_output_tokens"),
                override_acp_command=creds.get("command"),
                override_acp_args=creds.get("args"),
                role=effective_role,
            )
        except ValueError as exc:
            return [], str(exc)
        if _task_schema is not None:
            try:
                child._delegate_output_schema = _task_schema
            except Exception:
                logger.debug("Could not attach output schema to child %d", i)
        # Tee progress events into the live transcript (wrapper keeps the
        # _flush contract and swallows writer failures).
        _writer = live_writers[i] if i < len(live_writers) else None
        if _writer is not None:
            child.tool_progress_callback = wrap_progress_callback(
                getattr(child, "tool_progress_callback", None), _writer
            )
            child._live_transcript_path = str(_writer.path)
        if live_deleg_id:
            setattr(child, "_delegation_id", live_deleg_id)
            _ident_ref = getattr(child, "_progress_identity_ref", None)
            if isinstance(_ident_ref, dict):
                _ident_ref["delegation_id"] = live_deleg_id
        children.append((i, t, child))
    return children, None


def _finalize_live_transcripts(results: list, live_writers: list, live_paths: list) -> None:
    """Close out live transcripts (files are retained as the full-fidelity
    record; retention pruning happens on future dispatches)."""
    for entry in results:
        _idx = entry.get("task_index", -1)
        _w = live_writers[_idx] if isinstance(_idx, int) and 0 <= _idx < len(live_writers) else None
        if _w is not None:
            try:
                _w.finalize(entry)
            except Exception:
                logger.debug("Live transcript finalize failed", exc_info=True)
            if _idx < len(live_paths):
                entry["live_transcript"] = live_paths[_idx]


def _execute_and_aggregate(batch: _Batch, *, honor_parent_interrupt: bool = True) -> dict:
    """Run all built children, join, finalize (hooks + cost rollup), return the combined dict.

    Shared by the sync path and the background runner: even in the background
    the batch JOINS on itself here so ONE consolidated results block re-enters
    the conversation.
    """
    from tools.delegation_live_log import update_manifest_statuses

    results: list = []
    n_tasks = len(batch.task_list)
    if n_tasks == 1:
        _i, _t, child = batch.children[0]
        results.append(_run_single_child(
            _i,
            _t["goal"],
            child,
            batch.parent_agent,
            owner_session_id=batch.origin_ui_session_id or None,
            owner_transport=batch.origin_owner_transport,
            owner_session_record=batch.origin_owner_session_record,
        ))
    else:
        _run_children_parallel(
            batch.children,
            results,
            parent_agent=batch.parent_agent,
            n_tasks=n_tasks,
            max_children=batch.max_children,
            task_labels=[t["goal"][:40] for t in batch.task_list],
            live_deleg_id=batch.live_deleg_id,
            honor_parent_interrupt=honor_parent_interrupt,
            origin_ui_session_id=batch.origin_ui_session_id,
            origin_owner_transport=batch.origin_owner_transport,
            origin_owner_session_record=batch.origin_owner_session_record,
        )

    _finalize_child_results(results, batch.task_list, batch.children, batch.parent_agent)
    total_duration = round(time.monotonic() - batch.overall_start, 2)
    _finalize_live_transcripts(results, batch.live_writers, batch.live_paths)
    update_manifest_statuses(batch.live_deleg_id, results)

    combined: Dict[str, Any] = {"results": results, "total_duration_seconds": total_duration}
    if batch.live_paths:
        combined["live_transcripts"] = list(batch.live_paths)
    return combined


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    role: Optional[str] = None,
    background: Optional[bool] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    action: Optional[str] = None,
    subagent_id: Optional[str] = None,
    message: Optional[str] = None,
    parent_agent=None,
    credentials_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Spawn child agents (single ``goal`` or ``tasks=[...]`` batch) or control running ones.

    ``action`` list/steer/stop run synchronously and bypass the pause gate,
    depth limit and async dispatch. ``role`` is legacy (per-task beats
    top-level; capability is depth-derived). Returns JSON with one results
    entry per task, or a dispatch handle when running in the background.
    """
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    normalized_action = (action or "").strip().lower()
    if normalized_action in _CONTROL_ACTIONS:
        return _handle_control_action(normalized_action, subagent_id, message, parent_agent)
    if normalized_action and normalized_action != "spawn":
        return tool_error(f"Unknown action '{action}'. Use spawn (default), list, steer, or stop.")

    # Operator kill switch (TUI / delegation.pause RPC): blocks NEW spawns only.
    if is_spawn_paused():
        return tool_error(
            "Delegation spawning is paused. Clear the pause via the TUI "
            "(`p` in /agents) or the `delegation.pause` RPC before retrying."
        )

    top_role = _normalize_role(role)
    # background applies to single tasks AND batches: a batch is ONE async unit
    # that joins on every child and re-enters as a single consolidated message.
    background = is_truthy_value(background, default=False) if background is not None else False

    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return tool_error(
            f"Delegation depth limit reached (depth={depth}, "
            f"max_spawn_depth={max_spawn}). Raise "
            f"delegation.max_spawn_depth in config.yaml if deeper "
            f"nesting is required (no hard ceiling, but each level "
            f"multiplies API cost)."
        )

    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    # Caller-supplied max_iterations is ignored: the config value is authoritative
    # so budgets stay predictable (kwarg kept for internal callers/tests).
    if max_iterations is not None and max_iterations != default_max_iter:
        logger.debug(
            "delegate_task: ignoring caller-supplied max_iterations=%s; "
            "using delegation.max_iterations=%s from config",
            max_iterations, default_max_iter,
        )

    # credentials_cfg (internal callers only, e.g. /review → auxiliary.review) is
    # a per-call override shaped like the delegation config section.
    try:
        creds = _resolve_delegation_credentials(credentials_cfg if credentials_cfg else cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    max_children = _get_max_concurrent_children()
    task_list, err = _normalize_task_list(goal, context, tasks, output_schema, top_role, max_children)
    if err:
        return tool_error(err)
    task_schemas, err = _coerce_task_schemas(task_list, output_schema)
    if err:
        return tool_error(err)

    overall_start = time.monotonic()
    # Live transcripts: cache/delegation/live/<id>/task-<n>.log per task, a
    # side channel with zero effect on message content or prompt caching.
    # Best-effort: on failure live_paths is empty and delegation proceeds.
    from tools.delegation_live_log import create_live_transcripts

    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        task_list, context, model=creds.get("model"), provider=creds.get("provider")
    )
    _announce_batch(parent_agent, len(task_list), live_deleg_id)
    _origin_wake_sid, _origin_ui_session_id, _origin_owner_transport, _origin_owner_session_record = _capture_origin()

    children, err = _build_children(
        task_list,
        task_schemas,
        creds,
        top_role=top_role,
        max_iterations=default_max_iter,
        parent_agent=parent_agent,
        live_deleg_id=live_deleg_id,
        live_writers=live_writers,
    )
    if err:
        return tool_error(err)
    batch = _Batch(
        task_list, children, parent_agent, max_children, live_deleg_id, live_writers, live_paths,
        _origin_ui_session_id, _origin_owner_transport, _origin_owner_session_record, overall_start,
    )

    def _run(*, honor_parent_interrupt: bool = True) -> dict:
        return _execute_and_aggregate(batch, honor_parent_interrupt=honor_parent_interrupt)

    if background:
        return _dispatch_background(
            parent_agent=parent_agent,
            context=context,
            task_list=task_list,
            children=children,
            creds=creds,
            top_role=top_role,
            live_deleg_id=live_deleg_id,
            live_paths=live_paths,
            origin_wake_sid=_origin_wake_sid,
            origin_ui_session_id=_origin_ui_session_id,
            execute_and_aggregate=_run,
        )
    return json.dumps(_run(), ensure_ascii=False)


# ── OpenAI function-calling schema ──────────────────────────────────────────


def _build_top_level_description() -> str:
    """delegate_task description: ONLY guidance stated nowhere else in the schema
    (limits live in the 'tasks' parameter description, rebuilt per get_definitions())."""
    try:
        orchestration_available = _get_max_spawn_depth() >= 2 and _get_orchestrator_enabled()
    except Exception:
        orchestration_available = False

    # Mention recursion only where it's actually available. send_message is
    # deliberately not named (gateway-internal vocabulary); model_tools
    # session-filters the list to tools the session has.
    if orchestration_available:
        restrictions_rule = (
            "- Children cannot call clarify, memory, or cronjob.\n"
            "- Children can themselves delegate while depth remains "
            f"(max_spawn_depth={_get_max_spawn_depth()}); the runtime "
            "derives this from depth automatically.\n"
        )
    else:
        restrictions_rule = ("- Children cannot call delegate_task, clarify, memory, or " "cronjob.\n")

    return (
        "Spawn subagents in isolated contexts; each gets its own conversation, "
        "terminal session, and toolset, and only its final summary returns to "
        "you. Pass every task in `tasks` — one entry spawns one subagent, "
        "several run in parallel (limit in the tasks description).\n\n"
        "Runs in the background: dispatch returns immediately with live "
        "transcript paths, and the completed result (one consolidated message, "
        "results in task order) re-enters the conversation on its own. Do NOT "
        "wait or poll; continue other work. While children run, `action` "
        "(list/steer/stop) controls them live — steer when a transcript shows "
        "a child drifting.\n\n"
        "USE FOR: reasoning-heavy subtasks, work that would flood your context "
        "with intermediate data, or independent parallel workstreams.\n"
        "DO NOT USE FOR (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
        "- A single tool call -> call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot ask questions\n"
        "- Durable work that must survive this session -> cronjob or "
        "terminal(background=True, notify=True); /stop, /new, or "
        "process exit discards running subagents.\n\n"
        "RULES:\n"
        "- Children know nothing of this conversation: pass everything needed "
        "via 'context', including any required output language, tone, or "
        "style (e.g. \"respond in Chinese\").\n"
        "- Child summaries are SELF-REPORTS, not verified facts: a child "
        "claiming \"uploaded successfully\" or \"file written\" may be wrong. "
        "For external side effects (uploads, remote writes, publishing), "
        "require a verifiable handle (URL, ID, absolute path) and verify it "
        "yourself before telling the user the operation succeeded.\n"
        + restrictions_rule +
        "- Children inherit the parent model unless pinned via "
        "delegation.provider / delegation.model in config.yaml."
    )


def _build_tasks_param_description() -> str:
    """Compose the 'tasks' parameter description with current concurrency limit."""
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    return (
        f"The task(s), up to {max_children} in parallel for this user (set "
        "via delegation.max_concurrent_children). Each entry spawns one "
        "subagent with isolated context and terminal session; a single task "
        "is a one-entry array. Required when spawning."
    )


def _build_dynamic_schema_overrides() -> dict:
    """Per-call schema overrides (ToolEntry.dynamic_schema_overrides): every
    get_definitions() pass rewrites the descriptions to the user's actual limits."""
    overrides_params = {**DELEGATE_TASK_SCHEMA["parameters"]}
    # Copy properties so the static schema dict is never mutated.
    overrides_params["properties"] = {k: dict(v) for k, v in DELEGATE_TASK_SCHEMA["parameters"]["properties"].items()}
    overrides_params["properties"]["tasks"]["description"] = _build_tasks_param_description()

    return {"description": _build_top_level_description(), "parameters": overrides_params}


DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    # description / tasks.description are placeholders: the real text is built per
    # get_definitions() call by _build_dynamic_schema_overrides() so the model sees
    # the user's actual max_concurrent_children / max_spawn_depth. Lazy (not at
    # import) so cli.CLI_CONFIG isn't forced to load before the test conftest
    # redirects HERMES_HOME.
    "description": (
        "Spawn one or more subagents in isolated contexts. "
        "Description is rebuilt at every get_definitions() call to reflect "
        "the user's current delegation limits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            # The handler also accepts the legacy single-goal shape (top-level
            # `goal`/`context`/`output_schema`), wrapped into a one-entry batch at
            # dispatch. Unadvertised on purpose (old transcripts only); do not re-add.
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": (
                                "What this subagent should accomplish. Be "
                                "specific and self-contained — it knows "
                                "nothing about your conversation history."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Background THIS child needs: file paths, "
                                "error messages, constraints. Each child "
                                "sees only its own context — repeat shared "
                                "background in every task that needs it."
                            ),
                        },
                        "output_schema": {
                            "type": "object",
                            "description": (
                                "Optional JSON Schema this child's final "
                                "answer must validate against (told to the "
                                "child up front; parent validates with one "
                                "bounded correction retry; result gains "
                                "schema_valid, plus schema_errors on "
                                "failure). Keep it forgiving — require only "
                                "fields you will read."
                            ),
                        },
                    },
                    "required": ["goal"],
                },
                # No maxItems — the runtime limit (delegation.max_concurrent_children)
                # is enforced with a clear error in delegate_task(). A per-task `role`
                # is also accepted — legacy, ignored (capability is depth-derived);
                # unadvertised on purpose, do not re-add.
                "description": "(rebuilt at get_definitions() time)",
            },
            # `background` (bool) is also accepted — DEPRECATED, ignored: top-level
            # delegations always run in the background. Unadvertised; do not re-add.
            "action": {
                "type": "string",
                "enum": ["spawn", "list", "steer", "stop"],
                "description": (
                    "Default 'spawn'. Live control of running children: "
                    "'list' = ids/goals/status/transcripts; 'steer' = queue "
                    "course-correction text into one child (subagent_id + "
                    "message) without stopping it; 'stop' = end one child "
                    "early (subagent_id; partial result still returns). "
                    "Control actions return immediately; goal/tasks are "
                    "ignored unless spawning."
                ),
            },
            "subagent_id": {
                "type": "string",
                "description": (
                    "Target for action='steer'/'stop' (ids from the spawn "
                    "response or action='list')."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "For action='steer': the course correction, appended to "
                    "the child's next tool result mid-run. Be directive and "
                    "specific."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error


def _model_background_value(args: dict, parent_agent=None) -> bool:
    """Background flag for the MODEL-facing dispatch path (registry fallback).

    Top-level delegations always run in the background — the model does not
    choose — for single tasks and fan-out batches alike (one async unit, one
    consolidated result). The exception is an orchestrator subagent (depth > 0),
    which needs its workers' results within its own turn. The live path is
    ``run_agent._dispatch_delegate_task``; this mirrors it for the rare case the
    intercept is bypassed. Direct Python callers keep the synchronous default.
    """
    return not getattr(parent_agent, "_delegate_depth", 0) > 0


_MODEL_HIDDEN_TASK_FIELDS = {"acp_command", "acp_args"}


def _strip_model_hidden_task_fields(tasks: Any) -> Any:
    if not isinstance(tasks, list):
        return tasks
    stripped_tasks = []
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            stripped_tasks.append(task)
            continue
        stripped = {key: value for key, value in task.items() if key not in _MODEL_HIDDEN_TASK_FIELDS}
        changed = changed or len(stripped) != len(task)
        stripped_tasks.append(stripped)
    return stripped_tasks if changed else tasks


registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        tasks=_strip_model_hidden_task_fields(args.get("tasks")),
        max_iterations=args.get("max_iterations"),
        role=args.get("role"),
        background=_model_background_value(args, kw.get("parent_agent")),
        output_schema=args.get("output_schema"),
        action=args.get("action"),
        subagent_id=args.get("subagent_id"),
        message=args.get("message"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
    dynamic_schema_overrides=_build_dynamic_schema_overrides,
)
