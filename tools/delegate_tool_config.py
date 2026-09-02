"""Delegation config knobs (delegation.* keys) and child credential/provider resolution.

Split out of ``tools/delegate_tool.py``; every moved name is re-imported there, so
``tools.delegate_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from utils import base_url_hostname, is_truthy_value

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")

# Runtime-provider sentinel for providers that are not natively known; must
# match hermes_cli.runtime_provider.RUNTIME_PROVIDER_TYPE_CUSTOM.
_RUNTIME_PROVIDER_CUSTOM = "custom"

_DEFAULT_MAX_CONCURRENT_CHILDREN = 10
# One-shot guard: _get_max_concurrent_children() runs on every get_definitions()
# schema rebuild, so the >10 cost advisory would otherwise log on every turn.
_HIGH_CONCURRENCY_WARNED = False
MAX_DEPTH = 1  # flat by default: parent (0) -> child (1); deeper needs max_spawn_depth
_MIN_SPAWN_DEPTH = 1  # floor for the configurable cap; MAX_DEPTH stays the default
_LEGACY_MAX_ASYNC_WARNED = False
# No default wall-clock cap on children: legitimate heavy work (deep reviews,
# research fan-outs, slow reasoning models) was being killed mid-task. Stuck-child
# detection is the heartbeat staleness monitor; delegation.child_timeout_seconds
# opts back in.
DEFAULT_CHILD_TIMEOUT: Optional[float] = None


def _cfg() -> dict:
    """The ``delegation`` section, read through the origin so tests can patch it."""
    from tools.delegate_tool import _load_config

    return _load_config()


# ── Subagent approval callbacks ─────────────────────────────────────────────
# Subagent worker threads don't inherit the CLI's threading.local approval
# callback, so prompt_dangerous_approval() would fall back to input() and
# deadlock the parent's prompt_toolkit TUI. Every worker gets a non-interactive
# callback via ThreadPoolExecutor(initializer=...): deny by default, approve when
# delegation.subagent_auto_approve is true. Both warn for audit. Gateway sessions
# are unaffected (they resolve approvals via tools/approval.py's per-session queue).
def _subagent_auto_deny(command: str, description: str, **kwargs) -> str:
    """Auto-deny (safe default): returns 'deny' so the child sees a recoverable refusal."""
    logger.warning(
        "Subagent auto-denied dangerous command: %s (%s). "
        "Set delegation.subagent_auto_approve: true to allow.",
        command, description,
    )
    return "deny"


def _subagent_auto_approve(command: str, description: str, **kwargs) -> str:
    """Auto-approve (opt-in YOLO via delegation.subagent_auto_approve): returns 'once'."""
    logger.warning(
        "Subagent auto-approved dangerous command: %s (%s)",
        command, description,
    )
    return "once"


def _get_subagent_approval_callback():
    """Callback for subagent worker threads per delegation.subagent_auto_approve (default False)."""
    if is_truthy_value(_cfg().get("subagent_auto_approve", False)):
        return _subagent_auto_approve
    return _subagent_auto_deny


def _get_max_concurrent_children() -> int:
    """delegation.max_concurrent_children > DELEGATION_MAX_CONCURRENT_CHILDREN env > 10.

    Floor of 1 is the only bound enforced; there is no ceiling.
    """
    val = _cfg().get("max_concurrent_children")
    if val is not None:
        try:
            result = max(1, int(val))
        except (TypeError, ValueError):
            logger.warning(
                "delegation.max_concurrent_children=%r is not a valid integer; "
                "using default %d",
                val,
                _DEFAULT_MAX_CONCURRENT_CHILDREN,
            )
            return _DEFAULT_MAX_CONCURRENT_CHILDREN
        if result > 10:
            global _HIGH_CONCURRENCY_WARNED
            if not _HIGH_CONCURRENCY_WARNED:
                _HIGH_CONCURRENCY_WARNED = True
                logger.warning(
                    "delegation.max_concurrent_children=%d: each child consumes API tokens "
                    "independently. High values multiply cost linearly.",
                    result,
                )
        return result
    env_val = os.getenv("DELEGATION_MAX_CONCURRENT_CHILDREN")
    if env_val:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_CONCURRENT_CHILDREN
    return _DEFAULT_MAX_CONCURRENT_CHILDREN


def _get_worktree_isolation() -> bool:
    """delegation.worktree_isolation (bool, default False).

    When enabled each child gets its own git worktree off the parent's HEAD so
    parallel children never contend for one working copy. Git-only and
    local-backend-only; otherwise silently ignored (shared workspace as before).
    """
    return bool(_cfg().get("worktree_isolation", False))


def _get_max_async_children() -> int:
    """Concurrency cap for background delegations == delegation.max_concurrent_children.

    At capacity a new async dispatch is REJECTED (not queued) so a runaway model
    can't pile up unbounded background work; the caller then runs synchronously.
    A leftover ``delegation.max_async_children`` key is ignored with a one-time
    deprecation warning.
    """
    from tools.delegate_tool import _get_max_concurrent_children
    global _LEGACY_MAX_ASYNC_WARNED
    if _cfg().get("max_async_children") is not None and not _LEGACY_MAX_ASYNC_WARNED:
        _LEGACY_MAX_ASYNC_WARNED = True
        logger.warning(
            "delegation.max_async_children is deprecated and ignored; "
            "delegation.max_concurrent_children now caps background "
            "delegations too. Remove the stale key from config.yaml."
        )
    return _get_max_concurrent_children()


def _parse_timeout(raw: Any) -> Optional[float]:
    """Seconds → None (<= 0 disables) or max(30, value). Raises on non-numeric."""
    parsed = float(raw)
    return None if parsed <= 0 else max(30.0, parsed)


def _get_child_timeout() -> Optional[float]:
    """Hard wall-clock cap for one child, or None (default: no timeout).

    Failures should come from what the child does (API/tool errors, iteration
    budget), not a stopwatch; stuck children are caught by the heartbeat
    staleness monitor. delegation.child_timeout_seconds > 0 opts in (floor 30 s);
    0 or negative disables. Env fallback: DELEGATION_CHILD_TIMEOUT_SECONDS.
    """
    val = _cfg().get("child_timeout_seconds")
    if val is not None:
        try:
            return _parse_timeout(val)
        except (TypeError, ValueError):
            logger.warning(
                "delegation.child_timeout_seconds=%r is not a valid number; "
                "using default (no timeout)",
                val,
            )
    env_val = os.getenv("DELEGATION_CHILD_TIMEOUT_SECONDS")
    if env_val:
        try:
            return _parse_timeout(env_val)
        except (TypeError, ValueError):
            pass
    return DEFAULT_CHILD_TIMEOUT


def _get_max_spawn_depth() -> int:
    """delegation.max_spawn_depth floored at 1 (no ceiling).

    Depth 0 is the parent; agents at depths 0..N-1 may spawn, depth N is the
    leaf floor. Default 1 is flat. Each extra level multiplies API cost.
    """
    val = _cfg().get("max_spawn_depth")
    if val is None:
        return MAX_DEPTH
    try:
        ival = int(val)
    except (TypeError, ValueError):
        logger.warning(
            "delegation.max_spawn_depth=%r is not a valid integer; " "using default %d",
            val,
            MAX_DEPTH,
        )
        return MAX_DEPTH
    floored = max(_MIN_SPAWN_DEPTH, ival)
    if floored != ival:
        logger.warning(
            "delegation.max_spawn_depth=%d below floor %d; using %d",
            ival,
            _MIN_SPAWN_DEPTH,
            floored,
        )
    return floored


def _get_orchestrator_enabled() -> bool:
    """delegation.orchestrator_enabled kill switch (default True): False forces every child to leaf."""
    val = _cfg().get("orchestrator_enabled", True)
    if isinstance(val, bool):
        return val
    # Accept "true"/"false" strings from YAML that doesn't auto-coerce.
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "on"}
    return True


def _get_inherit_mcp_toolsets() -> bool:
    """Whether narrowed child toolsets should keep the parent's MCP toolsets."""
    return is_truthy_value(_cfg().get("inherit_mcp_toolsets"), default=True)


def _normalized_runtime_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _inherit_parent_capabilities(
    parent_agent, override_provider, override_base_url
) -> Optional[dict]:
    """Parent's endpoint-trust capability map for a child, or None.

    ``agent.capabilities`` is a trust decision scoped to one provider+endpoint:
    inherited ONLY when the child runs the parent's exact route; any provider or
    base_url override stays DEFAULT-DENY (matches the /model switch posture).
    """
    if override_provider or override_base_url:
        return None
    parent_caps = getattr(parent_agent, "capabilities", None)
    if not isinstance(parent_caps, dict):
        return None
    return {
        key: value
        for key, value in parent_caps.items()
        if isinstance(key, str) and isinstance(value, bool)
    }


def _inherit_parent_base_url(parent_agent, fallback_base_url: Optional[str]) -> Optional[str]:
    """Base URL the parent is actually calling (live client), not a stale attribute.

    ``parent_agent.base_url`` can lag the live client (old OpenRouter URL vs
    local Ollama); inheriting the stale one 401s with a dummy/local key.
    """
    surface_url = _normalized_runtime_url(fallback_base_url)
    client_kwargs = getattr(parent_agent, "_client_kwargs", None)
    client = getattr(parent_agent, "client", None)
    live_candidates = (
        client_kwargs.get("base_url") if isinstance(client_kwargs, dict) else None,
        # OpenAI SDK exposes base_url as httpx.URL — coerce before comparing.
        getattr(client, "base_url", "") if client is not None else None,
    )
    for raw in live_candidates:
        url = _normalized_runtime_url(raw)
        if url and url != surface_url and url.startswith(("http://", "https://")):
            return url
    return fallback_base_url or None


def _loaded_pool(key: Any):
    """``load_pool(key)`` when it holds credentials, else None."""
    from agent.credential_pool import load_pool

    pool = load_pool(key)
    return pool if pool is not None and pool.has_credentials() else None


def _resolve_child_credential_pool(
    effective_provider: Optional[str],
    parent_agent,
    effective_base_url: Optional[str] = None,
):
    """Credential pool for the child: parent's pool (same provider), that
    provider's own pool, or None (child keeps its fixed credential).

    Custom endpoints all collapse to ``provider="custom"``, so they are matched
    by endpoint identity (the ``custom:<name>`` pool key) — sharing the parent's
    pool across different custom endpoints would overwrite the child's delegated
    base_url on lease.
    """
    parent_pool = getattr(parent_agent, "_credential_pool", None)
    if not effective_provider:
        return parent_pool
    parent_provider = getattr(parent_agent, "provider", None) or ""

    if effective_provider == "custom":
        try:
            from agent.credential_pool import get_custom_provider_pool_key

            child_key = get_custom_provider_pool_key(effective_base_url)
            if child_key is None:
                # Unregistered endpoint (no custom_providers entry): keep the
                # child's fixed credential rather than inherit the parent's.
                return None
            parent_key = get_custom_provider_pool_key(
                getattr(parent_agent, "base_url", None)
            )
            if (
                parent_pool is not None
                and parent_provider == "custom"
                and parent_key is not None
                and parent_key == child_key
            ):
                return parent_pool
            return _loaded_pool(child_key)
        except Exception as exc:
            logger.debug(
                "Could not resolve custom credential pool for child endpoint '%s': %s",
                effective_base_url,
                exc,
            )
        return None

    if parent_pool is not None and effective_provider == parent_provider:
        return parent_pool
    try:
        return _loaded_pool(effective_provider)
    except Exception as exc:
        logger.debug(
            "Could not load credential pool for child provider '%s': %s",
            effective_provider,
            exc,
        )
    return None


def _merge_request_overrides(runtime_overrides, explicit_overrides):
    """Merge explicit ``delegation.request_overrides`` OVER runtime-derived ones.

    Explicit top-level keys win; ``extra_body`` is deep-merged ONE level so
    provider personality (e.g. ``thinking: {type: disabled}``) survives unless
    the explicit dict redefines that exact key. Both sides are deep-copied so
    transport-side mutation can't leak into the config/runtime cache. Returns
    None when both sides are empty.
    """
    import copy as _copy

    runtime_overrides = runtime_overrides if isinstance(runtime_overrides, dict) else None
    explicit_overrides = explicit_overrides if isinstance(explicit_overrides, dict) else None
    if not runtime_overrides and not explicit_overrides:
        return None
    merged = _copy.deepcopy(runtime_overrides) if runtime_overrides else {}
    explicit = _copy.deepcopy(explicit_overrides) if explicit_overrides else {}
    runtime_extra = merged.get("extra_body")
    explicit_extra = explicit.pop("extra_body", None)
    merged.update(explicit)
    if isinstance(runtime_extra, dict) and isinstance(explicit_extra, dict):
        runtime_extra.update(explicit_extra)
        merged["extra_body"] = runtime_extra
    elif explicit_extra is not None:
        merged["extra_body"] = explicit_extra
    return merged or None


# Native-SDK providers speak their own wire protocol and can't be reached via
# chat_completions against a base_url: always take the runtime-provider path
# (a configured base_url still flows through it, e.g. a Bedrock region).
_NATIVE_SDK_PROVIDERS = frozenset({"bedrock", "vertex", "google", "google-genai"})
_EXPLICIT_API_MODES = frozenset({"chat_completions", "codex_responses", "anthropic_messages"})


def _require_pinned_command(command: Optional[str], message: str) -> None:
    """A pinned ACP transport command must exist on PATH — refuse loudly rather
    than let the child silently fall back to another transport."""
    if not command:
        return
    import shutil as _shutil

    if not _shutil.which(command):
        raise ValueError(message)


def _direct_endpoint_credentials(cfg_values: dict, explicit_request_overrides) -> dict:
    """``delegation.base_url`` branch: provider/api_mode from URL heuristics."""
    configured_model, configured_provider, configured_base_url, configured_api_key, configured_api_mode = (
        cfg_values["model"], cfg_values["provider"], cfg_values["base_url"],
        cfg_values["api_key"], cfg_values["api_mode"],
    )
    # Shared URL-based api_mode detector so Anthropic-compatible direct
    # endpoints (/anthropic suffix: Azure AI Foundry, MiniMax, Zhipu, LiteLLM)
    # get the Messages transport instead of 404ing on chat_completions.
    from hermes_cli.runtime_provider import _detect_api_mode_for_url

    base_lower = configured_base_url.lower()
    host = base_url_hostname(configured_base_url)
    provider = "custom"
    api_mode = _detect_api_mode_for_url(configured_base_url) or "chat_completions"
    if host == "chatgpt.com" and "/backend-api/codex" in base_lower:
        provider, api_mode = "openai-codex", "codex_responses"
    elif host == "api.anthropic.com":
        provider, api_mode = "anthropic", "anthropic_messages"
    elif "api.kimi.com/coding" in base_lower:
        api_mode = "anthropic_messages"
    # Explicit delegation.api_mode always wins over the URL heuristic.
    if configured_api_mode in _EXPLICIT_API_MODES:
        api_mode = configured_api_mode

    # provider configured ALONGSIDE base_url: pull that provider's request
    # personality (request_overrides / max_output_tokens) onto the explicit
    # endpoint. Best-effort — a resolution failure only skips the overrides.
    request_overrides = None
    max_output_tokens = None
    if configured_provider:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(
                requested=configured_provider, target_model=configured_model
            )
            request_overrides = dict(runtime.get("request_overrides") or {}) or None
            max_output_tokens = runtime.get("max_output_tokens")
        except Exception as exc:
            logger.debug(
                "delegation.base_url: runtime resolution for provider '%s' "
                "failed; proceeding without request_overrides: %s",
                configured_provider,
                exc,
            )
    return {
        "model": configured_model,
        "provider": provider,
        "base_url": configured_base_url,
        "api_key": configured_api_key,  # None → inherited from parent in _build_child_agent
        "api_mode": api_mode,
        "request_overrides": _merge_request_overrides(request_overrides, explicit_request_overrides),
        "max_output_tokens": max_output_tokens,
    }


def _runtime_provider_credentials(cfg_values: dict, explicit_request_overrides) -> dict:
    """``delegation.provider`` branch: full bundle via the runtime provider system."""
    configured_model, configured_provider = cfg_values["model"], cfg_values["provider"]
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=configured_provider, target_model=configured_model)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation provider '{configured_provider}': {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or set delegation.base_url/delegation.api_key for a direct endpoint. "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes auth'."
        )
    pinned_command = runtime.get("command")
    _require_pinned_command(
        pinned_command,
        f"Delegation provider '{configured_provider}' is pinned to the "
        f"'{pinned_command}' command, which was not found on PATH. "
        f"Install it or choose a different delegation provider.",
    )
    return {
        "model": configured_model or runtime.get("model") or None,
        "provider": configured_provider if runtime.get("provider") == _RUNTIME_PROVIDER_CUSTOM else runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "request_overrides": _merge_request_overrides(
            runtime.get("request_overrides"), explicit_request_overrides
        )
        or {},
        "max_output_tokens": runtime.get("max_output_tokens"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve the child credential bundle from the ``delegation`` config section.

    Three branches: ``base_url`` set → direct endpoint (``api_key`` None means
    inherit the parent's key, so providers keyed outside OPENAI_API_KEY work);
    ``provider`` set → full bundle via the runtime provider system (same path as
    CLI/gateway startup); neither → None values, child inherits everything.
    ``request_overrides`` is honored on every branch. Raises ValueError with a
    user-facing message on credential failure.
    """
    values = {k: str(cfg.get(k) or "").strip() or None for k in ("model", "provider", "base_url", "api_key")}
    values["api_mode"] = str(cfg.get("api_mode") or "").strip().lower() or None
    explicit_request_overrides = (
        cfg.get("request_overrides")
        if isinstance(cfg.get("request_overrides"), dict)
        else None
    )
    is_native_sdk_provider = (values["provider"] or "").strip().lower() in _NATIVE_SDK_PROVIDERS

    if values["base_url"] and not is_native_sdk_provider:
        return _direct_endpoint_credentials(values, explicit_request_overrides)
    if not values["provider"]:
        # Pure inherit; explicit request_overrides still merge OVER the parent's.
        return {
            "model": values["model"],
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": _merge_request_overrides(
                getattr(parent_agent, "request_overrides", None),
                explicit_request_overrides,
            ),
            "max_output_tokens": None,
        }
    return _runtime_provider_credentials(values, explicit_request_overrides)


def _load_config() -> dict:
    """Return the ``delegation`` config section (read-only — do NOT mutate).

    Prefers the shared ``load_config_readonly()`` (follows HERMES_HOME/profile;
    no deepcopy, since this runs on every get_definitions() rebuild) over the
    legacy ``cli.CLI_CONFIG``, which can hide user-set keys. Exception:
    ``HERMES_IGNORE_USER_CONFIG=1`` is only honored by the legacy loader, so it
    stays authoritative when that flag is set.
    """
    if os.environ.get("HERMES_IGNORE_USER_CONFIG") != "1":
        try:
            from hermes_cli.config import load_config_readonly

            cfg = load_config_readonly().get("delegation") or {}
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation") or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}
