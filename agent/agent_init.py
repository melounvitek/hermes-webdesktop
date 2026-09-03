"""Implementation of :meth:`AIAgent.__init__` as ``init_agent(agent, ...)``.

``init_agent`` is a thin, ordered orchestrator over ``_init_*`` / ``_build_*`` phase
helpers (routing → callbacks → client → tools → session → config sections → compression →
context engine). Phase ORDER is load-bearing: later phases read attributes earlier ones set.
Symbols that tests patch on ``run_agent.*`` (``OpenAI``, ``get_tool_definitions``,
``logger``, …) are resolved through :func:`_ra` so the patch contract is preserved.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

from agent.context_compressor import ContextCompressor
from agent.iteration_budget import IterationBudget
from agent.memory_manager import StreamingContextScrubber
from agent.session_activity import ActivityProvenance
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH, fetch_model_metadata, is_local_endpoint, query_ollama_num_ctx
)
from agent.process_bootstrap import _install_safe_stdio
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.think_scrubber import StreamingThinkScrubber
from agent.tool_guardrails import (
    ToolCallGuardrailConfig, ToolCallGuardrailController, ToolGuardrailDecision
)
from hermes_cli.config import cfg_get
from hermes_cli.route_identity import normalize_route_base_url
from hermes_cli.timeouts import get_provider_request_timeout
from hermes_constants import get_hermes_home
from utils import base_url_host_matches, is_truthy_value

# Same logger name as run_agent so caplog/patches on "run_agent" see our records.
logger = logging.getLogger("run_agent")


# Memory providers already warned unavailable — the gateway builds a fresh AIAgent per
# message, so an un-deduped warning would fire every turn.
_warned_unavailable_providers: set[str] = set()


def _warn_memory_provider_unavailable(name: str, reason: str = "") -> None:
    """Warn once per provider that a configured memory provider is unavailable.

    ``is_available()`` is a side-effect-free hot-path check and can't log itself; without
    this the provider is silently dropped (common trigger: systemd/gateway services not
    inheriting ``~/.hermes/.env``). ``reason`` is the provider's ``unavailable_reason()``
    hint — this is the only place it can reach the user, so it is appended when present.
    """
    if name in _warned_unavailable_providers:
        return
    _warned_unavailable_providers.add(name)
    logger.warning(
        "Memory provider %r is selected but reports unavailable — external memory "
        "is disabled for this session (built-in memory still works). Check the "
        "provider's credentials/config with 'hermes memory status'. Note: "
        "systemd/gateway services do not inherit ~/.hermes/.env automatically; set "
        "any required variables in the service environment.%s",
        name,
        f" {reason}" if reason else "",
    )


def _ra():
    """Lazy ``run_agent`` so ``patch("run_agent.OpenAI")`` & co. reach this code path."""
    import run_agent
    return run_agent


# Canonicalize an endpoint URL for model-route identity comparisons.
_normalize_route_base_url = normalize_route_base_url


def _provider_default_routes(provider: str) -> set[str]:
    """Return known exact default routes for a canonical provider id."""
    routes: set[str] = set()

    def add(value):
        route = _normalize_route_base_url(value)
        if route:
            routes.add(route)

    try:
        from hermes_cli.providers import HERMES_OVERLAYS, get_provider

        overlay = HERMES_OVERLAYS.get(provider)
        provider_def = get_provider(provider, allow_network=False)
        add(getattr(overlay, "base_url_override", ""))
        add(getattr(provider_def, "base_url", ""))
    except Exception:
        pass

    try:
        from providers import get_provider_profile

        add(getattr(get_provider_profile(provider), "base_url", ""))
    except Exception:
        pass

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.models import normalize_provider as normalize_model_provider
        from hermes_cli.providers import normalize_provider as normalize_registry_provider

        for provider_id, config in PROVIDER_REGISTRY.items():
            if normalize_registry_provider(normalize_model_provider(provider_id)) == provider:
                add(getattr(config, "inference_base_url", ""))
    except Exception:
        pass

    if provider == "gemini":
        routes.update(f"{route.rstrip('/')}/openai" for route in list(routes))
    return routes


def _context_route_mismatch(
    configured_base_url: Any, active_base_url: Any, configured_provider: Any, active_provider: Any,
    *, already_normalized: bool = False,
) -> bool:
    """Return whether a context pin's configured route differs from runtime."""
    if already_normalized:
        configured_route = str(configured_base_url or "")
        active_route = str(active_base_url or "")
    else:
        configured_route = _normalize_route_base_url(configured_base_url)
        active_route = _normalize_route_base_url(active_base_url)
    if configured_route:
        return configured_route != active_route

    configured_provider = str(configured_provider or "").strip()
    active_provider = str(active_provider or "").strip()
    if not configured_provider:
        return False
    try:
        from hermes_cli.models import normalize_provider as normalize_model_provider

        configured_provider = normalize_model_provider(configured_provider)
        active_provider = normalize_model_provider(active_provider)
    except Exception:
        configured_provider = configured_provider.lower()
        active_provider = active_provider.lower()
    try:
        from hermes_cli.providers import normalize_provider as normalize_registry_provider

        configured_provider = normalize_registry_provider(configured_provider)
        active_provider = normalize_registry_provider(active_provider)
    except Exception:
        pass

    if active_route:
        configured_routes = _provider_default_routes(configured_provider)
        if configured_routes:
            return active_route not in configured_routes
        # Named/custom providers have no catalog default routes: an empty configured URL
        # with a matching provider identity is the same route (gateway display paths
        # compare the raw empty model.base_url and must not drop model.context_length).
        return not (active_provider and configured_provider == active_provider)
    return bool(
        configured_provider and active_provider and configured_provider != active_provider
    )


def _normalize_custom_provider_name(value: Any) -> str:
    """Mirror runtime normalization for a requested custom-provider identity."""
    return str(value or "").strip().lower().replace(" ", "-")


def _custom_provider_runtime_ids(value: Any) -> set[str]:
    """Return raw/menu identities that runtime accepts for a configured name."""
    normalized = _normalize_custom_provider_name(value)
    if not normalized:
        return set()
    return {normalized, f"custom:{normalized}"}


def _build_codex_gpt5_autoraise_notice(
    autoraise: Dict[str, Any], context_length: Optional[int] = None
) -> str:
    """Build the one-time notice shown when Codex gpt-5.x raises compaction.

    ``autoraise`` is ``{"model", "from", "to"}``. ``context_length`` is the live-resolved
    window (Codex's /models catalog is authoritative and shifts server-side), so the banner
    reports what this session actually got. The same text is printed for CLI users and
    replayed via ``status_callback`` for gateway users, so it must be self-contained and
    include the exact opt-back-out command.
    """
    model = str(autoraise.get("model") or "gpt-5.4/5.5").strip().lower().rsplit("/", 1)[-1]
    if isinstance(context_length, int) and context_length > 0:
        cap = f"{round(context_length / 1000)}K"
    else:
        # Static fallback: gpt-5.3-codex-spark has a native 128K window; the
        # gpt-5.4/5.5/5.6 family is capped at 272K by the Codex OAuth backend.
        cap = "128K" if model.startswith("gpt-5.3-codex-spark") else "272K"
    from_pct = int(round(autoraise["from"] * 100))
    to_pct = int(round(autoraise["to"] * 100))
    return (
        f"ℹ Codex {model} caps context at {cap}, so auto-compaction was raised "
        f"to {to_pct}% (from {from_pct}%) to use more of the window before "
        f"summarizing.\n"
        f"  Opt back out: hermes config set compression.codex_gpt55_autoraise false"
    )


def _resolve_compression_threshold(
    global_threshold: float, model_cthresh: Optional[float], *, model: Optional[str] = None,
    is_codex_autoraise: bool,
) -> tuple[float, Optional[Dict[str, Any]]]:
    """Combine the user's global compaction threshold with a per-model override.

    Returns ``(effective_threshold, autoraise_notice)``; the notice is
    ``{"model", "from", "to"}`` only when a Codex autoraise actually RAISES the threshold.
    Codex overrides never LOWER a higher user-configured threshold (the user deliberately
    keeps more raw context); other overrides (e.g. Arcee Trinity) stay unconditional.
    """
    if model_cthresh is None:
        return global_threshold, None
    if is_codex_autoraise:
        if model_cthresh <= global_threshold + 1e-9:
            return global_threshold, None
        return model_cthresh, {"model": model, "from": global_threshold, "to": model_cthresh}
    return model_cthresh, None


def _codex_gpt55_autoraise_notice_marker():
    """Per-profile marker path (``$HERMES_HOME`` is profile-scoped; not a config key)."""
    return get_hermes_home() / ".codex_gpt55_autoraise_notice"


def _codex_gpt55_autoraise_notice_state(autoraise: Dict[str, Any]) -> str:
    """Notice identity keyed on what it displays (model + from→to percentages).

    An unchanged threshold stays silent across restarts; a changed global threshold or a
    different autoraised Codex model re-notifies once.
    """
    model = str(autoraise.get("model") or "").strip().lower().rsplit("/", 1)[-1]
    from_pct = int(round(float(autoraise["from"]) * 100))
    to_pct = int(round(float(autoraise["to"]) * 100))
    return f"{model}:{from_pct}:{to_pct}"


def _codex_gpt55_autoraise_notice_seen(autoraise: Dict[str, Any]) -> bool:
    """True if this exact notice was already shown for this profile (unreadable = unseen)."""
    try:
        current = _codex_gpt55_autoraise_notice_state(autoraise)
        return _codex_gpt55_autoraise_notice_marker().read_text(
            encoding="utf-8"
        ).strip() == current
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _record_codex_gpt55_autoraise_notice(autoraise: Dict[str, Any]) -> None:
    """Persist that the notice was shown. Best-effort: a failure only re-shows it later."""
    try:
        marker = _codex_gpt55_autoraise_notice_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_codex_gpt55_autoraise_notice_state(autoraise), encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError):
        pass


def _normalized_custom_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _custom_provider_model_matches(agent_model: str, entry: Dict[str, Any]) -> bool:
    agent_model_norm = str(agent_model or "").strip().lower()
    # Multi-model entries (`providers.<name>.models` mapping / legacy `models:` list):
    # matching ANY catalog entry counts, else a provider whose `model` differs from the
    # session model drops its extra_body (e.g. OpenAI service_tier) → wrong billing tier.
    models = entry.get("models")
    catalog: List[str] = []
    if isinstance(models, dict):
        catalog = [str(k).strip().lower() for k in models]
    elif isinstance(models, (list, tuple)):
        catalog = [str(m).strip().lower() for m in models]
    if catalog and agent_model_norm in catalog:
        return True
    provider_model = str(entry.get("model", "") or "").strip().lower()
    if not provider_model and not catalog:
        return True
    return provider_model == agent_model_norm


def _custom_provider_extra_body_for_agent(
    *, provider: str, model: str, base_url: str, custom_providers: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    provider_norm = (provider or "").strip().lower()
    if provider_norm == "custom":
        provider_key_filter = ""
    elif provider_norm.startswith("custom:"):
        provider_key_filter = provider_norm.split(":", 1)[1].strip()
    else:
        return None

    target_url = _normalized_custom_base_url(base_url)
    if not target_url:
        return None

    fallback: Optional[Dict[str, Any]] = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        if provider_key_filter:
            entry_keys = {
                str(entry.get("provider_key", "") or "").strip().lower(),
                str(entry.get("name", "") or "").strip().lower(),
            }
            if provider_key_filter not in entry_keys:
                continue
        if _normalized_custom_base_url(entry.get("base_url")) != target_url:
            continue
        extra_body = entry.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body:
            continue
        provider_model = str(entry.get("model", "") or "").strip()
        if provider_model:
            if _custom_provider_model_matches(model, entry):
                return dict(extra_body)
        elif fallback is None:
            fallback = dict(extra_body)

    return fallback


def _merge_custom_provider_extra_body(agent, custom_providers: List[Dict[str, Any]]) -> None:
    extra_body = _custom_provider_extra_body_for_agent(
        provider=agent.provider, model=agent.model, base_url=agent.base_url,
        custom_providers=custom_providers,
    )
    if not extra_body:
        return

    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    merged_extra_body = dict(extra_body)
    existing_extra_body = overrides.get("extra_body")
    if isinstance(existing_extra_body, dict):
        merged_extra_body.update(existing_extra_body)
    overrides["extra_body"] = merged_extra_body
    agent.request_overrides = overrides


def _normalize_run_budget_seconds(value) -> Optional[float]:
    """Positive float or None (feature off). ``bool`` rejected: YAML ``true`` → 1s budget."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds <= 0:  # NaN or non-positive
        return None
    return seconds


def _refuse_checkpoint_required_on_codex_app_server(
    checkpoint_required: bool, api_mode: Optional[str]
) -> None:
    """Fail closed at init when the checkpoint gate cannot be honored.

    The codex app-server compacts its own thread without a truthful pre-compaction
    transcript boundary (in default "native" mode Hermes never initiates it), so refusing
    here keeps a turn from ever reaching a codex-owned compaction boundary — the
    compress_context() guard alone cannot cover native turns.
    """
    if checkpoint_required and api_mode == "codex_app_server":
        raise RuntimeError(
            "BLOCKED_MISSING_PREREQUISITE: compression.checkpoint_required "
            "is incompatible with the codex_app_server API mode: the codex "
            "agent compacts its own thread without a truthful pre-compaction "
            "transcript boundary, so a required pre-compress checkpoint "
            "cannot be guaranteed. Disable compression.checkpoint_required "
            "or use a non-app-server API mode."
        )


def _parse_config_int(raw: Any, default: int) -> int:
    """Strict int coercion: rejects bool (YAML ``true`` → 1) and fractional floats."""
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _cfg_flag(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    """Legacy string-set truthiness used by the ``compression`` section."""
    return str(cfg.get(key, default)).lower() in {"true", "1", "yes"}


def _cfg_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """``cfg[key]`` if it is a mapping, else ``{}`` (malformed sections are ignored)."""
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}


@dataclass
class CompressionSettings:
    """Parsed ``compression`` config section (see ``_parse_compression_config``)."""
    threshold: Any
    autoraise_notice_enabled: Any
    enabled: Any
    target_ratio: Any
    protect_last: Any
    tail_mode: Any
    min_tail_users: Any
    max_attempts: Any
    proactive_prune_tokens: Any
    proactive_prune_min_chars: Any
    proactive_prune_min_reclaim: Any
    protect_first: Any
    abort_on_summary_failure: Any
    model_thresholds: Any
    threshold_tokens: Any
    checkpoint_required: Any
    in_place: Any
    micro_compact: Any
    micro_compact_every_n_turns: Any
    micro_compact_defrag_tokens: Any
    codex_app_server_auto: Any
    codex_responses_native: Any
    codex_responses_compact_threshold: Any
    idle_compact_after_seconds: Any


_EXPLICIT_API_MODES = {
    "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
    "codex_app_server",
}


def _resolve_api_mode(agent, api_mode, provider_name, base_url):
    """Set ``agent.api_mode`` (and provider rewrites) — ordered ladder, first match wins."""
    host, url = agent._base_url_hostname, agent._base_url_lower
    if api_mode in _EXPLICIT_API_MODES:
        agent.api_mode = api_mode
    elif agent.provider in {"openai-codex", "xai", "xai-oauth"}:
        agent.api_mode = "codex_responses"
    elif provider_name is None and host == "chatgpt.com" and "/backend-api/codex" in url:
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
    elif provider_name is None and host == "api.x.ai":
        agent.api_mode = "codex_responses"
        agent.provider = "xai"
    elif agent.provider == "anthropic" or (provider_name is None and host == "api.anthropic.com"):
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
    elif url.rstrip("/").endswith("/anthropic"):
        # Third-party Anthropic-compatible endpoints (MiniMax, DashScope) end in /anthropic.
        agent.api_mode = "anthropic_messages"
    elif agent.provider == "bedrock" or (
        host.startswith("bedrock-runtime.") and base_url_host_matches(url, "amazonaws.com")
    ):
        agent.api_mode = "bedrock_converse"
    elif agent.provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal is dual-wire: anthropic/* → Messages, everything else → chat_completions.
        # Covers direct AIAgent construction without a resolved runtime.
        from hermes_cli.providers import nous_api_mode

        agent.api_mode = nous_api_mode(agent.model)
    else:
        # Host-mandated wire check — LAST, so the provider-slug rewrites above always win.
        # Covers api.meta.ai → codex_responses (prompt caching: 0% on chat vs 93-99%).
        # URL-driven, not provider-name-driven: `providers.meta` may point anywhere.
        try:
            from hermes_cli.providers import host_mandated_api_mode as _host_mandated_api_mode

            _mandated = _host_mandated_api_mode(base_url or "")
        except Exception:
            _mandated = None
        agent.api_mode = _mandated if _mandated is not None else "chat_completions"


def _finalize_routing(agent, api_mode, credential_pool):
    # Credential-pool validation runs AFTER provider auto-detection so a pool scoped to
    # "anthropic" isn't rejected for provider=None + anthropic.com URL.
    if credential_pool is not None:
        try:
            from agent.credential_pool import credential_pool_matches_provider

            if not credential_pool_matches_provider(
                credential_pool, agent.provider, base_url=agent.base_url,
            ):
                agent._credential_pool = None
        except Exception:
            agent._credential_pool = None

    # Eagerly warm the transport cache so import errors surface at init, not
    # mid-conversation. Non-fatal — transport may not exist for all modes yet.
    try:
        agent._get_transport()
    except Exception:
        pass

    try:
        from hermes_cli.model_normalize import (
            _AGGREGATOR_PROVIDERS, normalize_model_for_provider
        )

        if agent.provider not in _AGGREGATOR_PROVIDERS:
            agent.model = normalize_model_for_provider(agent.model, agent.provider)
    except Exception:
        pass

    # Auto-upgrade to Responses for GPT-5.x-style models and direct OpenAI URLs, unless:
    # api_mode was explicit, the runtime is ACP (`acp://` — ACP clients route themselves
    # and lack the Responses surface), or the URL is Azure OpenAI (gpt-5.x on
    # /chat/completions only). Provider exceptions live in
    # _provider_model_requires_responses_api.
    _base_lower = str(agent.base_url or "").lower()
    if (
        api_mode is None
        and agent.api_mode == "chat_completions"
        and agent.provider != "copilot-acp"
        and not _base_lower.startswith(("acp://", "acp+tcp://"))
        and not agent._is_azure_openai_url()
        and (
            agent._is_direct_openai_url()
            or agent._provider_model_requires_responses_api(agent.model, provider=agent.provider)
        )
    ):
        agent.api_mode = "codex_responses"
        # Invalidate the eager-warmed transport cache — api_mode changed after the warm.
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()

    # Pre-warm the OpenRouter model metadata cache (1h TTL) off-thread so the first pricing
    # estimate doesn't block. Process-level Event guard: the gateway builds an AIAgent per
    # message, and an unguarded spawn leaks one OS thread per message.
    if (agent.provider == "openrouter" or agent._is_openrouter_url()) and \
            not _ra()._openrouter_prewarm_done.is_set():
        _ra()._openrouter_prewarm_done.set()
        threading.Thread(
            target=fetch_model_metadata, daemon=True, name="openrouter-prewarm",
        ).start()


def _init_control_state(agent):
    # Tool execution state — allows _vprint during tool execution even when stream
    # consumers are registered (no tokens streaming then).
    agent._executing_tools = False
    agent._tool_guardrails = ToolCallGuardrailController()
    agent._tool_guardrail_halt_decision: ToolGuardrailDecision | None = None

    # Interrupt mechanism for breaking out of tool loops. Hard cancellation is separate
    # from redirect/message state; a thread-safe Event makes the cause atomic for pollers.
    agent._interrupt_requested = False
    agent._interrupt_message = None  # Optional message that triggered interrupt
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id: int | None = None  # Set at run_conversation() start
    agent._interrupt_thread_signal_pending = False
    agent._client_lock = threading.RLock()
    agent._model_request_active = threading.Event()
    agent._supports_active_turn_redirect = True

    # /steer — inject a user note into the next tool result without interrupting: the
    # drain hook appends it to the last tool result after the current batch, preserving
    # role alternation (no new user turn).
    agent._pending_steer: Optional[str] = None
    agent._pending_steer_lock = threading.Lock()

    # Active-turn redirect: unlike a hard /stop, preserve the valid turn prefix, cancel
    # only the in-flight request and rebuild its tail with the correction. Drained at a
    # role-safe boundary.
    agent._pending_redirect: Optional[str] = None
    agent._pending_redirect_lock = threading.Lock()

    # Concurrent-tool worker tids: `_set_interrupt` on `_execution_thread_id` alone doesn't
    # reach ThreadPoolExecutor workers, so interrupt()/clear_interrupt() fan out to these.
    agent._tool_worker_threads: set[int] = set()
    agent._tool_worker_threads_lock = threading.Lock()

    # Subagent delegation state
    agent._delegate_depth = 0        # 0 = top-level agent, incremented for children
    agent._active_children = []      # Running child AIAgents (for interrupt propagation)
    agent._active_children_lock = threading.Lock()

    # Background memory/skill review state (agent/background_review.py). The run is
    # installed before the worker starts and fences its first provider-capable phase; the
    # direct agent pointer keeps interrupt propagation available once the fork exists.
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._background_review_lock = threading.Lock()


def _init_prompt_cache_config(agent):
    # Anthropic prompt caching: auto-enabled for Claude on native Anthropic, OpenRouter and
    # anthropic_messages gateways (~75% input savings). Four breakpoints: static system
    # prefix, full system prompt, last two messages. See ``_anthropic_prompt_cache_policy``.
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy()
    )
    agent._cache_disabled = False
    # prompt_caching.cache_ttl: "5m" (default) or "1h" (2x write cost, pays off with
    # >5-minute pauses); unknown values keep "5m". A falsy value (false / null / "off" /
    # "disabled" / "no" / "none") disables caching entirely — OAuth plans billing cache
    # writes, or proxies adding their own cache_control. The disable survives /model
    # switches and fallback re-derivation via anthropic_prompt_cache_policy().
    agent._cache_ttl = "5m"
    try:
        from hermes_cli.config import load_config_readonly as _load_pc_cfg

        from agent.agent_runtime_helpers import cache_ttl_means_disabled

        _pc_cfg = _load_pc_cfg().get("prompt_caching", {}) or {}
        _ttl = _pc_cfg.get("cache_ttl", "5m")
        if _ttl in {"5m", "1h"}:
            agent._cache_ttl = _ttl
        elif cache_ttl_means_disabled(_ttl):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False
            agent._cache_ttl = None
            agent._cache_disabled = True
    except Exception:
        pass


def _init_turn_state(agent, run_budget_seconds):
    # Iteration budget: notify the LLM only on actual exhaustion (ONE message, one grace
    # call, then a forced summarise request). Intermediate pressure warnings made models
    # give up early on complex tasks.
    agent._budget_exhausted_injected = False
    agent._budget_grace_call = False

    # Wall-clock run budget (seconds per run_conversation turn). Explicit constructor arg
    # wins; else resolved from config.yaml (agent.run_budget_seconds) in
    # _apply_agent_section. None = fully off: no clock reads, no injection, no capping.
    agent.run_budget_seconds = _normalize_run_budget_seconds(run_budget_seconds)
    # Set by turn_context.prepare_turn when a run budget is active; None otherwise.
    agent._run_budget_started_at = None
    # One-shot latch for the 80% wrap-up notice (reset each turn).
    agent._run_budget_wrapup_injected = False

    # Activity tracking — updated on each API call, tool execution, and stream chunk. Read
    # by the gateway timeout handler and the "still working" notifications.
    agent._last_activity_ts: float = time.time()
    agent._last_activity_desc: str = "initializing"
    # Default paths and _touch_activity stamp unknown; named provenances are stamped by
    # compression writers (heartbeat / timeout / cooldown).
    agent._last_activity_provenance = ActivityProvenance.UNKNOWN
    # Rate-limit durable SessionDB activity stamps from _touch_activity.
    agent._session_activity_last_persist_mono: float = 0.0
    agent._current_tool: str | None = None
    agent._api_call_count: int = 0
    # Opt-out for the between-turns MCP tool refresh (build_turn_context). Set on internal
    # forks (background_review) that must keep ``tools[]`` byte-identical for cache parity.
    agent._skip_mcp_refresh = False
    # Registry generation the tool snapshot was derived from: lets a late/concurrent
    # refresh reject a stale rebuild instead of clobbering a newer one (set in _load_tools).
    agent._tool_snapshot_generation = 0
    # Rate limit tracking from x-ratelimit-* response headers; read by /usage.
    agent._rate_limit_state = None

    # Credits tracking (dev-only, behind HERMES_DEV_CREDITS) from x-nous-credits-* headers.
    # Session-start remaining is latched the first time a header is seen so cumulative
    # micros spent can be reported. Threshold-notice latch: sticky-notice keys + gates.
    agent._credits_state = None
    agent._credits_session_start_micros = None
    from agent.credits_tracker import new_credits_latch

    agent._credits_latch = new_credits_latch()

    # OpenRouter response cache hits (X-OpenRouter-Cache-Status: HIT in stream headers).
    agent._or_cache_hits: int = 0


def _setup_logging(agent):
    # agent.log (INFO+) and errors.log (WARNING+) under ~/.hermes/logs/. Idempotent, so
    # gateway mode (new AIAgent per message) won't duplicate handlers.
    from hermes_logging import setup_logging, setup_verbose_logging
    setup_logging(hermes_home=_ra()._hermes_home)

    if agent.verbose_logging:
        setup_verbose_logging()
        _ra().logger.info("Verbose logging enabled (third-party library logs suppressed)")
    # Quiet mode deliberately does NOT raise per-logger levels: that would starve the root
    # file handlers (isEnabledFor() is checked before propagation). setup_logging()
    # installs no console handler in quiet mode; noise reduction belongs in hermes_logging.


def _init_stream_state(agent):
    # Internal stream callback (streaming TTS); set here so _vprint can reference it early.
    agent._stream_callback = None
    # Deferred paragraph break — set after tool iterations so one "\n\n" precedes the next
    # real text delta.
    agent._stream_needs_break = False
    # Stateful scrubbers for <memory-context> / thinking spans split across stream deltas:
    # per-delta regexes can't survive chunk boundaries (both tags needed in one string).
    agent._stream_context_scrubber = StreamingContextScrubber()
    agent._stream_think_scrubber = StreamingThinkScrubber()
    # Visible assistant text already delivered via live token callbacks this response —
    # avoids re-sending commentary the provider later returns as a completed interim.
    agent._current_streamed_assistant_text = ""
    # Completed interim messages delivered this user turn; spans Codex continuation/tool
    # calls so repeated commentary is not re-sent before normalization dedups it.
    agent._delivered_interim_texts: set[str] = set()

    # Single-writer guard for the streaming delta sink: a superseded stream (reconnected
    # past, socket abort raced) must not interleave tokens with the retry's stream. Each
    # attempt claims a monotonic writer token; the sink drops chunks from threads holding a
    # stale one. Threads that never claimed are never fenced.
    agent._stream_writer_lock = threading.Lock()
    agent._stream_writer_token = 0
    agent._stream_writer_tls = threading.local()
    agent._stream_writer_dropped = 0

    # Current-turn user-message override when the API-facing message intentionally differs
    # from the persisted transcript (e.g. CLI voice mode's temporary prefix).
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None

    # Anthropic image-to-text fallbacks cached per image payload/URL so one tool loop
    # doesn't repeatedly re-run auxiliary vision on the same image history.
    agent._anthropic_image_fallback_cache: Dict[str, str] = {}


def _bedrock_region_from_url(base_url) -> str:
    """AWS region from a bedrock-runtime.<region>.amazonaws.com URL (default us-east-1)."""
    m = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
    return m.group(1) if m else "us-east-1"


def _print_key_banner(key, label: str, warn_missing: bool = False) -> None:
    """Masked credential line. ``key`` may be a callable Entra ID bearer provider (Azure
    Foundry) — never invoke or inspect it. Keys ≤ 12 chars (incl. "dummy-key") are not shown."""
    from agent.azure_identity_adapter import is_token_provider

    if is_token_provider(key):
        print("🔑 Using credentials: Microsoft Entra ID")
    elif isinstance(key, str) and len(key) > 12:
        print(f"🔑 Using {label}: {key[:8]}...{key[-4:]}")
    elif warn_missing:
        print("⚠️  Warning: API key appears invalid or missing")


def _init_anthropic_client(agent, api_key, base_url, _provider_timeout):
    """anthropic_messages: native Anthropic SDK (or AnthropicBedrock for Bedrock+Claude)."""
    from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
    agent.client = None
    agent._client_kwargs = {}
    agent._anthropic_base_url = base_url
    if agent.provider == "bedrock":
        # AnthropicBedrock SDK for full feature parity (prompt caching, thinking budgets).
        from agent.anthropic_adapter import build_anthropic_bedrock_client
        _br_region = agent._bedrock_region = _bedrock_region_from_url(base_url)
        agent._anthropic_client = build_anthropic_bedrock_client(_br_region)
        agent._anthropic_api_key = "aws-sdk"
        agent._is_anthropic_oauth = False
        agent.api_key = "aws-sdk"
        if not agent.quiet_mode:
            print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")
        return
    # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic. Other
    # anthropic_messages providers (MiniMax, Alibaba, …) must use their own key — falling
    # back would send Anthropic credentials to third-party endpoints.
    _is_native_anthropic = agent.provider == "anthropic"
    effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")

    # MiniMax OAuth tokens live ~15 min and the Anthropic SDK freezes ``api_key`` at
    # construction, so swap in a callable token provider: ``build_anthropic_client``
    # installs an httpx hook that mints a fresh bearer per request (re-reading auth.json,
    # so refreshes from other processes are seen). Cost: one file read per request.
    if agent.provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
        try:
            from hermes_cli.auth import build_minimax_oauth_token_provider
            effective_key = build_minimax_oauth_token_provider()
        except Exception as _mm_exc:  # noqa: BLE001 — never block startup on this
            logging.getLogger(__name__).warning(
                "MiniMax OAuth: failed to install per-request token provider "
                "(%s); falling back to static bearer that will expire ~15min in.",
                _mm_exc,
            )

    agent.api_key = effective_key
    agent._anthropic_api_key = effective_key
    # OAuth only for native Anthropic: third-party anthropic_messages providers must never
    # trip OAuth paths — those inject Claude-Code identity headers → 401/403.
    from agent.anthropic_adapter import _is_oauth_token as _is_oat
    agent._is_anthropic_oauth = _is_oat(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
    agent._anthropic_client = build_anthropic_client(effective_key, base_url, timeout=_provider_timeout)
    if not agent.quiet_mode:
        print(f"🤖 AI Agent initialized with model: {agent.model} (Anthropic native)")
        _print_key_banner(effective_key, "token")


def _init_moa_client(agent, api_key):
    """provider == "moa": virtual Mixture-of-Agents facade, no real HTTP client."""
    from agent.moa_loop import build_moa_facade
    agent.api_mode = "chat_completions"

    # build_moa_facade wires the reference relay ("moa.reference" / "moa.progress" /
    # "moa.phase" / "moa.aggregating" events through tool_progress_callback) so every
    # surface shows each reference's answer before the aggregator acts. Display-only;
    # shared with fallback-restore so a restored facade keeps emitting.
    agent.client = build_moa_facade(agent, agent.model)
    agent._client_kwargs = {}
    agent.api_key = api_key or "moa-virtual-provider"
    agent.base_url = "moa://local"
    if not agent.quiet_mode:
        print(f"🤖 AI Agent initialized with MoA preset: {agent.model}")


def _init_bedrock_client(agent, base_url):
    """bedrock_converse: boto3 directly, no OpenAI client."""
    agent._bedrock_region = _bedrock_region_from_url(base_url)
    # Guardrail config — read from config.yaml at init time.
    agent._bedrock_guardrail_config = None
    try:
        from hermes_cli.config import load_config_readonly as _load_br_cfg
        _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
        if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
            agent._bedrock_guardrail_config = {
                "guardrailIdentifier": _gr["guardrail_identifier"],
                "guardrailVersion": _gr["guardrail_version"],
            }
            if _gr.get("stream_processing_mode"):
                agent._bedrock_guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
            if _gr.get("trace"):
                agent._bedrock_guardrail_config["trace"] = _gr["trace"]
    except Exception:
        pass
    agent.client = None
    agent._client_kwargs = {}
    if not agent.quiet_mode:
        _gr_label = " + Guardrails" if agent._bedrock_guardrail_config else ""
        print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock, {agent._bedrock_region}{_gr_label})")


def _explicit_client_kwargs(agent, api_key, base_url, _provider_timeout) -> Dict[str, Any]:
    """OpenAI-client kwargs from explicit CLI/gateway credentials (auth already resolved)."""
    _parsed_url = urlparse(base_url)
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if _parsed_url.query:
        client_kwargs["base_url"] = urlunparse(_parsed_url._replace(query=""))
        client_kwargs["default_query"] = {k: v[0] for k, v in parse_qs(_parsed_url.query).items()}
    if _provider_timeout is not None:
        client_kwargs["timeout"] = _provider_timeout
    if agent.provider == "copilot-acp":
        client_kwargs["command"] = agent.acp_command
        client_kwargs["args"] = agent.acp_args
    # OpenCode Zen free tier (*-free slugs): the relay serves these ANONYMOUSLY and 401s any
    # unrecognized bearer — including our keyless placeholder. Send an empty Authorization
    # header to override the SDK's "Bearer <key>".
    try:
        from hermes_cli.models import (
            OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, opencode_zen_free_headers
        )
        if api_key == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER:
            client_kwargs["default_headers"] = opencode_zen_free_headers()
    except Exception:
        pass
    _headers_for = _host_default_headers_factory(base_url)
    if _headers_for is not None:
        client_kwargs["default_headers"] = _headers_for(api_key, base_url)
    elif "default_headers" not in client_kwargs:
        # Fall back to profile.default_headers for providers that declare custom headers
        # (Vercel AI Gateway attribution, Kimi User-Agent on non-kimi.com endpoints).
        try:
            from providers import get_provider_profile as _gpf
            _ph = _gpf(agent.provider)
            if _ph and _ph.default_headers:
                client_kwargs["default_headers"] = dict(_ph.default_headers)
        except Exception:
            pass
    return client_kwargs


def _routed_client_kwargs(agent, fallback_model, _provider_timeout) -> Dict[str, Any]:
    """OpenAI-client kwargs via the centralized provider router (no explicit creds).

    Falls through to the init-time fallback chain, then raises with the missing-key /
    no-provider diagnostic.
    """
    from agent.auxiliary_client import resolve_provider_client
    _routed_client, _ = resolve_provider_client(
        agent.provider or "auto", model=agent.model, raw_codex=True)
    if _routed_client is not None:
        return _client_kwargs_from_routed(_routed_client, _provider_timeout)
    # No credentials for the configured provider: try the user-configured fallback chain
    # BEFORE failing, whichever provider failed (an exhausted single-entry pool must not die
    # with a misleading "No LLM provider configured"). Only explicitly named providers keep
    # the missing-key diagnostic.
    _explicit = (agent.provider or "").strip().lower()
    for _fb in _fallback_entries(fallback_model):
        try:
            from hermes_cli.fallback_config import resolve_entry_api_key
            _fb_explicit_key = resolve_entry_api_key(_fb)
            _fb_client, _fb_model = resolve_provider_client(
                _fb["provider"], model=_fb["model"], raw_codex=True,
                explicit_base_url=_fb.get("base_url"), explicit_api_key=_fb_explicit_key,
            )
        except Exception as _fb_exc:
            logger.debug("Init-time fallback entry %s failed: %s", _fb.get("provider"), _fb_exc)
            continue
        if _fb_client is not None:
            agent.provider = _fb["provider"]
            agent.model = _fb_model or _fb["model"]
            agent._fallback_activated = True
            return _client_kwargs_from_routed(_fb_client, _provider_timeout)
    if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
        # Explicit non-OpenRouter provider with no creds and no usable fallback: fail fast.
        # Use the provider's real env var name (alibaba → DASHSCOPE_API_KEY).
        _env_hint = f"{_explicit.upper()}_API_KEY"
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
            _pcfg = PROVIDER_REGISTRY.get(_explicit)
            if _pcfg and _pcfg.api_key_env_vars:
                _env_hint = _pcfg.api_key_env_vars[0]
        except Exception:
            pass
        raise RuntimeError(
            f"Provider '{_explicit}' is set in config.yaml but no API key "
            f"was found. Set the {_env_hint} environment "
            f"variable, or switch to a different provider with `hermes model`."
        )
    raise RuntimeError(
        "No LLM provider configured. Run `hermes model` to "
        "select a provider, or run `hermes setup` for first-time "
        "configuration."
    )


_FINE_GRAINED_BETA = "fine-grained-tool-streaming-2025-05-14"


def _init_openai_client(agent, api_key, base_url, fallback_model, _provider_timeout):
    """OpenAI-wire client: resolve kwargs, apply header/TLS policy, construct."""
    if api_key and base_url:
        client_kwargs = _explicit_client_kwargs(agent, api_key, base_url, _provider_timeout)
    else:
        client_kwargs = _routed_client_kwargs(agent, fallback_model, _provider_timeout)
    try:
        from agent.bedrock_adapter import configure_bedrock_openai_client_kwargs
        configure_bedrock_openai_client_kwargs(client_kwargs, timeout=_provider_timeout)
    except Exception:
        if agent.provider == "bedrock" and "bedrock-mantle." in str(client_kwargs.get("base_url", "")):
            raise

    agent._client_kwargs = client_kwargs  # stored for rebuilding after interrupt

    # Fine-grained tool streaming for Claude on OpenRouter: without the beta header
    # Anthropic buffers the whole tool call and OpenRouter's proxy times out.
    _effective_base = str(client_kwargs.get("base_url", "")).lower()
    if base_url_host_matches(_effective_base, "openrouter.ai") and "claude" in (agent.model or "").lower():
        headers = client_kwargs.get("default_headers") or {}
        existing_beta = headers.get("x-anthropic-beta", "")
        if _FINE_GRAINED_BETA not in existing_beta:
            headers["x-anthropic-beta"] = ",".join(filter(None, (existing_beta, _FINE_GRAINED_BETA)))
            client_kwargs["default_headers"] = headers

    # model.default_headers (config.yaml) override provider/SDK defaults (WAFs that reject
    # the SDK's identifying headers). Mutates agent._client_kwargs — this same dict.
    agent._apply_user_default_headers()

    try:
        from hermes_cli.config import (
            apply_custom_provider_extra_headers_to_client_kwargs,
            apply_custom_provider_tls_to_client_kwargs, get_compatible_custom_providers,
            load_config,
        )

        _cp_entries = get_compatible_custom_providers(load_config())
        _cp_base_url = str(client_kwargs.get("base_url") or agent.base_url or "")
        apply_custom_provider_tls_to_client_kwargs(client_kwargs, _cp_base_url, _cp_entries)
        # Per-provider extra HTTP headers (providers.<name>.extra_headers /
        # custom_providers[].extra_headers). Applied last so the most specific config level
        # wins. SECURITY: values may carry credentials — never log them.
        apply_custom_provider_extra_headers_to_client_kwargs(client_kwargs, _cp_base_url, _cp_entries)
    except Exception:
        logger.debug("custom-provider TLS resolution skipped", exc_info=True)

    agent.api_key = client_kwargs.get("api_key", "")
    agent.base_url = client_kwargs.get("base_url", agent.base_url)
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback

        verify_ca_bundle_with_fallback()
        agent.client = agent._create_openai_client(client_kwargs, reason="agent_init", shared=True)
        if not agent.quiet_mode:
            print(f"🤖 AI Agent initialized with model: {agent.model}")
            if base_url:
                print(f"🔗 Using custom base URL: {base_url}")
            _print_key_banner(client_kwargs.get("api_key", "none"), "API key", warn_missing=True)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")


def _build_client(agent, api_key, base_url, fallback_model):
    # LLM client per wire mode. The provider router handles auth, base URL, headers and
    # Codex/Anthropic wrapping (raw_codex=True: the main agent needs direct
    # responses.stream()). One provider/model timeout up front so every construction path
    # applies it consistently (Bedrock Claude has its own).
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    _provider_timeout = get_provider_request_timeout(agent.provider, agent.model)
    if agent.api_mode == "anthropic_messages":
        _init_anthropic_client(agent, api_key, base_url, _provider_timeout)
    elif agent.provider == "moa":
        _init_moa_client(agent, api_key)
    elif agent.api_mode == "bedrock_converse":
        _init_bedrock_client(agent, base_url)
    else:
        _init_openai_client(agent, api_key, base_url, fallback_model, _provider_timeout)


def _lazy_headers(module: str, name: str, pass_key: bool = False, pass_base: bool = False):
    """Header factory ``(api_key, base_url) -> dict`` importing ``module.name`` at call time.

    ``pass_key`` forwards ``(key, base_url=base)`` (Codex Cloudflare headers); ``pass_base``
    forwards ``(base)`` (NVIDIA NIM); neither forwards nothing.
    """
    def factory(key, base):
        import importlib
        fn = getattr(importlib.import_module(module), name)
        if pass_key:
            return fn(key, base_url=base)
        return fn(base) if pass_base else fn()
    return factory


# Host → default_headers factory for explicit base_url client construction. Ordered: first
# host match wins; no match falls back to the provider profile's declared headers. ``_ra()``
# keeps the ``run_agent.*`` helpers patchable by tests.
_HOST_DEFAULT_HEADERS: List[tuple[str, Callable[[Any, str], Dict[str, str]]]] = [
    ("openrouter.ai", _lazy_headers("agent.auxiliary_client", "build_or_headers")),
    ("integrate.api.nvidia.com",
     _lazy_headers("agent.auxiliary_client", "build_nvidia_nim_headers", pass_base=True)),
    ("api.routermint.com", lambda _k, _b: _ra()._routermint_headers()),
    ("githubcopilot.com", _lazy_headers("hermes_cli.models", "copilot_default_headers")),
    ("api.kimi.com", lambda _k, _b: {"User-Agent": "claude-code/0.1.0"}),
    ("portal.qwen.ai", lambda _k, _b: _ra()._qwen_portal_headers()),
    ("chatgpt.com", _lazy_headers("agent.codex_headers", "codex_cloudflare_headers", pass_key=True)),
    ("x.ai", _lazy_headers("tools.xai_http", "hermes_xai_default_headers")),
]


def _host_default_headers_factory(base_url: str):
    for host, factory in _HOST_DEFAULT_HEADERS:
        if base_url_host_matches(base_url, host):
            return factory
    return None


def _client_kwargs_from_routed(client, timeout) -> Dict[str, Any]:
    """OpenAI-client kwargs mirroring a router-resolved client.

    Preserves provider-specific headers the router set: the OpenAI SDK stores caller-provided
    default_headers in ``_custom_headers``; older/mocked clients may expose
    ``default_headers`` / ``_default_headers`` instead.
    """
    kwargs = {"api_key": client.api_key, "base_url": str(client.base_url)}
    if timeout is not None:
        kwargs["timeout"] = timeout
    headers = (
        getattr(client, "_custom_headers", None)
        or getattr(client, "default_headers", None)
        or getattr(client, "_default_headers", None)
    )
    if headers:
        kwargs["default_headers"] = dict(headers)
    return kwargs


def _fallback_entries(fallback_model) -> List[Dict[str, Any]]:
    """Normalize legacy single-dict ``fallback_model`` / list ``fallback_providers``."""
    if isinstance(fallback_model, dict):
        fallback_model = [fallback_model]
    if not isinstance(fallback_model, list):
        return []
    return [
        f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]


def _init_fallback_chain(agent, fallback_model):
    # Stable identity for the pool entry that supplied this runtime: OAuth refreshes can
    # replace the token before a failed request is recovered, so the mutable API-key value
    # alone cannot attribute the failure to its source entry.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    # Provider fallback chain — ordered backups tried when the primary is exhausted
    # (rate-limit, overload, connection failure). Legacy single-dict or list format.
    agent._fallback_chain = _fallback_entries(fallback_model)
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    if agent._fallback_chain and not agent.quiet_mode:
        if len(agent._fallback_chain) == 1:
            fb = agent._fallback_chain[0]
            print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
        else:
            print(f"🔄 Fallback chain ({len(agent._fallback_chain)} providers): " +
                  " → ".join(f"{f['model']} ({f['provider']})" for f in agent._fallback_chain))


def _load_tools(agent, enabled_toolsets, disabled_toolsets):
    # A multiplexed gateway may enter a different HERMES_HOME after ``model_tools`` was first
    # imported; ensure that profile's plugin manager has discovered its registrations first.
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception:
        logger.warning("Plugin discovery failed during agent setup", exc_info=True)

    # Capture the registry generation FIRST so a later concurrent refresh can tell whether
    # it holds a newer or staler view (see refresh_agent_mcp_tools).
    try:
        from tools.registry import registry as _snapshot_registry
        agent._tool_snapshot_generation = _snapshot_registry._generation
    except Exception:
        agent._tool_snapshot_generation = 0
    agent.tools = _ra().get_tool_definitions(
        enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
        quiet_mode=agent.quiet_mode,
    )

    agent.valid_tool_names = set()
    if agent.tools:
        agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
        tool_names = sorted(agent.valid_tool_names)
        if not agent.quiet_mode:
            print(f"🛠️  Loaded {len(agent.tools)} tools: {', '.join(tool_names)}")
            if enabled_toolsets:
                print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
            if disabled_toolsets:
                print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
    elif not agent.quiet_mode:
        print("🛠️  No tools loaded (all tools filtered out or unavailable)")

    # Kanban lifecycle guidance is session-static (kanban_show is present iff
    # HERMES_KANBAN_TASK is set); resolve the ~835-token block once, not per prompt rebuild.
    from agent.prompt_builder import KANBAN_GUIDANCE
    agent._kanban_worker_guidance = (
        KANBAN_GUIDANCE if "kanban_show" in agent.valid_tool_names else ""
    )

    if agent.quiet_mode:
        return
    if agent.tools:
        requirements = _ra().check_toolset_requirements()
        missing_reqs = [name for name, available in requirements.items() if not available]
        if missing_reqs:
            print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
    if agent.save_trajectories:
        print("📝 Trajectory saving enabled")
    if agent.ephemeral_system_prompt:
        prompt_preview = agent.ephemeral_system_prompt[:60] + "..." if len(agent.ephemeral_system_prompt) > 60 else agent.ephemeral_system_prompt
        print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
    if agent._use_prompt_caching:
        if agent._use_native_cache_layout and agent.provider == "anthropic":
            source = "native Anthropic"
        elif agent._use_native_cache_layout:
            source = "Anthropic-compatible endpoint"
        else:
            source = "Claude via OpenRouter"
        print(f"💾 Prompt caching: ENABLED ({source}, {agent._cache_ttl} TTL)")


def _publish_session_id(session_id: str) -> None:
    """Expose the session ID to tools (terminal, execute_code) via ContextVar + os.environ.

    Both are kept in sync because different tool paths still read both. If the ContextVar
    bridge fails to import, keep the root-agent legacy env fallback but never let delegated
    construction publish a child ID process-wide.
    """
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(session_id)
    except Exception:
        try:
            from agent.delegation_context import is_delegated_child_context

            delegated_child = is_delegated_child_context()
        except Exception:
            delegated_child = False
        if not delegated_child:
            os.environ["HERMES_SESSION_ID"] = session_id


def _init_session_state(agent, session_id, session_db, parent_session_id, reasoning_config, max_tokens,
    checkpoints_enabled, checkpoint_max_snapshots, checkpoint_max_total_size_mb, checkpoint_max_file_size_mb):
    agent.session_start = datetime.now()
    agent.session_id = session_id or (
        f"{agent.session_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    _publish_session_id(agent.session_id)

    # ~/.hermes/sessions/ — kept unconditionally for request_dump_*.json debug breadcrumbs.
    agent.logs_dir = get_hermes_home() / "sessions"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    # Per-session JSON snapshot is opt-in (sessions.write_json_snapshots); state.db is canonical.
    agent._session_json_enabled = False
    try:
        from hermes_cli.config import load_config_readonly as _load_sess_cfg
        _sess_cfg = (_load_sess_cfg().get("sessions") or {})
        agent._session_json_enabled = bool(_sess_cfg.get("write_json_snapshots", False))
    except Exception:
        pass

    agent._session_messages: List[Dict[str, Any]] = []
    # Responses encrypted-reasoning replay: routes that 400 with ``invalid_encrypted_content``
    # make the loop disable it for the session (stateless continuity).
    agent._codex_reasoning_replay_enabled = True
    agent._memory_write_origin = "assistant_tool"
    agent._memory_write_context = "foreground"
    # Cached system prompt (built once, rebuilt on compression) + its cross-session-stable
    # prefix, kept separately only to place an early cache marker.
    agent._cached_system_prompt: Optional[str] = None
    agent._cached_system_prompt_static: Optional[str] = None

    # Filesystem checkpoint manager (transparent — not a tool)
    from tools.checkpoint_manager import CheckpointManager
    agent._checkpoint_mgr = CheckpointManager(
        enabled=checkpoints_enabled, max_snapshots=checkpoint_max_snapshots,
        max_total_size_mb=checkpoint_max_total_size_mb,
        max_file_size_mb=checkpoint_max_file_size_mb,
    )

    # SQLite session store (optional; CLI/gateway-provided). _owns_session_db False: a
    # caller-supplied handle is usually the SHARED launch handle; DEDICATED handles set True.
    agent._session_db = session_db
    agent._owns_session_db = False
    agent._parent_session_id = parent_session_id
    # Close flush and turn-start flush can overlap; the durable marker lives on each message
    # dict, so its test-and-append is serialized per agent.
    agent._session_persist_lock = threading.RLock()
    # CLI's just-accepted user dict, reused by turn setup so its durable marker survives a
    # close-persistence race.
    agent._pending_cli_user_message = None
    agent._last_flushed_db_idx = 0  # DB-write cursor (prevents duplicate writes)
    agent._session_db_created = False  # DB row deferred to run_conversation()
    # False on helper agents (compression / hygiene / review forks) that hand the session to
    # a continuation row that must stay open.
    agent._end_session_on_close = True
    # True on the background review fork: never persist, so its harness turn can't hijack
    # the live session.
    agent._persist_disabled = False
    agent._session_init_model_config = {
        "max_iterations": agent.max_iterations,
        "reasoning_config": reasoning_config,
        "max_tokens": max_tokens,
    }
    # Process-scoped --yolo is persisted so `hermes --resume` restores the bypass
    # (SessionDB.session_yolo_enabled); session-scoped /yolo toggles persist separately.
    try:
        from tools.approval import _YOLO_MODE_FROZEN
        if _YOLO_MODE_FROZEN:
            agent._session_init_model_config["yolo_mode"] = True
    except Exception:
        pass

    # In-memory todo list for task planning (one per agent/session)
    from tools.todo_tool import TodoStore
    agent._todo_store = TodoStore()


def _apply_display_config(agent, _agent_cfg, platform):
    # display.show_commentary (default true): Codex phase=commentary messages go to the
    # interim message path; false routes them to the reasoning channel.
    agent.show_commentary = bool(_cfg_dict(_agent_cfg, "display").get("show_commentary", True))

    # Window (seconds) for the bounded /fast auto|cold modes (agent.fast_mode).
    agent.fast_auto_seconds = (_agent_cfg.get("agent") or {}).get("fast_auto_seconds", 60)

    # model.lmstudio_load_mode: "explicit" (default, preload via LM Studio's management API)
    # or "jit" (LM Studio just-in-time / Auto-Evict path).
    _model_section = _cfg_dict(_agent_cfg, "model")
    agent.lmstudio_load_mode = "explicit"
    _load_mode = str(_model_section.get("lmstudio_load_mode", "explicit") or "explicit").strip().lower()
    if _load_mode in {"explicit", "jit"}:
        agent.lmstudio_load_mode = _load_mode
    else:
        logger.warning(
            "Invalid model.lmstudio_load_mode=%r; expected 'explicit' or 'jit'. Using explicit.",
            _model_section.get("lmstudio_load_mode"),
        )

    # API-transport streaming (``model.streaming``, default true). Some self-hosted backends
    # have broken streaming tool-call paths, so ``false`` seeds ``_disable_streaming`` — the
    # same non-streaming path the loop falls back to at runtime. Session-scoped (survives
    # model switches); orthogonal to ``display.streaming``.
    agent._disable_streaming = False
    _streaming = str(_model_section.get("streaming", "true")).strip().lower()
    if _streaming in {"false", "0", "no", "off"}:
        agent._disable_streaming = True
    elif _streaming not in {"true", "1", "yes", "on"}:
        logger.warning(
            "Invalid model.streaming=%r; expected a boolean. Using streaming (default).",
            _model_section.get("streaming"),
        )

    try:
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                _agent_cfg.get("tool_loop_guardrails", {}), platform=platform,
            )
        )
    except Exception as _tlg_err:
        _ra().logger.warning("Tool loop guardrail config ignored: %s", _tlg_err)
    # Only the derived auxiliary compression context override is cached (needed by the
    # startup feasibility check) — no broad pseudo-public config object on the agent.
    agent._aux_compression_context_length_config = None


def _memory_provider_init_kwargs(agent, platform) -> Dict[str, Any]:
    """Scoping kwargs for ``MemoryManager.initialize_all``.

    status_callback (deterministic retain indicator) is CLI-only — gateway status travels a
    different path and the indicator no-ops without it.
    """
    kwargs = {
        "session_id": agent.session_id,
        "platform": platform or "cli",
        "hermes_home": str(get_hermes_home()),
        "agent_context": "primary",
    }
    if kwargs["platform"] == "cli":
        kwargs["warning_callback"] = agent._emit_warning
        kwargs["status_callback"] = agent._emit_status
    # Session title (e.g. honcho derives chat-scoped session keys from it).
    if agent._session_db:
        try:
            _st = agent._session_db.get_session_title(agent.session_id)
            if _st:
                kwargs["session_title"] = _st
        except Exception:
            pass
    # Gateway user/chat identity for per-user scoping (gateway_session_key: stable per-chat
    # Honcho session isolation).
    for _ident in (
        "user_id", "user_id_alt", "user_name", "chat_id", "chat_name",
        "chat_type", "thread_id", "gateway_session_key",
    ):
        _val = getattr(agent, f"_{_ident}")
        if _val:
            kwargs[_ident] = _val
    # Profile identity for per-profile provider scoping
    try:
        from hermes_cli.profiles import get_active_profile_name
        kwargs["agent_identity"] = get_active_profile_name()
        kwargs["agent_workspace"] = "hermes"
    except Exception:
        pass
    return kwargs


def _init_memory(agent, _agent_cfg, skip_memory, platform):
    # Persistent memory (MEMORY.md + USER.md) — loaded from disk
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    # skip_memory=True skips the external *provider*; enabled_toolsets=["memory"] still gets
    # the built-in store so the memory tool never sees store=None. A memory entry on
    # disabled_toolsets is not a request.
    _memory_toolset_requested = (
        "memory" in (agent.enabled_toolsets or [])
        and "memory" not in (agent.disabled_toolsets or [])
    )
    if not skip_memory or _memory_toolset_requested:
        try:
            from tools.memory_tool import (
                get_builtin_memory_config, get_builtin_memory_store_flags
            )

            mem_config = get_builtin_memory_config(_agent_cfg)
            agent._memory_enabled, agent._user_profile_enabled = get_builtin_memory_store_flags(
                _agent_cfg
            )
            agent._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
            if agent._memory_enabled or agent._user_profile_enabled:
                from tools.memory_tool import MemoryStore
                agent._memory_store = MemoryStore(
                    memory_char_limit=mem_config.get("memory_char_limit", 2200),
                    user_char_limit=mem_config.get("user_char_limit", 1375),
                    memory_enabled=agent._memory_enabled,
                    user_profile_enabled=agent._user_profile_enabled,
                )
                agent._memory_store.load_from_disk()
        except Exception:
            pass  # Memory is optional — don't break agent init

    # Memory provider plugin (external — one at a time, alongside built-in), selected by
    # memory.provider.
    agent._memory_manager = None
    if not skip_memory:
        try:
            _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

            if _mem_provider_name and _mem_provider_name.strip():
                from agent.memory_manager import MemoryManager as _MemoryManager
                from plugins.memory import load_memory_provider as _load_mem
                agent._memory_manager = _MemoryManager()
                _mp = _load_mem(_mem_provider_name)
                if _mp and _mp.is_available():
                    agent._memory_manager.add_provider(_mp)
                elif _mp is not None and _mem_provider_name not in _warned_unavailable_providers:
                    # unavailable_reason() reads config/probes importlib — skip it once warned.
                    try:
                        _unavailable_reason = _mp.unavailable_reason()
                    except Exception:
                        _unavailable_reason = ""
                    _warn_memory_provider_unavailable(_mem_provider_name, _unavailable_reason)
                if agent._memory_manager.providers:
                    agent._memory_manager.initialize_all(
                        **_memory_provider_init_kwargs(agent, platform)
                    )
                    _ra().logger.info("Memory provider '%s' activated", _mem_provider_name)
                else:
                    _ra().logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                    agent._memory_manager = None
        except Exception as _mpe:
            _ra().logger.warning("Memory provider plugin init failed: %s", _mpe)
            agent._memory_manager = None

    from agent.memory_manager import inject_memory_provider_tools as _inject_memory_provider_tools
    _inject_memory_provider_tools(agent)


def _apply_agent_section(agent, _agent_cfg):
    # Skills config: nudge interval for skill creation reminders
    agent._skill_nudge_interval = 10
    try:
        agent._skill_nudge_interval = int(_agent_cfg.get("skills", {}).get("creation_nudge_interval", 10))
    except Exception:
        pass

    _agent_section = _cfg_dict(_agent_cfg, "agent")
    # Tool-use enforcement: "auto" (default — hardcoded model list), true, false, or list of
    # substrings. Execution-discipline guidance: same shape against EXECUTION_GUIDANCE_MODELS,
    # independent of enforcement (injection gate in agent/system_prompt.py).
    agent._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")
    agent._execution_guidance = _agent_section.get("execution_guidance", "auto")

    # Wall-clock run budget from config — only when the constructor arg was not given.
    if agent.run_budget_seconds is None:
        agent.run_budget_seconds = _normalize_run_budget_seconds(
            _agent_section.get("run_budget_seconds")
        )

    # Empty-response retry guard (``agent.empty_response_guard``): tolerant resolution — a
    # malformed section falls back to schema defaults (guard on, $0.25 threshold).
    from agent.empty_response_guard import resolve_guard_settings
    (
        agent._empty_guard_enabled, agent._empty_guard_cost_threshold_usd
    ) = resolve_guard_settings(_agent_section.get("empty_response_guard"))

    # Intent-ack continuation: "auto" (default — codex_responses only), true (all api_modes),
    # false, or a list of model-name substrings; resolved in the loop's intent-ack block.
    agent._intent_ack_continuation = _agent_section.get("intent_ack_continuation", "auto")

    # Runtime anti-stall guards (identical-call loop-breaker notice + continue-intent
    # extension of empty-response recovery). Notice-only — never blocks a call.
    agent._stall_guards = bool(_agent_section.get("stall_guards", True))

    # Universal guidance toggles (ALL models, unlike enforcement): task-completion,
    # parallel-tool-call batching, and the local Python toolchain probe.
    agent._task_completion_guidance = bool(_agent_section.get("task_completion_guidance", True))
    agent._parallel_tool_call_guidance = bool(_agent_section.get("parallel_tool_call_guidance", True))
    agent._environment_probe = bool(_agent_section.get("environment_probe", True))
    # Warm the probe off-thread (~0.5s of subprocesses) so the FIRST system-prompt build — on
    # the time-to-first-token path — finds the line already cached.
    if agent._environment_probe:
        try:
            from tools.env_probe import warm_environment_probe_async
            warm_environment_probe_async()
        except Exception:
            pass

    # Bot Mode teammate protocol section (tools/bot_mode_probe.py) — pure filesystem reads.
    agent._bot_mode_protocol = bool(_agent_section.get("bot_mode_protocol", True))
    # Session-title hint for the "Bot Chat" gate: hosts that defer the DB title write past
    # the first prompt build (tui_gateway pending_title) set this.
    agent._session_title_hint = None

    # Per-platform prompt-hint overrides (platform_hints: <platform>: {append|replace}),
    # stored verbatim; resolved in agent/system_prompt.py. Invalid shapes are ignored.
    agent._platform_hint_overrides = _cfg_dict(_agent_cfg, "platform_hints")

    # App-level API retry count (wraps each model API call). Default 3; 1 = single attempt.
    try:
        _api_retries = max(int(_agent_section.get("api_max_retries", 3)), 1)
    except (TypeError, ValueError):
        _api_retries = 3
    agent._api_max_retries = _api_retries


def _positive_int(raw: Any, *, reject: tuple = ()) -> Optional[int]:
    """``int(raw)`` when positive, else None. ``reject`` lists types refused outright (bool, float)."""
    if reject and isinstance(raw, reject):
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _compression_threshold(agent, cfg: Dict[str, Any]) -> tuple[float, bool]:
    """Global threshold merged with the per-model override; stashes the autoraise notice.

    Codex gpt-5.4/5.5 raise to 85% (backend caps at 272K, so 50% would compact at ~136K).
    The opt-out flag restores the global threshold; when the raise fires a one-time notice
    is stashed on the agent for the first turn, with its own display gate.
    """
    threshold = float(cfg.get("threshold", 0.50))
    autoraise = _cfg_flag(cfg, "codex_gpt55_autoraise", True)
    notice_enabled = _cfg_flag(cfg, "codex_gpt55_autoraise_notice", True)
    agent._compression_threshold_autoraised = None
    try:
        from agent.auxiliary_client import (
            _compression_threshold_for_model as _cthresh_fn,
            _is_codex_gpt54_or_gpt55 as _is_codex_gpt54_or_gpt55_fn,
            _is_codex_spark as _is_codex_spark_fn,
        )
        _model_cthresh = _cthresh_fn(
            agent.model, agent.provider, allow_codex_gpt55_autoraise=autoraise,
        )
        # Codex autoraises apply only when they RAISE; Arcee Trinity keeps its
        # unconditional override.
        threshold, agent._compression_threshold_autoraised = _resolve_compression_threshold(
            threshold,
            _model_cthresh,
            model=agent.model,
            is_codex_autoraise=(
                _is_codex_gpt54_or_gpt55_fn(agent.model, agent.provider)
                or _is_codex_spark_fn(agent.model, agent.provider)
            ),
        )
    except Exception:
        pass
    return threshold, notice_enabled


def _compression_codex_settings(cfg: Dict[str, Any]) -> tuple[str, bool, Optional[int]]:
    """``codex_app_server_auto`` / ``codex_responses_native`` / ``codex_responses_compact_threshold``."""
    app_server_auto = str(cfg.get("codex_app_server_auto", "native") or "native").lower()
    if app_server_auto not in {"native", "hermes", "off"}:
        _ra().logger.warning(
            "Invalid compression.codex_app_server_auto=%r; using 'native'. "
            "Valid values are: native, hermes, off.",
            app_server_auto,
        )
        app_server_auto = "native"
    # Native OpenAI Responses server-side compaction (opt-in; per-request gate in
    # agent/native_compaction.py). Truthy coercion: "false"/"off" strings stay disabled.
    responses_native = is_truthy_value(cfg.get("codex_responses_native", False))
    _raw = cfg.get("codex_responses_compact_threshold")
    compact_threshold = None
    if _raw is not None:
        compact_threshold = _positive_int(_raw, reject=(bool, float))
        if compact_threshold is None:
            _ra().logger.warning(
                "Invalid compression.codex_responses_compact_threshold=%r; "
                "using the automatic threshold derived from local compression.",
                _raw,
            )
    return app_server_auto, responses_native, compact_threshold


def _parse_compression_config(agent, _agent_cfg) -> CompressionSettings:
    """Parse the ``compression`` section. Defaults here MUST match DEFAULT_CONFIG."""
    cfg = _cfg_dict(_agent_cfg, "compression")
    threshold, autoraise_notice_enabled = _compression_threshold(agent, cfg)
    # Plain int()/float() coercions raise on garbage; evaluated up front, in config order.
    target_ratio = float(cfg.get("target_ratio", 0.20))
    protect_last = int(cfg.get("protect_last_n", 20))
    # max_attempts: retry rounds before "max compression attempts reached"; some sessions
    # need >3 (incompressible tool schemas). Default 3, floor 1, cap 10.
    max_attempts = _parse_config_int(cfg.get("max_attempts", 3), 3)
    if max_attempts < 1:
        max_attempts = 3
    # threshold_tokens: absolute cap — compression triggers at the lower of the ratio
    # threshold and this count; clamped to the window at apply-time (cap above window = no-op).
    threshold_tokens = cfg.get("threshold_tokens")
    if threshold_tokens is not None:
        threshold_tokens = _positive_int(threshold_tokens)
    # Non-system head messages to protect (system prompt is always protected); 0 is a
    # legitimate "system prompt + summary + tail".
    protect_first = max(0, int(cfg.get("protect_first_n", 3)))
    checkpoint_required = is_truthy_value(cfg.get("checkpoint_required"), default=False)
    _refuse_checkpoint_required_on_codex_app_server(
        checkpoint_required, getattr(agent, "api_mode", None)
    )
    app_server_auto, responses_native, compact_threshold = _compression_codex_settings(cfg)
    # Opt-in idle compaction: compact up front when a session resumes after this many
    # seconds idle (0 = disabled). Consumed by build_turn_context().
    idle_compact_after_seconds = max(0, int(cfg.get("idle_compact_after_seconds", 0)))
    return CompressionSettings(
        threshold=threshold,
        autoraise_notice_enabled=autoraise_notice_enabled,
        enabled=_cfg_flag(cfg, "enabled", True),
        target_ratio=target_ratio,
        protect_last=protect_last,
        # tail_mode: "lean" (default) keeps a clamped 2.5%/10K-25K verbatim tail — continuity
        # rides the summary. "legacy" restores the 0.20*threshold tail (hoards 100-240K on
        # big windows). Unknown values fall back to lean inside the compressor.
        tail_mode=str(cfg.get("tail_mode", "lean")).strip().lower(),
        # Actionable user messages guaranteed to survive in the tail (default 1, floor 1).
        min_tail_users=max(1, _parse_config_int(cfg.get("min_tail_user_messages", 1), 1)),
        max_attempts=min(max_attempts, 10),
        # Opt-in proactive tool-result prune trigger (0 = disabled; negatives = disabled).
        proactive_prune_tokens=max(0, _parse_config_int(cfg.get("proactive_prune_tokens", 0), 0)),
        proactive_prune_min_chars=_parse_config_int(
            cfg.get("proactive_prune_min_result_chars", 8000), 8000
        ),
        proactive_prune_min_reclaim=max(
            0, _parse_config_int(cfg.get("proactive_prune_min_reclaim_tokens", 4096), 4096)
        ),
        protect_first=protect_first,
        abort_on_summary_failure=_cfg_flag(cfg, "abort_on_summary_failure", False),
        # Per-model threshold overrides: keys substring-matched against the model name
        # (longest match wins); {} = global threshold for all models.
        model_thresholds={
            str(k): float(v) for k, v in _cfg_dict(cfg, "model_thresholds").items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        threshold_tokens=threshold_tokens,
        checkpoint_required=checkpoint_required,
        # In-place compaction rewrites messages + system prompt WITHOUT rotating the session
        # id. default=True MUST match DEFAULT_CONFIG — a False default flipped agents into
        # rotation mode whenever the merged config omitted the key.
        in_place=is_truthy_value(cfg.get("in_place"), default=True),
        # Opt-in (default False): micro-compaction rewrites already-sent history per turn,
        # breaking the prompt-cache prefix on a per-turn cadence.
        micro_compact=is_truthy_value(cfg.get("micro_compact"), default=False),
        # Pass cadence in completed turns; each pass costs one prompt-cache break (>= 1).
        micro_compact_every_n_turns=max(
            1, _parse_config_int(cfg.get("micro_compact_every_n_turns", 1), 1)
        ),
        # Rolling-summary defrag threshold, in tokens.
        micro_compact_defrag_tokens=max(
            1, _parse_config_int(cfg.get("micro_compact_defrag_threshold_tokens", 2000), 2000)
        ),
        codex_app_server_auto=app_server_auto,
        codex_responses_native=responses_native,
        codex_responses_compact_threshold=compact_threshold,
        idle_compact_after_seconds=idle_compact_after_seconds,
    )


def _warn_invalid_config_int(
    what: str, value: Any, requirement: str, fallback: str, print_fallback: str = "",
) -> None:
    """Log + stderr-print an invalid integer config value.

    ``print_fallback`` lets the user-facing line keep its historical wording where it
    differs from the log line.
    """
    _ra().logger.warning(
        "Invalid %s: %r — %s. Falling back to %s.", what, value, requirement, fallback,
    )
    print(
        f"\n⚠ Invalid {what}: {value!r}\n"
        f"  {requirement[0].upper() + requirement[1:]}.\n"
        f"  Falling back to {print_fallback or fallback}.\n",
        file=sys.stderr,
    )


def _custom_provider_configured_base_url(
    _configured_provider: str, _agent_cfg, _custom_providers
) -> str:
    """Base URL of a named custom provider (``providers.<name>`` first, then
    ``custom_providers``), normalized for route comparison; "" if unknown.
    Disabled ``providers.*`` entries also mask their ``custom_providers`` twin.
    """
    _wanted = _normalize_custom_provider_name(_configured_provider)
    _user_providers = _agent_cfg.get("providers")
    _disabled_ids: set[str] = set()
    if isinstance(_user_providers, dict):
        from hermes_cli.config import is_provider_enabled

        for _key, _entry in _user_providers.items():
            if not isinstance(_entry, dict):
                continue
            _ids = _custom_provider_runtime_ids(_key) | _custom_provider_runtime_ids(_entry.get("name"))
            if not is_provider_enabled(_entry):
                _disabled_ids.update(_ids)
                continue
            if _wanted in _ids:
                _url = _normalize_route_base_url(
                    _entry.get("api") or _entry.get("url") or _entry.get("base_url")
                )
                if _url:
                    return _url
    for _entry in _custom_providers:
        if not isinstance(_entry, dict):
            continue
        _key_ids = _custom_provider_runtime_ids(_entry.get("provider_key"))
        if _key_ids & _disabled_ids:
            continue
        if _wanted in _key_ids | _custom_provider_runtime_ids(_entry.get("name")):
            _url = _normalize_route_base_url(_entry.get("base_url"))
            if _url:
                return _url
    return ""


# Provider ids whose runtime is resolved first-hand (never a named custom provider).
_RUNTIME_FIRST_PROVIDER_IDS = {
    "auto", "moa", "vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai",
}


def _configured_default_base_url(_agent_cfg, _model_cfg, _custom_providers) -> str:
    """Normalized route of the configured default model (``model.base_url``, else the named
    custom provider's URL when ``model.provider`` is not a first-class/auth provider)."""
    _configured_base_url = _normalize_route_base_url(_model_cfg.get("base_url"))
    _configured_provider = str(_model_cfg.get("provider") or "").strip()
    _norm = _normalize_custom_provider_name(_configured_provider)
    _custom_provider_candidate = bool(_norm)
    if _norm in _RUNTIME_FIRST_PROVIDER_IDS:
        _custom_provider_candidate = False
    elif _custom_provider_candidate and _norm != "custom" and not _norm.startswith("custom:"):
        try:
            from hermes_cli.auth import resolve_provider as resolve_auth_provider

            _custom_provider_candidate = (
                str(resolve_auth_provider(_norm) or "").strip().lower() != _norm
            )
        except Exception:
            pass
    if not _configured_base_url and _custom_provider_candidate:
        _configured_base_url = _custom_provider_configured_base_url(
            _configured_provider, _agent_cfg, _custom_providers
        )
    return _configured_base_url


def _active_route_url(agent, base_url) -> str:
    """The runtime route, keeping the requested URL's query string when it is the same route."""
    _active_route_url = str(agent.base_url or "")
    _requested_route_url = str(base_url or "")
    if "?" in _requested_route_url.split("#", 1)[0]:
        try:
            _requested_without_query = urlunparse(
                urlparse(_requested_route_url)._replace(query="")
            )
            if _normalize_route_base_url(
                _requested_without_query
            ) == _normalize_route_base_url(_active_route_url):
                _active_route_url = _requested_route_url
        except (TypeError, ValueError):
            pass
    return _normalize_route_base_url(_active_route_url)


def _scope_context_length_to_default_runtime(
    agent, _agent_cfg, _model_cfg, _custom_providers, _config_context_length, base_url
) -> Optional[int]:
    """Return ``model.context_length`` only if it describes the active runtime.

    ``model.context_length`` describes the configured default model. A process launched
    with ``--model`` has already replaced ``agent.model`` before config loads, so carrying
    the default model's explicit window into that runtime is stale. Live switch/fallback
    paths already clear this override; direct-start stays consistent with them and lets
    provider metadata resolve the active model's window.
    """
    _default = _model_cfg.get("default")
    if isinstance(_default, dict):
        from hermes_cli.config import split_model_config_default
        _default, _ = split_model_config_default(_default)
    _configured_default_model = str(_default or "").strip()
    _configured_default_runtime_model = _configured_default_model
    _active_runtime_model = agent.model
    if _configured_default_model:
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider

            _configured_default_runtime_model = normalize_model_for_provider(
                _configured_default_model, agent.provider
            )
            _active_runtime_model = normalize_model_for_provider(agent.model, agent.provider)
        except Exception:
            pass
    _configured_base_url = _configured_default_base_url(_agent_cfg, _model_cfg, _custom_providers)
    _active_base_url = _active_route_url(agent, base_url)
    _route_mismatch = _context_route_mismatch(
        _configured_base_url, _active_base_url, str(_model_cfg.get("provider") or "").strip(),
        agent.provider, already_normalized=True,
    )
    _model_mismatch = bool(
        _configured_default_runtime_model
        and _configured_default_runtime_model != _active_runtime_model
    )
    if _model_mismatch or _route_mismatch:
        _ra().logger.debug(
            "Ignoring model.context_length=%s for startup runtime %s at %s "
            "(configured default is %s at %s)",
            _config_context_length,
            agent.model,
            _active_base_url or agent.provider,
            _configured_default_model,
            _configured_base_url or _model_cfg.get("provider"),
        )
        return None
    return _config_context_length


_CTX_LEN_REQUIREMENT = "must be a positive integer (e.g. 256000, not '256K')"


def _warn_invalid_custom_provider_context_length(agent, _custom_providers) -> None:
    """Surface a context_length the helper silently skipped (not a positive int)."""
    _target = _normalize_route_base_url(agent.base_url)
    if not _target:
        return
    for _cp_entry in _custom_providers:
        if not isinstance(_cp_entry, dict):
            continue
        if _normalize_route_base_url(_cp_entry.get("base_url")) != _target:
            continue
        _cp_models = _cp_entry.get("models", {})
        _cp_model_cfg = _cp_models.get(agent.model, {}) if isinstance(_cp_models, dict) else None
        _cp_ctx = _cp_model_cfg.get("context_length") if isinstance(_cp_model_cfg, dict) else None
        if _cp_ctx is not None and _positive_int(_cp_ctx) is None:
            _warn_invalid_config_int(
                f"context_length for model {agent.model!r} in custom_providers",
                _cp_ctx, _CTX_LEN_REQUIREMENT, "auto-detection", "auto-detected context window",
            )
        return


def _resolve_context_length(agent, _agent_cfg, base_url):
    # Explicit context_length for the auxiliary compression model: custom endpoints often
    # can't report it via /models, so the startup feasibility check needs the hint.
    try:
        _aux_cfg = cfg_get(_agent_cfg, "auxiliary", "compression", default={})
    except Exception:
        _aux_cfg = {}
    _aux_ctx = _aux_cfg.get("context_length") if isinstance(_aux_cfg, dict) else None
    try:
        agent._aux_compression_context_length_config = int(_aux_ctx) if _aux_ctx is not None else None
    except (TypeError, ValueError):
        agent._aux_compression_context_length_config = None

    # model.max_tokens from config when the caller did not pass one.
    _model_cfg = _agent_cfg.get("model", {})
    if agent.max_tokens is None and isinstance(_model_cfg, dict):
        _config_max_tokens = _model_cfg.get("max_tokens")
        if _config_max_tokens is not None:
            agent.max_tokens = _positive_int(_config_max_tokens, reject=(bool,))
            if agent.max_tokens is None:
                _warn_invalid_config_int(
                    "model.max_tokens in config.yaml", _config_max_tokens,
                    "must be a positive integer (e.g. 4096)", "provider default",
                )
    agent._session_init_model_config["max_tokens"] = agent.max_tokens

    _config_context_length = _model_cfg.get("context_length") if isinstance(_model_cfg, dict) else None
    if _config_context_length is not None:
        try:
            _config_context_length = int(_config_context_length)
        except (TypeError, ValueError):
            _warn_invalid_config_int(
                "model.context_length in config.yaml", _config_context_length,
                "must be a plain integer (e.g. 256000, not '256K')",
                "auto-detection", "auto-detected context window",
            )
            _config_context_length = None

    # Resolve custom_providers once before route-scoping a global context pin: a named custom
    # provider may keep its base URL only in this list.
    try:
        from hermes_cli.config import get_compatible_custom_providers
        _custom_providers = get_compatible_custom_providers(_agent_cfg)
    except Exception:
        _custom_providers = _agent_cfg.get("custom_providers")
        if not isinstance(_custom_providers, list):
            _custom_providers = []

    # ``model.context_length`` describes the configured default model; drop it when the
    # startup runtime (model or route) differs from that default.
    if _config_context_length is not None and isinstance(_model_cfg, dict):
        _config_context_length = _scope_context_length_to_default_runtime(
            agent, _agent_cfg, _model_cfg, _custom_providers, _config_context_length, base_url
        )

    # Reused by _check_compression_model_feasibility (aux compression model detection).
    agent._custom_providers = _custom_providers
    _merge_custom_provider_extra_body(agent, _custom_providers)

    if _config_context_length is None and _custom_providers:
        try:
            from hermes_cli.config import get_custom_provider_context_length
            _cp_ctx_resolved = get_custom_provider_context_length(
                model=agent.model, base_url=agent.base_url, custom_providers=_custom_providers
            )
            if _cp_ctx_resolved:
                _config_context_length = int(_cp_ctx_resolved)
        except Exception:
            pass
        if _config_context_length is None:
            _warn_invalid_custom_provider_context_length(agent, _custom_providers)

    # Persist for switch_model / fallback activation — AFTER the custom_providers branch so
    # per-model overrides aren't lost.
    agent._config_context_length = _config_context_length

    _lmstudio_runtime_context_length = agent._ensure_lmstudio_runtime_loaded(_config_context_length)
    if agent._lmstudio_load_was_unverified(_lmstudio_runtime_context_length):
        _ra().logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length; falling back to configured context"
        )
    _effective_context_length = agent._effective_lmstudio_context_length(
        _config_context_length, _lmstudio_runtime_context_length,
    )
    return _config_context_length, _custom_providers, _effective_context_length, _model_cfg


def _select_context_engine(_agent_cfg):
    """Config-driven context engine: ``context.engine`` → plugins/context_engine/<name>/ →
    general plugin system → None (built-in ContextCompressor)."""
    _engine_name = "compressor"
    try:
        _engine_name = _agent_cfg.get("context", {}).get("engine", "compressor") or "compressor"
    except Exception:
        pass
    if _engine_name == "compressor":
        return None  # built-in; don't auto-activate plugins
    _selected_engine = None
    _copy_failed = False
    try:
        from plugins.context_engine import load_context_engine
        _selected_engine = load_context_engine(_engine_name)
    except Exception as _ce_load_err:
        _ra().logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

    if _selected_engine is None:
        try:
            from hermes_cli.plugins import get_plugin_context_engine
            _candidate = get_plugin_context_engine()
        except Exception:
            _candidate = None
        if _candidate is not None and _candidate.name == _engine_name:
            # Deep-copy the shared plugin singleton so a child's update_model() can't mutate
            # the parent's compressor. Uncopyable state (locks, DB conns) → built-in
            # compressor with an ACCURATE message, not "not found".
            import copy
            try:
                _selected_engine = copy.deepcopy(_candidate)
            except Exception as _copy_err:
                _copy_failed = True
                _ra().logger.warning(
                    "Context engine '%s' could not be safely copied for this "
                    "agent (%s) — falling back to built-in compressor. Plugin "
                    "engines that hold uncopyable state (locks, DB connections) "
                    "should implement __deepcopy__ to copy only mutable budget "
                    "state.",
                    _engine_name, _copy_err,
                )

    if _selected_engine is None and not _copy_failed:
        _ra().logger.warning(
            "Context engine '%s' not found — falling back to built-in compressor", _engine_name
        )
    return _selected_engine


def _compressor_max_tokens(agent):
    """``agent.max_tokens``, or the native-Gemini adapter default when unset.

    With model.max_tokens unset the generateContent adapter still sends maxOutputTokens=65,535
    and the threshold is pct×(window − max_tokens): reserving 0 here let the provider 400
    before compaction fired, so mirror the adapter's default (native Gemini only).
    """
    if agent.max_tokens is not None:
        return agent.max_tokens
    try:
        from agent.gemini_native_adapter import (
            GEMINI_DEFAULT_MAX_OUTPUT_TOKENS, is_native_gemini_base_url
        )
        _gemini_provider = str(getattr(agent, "provider", "") or "").strip().lower() in {
            "gemini", "google", "google-gemini", "google-ai-studio",
        }
        if _gemini_provider or is_native_gemini_base_url(agent.base_url):
            return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    except Exception:
        pass
    return None


def _build_context_engine(agent, _agent_cfg, cs, _custom_providers, _effective_context_length, session_db):
    _selected_engine = _select_context_engine(_agent_cfg)
    if _selected_engine is not None:
        agent.context_compressor = _selected_engine
        # External engines own compaction policy — the host threshold (and its Codex
        # autoraise) never reaches the plugin, so drop the notice.
        agent._compression_threshold_autoraised = None
        from agent.model_metadata import get_model_context_length
        _plugin_ctx_len = get_model_context_length(
            agent.model, base_url=agent.base_url, api_key=getattr(agent, "api_key", ""),
            config_context_length=_effective_context_length, provider=agent.provider,
            custom_providers=_custom_providers,
        )
        # Per-model overrides BEFORE the initial update_model() so the first threshold
        # resolution already sees them.
        if cs.model_thresholds:
            agent.context_compressor.model_thresholds = cs.model_thresholds
        agent.context_compressor.update_model(
            model=agent.model, context_length=_plugin_ctx_len, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )
        if not agent.quiet_mode:
            _ra().logger.info("Using context engine: %s", _selected_engine.name)
    else:
        agent.context_compressor = ContextCompressor(
            model=agent.model, threshold_percent=cs.threshold, protect_first_n=cs.protect_first,
            protect_last_n=cs.protect_last, summary_target_ratio=cs.target_ratio,
            summary_model_override=None, quiet_mode=agent.quiet_mode, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), config_context_length=_effective_context_length,
            provider=agent.provider, api_mode=agent.api_mode,
            abort_on_summary_failure=cs.abort_on_summary_failure,
            max_tokens=_compressor_max_tokens(agent), model_thresholds=cs.model_thresholds,
            threshold_tokens_cap=cs.threshold_tokens,
            proactive_prune_tokens=cs.proactive_prune_tokens,
            proactive_prune_min_result_chars=cs.proactive_prune_min_chars,
            proactive_prune_min_reclaim_tokens=cs.proactive_prune_min_reclaim,
            min_tail_user_messages=cs.min_tail_users, tail_mode=cs.tail_mode,
        )
    _bind_session_state = getattr(agent.context_compressor, "bind_session_state", None)
    if callable(_bind_session_state):
        try:
            _bind_session_state(session_db=session_db, session_id=agent.session_id)
        except Exception:
            pass
    agent.compression_enabled = cs.enabled
    agent.compression_in_place = cs.in_place
    _cc = agent.context_compressor
    # checkpoint_required: micro-compaction is a lossy rewrite with no pre-compress
    # checkpoint hook, so suppress it while the gate is armed (mirrors native_compaction.py).
    if cs.checkpoint_required and cs.micro_compact:
        logger.warning(
            "compression.checkpoint_required is enabled: post-turn "
            "micro-compaction is disabled for this agent so every lossy "
            "rewrite passes through the checkpoint-gated compressor."
        )
        cs.micro_compact = False
    if hasattr(_cc, "_micro_compact_enabled"):
        _cc._micro_compact_enabled = cs.micro_compact
    if hasattr(_cc, "_micro_compact_every_n_turns"):
        _cc._micro_compact_every_n_turns = cs.micro_compact_every_n_turns
    if hasattr(_cc, "_micro_compact_defrag_threshold_tokens"):
        _cc._micro_compact_defrag_threshold_tokens = cs.micro_compact_defrag_tokens
    agent.compression_checkpoint_required = cs.checkpoint_required
    agent.codex_app_server_auto_compaction = cs.codex_app_server_auto
    agent.codex_responses_native_compaction = cs.codex_responses_native
    agent.codex_responses_compact_threshold = cs.codex_responses_compact_threshold
    from agent.native_compaction import resolve_native_compaction_capabilities
    agent.runtime_capabilities = resolve_native_compaction_capabilities(
        model=agent.model, base_url=agent.base_url, provider=agent.provider,
        is_codex_backend=(agent.provider or "").strip().lower() == "openai-codex",
    )
    agent.max_compression_attempts = cs.max_attempts
    agent.compression_idle_compact_after_seconds = cs.idle_compact_after_seconds


def _enforce_minimum_context(agent):
    # Reject windows below the 64K floor needed for reliable tool-calling; an explicit
    # positive model.context_length on LM Studio is allowed below the floor.
    _ctx = getattr(agent.context_compressor, "context_length", 0)
    _allow_lmstudio_explicit_below_floor = (
        str(getattr(agent, "provider", "") or "").strip().lower() == "lmstudio"
        and isinstance(agent._config_context_length, int)
        and not isinstance(agent._config_context_length, bool)
        and agent._config_context_length > 0
    )
    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not _allow_lmstudio_explicit_below_floor:
        raise ValueError(
            f"Model {agent.model} has a context window of {_ctx:,} tokens, "
            f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
            f"by Hermes Agent.  Choose a model with at least "
            f"{MINIMUM_CONTEXT_LENGTH // 1000}K context.  If your server "
            f"reports a window smaller than the model's true window, set "
            f"model.context_length in config.yaml to the real value "
            f"(this must be at least {MINIMUM_CONTEXT_LENGTH // 1000}K)."
        )


def _warn_nonagentic_hermes_model(agent):
    # Nous Hermes 3/4 are chat models, not tool-call-tuned. cli.py show_banner() already
    # warns on the CLI, so skip platform=="cli"; non-quiet non-CLI surfaces still get it.
    if agent.quiet_mode or (agent.platform or "cli") == "cli":
        return
    try:
        from hermes_cli.model_switch import _check_hermes_model_warning

        _hermes_warn = _check_hermes_model_warning(agent.model or "")
        if _hermes_warn:
            _user_msg = (
                "⚠ Nous Research Hermes 3 & 4 models are NOT agentic — they "
                "lack reliable tool-calling for agent workflows (delegation, "
                "cron, proactive tools). Consider an agentic model instead "
                "(Claude, GPT, Gemini, Qwen-Coder, etc.)."
            )
            if hasattr(agent, "_emit_warning"):
                agent._emit_warning(_user_msg)
            else:
                print(f"\n{_user_msg}\n", file=sys.stderr)
            _ra().logger.warning(_hermes_warn)
    except Exception:
        pass


def _inject_context_engine_tools(agent):
    # Inject context engine tool schemas (lcm_grep, lcm_describe, lcm_expand). Dedup against
    # existing names: plugin paths may register the same schemas via ctx.register_tool(), and
    # a duplicate trips provider-side 'duplicate tool name' errors. Gated on enabled_toolsets
    # like memory-provider tools so `platform_toolsets: telegram: []` can't leak lcm_* tools.
    agent._context_engine_tool_names: set = set()
    if (
        agent.context_compressor
        and agent.tools is not None
        and (agent.enabled_toolsets is None or "context_engine" in agent.enabled_toolsets)
    ):
        _existing_tool_names = {
            t.get("function", {}).get("name") for t in agent.tools if isinstance(t, dict)
        }
        from agent.memory_manager import normalize_tool_schema as _normalize_tool_schema
        for _raw_schema in agent.context_compressor.get_tool_schemas():
            _schema = _normalize_tool_schema(_raw_schema)
            if _schema is None:
                # A nameless tool makes strict providers 400 and disables the whole toolset.
                _ra().logger.warning(
                    "Context engine returned a tool schema with no resolvable "
                    "name; skipping to avoid poisoning the request (%r)",
                    _raw_schema,
                )
                continue
            _tname = _schema["name"]
            if _tname in _existing_tool_names:
                continue  # already registered via plugin/cache path
            agent.tools.append({"type": "function", "function": _schema})
            agent.valid_tool_names.add(_tname)
            agent._context_engine_tool_names.add(_tname)
            _existing_tool_names.add(_tname)

    if agent.context_compressor:
        try:
            agent.context_compressor.on_session_start(
                agent.session_id, hermes_home=str(get_hermes_home()),
                platform=agent.platform or "cli", model=agent.model,
                context_length=getattr(agent.context_compressor, "context_length", 0),
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
        except Exception as _ce_err:
            _ra().logger.debug("Context engine on_session_start: %s", _ce_err)


def _configure_ollama_num_ctx(agent, _model_cfg, _config_context_length):
    # Ollama defaults num_ctx to 2048 regardless of model, so detect the max window and pass
    # num_ctx on every request. model.ollama_num_ctx overrides; model.context_length caps the
    # detected value (VRAM budget).
    agent._ollama_num_ctx: int | None = None
    _override = _model_cfg.get("ollama_num_ctx") if isinstance(_model_cfg, dict) else None
    if _override is not None:
        try:
            agent._ollama_num_ctx = int(_override)
        except (TypeError, ValueError):
            _ra().logger.debug("Invalid ollama_num_ctx config value: %r", _override)
    if agent._ollama_num_ctx is None and agent.base_url and is_local_endpoint(agent.base_url):
        try:
            # api_key may be a callable (Entra token provider); detection needs a string.
            _key = agent.api_key if isinstance(agent.api_key, str) else ""
            _detected = query_ollama_num_ctx(agent.model, agent.base_url, api_key=_key or "")
            if _detected and _detected > 0:
                agent._ollama_num_ctx = _detected
        except Exception as exc:
            _ra().logger.debug("Ollama num_ctx detection failed: %s", exc)
    # Cap auto-detected num_ctx to the explicit context_length (GGUF metadata can advertise
    # 256K+ and Ollama would allocate that much VRAM); never override an explicit num_ctx.
    if (
        agent._ollama_num_ctx
        and _config_context_length
        and _override is None
        and agent._ollama_num_ctx > _config_context_length
    ):
        _ra().logger.info(
            "Ollama num_ctx capped: %d -> %d (model.context_length override)",
            agent._ollama_num_ctx, _config_context_length,
        )
        agent._ollama_num_ctx = _config_context_length
    if agent._ollama_num_ctx and not agent.quiet_mode:
        _ra().logger.info(
            "Ollama num_ctx: will request %d tokens (model max from /api/show)",
            agent._ollama_num_ctx,
        )
    # Recalibrate the compressor to the served window: every request runs at num_ctx, so a
    # trigger derived from the probed model window could sit above it and never fire.
    _cc_window = getattr(agent.context_compressor, "context_length", 0) or 0
    if agent._ollama_num_ctx and agent._ollama_num_ctx > 0 and _cc_window and agent._ollama_num_ctx < _cc_window:
        _ra().logger.info(
            "Compressor window clamped to Ollama num_ctx: %d -> %d",
            _cc_window, agent._ollama_num_ctx,
        )
        agent.context_compressor.update_model(
            model=agent.model, context_length=agent._ollama_num_ctx, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )


def _emit_compression_summary(agent, cs):
    # Codex gpt-5.x autoraise notice: at most once per profile/config state (persisted marker
    # — the gateway rebuilds the agent per message). A changed threshold/model re-notifies
    # once; the display gate suppresses the banner without disabling the autoraise.
    _autoraise = agent._compression_threshold_autoraised or {}
    _autoraise_notice = None
    if (
        bool(_autoraise)
        and cs.enabled
        and cs.autoraise_notice_enabled
        and not _codex_gpt55_autoraise_notice_seen(_autoraise)
    ):
        _autoraise_notice = _build_codex_gpt5_autoraise_notice(
            _autoraise, context_length=getattr(agent.context_compressor, "context_length", None)
        )

    if not agent.quiet_mode:
        if cs.enabled:
            # Report the active engine's own threshold — for a plugin engine the host
            # cs.threshold is not in effect and the percent would contradict the token count.
            _active_threshold_pct = getattr(
                agent.context_compressor, "threshold_percent", cs.threshold
            )
            _cap_note = ""
            _cap = getattr(agent.context_compressor, "threshold_tokens_cap", None)
            if _cap and _cap > 0:
                _cap_note = f" (capped at {_cap:,} tokens)"
            print(f"📊 Context limit: {agent.context_compressor.context_length:,} tokens (compress at {int(_active_threshold_pct*100)}% = {agent.context_compressor.threshold_tokens:,}{_cap_note})")
        else:
            print(f"📊 Context limit: {agent.context_compressor.context_length:,} tokens (auto-compression disabled)")
        # Printed inline for CLI users; gateway users get the same text replayed via
        # _compression_warning on turn 1.
        if _autoraise_notice:
            print(_autoraise_notice)

    # Gateway parity: status_callback isn't wired yet, so stash the text to replay on the
    # first run_conversation(). Mark shown so repeated inits in this profile stay silent.
    agent._compression_warning = _autoraise_notice
    if _autoraise_notice:
        _record_codex_gpt55_autoraise_notice(_autoraise)
    # Feasibility check is deferred to the first turn near the threshold (eager costs ~400ms
    # cold per init); run_conversation's preflight runs it at most once per agent.
    agent._compression_feasibility_checked = False


def _snapshot_primary_runtime(agent):
    # Snapshot primary runtime for per-turn restoration: when fallback activates during a
    # turn, the next turn restores these so the preferred model gets a fresh attempt. One
    # dict so new state fields are easy to add.
    _cc = agent.context_compressor
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_echo_flag": getattr(agent, "_reasoning_echo_flag", False),
        # Context engine state that _try_activate_fallback() overwrites. getattr because
        # plugin engines may lack these ContextCompressor-specific attrs.
        "compressor_model": getattr(_cc, "model", agent.model),
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(_cc, "api_key", ""),
        "compressor_provider": getattr(_cc, "provider", agent.provider),
        "compressor_context_length": _cc.context_length,
        "compressor_threshold_tokens": _cc.threshold_tokens,
    }
    if agent.api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })


def _init_usage_state(agent):
    from agent.runtime_cwd import scope_terminal_cwd as _scope_terminal_cwd

    agent._subdirectory_hints = SubdirectoryHintTracker(working_dir=_scope_terminal_cwd() or None)
    agent._user_turn_count = 0
    # Copilot x-initiator flag: first API call of a user turn sends "user".
    agent._is_user_initiated_turn = False

    # Usage-anchored context accounting (agent/model_metadata.py): last provider response's
    # exact usage + transcript snapshot. None until the first response with usage;
    # invalidated on compaction and session switches so stale anchors never suppress
    # compression.
    agent._usage_anchor = None
    agent._turn_base_usage_anchor = None

    # Cumulative token usage for the session
    for _counter in (
        "session_prompt_tokens", "session_completion_tokens", "session_total_tokens",
        "session_api_calls", "session_input_tokens", "session_output_tokens",
        "session_cache_read_tokens", "session_cache_write_tokens", "session_reasoning_tokens",
    ):
        setattr(agent, _counter, 0)
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    # Rolling history for status-bar avg latency / velocity (last 10 calls), shared by
    # conversation_loop and codex_runtime and readable by the CLI snapshot without IPC.
    from collections import deque as _deque
    agent._api_latency_history = _deque(maxlen=10)
    agent._api_output_history = _deque(maxlen=10)


# Constructor params stored verbatim under the same name.
_PASSTHROUGH_PARAMS = (
    "model", "max_iterations", "save_trajectories", "verbose_logging", "quiet_mode",
    "tool_progress_mode", "ephemeral_system_prompt", "platform", "skip_context_files",
    "load_soul_identity", "pass_session_id", "log_prefix_chars",
    # OpenRouter provider preferences
    "providers_allowed", "providers_ignored", "providers_order", "provider_sort",
    "provider_require_parameters", "provider_data_collection", "openrouter_min_coding_score",
    # Toolset filtering
    "enabled_toolsets", "disabled_toolsets",
    # Model response configuration (None = provider/model default)
    "max_tokens", "reasoning_config", "service_tier",
)
# Gateway identity params stored as ``agent._<name>``. gateway_session_key is the stable
# per-chat key (e.g. agent:main:telegram:dm:123).
_GATEWAY_IDENTITY_PARAMS = (
    "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id",
    "gateway_session_key",
)
_CALLBACK_PARAMS = (
    "tool_progress_callback", "tool_start_callback", "tool_complete_callback",
    "thinking_callback", "reasoning_callback", "clarify_callback",
    "read_terminal_callback", "read_preview_callback", "drive_preview_callback",
    "read_window_below_callback", "setup_mcp_callback", "tour_callback",
    "step_callback", "stream_delta_callback", "interim_assistant_callback",
    "status_callback", "notice_callback", "notice_clear_callback",
    "event_callback", "reaction_callback", "tool_gen_callback",
)


def init_agent(
    agent,
    base_url: str = None,
    api_key: str = None,
    provider: str = None,
    api_mode: str = None,
    acp_command: str = None,
    acp_args: list[str] | None = None,
    command: str = None,
    args: list[str] | None = None,
    model: str = "",
    max_iterations: int = sys.maxsize,  # Default: unlimited tool-calling iterations (shared with subagents)
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    save_trajectories: bool = False,
    verbose_logging: bool = False,
    quiet_mode: bool = False,
    tool_progress_mode: str = "all",
    ephemeral_system_prompt: str = None,
    log_prefix_chars: int = 100,
    log_prefix: str = "",
    providers_allowed: List[str] = None,
    providers_ignored: List[str] = None,
    providers_order: List[str] = None,
    provider_sort: str = None,
    provider_require_parameters: bool = False,
    provider_data_collection: str = None,
    openrouter_min_coding_score: Optional[float] = None,
    session_id: str = None,
    tool_progress_callback: callable = None,
    tool_start_callback: callable = None,
    tool_complete_callback: callable = None,
    thinking_callback: callable = None,
    reasoning_callback: callable = None,
    clarify_callback: callable = None,
    read_terminal_callback: callable = None,
    read_preview_callback: callable = None,
    drive_preview_callback: callable = None,
    read_window_below_callback: callable = None,
    setup_mcp_callback: callable = None,
    tour_callback: callable = None,
    step_callback: callable = None,
    stream_delta_callback: callable = None,
    interim_assistant_callback: callable = None,
    tool_gen_callback: callable = None,
    status_callback: callable = None,
    notice_callback: callable = None,
    notice_clear_callback: callable = None,
    event_callback: Optional[Callable[[str, dict], None]] = None,
    reaction_callback: Optional[Callable[[str], None]] = None,
    max_tokens: int = None,
    reasoning_config: Dict[str, Any] = None,
    service_tier: str = None,
    request_overrides: Dict[str, Any] = None,
    prefill_messages: List[Dict[str, Any]] = None,
    platform: str = None,
    user_id: str = None,
    user_id_alt: str = None,
    user_name: str = None,
    chat_id: str = None,
    chat_name: str = None,
    chat_type: str = None,
    thread_id: str = None,
    gateway_session_key: str = None,
    skip_context_files: bool = False,
    load_soul_identity: bool = False,
    skip_memory: bool = False,
    skip_background_review: bool = False,
    session_db=None,
    parent_session_id: str = None,
    iteration_budget: "IterationBudget" = None,
    run_budget_seconds: Optional[float] = None,
    fallback_model: Dict[str, Any] = None,
    credential_pool=None,
    checkpoints_enabled: bool = False,
    checkpoint_max_snapshots: int = 20,
    checkpoint_max_total_size_mb: int = 500,
    checkpoint_max_file_size_mb: int = 10,
    pass_session_id: bool = False,
    requested_provider: str = None,
    capabilities: Optional[Dict[str, bool]] = None,
):
    """Initialize the AI Agent (body of :meth:`AIAgent.__init__`).

    Non-obvious parameters:
      requested_provider: original provider identity before runtime canonicalization.
      openrouter_min_coding_score: coding-score floor for ``openrouter/pareto-code``
        only; None/empty lets OpenRouter pick.
      clarify_callback: ``(question, choices) -> str`` from the platform layer; if
        None the clarify tool returns an error.
      reasoning_config: None defaults to ``{"enabled": True, "effort": "medium"}``
        on OpenRouter.
      prefill_messages: prepended to history as priming context. Anthropic Sonnet/
        Opus 4.6+ reject a conversation ending on an assistant message (400) — use
        structured outputs instead of a trailing-assistant prefill there.
      skip_context_files: skip SOUL.md/.hermes.md/AGENTS.md/CLAUDE.md/.cursorrules
        injection (batch/data-generation runs). load_soul_identity keeps
        ~/.hermes/SOUL.md as identity even when skip_context_files=True.
    """
    _install_safe_stdio()

    _params = locals()
    for _name in _PASSTHROUGH_PARAMS:
        setattr(agent, _name, _params[_name])
    for _name in _GATEWAY_IDENTITY_PARAMS:
        setattr(agent, f"_{_name}", _params[_name])
    # Shared iteration budget — parent creates, children inherit; consumed by every LLM
    # turn across parent + all subagents.
    agent.iteration_budget = iteration_budget or IterationBudget(max_iterations)
    # Pluggable print function — CLI replaces this with _cprint so raw ANSI status lines go
    # through prompt_toolkit's renderer (patch_stdout's StdoutProxy would mangle them).
    # None = builtins.print.
    agent._print_fn = None
    agent.background_review_callback = None  # Optional sync callback for gateway delivery
    agent.memory_notifications = "on"  # Memory update notifications: "off", "on", "verbose"
    # Background review (memory/skill) opt-out: skips the end-of-turn fork (~30K tokens/event)
    # on cron-style sessions; single switch for both review paths.
    agent.skip_background_review = bool(skip_background_review)
    agent.log_prefix = f"{log_prefix} " if log_prefix else ""
    # Effective base URL for feature detection (prompt caching, reasoning, etc.)
    agent.base_url = base_url or ""
    provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
    agent.provider = provider_name or ""
    agent.requested_provider = (
        requested_provider.strip().lower()
        if isinstance(requested_provider, str) and requested_provider.strip()
        else agent.provider
    )
    agent.capabilities = {
        key: value for key, value in (capabilities or {}).items()
        if isinstance(key, str) and isinstance(value, bool)
    }
    agent._credential_pool = credential_pool
    agent.acp_command = acp_command or command
    agent.acp_args = list(acp_args or args or [])
    _resolve_api_mode(agent, api_mode, provider_name, base_url)
    _finalize_routing(agent, api_mode, credential_pool)

    # Platform callbacks are stored under their parameter names verbatim.
    for _cb in _CALLBACK_PARAMS:
        setattr(agent, _cb, _params[_cb])
    agent.suppress_status_output = False

    _init_control_state(agent)

    # Per-provider reasoning_content echo opt-in (see _reasoning_echo_opt_in). Read once at
    # init; switch_model / try_activate_fallback / restore keep it in sync.
    agent._reasoning_echo_flag = agent._read_reasoning_echo_from_config()
    agent.request_overrides = dict(request_overrides or {})
    agent.prefill_messages = prefill_messages or []  # Prefilled conversation turns
    agent._force_ascii_payload = False

    _init_prompt_cache_config(agent)
    _init_turn_state(agent, run_budget_seconds)
    _setup_logging(agent)
    _init_stream_state(agent)
    _build_client(agent, api_key, base_url, fallback_model)
    _init_fallback_chain(agent, fallback_model)
    _load_tools(agent, enabled_toolsets, disabled_toolsets)
    _init_session_state(
        agent, session_id, session_db, parent_session_id, reasoning_config, max_tokens,
        checkpoints_enabled, checkpoint_max_snapshots, checkpoint_max_total_size_mb, checkpoint_max_file_size_mb,
    )

    # Load config once for memory, skills, and compression sections
    try:
        from hermes_cli.config import load_config_readonly as _load_agent_config
        _agent_cfg = _load_agent_config()
    except Exception:
        _agent_cfg = {}

    _apply_display_config(agent, _agent_cfg, platform)
    _init_memory(agent, _agent_cfg, skip_memory, platform)
    _apply_agent_section(agent, _agent_cfg)
    cs = _parse_compression_config(agent, _agent_cfg)
    _config_context_length, _custom_providers, _effective_context_length, _model_cfg = _resolve_context_length(
        agent, _agent_cfg, base_url
    )
    _build_context_engine(agent, _agent_cfg, cs, _custom_providers, _effective_context_length, session_db)
    _enforce_minimum_context(agent)
    _warn_nonagentic_hermes_model(agent)
    _inject_context_engine_tools(agent)
    _init_usage_state(agent)
    _configure_ollama_num_ctx(agent, _model_cfg, _config_context_length)
    _emit_compression_summary(agent, cs)
    _snapshot_primary_runtime(agent)


__all__ = ["init_agent"]
