#!/usr/bin/env python3
"""
AI Agent Runner with Tool Calling

This module provides a clean, standalone agent that can execute AI models
with tool calling capabilities. It handles the conversation loop, tool execution,
and response management.

Features:
- Automatic tool calling loop until completion
- Configurable model parameters
- Error handling and recovery
- Message history management
- Support for multiple model providers

Usage:
    from run_agent import AIAgent

    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Missing hermes_bootstrap (partial `hermes update`) only skips Windows UTF-8 stdio setup.
    pass

import json
import logging
logger = logging.getLogger(__name__)
import os
import re
import sys
import time
import threading
import uuid
import warnings
from typing import List, Dict, Any, Optional, Callable
# `OpenAI` is a lazy proxy (SDK import costs ~240ms) that keeps the single `OpenAI(**kw)` call site and
# `patch("run_agent.OpenAI")` working. `fire` is imported only in __main__ so library imports never need it.
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home


def _launch_cwd_for_session(source: str) -> Optional[str]:
    """Working directory to stamp on a new session row, or None.

    Only local CLI sessions record a cwd (meaningful for ``hermes -c`` / ``--resume``). Gateway/cron/remote
    backends (non-"local" ``TERMINAL_ENV``) have no stable host cwd for the agent's tools, so they record
    nothing.
    """
    if source != "cli":
        return None
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend and backend != "local":
        return None
    try:
        return os.getcwd()
    except OSError:
        # cwd was unlinked out from under us — nothing meaningful to record.
        return None


def _session_source_for_agent(platform: Optional[str]) -> str:
    try:
        from gateway.session_context import get_session_env

        source = get_session_env("HERMES_SESSION_SOURCE", "")
    except Exception:
        source = os.environ.get("HERMES_SESSION_SOURCE", "")
    source = str(source or "").strip()
    if source:
        return source
    return platform or "cli"


def _gateway_origin_json(agent: "AIAgent") -> Optional[str]:
    """Build the gateway routing ``origin_json`` for a session row.

    Mirrors ``SessionSource.to_dict()`` so state.db consumers see the same fields
    ``record_gateway_session_peer`` writes. None when the agent carries no gateway identity.
    """
    chat_id = getattr(agent, "_chat_id", None)
    session_key = getattr(agent, "_gateway_session_key", None)
    user_id = getattr(agent, "_user_id", None)
    if not (chat_id or session_key or user_id):
        return None
    origin: Dict[str, Any] = {
        "platform": getattr(agent, "platform", None) or "",
        "chat_id": chat_id,
        "chat_name": getattr(agent, "_chat_name", None),
        "chat_type": getattr(agent, "_chat_type", None) or "dm",
        "user_id": user_id,
        "user_name": getattr(agent, "_user_name", None),
        "thread_id": getattr(agent, "_thread_id", None),
    }
    user_id_alt = getattr(agent, "_user_id_alt", None)
    if user_id_alt:
        origin["user_id_alt"] = user_id_alt
    profile = getattr(agent, "_profile_name", None)
    if not profile:
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile == "default":
                profile = None
        except Exception:
            profile = None
    if profile:
        origin["profile"] = profile
    try:
        return json.dumps(origin)
    except Exception:
        return None


# OpenAI lazy proxy + stdio/proxy helpers live in agent/process_bootstrap.py. The F401-suppressed
# re-exports below are reached via `patch("run_agent.<X>")`, `from run_agent import X`, or `_ra().<X>`.
from agent.process_bootstrap import (
    OpenAI,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.OpenAI")
    _SafeWriter,  # noqa: F401  # re-exported for tests that `from run_agent import _SafeWriter`
    _get_proxy_for_base_url,  # noqa: F401  # re-exported for tests
)
from agent.iteration_budget import IterationBudget


from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")


# Import our tool system
from model_tools import (
    get_tool_definitions,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.get_tool_definitions")
    get_toolset_for_tool,
    handle_function_call,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.handle_function_call")
    check_toolset_requirements,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.check_toolset_requirements")
)
from tools.terminal_tool import cleanup_vm, get_active_env
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


# Agent internals extracted to agent/ package for modularity
from agent.memory_provider import is_trivial_prompt
from agent.error_classifier import FailoverReason  # noqa: F401  # re-exported (`from run_agent import FailoverReason`)
from agent.client_lifecycle import (  # noqa: F401  # _routermint_headers/_qwen_portal_headers re-exported for agent_init's _ra()
    ClientLifecycleMixin,
    _qwen_portal_headers,
    _routermint_headers,
)
from agent.stream_delivery import StreamDeliveryMixin
from agent.status_output import StatusOutputMixin
from agent.api_request_hooks import ApiRequestHooksMixin
from agent.api_error_summary import ApiErrorSummaryMixin
from agent.interrupt_control import InterruptControlMixin
from agent.turn_explainers import TurnExplainersMixin
from agent.activity_tracking import ActivityTrackingMixin
from agent.rate_limit_credits import RateLimitCreditsMixin
from agent.session_persistence import (  # noqa: F401  # re-exported: cli/gateway/tui/tests import these from run_agent
    SessionPersistenceMixin,
    _DB_PERSISTED_MARKER,
    _EPHEMERAL_SCAFFOLDING_FLAGS,
    _is_ephemeral_scaffolding,
    _safe_session_filename_component,
)
from agent.compression_facade import CompressionFacadeMixin
from agent.turn_facade import TurnFacadeMixin
from agent.vision_message_prep import VisionMessagePrepMixin
from agent.reasoning_params import ReasoningParamsMixin
from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.session_activity import ActivityProvenance
from agent.model_metadata import (
    estimate_request_tokens_rough,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.estimate_request_tokens_rough")
    is_local_endpoint,
)
# Re-exported for tests that monkeypatch these symbols on run_agent.
from agent.context_compressor import (  # noqa: F401
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    user_originated_turn_view,
)
from agent.retry_utils import jittered_backoff  # noqa: F401
from agent.prompt_builder import (  # noqa: F401  # re-exported via _ra() / mock.patch("run_agent.<name>") / from run_agent import <name>
    DEFAULT_AGENT_IDENTITY,
    build_skills_system_prompt,
    build_context_files_prompt,
    build_environment_hints,
    load_soul_md,
)
from agent.process_bootstrap import _get_proxy_from_env  # noqa: F401
from agent.message_sanitization import (  # noqa: F401
    _SURROGATE_RE,
    _sanitize_surrogates,
    _sanitize_structure_surrogates,
    _sanitize_messages_surrogates,
    _escape_invalid_chars_in_json_strings,
    _repair_tool_call_arguments,
    _strip_non_ascii,
    _sanitize_messages_non_ascii,
    _sanitize_tools_non_ascii,
    _looks_like_image_content_rejection,
    _strip_images_from_messages,
    _sanitize_structure_non_ascii,
    coalesce_tool_call_id as _sanitize_coalesce_tool_call_id,
    uniquify_tool_call_ids as _sanitize_uniquify_tool_call_ids,
)
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _deterministic_call_id as _codex_deterministic_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
    _summarize_user_message_for_log,  # also used by _sync_external_memory_for_turn (memory boundary)
)
from agent.tool_guardrails import (
    ToolGuardrailDecision,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)
from agent.tool_dispatch_helpers import (
    _should_parallelize_tool_batch,  # noqa: F401  # re-exported for tests that `from run_agent import _should_parallelize_tool_batch`
    _is_destructive_command,  # noqa: F401  # re-exported for tests that access `run_agent._is_destructive_command`
    _extract_parallel_scope_path,  # noqa: F401  # re-exported for tests that `from run_agent import _extract_parallel_scope_path`
    _paths_overlap,  # noqa: F401  # re-exported for tests that `from run_agent import _paths_overlap`
    _append_subdir_hint_to_multimodal,  # noqa: F401  # re-exported for tests that `from run_agent import _append_subdir_hint_to_multimodal`
    _trajectory_normalize_msg,  # noqa: F401  # re-exported for tests that `from run_agent import _trajectory_normalize_msg`
)
from utils import base_url_host_matches, base_url_hostname, env_float, model_forces_max_completion_tokens


_MAX_TOOL_WORKERS = 8


# Spawn the OpenRouter pre-warm thread once per process, not per AIAgent (gateway thread leak).
_openrouter_prewarm_done = threading.Event()


def _pool_may_recover_from_rate_limit(pool) -> bool:
    """Decide whether to wait for credential-pool rotation instead of falling back.

    Rotation only helps when the pool has somewhere to go: with a single-credential pool the entry that
    just 429'd is the only one, so waiting retries the same exhausted quota. Fall back to ``fallback_model``
    instead.
    """
    if pool is None:
        return False
    if not pool.has_available():
        return False
    return len(pool.entries()) > 1


class _StreamErrorEvent(Exception):
    """Synthesized provider error surfaced from a Responses ``error`` SSE frame.

    Some Codex-style backends emit a standalone ``type=error`` frame instead of ``response.failed`` or an HTTP
    4xx. Raising this gives ``_summarize_api_error`` / the entitlement detector the familiar ``.body`` /
    ``.status_code`` shape.
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        param: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.status_code = status_code
        # OpenAI SDK-shaped body so _extract_api_error_context /
        # _summarize_api_error / classify_api_error all pick it up.
        self.body: Dict[str, Any] = {
            "error": {
                "message": message,
                "code": code,
                "param": param,
                "type": "error",
            }
        }


class AIAgent(
    ClientLifecycleMixin,
    StreamDeliveryMixin,
    StatusOutputMixin,
    ApiRequestHooksMixin,
    ApiErrorSummaryMixin,
    InterruptControlMixin,
    TurnExplainersMixin,
    ActivityTrackingMixin,
    RateLimitCreditsMixin,
    SessionPersistenceMixin,
    CompressionFacadeMixin,
    TurnFacadeMixin,
    VisionMessagePrepMixin,
    ReasoningParamsMixin,
):
    """AI Agent with tool calling capabilities."""

    _TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER = (
        "[hermes-agent: tool call arguments were corrupted in this session and "
        "have been dropped to keep the conversation alive. See issue #15236.]"
    )

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""
        self._base_url_hostname = base_url_hostname(value)

    def __init__(
        self,
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
        tool_delay: float = None,  # Deprecated: accepted for compatibility, ignored
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
        capabilities: Dict[str, bool] | None = None,
    ):
        """Forwarder — see ``agent.agent_init.init_agent`` (same keyword parameters, minus ``tool_delay``)."""
        init_kwargs = {k: v for k, v in locals().items() if k not in ("self", "tool_delay")}
        if tool_delay is not None:
            warnings.warn(
                "tool_delay is deprecated and ignored; sequential tool calls "
                "no longer sleep between executions.",
                DeprecationWarning,
                stacklevel=2,
            )
        from agent.agent_init import init_agent
        init_agent(self, **init_kwargs)

    def _get_session_db_for_recall(self):
        """Return a SessionDB for recall, lazily creating it if an entrypoint forgot.

        A missing ``session_db`` constructor arg degrades to opening the default state DB rather than
        making the advertised ``session_search`` tool unusable.
        """
        # Persistence-isolated forks (background review) must not lazily open the canonical state DB —
        # that would re-arm the flush to write the fork's harness turn into the user's real session.
        if getattr(self, "_persist_disabled", False):
            return None
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import get_shared_session_db

            self._session_db = get_shared_session_db()
            # We opened it here, so nothing else holds a reference — this agent
            # is its only owner and close() must release it.
            self._owns_session_db = True
            return self._session_db
        except Exception:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def _ensure_db_session(self) -> None:
        """Create session DB row on first use. Disables _session_db on failure."""
        if getattr(self, "_persist_disabled", False):
            return
        if self._session_db_created or not self._session_db:
            return
        source = _session_source_for_agent(self.platform)
        try:
            try:
                from hermes_cli.profiles import get_active_profile_name
                _profile_for_session = get_active_profile_name()
                # Persist the profile name explicitly, including "default": profile-keyed consumers treat NULL
                # as unowned (#94724 backfill, #99222).
            except Exception:
                _profile_for_session = None
            # Carry the live YOLO bypass into model_config: the row is created lazily on the first turn, so
            # this is the only chance to record a pre-first-turn /yolo toggle for `hermes --resume`.
            _init_model_config = self._session_init_model_config
            try:
                from tools.approval import is_session_yolo_enabled
                if is_session_yolo_enabled(self.session_id):
                    _init_model_config = dict(_init_model_config or {})
                    _init_model_config["yolo_mode"] = True
            except Exception:
                pass
            # Carry the gateway routing identity: when the gateway SessionStore degraded to JSONL (corrupt
            # state.db) this lazy create is the ONLY durable write, and an identity-less row is unrecoverable.
            self._session_db.create_session(
                session_id=self.session_id,
                source=source,
                model=self.model,
                model_config=_init_model_config,
                system_prompt=self._cached_system_prompt,
                user_id=getattr(self, "_user_id", None),
                session_key=getattr(self, "_gateway_session_key", None),
                chat_id=getattr(self, "_chat_id", None),
                chat_type=getattr(self, "_chat_type", None),
                thread_id=getattr(self, "_thread_id", None),
                display_name=(
                    getattr(self, "_chat_name", None)
                    or getattr(self, "_user_name", None)
                ),
                origin_json=_gateway_origin_json(self),
                parent_session_id=self._parent_session_id,
                cwd=_launch_cwd_for_session(source),
                profile_name=_profile_for_session,
            )
            self._session_db_created = True
        except Exception as e:
            # Transient failure (e.g. SQLite lock). Keep _session_db alive —
            # _session_db_created stays False so next run_conversation() retries.
            logger.warning(
                "Session DB creation failed (will retry next turn): %s", e
            )

    def _transition_context_engine_session(
        self,
        *,
        old_session_id: Optional[str] = None,
        new_session_id: Optional[str] = None,
        previous_messages: Optional[list] = None,
        carry_over_context: bool = False,
        reset_engine: bool = True,
        **extra_context,
    ) -> None:
        """Notify the active context engine about a host session transition.

        The built-in compressor keeps its reset behavior; plugin engines with richer hooks (``on_session_end``
        / ``on_session_reset`` / ``on_session_start`` / ``carry_over_new_session_context``) can flush, rebind
        and carry context.
        """
        engine = getattr(self, "context_compressor", None)
        if not engine:
            return

        if old_session_id and previous_messages is not None and hasattr(engine, "on_session_end"):
            try:
                engine.on_session_end(old_session_id, previous_messages)
            except Exception as exc:
                logger.debug("context engine on_session_end during transition: %s", exc)

        if reset_engine and hasattr(engine, "on_session_reset"):
            try:
                engine.on_session_reset()
            except Exception as exc:
                logger.debug("context engine on_session_reset during transition: %s", exc)

        should_start = bool(
            old_session_id
            or previous_messages is not None
            or carry_over_context
            or extra_context
        )
        target_session_id = new_session_id or getattr(self, "session_id", "") or ""
        if should_start and target_session_id and hasattr(engine, "on_session_start"):
            start_context = {
                "old_session_id": old_session_id,
                "carry_over_context": carry_over_context,
                "platform": _session_source_for_agent(getattr(self, "platform", None)),
                "model": getattr(self, "model", ""),
                "context_length": getattr(engine, "context_length", None),
                "conversation_id": getattr(self, "_gateway_session_key", None),
            }
            start_context.update(extra_context)
            start_context = {k: v for k, v in start_context.items() if v not in (None, "")}
            try:
                engine.on_session_start(target_session_id, **start_context)
            except Exception as exc:
                logger.debug("context engine on_session_start during transition: %s", exc)

        if (
            carry_over_context
            and old_session_id
            and target_session_id
            and hasattr(engine, "carry_over_new_session_context")
        ):
            try:
                engine.carry_over_new_session_context(old_session_id, target_session_id)
            except Exception as exc:
                logger.debug("context engine carry_over_new_session_context during transition: %s", exc)

    def reset_session_state(
        self,
        previous_messages: Optional[list] = None,
        old_session_id: Optional[str] = None,
        carry_over_context: bool = False,
    ):
        """Reset all session-scoped token/cost counters and compressor state for a fresh session.

        When ``previous_messages`` / ``old_session_id`` / ``carry_over_context`` are given, the context engine
        gets the full transition lifecycle (``_transition_context_engine_session``) instead of a bare reset.
        """
        # Token usage counters
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

        # Session boundary: the usage anchor describes the OLD transcript; fall back to full estimation.
        self._usage_anchor = None
        self._turn_base_usage_anchor = None

        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Copilot x-initiator: True for the first API call of a user turn,
        # False for tool-loop follow-ups (#3040).
        self._is_user_initiated_turn = False

        # Context engine reset/transition (works for built-in compressor and plugins)
        self._transition_context_engine_session(
            old_session_id=old_session_id,
            new_session_id=getattr(self, "session_id", None),
            previous_messages=previous_messages,
            carry_over_context=carry_over_context,
            reset_engine=True,
        )

        # Reset-only switches (/new, /resume, /branch) change session_id before this call; rebind the
        # built-in compressor's session-keyed cooldown state when no full start hook ran.
        engine = getattr(self, "context_compressor", None)
        target_session_id = getattr(self, "session_id", "") or ""
        bound_session_id = getattr(engine, "_session_id", "") if engine is not None else ""
        if (
            engine is not None
            and hasattr(engine, "bind_session_state")
            and target_session_id
            and target_session_id != bound_session_id
        ):
            try:
                engine.bind_session_state(getattr(self, "_session_db", None), target_session_id)
            except Exception as exc:
                logger.debug("context engine bind_session_state during reset: %s", exc)

    @staticmethod
    def _effective_lmstudio_context_length(
        config_context_length: Optional[int],
        runtime_context_length: Any,
    ) -> Optional[int]:
        """Return a safe context budget from explicit intent and verified runtime."""
        explicit = (
            config_context_length
            if isinstance(config_context_length, int)
            and not isinstance(config_context_length, bool)
            and config_context_length > 0
            else None
        )
        runtime_value = getattr(runtime_context_length, "context_length", runtime_context_length)
        runtime = (
            runtime_value
            if isinstance(runtime_value, int)
            and not isinstance(runtime_value, bool)
            and runtime_value > 0
            else None
        )
        if bool(getattr(runtime_context_length, "rejected", False)) or (
            bool(getattr(runtime_context_length, "load_attempted", False))
            and runtime is None
        ):
            return None
        if runtime is not None and explicit is not None:
            return min(runtime, explicit)
        return runtime if runtime is not None else explicit

    @staticmethod
    def _lmstudio_load_was_unverified(load_result: Any) -> bool:
        """Return true when a management load was rejected or unverifiable."""
        return bool(getattr(load_result, "rejected", False)) or (
            bool(getattr(load_result, "load_attempted", False))
            and getattr(load_result, "context_length", None) is None
        )

    def _ensure_lmstudio_runtime_loaded(
        self,
        config_context_length: Optional[int] = None,
    ) -> Any:
        """Preload LM Studio unless configured to rely on JIT loading."""
        if (self.provider or "").strip().lower() != "lmstudio":
            return None
        if (getattr(self, "lmstudio_load_mode", "explicit") or "explicit").strip().lower() == "jit":
            logger.debug("LM Studio explicit preload skipped: lmstudio_load_mode=jit")
            return None

        from hermes_cli.models import ensure_lmstudio_model_loaded

        if config_context_length is None:
            config_context_length = getattr(self, "_config_context_length", None)
        return ensure_lmstudio_model_loaded(
            self.model,
            self.base_url,
            getattr(self, "api_key", ""),
            config_context_length,
            return_load_result=True,
        )

    switch_model = _forward("agent.agent_runtime_helpers", "switch_model")

    def _disable_codex_reasoning_replay(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Disable Responses encrypted reasoning replay and strip cached state.

        Called on HTTP 400 ``invalid_encrypted_content``. Sets ``_codex_reasoning_replay_enabled=False``
        (consumed by the codex adapter/transport) and pops ``codex_reasoning_items`` from every assistant
        message. Returns ``{"messages": int, "items": int}`` for diagnostic logging.
        """
        stripped_messages = 0
        stripped_items = 0
        target_messages = messages if isinstance(messages, list) else []

        for msg in target_messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            items = msg.pop("codex_reasoning_items", None)
            if isinstance(items, list) and items:
                stripped_messages += 1
                stripped_items += len(items)

        self._codex_reasoning_replay_enabled = False
        return {"messages": stripped_messages, "items": stripped_items}

    # Stream-diagnostic class header preserved for backward compat —
    # actual list lives in ``agent.stream_diag.STREAM_DIAG_HEADERS``.
    from agent.stream_diag import STREAM_DIAG_HEADERS as _STREAM_DIAG_HEADERS  # noqa: E402

    _stream_diag_init = _forward_static("agent.stream_diag", "stream_diag_init")

    _stream_diag_capture_response = _forward("agent.stream_diag", "stream_diag_capture_response")

    _flatten_exception_chain = _forward_static("agent.stream_diag", "flatten_exception_chain")

    def _is_provider_stream_parse_error(self, error: BaseException) -> bool:
        """Return True for malformed provider streaming data from SDK parsers.

        The Anthropic SDK surfaces a malformed event-stream frame as a plain ``ValueError``; that is wire-
        format trouble, not local validation, so it follows the truncated-JSON retry path.
        """
        if getattr(self, "api_mode", None) != "anthropic_messages":
            return False
        if not isinstance(error, ValueError):
            return False
        if isinstance(error, (UnicodeEncodeError, json.JSONDecodeError)):
            return False
        message = str(error).strip().lower()
        return "expected ident at line" in message

    _log_stream_retry = _forward("agent.stream_diag", "log_stream_retry")

    _emit_stream_drop = _forward("agent.stream_diag", "emit_stream_drop")

    def _emit_auxiliary_failure(self, task: str, exc: BaseException) -> None:
        """Surface a compact warning for failed auxiliary work."""
        try:
            detail = self._summarize_api_error(exc)
        except Exception:
            detail = str(exc)
        detail = (detail or exc.__class__.__name__).strip()
        if len(detail) > 220:
            detail = detail[:217].rstrip() + "..."
        self._emit_warning(f"⚠ Auxiliary {task} failed: {detail}")

    def _current_main_runtime(self) -> Dict[str, str]:
        """Return the live main runtime for session-scoped auxiliary routing."""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
            "auth_mode": getattr(self, "auth_mode", "") or "",
        }

    _check_compression_model_feasibility = _forward("agent.conversation_compression", "check_compression_model_feasibility")

    _replay_compression_warning = _forward("agent.conversation_compression", "replay_compression_warning")

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        return hostname == "api.openai.com"

    def _is_azure_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets Azure OpenAI.

        Azure accepts the standard ``openai`` client but does NOT support the Responses API, so routing
        must treat it separately from direct OpenAI.
        """
        if base_url is not None:
            url = str(base_url).lower()
        else:
            url = getattr(self, "_base_url_lower", "") or ""
        return base_url_host_matches(url, "openai.azure.com")

    def _is_github_copilot_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets GitHub Copilot's OpenAI-compatible API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        if not hostname:
            return False
        return hostname == "api.githubcopilot.com" or hostname.endswith(".githubcopilot.com")

    def _resolved_api_call_timeout(self) -> float:
        """Resolve the effective per-call request timeout in seconds.

        Priority: per-model ``timeout_seconds`` > provider ``request_timeout_seconds`` >
        ``HERMES_API_TIMEOUT`` > 1800s.
        """
        cfg = get_provider_request_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg
        return env_float("HERMES_API_TIMEOUT", 1800.0)

    def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
        """Resolve the base non-stream stale timeout and whether it is implicit.

        Priority: per-model ``stale_timeout_seconds`` > provider-wide > ``HERMES_API_CALL_STALE_TIMEOUT`` >
        90s.
        Returns ``(seconds, uses_implicit_default)`` so callers can keep legacy behaviors (e.g. auto-disabling
        the detector for local endpoints) that apply only when the user did not configure one.
        """
        cfg = get_provider_stale_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg, False

        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
        if env_timeout is not None:
            return float(env_timeout), False

        # Reasoning-model floor for models whose cloud gateways idle-kill mid-think. uses_implicit_default
        # stays False so the local-endpoint short-circuit does not disable stale detection here.
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        reasoning_floor = get_reasoning_stale_timeout_floor(self.model)
        if reasoning_floor is not None:
            return reasoning_floor, False

        return 90.0, True

    def _compute_non_stream_stale_timeout(self, api_payload: Any) -> float:
        """Compute the effective non-stream stale timeout for this request.

        Accepts a full ``api_kwargs`` dict (Chat Completions or Responses) or a legacy ``messages`` list;
        context-size scaling applies identically via ``estimate_request_context_tokens``.
        """
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        from agent.chat_completion_helpers import estimate_request_context_tokens
        est_tokens = estimate_request_context_tokens(api_payload)
        if est_tokens > 100_000:
            timeout = max(stale_base, 240.0)
        elif est_tokens > 50_000:
            timeout = max(stale_base, 150.0)
        else:
            timeout = stale_base

        # Run-budget cap: an implicit stale timeout is capped at half the remaining budget (>= 60s) so one
        # hung call cannot eat the run. Never raises the timeout; explicit user config still wins.
        run_budget = getattr(self, "run_budget_seconds", None)
        if run_budget and not self._stale_timeout_is_explicit():
            started = getattr(self, "_run_budget_started_at", None)
            if started:
                remaining = float(run_budget) - (time.time() - started)
                deadline_cap = max(60.0, remaining * 0.5)
                if deadline_cap < timeout:
                    timeout = deadline_cap
        return timeout

    def _stale_timeout_is_explicit(self) -> bool:
        """True when the user explicitly configured the non-stream stale timeout (config or env var).

        Implicit values (reasoning floors, the 90s default) yield to the run-budget cap; explicit ones never
        do.
        """
        if get_provider_stale_timeout(self.provider, self.model) is not None:
            return True
        return os.getenv("HERMES_API_CALL_STALE_TIMEOUT") is not None

    def _codex_silent_hang_hint(self, model: Optional[str] = None) -> Optional[str]:
        """Actionable hint when this request matches a known Codex silent-reject configuration, else ``None``.

        The ChatGPT Codex backend has silently dropped some model requests (connection accepted, no events,
        no error); the stale detector ends the hang but a generic timeout gives no path forward. Currently
        flags the ``gpt-5.5`` family. Does not fix the backend — only makes the timeout actionable.
        """
        if self.api_mode != "codex_responses":
            return None
        from agent.codex_responses_adapter import classify_responses_route

        if not classify_responses_route(self).is_codex_backend:
            return None
        eff_model = (model if model is not None else self.model) or ""
        model_lower = eff_model.lower()
        # Match the gpt-5.5 family at word boundaries (bare, -codex, vendor-prefixed) but not gpt-5.50.
        if not re.search(r"(?:^|[/\-_])gpt-5\.5(?:$|[\-_])", model_lower):
            return None
        return (
            f"Codex backend appears to be silently rejecting {eff_model!r} "
            "on chatgpt.com/backend-api/codex (no stream events, no error). "
            "This is a known backend-side pattern that has affected ChatGPT "
            "Plus accounts intermittently. "
            "Workaround: try `gpt-5.4` on the same OAuth profile, or `gpt-5.3-codex`, "
            "or switch to a different model/provider in your fallback chain. "
            "Some ChatGPT Codex accounts do not support `gpt-5.4-codex`. "
            "See hermes-agent#21444 for symptom history."
        )

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return base_url_host_matches(self._base_url_lower, "openrouter.ai")

    def _is_copilot_url(self) -> bool:
        """Return True when the base URL targets GitHub Copilot or GitHub Models."""
        return (
            base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
            or base_url_host_matches(self._base_url_lower, "models.github.ai")
        )

    def _is_copilot_provider(self) -> bool:
        """True when the active provider is GitHub Copilot, however spelled.

        ``self.provider`` may hold the alias ``github-copilot`` / ``github`` rather than ``copilot``; a bare
        equality check silently skips credential recovery. Base URL is accepted as a fallback signal.
        """
        if (self.provider or "").strip().lower() in {"copilot", "github-copilot", "github"}:
            return True
        return self._is_copilot_url()

    def _is_codex_backend(self) -> bool:
        """Return True for the ChatGPT OAuth Codex Responses backend."""
        return (
            getattr(self, "api_mode", None) == "codex_responses"
            and getattr(self, "_base_url_hostname", "") == "chatgpt.com"
            and "/backend-api/codex"
            in (getattr(self, "_base_url_lower", "") or "")
        )

    _anthropic_prompt_cache_policy = _forward("agent.agent_runtime_helpers", "anthropic_prompt_cache_policy")

    _direct_native_anthropic_tool_cache_capability = _forward("agent.agent_runtime_helpers", "_direct_native_anthropic_tool_cache_capability")

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """Return True for models that require the Responses API path.

        GPT-5.x is rejected on /v1/chat/completions (``unsupported_api_for_model``) by OpenAI and OpenRouter.
        """
        m = model.lower()
        # Strip vendor prefix (e.g. "openai/gpt-5.4" → "gpt-5.4")
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        return m.startswith("gpt-5")

    @staticmethod
    def _provider_model_requires_responses_api(
        model: str,
        *,
        provider: Optional[str] = None,
    ) -> bool:
        """Return True when this provider/model pair should use Responses API."""
        normalized_provider = (provider or "").strip().lower()
        # Nous serves GPT-5.x models via its OpenAI-compatible chat
        # completions endpoint; its /v1/responses endpoint returns 404.
        if normalized_provider == "nous":
            return False
        if normalized_provider == "custom":
            # Generic custom endpoints may relay GPT-5 without full Responses semantics — only direct
            # OpenAI/xAI URLs auto-upgrade.
            return False
        if normalized_provider == "copilot":
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                return _should_use_copilot_responses_api(model)
            except Exception:
                # Fall back to the generic GPT-5 rule if Copilot-specific
                # logic is unavailable for any reason.
                pass
        return AIAgent._model_requires_responses_api(model)

    def _max_tokens_param(self, value: int) -> dict:
        """Return the correct max tokens kwarg for the current provider.

        Newer OpenAI families (and Azure / Copilot serving them) need ``max_completion_tokens``; others use
        ``max_tokens``. URL-first, then model-name fallback so third-party endpoints fronting those models
        work.
        """
        if (
            self._is_direct_openai_url()
            or self._is_azure_openai_url()
            or self._is_github_copilot_url()
            or model_forces_max_completion_tokens(self.model)
        ):
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    @staticmethod
    def _requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
        """Extract the outgoing response token cap from a prepared request."""
        if not isinstance(api_kwargs, dict):
            return None
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            raw = api_kwargs.get(key)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _has_content_after_think_block(self, content: str) -> bool:
        """Check if content has actual text after any reasoning/thinking blocks.

        Reasoning-only output is an incomplete generation to retry. Must stay in sync with
        ``_strip_think_blocks()`` tag variants.
        """
        if not content:
            return False

        # Remove all reasoning tag variants (must match _strip_think_blocks)
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())

    _strip_think_blocks = _forward("agent.agent_runtime_helpers", "strip_think_blocks")

    @staticmethod
    def _has_natural_response_ending(content: str) -> bool:
        """Heuristic: does visible assistant text look intentionally finished?"""
        if not content:
            return False
        stripped = content.rstrip()
        if not stripped:
            return False
        if stripped.endswith("```"):
            return True
        if stripped.endswith('^'):
            return True
        last = stripped[-1]
        if last in '.!?:)"\']}。！？：）】」』》^':
            return True
        # Emoji ranges (Misc Symbols, Dingbats, Emoticons, Supplemental, etc.)
        if ord(last) >= 0x1F300:
            return True
        return False

    def _is_ollama_glm_backend(self) -> bool:
        """Detect Ollama-hosted GLM models affected by finish_reason='stop' misreports.

        Matches only explicit Ollama signatures (port 11434, "ollama" in URL, provider ollama) — never
        arbitrary local proxies, which report correctly. Excludes Ollama Cloud (``ollama.com`` host,
        ``:cloud`` suffix): rewriting its stop→length manufactures false truncations and burns the
        continuation budget.
        """
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        base = self._base_url_lower
        # Ollama Cloud (hosted service or :cloud proxy) forwards finish_reason faithfully — do not rewrite.
        if "ollama.com" in base or ":cloud" in model_lower:
            return False
        if "ollama" in base or ":11434" in base:
            return True
        return provider_lower == "ollama"

    def _should_treat_stop_as_truncated(
        self,
        finish_reason: str,
        assistant_message,
        messages: Optional[list] = None,
    ) -> bool:
        """Detect conservative stop->length misreports for Ollama-hosted GLM models."""
        if finish_reason != "stop" or self.api_mode != "chat_completions":
            return False
        if not self._is_ollama_glm_backend():
            return False
        if not any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            for msg in (messages or [])
        ):
            return False
        if assistant_message is None or getattr(assistant_message, "tool_calls", None):
            return False

        content = getattr(assistant_message, "content", None)
        if not isinstance(content, str):
            return False

        visible_text = self._strip_think_blocks(content).strip()
        if not visible_text:
            return False
        if len(visible_text) < 20 or not re.search(r"\s", visible_text):
            return False

        return not self._has_natural_response_ending(visible_text)

    _looks_like_codex_intermediate_ack = _forward("agent.agent_runtime_helpers", "looks_like_codex_intermediate_ack")

    _extract_reasoning = _forward("agent.agent_runtime_helpers", "extract_reasoning")

    _cleanup_task_resources = _forward("agent.chat_completion_helpers", "cleanup_task_resources")

    # Background memory/skill review — prompts live in agent.background_review.
    from agent.background_review import (
        _MEMORY_REVIEW_PROMPT,
        _SKILL_REVIEW_PROMPT,
        _COMBINED_REVIEW_PROMPT,
    )

    _summarize_background_review_actions = _forward_static("agent.background_review", "summarize_background_review_actions")

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
        focus: Optional[str] = None,
        explicit: bool = False,
    ) -> None:
        """Post-turn review entry point: decide WHEN, then spawn.

        A review whose runtime is the MANAGED LOCAL llama-server is queued for machine idle (``defer:
        auto|never``)
        instead of hitting the user's GPU mid-session; everything else spawns immediately. ``explicit``
        (/refine)
        is never deferred but does not touch the ``focus``-keyed delegate/enabled gates.
        """
        # Gates run at enqueue/spawn time; the idle dispatcher re-checks `enabled` at dispatch time.
        if focus is None and getattr(self, "_delegate_depth", 0) > 0:
            return
        task_cfg = None
        if focus is None:
            from agent.background_review import load_background_review_settings
            enabled, task_cfg = load_background_review_settings()
            if not enabled:
                return

        # Structural clone at the single chokepoint: the fork sanitizes in place, and a shallow copy would
        # alias the live history's nested tool_calls/content (#100795).
        from agent.turn_finalizer import _clone_background_review_messages
        messages_snapshot = _clone_background_review_messages(messages_snapshot)

        kwargs = dict(
            messages_snapshot=messages_snapshot,
            review_memory=review_memory,
            review_skills=review_skills,
            focus=focus,
            task_cfg=task_cfg,
        )
        if focus is None and not explicit:
            from agent.review_idle_queue import (
                QUEUE,
                defer_mode,
                review_targets_managed_local,
            )
            if (defer_mode(task_cfg) == "auto"
                    and review_targets_managed_local(self, task_cfg)):
                session_key = str(getattr(self, "session_id", None) or id(self))
                QUEUE.enqueue(self, session_key, kwargs)
                return
        self._spawn_background_review_now(**kwargs)

    def _spawn_background_review_now(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
        focus: Optional[str] = None,
        task_cfg: Optional[Dict[str, Any]] = None,
        _requeue_attempts: int = 0,
    ) -> None:
        """Spawn the background memory/skill review thread.

        ``threading.Thread`` is constructed here so tests patching ``run_agent.threading.Thread`` keep
        working.
        ``focus`` is /refine steering text; ``task_cfg`` is the pre-loaded config block (None on direct
        calls).
        A deferred review preempted by a live turn is requeued (bounded) rather than lost.
        """
        from agent.background_review import (
            finish_background_review_run,
            prepare_background_review_run,
            spawn_background_review_thread,
        )
        from tools.thread_context import propagate_context_to_thread

        review_run = prepare_background_review_run(self)
        if review_run is None:
            return
        try:
            target, _prompt = spawn_background_review_thread(
                self,
                messages_snapshot,
                review_memory=review_memory,
                review_skills=review_skills,
                focus=focus,
                task_cfg=task_cfg,
                review_run=review_run,
            )

            def _target_with_requeue() -> None:
                target()
                self._maybe_requeue_preempted_review(
                    review_run,
                    dict(
                        messages_snapshot=messages_snapshot,
                        review_memory=review_memory,
                        review_skills=review_skills,
                        focus=focus,
                        task_cfg=task_cfg,
                        _requeue_attempts=_requeue_attempts + 1,
                    ),
                )

            # Carry the active profile into the review thread so MEMORY.md /
            # skill review writes land in the right profile (#54937).
            t = threading.Thread(
                target=propagate_context_to_thread(_target_with_requeue),
                daemon=True,
                name="bg-review",
            )
            t.start()
        except Exception:
            finish_background_review_run(self, review_run)
            raise

    _REVIEW_REQUEUE_MAX_ATTEMPTS = 3

    def _maybe_requeue_preempted_review(self, review_run, kwargs) -> None:
        """Requeue a deferred-mode review that a live turn cancelled.

        Only for automatic reviews on the managed local runtime; bounded attempts stop a busy box cycling
        forever.
        """
        try:
            if not review_run.cancel_requested.is_set():
                return  # ran to completion (or never admitted for other reasons)
            if kwargs.get("focus") is not None:
                return
            if kwargs.get("_requeue_attempts", 0) > self._REVIEW_REQUEUE_MAX_ATTEMPTS:
                logger.info("Preempted background review dropped after %d requeues",
                            self._REVIEW_REQUEUE_MAX_ATTEMPTS)
                return
            from agent.review_idle_queue import (
                QUEUE,
                defer_mode,
                review_targets_managed_local,
            )
            task_cfg = kwargs.get("task_cfg")
            if (defer_mode(task_cfg) != "auto"
                    or not review_targets_managed_local(self, task_cfg)):
                return
            session_key = str(getattr(self, "session_id", None) or id(self))
            # kwargs carries the incremented _requeue_attempts through the
            # queue so the cap survives the round trip.
            QUEUE.enqueue(self, session_key, dict(kwargs))
        except Exception:  # noqa: BLE001 — requeue is best-effort
            logger.debug("Preempted-review requeue failed", exc_info=True)

    _build_memory_write_metadata = _forward("agent.background_review", "build_memory_write_metadata")

    _apply_pending_steer_to_tool_results = _forward("agent.agent_runtime_helpers", "apply_pending_steer_to_tool_results")

    def get_activity_summary(self) -> dict:
        """Return a snapshot of the agent's current activity for diagnostics.

        Exposes ``last_activity_at`` / ``last_activity_description`` / ``last_activity_provenance`` plus the
        short aliases existing gateway and delegate readers use.
        """
        from agent.session_activity import (
            build_activity_snapshot,
        )

        provenance = getattr(self, "_last_activity_provenance", None)
        if provenance is None:
            provenance = ActivityProvenance.UNKNOWN
        return build_activity_snapshot(
            last_activity_at=getattr(self, "_last_activity_ts", None),
            last_activity_description=getattr(self, "_last_activity_desc", None) or "",
            last_activity_provenance=provenance,
            extra={
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
            },
        )

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine at session end.

        Idempotent: gateway cleanup and ``AIAgent.close()`` may share this ownership boundary.
        """
        if getattr(self, "_memory_provider_shutdown", False):
            return
        self._memory_provider_shutdown = True
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception as e:
                logger.warning("Memory provider on_session_end failed during shutdown: %s", e, exc_info=True)
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
        # Notify context engine of session end (flush DAG, close DBs, etc.)
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass

    def commit_memory_session(self, messages: list = None) -> None:
        """Trigger end-of-session extraction without tearing providers down.

        Called on session_id rotation (/new, compression); providers keep running, just flushing pending
        extraction.
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
        # Notify the context engine of session end (same lifecycle moment as the memory manager) so
        # per-session engine state does not leak into the next session (#22394).
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass

    def _sync_external_memory_for_turn(
        self,
        *,
        original_user_message: Any,
        final_response: Any,
        interrupted: bool,
        messages: list | None = None,
    ) -> None:
        """Mirror a completed turn into external memory providers (``sync_all`` + ``queue_prefetch_all``).

        Uses ``original_user_message`` — ``user_message`` may carry injected skill content. Interrupted turns
        are skipped entirely: partial output is not durable truth, and a prefetch keyed on it would fire
        against stale context. Strictly best-effort — an offline backend must never block the response.
        """
        if interrupted:
            return
        if not (self._memory_manager and final_response and original_user_message):
            return
        # Flatten multimodal parts to text (newline-joined for memory).
        user_text = _summarize_user_message_for_log(original_user_message, sep="\n")
        response_text = _summarize_user_message_for_log(final_response, sep="\n")
        if not (user_text and response_text):
            return
        try:
            sync_kwargs = {"session_id": self.session_id or ""}
            if messages is not None:
                sync_kwargs["messages"] = messages
            self._memory_manager.sync_all(
                user_text,
                response_text,
                **sync_kwargs,
            )
            # Sibling of the build_turn_context() prefetch gate: don't key recall on zero-signal prompts.
            if not is_trivial_prompt(user_text):
                self._memory_manager.queue_prefetch_all(
                    user_text,
                    session_id=self.session_id or "",
                )
        except Exception:
            pass

    def release_clients(self) -> None:
        """Release LLM client resources WITHOUT tearing down session tool state.

        For gateway cache eviction (LRU/idle): the session may resume with a fresh AIAgent on the same
        task_id, so process_registry entries, terminal sandbox, browser daemon, computer-use backend and
        memory provider are kept. Closes the OpenAI/httpx pool and active child subagents. Idempotent;
        distinct from ``close()``.
        """
        # Close active child agents (per-turn; no cross-turn persistence).
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.release_clients()
                except Exception:
                    # Fall back to full close on children; they're per-turn.
                    try:
                        child.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # Retire (don't hard-close) the shared client: eviction runs on the gateway memory-manager thread,
        # and a cross-thread close can release TLS FDs under a still-unwinding worker (#70773).
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._retire_shared_openai_client(client, reason="cache_evict")
                self.client = None
        except Exception:
            pass

        # Also drop the cached per-request wire client (reused across
        # sequential LLM calls) — same socket/memory rationale as above.
        try:
            self._close_cached_request_openai_client(reason="cache_evict")
        except Exception:
            pass
        try:
            self._close_cached_request_anthropic_client(reason="cache_evict")
        except Exception:
            pass

    def close(self) -> None:
        """Release all resources held by this agent instance (idempotent).

        Cleans up background processes, terminal sandbox, browser daemon, computer-use backend, child agents
        and client connections. Each step is independently guarded so one failure does not block the rest.
        """
        # close() is the hard owner boundary; shutdown_memory_provider() is idempotent so gateway
        # pre-calls never double-extract.
        try:
            session_messages = getattr(self, "_session_messages", None)
            self.shutdown_memory_provider(
                session_messages if isinstance(session_messages, list) else None
            )
        except Exception:
            pass

        task_id = getattr(self, "session_id", None) or ""

        # 1. Kill background processes for this task
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)
        except Exception:
            pass

        # 2. Clean terminal sandbox environments
        try:
            cleanup_vm(task_id)
        except Exception:
            pass

        # 3. Clean browser daemon sessions
        try:
            cleanup_browser(task_id)
        except Exception:
            pass

        # 4. Release the session-owned computer-use backend (lazy import keeps the core footprint narrow).
        try:
            from tools.computer_use import release_computer_use_session

            release_computer_use_session(task_id)
        except Exception:
            pass

        # 5. Close active child agents
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 6. Close the OpenAI/httpx client
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass

        # 6b. Close the cached per-request wire client (reused across
        # sequential LLM calls; see _create_request_openai_client).
        try:
            self._close_cached_request_openai_client(reason="agent_close")
        except Exception:
            pass
        try:
            self._close_cached_request_anthropic_client(reason="agent_close")
        except Exception:
            pass

        # 6c. Close the Codex app-server session; hard teardown had no owner and left the child running.
        # Clear the attribute BEFORE close() so a concurrent reader can't grab a half-closed session.
        try:
            codex_session = getattr(self, "_codex_session", None)
            if codex_session is not None:
                self._codex_session = None
                codex_session.close()
        except Exception:
            pass

        # 7. Free conversation history proactively (close() is the hard teardown; callers may still hold the
        # closed agent).
        try:
            self._session_messages = []
        except Exception:
            pass

        # Return freed heap pages to the OS on glibc; safe no-op elsewhere.
        try:
            from hermes_cli.mem_trim import trim_memory
            trim_memory(force=True, reason="agent close")
        except Exception:
            pass

        # 8. Finalize the owned session row unless ownership was handed forward (compression helpers,
        # review forks sharing the parent's id). end_session() is first-reason-wins and idempotent.
        session_db = getattr(self, "_session_db", None)
        try:
            if getattr(self, "_end_session_on_close", True):
                session_id = getattr(self, "session_id", None)
                if session_db and session_id:
                    session_db.end_session(session_id, "agent_close")
        except Exception:
            pass

        # 9. Close the SQLite handle ONLY when this agent owns it. A dedicated handle left open keeps its
        # fds and background token-writer thread (pinned via atexit) for the life of the process.
        # Cleared first so close() stays idempotent.
        try:
            if getattr(self, "_owns_session_db", False) and session_db is not None:
                self._owns_session_db = False
                # Shared instances no-op on close(); release the refcount
                # so the registry can close when the last caller is done (#90837).
                from hermes_state import release_or_close
                release_or_close(session_db)
        except Exception:
            pass

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """Recover todo state from conversation history.

        The gateway builds a fresh AIAgent per message, so replay the most recent todo tool response. Only
        results paired with an earlier assistant ``todo`` tool call count: caller-supplied history could
        otherwise seed the store with a forged bare ``role: tool`` message (GHSA-5g4g-6jrg-mw3g).
        """
        from tools.todo_tool import MAX_TODO_RESULT_CHARS

        # Walk history backwards to find the most recent todo tool response
        last_todo_response = None
        last_todo_revision = 0
        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # Only accept tool results paired with a prior assistant todo call.
            if not self._tool_response_matches_todo_call(history, idx):
                continue
            if len(content) > MAX_TODO_RESULT_CHARS:
                logger.warning(
                    "Skipping oversized todo tool response during hydration: "
                    "session=%s chars=%d",
                    self.session_id or "none",
                    len(content),
                )
                continue
            # Quick check: todo responses contain "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    last_todo_revision = data.get("revision", 1)
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        if last_todo_response is not None:
            # Restore only when history carries a newer revision than the store holds; empty lists are an
            # authoritative clear.
            current_revision = int(
                self._todo_store.snapshot().get("revision", 0) or 0
            )
            try:
                history_revision = max(0, int(last_todo_revision or 0))
            except (TypeError, ValueError):
                history_revision = 1
            if history_revision > current_revision:
                self._todo_store.restore(
                    last_todo_response,
                    revision=history_revision,
                )
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)

    @classmethod
    def _tool_response_matches_todo_call(
        cls,
        history: List[Dict[str, Any]],
        tool_index: int,
    ) -> bool:
        """Return True when a tool result belongs to a prior assistant todo call.

        Scans back to the nearest assistant message for a ``todo`` call with this ``tool_call_id``; a
        ``user``/``system`` boundary or missing id means unpaired → must not hydrate.
        """
        if tool_index < 0 or tool_index >= len(history):
            return False
        tool_msg = history[tool_index]
        tool_call_id = tool_msg.get("tool_call_id")
        if not tool_call_id:
            return False

        for prior_idx in range(tool_index - 1, -1, -1):
            prior = history[prior_idx]
            role = prior.get("role")
            if role == "assistant":
                return cls._assistant_has_todo_tool_call(prior, tool_call_id)
            if role in {"user", "system"}:
                return False
        return False

    @classmethod
    def _assistant_has_todo_tool_call(
        cls,
        assistant_msg: Dict[str, Any],
        tool_call_id: str,
    ) -> bool:
        """True when the assistant message issued a ``todo`` call with this id."""
        tool_calls = assistant_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return False

        for tool_call in tool_calls:
            if cls._get_tool_call_id_static(tool_call) != tool_call_id:
                continue
            if cls._get_tool_call_name_static(tool_call) == "todo":
                return True
        return False

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested

    _build_system_prompt = _forward("agent.system_prompt", "build_system_prompt")

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object).

        Policy owner: ``agent.message_sanitization.coalesce_tool_call_id``.
        """
        return _sanitize_coalesce_tool_call_id(tc)

    @staticmethod
    def _get_tool_call_name_static(tc) -> str:
        """Extract function name from a tool_call entry (dict or object).

        Gemini's OpenAI-compat endpoint requires the name on every ``role: tool`` message; others tolerate "".
        """
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                return fn.get("name", "") or ""
            return ""
        fn = getattr(tc, "function", None)
        return getattr(fn, "name", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    _sanitize_api_messages = _forward_static("agent.agent_runtime_helpers", "sanitize_api_messages")

    @staticmethod
    def _is_thinking_only_assistant(
        msg: Dict[str, Any],
        *,
        drop_codex_reasoning_items: bool = True,
    ) -> bool:
        """Return True if ``msg`` is an assistant turn whose only payload is reasoning (no text, no
        tool_calls).

        Providers that convert reasoning to thinking blocks reject such a message (400 "final block cannot be
        thinking"). The whole turn is dropped from the API copy; the transcript keeps the reasoning block.
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return False
        if msg.get("tool_calls"):
            return False
        # Prefill stubs are thinking-only by construction; check before content
        # inspection since repair_empty_non_final_messages may have healed content.
        if msg.get("_thinking_prefill"):
            return True
        # Does it have any actual output?
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                return False
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if block:  # non-empty non-dict string etc.
                        return False
                    continue
                btype = block.get("type")
                if btype in {"thinking", "redacted_thinking"}:
                    continue
                if btype == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return False
                    continue
                # tool_use, image, document, etc. — real payload
                return False
        elif content is not None and content != "":
            return False
        # A native compaction checkpoint makes a carrier never thinking-only, regardless of api_mode or
        # reasoning field. Checked above every reasoning branch so no carrier shape is dropped (#82108).
        from agent.native_compaction import has_compaction_checkpoint

        if has_compaction_checkpoint(msg.get("codex_reasoning_items")):
            return False
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return True
        # reasoning_details list form
        rd = msg.get("reasoning_details")
        if isinstance(rd, list) and rd:
            return True
        # Codex Responses keeps encrypted reasoning under a separate key; only real items count as
        # thinking-only, empty/junk lists fall through to generic empty-turn handling.
        codex_items = msg.get("codex_reasoning_items")
        if drop_codex_reasoning_items and isinstance(codex_items, list):
            return any(
                isinstance(item, dict) and item.get("type") == "reasoning"
                for item in codex_items
            )
        return False

    _drop_thinking_only_and_merge_users = _forward_static("agent.agent_runtime_helpers", "drop_thinking_only_and_merge_users")

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Truncate excess delegate_task tool_calls in one turn to max_concurrent_children, keeping all non-
        delegate calls.

        Returns the original list when no truncation was needed.
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < max_children:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "max_concurrent_children=%d limit",
            delegate_count - max_children, max_children,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Remove duplicate (tool_name, arguments) pairs within a single turn; first occurrence wins.

        Valid JSON arguments are canonicalized so key order / whitespace cannot evade dedup; malformed
        arguments keep their raw form. Returns the original list when nothing was removed.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            arguments = tc.function.arguments
            try:
                arguments = json.dumps(
                    json.loads(arguments), separators=(",", ":"), sort_keys=True
                )
            except (TypeError, ValueError):
                pass
            key = (tc.function.name, arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    @staticmethod
    def _uniquify_tool_call_ids(tool_calls: list) -> list:
        """Ensure every tool call in a single assistant turn has a distinct id (policy owner:
        ``message_sanitization``).

        Collisions get a deterministic ``<id>_d<n>`` suffix — never uuid4, for prompt-cache prefix stability.
        In place.
        """
        return _sanitize_uniquify_tool_call_ids(tool_calls)

    _repair_tool_call = _forward("agent.agent_runtime_helpers", "repair_tool_call")

    _invalidate_system_prompt = _forward("agent.system_prompt", "invalidate_system_prompt")

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """Generate a deterministic call_id from tool call content when the API omits one.

        Random UUIDs would make every request prefix unique and break the provider prompt cache.
        """
        return _codex_deterministic_call_id(fn_name, arguments, index)

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        return _codex_split_responses_tool_id(raw_id)

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """Build a valid Responses `function_call.id` (must start with `fc_`)."""
        return _codex_derive_responses_function_call_id(call_id, response_item_id)

    _interruptible_api_call = _forward("agent.chat_completion_helpers", "interruptible_api_call")

    # ── Unified streaming API call ─────────────────────────────────────────

    _interruptible_streaming_api_call = _forward("agent.chat_completion_helpers", "interruptible_streaming_api_call")

    _try_activate_fallback = _forward("agent.chat_completion_helpers", "try_activate_fallback")

    def _has_pending_fallback(self) -> bool:
        """Whether a fallback provider is actually available to switch to.

        Gates the "trying fallback..." status so we never announce a fallback that will not be attempted.
        Mirrors the early-return guard in ``try_activate_fallback``.
        """
        chain = getattr(self, "_fallback_chain", None) or []
        index = getattr(self, "_fallback_index", 0)
        return index < len(chain)

    # ── Per-turn primary restoration ─────────────────────────────────────

    _restore_primary_runtime = _forward("agent.agent_runtime_helpers", "restore_primary_runtime")

    _try_recover_primary_transport = _forward("agent.agent_runtime_helpers", "try_recover_primary_transport")

    _build_api_kwargs = _forward("agent.chat_completion_helpers", "build_api_kwargs")

    def _set_tool_guardrail_halt(self, decision: ToolGuardrailDecision) -> None:
        """Record the first guardrail decision that should stop this turn."""
        if decision.should_halt and self._tool_guardrail_halt_decision is None:
            self._tool_guardrail_halt_decision = decision

    def _toolguard_controlled_halt_response(self, decision: ToolGuardrailDecision) -> str:
        tool = decision.tool_name or "a tool"
        return (
            f"I stopped retrying {tool} because it hit the tool-call guardrail "
            f"({decision.code}) after {decision.count} repeated non-progressing "
            "attempts. The last tool result explains the blocker; the next step is "
            "to change strategy instead of repeating the same call."
        )

    def _append_guardrail_observation(
        self,
        tool_name: str,
        function_args: dict,
        function_result: str,
        *,
        failed: bool,
        tool_call_id: str = "",
    ) -> str:
        decision = self._tool_guardrails.after_call(
            tool_name,
            function_args,
            function_result,
            failed=failed,
        )
        # Identical-call stall guards: notice-only, observed on the RAW result (before the per-call loop
        # suffix) and applied at result construction so tool results stay append-only / cache-safe.
        stall_notice = None
        result_stub = None
        if self._stall_guards_enabled():
            try:
                observation = self._tool_guardrails.observe_call(
                    tool_name,
                    function_args,
                    function_result if isinstance(function_result, str) else None,
                    tool_call_id=tool_call_id,
                    failed=failed,
                )
                stall_notice = observation.notice
                result_stub = observation.stub
            except Exception as exc:
                logger.debug("stall-guard identical-call observation failed: %s", exc)
        # Result-reference stubbing: a 2nd+ identical call with a byte-identical FRESH result enters
        # context as a short stub. Not a cache — the tool ran; only plain-string results are stubbed.
        if result_stub and isinstance(function_result, str):
            function_result = result_stub
        if decision.action in {"warn", "halt"}:
            function_result = append_toolguard_guidance(function_result, decision)
        if decision.should_halt:
            self._set_tool_guardrail_halt(decision)
        else:
            # observe_call may have raised the identical-call streak halt
            # (hard_stop_enabled, tool-agnostic) — surface it the same way.
            streak_halt = self._tool_guardrails.halt_decision
            if streak_halt is not None and streak_halt.code == "identical_call_streak_halt":
                function_result = append_toolguard_guidance(function_result, streak_halt)
                self._set_tool_guardrail_halt(streak_halt)
        if stall_notice:
            function_result = (function_result or "") + "\n\n" + stall_notice
        return function_result

    def _stall_guards_enabled(self) -> bool:
        """Config gate for the runtime anti-stall guards (agent.stall_guards)."""
        return bool(getattr(self, "_stall_guards", True))

    def _guardrail_block_result(self, decision: ToolGuardrailDecision) -> str:
        self._set_tool_guardrail_halt(decision)
        return toolguard_synthetic_result(decision)

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        The segment planner splits the batch into maximal runs of parallel-safe calls (read-only, non-
        overlapping file targets, opted-in MCP) separated by sequential barriers; mixed batches run segment by
        segment in emission order so safe subsets stay concurrent while side-effect ordering is preserved.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if len(tool_calls) <= 1:
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_dispatch_helpers import _plan_tool_batch_segments
            _active_env = get_active_env(effective_task_id)
            _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
            segments = _plan_tool_batch_segments(tool_calls, execution_cwd=_exec_cwd)

            if len(segments) == 1:
                kind = segments[0][0]
                if kind == "parallel":
                    return self._execute_tool_calls_concurrent(
                        assistant_message, messages, effective_task_id, api_call_count
                    )
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_executor import execute_tool_calls_segmented
            return execute_tool_calls_segmented(
                self, assistant_message, messages, effective_task_id, api_call_count,
                segments=segments,
            )
        finally:
            self._executing_tools = False

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """Single call site for delegate_task dispatch; new DELEGATE_TASK_SCHEMA fields are added only here."""
        from tools.delegate_tool import (
            _strip_model_hidden_task_fields,
            delegate_task as _delegate_task,
        )
        # Top-level MODEL delegations always run in the background (handle returned, results re-enter as
        # messages). An ORCHESTRATOR SUBAGENT (depth > 0) stays synchronous — it needs results in-turn and
        # owns no gateway session. The schema-level `background` param is intentionally ignored.
        _is_subagent = getattr(self, "_delegate_depth", 0) > 0
        return _delegate_task(
            goal=function_args.get("goal"),
            context=function_args.get("context"),
            tasks=_strip_model_hidden_task_fields(function_args.get("tasks")),
            max_iterations=function_args.get("max_iterations"),
            role=function_args.get("role"),
            background=(not _is_subagent),
            action=function_args.get("action"),
            subagent_id=function_args.get("subagent_id"),
            message=function_args.get("message"),
            parent_agent=self,
        )

    _invoke_tool = _forward("agent.agent_runtime_helpers", "invoke_tool")

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """Word-wrap verbose tool output to the terminal width, wrapping each existing line separately.

        Returns ``label`` on the first line with continuation lines indented.
        """
        import shutil as _shutil
        import textwrap as _tw
        cols = _shutil.get_terminal_size((120, 24)).columns
        wrap_width = max(40, cols - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                wrapped = _tw.wrap(raw_line, width=wrap_width,
                                   break_long_words=True,
                                   break_on_hyphens=False)
                out_lines.extend(wrapped or [raw_line])
        body = ("\n" + indent).join(out_lines)
        return f"{indent}{label}{body}"

    _execute_tool_calls_concurrent = _forward("agent.tool_executor", "execute_tool_calls_concurrent")

    _execute_tool_calls_sequential = _forward("agent.tool_executor", "execute_tool_calls_sequential")

    _handle_max_iterations = _forward("agent.chat_completion_helpers", "handle_max_iterations")

    def _conversation_root_id(self) -> Optional[str]:
        """Resolve the stable conversation id for Portal usage attribution.

        Returns the session-lineage ROOT so one conversation keeps a single ``conversation=`` tag across
        compression rotation; delegate subagents resolve through ``_parent_session_id``. Falls back to the raw
        id.
        """
        sid = getattr(self, "session_id", None)
        if not sid:
            return None
        # Subagents may not have a DB row yet on their first turn; walking
        # from the parent id still lands on the right root.
        start = getattr(self, "_parent_session_id", None) or sid
        db = getattr(self, "_session_db", None)
        if db is not None:
            try:
                root = db.get_conversation_root(start)
                if root:
                    return root
            except Exception:
                logger.debug("Conversation root lineage walk failed", exc_info=True)
        return start


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-
        sonnet-4.6.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to
        trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection.
        Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses.
        Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)

    # Handle tool listing
    if list_tools:
        from model_tools import get_all_tool_names, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info

        print("📋 Available Tools & Toolsets:")
        print("-" * 50)

        # Show new toolsets system
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()

        # Group by category
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []

        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in {"web", "terminal", "vision", "creative", "reasoning"}:
                    basic_toolsets.append(entry)
                elif name in {"research", "development", "analysis", "content_creation", "full_stack"}:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)

        # Print basic toolsets
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")

        # Print composite toolsets
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")

        # Print scenario-specific toolsets
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")

        # Show legacy toolset compatibility
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")

        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")

        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return

    # Parse toolset selection arguments
    enabled_toolsets_list = None
    disabled_toolsets_list = None

    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")

    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")

    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")

    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # Use provided query or default to Python 3.13 example
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query

    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)

    # Run conversation
    result = agent.run_conversation(user_query)

    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")

    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])

    # Save sample trajectory to UUID-named file if requested
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"

        # Convert messages to trajectory format (same as batch_runner)
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )

        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }

        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # Pretty-print JSON with indent for readability
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")

    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
