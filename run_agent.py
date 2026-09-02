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

import asyncio
import base64
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import re
import sys
import tempfile
import time
import threading
import uuid
import warnings
from typing import List, Dict, Any, Optional, Callable
# `OpenAI` is a lazy proxy (SDK import costs ~240ms) that keeps the single `OpenAI(**kw)` call site
# and `patch("run_agent.OpenAI")` working. `fire` is imported only in __main__ so library imports never need
# it.
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
from agent.interrupt_compat import request_hard_interrupt


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
from agent.memory_manager import sanitize_context
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
from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.redact import redact_sensitive_text
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
from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES as _FILE_MUTATING_TOOLS,
    file_mutation_result_landed,
)
from agent.trajectory import (
    convert_scratchpad_to_think,
    save_trajectory as _save_trajectory_to_file,
)
from agent.tool_dispatch_helpers import (
    _should_parallelize_tool_batch,  # noqa: F401  # re-exported for tests that `from run_agent import _should_parallelize_tool_batch`
    _is_destructive_command,  # noqa: F401  # re-exported for tests that access `run_agent._is_destructive_command`
    _extract_parallel_scope_path,  # noqa: F401  # re-exported for tests that `from run_agent import _extract_parallel_scope_path`
    _paths_overlap,  # noqa: F401  # re-exported for tests that `from run_agent import _paths_overlap`
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,  # noqa: F401  # re-exported for tests that `from run_agent import _append_subdir_hint_to_multimodal`
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
    _extract_error_preview,
    _trajectory_normalize_msg,  # noqa: F401  # re-exported for tests that `from run_agent import _trajectory_normalize_msg`
)
from utils import atomic_json_write, base_url_host_matches, base_url_hostname, env_float, is_truthy_value, model_forces_max_completion_tokens


# Flags marking ephemeral empty-response/prefill recovery scaffolding. The loop pops these before
# appending the real response; persistence must skip them or a resumed session replays synthetic turns.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_thinking_prefill",
    # verify-on-stop / pre_verify nudges: persisting them poisons the resumed transcript and breaks
    # prompt-prefix cache reuse. The assistant candidate is NOT synthetic (#65919).
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    # kanban worker stop-guard: narrated exit without kanban_complete/block
    "_kanban_stop_synthetic",
    # dropped tool-call re-prompt pair: internal retry instruction, must not replay as user context on resume.
    "_dropped_toolcall_nudge",
)


def _is_ephemeral_scaffolding(msg: Any) -> bool:
    """Return True when ``msg`` is internal recovery scaffolding that must never be persisted to the
    durable transcript (SQLite session store or JSON log)."""
    return isinstance(msg, dict) and any(
        msg.get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS
    )


_MAX_TOOL_WORKERS = 8

# Intrinsic "already written to SQLite" marker. An id(msg) dedup set can alias a freed dict's address onto
# a new message and silently skip persisting it; a marker on the dict cannot. The `_` prefix is mandatory:
# wire sanitizers strip `_`-prefixed keys. CONTRACT (#92231): the marker asserts the dict's CONTENT is
# durable as written — any in-place mutation that must persist MUST pop it (see turn_finalizer,
# context_compressor).
_DB_PERSISTED_MARKER = "_db_persisted"


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


def _safe_session_filename_component(session_id: str) -> str:
    """Return a stable, path-safe filename component for a session ID.

    Session IDs may be untrusted (``X-Hermes-Session-Id``) and are interpolated into ``~/.hermes/sessions/``
    filenames. Collapses non ``[A-Za-z0-9_-]`` chars to ``_``, caps length, and appends a short content
    hash when sanitization changed the string so distinct IDs cannot collide.
    """
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")
    sanitized = sanitized[:96] or "session"
    if raw and sanitized == raw:
        return sanitized
    digest = hashlib.sha256(
        raw.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:12]
    return f"{sanitized}_{digest}"


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
                # as
                # unowned (#94724 backfill, #99222).
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
        /
        ``on_session_reset`` / ``on_session_start`` / ``carry_over_new_session_context``) can flush, rebind
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
        gets
        the full transition lifecycle (``_transition_context_engine_session``) instead of a bare reset.
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
        (consumed by
        the codex adapter/transport) and pops ``codex_reasoning_items`` from every assistant message.
        Returns ``{"messages": int, "items": int}`` for diagnostic logging.
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

    def _stream_diag_capture_response(
        self, diag: Dict[str, Any], http_response: Any
    ) -> None:
        """Forwarder — see ``agent.stream_diag.stream_diag_capture_response``."""
        from agent.stream_diag import stream_diag_capture_response
        stream_diag_capture_response(self, diag, http_response)

    _flatten_exception_chain = _forward_static("agent.stream_diag", "flatten_exception_chain")

    def _is_provider_stream_parse_error(self, error: BaseException) -> bool:
        """Return True for malformed provider streaming data from SDK parsers.

        The Anthropic SDK surfaces a malformed event-stream frame as a plain ``ValueError``; that is wire-
        format
        trouble, not local validation, so it follows the truncated-JSON retry path.
        """
        if getattr(self, "api_mode", None) != "anthropic_messages":
            return False
        if not isinstance(error, ValueError):
            return False
        if isinstance(error, (UnicodeEncodeError, json.JSONDecodeError)):
            return False
        message = str(error).strip().lower()
        return "expected ident at line" in message

    def _log_stream_retry(
        self,
        *,
        kind: str,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forwarder — see ``agent.stream_diag.log_stream_retry``."""
        from agent.stream_diag import log_stream_retry
        log_stream_retry(
            self, kind=kind, error=error, attempt=attempt,
            max_attempts=max_attempts, mid_tool_call=mid_tool_call, diag=diag,
        )

    def _emit_stream_drop(
        self,
        *,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forwarder — see ``agent.stream_diag.emit_stream_drop``."""
        from agent.stream_diag import emit_stream_drop
        emit_stream_drop(
            self, error=error, attempt=attempt, max_attempts=max_attempts,
            mid_tool_call=mid_tool_call, diag=diag,
        )

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

    def _check_compression_model_feasibility(self) -> None:
        """Forwarder — see ``agent.conversation_compression.check_compression_model_feasibility``."""
        from agent.conversation_compression import check_compression_model_feasibility
        check_compression_model_feasibility(self)

    def _replay_compression_warning(self) -> None:
        """Forwarder — see ``agent.conversation_compression.replay_compression_warning``."""
        from agent.conversation_compression import replay_compression_warning
        replay_compression_warning(self)

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
        arbitrary
        local proxies, which report correctly. Excludes Ollama Cloud (``ollama.com`` host, ``:cloud`` suffix):
        rewriting its stop→length manufactures false truncations and burns the continuation budget.
        """
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        base = self._base_url_lower
        # Ollama Cloud (hosted service or :cloud proxy) forwards finish_reason
        # faithfully — do not rewrite.
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

    @staticmethod
    def _summarize_background_review_actions(
        review_messages: List[Dict],
        prior_snapshot: List[Dict],
        notification_mode: str = "on",
    ) -> List[str]:
        """Forwarder — see ``agent.background_review.summarize_background_review_actions``."""
        from agent.background_review import summarize_background_review_actions
        return summarize_background_review_actions(
            review_messages,
            prior_snapshot,
            notification_mode=notification_mode,
        )

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

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some paths use an API-only user-message variant that must not leak into transcripts or resumed
        history;
        mutate the in-memory list in place so both persistence and returned history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        timestamp = getattr(self, "_persist_user_message_timestamp", None)
        platform_id = getattr(self, "_persist_user_message_platform_id", None)
        if idx is None or (
            override is None and timestamp is None and platform_id is None
        ):
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                # A plain-text override must not replace native image/audio blocks; a list override is the
                # clean
                # multimodal payload and does. Preflight compaction may re-anchor this index at a message
                # MERGED
                # with the compaction summary — overwriting it would drop the summary (see the twin guard in
                # _flush_messages_to_session_db_unlocked).
                if (
                    override is not None
                    and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                    and (
                        not isinstance(msg.get("content"), list)
                        or isinstance(override, list)
                    )
                ):
                    msg["content"] = override
                if timestamp is not None:
                    msg["timestamp"] = timestamp
                # Platform message id: load-bearing for restart drain-window recovery dedup
                # (has_platform_message_id). Stamped here too so it survives the override path.
                if platform_id is not None:
                    msg["platform_message_id"] = platform_id

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Trailing empty-response scaffolding is dropped from the live list. The persist user-message override
        is NOT applied here — ``_flush_messages_to_session_db`` writes it to the DB row only.
        """
        # Scaffolding removal mutates the live list on purpose. Close and turn-start persistence can run on
        # separate CLI threads, so the marker test-and-append below must be one critical section.
        from agent.agent_runtime_helpers import note_turn_persisted

        persist_lock = getattr(self, "_session_persist_lock", None)

        def _persist_and_drain() -> None:
            self._drop_trailing_empty_response_scaffolding(messages)
            self._session_messages = messages
            self._save_session_log(messages)
            self._flush_messages_to_session_db(messages, conversation_history)
            # Drain async token-accounting deltas at every persist point; cheap no-op when nothing queued.
            if self._session_db is not None:
                self._session_db.flush_token_counts()
            note_turn_persisted(self)

        if persist_lock is None:
            _persist_and_drain()
            return

        with persist_lock:
            _persist_and_drain()

    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Remove private empty-response retry/failure scaffolding from transcript tails.

        Also rewinds a trailing tool-result / assistant(tool_calls) pair the failed iteration left hanging;
        otherwise the next user turn lands as ``...tool, user`` and providers return empty content forever.
        """
        # Pass 1: strip the flagged scaffolding messages themselves.
        dropped_scaffolding = False
        while (
            messages
            and isinstance(messages[-1], dict)
            and (
                messages[-1].get("_empty_recovery_synthetic")
                or messages[-1].get("_empty_terminal_sentinel")
            )
        ):
            messages.pop()
            dropped_scaffolding = True

        # Pass 2: after stripping scaffolding, rewind trailing tool results and the assistant(tool_calls)
        # that produced them, so role alternation holds. Only runs when scaffolding was present.
        if not dropped_scaffolding:
            return

        # Drop any trailing tool-result messages
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "tool"
        ):
            messages.pop()

        # Drop the assistant(tool_calls) whose results were just popped — providers reject a dangling one.
        if (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            messages.pop()

    _repair_message_sequence = _forward("agent.agent_runtime_helpers", "repair_message_sequence")

    def _flush_messages_to_session_db(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ):
        """Serialize direct and turn-boundary session flushes per agent."""
        persist_lock = getattr(self, "_session_persist_lock", None)
        if persist_lock is None:
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)
        with persist_lock:
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)

    def _flush_messages_to_session_db_unlocked(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        _adoption_budget: int = 1,
    ):
        """Persist any un-flushed messages to the SQLite session store.

        Dedup is an intrinsic ``_DB_PERSISTED_MARKER`` on each written dict — not positional slices (drift
        after
        sequence repair) nor a retained ``id(msg)`` set (address reuse). ``_flushed_db_message_ids`` is only a
        one-shot seed translated to markers and cleared each flush.
        """
        # Persistence-isolated agents (background review fork) share the parent's session_id for cache
        # warmth; a write here would land the curator's harness turn in the user's real history. Hard-stop.
        if getattr(self, "_persist_disabled", False):
            return None
        if not self._session_db:
            return None
        # Persist user-message override (#48677): resolved here and applied ONLY to the written row, never
        # to the live dict — the early crash-resilience persist runs before the API call is built.
        _ov_idx = getattr(self, "_persist_user_message_idx", None)
        _ov_content = getattr(self, "_persist_user_message_override", None)
        _ov_timestamp = getattr(self, "_persist_user_message_timestamp", None)
        try:
            # Retry row creation if the earlier attempt failed transiently.
            if not self._session_db_created:
                self._ensure_db_session()
            # Positional slicing broke when repair_message_sequence shrank the list (#46053). Persistence is
            # tracked by an intrinsic per-message marker (see _DB_PERSISTED_MARKER); `_flushed_db_message_ids`
            # is honoured only as a one-shot seed translated to markers and then cleared.
            current_session_id = getattr(self, "session_id", None)
            flushed_session_id = getattr(self, "_flushed_db_message_session_id", None)
            if flushed_session_id != current_session_id or self._last_flushed_db_idx == 0:
                seed_ids = set()
            else:
                seed_ids = getattr(self, "_flushed_db_message_ids", None)
                if not isinstance(seed_ids, set):
                    seed_ids = set()
            self._flushed_db_message_session_id = current_session_id
            history_ids = {
                id(item) for item in (conversation_history or [])
                if isinstance(item, dict)
            }

            # Bounded scan: skip the identity-matched prefix of the previous flush's snapshot. Every message
            # in
            # it already got its final disposition, and no live dict has its marker popped in place.
            _scan_start = 0
            _prev_prefix = getattr(self, "_db_flush_scan_prefix", None)
            if isinstance(_prev_prefix, list):
                _limit = min(len(_prev_prefix), len(messages))
                while (
                    _scan_start < _limit
                    and messages[_scan_start] is _prev_prefix[_scan_start]
                    and bool(messages[_scan_start].get(_DB_PERSISTED_MARKER))
                ):
                    _scan_start += 1

            # Collect this flush's new rows and write them in ONE transaction
            # at the end of the scan (see append_messages_batch).
            _batch_rows: List[Dict[str, Any]] = []
            _batch_msgs: List[Dict] = []
            for _msg_idx in range(_scan_start, len(messages)):
                msg = messages[_msg_idx]
                if not isinstance(msg, dict):
                    continue
                # Never write ephemeral scaffolding: the flush is append-only, so a mid-turn persist could
                # commit a
                # synthetic turn that the end-of-turn drop cannot un-write. Skip regardless of position.
                if _is_ephemeral_scaffolding(msg):
                    continue
                if msg.get(_DB_PERSISTED_MARKER):
                    continue
                # Already-durable (history copy or caller-seeded): stamp so future flushes skip without id()
                # sets.
                if id(msg) in history_ids or id(msg) in seed_ids:
                    msg[_DB_PERSISTED_MARKER] = True
                    continue
                role = msg.get("role", "unknown")
                content = msg.get("content")
                # api_content sidecar: exact bytes sent to the API when they differ from clean content, so
                # replay
                # reproduces the sent prefix byte-for-byte.
                _row_api_content = msg.get("api_content")
                if not isinstance(_row_api_content, str):
                    _row_api_content = None
                _row_timestamp = msg.get("timestamp")
                # Apply the persist override to THIS row only. A list override replaces a noted payload; a
                # text
                # override must not erase an image/audio summary. Also match the staged CLI dict by identity —
                # the close safety-net may flush a shortened snapshot whose turn index refers to the full
                # history.
                pending_cli_message = getattr(self, "_pending_cli_user_message", None)
                is_current_turn_user = (
                    _ov_idx == _msg_idx or msg is pending_cli_message
                )
                if is_current_turn_user and msg.get("role") == "user":
                    # Preflight compaction may have re-anchored the index at a message MERGED with the
                    # compaction
                    # summary; overwriting it with the clean text would drop the summary from the durable
                    # transcript.
                    if (
                        _ov_content is not None
                        and (not isinstance(content, list) or isinstance(_ov_content, list))
                        and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                    ):
                        # Live content is what the wire sent, the override is the clean transcript; keep the
                        # sent bytes in
                        # api_content so replay matches the wire (#48677).
                        if (
                            _row_api_content is None
                            and isinstance(content, str)
                            and content != _ov_content
                        ):
                            _row_api_content = content
                        content = _ov_content
                    if _ov_timestamp is not None:
                        _row_timestamp = _ov_timestamp
                # Store the sidecar only when it actually differs.
                if _row_api_content == content:
                    _row_api_content = None
                # Load-time sanitize divergence: get_messages_as_conversation replays rows through
                # sanitize_context().strip(); capture the sent bytes when they would differ (compared in wire
                # form).
                if (
                    _row_api_content is None
                    and role in ("user", "assistant")
                    and isinstance(content, str)
                    and content
                    and sanitize_context(content).strip() != content.strip()
                ):
                    _row_api_content = content
                # Persist multimodal tool results as text summary only — base64 images bloat the DB.
                if _is_multimodal_tool_result(content):
                    content = _multimodal_text_summary(content)
                elif isinstance(content, list):
                    # List of OpenAI-style content parts: strip images, keep text.
                    _txt = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            _txt.append(str(p.get("text", "")))
                        elif isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
                            _txt.append("[screenshot]")
                    content = "\n".join(_txt) if _txt else None
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list) and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                _row = {
                    "role": role,
                    "content": content,
                    "tool_name": msg.get("tool_name"),
                    "tool_calls": tool_calls_data,
                    "tool_call_id": msg.get("tool_call_id"),
                    "finish_reason": msg.get("finish_reason"),
                    # Reasoning/codex fields are role-gated (assistant-only)
                    # inside _insert_message_rows — pass through untouched.
                    "reasoning": msg.get("reasoning"),
                    "reasoning_content": msg.get("reasoning_content"),
                    "reasoning_details": msg.get("reasoning_details"),
                    "codex_reasoning_items": msg.get("codex_reasoning_items"),
                    "codex_message_items": msg.get("codex_message_items"),
                    "_compressed_summary": bool(msg.get(COMPRESSED_SUMMARY_METADATA_KEY)),
                    "timestamp": _row_timestamp,
                    "api_content": _row_api_content,
                    # Standalone reference handoffs are always hidden so they never occupy the active user
                    # slot in
                    # retry/undo dispatch (#80622); merge-into-tail carriers keep prior visibility.
                    "display_kind": (
                        "hidden"
                        if (
                            msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                            and user_originated_turn_view(msg) is None
                            and (
                                ContextCompressor.classify_summary_content(
                                    msg.get("content")
                                )
                                == "standalone"
                                or not msg.get(
                                    "_compressed_summary_has_user_turn"
                                )
                            )
                        )
                        else msg.get("display_kind")
                    ),
                    "display_metadata": msg.get("display_metadata"),
                    # Platform message id — load-bearing for restart drain-window recovery dedup.
                    "platform_message_id": msg.get("platform_message_id"),
                }
                if isinstance(msg.get("_row_id"), int):
                    _row["_row_id"] = msg["_row_id"]
                _batch_rows.append(_row)
                _batch_msgs.append(msg)
            # One transaction for the turn's new rows. All-or-nothing pairs with the marker stamping below:
            # on failure no rows landed and no markers were stamped, so the next flush re-writes the tail.
            if _batch_rows:
                self._session_db.append_messages_batch(
                    session_id=self.session_id,
                    messages=_batch_rows,
                    compression_lock_holder=getattr(
                        self, "_active_compression_lock_holder", None
                    ),
                    turn_lease_holder=getattr(
                        self, "_active_session_turn_lease_holder", None
                    ),
                    turn_lease_ttl_seconds=getattr(
                        self, "_active_session_turn_lease_ttl_seconds", 300.0
                    )
                    or 300.0,
                )
                from agent.transcript_repair import sync_flushed_message_markers

                sync_flushed_message_markers(_batch_msgs, _batch_rows)
            # Markers are now the sole truth; reset the one-shot seed so no id() outlives this flush.
            self._flushed_db_message_ids = set()
            self._last_flushed_db_idx = len(messages)
            # Snapshot for the bounded scan above — only on full success, so
            # a partially-processed list can never be treated as settled.
            self._db_flush_scan_prefix = messages[:]
            return True
        except Exception as e:
            # Force a full re-scan on the next flush: an exception mid-loop
            # leaves messages with mixed dispositions.
            self._db_flush_scan_prefix = None
            # The only place the SQLite error is visible before it becomes a bare False — classify it so the
            # turn-end explanation can distinguish lock contention from disk-full/read-only.
            from hermes_state import (
                CompressionSessionClosedError,
                StateDbCorruptError,
                StateDbReplacedError,
                classify_persistence_error,
                divert_session_transcript_jsonl,
            )

            self._last_persistence_error_cause = classify_persistence_error(e)
            if isinstance(e, (StateDbReplacedError, StateDbCorruptError)):
                # Replaced/quarantined handle will not take this batch again — keep it on disk, not only in
                # RAM.
                try:
                    divert_session_transcript_jsonl(
                        getattr(self, "session_id", "") or "",
                        _batch_rows,
                    )
                except Exception:
                    logger.warning(
                        "JSONL divert failed after state.db %s for %s",
                        self._last_persistence_error_cause,
                        getattr(self, "session_id", None),
                        exc_info=True,
                    )
            if isinstance(e, CompressionSessionClosedError):
                # Compression race: another path rotated this session mid-write. Adopt the continuation tip
                # (get_compression_tip) ONLY when it is a different, live row, and retry exactly once; a
                # second
                # closed-parent write fails closed. tip == session_id means no continuation exists.
                if _adoption_budget > 0:
                    old_id = self.session_id
                    tip = None
                    try:
                        tip = self._session_db.get_compression_tip(old_id)
                    except Exception as tip_exc:
                        logger.warning(
                            "compression tip lookup failed for %s: %s",
                            old_id,
                            tip_exc,
                        )
                    if tip and tip != old_id:
                        tip_row = None
                        try:
                            tip_row = self._session_db.get_session(tip)
                        except Exception:
                            tip_row = None
                        if tip_row is not None and tip_row.get("ended_at") is None:
                            logger.warning(
                                "Adopted live compression tip %s for closed "
                                "session %s; retrying flush once",
                                tip,
                                old_id,
                            )
                            self.session_id = tip
                            self._flushed_db_message_ids = set()
                            self._last_flushed_db_idx = 0
                            self._compression_adoption_failed = False
                            return self._flush_messages_to_session_db_unlocked(
                                messages,
                                conversation_history,
                                _adoption_budget=0,
                            )
                # No live tip or budget exhausted: fail closed. The flag lets the turn explanation name
                # compression
                # rotation instead of misleading full-disk advice.
                self._compression_adoption_failed = True
                logger.warning("Session DB append_message failed: %s", e)
                return False
            logger.warning("Session DB append_message failed: %s", e)
            return False

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """Get messages up to (but not including) the last assistant turn.

        The rollback point when the final assistant message is incomplete or malformed.
        """
        if not messages:
            return []

        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()

        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]

    _format_tools_for_system_message = _forward("agent.system_prompt", "format_tools_for_system_message")

    _convert_to_trajectory_format = _forward("agent.agent_runtime_helpers", "convert_to_trajectory_format")

    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """Save conversation trajectory to JSONL file."""
        if not self.save_trajectories:
            return

        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)

    _extract_api_error_context = _forward_static("agent.agent_runtime_helpers", "extract_api_error_context")

    _dump_api_request_debug = _forward("agent.agent_runtime_helpers", "dump_api_request_debug")

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    @staticmethod
    def _redact_message_content(content):
        """Apply secret redaction to message content (str or list-of-parts).

        Only text fields pass through ``redact_sensitive_text``; image/binary parts are untouched.
        No-op when ``HERMES_REDACT_SECRETS`` disables redaction.
        """
        if content is None:
            return content
        if isinstance(content, str):
            return redact_sensitive_text(content)
        if isinstance(content, list):
            redacted = []
            for part in content:
                if isinstance(part, dict):
                    part = dict(part)
                    if isinstance(part.get("text"), str):
                        part["text"] = redact_sensitive_text(part["text"])
                    if isinstance(part.get("content"), str):
                        part["content"] = redact_sensitive_text(part["content"])
                redacted.append(part)
            return redacted
        return content

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """Optional per-session JSON snapshot writer (``sessions.write_json_snapshots``, default False).

        state.db is canonical; this exists for external tooling reading ``session_{sid}.json``. Rewrites the
        full
        list after every persistence point, never overwriting a larger log with fewer messages.
        """
        if not getattr(self, "_session_json_enabled", False):
            return
        messages = messages or self._session_messages
        if not messages:
            return

        # Re-derive the path each call so /branch and /compress land in the right file. Session IDs can be
        # untrusted (X-Hermes-Session-Id) — sanitize to a single traversal-free segment.
        try:
            safe_sid = _safe_session_filename_component(self.session_id)
            log_file = self.logs_dir / f"session_{safe_sid}.json"
        except Exception:
            return

        try:
            cleaned = []
            for msg in messages:
                # Mirror the SQLite flush: ephemeral recovery scaffolding is
                # internal retry state, never durable transcript content.
                if _is_ephemeral_scaffolding(msg):
                    continue
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                # Defence-in-depth: redact credentials from every message before persistence; respects
                # HERMES_REDACT_SECRETS via redact_sensitive_text (#19798, #19845).
                if "content" in msg:
                    msg = dict(msg)
                    msg["content"] = self._redact_message_content(msg.get("content"))
                cleaned.append(msg)

            # Never overwrite a larger session log with fewer messages (resumed agent with partial history).
            if log_file.exists():
                try:
                    existing = json.loads(log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": redact_sensitive_text(self._cached_system_prompt or ""),
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")

    def interrupt(
        self,
        message: Optional[str] = None,
        *,
        hard_cancel: bool = False,
        tool_reason: Optional[str] = None,
        require_generation: Optional[int] = None,
    ) -> bool:
        """Request the agent to interrupt its current tool-calling loop (call from another thread).

        ``message``: new message to include in the response context. ``hard_cancel``: explicit stop;
        compression
        may honor it even while ordinary interrupts are masked. ``tool_reason``: trusted fixed category safe
        for
        tool output. ``require_generation``: activity-generation claim — the interrupt is published only if
        the
        turn's generation still matches at the final mutation edge (claim reserved under the activity lock,
        consumed together with the first observable publication); returns False if the turn resumed meanwhile.
        """
        if require_generation is not None:
            # RESERVE the abort's generation claim under the SAME lock `_touch_activity` stamps with. Real
            # progress invalidates it; it is CONSUMED at the final mutation edge, so a resumed turn abandons
            # the abort.
            with self._liveness_activity_lock():
                if (
                    getattr(self, "_turn_liveness_activity_generation", 0)
                    != require_generation
                ):
                    return False
                self._turn_liveness_abort_claim = require_generation

        # A hard stop and redirect share one lock so /stop cannot race with an
        # accepted correction and accidentally turn itself into a retry.
        def _wait_for_compression_commit() -> None:
            # Pre-claim half of hard-cancel admission (#99758 P1): wait out a commit that already crossed its
            # boundary but mutate NOTHING — cancelling a pending fence is irreversible and must wait until the
            # generation claim survived the final mutation edge (_cancel_pending_compression_commit).
            fence = vars(self).get("_active_compression_commit_fence")
            if fence is None:
                return
            if not getattr(fence, "commit_in_flight", False):
                # No commit in flight — cancel_before_commit here WOULD cancel the pending commit; leave it to
                # the
                # destructive half.
                return
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # A commit holds the fence lock through finish_commit: this blocks until it finishes and
                    # returns
                    # False WITHOUT setting _cancelled.
                    cancel_before_commit(fence)
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence wait failed",
                        exc_info=True,
                    )

        def _cancel_pending_compression_commit() -> None:
            # Destructive half of hard-cancel admission (#99758 P1): runs only AFTER the claim survived, so a
            # declined abort never leaves the fence cancelled. A commit that started meanwhile owns the fence
            # and completes on its own; only a still-pending commit is cancelled here.
            fence = vars(self).get("_active_compression_commit_fence")
            if fence is None:
                return
            if getattr(fence, "commit_in_flight", False):
                return
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # Marks the fence cancelled (or waits out a just-started commit) without touching the
                    # hard-stop
                    # Event, which was published at the claim edge.
                    cancel_before_commit(fence)
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence admission failed",
                        exc_info=True,
                    )

        def _publish_interrupt_state() -> None:
            self._interrupt_requested = True
            self._interrupt_message = message
            self._tool_interrupt_reason = tool_interrupt_reason
            if hard_cancel:
                _hard_event = getattr(
                    self, "_hard_interrupt_requested", None
                )
                if _hard_event is not None:
                    _hard_event.set()

        def _consume_claim_and_publish_first_state() -> bool:
            # Final mutation edge: claim consumption and the FIRST observable interrupt publication are ONE
            # activity-lock critical section, so either the claim survives and commits before any later
            # activity stamp, or the stamp landed first and the abort declines without publishing.
            if require_generation is None:
                # No claim to race: publish WITHOUT the liveness lock. Bare AIAgent stand-ins in other suites
                # lack
                # the liveness seam and would AttributeError.
                _publish_interrupt_state()
                return True
            with self._liveness_activity_lock():
                if (
                    getattr(self, "_turn_liveness_abort_claim", None)
                    != require_generation
                ):
                    return False
                self._turn_liveness_abort_claim = None
                _publish_interrupt_state()
            return True

        # Tool cancellation attribution stays separate from _interrupt_message, which may carry the user's
        # full next message.
        tool_interrupt_reason = (
            (tool_reason or "explicit stop requested")
            if hard_cancel
            else ("user sent a new message" if message else "user interrupt")
        )

        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                # The blocking in-flight-commit wait runs BEFORE the atomic claim edge (redirect lock still
                # held);
                # the destructive pending-commit cancel runs AFTER the claim survives (#99758 P1).
                if hard_cancel:
                    _wait_for_compression_commit()
                if not _consume_claim_and_publish_first_state():
                    return False
                if hard_cancel:
                    _cancel_pending_compression_commit()
                self._pending_redirect = None
        else:
            if hard_cancel:
                _wait_for_compression_commit()
            if not _consume_claim_and_publish_first_state():
                return False
            if hard_cancel:
                _cancel_pending_compression_commit()
            self._pending_redirect = None

        # Codex app-server owns its model/tool loop and watches a private
        # interrupt event rather than Hermes' per-thread flag.
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _request_interrupt = getattr(_codex_session, "request_interrupt", None)
            if callable(_request_interrupt):
                try:
                    _request_interrupt()
                except Exception:
                    logger.debug(
                        "Failed to interrupt Codex app-server turn",
                        exc_info=True,
                    )

        # Cron turns request on the conversation thread (no nested interrupt-worker deadlock); their client
        # is registered here so this cross-thread interrupt can still shut the sockets.
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("interrupt_abort")
            except Exception:
                logger.debug("Failed to abort active inline request", exc_info=True)
        # Scope the tool interrupt to this agent's execution thread so other in-process agents are unaffected.
        if self._execution_thread_id is not None:
            _set_interrupt(
                True,
                self._execution_thread_id,
                reason=tool_interrupt_reason,
            )
            self._interrupt_thread_signal_pending = False
        else:
            # Interrupt arrived before run_conversation bound the execution thread: defer the tool-level
            # signal instead of targeting the caller thread.
            self._interrupt_thread_signal_pending = True
        # Fan out to concurrent-tool worker tids: is_interrupted() inside a tool only sees its own tid, so
        # without this a hung concurrent tool runs to its own timeout. getattr covers __init__-less stubs.
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(True, _wtid, reason=tool_interrupt_reason)
                except Exception:
                    pass
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                if hard_cancel:
                    request_hard_interrupt(
                        child,
                        message,
                        tool_reason=tool_interrupt_reason,
                    )
                else:
                    child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
        return True

    def hard_interrupt(
        self,
        message: Optional[str] = None,
        *,
        tool_reason: Optional[str] = None,
    ) -> None:
        """Request an explicit stop while preserving the ``interrupt()`` ABI.

        Frontends feature-detect this and fall back to legacy ``interrupt()`` for third-party agents.
        """
        # Bypass dynamic dispatch: legacy subclasses may override interrupt(message=None) without hard_cancel.
        AIAgent.interrupt(
            self,
            message,
            hard_cancel=True,
            tool_reason=tool_reason,
        )

    def clear_interrupt(self, *, preserve_redirect: bool = False) -> bool:
        """Clear the interrupt request and per-thread tool signal.

        ``preserve_redirect`` is only for the conversation loop rebuilding the same logical turn after
        cancelling a model request; public hard-stop paths clear everything.
        """
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                if preserve_redirect and not self._pending_redirect:
                    return False
                self._interrupt_requested = False
                self._interrupt_message = None
                self._tool_interrupt_reason = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                if not preserve_redirect:
                    self._pending_redirect = None
        else:
            if preserve_redirect and not getattr(self, "_pending_redirect", None):
                return False
            self._interrupt_requested = False
            self._interrupt_message = None
            self._tool_interrupt_reason = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            if not preserve_redirect:
                self._pending_redirect = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        # Also clear worker-thread bits so no stale interrupt survives a turn boundary onto a recycled tid.
        # getattr covers __init__-less test stubs.
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(False, _wtid)
                except Exception:
                    pass
        # A hard interrupt supersedes any pending /steer — its target iteration will no longer happen.
        _steer_lock = getattr(self, "_pending_steer_lock", None)
        if _steer_lock is not None:
            with _steer_lock:
                self._pending_steer = None
        return True

    def steer(self, text: str) -> bool:
        """Inject user text into the next tool result without interrupting the current tool.

        The text is appended to the LAST tool result once the batch finishes, so the model sees it on its next
        iteration. Thread-safe; multiple calls concatenate with newlines. Returns False for empty text.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            # __init__-less test stubs: fall back to a direct attribute set.
            existing = getattr(self, "_pending_steer", None)
            self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
            return True
        with _lock:
            if self._pending_steer:
                self._pending_steer = self._pending_steer + "\n" + cleaned
            else:
                self._pending_steer = cleaned
        return True

    def redirect(self, text: str) -> bool:
        """Redirect the active turn without converting it into a new task.

        During a model request this cancels only that request: completed messages/tool results are kept, the
        displayed partial reasoning becomes assistant context, the correction is appended as a real user
        message,
        and the loop retries. During tool execution it degrades to ``steer()``; Codex app-server uses native
        ``turn/steer``. Returns False when there is no live turn or the text is empty.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()

        # Codex owns its internal reasoning/tool loop, so use its first-class
        # active-turn steering protocol rather than interrupting the subprocess.
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _native_steer = getattr(_codex_session, "request_steer", None)
            if callable(_native_steer):
                _redirect_lock = getattr(self, "_pending_redirect_lock", None)
                if _redirect_lock is not None:
                    with _redirect_lock:
                        if self._interrupt_requested:
                            return False
                elif self._interrupt_requested:
                    return False
                try:
                    return bool(_native_steer(cleaned))
                except Exception:
                    logger.debug("Codex app-server turn/steer failed", exc_info=True)
                    return False

        # Never kill a tool to deliver guidance; the steer drain puts it on the final tool result.
        if getattr(self, "_executing_tools", False):
            return self.steer(cleaned)

        _model_active = getattr(self, "_model_request_active", None)
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            if _model_active is None or not _model_active.is_set():
                return False
            existing = getattr(self, "_pending_redirect", None)
            if self._interrupt_requested and not existing:
                return False
            self._pending_redirect = (
                f"{existing}\n\n[Additional user correction]\n{cleaned}"
                if existing
                else cleaned
            )
            self._interrupt_requested = True
            self._interrupt_message = None
        else:
            with _redirect_lock:
                if _model_active is None or not _model_active.is_set():
                    # The response completed before we acquired the state lock.
                    # Reject so the surface queues a new turn.
                    return False
                if self._interrupt_requested and not self._pending_redirect:
                    return False
                if self._pending_redirect:
                    self._pending_redirect = (
                        f"{self._pending_redirect}\n\n"
                        f"[Additional user correction]\n{cleaned}"
                    )
                else:
                    self._pending_redirect = cleaned
                self._interrupt_requested = True
                self._interrupt_message = None

        # Interrupt only the model request. Do not fan out to tool workers or
        # child agents as interrupt() does.
        _execution_thread_id = getattr(self, "_execution_thread_id", None)
        if _execution_thread_id is not None:
            _set_interrupt(True, _execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_thread_signal_pending = True
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("redirect_abort")
            except Exception:
                logger.debug("Failed to abort request for redirect", exc_info=True)
        return True

    def _has_pending_redirect(self) -> bool:
        """Return whether an active-turn redirect is waiting to be applied."""
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            return bool(getattr(self, "_pending_redirect", None))
        with _redirect_lock:
            return bool(self._pending_redirect)

    def _drain_pending_redirect(self) -> Optional[str]:
        """Return and clear pending active-turn correction text."""
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            text = getattr(self, "_pending_redirect", None)
            self._pending_redirect = None
            return text
        with _redirect_lock:
            text = self._pending_redirect
            self._pending_redirect = None
        return text

    def _drain_pending_steer(self) -> Optional[str]:
        """Return the pending steer text (if any) and clear the slot; None when nothing is pending."""
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            text = getattr(self, "_pending_steer", None)
            self._pending_steer = None
            return text
        with _lock:
            text = self._pending_steer
            self._pending_steer = None
        return text

    def _record_file_mutation_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        is_error: bool,
    ) -> None:
        """Record a ``write_file`` / ``patch`` outcome for the turn-end verifier.

        Failures store ``{path: {error_preview, tool}}``; a later success on the same path removes the entry.
        No-op when the per-turn state dict is not initialised (tool dispatched outside ``run_conversation``).
        """
        if tool_name not in _FILE_MUTATING_TOOLS:
            return
        state = getattr(self, "_turn_failed_file_mutations", None)
        if state is None:
            return
        targets = _extract_file_mutation_targets(tool_name, args)
        if not targets:
            return
        landed = file_mutation_result_landed(tool_name, result)
        if landed:
            landed_paths = _extract_landed_file_mutation_paths(tool_name, args, result)
            changed = getattr(self, "_turn_file_mutation_paths", None)
            if changed is not None:
                changed.update(landed_paths)
            # Feed the checkpoint agent-write ledger so /rollback's safe mode
            # can tell Hermes-authored content from later user hand-edits.
            mgr = getattr(self, "_checkpoint_mgr", None)
            if mgr is not None and getattr(mgr, "enabled", False):
                for _p in landed_paths:
                    try:
                        mgr.record_agent_write(_p)
                    except Exception:
                        pass
        if is_error and not landed:
            preview = _extract_error_preview(result)
            for path in targets:
                # Keep the FIRST error per path unless a later success replaces it.
                if path not in state:
                    state[path] = {
                        "tool": tool_name,
                        "error_preview": preview,
                    }
        else:
            for path in targets:
                state.pop(path, None)

    def _file_mutation_verifier_enabled(self) -> bool:
        """Check whether the per-turn file-mutation verifier footer is on.

        ``display.file_mutation_verifier`` (default True), cached per agent; ``HERMES_FILE_MUTATION_VERIFIER``
        overrides on every call and is never cached. A method so tests can patch one seam.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_FILE_MUTATION_VERIFIER")
            if env is not None:
                return env.strip().lower() not in {"0", "false", "no", "off"}
            cached = getattr(self, "_file_mutation_verifier_enabled_cache", None)
            if cached is not None:
                return cached
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "file_mutation_verifier" in _display:
                enabled = bool(_display.get("file_mutation_verifier"))
            else:
                enabled = True  # safe default: verifier on
            self._file_mutation_verifier_enabled_cache = enabled
            return enabled
        except Exception:
            pass
        return True  # safe default: verifier on

    # Bare absolute / home / Windows-drive paths in a footer line. Mirrors the gateway's
    # extract_local_files detector so anything it WOULD auto-attach is backticked first (#35584).
    _FOOTER_PATH_RE = re.compile(
        r"(?<![/:\w.`])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.[\w]+",
    )

    @classmethod
    def _neutralize_footer_paths(cls, text: str) -> str:
        """Wrap bare file paths in backticks so the gateway's ``extract_local_files`` never auto-attaches
        them.

        The extractor skips paths inside inline-code spans. Already-backticked paths are left alone (no
        double-wrap).
        """
        if not text:
            return text
        return cls._FOOTER_PATH_RE.sub(lambda m: f"`{m.group(0)}`", text)

    @classmethod
    def _format_file_mutation_failure_footer(cls, failed: Dict[str, Dict[str, Any]]) -> str:
        """Render the per-turn failed-mutation dict as a user-facing footer.

        Up to 10 paths with their first error preview, then an overflow count; empty string when nothing
        failed.
        Every path is backtick-wrapped via ``_neutralize_footer_paths`` so protected files cannot be auto-
        delivered.
        """
        if not failed:
            return ""
        lines = [
            "⚠️ File-mutation verifier: "
            f"{len(failed)} file(s) were NOT modified this turn despite any "
            "wording above that may suggest otherwise. Run `git status` or "
            "`read_file` to confirm."
        ]
        shown = 0
        for path, info in failed.items():
            if shown >= 10:
                break
            preview = (info.get("error_preview") or "").strip()
            tool = info.get("tool") or "patch"
            if preview:
                lines.append(f"  • `{path}` — [{tool}] {preview}")
            else:
                lines.append(f"  • `{path}` — [{tool}] failed")
            shown += 1
        remaining = len(failed) - shown
        if remaining > 0:
            lines.append(f"  • … and {remaining} more")
        # Neutralize paths the preview echoed; the lookbehind prevents double-wrapping the bullet path.
        return cls._neutralize_footer_paths("\n".join(lines))

    def _turn_completion_explainer_enabled(self) -> bool:
        """Check whether the end-of-turn completion explainer footer is on.

        ``display.turn_completion_explainer`` (default True), cached per agent;
        ``HERMES_TURN_COMPLETION_EXPLAINER``
        overrides on every call and is never cached. Mirrors ``_file_mutation_verifier_enabled``.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_TURN_COMPLETION_EXPLAINER")
            if env is not None:
                return env.strip().lower() not in {"0", "false", "no", "off"}
            cached = getattr(self, "_turn_completion_explainer_enabled_cache", None)
            if cached is not None:
                return cached
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "turn_completion_explainer" in _display:
                enabled = bool(_display.get("turn_completion_explainer"))
            else:
                enabled = True  # safe default: explainer on
            self._turn_completion_explainer_enabled_cache = enabled
            return enabled
        except Exception:
            pass
        return True  # safe default: explainer on

    @staticmethod
    def _format_turn_completion_explanation(
        turn_exit_reason: str, persistence_cause: Optional[str] = None
    ) -> str:
        """Render a user-facing explanation for an abnormal turn ending.

        Maps ``turn_exit_reason`` to an actionable message so a turn with no usable reply is never silent.
        ``persistence_cause`` refines ``session_persistence_failed`` wording (lock contention ≠ disk full).
        Returns "" for non-abnormal reasons so callers can concatenate unconditionally.
        """
        if not turn_exit_reason:
            return ""
        reason = str(turn_exit_reason)

        # Normal completion — stay quiet.  ``text_response(...)`` is the
        # healthy terminal; anything that produced a real reply is fine.
        if reason.startswith("text_response"):
            return ""

        prefix = "⚠️ No reply: "
        if reason == "empty_response_exhausted":
            return (
                prefix
                + "the model returned empty content after retries and any "
                "fallback providers. Try `continue`, switch model/provider, "
                "or inspect the tool output above."
            )
        if reason == "all_retries_exhausted_no_response":
            return (
                prefix
                + "all API retries were exhausted before a response was "
                "produced (provider errors / rate limits). Try `continue` "
                "or switch provider."
            )
        if reason == "partial_stream_recovery":
            return (
                prefix
                + "streaming stopped early and only a partial response was "
                "recovered. Send `continue` to resume from where it stopped."
            )
        if reason == "fallback_prior_turn_content":
            return (
                prefix
                + "no new content was produced this turn; showing recovered "
                "prior context. Send `continue` to retry."
            )
        if reason == "interrupted_during_api_call":
            return (
                prefix
                + "the request was interrupted mid-call before a reply was "
                "received. Send `continue` to retry."
            )
        if reason == "budget_exhausted":
            return (
                prefix
                + "the per-turn iteration/cost budget was exhausted before a "
                "final answer. Send `continue` to keep going."
            )
        if reason == "ollama_runtime_context_too_small":
            return (
                prefix
                + "the local model's context window was too small to finish. "
                "Increase the context size or use a larger model."
            )
        if reason.startswith("max_iterations_reached"):
            return (
                prefix
                + "the maximum tool-iteration limit was reached before a "
                "final answer. Send `continue` to keep going, or raise "
                "`max_iterations`."
            )
        if reason.startswith("error_near_max_iterations"):
            return (
                prefix
                + "an error occurred near the iteration limit before a final "
                "answer. Check the tool output above, then send `continue`."
            )
        if reason.startswith("repeated_outer_errors"):
            return (
                prefix
                + "the turn kept failing with repeated errors and was stopped "
                "early instead of retrying forever. Check the errors above, "
                "then send `continue` to retry."
            )
        if reason == "pending_tool_result":
            return (
                prefix
                + "the turn stopped while a tool result was still pending and "
                "the model produced no follow-up text. Send `continue` to "
                "let it summarize."
            )
        if reason == "session_persistence_failed":
            cause = persistence_cause or "unknown"
            if cause == "compression":
                return (
                    prefix
                    + "the turn was stopped because another process was "
                    "compressing this session. Your message should already be "
                    "saved — please send it again after compression completes."
                )
            if cause == "compression_closed":
                return (
                    prefix
                    + "the turn was stopped because this session was rotated "
                    "by context compression and its live continuation could "
                    "not be adopted. The storage itself is healthy — refresh "
                    "the client (or start a new turn) so it picks up the new "
                    "session id, then send your message again."
                )
            if cause == "turn_lease":
                return (
                    prefix
                    + "the turn was stopped because another Hermes process "
                    "took over this session. Your reply was not saved — wait "
                    "for the other process to finish, then send your message "
                    "again."
                )
            if cause == "locked":
                return (
                    prefix
                    + "the turn was stopped because session storage was busy "
                    "(another Hermes process was writing to the state "
                    "database). Your message should already be saved — "
                    "please send it again in a moment."
                )
            if cause == "replaced":
                return (
                    prefix
                    + "the turn was stopped because the state database file "
                    "was replaced underneath this process. Do not run "
                    "`hermes doctor --fix` or in-place FTS repair — stop "
                    "the process, restore the intended state.db, then "
                    "restart. Unwritten messages were diverted to "
                    "sessions/<session_id>.jsonl and, on the gateway, "
                    "pending_messages/pending-*.json."
                )
            if cause == "corrupt":
                return (
                    prefix
                    + "the turn was stopped because the state database "
                    "reported structural corruption (the transcript would "
                    "have been lost on restart). Freeing disk space will "
                    "not help. Recovery options:\n"
                    "1. Run `hermes doctor --fix`\n"
                    "2. Salvage with: sqlite3 ~/.hermes/state.db \".recover\" "
                    "(then replace state.db)\n"
                    "3. Restore from a backup in ~/.hermes/backups/\n"
                    "Then send your message again."
                )
            if cause == "disk":
                return (
                    prefix
                    + "the turn was stopped because session storage could not "
                    "be written (the transcript would have been lost on "
                    "restart). This is often a full disk — free some space "
                    "(or fix state.db permissions), then send your message "
                    "again."
                )
            return (
                prefix
                + "the turn was stopped because session storage could not be "
                "written (the transcript would have been lost on restart). "
                "Check the state database health (`hermes doctor`), then "
                "send your message again."
            )
        # Unknown/diagnostic-only reasons (e.g. "unknown", guardrail_halt
        # which already surfaces its own message) — don't second-guess.
        return ""

    _apply_pending_steer_to_tool_results = _forward("agent.agent_runtime_helpers", "apply_pending_steer_to_tool_results")

    def _liveness_activity_lock(self) -> "threading.Lock":
        """Shared lock for the activity clock and its generation counter.

        ``_touch_activity`` stamps under it and the liveness watchdog samples/commits under it, so a stall
        observation can never abort a turn that resumed in between. Lazy so ``__new__``-built doubles work.
        """
        _lock = getattr(self, "_turn_liveness_activity_lock", None)
        if _lock is None:
            _lock = threading.Lock()
            self._turn_liveness_activity_lock = _lock
        return _lock

    def _touch_activity(
        self,
        desc: str,
        *,
        provenance: Optional[ActivityProvenance] = None,
        force_persist: bool = False,
    ) -> None:
        """Update the last-activity timestamp and description (thread-safe).

        Bumps a monotonic generation under ``_liveness_activity_lock`` so the watchdog can bind a stall
        observation
        to the exact ``(generation, timestamp)`` it sampled. Also bridges (rate-limited, best-effort) to the
        kanban
        heartbeat when this is a dispatcher-spawned worker, and to the durable SessionDB activity projection.
        ``provenance`` names special writers (compression); ``force_persist`` bypasses the SessionDB rate
        limit.
        """
        from agent.session_activity import (
            bound_activity_description,
            normalize_activity_provenance,
            reset_session_activity_persist_window,
        )

        # Lazy per-instance lock, inline so SimpleNamespace doubles binding _touch_activity without the
        # class keep working (tests/run_agent/test_session_activity_persist.py).
        _clock_lock = getattr(self, "_turn_liveness_activity_lock", None)
        if _clock_lock is None:
            _clock_lock = threading.Lock()
            self._turn_liveness_activity_lock = _clock_lock
        with _clock_lock:
            self._turn_liveness_activity_generation = (
                getattr(self, "_turn_liveness_activity_generation", 0) + 1
            )
            self._last_activity_ts = time.time()
            self._last_activity_desc = bound_activity_description(desc)
            self._last_activity_provenance = normalize_activity_provenance(provenance)
            # Real progress invalidates a reserved abort claim; an in-flight watchdog interrupt must abandon
            # itself at the final mutation edge.
            self._turn_liveness_abort_claim = None
        if os.environ.get("HERMES_KANBAN_TASK"):
            try:
                from tools.kanban_tools import (
                    heartbeat_current_worker_from_env,
                    inject_new_comments_from_env,
                )
                heartbeat_current_worker_from_env()
                # Fold any new operator notes into the running turn (OUT-OF-BAND
                # steer) so the user can talk to a live task without a restart.
                inject_new_comments_from_env(self)
            except Exception:
                # Never let the bridge break the loop; this guard covers import-time failures.
                pass
        if force_persist:
            reset_session_activity_persist_window(self)
        self._persist_session_activity_if_due()

    def _persist_session_activity_if_due(self) -> None:
        """Best-effort durable activity heartbeat for SessionDB consumers.

        Cadence pinned by ``SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS`` (config-independent). Fail-open:
        a failed write never raises into the agent loop.
        """
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        touch = getattr(session_db, "touch_session_activity", None)
        if not callable(touch):
            return
        from agent.session_activity import (
            SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS,
            normalize_activity_provenance,
        )

        now_mono = time.monotonic()
        last_mono = getattr(self, "_session_activity_last_persist_mono", 0.0)
        if (now_mono - last_mono) < SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS:
            return
        self._session_activity_last_persist_mono = now_mono
        try:
            touch(
                session_id,
                getattr(self, "_last_activity_ts", None),
                description=getattr(self, "_last_activity_desc", None),
                provenance=normalize_activity_provenance(
                    getattr(self, "_last_activity_provenance", None)
                ),
            )
        except Exception:
            # Heartbeat is observation-only; never let its I/O break the loop.
            logger.debug(
                "session activity heartbeat write failed (ignored)",
                exc_info=True,
            )

    def _reset_activity_labels_after_turn(self) -> None:
        """Drop mid-turn activity labels once the turn is no longer running.

        Keeps ``_last_activity_ts`` so idle/watchdog clocks stay continuous across turns; clears description +
        provenance so idle agents / SessionDB listings stop advertising the last mid-turn stamp.
        """
        from agent.session_activity import ActivityProvenance

        self._last_activity_desc = ""
        self._last_activity_provenance = ActivityProvenance.UNKNOWN
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        clear = getattr(session_db, "clear_session_activity_labels", None)
        if not callable(clear):
            return
        try:
            clear(session_id)
        except Exception:
            # Never let durable cleanup I/O break turn teardown.
            pass

    def _capture_rate_limits(self, http_response: Any) -> None:
        """Parse x-ratelimit-* headers from an HTTP response and cache the state.

        Called after each streaming call; the httpx Response is available as ``stream.response``.
        """
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

    def _capture_anthropic_response_headers(self, http_response: Any) -> None:
        """Capture out-of-band state from Anthropic Messages response headers.

        The SDK's aggregated ``Message`` drops headers, where Portal puts rate-limit and credits state. Fail-
        open.
        """
        self._capture_rate_limits(http_response)
        self._capture_credits(http_response)

    def _capture_credits(self, http_response: Any) -> None:
        """Parse x-nous-credits-* headers, cache CreditsState, fire threshold notices.

        The PARSE is swallowed (miss → keep last-known); the notice EVALUATION is a separate block that WARNS
        on
        failure so a depletion-notice bug cannot vanish silently.
        """
        # Dev test fixture (HERMES_DEV_CREDITS_FIXTURE): inject a chosen notice state
        # each turn for repeatable testing, bypassing real headers. Throwaway scaffolding.
        try:
            from agent.credits_tracker import dev_fixture_credits_state
            _fixture = dev_fixture_credits_state()
        except Exception:
            _fixture = None
        if _fixture is not None:
            self._credits_state = _fixture
            if self._credits_session_start_micros is None:
                self._credits_session_start_micros = _fixture.remaining_micros
            _latch = getattr(self, "_credits_latch", None)
            if isinstance(_latch, dict):
                # Only seen_below_90 — priming seen_grant_unspent would fire grant_spent on first observation.
                _latch["seen_below_90"] = True  # let warn90 fire without a real crossing
            _used = _fixture.used_fraction
            logger.info(
                "credits ▸ [FIXTURE] remaining=%d (%s) · paid=%s · denom=%s · used=%s "
                "(real headers bypassed — `echo clear` / unset HERMES_DEV_CREDITS_FIXTURE to restore)",
                _fixture.remaining_micros,
                _fixture.remaining_usd or "?",
                _fixture.paid_access,
                _fixture.denominator_kind,
                ("%.0f%%" % (_used * 100)) if _used is not None else "n/a",
            )
            self._emit_credits_notices()
            return
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        _dev = is_truthy_value(os.environ.get("HERMES_DEV_CREDITS"))

        # ── Parse (fail-open → miss; never overwrite good state with None) ──
        try:
            from agent.credits_tracker import parse_credits_headers
            state = parse_credits_headers(headers, provider=self.provider)
        except Exception:
            return  # parse error → treat as a miss, keep last-known
        if state is None:
            if _dev:
                logger.info(
                    "credits ▸ response had no valid x-nous-credits-* headers "
                    "(miss — producer off / non-Nous path / >TTL stale)"
                )
            return

        # retain-last-known: only overwrite on a fresh valid parse
        self._credits_state = state
        # Latch session-start remaining the first time we ever see a header
        if self._credits_session_start_micros is None:
            self._credits_session_start_micros = state.remaining_micros
        if _dev:
            # HERMES_DEV_CREDITS: stream each capture to agent.log — watch live with
            # `hermes logs -f` (grep 'credits ▸'). Dev-only; silent for normal users.
            spent = self.get_credits_spent_micros()
            used = state.used_fraction
            logger.info(
                "credits ▸ remaining=%d (%s) · paid=%s · denom=%s · used=%s "
                "· Δspent=%s · age=%s%s",
                state.remaining_micros,
                state.remaining_usd or "?",
                state.paid_access,
                state.denominator_kind,
                ("%.0f%%" % (used * 100)) if used is not None else "n/a",
                ("%.1f¢" % (spent / 10000)) if spent is not None else "n/a",
                ("%.0fs" % state.age_seconds) if state.age_seconds != float("inf") else "n/a",
                (" · disabled=%s" % state.disabled_reason) if state.disabled_reason else "",
            )

        # Threshold notices — shared with the cold-start seed (see _emit_credits_notices).
        self._emit_credits_notices()

    def _emit_credits_notices(self) -> None:
        """Run the threshold policy on the current credits state and emit notices.

        Shared by the warm path and the cold-start seed so an already-depleted session warns immediately. Runs
        only
        when a notice consumer is bound. WARNS on failure. Emits clears FIRST so depleted lands last (latest-
        wins slot).
        """
        if getattr(self, "notice_callback", None) is None and getattr(self, "notice_clear_callback", None) is None:
            return
        if not self._credits_notices_enabled():
            return
        state = getattr(self, "_credits_state", None)
        if state is None:
            return
        try:
            from agent.credits_tracker import evaluate_credits_notices, is_free_tier_model, new_credits_latch
            latch = getattr(self, "_credits_latch", None)
            if latch is None:
                latch = self._credits_latch = new_credits_latch()
            # Free-model gate: a depleted account can still inference on a free model. Local data only.
            model_is_free = is_free_tier_model(
                getattr(self, "model", "") or "",
                getattr(self, "base_url", "") or "",
            )
            to_show, to_clear = evaluate_credits_notices(state, latch, model_is_free=model_is_free)
            for key in to_clear:        # clears FIRST …
                self._emit_notice_clear(key)
            for notice in to_show:      # … then shows (depleted lands last in a latest-wins slot)
                self._emit_notice(notice)
        except Exception:
            logger.warning("credits notice evaluation/emit failed", exc_info=True)

    def _credits_notices_enabled(self) -> bool:
        """Whether credits notices are enabled (``display.credits_notices``).

        Read once per agent and cached (governs UI noise, not correctness); fail-open True.
        """
        cached = getattr(self, "_credits_notices_enabled_cache", None)
        if cached is not None:
            return cached
        enabled = True
        try:
            from hermes_cli.config import load_config as _load_config
            _cfg = _load_config() or {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "credits_notices" in _display:
                enabled = bool(_display.get("credits_notices"))
        except Exception:
            enabled = True
        self._credits_notices_enabled_cache = enabled
        return enabled

    def get_credits_state(self):
        """Return the last captured CreditsState, or None."""
        return self._credits_state

    def get_credits_spent_micros(self):
        """Session-cumulative micros spent = first_seen_remaining - current_remaining. None if no data."""
        if self._credits_session_start_micros is None or self._credits_state is None:
            return None
        return self._credits_session_start_micros - self._credits_state.remaining_micros

    def _check_openrouter_cache_status(self, http_response: Any) -> None:
        """Read X-OpenRouter-Cache-Status from response headers and log it; HITs count in ``_or_cache_hits``."""
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            status = headers.get("x-openrouter-cache-status")
            if not status:
                return
            if status.upper() == "HIT":
                self._or_cache_hits += 1
                logger.info("OpenRouter response cache HIT (total: %d)", self._or_cache_hits)
            else:
                logger.debug("OpenRouter response cache %s", status.upper())
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_activity_summary(self) -> dict:
        """Return a snapshot of the agent's current activity for diagnostics.

        Exposes ``last_activity_at`` / ``last_activity_description`` / ``last_activity_provenance`` plus the
        short aliases existing gateway and delegate readers use.
        """
        from agent.session_activity import (
            ActivityProvenance,
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
        are
        skipped entirely: partial output is not durable truth, and a prefetch keyed on it would fire against
        stale context. Strictly best-effort — an offline backend must never block the response.
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
        task_id, so
        process_registry entries, terminal sandbox, browser daemon, computer-use backend and memory provider
        are
        kept. Closes the OpenAI/httpx pool and active child subagents. Idempotent; distinct from ``close()``.
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
        and
        client connections. Each step is independently guarded so one failure does not block the rest.
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

        # 7. Free conversation history proactively (close() is the hard teardown; callers may still hold
        # the closed agent).
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
        results
        paired with an earlier assistant ``todo`` tool call count: caller-supplied history could otherwise
        seed
        the store with a forged bare ``role: tool`` message (GHSA-5g4g-6jrg-mw3g).
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

    def _build_system_prompt(self, system_message: str = None) -> str:
        """Forwarder — see ``agent.system_prompt.build_system_prompt``."""
        from agent.system_prompt import build_system_prompt
        return build_system_prompt(self, system_message=system_message)

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

    def _invalidate_system_prompt(self):
        """Forwarder — see ``agent.system_prompt.invalidate_system_prompt``."""
        from agent.system_prompt import invalidate_system_prompt
        invalidate_system_prompt(self)

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

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    # 20 MB base64 ≈ 15 MB decoded — prevents OOM from an oversized data: URL in a shared gateway process.
    _MAX_DATA_URL_BASE64_BYTES = 20 * 1024 * 1024

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        if len(data) > AIAgent._MAX_DATA_URL_BASE64_BYTES:
            logger.warning(
                "data-URL payload too large (%d bytes), skipping", len(data)
            )
            return "", None
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        try:
            with tmp:
                tmp.write(base64.b64decode(data))
        except Exception:
            # delete=False means a corrupt/unsupported data URL would otherwise
            # leak a zero-byte temp file on every failed materialization.
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _model_supports_vision(self) -> bool:
        """Return True if the active provider+model reports native vision.

        Resolution: ``model.supports_vision`` > ``providers.<p>.models.<m>.supports_vision`` > models.dev
        lookup
        (see ``image_routing._supports_vision_override``). Custom/local models absent from models.dev would
        otherwise be misclassified and have their images stripped.
        """
        try:
            from hermes_cli.config import load_config
            from agent.image_routing import _lookup_supports_vision
            cfg = load_config()
            provider = (getattr(self, "provider", "") or "").strip()
            model = (getattr(self, "model", "") or "").strip()
            return _lookup_supports_vision(provider, model, cfg) is True
        except Exception:
            return False

    def _provider_supports_vision_tool_messages(self) -> bool:
        """Return True if the active provider accepts list-type tool content.

        Some providers (Xiaomi MiMo) accept multimodal user messages but 400 on list-type tool content;
        reads the provider profile's ``supports_vision_tool_messages``.
        """
        try:
            from providers import get_provider_profile
            provider = (getattr(self, "provider", "") or "").strip()
            profile = get_provider_profile(provider)
            if profile is not None:
                return getattr(profile, "supports_vision_tool_messages", True)
        except Exception:
            pass
        return True  # default: assume compatible

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _get_transport(self, api_mode: str = None):
        """Return the cached transport for the given (or current) api_mode (lazy; None if unregistered)."""
        mode = api_mode or self.api_mode
        cache = getattr(self, "_transport_cache", None)
        if cache is None:
            cache = {}
            self._transport_cache = cache
        t = cache.get(mode)
        if t is None:
            from agent.transports import get_transport
            t = get_transport(mode)
            cache[mode] = t
        return t

    def _prepare_messages_for_non_vision_model(self, api_messages: list) -> list:
        """Replace native image parts with cached vision_analyze text when the active model lacks vision.

        Vision-capable models pass through unchanged (the provider adapter — including the Anthropic one —
        handles image parts natively). The text fallback is the historically Anthropic-named preprocessor.
        """
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        if self._model_supports_vision():
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    # Same transform for the Anthropic route (callers/tests patch this name independently).
    _prepare_anthropic_messages_for_api = _prepare_messages_for_non_vision_model

    def _tool_result_content_for_active_model(self, tool_name: str, result: Any) -> Any:
        """Return the tool message content that is safe for the active model.

        Text-only providers must not receive image parts: a rejected tool result becomes canonical history
        and can make the next user turn fail before the agent can recover.
        """
        if not _is_multimodal_tool_result(result):
            return result

        content = result.get("content") or []
        if not self._content_has_image_parts(content):
            return content

        if self._model_supports_vision():
            # Vision on paper, but the provider rejects list-type tool content (or we already learned that
            # in-session): short-circuit to a text summary.
            if not self._provider_supports_vision_tool_messages():
                logger.debug(
                    "Tool %s: provider %s does not accept list-type tool "
                    "content — sending text summary",
                    tool_name, getattr(self, "provider", ""),
                )
                return _multimodal_text_summary(result)
            key = (
                (getattr(self, "provider", "") or "").strip().lower(),
                (getattr(self, "model", "") or "").strip(),
            )
            no_list = getattr(self, "_no_list_tool_content_models", None)
            if no_list and key in no_list:
                logger.debug(
                    "Tool %s: model %s/%s known to reject list-type tool "
                    "content this session — sending text summary",
                    tool_name, key[0], key[1],
                )
                return _multimodal_text_summary(result)
            return content

        summary = _multimodal_text_summary(result)
        if tool_name == "computer_use":
            return json.dumps({
                "error": (
                    "computer_use returned screenshot/image content, but the active "
                    "model/provider does not support image input. Switch to a "
                    "vision-capable model for desktop computer use, or use browser "
                    "tools for browser tasks."
                ),
                "text_summary": summary,
            })

        logger.warning(
            "Tool %s returned image content for non-vision model %s/%s; "
            "falling back to text summary",
            tool_name,
            self.provider,
            self.model,
        )
        return summary

    def _try_shrink_image_parts_in_messages(
        self,
        api_messages: list,
        *,
        max_dimension: int = 8000,
    ) -> bool:
        """Forwarder — see ``agent.conversation_compression.try_shrink_image_parts_in_messages``."""
        from agent.conversation_compression import try_shrink_image_parts_in_messages
        return try_shrink_image_parts_in_messages(
            api_messages,
            max_dimension=max_dimension,
        )

    def _try_strip_image_parts_from_tool_messages(
        self,
        api_messages: list,
        *,
        remember_model: bool = True,
    ) -> bool:
        """Downgrade list-type tool messages to text summaries in place; returns True if any were downgraded.

        Recovery for providers that 400 on list-type tool content (e.g. MiMo "text is not set"). By default
        records the (provider, model) in ``_no_list_tool_content_models`` so later results downgrade without a
        round-trip; 413 recovery passes ``remember_model=False`` (body too large ≠ provider rejects lists).
        """
        if not isinstance(api_messages, list):
            return False

        if remember_model:
            # Record (provider, model) so we don't relearn this lesson.
            key = (
                (getattr(self, "provider", "") or "").strip().lower(),
                (getattr(self, "model", "") or "").strip(),
            )
            if not hasattr(self, "_no_list_tool_content_models"):
                self._no_list_tool_content_models = set()
            if key[1]:  # only record when we actually have a model id
                self._no_list_tool_content_models.add(key)

        changed = False
        for msg in api_messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            # Salvage any text parts so the model still sees some signal.
            text_parts: List[str] = []
            had_image = False
            for part in content:
                if not isinstance(part, dict):
                    if isinstance(part, str) and part.strip():
                        text_parts.append(part.strip())
                    continue
                ptype = part.get("type")
                if ptype == "image_url" or ptype == "input_image":
                    had_image = True
                    continue
                if ptype in {"text", "input_text"}:
                    text = str(part.get("text") or "").strip()
                    if text:
                        text_parts.append(text)

            if not had_image:
                # List content without image parts — leave alone; stripping wouldn't reduce ambiguity.
                continue

            if text_parts:
                msg["content"] = "\n\n".join(text_parts)
            else:
                msg["content"] = (
                    "[image content removed — provider does not accept "
                    "list-type tool message content]"
                )
            changed = True

        return changed

    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.

        DashScope, MiniMax, Xiaomi MiMo, OpenCode Go/Zen (non-Claude), ZAI/Zhipu keep dots; AWS Bedrock uses
        dotted inference-profile IDs and rejects the hyphenated form with HTTP 400.
        """
        if (getattr(self, "provider", "") or "").lower() in {
            "alibaba", "minimax", "minimax-cn",
            "opencode-go", "opencode-zen",
            "zai", "bedrock",
            "xiaomi", "vertex",
        }:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        host = base_url_hostname(base)
        return (
            "dashscope" in host
            or base_url_host_matches(base, "aliyuncs.com")
            or "minimax" in host
            or (base_url_host_matches(base, "opencode.ai") and "/zen/" in base)
            or base_url_host_matches(base, "bigmodel.cn")
            or base_url_host_matches(base, "xiaomimimo.com")
            # Vertex AI OpenAI-compat endpoint — Gemini model ids keep dots
            # (e.g. google/gemini-3.5-flash); the hyphenated form is wrong.
            or base_url_host_matches(base, "aiplatform.googleapis.com")
            # AWS Bedrock runtime endpoints — defense-in-depth when
            # ``provider`` is unset but ``base_url`` still names Bedrock.
            or host.startswith("bedrock-runtime.")
        )

    def _is_qwen_portal(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return base_url_host_matches(self._base_url_lower, "portal.qwen.ai")

    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        prepared = copy.deepcopy(api_messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Normalize: convert bare strings to text dicts, keep dicts as-is.
                # deepcopy already created independent copies, no need for dict().
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # Inject cache_control on the last part of the system message.
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared

    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """In-place variant — mutates an already-copied message list."""
        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    def _build_api_kwargs(self, api_messages: list, tools_for_api: Optional[list] = None) -> dict:
        """Forwarder — see ``agent.chat_completion_helpers.build_api_kwargs``."""
        from agent.chat_completion_helpers import build_api_kwargs
        return build_api_kwargs(self, api_messages, tools_for_api=tools_for_api)

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body upstream and some routes 400 on ``reasoning``; gate to known
        reasoning-capable families and direct Nous Portal.
        """
        if base_url_host_matches(self._base_url_lower, "nousresearch.com"):
            return True
        if base_url_host_matches(self._base_url_lower, "ai-gateway.vercel.sh"):
            return True
        if (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "githubcopilot.com")
        ):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if (self.provider or "").strip().lower() == "lmstudio":
            opts = self._lmstudio_reasoning_options_cached()
            # "off-only" (or absent) means no real reasoning capability.
            return any(opt and opt != "off" for opt in opts)
        # Ollama Cloud: /api/show capabilities are authoritative — emit reasoning_effort only for models
        # declaring "thinking". Cached per (model, base_url).
        if base_url_host_matches(self._base_url_lower, "ollama.com"):
            return self._ollama_supports_thinking_cached()
        if not self._is_openrouter_url():
            return False
        if base_url_host_matches(self._base_url_lower, "api.mistral.ai"):
            return False

        model = (self.model or "").lower()
        # Live-catalog metadata first (OpenRouter /v1/models supported_parameters) — the static prefix
        # allowlist repeatedly went stale one vendor at a time (#75386). Unknown falls back to the static
        # list.
        try:
            from hermes_cli.models import (
                openrouter_model_reasoning_capabilities,
                warm_openrouter_reasoning_caps_async,
            )
            caps = openrouter_model_reasoning_capabilities(self.model)
            if caps is None:
                # Cache cold — warm in the background; never block this turn on HTTP.
                warm_openrouter_reasoning_caps_async()
        except Exception:
            caps = None
        if caps is not None:
            return bool(caps.get("supports_reasoning"))
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "google/gemma-4",
            "qwen/qwen3",
            "tencent/hy",
            "xiaomi/",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _lmstudio_reasoning_options_cached(self) -> list[str]:
        """Probe LM Studio's published reasoning ``allowed_options`` once per (model, base_url).

        Needed for the supports-reasoning gate and to clamp ``reasoning_effort`` so toggle-style models don't
        400
        on ``high``. Non-empty results cache permanently; empty ones (transient failure OR non-reasoning
        model)
        cache with a 60s TTL to avoid a round-trip per turn while retrying soon.
        """
        import time as _time

        cache = getattr(self, "_lm_reasoning_opts_cache", None)
        if cache is None:
            cache = self._lm_reasoning_opts_cache = {}
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            opts, ts = cached
            # Non-empty → permanent. Empty → 60s TTL.
            if opts or (_time.monotonic() - ts) < 60:
                return opts
        try:
            from hermes_cli.models import lmstudio_model_reasoning_options
            opts = lmstudio_model_reasoning_options(
                self.model, self.base_url, getattr(self, "api_key", ""),
            )
        except Exception:
            opts = []
        cache[key] = (opts, _time.monotonic())
        return opts

    def _ollama_supports_thinking_cached(self) -> bool:
        """Probe Ollama's ``/api/show`` capabilities once per (model, base_url); True only if ``thinking`` is
        declared.

        True/False cache permanently; a probe failure (None) caches 60s so an outage neither suppresses
        reasoning
        for the session nor round-trips every turn.
        """
        import time as _time

        cache = getattr(self, "_ollama_thinking_cache", None)
        if cache is None:
            cache = self._ollama_thinking_cache = {}
        key = (self.model, self.base_url)
        cached = cache.get(key)
        if cached is not None:
            supported, ts = cached
            # Definitive True/False → permanent. Unknown (None) → 60s TTL.
            if supported is not None or (_time.monotonic() - ts) < 60:
                return bool(supported)
        try:
            from hermes_cli.models import ollama_model_supports_thinking
            supported = ollama_model_supports_thinking(
                self.model, self.base_url, getattr(self, "api_key", "")
            )
        except Exception:
            supported = None
        cache[key] = (supported, _time.monotonic())
        return bool(supported)

    def _resolve_lmstudio_summary_reasoning_effort(self) -> Optional[str]:
        """Resolve a safe top-level ``reasoning_effort`` for LM Studio.

        The iteration-limit summary calls ``chat.completions.create()`` directly, bypassing the transport;
        share the helper so effort resolution and clamping cannot drift.
        """
        from agent.lmstudio_reasoning import resolve_lmstudio_effort
        return resolve_lmstudio_effort(
            self.reasoning_config,
            self._lmstudio_reasoning_options_cached(),
        )

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "xhigh" not in supported_efforts and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    _build_assistant_message = _forward("agent.chat_completion_helpers", "build_assistant_message")

    def _needs_thinking_reasoning_pad(self) -> bool:
        """Return True when the active provider enforces ``reasoning_content`` echo-back on tool-call replays.

        DeepSeek thinking, Kimi/Moonshot thinking and Xiaomi MiMo thinking all 400 without it. Cached per
        (provider, model, base_url) and invalidated by ``switch_model()`` / ``_try_activate_fallback()`` —
        the loop calls this ~16× per turn and each miss re-runs several ``urlparse`` host matches.
        """
        key = (self.provider, self.model, getattr(self, "_base_url_lower", self.base_url))
        cached = getattr(self, "_thinking_pad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = (
            self._needs_deepseek_tool_reasoning()
            or self._needs_kimi_tool_reasoning()
            or self._needs_mimo_tool_reasoning()
            or self._reasoning_echo_opt_in()
        )
        self._thinking_pad_cache = (key, result)
        return result

    def _reasoning_echo_opt_in(self) -> bool:
        """True when the user opted in to ``reasoning_content`` echo-back for the *current* provider via
        config.

        Covers custom providers / gateways proxying thinking models that the host-based
        ``_REASONING_ECHO_RULES``
        miss. Per-active-provider: primary from ``model.reasoning_echo``, fallback from the fallback entry's
        field,
        restored by ``restore_primary_runtime()`` — so falling back to a strict provider still strips it.
        """
        return bool(getattr(self, "_reasoning_echo_flag", False))

    @staticmethod
    def _read_reasoning_echo_from_config() -> bool:
        """Read ``model.reasoning_echo`` from config; False on any error."""
        try:
            from hermes_cli.config import load_config_readonly
            return bool(
                (load_config_readonly().get("model") or {}).get("reasoning_echo")
            )
        except Exception:
            return False

    def _needs_kimi_tool_reasoning(self) -> bool:
        """Return True when the current provider is Kimi / Moonshot thinking mode (requires
        ``reasoning_content`` echo).

        Host-driven, not model-name-driven: aggregators re-exporting Kimi reject the echo. Rule table:
        ``message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "kimi", self.provider, None, self.base_url
        )

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """Return True when the current provider is DeepSeek thinking mode (requires ``reasoning_content``
        echo).

        Rule table: ``message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "deepseek", (self.provider or "").lower(), self.model, self.base_url
        )

    def _needs_mimo_tool_reasoning(self) -> bool:
        """Return True when the current provider is Xiaomi MiMo thinking mode (requires ``reasoning_content``
        echo).

        Rule table: ``message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "mimo", (self.provider or "").lower(), self.model, self.base_url
        )

    _copy_reasoning_content_for_api = _forward("agent.agent_runtime_helpers", "copy_reasoning_content_for_api")

    _reapply_reasoning_echo_for_provider = _forward("agent.agent_runtime_helpers", "reapply_reasoning_echo_for_provider")

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict, model: "str | None" = None) -> dict:
        """Strip Codex Responses fields (call_id, response_item_id, extra_content) from tool_calls for strict
        providers.

        Strict Chat Completions APIs (Mistral, Fireworks) 400/422 on unknown fields. ``extra_content`` (Gemini
        thought_signature) is kept only when the outgoing model is Gemini-family (it 400s without it). Builds
        new dicts so the internal history retains the Codex fields for a later fallback.
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        from agent.transports.chat_completions import _model_consumes_thought_signature
        _STRIP_KEYS = {"call_id", "response_item_id"}
        if not _model_consumes_thought_signature(model):
            _STRIP_KEYS = _STRIP_KEYS | {"extra_content"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    _sanitize_tool_call_arguments = _forward_static("agent.agent_runtime_helpers", "sanitize_tool_call_arguments")

    def _should_sanitize_tool_calls(self) -> bool:
        """Determine if tool_calls need sanitization (True for every non-Codex API).

        Codex Responses fields (call_id, response_item_id) are not Chat Completions schema and 400 elsewhere.
        """
        return self.api_mode != "codex_responses"

    def _compress_context(
        self,
        messages: list,
        system_message: str,
        *,
        approx_tokens: int = None,
        task_id: str = "default",
        focus_topic: str = None,
        force: bool = False,
        bypass_cooldown: bool = False,
        defer_context_engine_notification: bool = False,
        commit_fence=None,
    ) -> tuple:
        """Forwarder — see ``agent.conversation_compression.compress_context``.

        ``force=True`` (manual /compress) bypasses the summary-failure cooldown; ``bypass_cooldown=True``
        (provider-proven overflow recovery) runs one real attempt while the cooldown stays armed.
        """
        # Per-attempt timeout signal for turn-start preflight and in-loop consumers (#98424): a stalled
        # compression must not be mistaken for a structural no-op. Thread-local + per-agent lock (#98741).
        from agent.conversation_compression import (
            CompressionCommitFence,
            compress_context,
            mark_context_compression_timed_out,
            reset_context_compression_timeout_outcome,
            resolve_context_compression_timeouts,
            run_compress_context_with_progress_timeout,
        )
        reset_context_compression_timeout_outcome(self)
        from agent.portal_tags import (
            get_affinity_scope,
            get_conversation_context,
            reset_affinity_scope,
            reset_conversation_context,
            set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        # Out-of-turn compaction (/compact, gateway /compress, partial head compression) runs outside
        # run_conversation's ambient scope; publish the root as a fallback so the summarizer's call carries
        # the conversation tag. No-op for in-turn callers.
        token = None
        if get_conversation_context() is None:
            root = self._conversation_root_id()
            if root:
                token = set_conversation_context(root)
        # Same fallback for the ROUTING scope, only when the host declared one (pre-#96811 fallback
        # otherwise).
        affinity_token = None
        if get_affinity_scope() is None:
            declared = declared_conversation_scope_safe(self)
            if declared:
                affinity_token = set_affinity_scope(declared)
        # Every compression has a fence; hard_interrupt() uses this exact instance to serialize cancel
        # admission against begin_commit().
        active_fence = commit_fence or CompressionCommitFence()
        # Serialize fence publication so overlapping automatic/manual entrypoints cannot replace the
        # fence of the attempt currently committing.
        fence_registration_lock = vars(self).setdefault(
            "_compression_commit_fence_lock", threading.RLock()
        )
        with fence_registration_lock:
            missing_fence = object()
            previous_fence = vars(self).get(
                "_active_compression_commit_fence", missing_fence
            )
            self._active_compression_commit_fence = active_fence
        try:
            def _run(fence=None, target_messages=None):
                return compress_context(
                    self,
                    target_messages if target_messages is not None else messages,
                    system_message,
                    approx_tokens=approx_tokens, task_id=task_id,
                    focus_topic=focus_topic,
                    force=force,
                    bypass_cooldown=bypass_cooldown,
                    defer_context_engine_notification=(
                        defer_context_engine_notification
                    ),
                    commit_fence=fence,
                )

            # Callers that already own a progress-aware wait (gateway session
            # hygiene) pass commit_fence and must not be double-wrapped.
            direct_path = commit_fence is not None
            idle_timeout = total_ceiling = None
            if not direct_path:
                idle_timeout, total_ceiling = resolve_context_compression_timeouts()
                if idle_timeout <= 0:
                    direct_path = True

            if direct_path:
                result = _run(active_fence)
            else:
                def _snapshot_worker(fence=None):
                    # #76354 F3: the pooled worker must NEVER share the caller's live transcript — a late
                    # engine after a
                    # host timeout could rewrite it. Deep-snapshot on the worker; results publish only via an
                    # ADMITTED commit.
                    snapshot = copy.deepcopy(messages)
                    result_msgs, result_prompt = _run(
                        fence, target_messages=snapshot
                    )
                    if result_msgs is snapshot:
                        # No-op/abort returned the snapshot unchanged: hand back the ORIGINAL list so
                        # identity-based
                        # semantics keep working.
                        return messages, result_prompt
                    return result_msgs, result_prompt

                # Resolve the fallback prompt lazily: an eager rebuild would raise before compress_context
                # runs
                # when _cached_system_prompt is unset and _build_system_prompt fails.
                def _fallback_prompt():
                    cached = getattr(self, "_cached_system_prompt", None)
                    if cached:
                        return cached
                    try:
                        return self._build_system_prompt(system_message)
                    except Exception:
                        logger.debug(
                            "compress_context timeout fallback prompt rebuild "
                            "failed; using raw system_message",
                            exc_info=True,
                        )
                        return system_message or ""

                timeout_cause = {
                    "total_exhausted": False,
                    "progress_observed": False,
                }

                def _on_timeout_cause(total_exhausted, progress_observed):
                    timeout_cause["total_exhausted"] = total_exhausted
                    timeout_cause["progress_observed"] = progress_observed

                def _on_timeout(idle, waited, since_progress):
                    mark_context_compression_timed_out(self)
                    total_exhausted = timeout_cause["total_exhausted"]
                    progress_observed = timeout_cause["progress_observed"]
                    if total_exhausted:
                        logger.warning(
                            "Context compression reached its total ceiling "
                            "after %.1fs (progress observed=%s); continuing "
                            "without compression",
                            waited,
                            progress_observed,
                        )
                    else:
                        logger.warning(
                            "Context compression made no progress for %.1fs "
                            "(total wait %.1fs, ceiling %.1fs); continuing "
                            "without compression",
                            since_progress,
                            waited,
                            total_ceiling,
                        )
                    touch = getattr(self, "_touch_activity", None)
                    if callable(touch):
                        try:
                            touch(
                                "context compression timed out",
                                provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
                            )
                        except Exception:
                            logger.debug(
                                "compress_context timeout activity touch failed",
                                exc_info=True,
                            )
                    # Same timeout cooldown ladder as summary-LLM timeouts
                    # (#62452): avoid re-burning the full idle budget every turn.
                    compressor = getattr(self, "context_compressor", None)
                    if compressor is not None:
                        record = getattr(compressor, "record_timeout_failure", None)
                        if callable(record):
                            try:
                                reason = (
                                    "host compress_context total ceiling "
                                    "exhausted"
                                    if total_exhausted
                                    else "host compress_context timeout "
                                    "(no summary progress)"
                                )
                                record(
                                    reason,
                                    failure_kind=(
                                        "ceiling_exhausted"
                                        if total_exhausted
                                        else "stalled"
                                    ),
                                )
                            except Exception:
                                logger.debug(
                                    "failed to record compress_context timeout "
                                    "cooldown",
                                    exc_info=True,
                                )
                    emit = getattr(self, "_emit_warning", None)
                    if callable(emit):
                        if total_exhausted:
                            progress = (
                                " after summary output was observed"
                                if progress_observed
                                else ""
                            )
                            emit(
                                "⚠ Context compression reached its total ceiling "
                                f"after {waited:.1f}s{progress}. No messages were "
                                "dropped — continuing without compression. Run "
                                "/compress to retry or /new for a clean session."
                            )
                        else:
                            emit(
                                "⚠ Context compression timed out "
                                f"after {idle:.1f}s with no output from the summary "
                                "model. No messages were dropped — continuing "
                                "without compression. Run /compress to retry, /new "
                                "for a clean session, or check "
                                "auxiliary.compression."
                            )

                def _on_commit_overrun(waited, ceiling):
                    # Commit-phase ceiling breach: the SessionDB mutation must complete, so this only surfaces
                    # the overrun.
                    emit = getattr(self, "_emit_warning", None)
                    if callable(emit):
                        emit(
                            "⚠ Context compression commit is taking unusually "
                            f"long ({waited:.0f}s, ceiling {ceiling:.0f}s). "
                            "Waiting for it to finish safely — if this persists, "
                            "check SessionDB health (disk / lock contention)."
                        )

                def _publish_new_fence():
                    # The stall-fallback retry (#78981) needs a fence the aborted attempt cannot veto; publish
                    # it on
                    # the slot hard_interrupt() reads. The finally restores the caller's fence.
                    retry_fence = CompressionCommitFence()
                    with fence_registration_lock:
                        self._active_compression_commit_fence = retry_fence
                    return retry_fence

                result = run_compress_context_with_progress_timeout(
                    worker=_snapshot_worker,
                    messages=messages,
                    system_prompt_fallback=_fallback_prompt,
                    idle_timeout_seconds=idle_timeout,
                    total_ceiling_seconds=total_ceiling,
                    on_timeout=_on_timeout,
                    on_timeout_cause=_on_timeout_cause,
                    on_commit_overrun=_on_commit_overrun,
                    fence=active_fence,
                    telemetry_agent=self,
                    new_fence=_publish_new_fence,
                )
            # Imported UNCONDITIONALLY: a silent fallback literal would split the stamping key from the
            # flush's and resurrect the duplicate-row bug.
            from agent.context_compressor import _DB_PERSISTED_MARKER
            from agent.conversation_compression import (
                _messages_match_scoped_identity,

            )

            def _sync_persisted_markers(target_messages, source_messages):
                if not isinstance(target_messages, list) or not isinstance(
                    source_messages, list
                ):
                    return
                # Stamps land on the worker's snapshot first; mirror them onto the live lists by scoped
                # identity.
                # Timestamp-less repeated content is ambiguous, so every scoped match is stamped.
                for source_message in source_messages:
                    if not (
                        isinstance(source_message, dict)
                        and source_message.get(_DB_PERSISTED_MARKER)
                    ):
                        continue
                    source_timestamp = source_message.get("timestamp")
                    matched_exact_timestamp = False
                    if source_timestamp is not None:
                        for target_message in target_messages:
                            if not isinstance(target_message, dict):
                                continue
                            if target_message.get(_DB_PERSISTED_MARKER):
                                continue
                            if not _messages_match_scoped_identity(
                                target_message, source_message
                            ):
                                continue
                            if target_message.get("timestamp") != source_timestamp:
                                continue
                            target_message[_DB_PERSISTED_MARKER] = True
                            matched_exact_timestamp = True
                        if matched_exact_timestamp:
                            continue
                    for target_message in target_messages:
                        if not isinstance(target_message, dict):
                            continue
                        if target_message.get(_DB_PERSISTED_MARKER):
                            continue
                        if not _messages_match_scoped_identity(
                            target_message, source_message
                        ):
                            continue
                        target_message[_DB_PERSISTED_MARKER] = True

            if isinstance(result, tuple) and result:
                result_messages = result[0]
                if isinstance(result_messages, list):
                    # Direct-path callers bypass the snapshot worker but still need the post-publish mirror.
                    if direct_path or result_messages is not messages:
                        _sync_persisted_markers(messages, result_messages)
                    session_messages = getattr(self, "_session_messages", None)
                    if (
                        isinstance(session_messages, list)
                        and session_messages is not messages
                    ):
                        # Durable-parent adoption can leave `_session_messages` on the pre-adoption list; sync
                        # both.
                        _sync_persisted_markers(session_messages, result_messages)
            # The worker thread rotated hermes_logging's thread-local session id; propagate to this thread
            # (#34089).
            try:
                from hermes_logging import set_session_context
                set_session_context(self.session_id)
            except Exception:
                pass
            # #76354 F5: rebind the session ContextVar in the CALLER's context so post-compression tools
            # resolve HERMES_SESSION_ID to the child id (idempotent when no rotation happened).
            try:
                from gateway.session_context import set_current_session_id
                if self.session_id:
                    set_current_session_id(self.session_id)
            except Exception:
                logger.debug(
                    "post-compression session ContextVar rebind failed",
                    exc_info=True,
                )
            return result
        finally:
            with fence_registration_lock:
                if previous_fence is missing_fence:
                    vars(self).pop("_active_compression_commit_fence", None)
                else:
                    self._active_compression_commit_fence = previous_fence
            # Restore whatever the caller had, so a compaction never leaks its
            # tag into the surrounding scope.
            if token is not None:
                reset_conversation_context(token)
            if affinity_token is not None:
                reset_affinity_scope(affinity_token)

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
        overlapping
        file targets, opted-in MCP) separated by sequential barriers; mixed batches run segment by segment in
        emission order so safe subsets stay concurrent while side-effect ordering is preserved.
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

    def run_conversation(
        self,
        user_message: Any,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        persist_user_display_metadata: Optional[Dict[str, Any]] = None,
        persist_user_platform_id: Optional[str] = None,
        moa_config: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forwarder — see ``agent.conversation_loop.run_conversation``."""
        # A review shares this session_id for cache parity: fence review startup or interrupt an admitted
        # request and await its exit before opening live-turn instrumentation (#84423).
        from agent.background_review import cancel_background_review_for_live_turn

        cancel_background_review_for_live_turn(self)

        # Turn liveness for the deferred-review idle queue; start-mark is the first statement of the try
        # so the finally's note_turn_finished balances every exit.
        from agent.review_idle_queue import QUEUE as _review_queue

        from agent.aux_accounting import (
            reset_accounting_context,
            set_accounting_context,
        )
        from agent import relay_runtime
        from agent.conversation_loop import run_conversation
        from agent.portal_tags import (
            reset_affinity_scope,
            reset_conversation_context,
            set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        from hermes_cli.observability.relay_shared_metrics import (
            finish_task_run,
            start_task_run,
        )
        from agent.subagent_lifecycle import bind_subagent_parent
        effective_task_id = task_id or str(uuid.uuid4())
        session_id = str(getattr(self, "session_id", None) or "")
        task_context = {
            "session_id": session_id,
            "task_id": effective_task_id,
            "platform": getattr(self, "platform", None) or "",
        }
        relay_turn_id = (
            f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
        )
        self._relay_pending_turn_id = relay_turn_id
        relay_parent_session_id = (
            str(getattr(self, "_parent_session_id", None) or "")
            if task_context["platform"] == "subagent"
            else ""
        )
        relay_lease = None
        relay_turn = None
        durable_turn_lease = None
        durable_turn_lease_stop = None
        durable_turn_lease_thread = None
        durable_turn_liveness_thread = None
        durable_turn_lease_activity_lock = threading.Lock()
        durable_turn_lease_turn_active = False
        durable_turn_lease_interrupt_message = None
        token = None
        # Initialized alongside `token`: early returns leave the try before set_affinity_scope() and the
        # finally reads this unconditionally (PR #97158).
        affinity_token = None
        acct_token = None
        task_started = False
        task_finished = False
        relay_outcome = "failed"

        def _stop_durable_turn_lease_refresher() -> None:
            nonlocal durable_turn_lease_turn_active
            with durable_turn_lease_activity_lock:
                durable_turn_lease_turn_active = False
                if durable_turn_lease_stop is not None:
                    durable_turn_lease_stop.set()

        def _clear_durable_turn_lease_interrupt() -> None:
            """Clear only the interrupt admitted by this turn's refresher."""
            message = durable_turn_lease_interrupt_message
            if not message:
                return

            def _clear_if_owned() -> None:
                if getattr(self, "_interrupt_message", None) != message:
                    return
                self._interrupt_requested = False
                self._interrupt_message = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                self._interrupt_thread_signal_pending = False
                if self._execution_thread_id is not None:
                    _set_interrupt(False, self._execution_thread_id)

            redirect_lock = getattr(self, "_pending_redirect_lock", None)
            if redirect_lock is None:
                _clear_if_owned()
            else:
                with redirect_lock:
                    _clear_if_owned()

        try:
            _review_queue.note_turn_started()
            # Durable cross-process lease over load -> run -> flush (Desktop, CLI resume, gateway, background
            # delivery sharing state.db, #84234).
            _turn_db = getattr(self, "_session_db", None)
            _durable_session_exists = False
            if _turn_db is not None and session_id:
                try:
                    _durable_session_exists = _turn_db.get_session(session_id) is not None
                except Exception:
                    # A locked / non-WAL read is not proof the row is absent; treating probe failure as
                    # "fresh" ran
                    # fail-open at the exact contention point (#84234). Acquire, or fail closed.
                    logger.warning(
                        "Could not check durable session before turn lease; "
                        "will acquire rather than run without serialization",
                        exc_info=True,
                    )
                    _durable_session_exists = True
            if (
                _turn_db is not None
                and session_id
                and not getattr(self, "_persist_disabled", False)
                # A fresh session id has no durable transcript to race over, and callers may supply an in-
                # memory
                # seed before the row exists — reloading would erase it.
                and _durable_session_exists
                # Check the concrete type: MagicMock-style shims accept any attribute without the protocol.
                and callable(
                    getattr(type(_turn_db), "acquire_session_turn_lease", None)
                )
            ):
                # Row proven to exist — suppress the redundant create attempt.
                self._session_db_created = True
                _durable_holder = (
                    f"pid={os.getpid()}:turn={relay_turn_id}:platform="
                    f"{task_context['platform'] or 'unknown'}"
                )
                _lease_ttl = 300.0
                _lease_waited = False

                def _on_session_turn_lease_wait(elapsed: float) -> None:
                    nonlocal _lease_waited
                    _lease_waited = True
                    if elapsed < 1.0:
                        self._emit_status(
                            "⏳ Another Hermes process is using this session; "
                            "waiting for it to finish before starting your turn..."
                        )
                    else:
                        self._emit_status(
                            "⏳ Still waiting for the other Hermes process on "
                            f"this session ({int(elapsed)}s)..."
                        )

                if not _turn_db.acquire_session_turn_lease(
                    session_id,
                    _durable_holder,
                    ttl_seconds=_lease_ttl,
                    wait_seconds=1800.0,
                    on_wait=_on_session_turn_lease_wait,
                    should_abort=lambda: getattr(self, "_interrupt_requested", False),
                ):
                    if getattr(self, "_interrupt_requested", False):
                        logger.info(
                            "session turn lease wait aborted by interrupt: %s",
                            session_id,
                        )
                        relay_outcome = "cancelled"
                        interrupt_msg = (
                            "Stopped waiting for another Hermes process on "
                            "this session. Your message was not processed."
                        )
                        interrupt_result = {
                            "final_response": interrupt_msg,
                            "messages": list(conversation_history or []),
                            "api_calls": 0,
                            "completed": False,
                            "interrupted": True,
                        }
                        interrupt_message = getattr(
                            self, "_interrupt_message", None
                        )
                        if interrupt_message:
                            interrupt_result["interrupt_message"] = (
                                interrupt_message
                            )
                        # The finalizer never runs on this early return; clear so a cached agent doesn't fail-
                        # close the next turn.
                        try:
                            self.clear_interrupt()
                        except Exception:
                            self._interrupt_requested = False
                            self._interrupt_message = None
                        return interrupt_result
                    # Fail closed like gateway TurnLeaseTimeoutError: surface a resend notice, not a bare
                    # TimeoutError.
                    timeout_msg = (
                        "⏳ Another Hermes process kept this session busy too "
                        "long. Your message was not processed - wait for the "
                        "other process to finish, then send it again."
                    )
                    logger.error(
                        "session turn lease wait timed out for %s",
                        session_id,
                    )
                    try:
                        self._emit_warning(timeout_msg)
                    except Exception:
                        logger.debug(
                            "Failed to emit session turn lease timeout warning",
                            exc_info=True,
                        )
                    relay_outcome = "timed_out"
                    return {
                        "final_response": timeout_msg,
                        "messages": list(conversation_history or []),
                        "api_calls": 0,
                        "completed": False,
                        "failed": True,
                        "error": f"session_turn_lease_timeout:{session_id}",
                    }

                # Assign only after admission so the finally cannot release a holder that never owned the row;
                # persist paths read the agent attr so a late flush is fenced in the same SQLite transaction.
                durable_turn_lease = _durable_holder
                self._active_session_turn_lease_holder = _durable_holder
                self._active_session_turn_lease_ttl_seconds = _lease_ttl
                if _lease_waited:
                    self._emit_status(
                        "Session is free; loading the latest transcript..."
                    )

                # The holder may have compressed/rotated the session while we waited: reload only AFTER
                # admission,
                # and skip when acquisition was immediate (avoids a needless prompt-cache miss).
                if _lease_waited:
                    latest_session_id = _turn_db.resolve_resume_session_id(session_id)
                    if latest_session_id:
                        self.session_id = latest_session_id
                        task_context["session_id"] = latest_session_id
                    conversation_history = _turn_db.get_messages_as_conversation(
                        self.session_id,
                        repair_alternation=True,
                        include_row_ids=True,
                    )

                # Long turns outlive a fixed TTL: refresh in a daemon thread; holder-qualified UPDATE/DELETE
                # fence
                # a late refresher from a successor lease.
                durable_turn_lease_stop = threading.Event()
                _lease_refresh_interval = float(
                    getattr(self, "_session_turn_lease_refresh_interval", 60.0)
                )

                # ── Turn liveness watchdog (#95548) ──
                # Lease renewal is NOT evidence of progress; a silently stalled turn would renew forever.
                # Policy
                # lives in agent/turn_liveness.py; this block only wires config + commit/deactivate callbacks.
                try:
                    from hermes_cli.config import (
                        load_config_readonly as _liveness_load_config,
                    )
                    _liveness_config = _liveness_load_config() or {}
                except Exception:
                    _liveness_config = {}
                from agent import turn_liveness

                _liveness_timeout, _liveness_poll = (
                    turn_liveness.resolve_turn_liveness_settings(_liveness_config)
                )

                def _interrupt_turn(message: str) -> None:
                    # Lease-loss interrupts fire UNCONDITIONALLY (no generation claim): a lost lease means
                    # this
                    # process no longer owns the session. Only the watchdog's stalls can be spuriously stale.
                    nonlocal durable_turn_lease_interrupt_message
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return
                        durable_turn_lease_interrupt_message = message
                        try:
                            self.interrupt(message, hard_cancel=True)
                        except Exception:
                            self._interrupt_requested = True
                            self._interrupt_message = message

                def _commit_turn_liveness_abort(
                    snapshot: "turn_liveness.ActivitySnapshot",
                    message: str,
                ) -> bool:
                    """Commit point for the watchdog's stall observation.

                    Revalidates the observed ``(generation, timestamp)`` under the SAME lock
                    ``_touch_activity`` uses, so a turn
                    that resumed while the stall was logged is never hard-cancelled. The revalidated
                    generation is carried into
                    ``interrupt`` as ``require_generation``, which consumes it with the first publication in
                    ONE critical section.
                    If ``interrupt`` raises, the abort declines FAIL-CLOSED. Returns False when stale or
                    already winding down.
                    """
                    nonlocal durable_turn_lease_interrupt_message
                    with self._liveness_activity_lock():
                        current_generation = getattr(
                            self, "_turn_liveness_activity_generation", 0
                        )
                        if (
                            current_generation,
                            getattr(self, "_last_activity_ts", None),
                        ) != (snapshot.generation, snapshot.activity_ts):
                            return False
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return False
                    try:
                        published = self.interrupt(
                            message,
                            hard_cancel=True,
                            require_generation=current_generation,
                        )
                    except Exception:
                        # Round-4 (#95663): fail closed — an exceptional path must not turn an unvalidated
                        # claim into
                        # unconditional abort authority.
                        logger.debug(
                            "Turn liveness abort interrupt raised; "
                            "declining the abort",
                            exc_info=True,
                        )
                        published = False
                    if published is False:
                        # Claim went stale between revalidation and the hammer: real progress landed, abandon
                        # the abort.
                        return False
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_interrupt_message = message
                    return True

                def _deactivate_turn_after_liveness_abort() -> None:
                    """Stop lease renewal after a committed liveness abort.

                    A wedge the hard interrupt cannot unwind must not keep the lease alive forever; TTL expiry
                    lets
                    stale-turn cleanup reclaim the row.
                    """
                    nonlocal durable_turn_lease_turn_active
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_stop.set()
                        durable_turn_lease_turn_active = False

                def _turn_is_active() -> bool:
                    with durable_turn_lease_activity_lock:
                        return durable_turn_lease_turn_active

                def _refresh_durable_turn_lease() -> None:
                    while not durable_turn_lease_stop.wait(_lease_refresh_interval):
                        try:
                            if not _turn_db.refresh_session_turn_lease(
                                getattr(self, "session_id", None) or session_id,
                                durable_turn_lease,
                                ttl_seconds=_lease_ttl,
                            ):
                                # finally sets stop then releases; a late holder-fenced miss must not hard-
                                # interrupt the next turn.
                                if durable_turn_lease_stop.is_set():
                                    return
                                logger.error(
                                    "Lost session turn lease while turn is active: %s",
                                    getattr(self, "session_id", None) or session_id,
                                )
                                _interrupt_turn(
                                    "Session turn lease lost; stopping to protect "
                                    "the transcript."
                                )
                                return
                        except Exception:
                            if durable_turn_lease_stop.is_set():
                                return
                            logger.warning(
                                "Failed to refresh session turn lease: %s",
                                getattr(self, "session_id", None) or session_id,
                                exc_info=True,
                            )
                            _interrupt_turn(
                                "Session turn lease could not be refreshed; "
                                "stopping to protect the transcript."
                            )
                            return

                durable_turn_lease_thread = threading.Thread(
                    target=_refresh_durable_turn_lease,
                    name="session-turn-lease-refresh",
                    daemon=True,
                )
                if _liveness_timeout is not None:
                    durable_turn_liveness_thread = (
                        turn_liveness.TurnLivenessWatchdog(
                            self,
                            session_id=getattr(self, "session_id", None) or session_id,
                            timeout_s=_liveness_timeout,
                            poll_s=_liveness_poll,
                            stop_event=durable_turn_lease_stop,
                            activity_lock=self._liveness_activity_lock(),
                            is_turn_active=_turn_is_active,
                            commit_abort=_commit_turn_liveness_abort,
                            deactivate_turn=_deactivate_turn_after_liveness_abort,
                        ).make_thread()
                    )

            relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=task_context["session_id"],
                platform=task_context["platform"],
                parent_session_id=relay_parent_session_id,
                model=str(getattr(self, "model", None) or ""),
            )
            relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                relay_lease,
                turn_id=relay_turn_id,
                task_id=effective_task_id,
            )
            # Keep existing tests and external relay-runtime shims that return
            # a minimal turn object compatible with the new opt-out flag.
            if getattr(relay_turn, "relay_enabled", True):
                start_task_run(
                    **task_context,
                    parent_session_id=getattr(self, "_parent_session_id", None) or "",
                )
                task_started = True
            # Publish the conversation id for ambient Nous Portal tagging: every LLM call in this turn (loop,
            # compression, vision, MoA, review forks) inherits `conversation=<root>`.
            token = set_conversation_context(self._conversation_root_id())
            # Routing/affinity scope the HOST declared; providers fall back to the id above when unset
            # (#96811).
            affinity_token = set_affinity_scope(
                declared_conversation_scope_safe(self)
            )
            # Publish session accounting handles so auxiliary calls record usage into session_model_usage
            # (#23270).
            acct_token = set_accounting_context(
                getattr(self, "_session_db", None),
                getattr(self, "session_id", None),
            )
            from agent.auxiliary_client import scoped_runtime_main

            # Keep the ContextVar scope local (tokens on the agent may be observed from another thread).
            with bind_subagent_parent(self), scoped_runtime_main({}):
                try:
                    if durable_turn_lease_thread is not None:
                        with durable_turn_lease_activity_lock:
                            durable_turn_lease_turn_active = True
                        # Stamp the activity clock at turn entry (#95663): `_last_activity_ts` persists across
                        # turns, so
                        # without this the watchdog would measure idle from the PREVIOUS turn and abort a
                        # fresh one.
                        self._touch_activity("starting new turn")
                        durable_turn_lease_thread.start()
                        if durable_turn_liveness_thread is not None:
                            durable_turn_liveness_thread.start()
                    result = run_conversation(
                        self,
                        user_message,
                        system_message,
                        conversation_history,
                        effective_task_id,
                        stream_callback,
                        persist_user_message,
                        persist_user_timestamp=persist_user_timestamp,
                        persist_user_display_kind=persist_user_display_kind,
                        persist_user_display_metadata=persist_user_display_metadata,
                        persist_user_platform_id=persist_user_platform_id,
                        moa_config=moa_config,
                    )
                finally:
                    # Post-loop relay/task finalization must not receive a late refresh interrupt.
                    _stop_durable_turn_lease_refresher()
                    # Interrupt clear is deferred until after thread join (outer finally) so a refresher
                    # firing
                    # between stop and join cannot leave an interrupt behind.
            terminal = result if isinstance(result, dict) else {}
            if terminal.get("interrupted") is True:
                relay_outcome = "cancelled"
            elif terminal.get("failed") is True:
                relay_outcome = "failed"
            else:
                relay_outcome = "success"
            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                relay_turn,
                outcome=relay_outcome,
            )
            if task_started:
                task_finished = True
                finish_task_run(**task_context, result=result)
            return result
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                type(exc).__name__ == "CancelledError"
            ):
                relay_outcome = "cancelled"
            elif isinstance(exc, TimeoutError):
                relay_outcome = "timed_out"
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                    relay_turn,
                    outcome=relay_outcome,
                )
            if task_started and not task_finished:
                task_finished = True
                finish_task_run(**task_context, error=exc)
            raise
        finally:
            try:
                if relay_turn is not None:
                    relay_runtime.SESSION_COORDINATOR.end_turn(
                        relay_turn,
                        outcome=relay_outcome,
                    )
            finally:
                try:
                    if relay_lease is not None:
                        relay_runtime.SESSION_COORDINATOR.release_conversation(
                            relay_lease
                        )
                finally:
                    _stop_durable_turn_lease_refresher()
                    for _durable_thread in (
                        durable_turn_lease_thread,
                        durable_turn_liveness_thread,
                    ):
                        if (
                            _durable_thread is not None
                            and _durable_thread.is_alive()
                        ):
                            _durable_thread.join(timeout=1.0)
                    # Clear any refresher interrupt fired between stop and join; must run AFTER join.
                    _clear_durable_turn_lease_interrupt()
                    if durable_turn_lease is not None:
                        try:
                            _turn_db.release_session_turn_lease(
                                session_id, durable_turn_lease
                            )
                        except Exception:
                            logger.error(
                                "Failed to release session turn lease: %s",
                                session_id,
                                exc_info=True,
                            )
                        if (
                            getattr(self, "_active_session_turn_lease_holder", None)
                            == durable_turn_lease
                        ):
                            self._active_session_turn_lease_holder = None
                            self._active_session_turn_lease_ttl_seconds = None
                    # Always clear mid-turn labels when the turn exits — including
                    # interrupted early returns that skip finalize_turn. Keep ts.
                    try:
                        self._reset_activity_labels_after_turn()
                    except Exception:
                        pass
                    if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                        self._relay_pending_turn_id = None
                    if acct_token is not None:
                        reset_accounting_context(acct_token)
                    if token is not None:
                        reset_conversation_context(token)
                    if affinity_token is not None:
                        reset_affinity_scope(affinity_token)
                    # Balance note_turn_started on every exit so the idle queue's live-turn count cannot leak.
                    try:
                        _review_queue.note_turn_finished()
                    except Exception:
                        pass

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """Simple chat interface that returns just the final response string.

        ``stream_callback`` is invoked with each text delta during streaming.
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]

    _run_codex_app_server_turn = _forward("agent.codex_runtime", "run_codex_app_server_turn")


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
