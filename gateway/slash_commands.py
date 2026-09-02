"""Gateway slash-command handlers for GatewayRunner.

The in-session slash commands (/model, /reset, /usage, ...) dispatched from ``_handle_message``,
lifted out of ``gateway/run.py`` into a mixin so every ``self._handle_*_command`` reference keeps
working via the MRO. run.py helpers (``_hermes_home``, ``_load_gateway_config``, ...) are imported
lazily inside handler bodies — a deferred ``from gateway.run import ...`` avoids the import cycle.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import logging
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.i18n import t
from agent.turn_context import extract_api_content_sidecar
from gateway.config import HomeChannel, Platform, PlatformConfig, persist_home_channel
from gateway.platforms.base import EphemeralReply, MessageEvent, MessageType
from gateway.session import (
    AsyncSessionStore,
    SessionSource,
    build_session_key,
    is_shared_multi_user_session,
)
from hermes_cli.config import atomic_config_write, cfg_get, clear_model_endpoint_credentials
from utils import (
    atomic_json_write,
    base_url_host_matches,
    is_truthy_value,
)
import contextlib

logger = logging.getLogger("gateway.run")

# Upper bound on the off-loop agent-resource cleanup during a /new or /reset (see
# _handle_reset_command). A stuck teardown must not block the event loop; past this the reset
# proceeds and the cleanup is left to finish (or leak) in its worker thread.
_RESET_CLEANUP_TIMEOUT_S = 30.0

# /rollback result keys -> i18n line for files the safe restore left alone.
_ROLLBACK_SKIP_LINES = (
    ("skipped_user_edits", "gateway.rollback.kept_user_edits"),
    ("skipped_oversize", "gateway.rollback.kept_oversize"),
    ("failed_deletes", "gateway.rollback.failed_deletes"),
)

# /busy input modes -> (status-card behavior, set-confirmation behavior).
_BUSY_MODE_BEHAVIOR = {
    "queue": ("queues for next turn", "Messages will be queued for the next turn while Hermes is busy."),
    "steer": (
        "steers into current run (after next tool call)",
        "Messages will be steered into the current run (after the next tool call).",
    ),
    "interrupt": ("interrupts current run", "Messages will interrupt the current run while Hermes is busy."),
}

# /voice subcommand -> stored mode (None = auto-TTS disabled), confirmation i18n key.
_VOICE_MODE_BY_ARG = {
    "on": ("voice_only", "gateway.voice.enabled_voice_only"),
    "enable": ("voice_only", "gateway.voice.enabled_voice_only"),
    "off": ("off", "gateway.voice.disabled_text"),
    "disable": ("off", "gateway.voice.disabled_text"),
    "tts": ("all", "gateway.voice.tts_enabled"),
}


def _clean_str(value: Any) -> str:
    """Strip and return a non-empty string value, or empty string."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _int_value(value: Any) -> int:
    """Safely coerce to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _model_switch_skew_guard() -> Optional[str]:
    """Refuse a model switch when the gateway is running stale code.

    A long-lived gateway keeps boot-time modules in memory; if the checkout changed underneath it,
    a first-time lazy import on a new code path can crash on a stale cached dependency. Detect the
    drift and ask for a restart. Scoped to model switching only (the highest-risk trigger).
    """
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return t(
        "gateway.model.error_prefix",
        error=(
            f"This gateway is running code from {boot_rev} but the checkout on "
            f"disk is now {disk_rev}. Switching models would risk a stale-module "
            f"crash — restart the gateway to load the new code: hermes gateway restart"
        ),
    )


def _manual_compression_reply_lines(summary: dict, compressor, focus_topic) -> list[str]:
    """Lines for the manual /compress confirmation, surfacing summariser/aux-model failures.

    ``_last_compress_aborted`` = no usable summary, messages unchanged (force=True bypasses any
    cooldown). Provider exception text is force-redacted at this UI boundary even when global
    redaction is off. A configured aux model that failed and was recovered via main is an info
    note so the user can fix their config.
    """
    lines = [f"🗜️ {summary['headline']}"]
    if focus_topic:
        lines.append(t("gateway.compress.focus_line", topic=focus_topic))
    lines.append(summary["token_line"])
    if summary["note"]:
        lines.append(summary["note"])
    summary_err = getattr(compressor, "_last_summary_error", None)
    if summary_err:
        from agent.redact import redact_sensitive_text
        summary_err = redact_sensitive_text(summary_err, force=True)
    aux_fail_model = getattr(compressor, "_last_aux_model_failure_model", None)
    if getattr(compressor, "_last_compress_aborted", False):
        lines.append(t("gateway.compress.aborted", error=(summary_err or "unknown error")))
    elif aux_fail_model:
        lines.append(
            t(
                "gateway.compress.aux_failed",
                model=aux_fail_model,
                error=(getattr(compressor, "_last_aux_model_failure_error", None) or "unknown error"),
            )
        )
    return lines


async def _persist_model_switch_to_config(result, config_path) -> None:
    """Write-through a resolved /model switch to ``config_path`` (model.default/provider/base_url).

    Write-back round-trip: raw read is correct (merged defaults must not be persisted back to the
    user's file). A scalar/None ``model:`` is coerced into a dict first — otherwise
    ``cfg.setdefault("model", {})`` returns the existing scalar and the next assignment raises
    ``TypeError``. Named providers re-resolve base_url/api_mode fresh, so leftovers are cleared
    unconditionally; custom providers have no registry entry to re-derive from, so they need an
    explicit set-or-clear (a lone ``if base_url:`` leaves stale values).
    """
    from hermes_cli.config import read_user_config_raw, save_config

    cfg = read_user_config_raw(config_path)
    raw_model = cfg.get("model")
    if isinstance(raw_model, dict):
        model_cfg = raw_model
    elif isinstance(raw_model, str) and raw_model.strip():
        model_cfg = cfg["model"] = {"default": raw_model.strip()}
    else:
        model_cfg = cfg["model"] = {}
    try:
        from hermes_cli.route_identity import should_clear_context_pin_async

        if await should_clear_context_pin_async(
            model_cfg.get("default") or model_cfg.get("model"),
            result.new_model,
            model_cfg.get("base_url"),
            result.base_url,
            model_cfg.get("provider"),
            result.target_provider,
        ):
            model_cfg.pop("context_length", None)
    except Exception:
        model_cfg.pop("context_length", None)
    model_cfg["default"] = result.new_model
    model_cfg["provider"] = result.target_provider
    is_custom_target = str(result.target_provider or "").strip().lower() == "custom"
    if result.base_url:
        model_cfg["base_url"] = result.base_url
    elif is_custom_target:
        model_cfg.pop("base_url", None)
    if is_custom_target:
        if result.api_mode:
            model_cfg["api_mode"] = result.api_mode
        else:
            model_cfg.pop("api_mode", None)
    else:
        clear_model_endpoint_credentials(model_cfg, clear_base_url=True)
    save_config(cfg)


def _read_model_command_config(config_path):
    """Current (model, provider, base_url, user_providers, custom_providers, excluded) for /model.

    Fail-open: any config read error yields the defaults (``provider="openrouter"``).
    """
    from gateway.run import _load_gateway_config

    current_model, current_provider, current_base_url = "", "openrouter", ""
    user_provs = custom_provs = None
    excluded_provs: list = []
    try:
        cfg = _load_gateway_config(config_path=config_path)
        if cfg:
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                current_model = model_cfg.get("default", "")
                current_provider = model_cfg.get("provider", current_provider)
                current_base_url = model_cfg.get("base_url", "")
            user_provs = cfg.get("providers")
            try:
                from hermes_cli.config import get_compatible_custom_providers
                custom_provs = get_compatible_custom_providers(cfg)
            except Exception:
                custom_provs = cfg.get("custom_providers")
            _excl = cfg.get("model_catalog", {}).get("excluded_providers")
            if isinstance(_excl, list):
                excluded_provs = _excl
    except Exception:
        pass
    return current_model, current_provider, current_base_url, user_provs, custom_provs, excluded_provs


def _model_provider_listing_lines(providers) -> list[str]:
    """Text-list body for ``/model`` with no args on platforms without a picker."""
    lines: list[str] = []
    for p in providers:
        tag = t("gateway.model.current_tag") if p["is_current"] else ""
        lines.append(f"**{p['name']}** `--provider {p['slug']}`{tag}:")
        if p["models"]:
            model_strs = ", ".join(f"`{m}`" for m in p["models"])
            extra = t("gateway.model.more_models_suffix", count=p["total_models"] - len(p["models"])) if p["total_models"] > len(p["models"]) else ""
            lines.append(f"  {model_strs}{extra}")
        elif p.get("api_url"):
            lines.append(f"  `{p['api_url']}`")
        lines.append("")
    return lines


def _status_model_route(status_agent, persisted_route: dict, session_row: dict, session_entry):
    """``(model, provider, context_used, context_total)`` for /status.

    Order: live/cached agent route -> persisted dominant route -> SessionDB row -> gateway config
    (only loaded when something is still missing).
    """
    from gateway.run import _AGENT_PENDING_SENTINEL, _load_gateway_config, _resolve_gateway_model

    model_name = provider_name = ""
    route_resolved = False
    context_used = context_total = 0
    if status_agent is not None and status_agent is not _AGENT_PENDING_SENTINEL:
        live_model = _clean_str(getattr(status_agent, "model", ""))
        live_provider = _clean_str(getattr(status_agent, "provider", ""))
        if live_model and live_provider:
            model_name, provider_name, route_resolved = live_model, live_provider, True
        ctx = getattr(status_agent, "context_compressor", None)
        if ctx is not None:
            context_used = _int_value(getattr(ctx, "last_prompt_tokens", 0))
            context_total = _int_value(getattr(ctx, "context_length", 0))

    persisted_model = _clean_str(persisted_route.get("model"))
    persisted_provider = _clean_str(persisted_route.get("billing_provider"))
    if not route_resolved and persisted_model and persisted_provider:
        model_name, provider_name, route_resolved = persisted_model, persisted_provider, True
    if not route_resolved:
        model_name = _clean_str(session_row.get("model"))
        provider_name = _clean_str(session_row.get("billing_provider"))
    context_used = context_used or _int_value(getattr(session_entry, "last_prompt_tokens", 0))

    user_config: dict[str, Any] = {}
    if not model_name or not provider_name or not context_total:
        try:
            user_config = _load_gateway_config()
        except Exception:
            user_config = {}
    model_cfg = user_config.get("model", {}) if isinstance(user_config, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not model_name:
        model_name = _resolve_gateway_model(user_config)
    if not provider_name:
        provider_name = _clean_str(model_cfg.get("provider"))
    if not context_total:
        configured_context = model_cfg.get("context_length")
        if isinstance(configured_context, int) and configured_context > 0:
            context_total = configured_context
    return model_name, provider_name, context_used, context_total


def _context_compressor_lines(agent, ctx, used: int) -> list[str]:
    """/context full view: auto-compression threshold/headroom, compression count + last savings,
    and cumulative throughput (labelled as throughput, NOT context size)."""
    lines: list[str] = []
    threshold = getattr(ctx, "threshold_tokens", 0) or 0
    threshold_pct = (getattr(ctx, "threshold_percent", 0) or 0) * 100
    if threshold > 0:
        if used >= threshold:
            lines.append(
                t("gateway.context.over_threshold", threshold=f"{threshold:,}", threshold_pct=f"{threshold_pct:.0f}")
            )
        else:
            lines.append(
                t(
                    "gateway.context.threshold",
                    threshold=f"{threshold:,}",
                    threshold_pct=f"{threshold_pct:.0f}",
                    to_go=f"{threshold - used:,}",
                )
            )
    compressions = getattr(ctx, "compression_count", 0) or 0
    lines.append(t("gateway.context.compressions", count=compressions))
    if compressions:
        savings = getattr(ctx, "_last_compression_savings_pct", None)
        if savings is not None:
            lines.append(t("gateway.context.last_savings", savings=f"{savings:.0f}"))

    def _n(attr):
        return getattr(agent, attr, 0) or 0

    lines.append("")
    lines.append(t("gateway.context.totals_header", calls=_n("session_api_calls")))
    lines.append(
        t(
            "gateway.context.totals_line",
            input=f"{_n('session_input_tokens'):,}",
            output=f"{_n('session_output_tokens'):,}",
            reasoning=f"{_n('session_reasoning_tokens'):,}",
        )
    )
    lines.append(t("gateway.context.total_billed", total=f"{_n('session_total_tokens'):,}"))
    lines.append(t("gateway.context.throughput_note"))
    return lines


def _agents_delegation_lines(d: dict) -> list[str]:
    """/agents rows for one background delegation. Live per-child activity comes from the
    registry's progress sampler: api calls, current tool, seconds since last activity."""
    goal = " ".join(str(d.get("goal") or "").split())
    if len(goal) > 70:
        goal = goal[:67] + "..."
    status = d.get("status", "?")
    row = f"- `{d.get('delegation_id', '?')}` · {status}"
    if status == "stalling":
        quiet = d.get("stalled_after_quiet_seconds")
        if quiet is not None:
            row += f" · no progress {quiet:.0f}s"
    elif d.get("seconds_since_progress", 0) >= 60:
        row += f" · quiet {d['seconds_since_progress']:.0f}s"
    if goal:
        row += f" · {goal}"
    lines = [row]
    for i, child in enumerate(d.get("children_activity") or []):
        if not isinstance(child, dict):
            continue
        tool = child.get("current_tool")
        doing = f"`{tool}`" if tool else "between turns"
        part = f"  - child {i + 1}: {child.get('api_calls', '?')} api calls · {doing}"
        idle = child.get("seconds_since_activity")
        if idle is not None:
            part += f" · active {idle:.0f}s ago"
        lines.append(part)
    return lines


def _usage_agent_stats_lines(agent) -> list[str]:
    """/usage session block for a live agent: rate limits, token breakdown (matches the CLI),
    context window and compression count."""
    lines: list[str] = []
    rl_state = agent.get_rate_limit_state()
    if rl_state and rl_state.has_data:
        from agent.rate_limit_tracker import format_rate_limit_compact
        lines.append(t("gateway.usage.rate_limits", state=format_rate_limit_compact(rl_state)))
        lines.append("")
    input_tokens = getattr(agent, "session_input_tokens", 0) or 0
    output_tokens = getattr(agent, "session_output_tokens", 0) or 0
    lines.append(t("gateway.usage.header_session"))
    lines.append(t("gateway.usage.label_model", model=agent.model))
    lines.append(t("gateway.usage.label_input_tokens", count=f"{input_tokens:,}"))
    lines.append(t("gateway.usage.label_output_tokens", count=f"{output_tokens:,}"))
    lines.append(t("gateway.usage.label_total", count=f"{agent.session_total_tokens:,}"))
    lines.append(t("gateway.usage.label_api_calls", count=agent.session_api_calls))
    ctx = agent.context_compressor
    _lpt = ctx.last_prompt_tokens if ctx.last_prompt_tokens > 0 else 0
    if _lpt:
        pct = min(100, _lpt / ctx.context_length * 100) if ctx.context_length else 0
        lines.append(t("gateway.usage.label_context", used=f"{_lpt:,}", total=f"{ctx.context_length:,}", pct=f"{pct:.0f}"))
    if ctx.compression_count:
        lines.append(t("gateway.usage.label_compressions", count=ctx.compression_count))
    return lines


def _home_thread_from_source(source) -> Optional[str]:
    """The thread id /sethome should persist on the home target, or None.

    Slack thread-per-message keying stamps a top-level message's own id as ``source.thread_id`` (a
    session key, not a location); persisting it would pin HOME to that ephemeral thread. A thread
    id equal to the message's own id is synthetic and dropped; a real thread (id = parent's) is kept.
    """
    thread_id = getattr(source, "thread_id", None)
    if not thread_id:
        return None
    if (
        getattr(source, "platform", None) == Platform.SLACK
        and getattr(source, "message_id", None)
        and str(thread_id) == str(source.message_id)
    ):
        return None
    return str(thread_id)


class GatewaySlashCommandsMixin:
    """In-session slash-command handlers for GatewayRunner."""

    async_session_store: AsyncSessionStore

    # ------------------------------------------------------------------ shared helpers
    def _cached_agent_for(self, session_key: str):
        """Peek the cached AIAgent for *session_key* without evicting it, or None.

        Cache entries are ``(agent, signature, ...)`` tuples; bare agents (test doubles) are
        accepted too. Lock/cache may be absent on fixtures that skip ``__init__``.
        """
        cache = getattr(self, "_agent_cache", None)
        if cache is None:
            return None
        lock = getattr(self, "_agent_cache_lock", None)
        try:
            if lock is not None:
                with lock:
                    entry = cache.get(session_key)
            else:
                entry = cache.get(session_key)
        except Exception:
            return None
        if isinstance(entry, (tuple, list)):
            return entry[0] if entry else None
        return entry or None

    def _resident_agent_for(self, session_key: str):
        """The live running agent for *session_key*, else the cached one, else None.

        The pending sentinel (a run that is starting) never counts as a usable agent.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL

        agent = self._running_agents.get(session_key)
        if agent is not None and agent is not _AGENT_PENDING_SENTINEL:
            return agent
        return self._cached_agent_for(session_key)

    @staticmethod
    def _session_db_unavailable_reply() -> str:
        from hermes_state import format_session_db_unavailable
        return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

    def _reply_metadata(self, event: MessageEvent):
        """Thread/reply metadata for an outbound send anchored on *event*."""
        return self._thread_metadata_for_source(event.source, self._reply_anchor_for_event(event))

    def _adapter_and_key_for(self, event: MessageEvent):
        """``(adapter, session_key)`` for the event's source, either None when no source."""
        if not event.source:
            return None, None
        return self.adapters.get(event.source.platform), self._session_key_for_source(event.source)

    def _telegramized_command_reply(self, event: MessageEvent, text: str) -> str:
        from gateway.run import _telegramize_command_mentions
        return _telegramize_command_mentions(
            text, getattr(getattr(event, "source", None), "platform", None)
        )

    def _checkpoint_manager(self):
        """A CheckpointManager from gateway config, or None when checkpoints are disabled."""
        from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config
        from tools.checkpoint_manager import CheckpointManager

        cp_kwargs = _checkpoint_agent_kwargs(_load_gateway_config())
        if not cp_kwargs["checkpoints_enabled"]:
            return None
        return CheckpointManager(
            enabled=True,
            max_snapshots=cp_kwargs["checkpoint_max_snapshots"],
            max_total_size_mb=cp_kwargs["checkpoint_max_total_size_mb"],
            max_file_size_mb=cp_kwargs["checkpoint_max_file_size_mb"],
        )

    def _write_approval_setter(self, section: str, session_key: str):
        """``set_mode_fn`` for /memory and /skills: persist ``<section>.write_approval``.

        Write-back round-trip: raw read is correct (merged defaults must not be persisted back to
        the user's file). The new setting must take effect next message, so the cached agent is dropped.
        """
        from gateway.run import _gateway_config_home
        from hermes_cli.config import read_user_config_raw
        config_path = _gateway_config_home() / "config.yaml"

        def _set_approval(enabled: bool):
            user_config = read_user_config_raw(config_path)
            user_config.setdefault(section, {})["write_approval"] = bool(enabled)
            atomic_config_write(config_path, user_config)
            self._evict_cached_agent(session_key)

        return _set_approval

    async def _deliver_approval_confirmation(self, event: MessageEvent, confirmation_text: str, verb: str):
        """Return *confirmation_text* for normal delivery, or push it on native-streaming adapters.

        Native-streaming adapters (WeCom msgtype:"stream") need the confirmation sent directly with
        control-lane metadata (reliable proactive send, not the finalized reply stream). Everyone
        else returns text for normal delivery. (``is not True``: mocks auto-create attrs.)
        """
        source = event.source
        adapter = self.adapters.get(source.platform)
        if adapter:
            adapter.resume_typing_for_chat(source.chat_id)  # agent is about to continue
        if getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False) is not True:
            return confirmation_text
        if adapter:
            try:
                await adapter.send(
                    source.chat_id,
                    confirmation_text,
                    reply_to=event.message_id,
                    metadata={"is_approval_prompt": True, "force_proactive_send": True},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to send /%s confirmation to %s: %s", verb, source.chat_id, exc, exc_info=True,
                )
        return None

    def _typed_command_prefix_for(self, platform) -> str:
        """Return the prefix users can always type to reach Hermes commands.

        Adapter ``typed_command_prefix`` capability (default "/"). Slack and Matrix use "!" because
        typed "/" is blocked/reserved there; their adapters rewrite "!command" to "/command".
        """
        adapter = self.adapters.get(platform) if getattr(self, "adapters", None) else None
        return getattr(adapter, "typed_command_prefix", "/") if adapter is not None else "/"

    async def _handle_reset_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /new or /reset command."""
        source = event.source

        # Get existing session key
        session_key = self._session_key_for_source(source)
        self._invalidate_session_run_generation(session_key, reason="session_reset")
        # Evict the running-agent slot now that the generation is bumped: the in-flight run's own
        # guarded release (old generation) returns False and would leave a zombie slot that silently
        # drops all later messages. Idempotent, so the run's finally calling it again is harmless.
        self._release_running_agent_state(session_key)

        # Snapshot the old entry so on_session_finalize can report the
        # expiring session id before reset_session() rotates it.
        old_entry = self.session_store._entries.get(session_key)

        # Close the old agent's tool resources (sandboxes, browser daemons, subprocesses) before
        # evicting it; getattr-guarded since test fixtures may skip __init__. _cleanup_agent_resources
        # is blocking and this handler runs ON the event loop (confirm-button click), so an inline
        # call wedges the loop — offload to a worker thread with a bounded timeout.
        _old_agent = self._cached_agent_for(session_key)
        if _old_agent is not None:
            try:
                await asyncio.wait_for(
                    self._run_in_executor_with_context(self._cleanup_agent_resources, _old_agent),
                    timeout=_RESET_CLEANUP_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                # wait_for cancels the await, but the worker thread cannot be cancelled — a wedged
                # teardown keeps running (or leaks) for the gateway's lifetime. The reset proceeds.
                logger.warning(
                    "Agent resource cleanup for session %s exceeded %ss during "
                    "/new reset; proceeding with reset (the worker thread is left "
                    "to finish on its own). (#35994)",
                    session_key, _RESET_CLEANUP_TIMEOUT_S,
                )
            except Exception as cleanup_exc:
                logger.warning(
                    "Agent resource cleanup for session %s failed during /new "
                    "reset: %s (#35994)",
                    session_key, cleanup_exc,
                )
        self._evict_cached_agent(session_key)

        # Conversation boundary: clear ALL conversation-scoped per-session state (model/reasoning
        # overrides, one-turn restores, model notes, last-resolved cache, /queue overflow) +
        # security state in one funnel call. See _CONVERSATION_SCOPED_STATE in gateway/run.py.
        self._clear_conversation_scope(session_key, reason="session_reset")

        # The old conversation's in-flight async delegations end WITH it: once the session id rotates
        # their completions have no live owner (orphaned payload on the shared queue, wasted tokens).
        # Interrupt by expiring durable session id (parent_session_id), routing key as legacy fallback.
        try:
            from tools.async_delegation import interrupt_for_session

            interrupt_for_session(
                session_key=session_key,
                parent_session_id=str(getattr(old_entry, "session_id", "") or ""),
                reason="session_reset",
            )
        except Exception:
            pass

        try:
            from tools.env_passthrough import clear_env_passthrough
            clear_env_passthrough()
        except Exception:
            pass

        try:
            from tools.credential_files import clear_credential_files
            clear_credential_files()
        except Exception:
            pass

        # Reset the session
        new_entry = await self.async_session_store.reset_session(session_key)

        # (Conversation-scoped overrides + security state were already
        # cleared via _clear_conversation_scope above.)

        _old_sid = old_entry.session_id if old_entry else None

        # Fire plugin on_session_finalize hook (session boundary). Off-loop + bounded: finalize
        # hooks can block arbitrarily (observability trace exports) and this handler runs on the
        # gateway event loop (see GatewayRunner._finalize_session_off_loop).
        with contextlib.suppress(Exception):
            await self._finalize_session_off_loop(
                session_id=_old_sid,
                platform=source.platform.value if source.platform else "",
                reason="new_session",
                old_session_id=_old_sid,
                new_session_id=new_entry.session_id if new_entry else None,
            )

        # Emit session:end hook (session is ending)
        await self.hooks.emit("session:end", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Emit session:reset hook
        await self.hooks.emit("session:reset", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Resolve session config info to surface to the user, scoped to the
        # profile serving this source so a multiplexed /reset //new banner
        # reports the profile's model, not the base config's (#59003).
        try:
            session_info = await asyncio.to_thread(
                self._reset_notice_session_info, source
            )
        except Exception:
            session_info = ""

        if new_entry:
            header = await asyncio.to_thread(self._telegram_topic_new_header, source) or t("gateway.reset.header_default")
        else:
            # No existing session, just create one
            new_entry = await self.async_session_store.get_or_create_session(source, force_new=True)
            header = await asyncio.to_thread(self._telegram_topic_new_header, source) or t("gateway.reset.header_new")

        # Set session title if provided with /new <title>
        _title_arg = event.get_command_args().strip()
        if _title_arg and self._session_db and new_entry:
            header = await self._reset_titled_header(header, new_entry.session_id, _title_arg)

        # When /new runs inside a Telegram DM topic lane, rewrite the (chat_id, thread_id) →
        # session_id binding so the next message uses the freshly-created session. Otherwise the
        # binding-lookup at the top of _handle_message_with_agent switches right back to the old one.
        if await asyncio.to_thread(self._is_telegram_topic_lane, source) and new_entry is not None:
            try:
                await asyncio.to_thread(self._record_telegram_topic_binding, source, new_entry)
            except Exception:
                logger.debug("Failed to rebind Telegram topic after /new", exc_info=True)

        # Fire plugin on_session_reset hook (new session guaranteed to exist)
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _new_sid = new_entry.session_id if new_entry else None
            _invoke_hook(
                "on_session_reset",
                session_id=_new_sid,
                platform=source.platform.value if source.platform else "",
                reason="new_session",
                old_session_id=_old_sid,
                new_session_id=_new_sid,
            )
        except Exception:
            pass

        # Append a random tip to the reset message
        try:
            from hermes_cli.tips import get_random_tip
            _tip_line = t("gateway.reset.tip", tip=get_random_tip())
        except Exception:
            _tip_line = ""

        if session_info:
            return EphemeralReply(f"{header}\n\n{session_info}{_tip_line}")
        return EphemeralReply(f"{header}{_tip_line}")

    async def _reset_titled_header(self, header: str, session_id: str, title_arg: str) -> str:
        """Apply ``/new <title>``: titled header on success, else the header plus a rejection note."""
        from hermes_state import SessionDB
        note = ""
        try:
            sanitized = SessionDB.sanitize_title(title_arg)
        except ValueError as e:
            sanitized = None
            note = t("gateway.reset.title_rejected", error=str(e))
        if sanitized:
            try:
                await self._session_db.set_session_title(session_id, sanitized)
                header = t("gateway.reset.header_titled", title=sanitized)
            except ValueError as e:
                note = t("gateway.reset.title_error_untitled", error=str(e))
            except Exception:
                pass
        elif not note:
            # sanitize_title returned empty (whitespace-only / unprintable)
            note = t("gateway.reset.title_empty_untitled")
        return header + note

    async def _handle_profile_command(self, event: MessageEvent) -> str:
        """Handle /profile — show the profile serving this source and its home.

        On a multiplexed gateway the process-level active profile is the multiplexer's own, so it
        would read "default" in every chat. With ``multiplex_profiles`` on, report ``source.profile``
        and resolve home under that profile's runtime scope (like the scoped /reset banner); when
        off the stamp is ignored, mirroring ``_run_agent``.
        """
        from hermes_constants import display_hermes_home
        from hermes_cli.slash_exec import CommandContext, execute_command

        multiplexed = getattr(
            getattr(self, "config", None), "multiplex_profiles", False
        )
        source = getattr(event, "source", None)

        profile_name = ""
        display = ""
        if multiplexed:
            profile_name = (getattr(source, "profile", "") or "").strip()
            try:
                from gateway.run import _profile_runtime_scope

                profile_home = self._resolve_profile_home_for_source(source)
                with _profile_runtime_scope(profile_home):
                    display = display_hermes_home()
            except Exception:
                display = display_hermes_home()

        # Shared executor resolves process-level fallbacks; the multiplexed
        # per-source overrides (when any) ride in via options.
        reply = execute_command(
            "profile",
            CommandContext(
                surface="gateway",
                options={"profile_name": profile_name, "home_display": display},
            ),
        )

        lines = [
            t("gateway.profile.header", profile=reply.data["profile"]),
            t("gateway.profile.home", home=reply.data["home"]),
        ]

        return "\n".join(lines)

    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        """Handle /whoami — show the user's slash command access on this scope.

        Always allowed (slash_access floor). Reports platform, DM-vs-group scope, tier, and the
        commands the user can actually run here.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        source = event.source
        policy = _policy_for_source(self.config, source)
        platform = source.platform.value if source and source.platform else "?"
        chat_type = (source.chat_type if source else "") or "dm"
        scope = "DM" if chat_type.lower() in {"dm", "direct", "private", ""} else "group/channel"
        user_id = (source.user_id if source else None) or "?"

        if not policy.enabled:
            return (
                f"**You** — {platform} ({scope})\n"
                f"User ID: `{user_id}`\n"
                f"Tier: unrestricted (no admin list configured for this scope)\n"
                f"Slash commands: all available"
            )

        if policy.is_admin(user_id):
            return (
                f"**You** — {platform} ({scope})\n"
                f"User ID: `{user_id}`\n"
                f"Tier: **admin**\n"
                f"Slash commands: all available"
            )

        # Non-admin user. Show what's actually reachable.
        floor = ["help", "whoami"]  # mirrors slash_access._ALWAYS_ALLOWED_FOR_USERS
        configured = sorted(policy.user_allowed_commands)
        # Combine + dedupe, preserve order: floor first, then operator additions.
        seen: set[str] = set()
        runnable: list[str] = []
        for c in floor + configured:
            if c not in seen:
                seen.add(c)
                runnable.append(c)
        runnable_str = ", ".join(f"/{c}" for c in runnable) if runnable else "(none)"
        return (
            f"**You** — {platform} ({scope})\n"
            f"User ID: `{user_id}`\n"
            f"Tier: user\n"
            f"Slash commands you can run: {runnable_str}"
        )

    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        """Handle /kanban — delegate to the shared kanban CLI.

        DB work runs in a thread pool to keep the event loop responsive. Reads and mutations are
        allowed while an agent runs: the board is profile-agnostic and never touches agent state.
        """
        from hermes_cli.kanban import run_slash

        text = (event.text or "").strip()
        # Strip the leading "/kanban" (with or without slash), leaving args.
        if text.startswith("/"):
            text = text.lstrip("/")
        if text.startswith("kanban"):
            text = text[len("kanban"):].lstrip()

        tokens = shlex.split(text) if text else []
        requested_board = None
        action = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--board":
                if i + 1 >= len(tokens):
                    break
                requested_board = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--board="):
                requested_board = tok.split("=", 1)[1]
                i += 1
                continue
            action = tok
            break

        is_create = action == "create"

        try:
            output = await asyncio.to_thread(run_slash, text)
        except Exception as exc:  # pragma: no cover - defensive
            return t("gateway.kanban.error_prefix", error=exc)

        # Auto-subscribe on create. Parse the task id from the CLI's standard success line ("Created
        # t_abcd  (ready, assignee=...)"). If the user passed --json we don't subscribe; they're
        # clearly scripting and can call /kanban notify-subscribe explicitly.
        if is_create and output:
            m = re.search(r"Created\s+(t_[0-9a-f]+)\b", output)
            if m:
                task_id = m.group(1)
                try:
                    if await self._kanban_auto_subscribe(event, task_id, requested_board):
                        output = (
                            output.rstrip()
                            + "\n"
                            + t("gateway.kanban.subscribed_suffix", task_id=task_id)
                        )
                except Exception as exc:
                    logger.warning("kanban create auto-subscribe failed: %s", exc)

        # Gateway messages have practical length caps; truncate long
        # listings to keep the UX reasonable.
        if len(output) > 3800:
            output = output[:3800] + "\n" + t("gateway.kanban.truncated_suffix")
        return output or t("gateway.kanban.no_output")

    async def _kanban_auto_subscribe(self, event: MessageEvent, task_id: str, requested_board) -> bool:
        """Subscribe the event's chat to *task_id* notifications (notify+wake). False when the
        source has no platform/chat to route back to."""
        source = event.source
        platform = getattr(source, "platform", None)
        platform_str = (platform.value if hasattr(platform, "value") else str(platform or "")).lower()
        chat_id = str(getattr(source, "chat_id", "") or "")
        chat_type = str(getattr(source, "chat_type", "") or "") or None
        thread_id = str(getattr(source, "thread_id", "") or "")
        user_id = str(getattr(source, "user_id", "") or "") or None
        # Also persist the stable alt id (Signal UUID, Feishu union_id): build_session_key keys the
        # participant on ``user_id_alt or user_id``, so a replayed wake rebuilds the same session
        # key only when the alt id survives the round-trip.
        user_id_alt = str(getattr(source, "user_id_alt", "") or "") or None
        delivery_metadata = self._reply_metadata(event) or None
        if isinstance(delivery_metadata, dict):
            chat_type = str(getattr(source, "chat_type", "") or "")
            if chat_type:
                delivery_metadata.setdefault("chat_type", chat_type)
        if not (platform_str and chat_id):
            return False

        def _sub():
            from hermes_cli import kanban_db as _kb
            conn = _kb.connect(board=requested_board)
            try:
                _kb.add_notify_sub(
                    conn, task_id=task_id,
                    platform=platform_str, chat_id=chat_id,
                    chat_type=chat_type,
                    thread_id=thread_id or None,
                    user_id=user_id,
                    user_id_alt=user_id_alt,
                    notifier_profile=getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name(),
                    # Subscribing from chat: deliver the passive message and wake the destination agent.
                    delivery_mode="notify+wake",
                    delivery_metadata=delivery_metadata,
                )
            finally:
                conn.close()
        await asyncio.to_thread(_sub)
        return True

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)

        connected_platforms = [p.value for p in self.adapters]

        # Check if there's an active agent. Keep the sentinel distinct: a
        # starting/pending run should not be treated as a fully usable agent for
        # model/context display, but it still occupies the session slot.
        session_key = session_entry.session_key
        agent = self._running_agents.get(session_key)
        is_running = agent is not None and agent is not _AGENT_PENDING_SENTINEL

        # Count pending /queue follow-ups (slot + overflow).
        adapter = self.adapters.get(source.platform) if source else None
        queue_depth = self._queue_depth(session_key, adapter=adapter)

        title, session_row, db_total_tokens, persisted_route = await self._status_session_db_facts(
            session_entry.session_id
        )
        # Resolve model/context for cockpit-style status. Prefer the live or cached agent because it
        # carries the actual runtime route and context compressor; fall back to SessionDB metadata +
        # last_prompt_tokens so /status stays useful between turns without billing/account calls.
        status_agent = agent if is_running else self._cached_agent_for(session_key)
        model_name, provider_name, context_used, context_total = _status_model_route(
            status_agent, persisted_route, session_row, session_entry
        )

        model_line = ""
        if model_name:
            if provider_name:
                model_line = t("gateway.status.model_provider", model=model_name, provider=provider_name)
            else:
                model_line = t("gateway.status.model", model=model_name)

        context_line = ""
        if context_total:
            pct = min(100, round((context_used / context_total) * 100)) if context_total else 0
            context_line = t(
                "gateway.status.context",
                used=f"{context_used:,}",
                total=f"{context_total:,}",
                pct=f"{pct}",
            )
        elif context_used:
            context_line = t("gateway.status.context_used", used=f"{context_used:,}")

        lines = [
            t("gateway.status.header"),
            "",
            t("gateway.status.session_id", session_id=session_entry.session_id),
        ]
        if title:
            lines.append(t("gateway.status.title", title=title))
        lines.extend([
            t("gateway.status.created", timestamp=session_entry.created_at.strftime('%Y-%m-%d %H:%M')),
            t("gateway.status.last_activity", timestamp=session_entry.updated_at.strftime('%Y-%m-%d %H:%M')),
        ])
        if model_line:
            lines.append(model_line)
        if context_line:
            lines.append(context_line)
        lines.extend([
            t("gateway.status.tokens", tokens=f"{db_total_tokens:,}"),
            t("gateway.status.agent_running", state=t("gateway.status.state_yes") if is_running else t("gateway.status.state_no")),
        ])
        if queue_depth:
            lines.append(t("gateway.status.queued", count=queue_depth))
        if source.platform == Platform.MATRIX:
            scope = getattr(self.adapters.get(Platform.MATRIX), "_matrix_session_scope", os.getenv("MATRIX_SESSION_SCOPE", "auto"))
            thread = source.thread_id or "none"
            lines.extend([
                "",
                t("gateway.status.matrix_scope_header"),
                t("gateway.status.matrix_scope_room", room=source.chat_name or source.chat_id),
                t("gateway.status.matrix_scope_room_id", room_id=source.chat_id),
                t("gateway.status.matrix_scope_thread", thread_id=thread),
                t("gateway.status.matrix_scope_mode", scope=scope),
                t(
                    "gateway.status.matrix_scope_key",
                    session_key=self._redact_matrix_session_key(session_key),
                ),
            ])
        lines.extend([
            "",
            t("gateway.status.platforms", platforms=', '.join(connected_platforms)),
        ])

        return "\n".join(lines)

    async def _status_session_db_facts(self, session_id: str):
        """``(title, session_row, db_total_tokens, persisted_route)`` for /status; each fail-open.

        Token totals come from the SQLite session DB rather than the in-memory SessionStore: the
        agent's per-turn token deltas are persisted into sessions_db (run_agent.py), not into
        SessionEntry, so session_entry.total_tokens is always 0.
        """
        title = None
        session_row: dict[str, Any] = {}
        db_total_tokens = 0
        persisted_route: dict[str, Any] = {}
        if not self._session_db:
            return title, session_row, db_total_tokens, persisted_route
        try:
            title = await self._session_db.get_session_title(session_id)
        except Exception:
            title = None
        try:
            row = await self._session_db.get_session(session_id)
            if isinstance(row, dict):
                session_row = row
                db_total_tokens = sum(
                    _int_value(row.get(k))
                    for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
                )
        except Exception:
            db_total_tokens = 0
        try:
            route = await self._session_db.get_dominant_session_model_route(session_id)
            if isinstance(route, dict):
                persisted_route = route
        except Exception:
            persisted_route = {}
        return title, session_row, db_total_tokens, persisted_route

    @staticmethod
    def _redact_matrix_session_key(session_key: str) -> str:
        """Return a stable Matrix session-key fingerprint for shared room status."""
        text = str(session_key or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

    async def _handle_context_command(self, event: MessageEvent) -> str:
        """Handle /context — the dedicated context-window view.

        /status shows a one-line ``used / total`` summary; this command is the deep view: a usage
        gauge, auto-compression threshold and headroom, compression count and last savings, and
        cumulative throughput — the last clearly labelled as throughput, NOT context size.
        Resolution order: running agent, cached agent, SessionStore/SessionDB metadata, and a
        transcript estimate only as last resort. ``/context all`` adds per-skill/toolset listings.
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        session_entry = await self.async_session_store.get_or_create_session(source)
        expanded = event.get_command_args().strip().lower() in {"all", "full", "details"}

        # Running agent first (mid-turn), then cached agent (between turns).
        agent = self._resident_agent_for(session_key)
        has_agent = bool(agent)

        ctx = getattr(agent, "context_compressor", None) if has_agent else None
        used, context_length, model_name = await self._resolve_context_figures(
            agent if has_agent else None, ctx, session_entry, source
        )

        # Gauge path: real current-context figure
        if used > 0 and context_length > 0:
            pct = min(100.0, used / context_length * 100)
            headroom = max(0, context_length - used)
            BAR_WIDTH = 24
            filled = int(round(pct / 100 * BAR_WIDTH))
            bar = "█" * max(0, filled) + "░" * max(0, BAR_WIDTH - filled)

            lines = [
                t("gateway.context.header"),
                "",
                t("gateway.context.model", model=model_name or "?"),
                t("gateway.context.window", total=f"{context_length:,}"),
                t(
                    "gateway.context.in_use",
                    used=f"{used:,}",
                    total=f"{context_length:,}",
                    pct=f"{pct:.0f}",
                ),
                t("gateway.context.bar", bar=bar),
                t("gateway.context.headroom", headroom=f"{headroom:,}"),
                "",
            ]
            # Full view — compression / throughput need the live agent.
            if ctx is not None:
                lines.extend(_context_compressor_lines(agent, ctx, used))
            else:
                lines.append(t("gateway.context.detail_after_first"))

            # Per-category estimated breakdown (+ optional expanded listings). Same chars/4 engine
            # the desktop popover and /usage use; plain text (no glyph grid — monospace isn't
            # guaranteed on messaging platforms). Fail-open: rendering errors never break /context.
            if has_agent:
                breakdown = await asyncio.to_thread(
                    self._context_breakdown_block, agent, source, expanded
                )
                if breakdown:
                    lines.append("")
                    lines.extend(breakdown)

            return "\n".join(lines)

        # Last resort: rough estimate from transcript
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough

            msgs = [
                m
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            approx = estimate_messages_tokens_rough(msgs)
            return "\n".join(
                [
                    t("gateway.context.header"),
                    "",
                    t(
                        "gateway.context.estimated",
                        count=f"{approx:,}",
                        messages=len(msgs),
                    ),
                    t("gateway.context.detail_after_first"),
                ]
            )
        return t("gateway.context.no_data")

    async def _resolve_context_figures(self, agent, ctx, session_entry, source):
        """``(used, context_length, model_name)`` for /context with cascading fallbacks.

        used  : compressor.last_prompt_tokens -> SessionStore.last_prompt_tokens
        model : agent.model -> SessionDB row model
        window: compressor.context_length -> effective gateway model route -> model metadata
        """
        used = context_length = 0
        if ctx is not None:
            used = getattr(ctx, "last_prompt_tokens", 0) or 0
            context_length = getattr(ctx, "context_length", 0) or 0
        model_name = _clean_str(getattr(agent, "model", "")) if agent is not None else ""
        if not used:
            used = _int_value(getattr(session_entry, "last_prompt_tokens", 0))
        if not model_name and self._session_db:
            try:
                row = await self._session_db.get_session(session_entry.session_id) or {}
                if isinstance(row, dict):
                    model_name = _clean_str(row.get("model", ""))
            except Exception:
                model_name = ""
        if not context_length:
            try:
                from gateway.run import _profile_runtime_scope, _resolve_gateway_model_context

                def _resolve_nonresident_context():
                    if getattr(getattr(self, "config", None), "multiplex_profiles", False):
                        profile_home = self._resolve_profile_home_for_source(source)
                        with _profile_runtime_scope(profile_home):
                            return _resolve_gateway_model_context(model_name or None)
                    return _resolve_gateway_model_context(model_name or None)

                resolved = await asyncio.to_thread(_resolve_nonresident_context)
                model_name = model_name or resolved.model
                context_length = _int_value(resolved.context_length)
            except Exception:
                context_length = 0
        if not context_length and model_name:
            try:
                from agent.model_metadata import get_model_context_length

                context_length = _int_value(await asyncio.to_thread(get_model_context_length, model_name))
            except Exception:
                context_length = 0
        return used, context_length, model_name

    def _gateway_session_origin_for_id(self, session_id: str) -> Optional[SessionSource]:
        """Best-effort origin lookup for gateway session IDs."""
        lookup = getattr(type(self.session_store), "lookup_by_session_id", None)
        if callable(lookup):
            entry = lookup(self.session_store, session_id)
            return getattr(entry, "origin", None) if entry is not None else None

        # Test doubles and older stores may not expose the public lookup helper.
        # Keep the Matrix resume guard fail-closed if no origin can be resolved.
        entries = getattr(self.session_store, "_entries", {}) or {}
        for entry in entries.values():
            if getattr(entry, "session_id", None) == session_id:
                return getattr(entry, "origin", None)
        return None

    @staticmethod
    def _same_matrix_room(current: SessionSource, origin: Optional[SessionSource]) -> bool:
        return (
            origin is not None
            and origin.platform == Platform.MATRIX
            and current.platform == Platform.MATRIX
            and origin.chat_id == current.chat_id
            # thread_id is part of the session key (build_session_key appends it for every chat
            # type when present) and Matrix scopes a turn to the current room/thread, so a live
            # session in another thread of the SAME room is a DIFFERENT session: thread A must not
            # resume/enumerate a target from thread B. Non-threaded rooms compare "" == "" unchanged.
            and str(getattr(current, "thread_id", "") or "")
            == str(getattr(origin, "thread_id", "") or "")
        )

    def _same_origin_chat(self, current: SessionSource, origin: Optional[SessionSource]) -> bool:
        """Platform-agnostic counterpart to ``_same_matrix_room``.

        Per-participant sessions (``build_session_key`` with the default ``group_sessions_per_user``)
        must be participant-scoped here too, else a co-member could resume another member's live
        session (IDOR). Only an explicitly shared group/thread (``is_shared_multi_user_session``) shares.
        """
        if origin is None or current is None:
            return False
        if origin.platform != current.platform:
            return False
        if origin.chat_id != current.chat_id:
            return False
        # thread_id is part of the session key for every chat type (build_session_key appends it
        # unconditionally), so threads of the same parent chat are DIFFERENT sessions.
        # is_shared_multi_user_session only decides sharing WITHIN a thread — require thread equality
        # before any sharing logic so a live origin in thread A cannot match a caller in thread B.
        if str(getattr(current, "thread_id", "") or "") != str(
            getattr(origin, "thread_id", "") or ""
        ):
            return False
        chat_type = (getattr(current, "chat_type", "") or "").lower()
        # DM-like chats are always per-user.
        if chat_type in {"dm", "direct", "private", ""}:
            # chat_id was already required equal above and, when present, IS the DM session key, so
            # an equal non-empty chat_id suffices. build_session_key falls back to the participant
            # (``user_id_alt or user_id`` — Signal/Feishu key on user_id_alt) only when there is NO
            # chat_id; mirror that and fail closed on a missing/different participant so two
            # no-chat_id DM origins are never conflated.
            if str(getattr(current, "chat_id", "") or ""):
                return True
            cur_pid = str(current.user_id_alt or current.user_id or "")
            org_pid = str(origin.user_id_alt or origin.user_id or "")
            return bool(cur_pid) and cur_pid == org_pid
        # Non-DM: scope by participant whenever the session key for this source
        # is per-user. is_shared_multi_user_session mirrors build_session_key's
        # isolation rules exactly, so the guard stays in lock-step with the key.
        shared = is_shared_multi_user_session(
            current,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )
        if shared:
            return True
        # Per-user key: compare the participant id the key is actually built
        # from (user_id_alt or user_id — Signal/Feishu key on user_id_alt).
        cur_pid = current.user_id_alt or current.user_id
        org_pid = origin.user_id_alt or origin.user_id
        if cur_pid and org_pid:
            return cur_pid == org_pid
        # Per-user key but a participant id is missing on one side: cannot prove
        # the same owner — fail closed.
        return False

    def _resume_caller_is_admin(self, source: SessionSource) -> bool:
        """Whether *source* is an EXPLICITLY-configured admin allowed cross-origin /resume or /sessions.

        Stricter than ``SlashAccessPolicy.is_admin()``, which returns True for every allowed caller
        when slash gating is DISABLED; cross-origin DATA ACCESS needs a real configured admin, else
        the default (no admin list) config would make every caller cross-origin-capable (IDOR).
        """
        try:
            from gateway.slash_access import policy_for_source
            policy = policy_for_source(self.config, source)
            uid = getattr(source, "user_id", None)
            return bool(policy.enabled and uid and policy.is_admin(uid))
        except Exception:
            return False

    async def _resume_target_allowed(
        self, source: SessionSource, target_id: str, allow_override: bool = False
    ) -> bool:
        """Whether *source* may resume the persisted session *target_id*.

        Generalizes the Matrix-only room guard to every adapter so a caller cannot bind to another
        user's/room's session (IDOR). Uses the live origin when the target is active, else the DB
        row's source + user_id; the row must PROVE ownership or fail closed. Admin ``--all`` bypasses.
        """
        if allow_override and self._resume_caller_is_admin(source):
            return True
        # Use the live origin only when it resolves to a real SessionSource; a
        # store that can't resolve it (or an unexpected lookup error) must not
        # silently allow/deny — fall through to the deterministic DB scoping.
        try:
            origin = self._gateway_session_origin_for_id(target_id)
        except Exception:
            origin = None
        if isinstance(origin, SessionSource):
            return self._same_origin_chat(source, origin)
        # Inactive/persisted-only: best-effort scope by DB row source + user.
        try:
            row = await self._session_db.get_session(target_id) or {}
        except Exception:
            return False
        caller_src = source.platform.value if source.platform else None
        row_src = row.get("source")
        if row_src and caller_src and str(row_src) != str(caller_src):
            return False  # different platform / source
        caller_uid = str(getattr(source, "user_id", "") or "")
        row_uid = str(row.get("user_id") or "")
        # Chat/thread origin recorded at session creation. Rows once stored only source + user_id,
        # so a same-user row could belong to a DIFFERENT chat; comparing the persisted origin closes
        # that gap. Legacy rows (NULL) fail closed — resume via a live session or an admin override.
        caller_chat = str(getattr(source, "chat_id", "") or "")
        row_chat = str(row.get("chat_id") or "")
        caller_thread = str(getattr(source, "thread_id", "") or "")
        row_thread = str(row.get("thread_id") or "")
        chat_type = (getattr(source, "chat_type", "") or "").lower()
        caller_is_dm = chat_type in {"dm", "direct", "private", ""}
        # build_session_key keys the participant on ``user_id_alt or user_id``, but the sessions table
        # has no user_id_alt column, so a row cannot prove the canonical participant for an alt-keyed
        # (Signal/Feishu) caller: per-user row_uid == caller_uid checks must fail closed (CWE-639).
        caller_keys_on_alt = bool(str(getattr(source, "user_id_alt", "") or ""))
        if caller_uid:
            # Identity-bearing caller: the row must PROVE the same owner AND platform AND chat/thread.
            # A blank/legacy source can't prove the platform (row_src above only rejects a *mismatching*
            # non-blank one); a different thread is a different session. Any gap fails closed.
            origin_ok = (
                bool(row_src) and bool(caller_src)
                and str(row_src) == str(caller_src)
                and row_thread == caller_thread
            )
            if not origin_ok:
                return False
            if caller_is_dm:
                # DMs are keyed on user_id; require the same owner. A no-chat_id DM is keyed PURELY on
                # the participant (so an alt-keyed caller fails closed); when both sides carry chat_id,
                # equality is the DM key and suffices, and a mismatching chat_id is rejected.
                if caller_keys_on_alt and not (bool(row_chat) and bool(caller_chat)):
                    return False
                return (
                    bool(row_uid) and row_uid == caller_uid
                    and row_chat == caller_chat
                )
            # Non-DM (group/channel/forum/thread): build_session_key includes chat_id, so a row (or
            # caller) with NO chat provenance cannot prove same-chat. Require both non-blank and
            # equal — a legacy NULL-chat row fails closed even when both normalize to "". (CWE-639)
            if not (bool(row_chat) and bool(caller_chat) and row_chat == caller_chat):
                return False
            # Same non-DM chat/thread: mirror build_session_key's participant scoping. A SHARED
            # group/thread session (group_sessions_per_user=False, or a shared thread) is one session
            # for every participant, so the same-chat proof suffices — do NOT also require user-id
            # equality (it would block co-members). A per-user session still requires the same owner.
            shared = is_shared_multi_user_session(
                source,
                group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
                thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
            )
            if shared:
                return True
            # Per-user non-DM: the session key includes the participant (``user_id_alt or
            # user_id``). If the caller keys on user_id_alt, the persisted row (user_id only) cannot
            # prove the canonical participant, so fail closed rather than matching on user_id alone.
            if caller_keys_on_alt:
                return False
            return bool(row_uid) and row_uid == caller_uid
        # No caller identity: the row carries only source + user_id, so a same-platform row can belong
        # to a DIFFERENT chat or user — same platform alone is NOT ownership proof; fail closed
        # (CWE-639). Same-chat resume of an ACTIVE session still works via the live-origin branch.
        return False

    async def _resume_row_visible(
        self, source: SessionSource, row: dict, allow_all: bool
    ) -> bool:
        """Whether a titled-session listing *row* belongs to the caller's origin.

        Prevents cross-origin enumeration of session ids/previews via the numbered /resume list;
        keeps Matrix room-scoping, scopes every other platform to the caller unless admin ``--all``.
        """
        sid = str(row.get("id") or "")
        if source.platform == Platform.MATRIX:
            # Cross-room enumeration is cross-ORIGIN data access: gate the ``--all`` short-circuit
            # behind a real configured admin, exactly like the non-Matrix branch below.
            if allow_all and self._resume_caller_is_admin(source):
                return True
            return self._same_matrix_room(source, self._gateway_session_origin_for_id(sid))
        if allow_all and self._resume_caller_is_admin(source):
            return True
        return await self._resume_target_allowed(source, sid, allow_override=False)

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents command - list active agents and running tasks."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        from tools.process_registry import format_uptime_short, process_registry

        now = time.time()
        current_session_key = self._session_key_for_source(event.source)

        running_agents: dict = getattr(self, "_running_agents", {}) or {}
        running_started: dict = getattr(self, "_running_agents_ts", {}) or {}

        agent_rows: list[dict] = []
        for session_key, agent in running_agents.items():
            started = float(running_started.get(session_key, now))
            elapsed = max(0, int(now - started))
            is_pending = agent is _AGENT_PENDING_SENTINEL
            agent_rows.append(
                {
                    "session_key": session_key,
                    "elapsed": elapsed,
                    "state": t("gateway.agents.state_starting") if is_pending else t("gateway.agents.state_running"),
                    "session_id": "" if is_pending else str(getattr(agent, "session_id", "") or ""),
                    "model": "" if is_pending else str(getattr(agent, "model", "") or ""),
                }
            )

        agent_rows.sort(key=lambda row: row["elapsed"], reverse=True)

        running_processes: list[dict] = []
        try:
            running_processes = [
                p for p in process_registry.list_sessions()
                if p.get("status") == "running"
            ]
        except Exception:
            running_processes = []

        background_tasks = [
            t for t in (getattr(self, "_background_tasks", set()) or set())
            if hasattr(t, "done") and not t.done()
        ]

        lines = [
            t("gateway.agents.header"),
            "",
            t("gateway.agents.active_agents", count=len(agent_rows)),
        ]

        if agent_rows:
            for idx, row in enumerate(agent_rows[:12], 1):
                current = t("gateway.agents.this_chat") if row["session_key"] == current_session_key else ""
                sid = f" · `{row['session_id']}`" if row["session_id"] else ""
                model = f" · `{row['model']}`" if row["model"] else ""
                lines.append(
                    f"{idx}. `{row['session_key']}` · {row['state']} · "
                    f"{format_uptime_short(row['elapsed'])}{sid}{model}{current}"
                )
            if len(agent_rows) > 12:
                lines.append(t("gateway.agents.more", count=len(agent_rows) - 12))

        lines.extend(
            [
                "",
                t("gateway.agents.running_processes", count=len(running_processes)),
            ]
        )
        if running_processes:
            for proc in running_processes[:12]:
                cmd = " ".join(str(proc.get("command", "")).split())
                if len(cmd) > 90:
                    cmd = cmd[:87] + "..."
                lines.append(
                    f"- `{proc.get('session_id', '?')}` · "
                    f"{format_uptime_short(int(proc.get('uptime_seconds', 0)))} · `{cmd}`"
                )
            if len(running_processes) > 12:
                lines.append(t("gateway.agents.more", count=len(running_processes) - 12))

        lines.extend(
            [
                "",
                t("gateway.agents.async_jobs", count=len(background_tasks)),
            ]
        )

        # Background (async) delegations — delegate_task(background=true).
        try:
            from tools.async_delegation import list_async_delegations
            delegations = [
                d for d in list_async_delegations()
                if d.get("status") in ("running", "stalling", "finalizing")
            ]
        except Exception:
            delegations = []
        if delegations:
            lines.extend(["", t("gateway.agents.background_delegations", count=len(delegations))])
            for d in delegations[:12]:
                lines.extend(_agents_delegation_lines(d))
            if len(delegations) > 12:
                lines.append(t("gateway.agents.more", count=len(delegations) - 12))

        if (
            not agent_rows
            and not running_processes
            and not background_tasks
            and not delegations
        ):
            lines.append("")
            lines.append(t("gateway.agents.none"))

        return "\n".join(lines)

    async def _handle_stop_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /stop command - interrupt a running agent.

        When an agent is truly hung (blocked thread that never checks _interrupt_requested), the
        early intercept in _handle_message() handles /stop before this method is reached; this
        handler fires only via normal dispatch (no running agent) or as a fallback, and force-cleans
        the session lock in all cases. The session is preserved so the user can continue.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_key = session_entry.session_key

        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            # Force-clean the sentinel so the session is unlocked.
            await self._interrupt_and_clear_session(
                session_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command_pending",
            )
            logger.info("STOP (pending) for session %s — sentinel cleared", session_key)
            return EphemeralReply(t("gateway.stop.stopped_pending"))
        if agent:
            # Force-clean the session lock so a truly hung agent doesn't
            # keep it locked forever.
            await self._interrupt_and_clear_session(
                session_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command_handler",
            )
            return EphemeralReply(t("gateway.stop.stopped"))

        # No run under the caller's own key. In a per-user thread (thread_sessions_per_user=True) a
        # run another user started lives under a different key, yet authorized users must still be
        # able to /stop it: fall back to sibling runs in this thread, gated on authorization.
        sibling_keys = self._sibling_thread_run_keys(source, session_key)
        if sibling_keys and self._is_user_authorized(source):
            for sibling_key in sibling_keys:
                await self._interrupt_and_clear_session(
                    sibling_key,
                    source,
                    interrupt_reason=_INTERRUPT_REASON_STOP,
                    invalidation_reason="stop_command_thread_sibling",
                )
            logger.info(
                "STOP (thread sibling) by %s — interrupted %d run(s) in thread: %s",
                session_key,
                len(sibling_keys),
                ", ".join(sibling_keys),
            )
            return EphemeralReply(t("gateway.stop.stopped"))

        # No running agent anywhere for this scope. A platform status indicator can still be stuck —
        # e.g. Slack's persistent assistant.threads.setStatus survives a gateway restart or a turn
        # that died without a final send.
        adapter = getattr(self, "adapters", {}).get(source.platform)
        if adapter and hasattr(adapter, "_stop_typing_with_metadata"):
            try:
                await adapter._stop_typing_with_metadata(source.chat_id, self._reply_metadata(event))
            except Exception:
                logger.debug(
                    "Failed to clear typing on /stop with no active agent",
                    exc_info=True,
                )

        return t("gateway.stop.no_active")

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — inspect and manually control failed/paused
        gateway adapters (pause stops the reconnect watcher; resume re-queues for retry).
        """
        text = (getattr(event, "content", "") or "").strip()
        # Strip the leading "/platform" (or "/PLATFORM") token if present
        parts = text.split(maxsplit=2)
        if parts and parts[0].lower().lstrip("/").startswith("platform"):
            parts = parts[1:]
        action = (parts[0] if parts else "list").lower()
        target = parts[1].lower() if len(parts) > 1 else ""

        # Resolve platform name (case-insensitive, value match)
        def _resolve_platform(name: str):
            if not name:
                return None
            for p in Platform.__members__.values():
                if p.value.lower() == name:
                    return p
            return None

        if action == "list":
            lines = ["**Gateway platforms**"]
            connected = sorted(p.value for p in self.adapters)
            if connected:
                lines.append("Connected: " + ", ".join(connected))
            else:
                lines.append("Connected: (none)")
            failed = getattr(self, "_failed_platforms", {}) or {}
            if failed:
                for p, info in failed.items():
                    if info.get("paused"):
                        reason = info.get("pause_reason") or "paused"
                        lines.append(
                            f"  · {p.value} — PAUSED ({reason}). "
                            f"Resume with `/platform resume {p.value}`."
                        )
                    else:
                        attempts = info.get("attempts", 0)
                        lines.append(
                            f"  · {p.value} — retrying (attempt {attempts})"
                        )
            else:
                lines.append("Failed/paused: (none)")
            return "\n".join(lines)

        if action in {"pause", "resume"}:
            if not target:
                return f"Usage: /platform {action} <name>"
            platform = _resolve_platform(target)
            if platform is None:
                return f"Unknown platform: {target}"
            failed = getattr(self, "_failed_platforms", {}) or {}
            if action == "pause":
                if platform not in failed:
                    return (
                        f"{platform.value} is not in the retry queue "
                        f"(it's either connected or not enabled)."
                    )
                if failed[platform].get("paused"):
                    return f"{platform.value} is already paused."
                self._pause_failed_platform(platform, reason="paused via /platform pause")
                return (
                    f"✓ {platform.value} paused. "
                    f"Resume with `/platform resume {platform.value}` or "
                    f"`hermes gateway restart` to reset."
                )
            # action == "resume"
            if platform not in failed:
                return (
                    f"{platform.value} is not in the retry queue — "
                    f"nothing to resume."
                )
            if not failed[platform].get("paused"):
                return (
                    f"{platform.value} is already retrying — "
                    f"no resume needed."
                )
            self._resume_paused_platform(platform)
            return f"✓ {platform.value} resumed — retrying on next watcher tick."

        return (
            "Usage: /platform <list|pause|resume> [name]\n"
            "  /platform list — show platform status\n"
            "  /platform pause <name> — stop retrying a failing platform\n"
            "  /platform resume <name> — re-queue a paused platform"
        )

    async def _handle_restart_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /restart command - drain active work, then restart the gateway."""
        from gateway.run import _hermes_home
        # Idempotency check: if the previous gateway process recorded this same /restart (platform +
        # update_id) and we see it *again*, it's a redelivery from PTB's graceful-shutdown get_updates
        # ACK failing on the way out. Ignoring it prevents a loop where every fresh gateway re-restarts.
        if self._is_stale_restart_redelivery(event):
            logger.info(
                "Ignoring redelivered /restart (platform=%s, update_id=%s) — "
                "already processed by a previous gateway instance.",
                event.source.platform.value if event.source and event.source.platform else "?",
                event.platform_update_id,
            )
            return ""

        if self._restart_requested or self._draining:
            count = self._running_agent_count()
            if count:
                return t("gateway.draining", count=count)
            return EphemeralReply(t("gateway.restart.in_progress"))

        # Save the requester's routing info so the new gateway process can
        # notify them once it comes back online.
        try:
            notify_data = {
                "platform": event.source.platform.value if event.source.platform else None,
                "chat_id": event.source.chat_id,
                "chat_type": event.source.chat_type,
            }
            if event.source.delivered_via_upstream_relay is True:
                notify_data["delivered_via_upstream_relay"] = True
                if event.source.user_id:
                    notify_data["user_id"] = event.source.user_id
                if event.source.scope_id:
                    notify_data["scope_id"] = event.source.scope_id
            if event.source.thread_id:
                notify_data["thread_id"] = event.source.thread_id
            if event.message_id:
                notify_data["message_id"] = event.message_id
            if event.source is not None:
                try:
                    self._restart_command_source = dataclasses.replace(
                        event.source,
                        message_id=str(event.message_id)
                        if event.message_id is not None
                        else event.source.message_id,
                    )
                except Exception:
                    self._restart_command_source = event.source
            await asyncio.to_thread(
                atomic_json_write,
                _hermes_home / ".restart_notify.json",
                notify_data,
                indent=None,
            )
        except Exception as e:
            logger.debug("Failed to write restart notify file: %s", e)

        # Record the triggering platform + update_id in a dedicated dedup marker. Unlike
        # .restart_notify.json (unlinked once the new gateway sends its notification) this persists
        # so a delayed Telegram redelivery is still detectable. Overwritten on every /restart.
        try:
            dedup_data = {
                "platform": event.source.platform.value if event.source.platform else None,
                "requested_at": time.time(),
            }
            if event.platform_update_id is not None:
                dedup_data["update_id"] = event.platform_update_id
            await asyncio.to_thread(
                atomic_json_write,
                _hermes_home / ".restart_last_processed.json",
                dedup_data,
                indent=None,
            )
        except Exception as e:
            logger.debug("Failed to write restart dedup marker: %s", e)

        active_agents = self._running_agent_count()
        # Under a service manager (systemd/launchd) or Docker/Podman, exit 75 so the supervisor /
        # restart policy restarts us — detached setsid+bash fails there (systemd KillMode=mixed kills
        # the cgroup; tini exits with the gateway). The explicit marker covers ``sudo env -i`` wrappers.
        from gateway.restart import (
            is_container_restart_context,
            is_gateway_supervisor_process,
        )

        _under_service = is_gateway_supervisor_process()
        _in_container = is_container_restart_context()
        if _under_service or _in_container:
            self.request_restart(detached=False, via_service=True)
        else:
            self.request_restart(detached=True, via_service=False)
        if active_agents:
            return t("gateway.draining", count=active_agents)
        return EphemeralReply(t("gateway.restart.restarting"))

    async def _handle_version_command(self, event: MessageEvent) -> str:
        """Handle /version — show the running Hermes Agent version."""
        from hermes_cli.slash_exec import CommandContext, execute_command

        return execute_command("version", CommandContext(surface="gateway")).text

    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("help", CommandContext(surface="gateway"))
        return self._telegramized_command_reply(event, reply.text)

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        from hermes_cli.slash_exec import CommandContext, execute_command

        # Page size is a surface parameter (Telegram messages are shorter).
        page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
        reply = execute_command(
            "commands",
            CommandContext(
                surface="gateway",
                args=event.get_command_args(),
                options={"page_size": page_size},
            ),
        )
        return self._telegramized_command_reply(event, reply.text)

    async def _perform_model_switch(
        self,
        switch_model,
        *,
        raw_input: str,
        explicit_provider,
        session_key: str,
        source,
        current_model,
        current_provider,
        current_base_url,
        current_api_key,
        persist_global: bool,
        user_provs,
        custom_provs,
    ):
        """Resolve a /model switch off-loop. Returns ``(result, None)`` or ``(None, error_text)``."""
        from gateway.run import _load_gateway_config

        skew_error = _model_switch_skew_guard()
        if skew_error:
            return None, skew_error
        # Offload the switch off the event loop — switch_model() can fall through to a synchronous
        # models.dev HTTP fetch (requests.get, 15s timeout) on a cold/expired cache, which freezes
        # the gateway otherwise.
        result = await asyncio.to_thread(
            switch_model,
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key=current_api_key,
            is_global=persist_global,
            explicit_provider=explicit_provider,
            user_providers=user_provs,
            custom_providers=custom_provs,
        )
        if not result.success:
            return None, t("gateway.model.error_prefix", error=result.error_message)
        try:
            from hermes_cli.context_switch_guard import enrich_model_switch_warnings_for_gateway

            # Offload: merge_preflight_compression_warning() calls the sync
            # resolve_display_context_length() provider probe ladder — must not run on the loop.
            await asyncio.to_thread(
                enrich_model_switch_warnings_for_gateway,
                result,
                self,
                session_key=session_key,
                source=source,
                custom_providers=custom_provs,
                load_gateway_config=_load_gateway_config,
            )
        except Exception as exc:
            logger.debug("preflight-compression switch warning failed: %s", exc)
        return result, None

    async def _commit_model_switch(
        self,
        result,
        *,
        session_key: str,
        source,
        current_model,
        current_base_url,
        current_api_key,
        custom_provs,
        persist_global: bool,
        config_path,
        one_turn: bool = False,
        restore_snapshot=None,
        picker: bool = False,
    ) -> str:
        """Apply a resolved switch (cached agent, session, config) and build the confirmation.

        Shared by the typed ``/model <name>`` path and the picker callback (``picker=True``).
        """
        from gateway.run import _load_gateway_config
        from hermes_cli.model_switch import format_model_for_display, resolve_display_context_length_async

        # If there's a cached agent, update it in-place
        cached_agent = self._cached_agent_for(session_key)
        if cached_agent is not None:
            try:
                cached_agent.switch_model(
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
                    capabilities=getattr(result, "runtime_capabilities", None),
                )
            except Exception as exc:
                # In-place swap rolled back to the OLD working model/client and re-raised. Abort the
                # commit (DB persist, session override, cache eviction, config write) so a failed switch
                # is a no-op — otherwise the next message rebuilds a broken agent from the override.
                logger.warning(
                    "%s model switch failed for cached agent: %s", "Picker" if picker else "In-place", exc
                )
                return t(
                    "gateway.model.error_prefix",
                    error=f"Model switch to {result.new_model} failed ({exc}); staying on {current_model}.",
                )

        # Persist the new model to the session DB so the dashboard shows the updated model.
        _sess_db = getattr(self, "_session_db", None)
        if _sess_db is not None:
            try:
                _sess_entry = await self.async_session_store.get_or_create_session(source)
                # Typed path: if this session was auto-reset, consume the flag so the next regular
                # message's cleanup does not wipe the model override just stored below.
                if not picker and getattr(_sess_entry, "was_auto_reset", False):
                    _sess_entry.was_auto_reset = False
                await _sess_db.update_session_model(
                    _sess_entry.session_id, result.new_model,
                    provider=result.target_provider,
                )
            except Exception as exc:
                logger.debug("Failed to persist model switch to DB: %s", exc)

        # Store a note to prepend to the next user message so the model knows about the switch
        # (avoids system messages mid-history). Display form strips opaque Palantir RID
        # prefixes; the override map below keeps the full ID for the wire.
        if not hasattr(self, "_pending_model_notes"):
            self._pending_model_notes = {}
        self._pending_model_notes[session_key] = (
            f"[Note: model was just switched from {format_model_for_display(current_model)} to "
            f"{format_model_for_display(result.new_model)} "
            f"via {result.provider_label or result.target_provider}. "
            f"{'This override applies to the next turn only. ' if one_turn else ''}"
            f"Adjust your self-identification accordingly.]"
        )

        # Store session override so next agent creation uses the new model
        self._session_model_overrides[session_key] = {
            "model": result.new_model,
            "provider": result.target_provider,
            "api_key": result.api_key,
            "base_url": result.base_url,
            "api_mode": result.api_mode,
            "request_overrides": dict(result.request_overrides or {}),
            "capabilities": dict(result.runtime_capabilities or {}),
        }
        if one_turn:
            if not hasattr(self, "_pending_one_turn_model_restores"):
                self._pending_one_turn_model_restores = {}
            self._pending_one_turn_model_restores[session_key] = (
                restore_snapshot or {"had_override": False, "override": None}
            )
        elif not picker and hasattr(self, "_pending_one_turn_model_restores"):
            self._pending_one_turn_model_restores.pop(session_key, None)

        # Write-through the non-secret parts (model/provider/base_url) so the override survives a
        # restart; api_key/api_mode are never persisted (re-resolved on rehydration). /model --once is
        # EXCLUDED: a one-turn override must not outlive a restart; the pre-once value stays persisted.
        if not one_turn:
            try:
                await self.async_session_store.set_model_override(
                    session_key, self._session_model_overrides[session_key]
                )
            except Exception:
                logger.debug("Failed to persist session model override", exc_info=True)

        # Evict cached agent so the next turn creates a fresh agent from the
        # override rather than relying on cache signature mismatch detection.
        self._evict_cached_agent(session_key)

        # Persist to config (default) unless --session opted out
        if persist_global:
            try:
                await _persist_model_switch_to_config(result, config_path)
            except Exception as e:
                logger.warning("Failed to persist model switch: %s", e)

        # Build confirmation message with full metadata. Display form shortens opaque Palantir
        # IDs (ri.language-model-service..*) to their trailing slug.
        provider_label = result.provider_label or result.target_provider
        lines = [t("gateway.model.switched", model=format_model_for_display(result.new_model))]
        lines.append(t("gateway.model.provider_label", provider=provider_label))

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry.
        mi = result.model_info
        _sw_config_ctx = None
        _sw_model_cfg = {}
        try:
            _sw_model_cfg = _load_gateway_config().get("model", {})
            if isinstance(_sw_model_cfg, dict):
                _sw_raw = _sw_model_cfg.get("context_length")
                if _sw_raw is not None:
                    _sw_config_ctx = int(_sw_raw)
        except Exception:
            pass
        if not isinstance(_sw_model_cfg, dict):
            _sw_model_cfg = {}
        ctx = await resolve_display_context_length_async(
            result.new_model,
            result.target_provider,
            base_url=result.base_url or current_base_url or "",
            api_key=result.api_key or current_api_key or "",
            model_info=mi,
            custom_providers=custom_provs,
            config_context_length=_sw_config_ctx,
            configured_model=_sw_model_cfg.get("default") or _sw_model_cfg.get("model"),
            configured_provider=_sw_model_cfg.get("provider"),
            configured_base_url=_sw_model_cfg.get("base_url"),
        )
        if ctx:
            lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
        if mi:
            if mi.max_output:
                lines.append(t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}"))
            lines.append(t("gateway.model.capabilities_label", capabilities=mi.format_capabilities()))

        if not picker:
            cache_enabled = (
                (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
                or result.api_mode == "anthropic_messages"
            )
            if cache_enabled:
                lines.append(t("gateway.model.prompt_caching_enabled"))

        if result.warning_message:
            lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))

        if persist_global:
            lines.append(t("gateway.model.saved_global"))
        elif one_turn:
            lines.append("    (next turn only — restores after one response)")
        else:
            lines.append(t("gateway.model.session_only_hint"))
        return "\n".join(lines)

    async def _send_model_picker(self, event: MessageEvent, source, adapter, session_key: str, listing_kwargs: dict, on_model_selected) -> bool:
        """Send the interactive /model picker; False when nothing was sent (text fallback).

        *source* is the session-key-normalized source (Telegram topic recovery), so the picker's
        thread metadata lands where the next turn reads.
        """
        from hermes_cli.model_switch import list_picker_providers

        try:
            # Offload blocking provider-listing (can fall through to a synchronous urllib HTTP fetch
            # on a stale cache) off the event loop so the gateway doesn't freeze. See #41289.
            providers = await asyncio.to_thread(
                list_picker_providers, max_models=50, include_moa=True, **listing_kwargs
            )
        except Exception:
            providers = []
        if not providers:
            return False
        result = await adapter.send_model_picker(
            chat_id=source.chat_id,
            providers=providers,
            current_model=listing_kwargs["current_model"],
            current_provider=listing_kwargs["current_provider"],
            session_key=session_key,
            on_model_selected=on_model_selected,
            metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
        )
        return bool(result.success)

    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /model command — switch model."""
        from gateway.run import _hermes_home, _load_gateway_config
        from hermes_cli.model_switch import (
            switch_model as _switch_model, parse_model_switch_args,
            resolve_persist_behavior,
            list_authenticated_providers,
        )
        from hermes_cli.providers import get_label

        raw_args = event.get_command_args().strip()
        source = event.source
        _command_profile_home = None
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            _command_profile_home = self._resolve_profile_home_for_source(source)

        # Parse --provider, --global, --session, --once, and --refresh flags
        # via the shared single-owner parser (hermes_cli.model_switch).
        request = parse_model_switch_args(raw_args)
        model_input = request.target
        explicit_provider = request.explicit_provider
        is_global_flag = request.is_global
        force_refresh = request.force_refresh
        is_session = request.is_session
        one_turn = request.is_once
        if request.errors:
            # Gateway decoration: "❌ " prefix over the canonical error copy.
            return f"❌ {request.error_messages()[0]}"
        persist_global = resolve_persist_behavior(
            is_global_flag,
            is_session,
            is_once=one_turn,
            explicit_provider=explicit_provider,
        )

        # --refresh: bust the disk cache so the picker shows live data.
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
            except Exception:
                pass

        # Read current model/provider from config
        config_path = (_command_profile_home or _hermes_home) / "config.yaml"
        current_model, current_provider, current_base_url, user_provs, custom_provs, excluded_provs = (
            _read_model_command_config(config_path)
        )
        current_api_key = ""

        # Check for session override. Normalize the source the same way a normal message turn does
        # (Telegram DM topic recovery) before deriving the override key, so the override is stored
        # under the key the next message turn reads.
        source = await asyncio.to_thread(self._normalize_source_for_session_key, source)
        session_key = self._session_key_for_source(source)
        override = self._session_model_overrides.get(session_key, {})
        restore_snapshot = (
            self._snapshot_session_model_override(session_key) if one_turn else None
        )
        if override:
            current_model = override.get("model", current_model)
            current_provider = override.get("provider", current_provider)
            current_base_url = override.get("base_url", current_base_url)
            current_api_key = override.get("api_key", current_api_key)

        async def perform_switch(model_id: str, provider_slug, *, src=source):
            return await self._perform_model_switch(
                _switch_model,
                raw_input=model_id,
                explicit_provider=provider_slug,
                session_key=session_key,
                source=src,
                current_model=current_model,
                current_provider=current_provider,
                current_base_url=current_base_url,
                current_api_key=current_api_key,
                persist_global=persist_global,
                user_provs=user_provs,
                custom_provs=custom_provs,
            )

        async def commit_switch(result, *, picker: bool = False, src=source) -> str:
            """Apply the resolved switch (agent, session, config) and build the reply."""
            return await self._commit_model_switch(
                result,
                session_key=session_key,
                source=src,
                current_model=current_model,
                current_base_url=current_base_url,
                current_api_key=current_api_key,
                custom_provs=custom_provs,
                persist_global=persist_global,
                config_path=config_path,
                one_turn=False if picker else one_turn,
                restore_snapshot=None if picker else restore_snapshot,
                picker=picker,
            )

        async def switch_and_commit(model_id: str, provider_slug, *, picker: bool) -> str:
            # The picker callback binds the raw event source (pre-normalization), as it always has.
            src = event.source if picker else source
            result, error = await perform_switch(model_id, provider_slug, src=src)
            if error is not None:
                return error
            return await commit_switch(result, picker=picker, src=src)

        # No args: show interactive picker (Telegram/Discord) or text list
        if not model_input and not explicit_provider:
            listing_kwargs = dict(
                current_provider=current_provider,
                current_base_url=current_base_url,
                current_model=current_model,
                user_providers=user_provs,
                custom_providers=custom_provs,
                excluded_providers=excluded_provs,
            )
            # Try interactive picker if the platform supports it
            adapter = self._adapter_for_source(source)
            if adapter is not None and getattr(type(adapter), "send_model_picker", None) is not None:
                async def _on_model_selected(_chat_id: str, model_id: str, provider_slug: str) -> str:
                    """Perform the model switch and return confirmation text."""
                    if _command_profile_home is None:
                        return await switch_and_commit(model_id, provider_slug, picker=True)
                    from gateway.run import _profile_runtime_scope

                    with _profile_runtime_scope(_command_profile_home):
                        return await switch_and_commit(model_id, provider_slug, picker=True)

                if await self._send_model_picker(event, source, adapter, session_key, listing_kwargs, _on_model_selected):
                    return None  # Picker sent — adapter handles the response

            # Fallback: text list (for platforms without picker or if picker failed)
            lines = [t("gateway.model.current_label", model=current_model or "unknown", provider=get_label(current_provider)), ""]
            try:
                # Offload blocking provider-listing off the event loop so the
                # gateway doesn't freeze on a stale-cache HTTP fetch. See #41289.
                providers = await asyncio.to_thread(list_authenticated_providers, max_models=5, **listing_kwargs)
                lines.extend(_model_provider_listing_lines(providers))
            except Exception:
                pass
            lines.append(t("gateway.model.usage_switch_model"))
            lines.append(t("gateway.model.usage_switch_provider"))
            lines.append(t("gateway.model.usage_persist"))
            return "\n".join(lines)

        # Perform the switch
        result, error = await perform_switch(model_input, explicit_provider)
        if error is not None:
            return error

        # Selection-guard confirmation for the typed /model <name> path (pickers confirm via their own
        # UI). Runs the unified registry (cost + data-policy guards); pricing lookups may hit
        # models.dev or a /models endpoint on a cache miss, so run it off the event loop.
        _cost_warning = None
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning

            _cost_warning = await asyncio.to_thread(
                combined_selection_warning,
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or current_base_url or "",
                api_key=result.api_key or current_api_key or "",
                model_info=result.model_info,
            )
        except Exception:
            _cost_warning = None
        if _cost_warning is not None:
            async def _on_cost_confirm(choice: str) -> str:
                if choice == "cancel":
                    return (
                        f"🟡 Model switch cancelled. Current model unchanged "
                        f"({current_model or 'unknown'})."
                    )
                # "once" and "always" both proceed — there is no persistent
                # opt-out for selection guards (each guarded switch should be
                # an explicit decision).
                return await commit_switch(result)

            _p = self._typed_command_prefix_for(event.source.platform)
            return await self._request_slash_confirm(
                event=event,
                command="model",
                title=_cost_warning.title,
                message=(
                    f"⚠️ **{_cost_warning.title}**\n\n{_cost_warning.message}\n\n"
                    f"_Text fallback: reply `{_p}approve` to switch or `{_p}cancel` to keep "
                    "the current model._"
                ),
                handler=_on_cost_confirm,
            )

        return await commit_switch(result)

    async def _handle_codex_runtime_command(self, event: MessageEvent) -> str:
        """Handle /codex-runtime command in the gateway.

        On change the cached agent is evicted so the next message builds a fresh AIAgent with the
        new api_mode (avoids prompt-cache invalidation mid-session).
        """
        from hermes_cli import codex_runtime_switch as crs

        raw_args = event.get_command_args().strip() if event else ""
        new_value, errors = crs.parse_args(raw_args)
        if errors:
            return "❌ " + "\n❌ ".join(errors)

        # Load + persist via the same helpers used for /model and /yolo
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            return f"❌ Could not load config: {exc}"
        cfg = load_config()

        result = crs.apply(
            cfg,
            new_value,
            persist_callback=(save_config if new_value is not None else None),
        )

        # On a real change, evict the cached agent so the new runtime takes
        # effect on the next message rather than waiting for cache TTL.
        if result.success and new_value is not None and result.requires_new_session:
            try:
                session_key = self._session_key_for_source(event.source)
                self._evict_cached_agent(session_key)
            except Exception:
                logger.debug("could not evict cached agent after codex-runtime change",
                             exc_info=True)

        prefix = "✓" if result.success else "✗"
        return f"{prefix} {result.message}"

    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality.

        All resolution/persistence goes through hermes_cli.personality, the single owner of state.
        """
        from gateway.run import _load_gateway_config
        from hermes_cli.personality import (
            active_personality_name,
            available_personalities,
            describe_personality,
            persist_personality,
            resolve_personality,
        )

        args = event.get_command_args().strip()

        try:
            config = _load_gateway_config()
        except Exception:
            config = {}
        personalities = available_personalities(config)

        if not args:
            current = active_personality_name(config)
            lines = [t("gateway.personality.header")]
            lines.append(t("gateway.personality.none_option"))
            for name, prompt in personalities.items():
                marker = " ✓" if name == current else ""
                lines.append(
                    t(
                        "gateway.personality.item",
                        name=f"{name}{marker}",
                        preview=describe_personality(prompt),
                    )
                )
            lines.append(t("gateway.personality.usage"))
            return "\n".join(lines)

        try:
            name, _new_prompt = resolve_personality(args, config)
        except ValueError:
            available = "`none`, " + ", ".join(f"`{n}`" for n in personalities)
            return t("gateway.personality.unknown", name=args.lower(), available=available)

        # Persist the selection only — hermes_cli.personality never writes agent.system_prompt (user-
        # owned overlay). persist_personality writes get_hermes_home()/config.yaml (the routed profile
        # under multiplex) and the next turn re-resolves the prompt from it: no process-global state.
        if not persist_personality(name):
            return t("gateway.personality.save_failed", error="config write failed")

        if not name:
            return t("gateway.personality.cleared")
        return t("gateway.personality.set_to", name=name)

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        # Find the last *real* user message. Timeline bookkeeping rows carry role=user +
        # display_kind (model_switch / async_delegation_complete / auto_continue / hidden); clients
        # never count them as user turns.
        last_user_idx = None
        # The canonical projection excludes bookkeeping and pure handoffs while
        # still recognizing a real ask embedded in a compaction carrier.
        from agent.context_compressor import (
            history_before_user_originated_turn,
            retryable_user_text,
            split_user_originated_turn,
            user_originated_turn_view,
        )

        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if user_originated_turn_view(msg) is not None:
                last_user_idx = i
                break

        if last_user_idx is None:
            return t("gateway.retry.no_previous")

        # Resolve the live text and the scaffold-preserving prefix before any
        # transcript write. Messaging retries cannot reconstruct attachments;
        # reject media/unknown content without truncating the session.
        try:
            truncated, live_view = history_before_user_originated_turn(
                history, last_user_idx
            )
            last_user_msg = retryable_user_text(live_view.get("content"))
            handoff, _ = split_user_originated_turn(history[last_user_idx])
        except ValueError as exc:
            return f"Cannot retry that message safely: {exc}"

        if handoff is not None:
            # A composite carrier is one physical row containing both the retained summary and the
            # live ask. Let the carrier-aware rewind archive that row/tail and insert its pure
            # scaffold atomically.
            try:
                rewind_result = await self.async_session_store.rewind_session(
                    session_entry.session_id,
                    1,
                    require_retryable_composite=True,
                )
            except ValueError as exc:
                return f"Cannot retry that message safely: {exc}"
            if rewind_result is None:
                return "Retry failed; transcript was not changed."
            # The store reselects and validates the latest carrier on the same
            # snapshot used by the atomic rewind.  A concurrent newer turn can
            # therefore never be removed while this handler resends stale text.
            last_user_msg = rewind_result["target_text"]
        else:
            # After in-place compaction the pre-compaction transcript lives on as
            # active=0/compacted=1 rows under this session id. active_only preserves that archive; a
            # separate existence probe could fail open or race with the write.
            if not await self.async_session_store.rewrite_transcript(
                session_entry.session_id,
                truncated,
                active_only=True,
                reject_active_turn_lease=True,
            ):
                return "Retry failed; transcript was not changed."
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0

        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
            channel_prompt=event.channel_prompt,
        )

        # Let the normal message handler process it
        return await self._handle_message(retry_event)

    async def _handle_goal_command(self, event: "MessageEvent") -> str:
        """Handle /goal for gateway platforms.

        Subcommands: status / pause / resume / clear. Setting a new goal queues the goal text as the
        next turn so the agent starts immediately; the post-turn continuation hook takes over after.
        """
        args = (event.get_command_args() or "").strip()
        lower = args.lower()

        mgr, session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")

        if not args or lower == "status":
            return mgr.status_line()
        if lower == "show":
            return f"{mgr.status_line()}\n{mgr.render_contract()}"
        if lower == "unwait":
            return "▶ Wait barrier cleared — goal loop resumes." if mgr.stop_waiting() else "No wait barrier set."
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            self._clear_goal_continuations(event, "clear")
            return t("gateway.goal_cleared") if had else t("gateway.no_active_goal")
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            if state is None:
                return t("gateway.goal.no_goal_set")
            self._clear_goal_continuations(event, "pause")
            return t("gateway.goal.paused", goal=state.goal)
        if lower == "resume":
            return self._goal_resume(mgr, event)
        # Verb-prefixed forms take the remainder as their argument.
        for verb, handler in (("wait ", self._goal_wait), ("gate ", self._goal_gate)):
            if lower == verb.strip() or lower.startswith(verb):
                return handler(mgr, args[len(verb) - 1:].strip(), event)
        return await self._goal_set(mgr, args, lower, event)

    def _clear_goal_continuations(self, event: MessageEvent, verb: str) -> None:
        try:
            adapter, _quick_key = self._adapter_and_key_for(event)
            if adapter and _quick_key:
                self._clear_goal_pending_continuations(_quick_key, adapter)
        except Exception as exc:
            logger.debug("goal %s: pending continuation cleanup failed: %s", verb, exc)

    def _goal_resume(self, mgr, event: MessageEvent) -> str:
        state = mgr.resume()
        if state is None:
            return t("gateway.goal.no_resume")
        # Resume must restart work, not just flip persisted state: enqueue the canonical
        # continuation through the adapter FIFO — the same path the post-turn judge uses — so
        # the next turn fires as soon as this reply is delivered.
        prompt = mgr.next_continuation_prompt()
        try:
            adapter, _quick_key = self._adapter_and_key_for(event)
            if prompt and adapter and _quick_key:
                cont_event = MessageEvent(
                    text=prompt,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=None,
                    channel_prompt=None,
                )
                self._enqueue_fifo(_quick_key, cont_event, adapter)
        except Exception as exc:
            logger.debug("goal resume: continuation enqueue failed: %s", exc)
        return t("gateway.goal.resumed", goal=state.goal)

    @staticmethod
    def _goal_wait(mgr, wait_arg: str, event: MessageEvent) -> str:
        """/goal wait <pid> [reason] — park the loop on a background process."""
        if not wait_arg:
            return "Usage: /goal wait <pid> [reason]"
        wtokens = wait_arg.split(None, 1)
        try:
            pid = int(wtokens[0])
        except ValueError:
            return "/goal wait: <pid> must be an integer process id."
        reason = wtokens[1].strip() if len(wtokens) > 1 else ""
        try:
            mgr.wait_on(pid, reason=reason)
        except (RuntimeError, ValueError) as exc:
            return f"/goal wait: {exc}"
        rtxt = f" ({reason})" if reason else ""
        return f"⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits."

    def _goal_gate(self, mgr, gate_arg: str, event: MessageEvent) -> str:
        """/goal gate [list | add <command> | remove <N> | clear] — deterministic quality gates."""
        gate_lower = gate_arg.lower()
        if not gate_arg or gate_lower == "list":
            return mgr.render_gates()
        if gate_lower.startswith("add "):
            # SECURITY: a gate is persisted and later executed with shell=True at every goal turn
            # boundary (run_gate), with no approval prompt. Letting an allowed but non-admin gateway
            # sender choose that string is authenticated RCE under the Hermes process account — and
            # with no admin list configured (the backward-compatible default) every allowed sender
            # is treated as unrestricted. Gate ONLY this shell-creating operation behind a real,
            # explicitly-configured admin (the same fail-closed check that guards cross-origin
            # /resume); list/remove/clear stay open so a non-admin can still recover.
            if not self._resume_caller_is_admin(event.source):
                return (
                    "⛔ /goal gate add requires an explicitly configured "
                    "gateway admin (allow_admin_from for DMs, "
                    "group_allow_admin_from for groups)."
                )
            try:
                gate = mgr.add_gate(gate_arg[len("add"):].strip())
            except (RuntimeError, ValueError) as exc:
                return f"/goal gate add: {exc}"
            return (
                f"⚿ Gate added: $ {gate.command} "
                f"({gate.max_retries} retries, {gate.timeout_seconds}s timeout). "
                f"It must pass before the goal can complete."
            )
        if gate_lower.startswith("remove ") or gate_lower.startswith("rm "):
            try:
                removed = mgr.remove_gate(int(gate_arg.split(None, 1)[1].strip()))
            except (RuntimeError, ValueError, IndexError) as exc:
                return f"/goal gate remove: {exc}"
            return f"✓ Gate removed: $ {removed}"
        if gate_lower == "clear":
            try:
                prev = mgr.clear_gates()
            except RuntimeError as exc:
                return f"/goal gate clear: {exc}"
            return f"✓ Cleared {prev} gate{'s' if prev != 1 else ''}."
        return "Usage: /goal gate [list | add <command> | remove <N> | clear]"

    async def _goal_set(self, mgr, args: str, lower: str, event: MessageEvent) -> str:
        """Set a new goal from free text, inline ``field: value`` contract lines, or ``draft <objective>``."""
        if lower.startswith("draft"):
            # Draft a structured completion contract, then set it. The aux LLM call is sync;
            # run it off the event loop.
            objective = args[len("draft"):].strip()
            if not objective:
                return "Usage: /goal draft <objective in plain language>"
            try:
                from hermes_cli.goals import draft_contract

                # _run_in_executor_with_context, not a bare hop: drafting a contract calls the
                # auxiliary LLM, whose provider/credential resolution reads the profile secret scope
                # — a contextvar that a default-executor hop drops, leaving it unscoped.
                contract = await self._run_in_executor_with_context(draft_contract, objective)
            except Exception as exc:
                logger.debug("goal draft failed: %s", exc)
                contract = None
            args = objective  # the goal text is the objective
        else:
            # Inline `field: value` lines parse into a completion contract; the remaining prose is
            # the goal headline. Plain free-form goals (no such lines) behave exactly as before.
            from hermes_cli.goals import parse_contract

            headline, parsed = parse_contract(args)
            args = headline or args
            contract = parsed if not parsed.is_empty() else None

        try:
            state = mgr.set(args, contract=contract)
        except ValueError as exc:
            return t("gateway.goal.invalid", error=str(exc))

        # Queue the goal text as an immediate first turn so the agent starts making progress. The
        # post-turn hook takes over after.
        adapter, _quick_key = self._adapter_and_key_for(event)
        if adapter and _quick_key:
            try:
                kickoff_event = MessageEvent(
                    text=state.goal,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=event.channel_prompt,
                )
                self._enqueue_fifo(_quick_key, kickoff_event, adapter)
            except Exception as exc:
                logger.debug("goal kickoff enqueue failed: %s", exc)

        base = t("gateway.goal.set", budget=state.max_turns, goal=state.goal)
        if state.has_contract():
            return f"{base}\nCompletion contract:\n{state.contract.render_block()}"
        if lower.startswith("draft"):
            # Drafting was requested but the aux model couldn't produce one.
            return f"{base}\n(Couldn't draft a contract — running as a free-form goal.)"
        return base

    async def _handle_heartbeat_command(self, event: "MessageEvent") -> str:
        """Handle /heartbeat for gateway platforms (mirror of CLI handler).

        Manages the session's one recurring re-entry prompt. The gateway-wide poller injects due
        heartbeats through the adapter FIFO as ordinary user turns, so alternation and caching hold.
        """
        from hermes_cli.heartbeat import parse_interval, format_interval, MIN_INTERVAL_SECONDS

        args = (event.get_command_args() or "").strip()
        lower = args.lower()

        mgr, session_entry = await self._get_heartbeat_manager_for_event(event)
        if mgr is None:
            return "Heartbeats unavailable (no session)."

        quick_key = self._session_key_for_source(event.source) if event.source else None

        if not args or lower == "status":
            return mgr.status_line()

        if lower == "pause":
            state = mgr.pause()
            return f"⏸ Heartbeat paused: {state.prompt}" if state else "No heartbeat set."

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return "No heartbeat to resume."
            if quick_key and event.source is not None:
                self._register_heartbeat_watch(quick_key, event.source, mgr.session_id)
            return f"▶ Heartbeat resumed (every {format_interval(state.interval_seconds)}): {state.prompt}"

        if lower in {"clear", "stop", "off"}:
            had = mgr.clear()
            if quick_key:
                self._unregister_heartbeat_watch(quick_key)
            return "✓ Heartbeat cleared." if had else "No heartbeat set."

        # Set: `/heartbeat every 10m <prompt>` (also accepts `10m <prompt>`).
        tokens = args.split(None, 2)
        interval = None
        prompt = ""
        if tokens and tokens[0].lower() == "every" and len(tokens) >= 2:
            interval = parse_interval(f"every {tokens[1]}")
            prompt = tokens[2] if len(tokens) > 2 else ""
        elif tokens:
            interval = parse_interval(tokens[0])
            prompt = args[len(tokens[0]):].strip() if interval and interval > 0 else ""

        if interval is None:
            return (
                "Usage: /heartbeat every <interval> <prompt>  (e.g. /heartbeat every 10m Check CI)\n"
                "Also: /heartbeat status | pause | resume | clear"
            )
        if interval < 0:
            return f"Interval too small — minimum is {MIN_INTERVAL_SECONDS}s."
        if not prompt.strip():
            return "Usage: /heartbeat every <interval> <prompt> — the prompt is required."

        try:
            state = mgr.set(prompt, interval)
        except ValueError as exc:
            return f"Invalid heartbeat: {exc}"
        if quick_key and event.source is not None:
            self._register_heartbeat_watch(quick_key, event.source, mgr.session_id)
        return (
            f"♥ Heartbeat set (every {format_interval(state.interval_seconds)}): {state.prompt}\n"
            "Fires as a normal turn whenever this session is idle and the interval has "
            "elapsed. Lives while the gateway runs — use `hermes cron` for durable schedules."
        )

    def _idle_cached_agent_or_error(self, event: MessageEvent, verb: str):
        """``(session_key, cached_agent, None)`` for /refine and /review, or ``(_, _, error_text)``.

        Both need a cached agent from a completed turn and refuse while a run is in flight.
        """
        quick_key = self._session_key_for_source(event.source) if event.source else None
        if not quick_key:
            return None, None, f"{verb.capitalize()} unavailable (no session)."
        if quick_key in self._running_agents:
            return quick_key, None, f"Agent is running — wait for the turn to finish, then /{verb}."
        agent = self._cached_agent_for(quick_key)
        if agent is None:
            return quick_key, None, f"Nothing to {verb} yet — send a message first."
        return quick_key, agent, None

    async def _handle_refine_command(self, event: "MessageEvent") -> str:
        """Handle /refine — run the memory/skill review fork on demand.

        Runs in a daemon thread against a snapshot of the cached AIAgent's conversation; the live
        session and prompt cache are untouched. Requires at least one completed turn.
        """
        args = (event.get_command_args() or "").strip()
        quick_key, agent, error = self._idle_cached_agent_or_error(event, "refine")
        if error:
            return error

        snapshot = list(getattr(agent, "_session_messages", None) or [])
        if not snapshot:
            return "Nothing to refine yet — the conversation is empty."

        review_skills = "skill_manage" in getattr(agent, "valid_tool_names", set())
        try:
            agent._spawn_background_review(
                messages_snapshot=snapshot,
                review_memory=True,
                review_skills=review_skills,
                focus=args or None,
            )
        except Exception as exc:
            return f"/refine failed to start: {exc}"
        tail = f" (focus: {args})" if args else ""
        return (
            f"⚗ Reviewing this conversation in the background{tail} — "
            f"any memory/skill updates will be reported when done."
        )

    async def _handle_review_command(self, event: "MessageEvent") -> str:
        """Handle /review — spawn an independent reviewer subagent.

        The approval session-key contextvar is only bound during agent turns, so bind it explicitly
        here or the completion event carries no gateway route and never re-enters this chat.
        """
        args = (event.get_command_args() or "").strip()
        quick_key, agent, error = self._idle_cached_agent_or_error(event, "review")
        if error:
            return error

        snapshot = list(getattr(agent, "_session_messages", None) or [])

        from tools.approval import (
            reset_current_session_key,
            set_current_session_key,
        )

        def _dispatch():
            token = set_current_session_key(quick_key)
            try:
                from agent.review_engine import start_review

                return start_review(agent, snapshot, args)
            finally:
                reset_current_session_key(token)

        try:
            # _run_in_executor_with_context, not a bare hop: the reviewer
            # subagent is spawned from the worker and inherits its context,
            # so a bare hop would run it under the launch home / no secret scope.
            result = await self._run_in_executor_with_context(_dispatch)
        except ValueError as exc:
            return str(exc)
        except Exception as exc:
            return f"/review failed to start: {exc}"

        from agent.review_engine import format_dispatch_note

        return format_dispatch_note(result, args)

    async def _handle_subgoal_command(self, event: "MessageEvent") -> str:
        """Handle /subgoal for gateway platforms (mirror of CLI handler).

        Subgoals are extra criteria appended to the active goal mid-loop. They modify state read
        at the next turn boundary, so this is safe to invoke while the agent is running.
        """
        args = (event.get_command_args() or "").strip()
        mgr, _session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")
        if not mgr.has_goal():
            return "No active goal. Set one with /goal <text>."

        # No args → list current subgoals.
        if not args:
            return f"{mgr.status_line()}\n{mgr.render_subgoals()}"

        tokens = args.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        if verb == "remove":
            if not rest:
                return "Usage: /subgoal remove <n>"
            try:
                idx = int(rest.split()[0])
            except ValueError:
                return "/subgoal remove: <n> must be an integer (1-based index)."
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as exc:
                return f"/subgoal remove: {exc}"
            return f"✓ Removed subgoal {idx}: {removed}"

        if verb == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as exc:
                return f"/subgoal clear: {exc}"
            if prev:
                return f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}."
            return "No subgoals to clear."

        try:
            text = mgr.add_subgoal(args)
        except (ValueError, RuntimeError) as exc:
            return f"/subgoal: {exc}"
        idx = len(mgr.state.subgoals) if mgr.state else 0
        return f"✓ Added subgoal {idx}: {text}"

    async def _get_loop_manager_for_event(self, event: "MessageEvent"):
        """Return a LoopManager bound to the session for this gateway event.

        Returns ``(manager, session_entry)``, or ``(None, None)`` when the loops module or session
        can't be loaded. Mirrors ``_get_goal_manager_for_event``.
        """
        try:
            from hermes_cli.loops import LoopManager
        except Exception as exc:
            logger.debug("loop manager unavailable: %s", exc)
            return None, None
        # Warm the SessionDB cache off-loop. A cold cache drops the first
        # /loop write while the reply claims the loop was set (same class
        # as the /goal false-ack fix).
        await self._warm_goals_session_db("loop manager")
        try:
            session_entry = await self.async_session_store.get_or_create_session(event.source)
        except Exception:
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        return LoopManager(session_id=sid), session_entry

    async def _handle_loop_command(self, event: "MessageEvent") -> str:
        """Handle /loop for gateway platforms — recurring in-session wakeups.

        Mirrors the CLI handler via ``dispatch_loop_command``. New loops capture the event's routing
        (platform/chat/thread) so the idle loop-wakeup watcher can inject ticks here after a restart.
        """
        try:
            from hermes_cli.loops import dispatch_loop_command, goal_blocks_loop_tick
        except Exception as exc:
            logger.debug("loops module unavailable: %s", exc)
            return "Loops unavailable."

        mgr, _session_entry = await self._get_loop_manager_for_event(event)
        if mgr is None:
            return "Loops unavailable (no active session)."

        route: dict = {}
        try:
            src = event.source
            if src is not None:
                platform = getattr(src, "platform", "")
                route = {
                    "platform": platform.value if hasattr(platform, "value") else str(platform or ""),
                    "chat_id": str(getattr(src, "chat_id", "") or ""),
                    "chat_type": str(getattr(src, "chat_type", "") or ""),
                    "thread_id": str(getattr(src, "thread_id", "") or ""),
                    "user_id": str(getattr(src, "user_id", "") or ""),
                    "user_name": str(getattr(src, "user_name", "") or ""),
                }
                route = {k: v for k, v in route.items() if v}
        except Exception:
            route = {}

        args = (event.get_command_args() or "").strip()
        result = dispatch_loop_command(mgr, args, route=route)
        output = result.get("output") or ""
        if result.get("created"):
            try:
                if goal_blocks_loop_tick(mgr.session_id):
                    output += (
                        "\nNote: an active /goal is driving this session — loop "
                        "wakeups defer until the goal finishes, pauses, or parks."
                    )
            except Exception:
                pass
        return output

    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo [N] — back up N user turns (default 1), soft-deleting the truncated rows and
        echoing the backed-up text. Evicts the cached agent so the next message rebuilds context
        from the active-only transcript (gateway analogue of the CLI's history surgery).
        """
        source = event.source

        # Parse optional turn count: "/undo" → 1, "/undo 3" → 3.
        n = 1
        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                n = int(raw_args.split()[0])
            except (ValueError, IndexError):
                return t("gateway.undo.invalid_count", arg=raw_args.split()[0])
            if n < 1:
                n = 1

        session_entry = await self.async_session_store.get_or_create_session(source)
        result = await self.async_session_store.rewind_session(session_entry.session_id, n)

        if result is None:
            return t("gateway.undo.nothing")

        # Reset stored token count — transcript was truncated.
        session_entry.last_prompt_tokens = 0
        # Evict the cached agent so the next turn rebuilds from the active-only
        # transcript and memory providers refresh their per-session caches.
        try:
            session_key = build_session_key(source)
            self._evict_cached_agent(session_key)
        except Exception as e:
            logger.debug("undo: cached-agent eviction skipped: %s", e)

        target_text = result["target_text"]
        preview = target_text[:200] + "..." if len(target_text) > 200 else target_text
        return t(
            "gateway.undo.removed",
            turns=result["turns_undone"],
            count=result["rewound_count"],
            preview=preview,
        )

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        from gateway.run import _home_target_env_var, _home_thread_env_var
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        if source.platform is None:
            return t("gateway.set_home.save_failed", error="Missing logical platform")

        via_relay = getattr(source, "delivered_via_upstream_relay", False) is True
        if via_relay:
            adapter_for_source = getattr(self, "_adapter_for_source", None)
            relay_adapter = adapter_for_source(source) if callable(adapter_for_source) else None
            fronts_platform = getattr(relay_adapter, "fronts_platform", None)
            if (
                source.platform in {None, Platform.LOCAL, Platform.RELAY}
                or not getattr(source, "user_id", None)
                or not callable(fronts_platform)
                or not fronts_platform(source.platform)
            ):
                return t(
                    "gateway.set_home.save_failed",
                    error="Relay does not authenticate this logical home target",
                )

        thread_id = _home_thread_from_source(source)
        home = HomeChannel(
            platform=source.platform,
            chat_id=str(chat_id),
            name=chat_name,
            thread_id=str(thread_id) if thread_id else None,
            user_id=(
                str(source.user_id)
                if getattr(source, "user_id", None)
                else None
            ),
            scope_id=(
                str(source.scope_id)
                if getattr(source, "scope_id", None)
                else None
            ),
        )

        # config.yaml is canonical because it can persist the authenticated
        # logical-target provenance required by Relay after a restart.
        try:
            persist_home_channel(home, enabled_if_new=not via_relay)
        except Exception as e:
            return t("gateway.set_home.save_failed", error=e)

        # Preserve legacy home env vars for existing cron/setup consumers.
        env_key = _home_target_env_var(platform_name)
        thread_env_key = _home_thread_env_var(platform_name)
        try:
            from hermes_cli.config import save_env_value
            save_env_value(env_key, str(chat_id))
            save_env_value(thread_env_key, str(thread_id or ""))
        except Exception as e:
            logger.warning("Home config saved but legacy env persistence failed: %s", e)

        # Keep the running gateway config in sync too. The pre-restart
        # notification path reads self.config before the process reloads config.
        platform_config = self.config.platforms.setdefault(
            source.platform,
            PlatformConfig(enabled=not via_relay),
        )
        platform_config.home_channel = home

        return t("gateway.set_home.success", name=chat_name, chat_id=chat_id)

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice [on|off|tts|channel|leave|status] command."""
        args = event.get_command_args().strip().lower()
        chat_id = event.source.chat_id
        # Voice state belongs to the (bot, chat) pair: resolve the adapter that
        # received the command and key the mode by its owning profile so two
        # multiplexed bots in one chat keep independent /voice state (#75198).
        voice_key = self._voice_key_for_source(event.source)

        adapter = self._adapter_for_source(event.source)

        def _set_mode(mode: str) -> None:
            self._voice_mode[voice_key] = mode
            self._save_voice_modes()
            if adapter:
                if mode == "off":
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
                else:
                    self._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)

        if args in _VOICE_MODE_BY_ARG:
            mode, reply_key = _VOICE_MODE_BY_ARG[args]
            _set_mode(mode)
            return t(reply_key)
        elif args in {"channel", "join"}:
            return await self._handle_voice_channel_join(event)
        elif args == "leave":
            return await self._handle_voice_channel_leave(event)
        elif args == "status":
            mode = self._voice_mode.get(voice_key, "off")
            labels = {
                "off": t("gateway.voice.label_off"),
                "voice_only": t("gateway.voice.label_voice_only"),
                "all": t("gateway.voice.label_all"),
            }
            # Append voice channel info if connected
            guild_id = self._get_guild_id(event)
            if guild_id and hasattr(adapter, "get_voice_channel_info"):
                info = adapter.get_voice_channel_info(guild_id)
                if info:
                    lines = [
                        t("gateway.voice.status_mode", label=labels.get(mode, mode)),
                        t("gateway.voice.status_channel", channel=info['channel_name']),
                        t("gateway.voice.status_participants", count=info['member_count']),
                    ]
                    for m in info["members"]:
                        status = t("gateway.voice.speaking") if m.get("is_speaking") else ""
                        lines.append(t("gateway.voice.status_member", name=m['display_name'], status=status))
                    return "\n".join(lines)
            return t("gateway.voice.status_mode", label=labels.get(mode, mode))
        else:
            # Toggle: off → on, on/all → off
            if self._voice_mode.get(voice_key, "off") == "off":
                _set_mode("voice_only")
                toggle_line = t("gateway.voice.enabled_short")
            else:
                _set_mode("off")
                toggle_line = t("gateway.voice.disabled_short")
            # Bare /voice still toggles, but append an explainer so users discover the
            # on/off/tts/status subcommands (and, on Discord, live voice-channel join/leave). The
            # toggle result is shown first via the {toggle} placeholder.
            supports_voice_channels = adapter is not None and hasattr(
                adapter, "join_voice_channel"
            )
            channels = (
                t("gateway.voice.help_channels") if supports_voice_channels else ""
            )
            return t("gateway.voice.help", toggle=toggle_line, channels=channels)

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import format_checkpoint_list

        mgr = self._checkpoint_manager()
        if mgr is None:
            return t("gateway.rollback.not_enabled")

        from tools.terminal_scope import terminal_env as _tenv

        cwd = _tenv("TERMINAL_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        # --all / --force: classic full restore, overwriting user edits too.
        restore_all = False
        arg_parts = []
        for tok in arg.split():
            if tok.lower() in ("--all", "--force"):
                restore_all = True
            else:
                arg_parts.append(tok)
        arg = " ".join(arg_parts)

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return t("gateway.rollback.none_found", cwd=cwd)

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return t("gateway.rollback.invalid_number", max=len(checkpoints))
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash, safe=not restore_all)
        if result["success"]:
            msg = t(
                "gateway.rollback.restored",
                hash=result["restored_to"],
                reason=result["reason"],
            )
            for result_key, i18n_key in _ROLLBACK_SKIP_LINES:
                files = result.get(result_key) or []
                if files:
                    shown = ", ".join(files[:5])
                    more = f" (+{len(files) - 5})" if len(files) > 5 else ""
                    msg += "\n" + t(i18n_key, files=shown + more)
            return msg
        return t("gateway.rollback.restore_failed", error=result["error"])

    async def _handle_diff_command(self, event: MessageEvent) -> str:
        """Handle /diff — show git changes in the working directory.

        Diff body is truncated hard here (chat is not a pager); platform senders clamp further.
        """
        args = event.get_command_args().strip()

        stat_only = False
        mode = "working"
        for arg in args.split():
            low = arg.lower()
            if low in ("--stat", "stat"):
                stat_only = True
            elif low in ("staged", "--staged", "cached", "--cached"):
                mode = "staged"
            elif low in ("all", "--all", "head"):
                mode = "all"
            elif low == "session":
                mode = "session"

        from tools.terminal_scope import terminal_env as _tenv

        cwd = _tenv("TERMINAL_CWD", str(Path.home()))

        if mode == "session":
            return await self._gateway_session_diff(cwd, stat_only)

        from tools.working_diff import collect_working_diff

        result = await asyncio.to_thread(collect_working_diff, cwd, mode)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        untracked = result.get("untracked", [])
        if result.get("empty") or (not stat and not diff and not untracked):
            return t("gateway.diff.no_changes")

        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if untracked:
            shown = "\n".join(f"+ {rel}" for rel in untracked[:15])
            more = f"\n... and {len(untracked) - 15} more" if len(untracked) > 15 else ""
            out.append(f"**Untracked:**\n```\n{shown}{more}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    async def _gateway_session_diff(self, cwd: str, stat_only: bool) -> str:
        """Cumulative checkpoint-baseline diff for /diff session (gateway)."""
        mgr = self._checkpoint_manager()
        if mgr is None:
            return t("gateway.diff.not_enabled")

        result = await asyncio.to_thread(mgr.session_diff, cwd)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        if result.get("empty") or (not stat and not diff):
            return t("gateway.diff.no_changes")

        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    @staticmethod
    def _fenced_truncated_diff(diff: str, max_lines: int = 60,
                               max_chars: int = 3000) -> str:
        """Fence a diff body, truncating to messaging-friendly size."""
        diff_lines = diff.splitlines()
        truncated = False
        if len(diff_lines) > max_lines:
            diff = "\n".join(diff_lines[:max_lines])
            truncated = True
        if len(diff) > max_chars:
            diff = diff[:max_chars]
            truncated = True
        note = ""
        if truncated:
            note = (
                f"\n... (truncated — {len(diff_lines)} lines total; "
                "use /diff --stat for a summary)"
            )
        return f"```diff\n{diff}{note}\n```"

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /bg <prompt> — run a prompt in a background thread with its own session; the
        result is sent to the same chat without touching the active session's history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return t("gateway.background.usage")

        source = event.source
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"

        event_message_id = self._reply_anchor_for_event(event)

        # Forward image/audio attachments so the background agent can see them.
        media_urls = list(event.media_urls) if event.media_urls else []
        media_types = list(event.media_types) if event.media_types else []

        # Fire-and-forget the background task
        _task = asyncio.create_task(
            self._run_background_task(
                prompt,
                source,
                task_id,
                event_message_id=event_message_id,
                media_urls=media_urls,
                media_types=media_types,
            )
        )
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        return t("gateway.background.started", preview=preview, task_id=task_id)

    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw <question> — answer a side question via a one-shot auxiliary LLM call on a
        transcript snapshot; live history is never touched (alternation + prompt cache intact,
        current turn keeps running). Unlike /bg, which spawns a fresh contextless session.
        """
        question = event.get_command_args().strip()
        if not question:
            return t("gateway.btw.usage")

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if not history:
            return t("gateway.btw.no_history")

        try:
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
            )
        except Exception:
            model, runtime_kwargs = None, {}
        if not runtime_kwargs.get("api_key"):
            return t("gateway.btw.no_provider")

        main_runtime = {
            "model": model,
            "provider": runtime_kwargs.get("provider"),
            "base_url": runtime_kwargs.get("base_url"),
            "api_key": runtime_kwargs.get("api_key"),
            "api_mode": runtime_kwargs.get("api_mode"),
        }
        history_snapshot = list(history)
        # Prefer the cache-parity fork when a live cached AIAgent exists: it replays the snapshot
        # against the warm provider prefix cache, giving FULL context at cache-read prices. With no
        # cached agent the cache is cold anyway — answer_side_question's digest fallback handles it.
        try:
            parent_agent = self._cached_agent_for(self._session_key_for_source(source))
        except Exception:
            parent_agent = None
        _thread_metadata = self._reply_metadata(event)
        adapter = self._adapter_for_source(source)
        preview = question[:60] + ("..." if len(question) > 60 else "")

        async def _run_side_question() -> None:
            from agent.side_question import answer_side_question
            try:
                answer = await asyncio.to_thread(
                    answer_side_question,
                    question,
                    history_snapshot,
                    parent_agent=parent_agent,
                    main_runtime=main_runtime,
                )
            except Exception as e:
                logger.warning("/btw side question failed: %s", e)
                if adapter is not None:
                    await adapter.send(
                        source.chat_id,
                        t("gateway.btw.failed", preview=preview, error=str(e)),
                        metadata=_thread_metadata,
                    )
                return
            if adapter is not None:
                await adapter.send(
                    source.chat_id,
                    t("gateway.btw.answer", preview=preview, answer=answer or ""),
                    metadata=_thread_metadata,
                )

        _task = asyncio.create_task(_run_side_question())
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        return t("gateway.btw.started", preview=preview)

    def _save_gateway_config_key(self, key_path: str, value) -> bool:
        """Save a dot-separated key to config.yaml (shared by /reasoning, /fast
        and their interactive pickers)."""
        from gateway.run import _gateway_config_home
        from hermes_cli.config import read_user_config_raw
        config_path = _gateway_config_home() / "config.yaml"
        try:
            # Write-back round-trip: raw read is correct (merged defaults must
            # not be persisted back to the user's file).
            user_config = read_user_config_raw(config_path)
            keys = key_path.split(".")
            current = user_config
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            atomic_config_write(config_path, user_config)
            return True
        except Exception as e:
            logger.error("Failed to save config key %s: %s", key_path, e)
            return False

    def _apply_reasoning_selection(
        self,
        session_key: str,
        platform_key: str,
        value: str,
        persist_global: bool = False,
    ) -> str:
        """Apply a /reasoning argument (typed or picked) and return the reply.

        Single path shared by `/reasoning <arg>` and the choice picker so both match the parser.
        """
        from hermes_constants import parse_reasoning_effort

        value = (value or "").strip().lower()

        # Display toggle (per-platform)
        if value in {"show", "on"}:
            self._show_reasoning = True
            self._save_gateway_config_key(
                f"display.platforms.{platform_key}.show_reasoning", True
            )
            return t("gateway.reasoning.display_set_on", platform=platform_key)
        if value in {"hide", "off"}:
            self._show_reasoning = False
            self._save_gateway_config_key(
                f"display.platforms.{platform_key}.show_reasoning", False
            )
            return t("gateway.reasoning.display_set_off", platform=platform_key)

        if value == "reset":
            if persist_global:
                return t("gateway.reasoning.reset_global_unsupported")
            self._set_session_reasoning_override(session_key, None)
            self._reasoning_config = self._load_reasoning_config()
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.reset_done")

        parsed = parse_reasoning_effort(value)
        if parsed is None:
            return t("gateway.reasoning.unknown_arg", arg=value)

        self._reasoning_config = parsed
        if persist_global:
            if self._save_gateway_config_key("agent.reasoning_effort", value):
                self._set_session_reasoning_override(session_key, None)
                self._evict_cached_agent(session_key)
                return t("gateway.reasoning.set_global", effort=value)
            self._set_session_reasoning_override(session_key, parsed)
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.set_global_save_failed", effort=value)

        self._set_session_reasoning_override(session_key, parsed)
        self._evict_cached_agent(session_key)
        return t("gateway.reasoning.set_session", effort=value)

    def _reasoning_picker_choices(self, current_effort: str) -> list:
        """Build the choice list for the interactive /reasoning picker."""
        from hermes_constants import VALID_REASONING_EFFORTS

        choices = [
            {
                "value": "none",
                "label": t("gateway.reasoning.choice_none"),
                "is_current": current_effort == "none",
            }
        ]
        for level in VALID_REASONING_EFFORTS:
            choices.append(
                {
                    "value": level,
                    "label": level,
                    "is_current": level == current_effort,
                }
            )
        choices.extend(
            [
                {"value": "reset", "label": t("gateway.reasoning.choice_reset"), "is_current": False},
                {"value": "show", "label": t("gateway.reasoning.choice_show"), "is_current": False},
                {"value": "hide", "label": t("gateway.reasoning.choice_hide"), "is_current": False},
            ]
        )
        return choices

    async def _try_send_choice_picker(
        self,
        event: MessageEvent,
        session_key: str,
        title: str,
        choices: list,
        on_choice_selected,
    ) -> bool:
        """Send an interactive choice picker when the platform supports it.

        Mirrors the `/model` gate: capability is detected on the adapter *type*
        (``send_choice_picker``); a failed send returns False (text fallback) instead of erroring.
        """
        adapter = self._adapter_for_source(event.source)
        has_picker = (
            adapter is not None
            and getattr(type(adapter), "send_choice_picker", None) is not None
        )
        if not has_picker:
            return False
        try:
            metadata = self._reply_metadata(event)
            result = await adapter.send_choice_picker(
                chat_id=event.source.chat_id,
                title=title,
                choices=choices,
                session_key=session_key,
                on_choice_selected=on_choice_selected,
                metadata=metadata,
            )
            return bool(getattr(result, "success", False))
        except Exception as e:
            logger.warning("send_choice_picker failed, falling back to text: %s", e)
            return False

    async def _handle_reasoning_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reasoning command — manage reasoning effort and display toggle."""
        from gateway.run import _platform_config_key

        raw_args = event.get_command_args().strip()
        args, persist_global = self._parse_reasoning_command_args(raw_args)
        # Normalize the source (Telegram DM topic recovery) before deriving
        # the override key so storage matches the key the next message turn
        # reads — same fix as /model (#30479).
        _reasoning_source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(_reasoning_source)
        self._show_reasoning = self._load_show_reasoning()
        # Use the session's effective model (session /model override wins over
        # config default) so per-model reasoning_overrides display correctly.
        _session_model = str(
            ((getattr(self, "_session_model_overrides", {}) or {}).get(session_key) or {}).get("model") or ""
        )
        self._reasoning_config = self._resolve_session_reasoning_config(
            source=event.source,
            session_key=session_key,
            model=_session_model,
        )

        if not raw_args:
            # Show current state
            rc = self._reasoning_config
            if rc is None:
                level = t("gateway.reasoning.level_default")
                current_effort = "medium"
            elif rc.get("enabled") is False:
                level = t("gateway.reasoning.level_disabled")
                current_effort = "none"
            else:
                level = rc.get("effort", "medium")
                current_effort = level
            display_state = (
                t("gateway.reasoning.display_on")
                if self._show_reasoning
                else t("gateway.reasoning.display_off")
            )
            has_session_override = session_key in (getattr(self, "_session_reasoning_overrides", {}) or {})
            scope = (
                t("gateway.reasoning.scope_session")
                if has_session_override
                else t("gateway.reasoning.scope_global")
            )

            # Interactive picker on platforms that support it (parity with the
            # /model picker). Falls through to the text status card otherwise.
            _picker_platform_key = _platform_config_key(event.source.platform)

            async def _on_reasoning_choice(_chat_id: str, value: str) -> str:
                return self._apply_reasoning_selection(
                    session_key, _picker_platform_key, value
                )

            picker_sent = await self._try_send_choice_picker(
                event,
                session_key,
                title=t(
                    "gateway.reasoning.picker_title",
                    level=level,
                    scope=scope,
                    display=display_state,
                ),
                choices=self._reasoning_picker_choices(current_effort),
                on_choice_selected=_on_reasoning_choice,
            )
            if picker_sent:
                return None  # Picker sent — adapter handles the response

            return t(
                "gateway.reasoning.status",
                level=level,
                scope=scope,
                display=display_state,
            )

        # Typed argument path — same applier the picker uses.
        platform_key = _platform_config_key(event.source.platform)
        return self._apply_reasoning_selection(
            session_key, platform_key, args, persist_global=persist_global
        )

    async def _handle_memory_command(self, event: MessageEvent) -> str:
        """Handle /memory — review pending memory writes + toggle the approval gate.

        Entries are small enough to review inline, so the full flow works on every platform.
        """
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        from tools.memory_tool import load_on_disk_store

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []
        _set_approval = self._write_approval_setter("memory", self._session_key_for_source(event.source))

        # Apply approved writes against a fresh on-disk store (the gateway has
        # no long-lived agent; the store persists to the same MEMORY/USER.md).
        # load_on_disk_store() honors the user's configured char limits.
        store = load_on_disk_store()

        out = handle_pending_subcommand(
            wa.MEMORY, args, memory_store=store, set_mode_fn=_set_approval,
        )
        if out is None:
            out = ("Unknown /memory subcommand. Use: pending, approve <id>, "
                   "reject <id>, approval <on|off>.")
        return out

    async def _handle_skills_command(self, event: MessageEvent) -> str:
        """Handle /skills on the gateway — pending skill-write review only (hub stays CLI-only).

        Gated by ``skills.write_approval`` but still answers when staged writes exist after the
        gate is off (never stranded). ``diff`` is truncated for chat.
        """
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []

        gate_on = wa.write_approval_enabled(wa.SKILLS)
        wants_toggle = bool(args) and args[0].lower() in {"approval", "mode"}
        if not gate_on and not wants_toggle and wa.pending_count(wa.SKILLS) == 0:
            return ("Skill write approval is off (skills.write_approval). "
                    "Enable it with /skills approval on, then review staged "
                    "writes here with /skills pending.")

        out = handle_pending_subcommand(
            wa.SKILLS, args,
            set_mode_fn=self._write_approval_setter("skills", self._session_key_for_source(event.source)),
        )
        if out is None:
            return ("Unknown /skills subcommand on this platform. Use: pending, "
                    "approve <id>, reject <id>, diff <id>, approval <on|off>. "
                    "(Search/install are CLI-only.)")

        # Chat bubbles can't hold a full skill diff — truncate and point at the pending JSON file
        # (NOT `hermes skills diff <name>`, which diffs a bundled skill against its stock version).
        if args and args[0].lower() == "diff" and len(out) > 3000:
            pending_id = args[1] if len(args) > 1 else "<id>"
            out = (out[:3000]
                   + "\n… (truncated — full diff in "
                     f"~/.hermes/pending/skills/{pending_id}.json)")
        return out

    async def _handle_fast_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /fast — mirror the CLI Priority Processing toggle in gateway chats.

        Session-scoped by default; ``--global`` persists agent.service_tier (parity with /model).
        """
        from gateway.run import _load_gateway_config, _resolve_gateway_model
        from hermes_cli.models import model_supports_fast_mode

        raw_args = event.get_command_args().strip().lower()
        # Reuse the /reasoning arg parser: strips --global (any position),
        # normalizes unicode dashes.
        args, persist_global = self._parse_reasoning_command_args(raw_args)
        session_key = self._session_key_for_source(event.source)
        self._service_tier = self._resolve_session_service_tier(
            session_key=session_key
        )

        user_config = _load_gateway_config()
        model = _resolve_gateway_model(user_config)
        if not model_supports_fast_mode(model):
            return t("gateway.fast.not_supported")

        def _apply_fast_selection(value: str, persist: bool = False) -> str:
            """Apply a /fast argument (typed or picked) and return the reply."""
            if value in {"fast", "on"}:
                tier = "priority"
                saved_value = "fast"
                label = t("gateway.fast.label_fast")
            elif value in {"normal", "off"}:
                tier = None
                saved_value = "normal"
                label = t("gateway.fast.label_normal")
            elif value in {"auto", "cold"}:
                tier = saved_value = value
                label = value.upper()
            else:
                return t("gateway.fast.unknown_arg", arg=value)
            self._service_tier = tier
            if persist:
                if self._save_gateway_config_key("agent.service_tier", saved_value):
                    # Global write supersedes any session override.
                    self._set_session_service_tier_override(
                        session_key, None, clear=True
                    )
                    self._evict_cached_agent(session_key)
                    return t("gateway.fast.saved", label=label)
                # Config write failed — fall back to a session override so the
                # user's choice still applies (mirrors /reasoning --global).
                self._set_session_service_tier_override(session_key, tier)
                self._evict_cached_agent(session_key)
                return t("gateway.fast.session_only", label=label)
            self._set_session_service_tier_override(session_key, tier)
            self._evict_cached_agent(session_key)
            return t("gateway.fast.session_only", label=label)

        if not args or args == "status":
            is_fast = self._service_tier == "priority"
            mode = "fast" if is_fast else (self._service_tier or "normal")
            status = {"fast": t("gateway.fast.status_fast"), "normal": t("gateway.fast.status_normal")}.get(mode, mode)

            async def _on_fast_choice(_chat_id: str, value: str) -> str:
                return _apply_fast_selection(value, persist=persist_global)

            picker_sent = await self._try_send_choice_picker(
                event,
                session_key,
                title=t("gateway.fast.picker_title", mode=status),
                choices=[
                    {"value": v, "label": t(f"gateway.fast.choice_{v}"), "is_current": mode == v}
                    for v in ("fast", "normal", "auto", "cold")
                ],
                on_choice_selected=_on_fast_choice,
            )
            if picker_sent:
                return None  # Picker sent — adapter handles the response

            return t("gateway.fast.status", mode=status)

        return _apply_fast_selection(args, persist=persist_global)

    async def _handle_approvals_command(self, event: MessageEvent) -> str:
        """Show or persist the profile-wide dangerous-command approval mode."""
        from gateway.slash_access import policy_for_source
        from hermes_cli.approval_mode import run_approval_mode_command

        requested = event.get_command_args().strip() or None
        # This mutates profile-wide security policy. The central slash gate can
        # allow selected commands to non-admin users, so enforce admin again at
        # this side-effect boundary. Unconfigured policies remain unrestricted.
        policy = policy_for_source(self.config, event.source)
        if requested and not policy.is_admin(event.source.user_id):
            return "Only gateway admins can change the persistent approval mode."
        result = run_approval_mode_command(requested)
        # Approval checks load config dynamically; do not evict the cached agent
        # or alter its system prompt/tool schema (prompt-cache prefix is sacred).
        return result.message

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        session_key = self._session_key_for_source(event.source)
        current = is_session_yolo_enabled(session_key)
        if current:
            disable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.disabled"))
        else:
            enable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.enabled"))

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` (default off). Cycles off → new → all → verbose
        per *current platform*, saved to ``display.platforms.<platform>.tool_progress``.
        """
        from gateway.run import _gateway_config_home, _load_gateway_config, _platform_config_key

        config_path = _gateway_config_home() / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- check config gate ------------------------------------------------
        try:
            user_config = _load_gateway_config()
            gate_enabled = is_truthy_value(
                cfg_get(user_config, "display", "tool_progress_command"),
                default=False,
            )
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return t("gateway.verbose.not_enabled")

        # --- cycle mode (per-platform) ----------------------------------------
        cycle = ["off", "new", "all", "verbose", "log"]
        descriptions = {
            "off": t("gateway.verbose.mode_off"),
            "new": t("gateway.verbose.mode_new"),
            "all": t("gateway.verbose.mode_all"),
            "verbose": t("gateway.verbose.mode_verbose"),
            "log": t("gateway.verbose.mode_log"),
        }

        # Read current effective mode for this platform via the resolver
        from gateway.display_config import resolve_display_setting
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        if current not in cycle:
            current = "all"
        idx = (cycle.index(current) + 1) % len(cycle)
        new_mode = cycle[idx]

        # Save to display.platforms.<platform>.tool_progress
        try:
            if "display" not in user_config or not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if "platforms" not in display or not isinstance(display.get("platforms"), dict):
                display["platforms"] = {}
            if platform_key not in display["platforms"] or not isinstance(display["platforms"].get(platform_key), dict):
                display["platforms"][platform_key] = {}
            display["platforms"][platform_key]["tool_progress"] = new_mode
            atomic_config_write(config_path, user_config)
            return (
                f"{descriptions[new_mode]}\n"
                + t("gateway.verbose.saved_suffix", platform=platform_key)
            )
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{descriptions[new_mode]}\n" + t("gateway.verbose.save_failed", error=e)

    async def _handle_busy_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /busy — control what happens when messaging while Hermes is working."""
        arg = event.get_command_args().strip().lower()
        if not arg or arg == "status":
            mode = self._effective_busy_input_mode(event.source)
            behavior = _BUSY_MODE_BEHAVIOR.get(mode, _BUSY_MODE_BEHAVIOR["interrupt"])[0]
            return EphemeralReply(
                f"**Busy input mode: `{mode}`" + "\n"
                f"Messages while busy: _{behavior}_" + "\n"
                f"Change with `/busy queue`, `/busy steer`, or `/busy interrupt`."
            )

        if arg not in _BUSY_MODE_BEHAVIOR:
            return EphemeralReply(
                f"Unknown mode `{arg}`. Use `/busy queue`, `/busy steer`, or `/busy interrupt`."
            )

        # Persist before mutate
        from cli import save_config_value
        if save_config_value("display.busy_input_mode", arg):
            profile_name = self._busy_profile_name_for_source(event.source)
            if profile_name:
                from gateway.run import _load_gateway_runtime_config

                self._snapshot_profile_busy_modes(
                    profile_name,
                    _load_gateway_runtime_config(),
                )
            else:
                self._busy_input_mode = arg
                # busy_input_mode is also the source of truth for the text mode — re-derive it so the
                # adapter refresh below doesn't keep a stale value and keep interrupting.
                self._busy_text_mode = self._load_busy_text_mode()

            adapter = self._adapter_for_source(event.source)
            if adapter is not None:
                adapter._busy_text_mode = self._effective_busy_text_mode(event.source)

            behavior = _BUSY_MODE_BEHAVIOR[arg][1]
            return EphemeralReply(
                f"Busy input mode set to **`{arg}`** (saved)." + "\n"
                f"_{behavior}_"
            )
        return EphemeralReply(
            f"Busy input mode could not be saved to config. Mode unchanged."
        )

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        """Handle /footer command — toggle the runtime-metadata footer."""
        from gateway.run import _gateway_config_home, _load_gateway_config, _platform_config_key, _resolve_gateway_model
        from gateway.runtime_footer import resolve_footer_config

        config_path = _gateway_config_home() / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- parse argument -------------------------------------------------
        arg = ""
        try:
            text = (getattr(event, "message", None) or "").strip()
            if text.startswith("/"):
                parts = text.split(None, 1)
                if len(parts) > 1:
                    arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        # --- load config ----------------------------------------------------
        try:
            user_config: dict = _load_gateway_config()
        except Exception as e:
            return t("gateway.config_read_failed", error=e)

        effective = resolve_footer_config(user_config, platform_key)

        if arg in {"status", "?"}:
            state = t("gateway.footer.state_on") if effective["enabled"] else t("gateway.footer.state_off")
            fields = ", ".join(effective.get("fields") or [])
            return t(
                "gateway.footer.status",
                state=state,
                fields=fields,
                platform=platform_key,
            )

        if arg in {"on", "enable", "true", "1"}:
            new_state = True
        elif arg in {"off", "disable", "false", "0"}:
            new_state = False
        elif arg == "":
            new_state = not effective["enabled"]
        else:
            return t("gateway.footer.usage")

        # --- write global flag ---------------------------------------------
        try:
            if not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if not isinstance(display.get("runtime_footer"), dict):
                display["runtime_footer"] = {}
            display["runtime_footer"]["enabled"] = new_state
            atomic_config_write(config_path, user_config)
        except Exception as e:
            logger.warning("Failed to save runtime_footer.enabled: %s", e)
            return t("gateway.config_save_failed", error=e)

        state = t("gateway.footer.state_on") if new_state else t("gateway.footer.state_off")
        example = ""
        if new_state:
            # Show a preview using current agent state if available.
            from gateway.runtime_footer import format_runtime_footer
            preview = format_runtime_footer(
                model=_resolve_gateway_model(user_config) or None,
                context_tokens=0,
                context_length=None,
                fields=effective.get("fields") or ["model", "context_pct", "cwd"],
            )
            if preview:
                example = t("gateway.footer.example_line", preview=preview)
        return t("gateway.footer.saved", state=state, example=example)

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Profile-scoping wrapper around manual /compress.

        Multiplexed gateways resolve credentials through the fail-closed per-profile secret scope;
        slash dispatch (unlike ``_run_agent``) does not install it, so an unscoped /compress would
        raise ``UnscopedSecretError``. Single-profile gateways skip this.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._handle_compress_command_inner(event)

        from gateway.run import _profile_runtime_scope

        profile_home = self._resolve_profile_home_for_source(event.source)
        with _profile_runtime_scope(profile_home):
            return await self._handle_compress_command_inner(event)

    async def _compress_codex_app_server_session(
        self, session_key: str, session_id: str
    ) -> str:
        """Manual /compress for codex_app_server sessions.

        Compacts the LIVE cached agent's app-server thread (``thread/compact/start``, ``force=True``
        bypasses the ``codex_app_server_auto`` gate) and keeps the agent cached. Never builds a
        temporary agent or rewrites the mirror: neither can shrink the server-side thread.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL

        agent = self._cached_agent_for(session_key)
        if (
            agent is None
            or agent is _AGENT_PENDING_SENTINEL
            or getattr(agent, "_codex_session", None) is None
        ):
            return (
                "🗜️ Nothing to compact: this session runs on the Codex "
                "app-server runtime, whose context lives in a Codex-owned "
                "thread that only exists while the agent is active. Send a "
                "message first, then /compress — or /reset to start fresh."
            )

        compressor = getattr(agent, "context_compressor", None)
        count_before = getattr(compressor, "compression_count", 0)
        try:
            await self._run_in_executor_with_context(
                lambda: agent._compress_context(
                    [], "", force=True,
                )
            )
        except Exception as exc:
            return t("gateway.compress.failed", error=exc)
        count_after = getattr(compressor, "compression_count", 0)
        if count_after > count_before:
            return (
                "🗜️ Codex app-server thread compacted (thread/compact). "
                "The transcript mirror is unchanged by design — the "
                "app-server now carries the compacted context."
            )
        return (
            "⚠️ Codex app-server compaction did not complete — the thread "
            "is unchanged. Check the app-server logs, retry /compress, or "
            "/reset for a clean session."
        )

    async def _handle_compress_command_inner(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context.

        Optional ``/compress <focus>`` tells the summariser what to preserve, discarding the rest.
        """
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return t("gateway.compress.not_enough")

        # Parse args: either a focus topic (full compress) or the
        # boundary-aware "here [N]" form (partial compress).
        from hermes_cli.partial_compress import (
            extract_compress_flags,
            parse_partial_compress_args,
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
            summarize_compress_preview,
        )
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )
        _raw_args = (event.get_command_args() or "").strip()
        # Strip --preview/--dry-run/--aggressive before positional parsing
        # so the flags coexist with 'here [N]' / focus-topic forms.
        _raw_args, _preview, _aggressive = extract_compress_flags(_raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)

        _agg_note = ""
        if _aggressive:
            # LLM-free hard truncation is not supported on this surface — it would need its own
            # transcript-persistence branch outside the guarded _compress_context rotation machinery.
            _agg_note = t("gateway.compress.aggressive_unsupported")
            if not _preview:
                return _agg_note

        if _preview:
            # Report what WOULD be compressed — no agent, no writes.
            from agent.model_metadata import estimate_request_tokens_rough
            _pv_msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            report = summarize_compress_preview(
                _pv_msgs, partial, keep_last, focus_topic, estimate_request_tokens_rough(_pv_msgs)
            )
            lines = [f"🗜️ {line}" for line in report["lines"]]
            if _aggressive:
                lines.append(_agg_note)
            return "\n".join(lines)

        try:
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough
            from gateway.run import _platform_config_key

            session_key = self._session_key_for_source(source)
            # Preserve the platform + stable gateway session identity of a normal turn so external
            # context engines bind this agent to the original conversation, not a default "cli" host.
            platform_key = (
                _platform_config_key(source.platform) if source.platform else None
            )
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
            )
            if str(runtime_kwargs.get("api_mode") or "").lower() == "codex_app_server":
                # codex app-server: the model's context is the server-side thread owned by the LIVE
                # cached agent; a temporary agent has none (and finally-eviction would destroy the
                # real context). Compact the live thread and KEEP the agent cached; no mirror fallback.
                return await self._compress_codex_app_server_session(
                    session_key, session_entry.session_id
                )
            if not runtime_kwargs.get("api_key"):
                return t("gateway.compress.no_provider")

            # Pass the FULL transcript (tool results included), like auto-compress: user/assistant-
            # only starves tool-result pruning and can trip the protect-first/last early-return.
            msgs = [
                m for m in history
                if m.get("role") in {"user", "assistant", "tool"}
            ]

            # Boundary-aware split: only the head is summarized; the most recent `keep_last`
            # exchanges are preserved verbatim. The split snaps the tail to a user-turn start so the
            # rejoined transcript keeps role alternation valid.
            tail: list = []
            head = msgs
            if partial:
                head, tail = split_history_for_partial_compress(msgs, keep_last)
                if not tail:
                    # Degenerate split — fall back to full compression.
                    partial = False
                    head = msgs

            # Bind the temporary compression agent to the source's platform + stable gateway session
            # key. Assign directly (not setdefault: a resolver value would be a stale placeholder,
            # and it avoids duplicate-kwarg TypeError); platform only when known so None -> "cli" holds.
            if platform_key is not None:
                runtime_kwargs["platform"] = platform_key
            runtime_kwargs["gateway_session_key"] = session_key

            tmp_agent = await self._build_manual_compression_agent(
                session_entry.session_id, model, runtime_kwargs
            )
            try:
                # Estimate with system prompt + tool schemas included so the figure reflects real
                # request pressure, not a transcript-only underestimate. Must be computed after
                # tmp_agent is built so _cached_system_prompt/tools are populated.
                _sys_prompt = getattr(tmp_agent, "_cached_system_prompt", "") or ""
                _tools = getattr(tmp_agent, "tools", None) or None
                approx_tokens = estimate_request_tokens_rough(
                    msgs, system_prompt=_sys_prompt, tools=_tools
                )

                compressor = tmp_agent.context_compressor
                if not compressor.has_content_to_compress(head):
                    return t("gateway.compress.nothing_to_do")

                # Not a bare run_in_executor: the profile secret scope is a contextvar and the
                # default-executor hop would drop it, making the compressor's aux-client credential
                # resolution fail closed under multiplexing.
                compressed, _ = await self._run_in_executor_with_context(
                    lambda: tmp_agent._compress_context(
                        head,
                        "",
                        approx_tokens=approx_tokens,
                        focus_topic=focus_topic,
                        force=True,
                        defer_context_engine_notification=True,
                    )
                )

                # If _compress_context returned unchanged because a concurrent compression lock is
                # held, tell the user clearly instead of showing the misleading "No changes from
                # compression" no-op text.
                _lock_skipped = getattr(tmp_agent, "_compression_skipped_due_to_lock", None)
                if _lock_skipped is True or isinstance(_lock_skipped, str):
                    from agent.manual_compression_feedback import (
                        describe_compression_lock_skip,
                    )
                    return describe_compression_lock_skip(_lock_skipped)

                if partial and tail:
                    compressed = rejoin_compressed_head_and_tail(compressed, tail)

                await self._persist_manual_compression(tmp_agent, session_entry, source, compressed)
                finalize_context_engine_compression_notification(
                    tmp_agent,
                    committed=True,
                )
                new_tokens = estimate_request_tokens_rough(
                    compressed, system_prompt=_sys_prompt, tools=_tools
                )
                summary = summarize_manual_compression(
                    msgs,
                    compressed,
                    approx_tokens,
                    new_tokens,
                    compression_state=compressor,
                )
            finally:
                finalize_context_engine_compression_notification(
                    tmp_agent,
                    committed=False,
                )
                # Evict cached agent so next turn rebuilds system prompt
                # from current files (SOUL.md, memory, etc.).
                self._evict_cached_agent(session_key)
                # Off-loop + bounded: temporary-agent teardown can block on
                # subprocess/network/SQLite work.
                await self._cleanup_agent_resources_off_loop(
                    tmp_agent, context="manual compression"
                )
            return "\n".join(_manual_compression_reply_lines(summary, compressor, focus_topic))
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return t("gateway.compress.failed", error=e)

    async def _build_manual_compression_agent(self, session_id: str, model, runtime_kwargs: dict):
        """Build the throwaway AIAgent that performs a manual /compress rewrite of *session_id*."""
        from run_agent import AIAgent
        from gateway.run import _GATEWAY_HYGIENE_PLATFORM, _seed_hygiene_system_prompt

        # The manual compression helper runs outside the live session's fully initialized prompt
        # environment and _compress_context may persist its cached system prompt — restore the
        # exact live-session prompt so provider blocks are retained.
        session_row = None
        get_session = getattr(self._session_db, "get_session", None)
        if callable(get_session):
            try:
                session_row = await get_session(session_id)
            except Exception as exc:
                logger.warning(
                    "Manual compression could not restore the system prompt "
                    "for session %s: %s. Preserving an empty prompt so the "
                    "live turn rebuilds it with its configured providers.",
                    session_id,
                    exc,
                    exc_info=True,
                )

        # This agent performs a lossy rewrite. When compression.checkpoint_required is on, the
        # memory provider must be loaded so _compress_context() can write the pre-compression
        # checkpoint; otherwise keep the historical fast path (no provider init).
        from hermes_cli.config import load_config as _load_cfg
        from utils import is_truthy_value as _is_truthy

        _checkpoint_required = _is_truthy(
            ((_load_cfg() or {}).get("compression") or {}).get("checkpoint_required"),
            default=False,
        )
        tmp_agent = AIAgent(
            **runtime_kwargs,
            model=model,
            max_iterations=4,
            quiet_mode=True,
            skip_memory=not _checkpoint_required,
            enabled_toolsets=["memory"],
            session_id=session_id,
            session_db=getattr(self._session_db, "_db", self._session_db),
        )
        _seed_hygiene_system_prompt(tmp_agent, session_row)
        # Keep the real source platform during construction so external context engines bind
        # correctly. If compression has to rebuild the prompt, stamp that provider-less fallback
        # as stale for the next real gateway turn.
        tmp_agent.platform = _GATEWAY_HYGIENE_PLATFORM
        tmp_agent._print_fn = lambda *a, **kw: None
        # Prevent close() from ending the newly rotated session — the gateway session entry now
        # points at the new id and must remain open for the next user turn.
        tmp_agent._end_session_on_close = False
        return tmp_agent

    async def _persist_manual_compression(self, tmp_agent, session_entry, source, compressed) -> None:
        """Commit a manual /compress result to the session store.

        _compress_context either rotated (new continuation id — write compressed messages into the
        NEW session so the original stays searchable) or compacted in place (compression.in_place:
        same id, transcript replaced). Persist BEFORE repointing the live session: repoint first +
        failed DB write would leave the entry on an empty session while reporting success; a failed
        write is fatal so old history stays reachable. Only rewrite when rotation produced a NEW id:
        in-place compaction already archived + inserted rows and rewrite_transcript()
        (active_only=False) would DELETE the archived turns; an unchanged id without in-place means
        rotation FAILED and a rewrite would leave only the summary.
        """
        new_session_id = tmp_agent.session_id
        if new_session_id != session_entry.session_id:
            if not await self.async_session_store.rewrite_transcript(new_session_id, compressed):
                raise RuntimeError(
                    f"failed to persist compressed transcript for session {new_session_id}"
                )
            session_entry.session_id = new_session_id
            await self.async_session_store._save()
            await asyncio.to_thread(
                self._sync_telegram_topic_binding,
                source, session_entry, reason="compress-command",
            )
        elif not getattr(tmp_agent, "_last_compaction_in_place", False):
            logger.warning(
                "Manual /compress: session rotation did not occur "
                "(session_id unchanged) and in-place mode is off — "
                "preserving original transcript instead of overwriting "
                "it (#44794)."
            )
        # Reset stored token count — transcript changed, old value is stale
        await self.async_session_store.update_session(session_entry.session_key, last_prompt_tokens=0)

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic for Telegram DM user-managed topic sessions."""
        source = event.source
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return t("gateway.topic.not_telegram_dm")
        if not self._session_db:
            return self._session_db_unavailable_reply()

        # Authorization: /topic activates multi-session mode and mutates SQLite side tables.
        # Unauthorized senders (not in allowlist) must not be able to do that. Gateway routes
        # already authorize the message before reaching here, but defense in depth.
        auth_fn = getattr(self, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                if not auth_fn(source):
                    return t("gateway.topic.unauthorized")
            except Exception:
                logger.debug("Topic auth check failed", exc_info=True)

        args = event.get_command_args().strip()

        # /topic help — inline usage without leaving the bot.
        if args.lower() in {"help", "?", "-h", "--help"}:
            return self._telegram_topic_help_text()

        # /topic off — clean disable path so users don't have to edit the DB.
        if args.lower() in {"off", "disable", "stop"}:
            return await self._disable_telegram_topic_mode_for_chat(source)

        if args:
            if not source.thread_id:
                return t("gateway.topic.restore_needs_topic")
            return await self._restore_telegram_topic_session(event, args)

        capabilities = await self._get_telegram_topic_capabilities(source)
        if capabilities.get("checked"):
            if capabilities.get("has_topics_enabled") is False:
                # Debounce the BotFather screenshot: don't re-send on every
                # /topic while threads are still disabled.
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t("gateway.topic.topics_disabled")
            if capabilities.get("allows_users_to_create_topics") is False:
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t("gateway.topic.topics_user_disallowed")

        try:
            await self._session_db.enable_telegram_topic_mode(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                profile_name=self._telegram_topic_profile_name(source),
                has_topics_enabled=capabilities.get("has_topics_enabled"),
                allows_users_to_create_topics=capabilities.get("allows_users_to_create_topics"),
            )
        except Exception as exc:
            logger.exception("Failed to enable Telegram topic mode")
            return t("gateway.topic.enable_failed", error=exc)

        if not source.thread_id:
            await self._ensure_telegram_system_topic(source)

        if source.thread_id:
            try:
                binding = await self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    profile_name=self._telegram_topic_profile_name(source),
                )
            except Exception:
                logger.debug("Failed to read Telegram topic binding", exc_info=True)
                binding = None
            if binding:
                session_id = str(binding.get("session_id") or "")
                title = None
                try:
                    title = await self._session_db.get_session_title(session_id)
                except Exception:
                    title = None
                session_label = title or t("gateway.topic.untitled_session")
                return t(
                    "gateway.topic.bound_status",
                    label=session_label,
                    session_id=session_id,
                )
            return t("gateway.topic.thread_ready")

        return await self._telegram_topic_root_status_message(source)

    async def _handle_save_command(self, event: MessageEvent) -> str:
        """Handle /save — export the current session and send it as a document."""
        from hermes_cli.session_export import (
            SAVE_USAGE,
            default_save_filename,
            normalize_save_format,
            render_session_for_save,
        )

        parts = event.get_command_args().split()
        if not parts:
            return SAVE_USAGE
        redact = False
        if parts[-1].lower() in ("redact", "--redact"):
            redact = True
            parts = parts[:-1]
            if not parts:
                return SAVE_USAGE

        try:
            fmt = normalize_save_format(parts[0])
        except ValueError as e:
            return f"{e}\n\n{SAVE_USAGE}"

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            return "Session database not available."
        filename = parts[1] if len(parts) > 1 else default_save_filename(session_id, fmt)
        # The filename is echoed to the platform only — never trust path
        # separators from chat input.
        filename = os.path.basename(filename) or default_save_filename(session_id, fmt)

        # self._session_db is an AsyncSessionDB — every forwarded call is
        # offloaded to a thread and must be awaited.
        export_data = await self._session_db.export_session(session_id)
        if not export_data:
            return f"No stored messages found for this session ({session_id})."

        if redact:
            from hermes_cli.session_export_md import redact_session_data

            export_data = redact_session_data(export_data)

        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="hermes_save_")
        temp_path = os.path.join(temp_dir, filename)
        try:
            # Off-loop: rendering a long session and writing it to disk are CPU/disk-bound and scale
            # with transcript size (multi-MB for long sessions). Inline they stall every other chat
            # on the gateway event loop (Pattern A). One thread hop covers both.
            def _render_and_write() -> None:
                rendered = render_session_for_save(export_data, fmt)
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(rendered)

            await asyncio.to_thread(_render_and_write)

            adapter = self.get_adapter(source.platform)
            if adapter:
                await adapter.send_document(
                    chat_id=source.chat_id,
                    file_path=temp_path,
                    caption=f"Session export: {filename}",
                    file_name=filename,
                )
                return "Export complete."
            return "Platform adapter not found to send the document."
        except Exception as e:
            logger.warning("Session /save failed: %s", e)
            return f"Error exporting session: {e}"
        finally:
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            return self._session_db_unavailable_reply()

        # Ensure session exists in SQLite DB (it may only exist in session_store
        # if this is the first command in a new session)
        existing_title = await self._session_db.get_session_title(session_id)
        if existing_title is None:
            # Session doesn't exist in DB yet — create it
            try:
                await self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id,
                    # Persist the messaging origin so a later /resume of this
                    # titled-but-now-inactive session can prove it belongs to the
                    # caller's chat/thread (IDOR scoping).
                    chat_id=source.chat_id,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                )
            except Exception:
                pass  # Session might already exist, ignore errors

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                from hermes_state import SessionDB
                sanitized = SessionDB.sanitize_title(title_arg)
            except ValueError as e:
                return t("gateway.shared.warn_passthrough", error=e)
            if not sanitized:
                return t("gateway.title.empty_after_clean")
            # Set the title
            try:
                if await self._session_db.set_session_title(session_id, sanitized):
                    # Propagate the user-chosen title to the visible Telegram forum topic name too.
                    # Auto-generated titles already rename the topic; without this, /title only
                    # updated the DB title and the topic kept its auto-assigned name.
                    schedule_rename = getattr(
                        self, "_schedule_telegram_topic_title_rename", None
                    )
                    if callable(schedule_rename):
                        try:
                            await asyncio.to_thread(schedule_rename, source, session_id, sanitized)
                        except Exception:
                            logger.debug(
                                "Failed to rename Telegram topic from /title",
                                exc_info=True,
                            )
                    return t("gateway.title.set_to", title=sanitized)
                else:
                    return t("gateway.title.not_found")
            except ValueError as e:
                return t("gateway.shared.warn_passthrough", error=e)
        else:
            # Show the current title and session ID
            title = await self._session_db.get_session_title(session_id)
            if title:
                return t("gateway.title.current_with_title", session_id=session_id, title=title)
            else:
                return t("gateway.title.current_no_title", session_id=session_id)

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — list or switch to a previous session."""
        if not self._session_db:
            return self._session_db_unavailable_reply()

        source = await asyncio.to_thread(
            self._normalize_source_for_session_key, event.source
        )
        session_key = self._session_key_for_source(source)
        raw_args = event.get_command_args().strip()
        try:
            parts = shlex.split(raw_args)
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)
        allow_all = "--all" in parts
        allow_cross_room = "--cross-room" in parts
        name = " ".join(p for p in parts if p not in {"--all", "--cross-room"}).strip()

        # Strip common outer brackets/quotes users may type literally from the
        # usage hint (e.g. ``/resume <abc123>``). Mirrors the CLI behavior.
        if len(name) >= 2 and (
            (name[0] == "<" and name[-1] == ">")
            or (name[0] == "[" and name[-1] == "]")
            or (name[0] == '"' and name[-1] == '"')
            or (name[0] == "'" and name[-1] == "'")
        ):
            name = name[1:-1].strip()

        async def _list_titled_sessions() -> list[dict]:
            """Titled sessions visible to the caller (origin-scoped unless admin ``--all``)."""
            user_source = source.platform.value if source.platform else None
            widen = allow_all and self._resume_caller_is_admin(source)
            sessions = await self._session_db.list_sessions_rich(
                source=user_source,
                session_key=None if widen else session_key,
                limit=10,
            )
            titled = [s for s in sessions if s.get("title")][:10]
            return [s for s in titled if await self._resume_row_visible(source, s, allow_all)]

        if not name:
            # List recent titled sessions for this user/platform
            try:
                titled = await _list_titled_sessions()
                return self._resume_listing_reply(source, titled, allow_all)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return t("gateway.resume.list_failed", error=e)

        # Resolve a numbered choice or a title to a session ID.
        if name.isdigit():
            try:
                titled = await _list_titled_sessions()
            except Exception as e:
                logger.debug("Failed to list titled sessions for numeric resume: %s", e)
                return t("gateway.resume.list_failed", error=e)
            index = int(name)
            if index < 1 or index > len(titled):
                return t("gateway.resume.out_of_range", index=index)
            target = titled[index - 1]
            target_id = target.get("id")
            name = target.get("title") or name
        else:
            # Try direct session ID lookup first (so `/resume <session_id>`
            # works in the gateway, not just `/resume <title>`).
            session = await self._session_db.get_session(name)
            if session:
                target_id = session["id"]
            else:
                target_id = await self._session_db.resolve_session_by_title(name)
        if not target_id:
            return t("gateway.resume.not_found", name=name)
        # Compression creates child continuations that hold the live transcript.
        # Follow that chain so gateway /resume matches CLI behavior (#15000).
        try:
            target_id = await self._session_db.resolve_resume_session_id(target_id)
        except Exception as e:
            logger.debug("Failed to resolve resume continuation for %s: %s", target_id, e)

        if source.platform == Platform.MATRIX:
            target_origin = self._gateway_session_origin_for_id(target_id)
            if not self._same_matrix_room(source, target_origin) and not allow_cross_room:
                if target_origin is None:
                    return t("gateway.resume.matrix_blocked_no_origin", name=name)
                return t(
                    "gateway.resume.matrix_blocked_other_room",
                    room=target_origin.chat_name or target_origin.chat_id,
                    name=name,
                )
        elif not await self._resume_target_allowed(
            source, target_id, allow_override=(allow_all or allow_cross_room)
        ):
            # IDOR guard: a session id/title is a routing handle, not authority. Bind /resume to the
            # caller's own platform/user/chat on every non-Matrix adapter so one user can't attach
            # to another's persisted transcript.
            return t("gateway.resume.blocked_not_owner", name=name)

        # Check if already on that session
        current_entry = await self.async_session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return t("gateway.resume.already_on", name=name)

        # Clear any running agent for this session key
        self._release_running_agent_state(session_key)

        # Switch the session entry to point at the old session
        new_entry = await self.async_session_store.switch_session(session_key, target_id)
        if not new_entry:
            return t("gateway.resume.switch_failed")

        # Conversation boundary: clear ALL conversation-scoped per-session state (model/reasoning
        # overrides #10702, one-turn restores, model notes, last-resolved cache #58403, /queue
        # overflow) + security state in one funnel call.
        self._clear_conversation_scope(session_key, reason="resume")

        # Evict any cached agent for this session so the next message rebuilds with the correct
        # session_id end-to-end — mirrors /branch and /reset. Otherwise the cached AIAgent (and its
        # memory provider, which cached _session_id at initialize()) keeps writing to the wrong session.
        self._evict_cached_agent(session_key)

        # Get the title for confirmation
        title = await self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = await self.async_session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

        if source.platform == Platform.MATRIX and allow_cross_room:
            return t(
                "gateway.resume.matrix_cross_room_success",
                title=title,
                room=source.chat_name or source.chat_id,
                msg_part=msg_part,
            )
        if not msg_count:
            return t("gateway.resume.resumed_no_count", title=title)
        if msg_count == 1:
            return t("gateway.resume.resumed_one", title=title, count=msg_count)
        return t("gateway.resume.resumed_many", title=title, count=msg_count)

    def _resume_listing_reply(self, source, titled: list[dict], allow_all: bool) -> str:
        """Numbered /resume list. A non-admin ``--all`` silently falls back to same-origin scoping;
        say so instead of rendering an unexplained narrower list (sibling of the /sessions notice)."""
        scope_note = (
            t("gateway.resume.all_requires_admin")
            if allow_all and not self._resume_caller_is_admin(source)
            else None
        )
        if not titled:
            if source.platform == Platform.MATRIX and not allow_all:
                return t("gateway.resume.matrix_no_named_sessions")
            base = t("gateway.resume.no_named_sessions")
            return f"{base}\n{scope_note}" if scope_note else base
        lines = [t("gateway.resume.list_header")]
        for idx, s in enumerate(titled[:10], start=1):
            title = s["title"]
            if source.platform == Platform.MATRIX and allow_all:
                origin = self._gateway_session_origin_for_id(str(s.get("id") or ""))
                if origin:
                    title = f"{title} — {origin.chat_name or origin.chat_id}"
            preview = s.get("preview", "")[:40]
            preview_part = t("gateway.resume.list_preview_suffix", preview=preview) if preview else ""
            lines.append(t("gateway.resume.list_item_numbered", index=idx, title=title, preview_part=preview_part))
        if scope_note:
            lines.append(scope_note)
        lines.append(t("gateway.resume.list_footer_numbered"))
        return "\n".join(lines)

    async def _handle_sessions_command(self, event: MessageEvent) -> str:
        """Handle /sessions — list previous sessions for gateway chats."""
        if not self._session_db:
            return self._session_db_unavailable_reply()

        from hermes_cli.session_listing import (
            format_gateway_session_listing,
            parse_session_listing_args,
            query_session_listing,
        )

        raw_args = event.get_command_args().strip()
        try:
            include_all, include_unnamed, target, search_query = (
                parse_session_listing_args(raw_args)
            )
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)

        if search_query == "":
            return "Usage: `/sessions search <query>`"

        if target:
            resume_event = dataclasses.replace(event, text=f"/resume {target}")
            return await self._handle_resume_command(resume_event)

        source = await asyncio.to_thread(
            self._normalize_source_for_session_key, event.source
        )
        session_key = self._session_key_for_source(source)

        # A cross-origin listing (`/sessions all`) is honored only for an admin, mirroring the
        # `/resume --all` override. `all` is just a parsed user argument; ungated, any caller could
        # enumerate other origins' session ids/titles/previews — the enumeration half of the IDOR.
        cross_origin = include_all and self._resume_caller_is_admin(source)
        # Don't silently no-op a requested widening: a non-admin `/sessions all`
        # used to render the same scoped list with zero feedback, which reads
        # as "my session vanished" (community report, Aug 2026).
        scope_notice = None
        if include_all and not cross_origin:
            scope_notice = (
                "_Note: `all` (cross-chat listing) requires a configured admin; "
                "showing this chat's sessions only._"
            )
        current_entry = await self.async_session_store.get_or_create_session(source)
        rows = await asyncio.to_thread(
            query_session_listing,
            getattr(self._session_db, "_db", self._session_db),
            source=source.platform.value if source.platform else None,
            session_key=None if cross_origin else session_key,
            current_session_id=current_entry.session_id,
            include_current_session=True,
            include_all_sources=cross_origin,
            include_unnamed=include_unnamed,
            search_query=search_query,
            # Search filters at SQL level, so over-fetch before the visibility
            # cut: origin-invisible matches would otherwise consume the page.
            limit=50 if search_query else 10,
            exclude_sources=["tool"],
        )
        if not cross_origin:
            # Scope the listing to the caller's own origin on every adapter so
            # session ids/previews from other users/rooms aren't enumerable.
            rows = [
                row for row in rows
                if await self._resume_row_visible(source, row, allow_all=False)
            ]
        rows = rows[:10]
        if search_query:
            title = f"Sessions matching “{search_query}”"
        else:
            title = "Sessions" if include_unnamed else "Named Sessions"
        return format_gateway_session_listing(
            rows,
            include_source=cross_origin,
            title=title,
            notice=scope_notice,
        )

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into a new independent copy so the
        user can explore a different approach without losing the original.
        """
        import uuid as _uuid

        if not self._session_db:
            return self._session_db_unavailable_reply()

        source = event.source
        session_key = self._session_key_for_source(source)

        # Load the current session and its transcript
        current_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(current_entry.session_id)
        if not history:
            return t("gateway.branch.no_conversation")

        branch_name = event.get_command_args().strip()

        # Generate the new session ID
        from datetime import datetime as _dt
        now = _dt.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = _uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            current_title = await self._session_db.get_session_title(current_entry.session_id)
            base = current_title or "branch"
            branch_title = await self._session_db.get_next_title_in_lineage(base)

        parent_session_id = current_entry.session_id

        # Serialize the parent's full origin (same shape as the reset path's db_create_kwargs in
        # gateway/session.py, #82633) so the branch row carries complete identity from birth. Prefer
        # the live entry's origin (it may hold richer metadata than the triggering event's source).
        _branch_origin = current_entry.origin or source
        _branch_origin_json = None
        if _branch_origin is not None:
            try:
                import json as _json

                _branch_origin_json = _json.dumps(_branch_origin.to_dict())
            except Exception:
                _branch_origin_json = None

        # Create the new session with parent link. Persist a stable ``_branched_from`` marker in
        # model_config so list_sessions_rich() keeps the branch visible in /resume and /sessions
        # even after the parent is reopened and re-ended with a different end_reason.
        try:
            await self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                model_config={"_branched_from": parent_session_id},
                parent_session_id=parent_session_id,
                # Forward ALL gateway routing columns at CREATE time: otherwise they're NULL until
                # switch_session() calls _record_gateway_session_peer(), and a crash in between (each
                # append_message is best-effort) leaves the branch unroutable — by chat/thread lookup
                # and by /resume's IDOR guard. user_id feeds the full-peer-tuple fallback lookup;
                # origin_json/display_name complete the identity (same shape as session.py's reset
                # path) so state.db consumers see a fully formed row with no backfill gap.
                user_id=source.user_id,
                session_key=session_key,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
                thread_id=source.thread_id,
                origin_json=_branch_origin_json,
                display_name=current_entry.display_name,
            )
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return t("gateway.branch.create_failed", error=e)

        # Copy conversation history to the new session in bounded-chunk transactions: one txn per
        # row was the removed write-amplification pattern, and a history can be hundreds of rows.
        # Best-effort like the old loop — a failed copy still yields a usable (partial) branch.
        try:
            await self._session_db.append_messages_batch(
                new_session_id,
                [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content"),
                        "tool_name": msg.get("tool_name") or msg.get("name"),
                        "tool_calls": msg.get("tool_calls"),
                        "tool_call_id": msg.get("tool_call_id"),
                        "finish_reason": msg.get("finish_reason"),
                        "reasoning": msg.get("reasoning"),
                        "reasoning_content": msg.get("reasoning_content"),
                        "reasoning_details": msg.get("reasoning_details"),
                        "codex_reasoning_items": msg.get("codex_reasoning_items"),
                        "codex_message_items": msg.get("codex_message_items"),
                        # Keep the api_content sidecar so the branch's first turn
                        # replays the parent's exact wire bytes (warm provider
                        # prompt cache) instead of a full cold prefill.
                        "api_content": extract_api_content_sidecar(msg),
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in history
                ],
                chunk_rows=500,
            )
        except Exception:
            pass  # Best-effort copy

        # Set title
        with contextlib.suppress(Exception):
            await self._session_db.set_session_title(new_session_id, branch_title)

        # Switch the session store entry to the new session
        new_entry = await self.async_session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return t("gateway.branch.switch_failed")
        self._clear_session_boundary_security_state(session_key)

        # Evict any cached agent for this session
        self._evict_cached_agent(session_key)

        msg_count = len([m for m in history if m.get("role") == "user"])
        key = "gateway.branch.branched_one" if msg_count == 1 else "gateway.branch.branched_many"
        return t(key, title=branch_title, count=msg_count, parent=parent_session_id, new=new_session_id)

    async def _handle_topup_command(self, event: MessageEvent) -> str:
        """Handle /topup -- show the Nous balance and hand off to the portal.

        Does NOT charge, confirm, or track payment — that happens in the browser; the next /topup
        shows the new balance. Fetched off the event loop; fail-open.
        """
        from agent.account_usage import build_credits_view

        try:
            view = await asyncio.to_thread(build_credits_view, markdown=True)
        except Exception:
            view = None

        if view is None or not view.logged_in:
            return t("gateway.credits.not_logged_in")

        lines: list[str] = ["💳 **Nous balance**"]
        for line in view.balance_lines:
            if line.lstrip().startswith("📈"):
                continue  # drop the helper's header; we print our own
            lines.append(line)
        if view.identity_line:
            lines.append("")
            lines.append(view.identity_line)
        if view.topup_url:
            lines.append("")
            lines.append(f"Manage billing on the portal: {view.topup_url}")
            lines.append("Top up and manage billing in the browser — your balance updates here after.")
        return "\n".join(lines)

    def _context_breakdown_block(self, agent, source, expanded: bool) -> list[str]:
        """Render the /context per-category block (plain text, no grid).

        Estimated (chars/4), same engine as /usage. Runs in a thread; returns [] and never raises.
        """
        try:
            from agent.context_breakdown import (
                compute_context_details,
                compute_session_context_breakdown,
                render_context_breakdown_lines,
            )

            history: list[dict] = []
            try:
                entry = self.session_store.get_or_create_session(source)
                history = self.session_store.load_transcript(entry.session_id) or []
            except Exception:
                history = []

            payload = compute_session_context_breakdown(agent, history)
            if not (payload.get("categories") or []):
                return []

            details = None
            if expanded:
                try:
                    details = compute_context_details(agent)
                except Exception:
                    details = {"skills": [], "toolsets": []}

            return render_context_breakdown_lines(payload, details=details, grid=False)
        except Exception:
            return []

    def _context_breakdown_lines(self, agent, source) -> list[str]:
        """Render the per-category context breakdown for /usage.

        Estimated (chars/4). Returns [] and never raises so /usage stays robust.
        """
        try:
            from agent.context_breakdown import compute_session_context_breakdown

            history: list[dict] = []
            try:
                entry = self.session_store.get_or_create_session(source)
                history = self.session_store.load_transcript(entry.session_id) or []
            except Exception:
                history = []

            payload = compute_session_context_breakdown(agent, history)
            categories = payload.get("categories") or []
            if not categories:
                return []

            total = payload.get("estimated_total") or 0
            out = [t("gateway.usage.breakdown_header")]
            for cat in categories:
                tokens = int(cat.get("tokens") or 0)
                if tokens <= 0:
                    continue
                cat_id = str(cat.get("id") or "")
                label = t(f"gateway.usage.breakdown_cat_{cat_id}")
                # Missing key → t() echoes the key back; fall back to the
                # English label the engine already provides.
                if label.endswith(f"breakdown_cat_{cat_id}"):
                    label = str(cat.get("label") or cat_id)
                pct = round(tokens / total * 100) if total else 0
                out.append(
                    t("gateway.usage.breakdown_line", label=label, count=f"{tokens:,}", pct=pct)
                )
            return out if len(out) > 1 else []
        except Exception:
            return []

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Checks both _running_agents (mid-turn) and _agent_cache (between turns) so details are
        available whenever the user asks.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        # `/usage reset [--force]` — redeem one banked Codex rate-limit reset
        # credit. Parsed before the display path so it never mixes with the
        # stats rendering below.
        raw_args = event.get_command_args().strip()
        args = [a.lower() for a in raw_args.split()] if raw_args else []
        wants_reset = bool(args) and args[0] == "reset"
        if args and not wants_reset:
            return t("gateway.usage.unknown_subcommand", args=raw_args)

        # Running agent first (mid-turn), then cached agent (between turns).
        agent = self._resident_agent_for(session_key)

        # Resolve provider/base_url/api_key for the account-usage fetch. Prefer the live agent; fall
        # back to persisted billing data on the SessionDB row so `/usage` still returns account info
        # between turns when no agent is resident.
        provider = getattr(agent, "provider", None) if agent else None
        base_url = getattr(agent, "base_url", None) if agent else None
        api_key = getattr(agent, "api_key", None) if agent else None
        if not provider and getattr(self, "_session_db", None) is not None:
            provider, base_url = await self._persisted_billing_route(source)

        if wants_reset:
            normalized_provider = str(provider or "").strip().lower()
            if normalized_provider != "openai-codex":
                return t("gateway.usage.reset_wrong_provider")
            force = "--force" in args[1:]
            from agent.account_usage import redeem_codex_reset_credit

            result = await asyncio.to_thread(
                redeem_codex_reset_credit,
                base_url=base_url,
                api_key=api_key,
                force=force,
            )
            return result.message

        # Fetch account usage off the event loop so slow provider APIs don't
        # block the gateway. Failures are non-fatal -- account_lines stays [].
        account_lines: list[str] = []
        credits_lines: list[str] = []
        if provider:
            try:
                account_snapshot = await asyncio.to_thread(
                    fetch_account_usage,
                    provider,
                    base_url=base_url,
                    api_key=api_key,
                )
            except Exception:
                account_snapshot = None
            if account_snapshot:
                account_lines = render_account_usage_lines(account_snapshot, markdown=True)

        # ── Nous credits magnitudes + monthly-grant % gauge ─────────────
        # Shared with CLI/TUI via nous_credits_lines(); run off the event loop. Gates on "a Nous
        # account is logged in" — NOT the inference provider, NOT under `if provider:` — so a Nous
        # user inferring elsewhere still sees a balance. No recovery trigger; fail-open.
        try:
            from agent.account_usage import nous_credits_lines

            credits_lines = await asyncio.to_thread(nous_credits_lines, markdown=True)
        except Exception:
            credits_lines = []  # fail-open: never break /usage

        def _with_account_blocks(lines: list[str]) -> str:
            # Each block is preceded by a blank divider only when something precedes it.
            for block in (account_lines, credits_lines):
                if block:
                    if lines:
                        lines.append("")
                    lines.extend(block)
            return "\n".join(lines)

        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = _usage_agent_stats_lines(agent)
            # Per-category context breakdown (estimated — chars/4 heuristic). Same engine the
            # desktop popover uses. The system prompt / tools / skills / memory slices read off the
            # live agent; the conversation slice is estimated from the session transcript.
            breakdown_lines = await asyncio.to_thread(self._context_breakdown_lines, agent, source)
            if breakdown_lines:
                lines.append("")
                lines.extend(breakdown_lines)
            return _with_account_blocks(lines)

        # No agent at all -- check session history for a rough count
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            return _with_account_blocks([
                t("gateway.usage.header_session_info"),
                t("gateway.usage.label_messages", count=len(msgs)),
                t("gateway.usage.label_estimated_context", count=f"{approx:,}"),
                t("gateway.usage.detailed_after_first"),
            ])
        if account_lines or credits_lines:
            return _with_account_blocks([])
        return t("gateway.usage.no_data")

    async def _persisted_billing_route(self, source):
        """``(provider, base_url)`` from the SessionDB row / dominant route when no agent is resident."""
        try:
            entry = await self.async_session_store.get_or_create_session(source)
            persisted = await self._session_db.get_session(entry.session_id) or {}
            route = await self._session_db.get_dominant_session_model_route(entry.session_id)
            persisted_route = route if isinstance(route, dict) else {}
        except Exception:
            persisted = {}
            persisted_route = {}
        if persisted_route.get("billing_provider"):
            return persisted_route["billing_provider"], persisted_route.get("billing_base_url")
        return persisted.get("billing_provider"), persisted.get("billing_base_url")

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        args = event.get_command_args().strip()

        # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
        args = re.sub(r'[\u2012\u2013\u2014\u2015](days|source)', r'--\1', args)

        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return t("gateway.insights.invalid_days", value=parts[i + 1])
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import get_shared_session_db
            from agent.insights import InsightsEngine

            def _run_insights():
                db = get_shared_session_db()
                try:
                    engine = InsightsEngine(db)
                    report = engine.generate(days=days, source=source)
                    result = engine.format_gateway(report)
                    return result
                finally:
                    from hermes_state import release_or_close
                    release_or_close(db)

            # Not a bare hop: ``SessionDB()`` resolves ``get_hermes_home()`` at call time, which is
            # a contextvar set by ``_profile_runtime_scope``; a default-executor hop starts with an
            # EMPTY context and would read the DEFAULT profile's state.db.
            return await self._run_in_executor_with_context(_run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return t("gateway.insights.error", error=e)

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reload-mcp — reconnect MCP servers and rebuild the cached agent.

        Reloading invalidates the provider prompt cache (tool schemas live in the system prompt),
        so it routes through slash-confirm; "Always Approve" persists
        ``approvals.mcp_reload_confirm: false``.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        # Read the gate fresh from disk so a prior "always" click takes
        # effect on the next invocation without restarting the gateway.
        user_config = self._read_user_config()
        approvals = user_config.get("approvals") if isinstance(user_config, dict) else None
        confirm_required = True
        if isinstance(approvals, dict):
            confirm_required = bool(approvals.get("mcp_reload_confirm", True))

        if not confirm_required:
            return await self._execute_mcp_reload(event)

        # Route through slash-confirm. The primitive sends the prompt and stores the resume handler;
        # the button/text response triggers ``_resolve_slash_confirm`` which invokes the handler
        # with the chosen outcome.
        async def _on_confirm(choice: str) -> Optional[str]:
            if choice == "cancel":
                return t("gateway.reload_mcp.cancelled")
            if choice == "always":
                # Persist the opt-out and run the reload.
                try:
                    from cli import save_config_value
                    save_config_value("approvals.mcp_reload_confirm", False)
                    logger.info(
                        "User opted out of /reload-mcp confirmation (session=%s)",
                        session_key,
                    )
                except Exception as exc:
                    logger.warning("Failed to persist mcp_reload_confirm=false: %s", exc)
            # once / always → run the reload
            result = await self._execute_mcp_reload(event)
            if choice == "always":
                return f"{result}\n\n" + t("gateway.reload_mcp.always_followup")
            return result

        prompt_message = t("gateway.reload_mcp.confirm_prompt")
        return await self._request_slash_confirm(
            event=event,
            command="reload-mcp",
            title="/reload-mcp",
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _handle_reload_skills_command(self, event: MessageEvent) -> str:
        """Handle /reload-skills — rescan skills dir, queue a note for next turn.

        Skills are invoked at runtime, not baked into the system prompt, so this does NOT clear the
        prompt cache. Added/removed skills go into ``_pending_skills_reload_notes[session_key]``,
        prepended to the NEXT user message — nothing out-of-band, so alternation is preserved.
        """
        try:
            from agent.skill_commands import reload_skills

            # _run_in_executor_with_context, not a bare hop: the rescan walks
            # get_hermes_home()/skills, a contextvar override under multiplex.
            result = await self._run_in_executor_with_context(reload_skills)
            added = result.get("added", [])      # [{"name", "description"}, ...]
            removed = result.get("removed", [])  # [{"name", "description"}, ...]
            total = result.get("total", 0)

            # Let adapters refresh platform-side state that cached the skill list at startup (today:
            # Discord /skill autocomplete — otherwise new skills stay invisible and deleted ones
            # error). Adapters without refresh_skill_group are skipped; the in-process reload suffices.
            for adapter in list(self.adapters.values()):
                refresh = getattr(adapter, "refresh_skill_group", None)
                if not callable(refresh):
                    continue
                try:
                    maybe = refresh()
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception as exc:
                    logger.warning(
                        "Adapter %s refresh_skill_group raised: %s",
                        getattr(adapter, "name", adapter), exc,
                    )

            lines = [t("gateway.reload_skills.header")]
            if not added and not removed:
                lines.append(t("gateway.reload_skills.no_new"))
                lines.append(t("gateway.reload_skills.total", count=total))
                return "\n".join(lines)

            def _fmt_line(item: dict) -> str:
                nm = item.get("name", "")
                desc = item.get("description", "")
                if desc:
                    return t("gateway.reload_skills.item_with_desc", name=nm, desc=desc)
                return t("gateway.reload_skills.item_no_desc", name=nm)

            # Queue a one-shot note for the next user turn in this session too. Format matches how
            # the system prompt renders pre-existing skills (``    - name: description``) so the
            # model reads the diff in the same shape as its original skill catalog.
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            for i18n_key, note_header, items in (
                ("gateway.reload_skills.added_header", "Added Skills:", added),
                ("gateway.reload_skills.removed_header", "Removed Skills:", removed),
            ):
                if items:
                    lines.append(t(i18n_key))
                    lines.extend(_fmt_line(item) for item in items)
                    sections.extend(["", note_header])
                    sections.extend(_fmt_line(item) for item in items)
            lines.append(t("gateway.reload_skills.total", count=total))
            sections.append("")
            sections.append("Use skills_list to see the updated catalog.]")
            note = "\n".join(sections)

            session_key = self._session_key_for_source(event.source)
            if not hasattr(self, "_pending_skills_reload_notes"):
                self._pending_skills_reload_notes = {}
            if session_key:
                self._pending_skills_reload_notes[session_key] = note

            return "\n".join(lines)

        except Exception as e:
            logger.warning("Skills reload failed: %s", e)
            return t("gateway.reload_skills.failed", error=e)

    async def _handle_bundles_command(self, event: MessageEvent) -> str:
        """Handle /bundles — list installed skill bundles (mirrors the CLI handler).

        Bundles are loaded by invoking their own ``/<slug>`` command, not by this one.
        """
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("bundles", CommandContext(surface="gateway"))
        if "error" in reply.data:
            logger.warning("Bundles command unavailable: %s", reply.data["error"])
            return reply.text

        bundles = reply.data["bundles"]
        if not bundles:
            return (
                "No skill bundles installed.\n"
                "Create one on the host with:\n"
                "  `hermes bundles create <name> --skill <s1> --skill <s2>`\n"
                f"Directory: `{reply.data['dir']}`"
            )

        lines = [f"**Skill Bundles** ({len(bundles)} installed):", ""]
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or f"Load {skill_count} skills"
            lines.append(
                f"• `/{info['slug']}` — {desc} _({skill_count} skills)_"
            )
            for s in info.get("skills", []):
                lines.append(f"    · {s}")
        lines.append("")
        lines.append("Invoke a bundle with `/<slug>` to load all its skills.")
        return "\n".join(lines)

    def _blocking_approval_or_stale(self, event: MessageEvent, stale_key: str, none_key: str):
        """``(session_key, None)`` when an agent thread is blocked on approval, else the reply to send.

        A pending-approvals entry with no blocked thread is a stale prompt: drop it and say so.
        """
        from tools.approval import has_blocking_approval

        session_key = self._session_key_for_source(event.source)
        if has_blocking_approval(session_key):
            return session_key, None
        if session_key in self._pending_approvals:
            self._pending_approvals.pop(session_key)
            return session_key, t(stale_key)
        return session_key, t(none_key)

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        Agent threads block inside tools/approval.py; signalling the event resumes them so the
        command executes inline — same flow as the CLI's synchronous approval.
        """
        from tools.approval import resolve_gateway_approval

        session_key, stale = self._blocking_approval_or_stale(event, "gateway.approval_expired", "gateway.approve.no_pending")
        if stale:
            return stale

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in {"always", "permanent", "permanently"} for a in remaining):
            choice = "always"
        elif any(a in {"session", "ses"} for a in remaining):
            choice = "session"
        else:
            choice = "once"

        count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        if not count:
            return t("gateway.approve.no_pending")

        plural = "plural" if count > 1 else "singular"
        confirmation_text = t(f"gateway.approve.{choice}_{plural}", count=count)
        logger.info("User approved %d dangerous command(s) via /approve (%s)", count, choice)
        return await self._deliver_approval_confirmation(event, confirmation_text, "approve")

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked thread(s) with a 'deny' result so they get a definitive BLOCKED message,
        as in the CLI. ``/deny`` denies the oldest; ``/deny all`` denies everything.
        """
        from tools.approval import resolve_gateway_approval

        session_key, stale = self._blocking_approval_or_stale(event, "gateway.deny.stale", "gateway.deny.no_pending")
        if stale:
            return stale

        # Parse args: a leading "all" token denies every pending command;
        # anything after it (or the whole arg string when "all" is absent) is
        # captured verbatim as the optional deny reason relayed to the agent.
        raw_args = event.get_command_args().strip()
        tokens = raw_args.split()
        resolve_all = bool(tokens) and tokens[0].lower() == "all"
        reason = raw_args[len(tokens[0]):].strip() if resolve_all else raw_args
        # Cap to a sane one-liner; the agent only needs a short hint.
        if reason:
            reason = reason[:280].strip()

        count = resolve_gateway_approval(
            session_key, "deny", resolve_all=resolve_all,
            reason=reason or None,
        )
        if not count:
            return t("gateway.deny.no_pending")

        logger.info(
            "User denied %d dangerous command(s) via /deny%s",
            count, " (with reason)" if reason else "",
        )
        if reason:
            if count > 1:
                confirmation_text = t("gateway.deny.denied_reason_plural", count=count, reason=reason)
            else:
                confirmation_text = t("gateway.deny.denied_reason_singular", reason=reason)
        elif count > 1:
            confirmation_text = t("gateway.deny.denied_plural", count=count)
        else:
            confirmation_text = t("gateway.deny.denied_singular")
        return await self._deliver_approval_confirmation(event, confirmation_text, "deny")

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report (summary only) and return paste URLs.

        Uploads ONLY the summary (system info + log tails), never full logs, to protect privacy;
        use ``hermes debug share`` from the CLI for full uploads.
        """
        from hermes_cli.debug import (
            _capture_dump, collect_debug_report,
            upload_to_pastebin, _schedule_auto_delete,
            _GATEWAY_PRIVACY_NOTICE, _best_effort_sweep_expired_pastes,
        )

        # Run blocking I/O (dump capture, log reads, uploads) in a thread.
        def _collect_and_upload():
            _best_effort_sweep_expired_pastes()
            dump_text = _capture_dump()
            report = collect_debug_report(log_lines=200, dump_text=dump_text)

            urls = {}
            try:
                urls["Report"] = upload_to_pastebin(report)
            except Exception as exc:
                return t("gateway.debug.upload_failed", error=exc)

            # Schedule auto-deletion after 6 hours
            _schedule_auto_delete(list(urls.values()))

            lines = [_GATEWAY_PRIVACY_NOTICE, "", t("gateway.debug.header"), ""]
            label_width = max(len(k) for k in urls)
            for label, url in urls.items():
                lines.append(f"`{label:<{label_width}}`  {url}")

            lines.append("")
            lines.append(t("gateway.debug.auto_delete"))
            lines.append(t("gateway.debug.full_logs_hint"))
            lines.append(t("gateway.debug.share_hint"))
            return "\n".join(lines)

        # _run_in_executor_with_context, not a bare hop: this collects the profile's logs/config off
        # ``get_hermes_home()`` and uploads them to a public paste. Losing the contextvar override
        # would publish the DEFAULT profile's diagnostics from another profile's chat.
        return await self._run_in_executor_with_context(_collect_and_upload)

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` detached (``setsid``) so it survives the gateway restart it may
        trigger; marker files let this or the next gateway process notify the user on completion.
        """
        from gateway.run import _hermes_home, _resolve_hermes_bin
        import json
        import shutil
        import subprocess
        from datetime import datetime
        from hermes_cli.config import is_managed, format_managed_message

        # Block non-messaging platforms (API server, webhooks, ACP)
        platform = event.source.platform
        _allowed = self._UPDATE_ALLOWED_PLATFORMS
        # Plugin platforms with allow_update_command=True are also allowed
        if platform not in _allowed:
            try:
                from gateway.platform_registry import platform_registry
                entry = platform_registry.get(platform.value)
                if not entry or not entry.allow_update_command:
                    return t("gateway.update.platform_not_messaging")
            except Exception:
                return t("gateway.update.platform_not_messaging")

        if is_managed():
            return f"✗ {format_managed_message('update Hermes Agent')}"

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return t("gateway.update.not_git_repo")

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            return t("gateway.update.hermes_cmd_not_found")

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        session_key = self._session_key_for_source(event.source)
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "chat_type": event.source.chat_type,
            "user_id": event.source.user_id,
            "session_key": session_key,
            "timestamp": datetime.now().isoformat(),
        }
        if event.source.thread_id:
            pending["thread_id"] = event.source.thread_id
        if event.message_id:
            pending["message_id"] = event.message_id
        _tmp_pending = pending_path.with_suffix(".tmp")
        _tmp_pending.write_text(json.dumps(pending), encoding="utf-8")
        _tmp_pending.replace(pending_path)
        exit_code_path.unlink(missing_ok=True)

        # Spawn `hermes update --gateway` detached (setsid: portable, works where systemd-run --user
        # lacks a D-Bus session) so it survives gateway restart. --gateway enables file-based IPC
        # for interactive prompts so the gateway forwards them instead of skipping. PYTHONUNBUFFERED
        # lets the gateway stream output in near-real-time.
        # Windows: no setsid chain — an inline Python helper via sys.executable runs the command,
        # redirects both outputs to the same files, and writes the exit code.
        try:
            if sys.platform == "win32":
                import textwrap
                from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

                # Invoke the updater as a module under this interpreter rather than through
                # hermes_cmd (venv\Scripts\hermes.exe): the shim launcher holds its own file open
                # for the whole run, and the update has to replace it.
                helper = textwrap.dedent(
                    """
                    import os, subprocess, sys
                    output_path = sys.argv[1]
                    exit_code_path = sys.argv[2]
                    cmd = sys.argv[3:]
                    env = dict(os.environ)
                    env["PYTHONUNBUFFERED"] = "1"
                    with open(output_path, "wb") as f:
                        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                        rc = proc.wait(timeout=3600)
                    with open(exit_code_path, "w", encoding="utf-8") as f:
                        f.write(str(rc))
                    """
                ).strip()
                subprocess.Popen(
                    [
                        sys.executable, "-c", helper,
                        str(output_path), str(exit_code_path),
                        sys.executable, "-m", "hermes_cli.main",
                        "update", "--gateway",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **windows_detach_popen_kwargs(),
                )
            else:
                hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
                update_cmd = (
                    f"PYTHONUNBUFFERED=1 {hermes_cmd_str} update --gateway"
                    f" > {shlex.quote(str(output_path))} 2>&1; "
                    # Avoid `status=$?`: `status` is read-only in zsh and this template is reused in
                    # macOS/zsh operator wrappers, so keep it zsh-safe even though bash runs it here.
                    f"rc=$?; printf '%s' \"$rc\" > {shlex.quote(str(exit_code_path))}"
                )
                setsid_bin = shutil.which("setsid")
                if setsid_bin:
                    # Preferred: setsid creates a new session, fully detached
                    subprocess.Popen(
                        [setsid_bin, "bash", "-c", update_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                else:
                    # Fallback: start_new_session=True calls os.setsid() in child
                    subprocess.Popen(
                        ["bash", "-c", update_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)
            return t("gateway.update.start_failed", error=e)

        self._schedule_update_notification_watch()
        return t("gateway.update.starting")
