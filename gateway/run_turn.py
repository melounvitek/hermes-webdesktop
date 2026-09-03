"""Agent-turn execution (_handle_message_with_agent, _run_agent*, proxy path, background tasks, MCP reload) for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import dataclasses
import inspect
import json
import os
import queue
import threading
import time
from agent.i18n import t
from contextlib import suppress
from contextvars import copy_context
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.session import (
    SessionSource,
    TranscriptReadError,
    _session_key_namespace,
    build_channel_continuity_note,
    build_session_context,
)
from gateway.turn_context import TurnContext
from gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from hermes_constants import get_hermes_home_override
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import base_url_hostname

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTurnMixin:
    """Agent-turn execution (_handle_message_with_agent, _run_agent*, proxy path, background tasks, MCP reload) for GatewayRunner."""

    def _resolve_session_agent_runtime(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session.

        Priority (highest first): session ``/model`` → ``channel_overrides`` → global config/env
        (``_resolve_gateway_model(user_config)`` and default provider resolution).
        """
        from gateway.run import (
            _credential_pool_for_provider,
            _get_channel_override,
            _resolve_gateway_model,
            _resolve_runtime_agent_kwargs,
            _resolve_runtime_agent_kwargs_for_provider,
        )
        skey = self._resolve_session_key_or_none(source, session_key)

        model = _resolve_gateway_model(user_config)
        if skey:
            self._rehydrate_session_model_override(skey)
        _override_state = self._peek_session_state(skey) if skey else None
        override = _override_state.conversation.model_override if _override_state else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                k: override.get(k) for k in (
                    "provider", "requested_provider", "api_key", "base_url", "api_mode",
                    "max_tokens", "credential_pool", "request_overrides", "capabilities",
                )
            }
            override_runtime["capabilities"] = dict(override_runtime["capabilities"] or {})
            if override_runtime.get("api_key"):
                if override_runtime.get("credential_pool") is None:
                    override_runtime["credential_pool"] = _credential_pool_for_provider(
                        override.get("provider")
                    )
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    skey or "", model, override_model, override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # No api_key on the override: fall through to env-based resolution and apply the
            # override's model/provider on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                skey or "", model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                skey or "", model,
                [
                    _key for _key, _st in list(self._sessions_map().items())
                    if _st.conversation.model_override is not None
                ][:5] or "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info("Runtime provider supplied explicit model override: %s -> %s", model, runtime_model)
            model = runtime_model

        cfg = getattr(self, "config", None)
        if cfg and source is not None:
            ch = _get_channel_override(
                cfg,
                source.platform,
                str(source.chat_id) if source.chat_id else "",
                thread_id=str(source.thread_id) if getattr(source, "thread_id", None) else None,
                parent_id=str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
            )
            if ch:
                if ch.model:
                    model = ch.model
                if ch.provider:
                    runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(ch.provider)
                    ch_runtime_model = runtime_kwargs.pop("model", None)
                    # Adopt the provider's bundled model only when the override named none.
                    if ch_runtime_model and not ch.model:
                        model = ch_runtime_model

        if override and skey:
            model, runtime_kwargs = self._apply_session_model_override(skey, model, runtime_kwargs)

        # No model.default but a provider resolved (e.g. `hermes auth add openai-codex` without
        # `hermes model`): fall back to the provider's first catalog model so the API call has one.
        if not model and runtime_kwargs.get("provider"):
            try:
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"],
                    )
            except Exception:
                pass

        # Final safety net: an empty model (e.g. transient config-cache miss on a post-interrupt
        # recovery turn) makes every API call fail HTTP 400 and the session goes silent — reuse the
        # last model resolved for this session, else the most recent process-wide.
        if not model:
            _lr_state = self._peek_session_state(skey) if skey else None
            _lr_star = self._peek_session_state("*")
            _recovered = (
                (_lr_state.conversation.last_resolved_model if _lr_state else "")
                or (_lr_star.conversation.last_resolved_model if _lr_star else "")
            )
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)",
                    skey or "", _recovered,
                )
                model = _recovered
        else:
            # Cache the good resolution for future recovery turns.
            if skey:
                self._session_state(skey).conversation.last_resolved_model = model
            self._session_state("*").conversation.last_resolved_model = model

        return model, runtime_kwargs

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Build the effective model/runtime config for a single turn.

        Always uses the session's primary model/provider. With `/fast` priority on and a model that
        supports it, fast-mode ``request_overrides`` are deep-merged OVER the per-provider
        ``request_overrides`` (e.g. ``custom_providers`` ``extra_body``) so both reach the model.
        """
        from gateway.run import _deep_merge_request_overrides
        from hermes_cli.models import resolve_fast_mode_overrides

        # Tests bind this method onto bare namespaces, so no class-level tables here.
        runtime = {
            k: runtime_kwargs.get(k) for k in (
                "api_key", "base_url", "provider", "requested_provider", "api_mode", "command", "args",
                "credential_pool", "max_tokens", "capabilities",
            )
        }
        runtime["args"] = list(runtime["args"] or [])
        runtime["capabilities"] = dict(runtime["capabilities"] or {})
        base_request_overrides = dict(runtime_kwargs.get("request_overrides") or {})
        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model, runtime["provider"], runtime["requested_provider"], runtime["base_url"],
                runtime["api_mode"], runtime["command"], tuple(runtime["args"]),
            ),
        }
        if getattr(self, "_service_tier", None) != "priority":
            # None (normal) or auto/cold — the bounded window is applied per request by
            # agent.fast_mode, not pinned into request_overrides.
            route["request_overrides"] = base_request_overrides
            return route
        try:
            overrides = resolve_fast_mode_overrides(
                route["model"], provider=runtime["provider"], base_url=runtime["base_url"],
            )
        except Exception:
            overrides = None
        # Fast-mode keys (service_tier / speed) are top-level and don't collide with extra_body.
        route["request_overrides"] = _deep_merge_request_overrides(base_request_overrides, overrides or {})
        return route

    def _sync_session_model_from_agent(self, session_id: str, agent: Any) -> None:
        """Persist the runtime model/provider a gateway turn actually used (provider fallback can
        switch them after the row was created). Runs in the ``run_sync`` executor thread, so it
        uses the sync ``SessionDB`` (``_db``), not the AsyncSessionDB forwarder."""
        if not session_id or agent is None or self._session_db is None:
            return
        model = getattr(agent, "model", None)
        if not model:
            return
        runtime = {
            "provider": getattr(agent, "provider", None),
            "base_url": getattr(agent, "base_url", None),
            "api_mode": getattr(agent, "api_mode", None),
            "fallback_active": bool(getattr(agent, "_fallback_activated", False)),
        }
        runtime = {k: v for k, v in runtime.items() if v not in (None, "")}
        try:
            db = self._session_db._db
            row = db.get_session(session_id)
            if not row:
                return
            raw_config = row.get("model_config")
            try:
                config = json.loads(raw_config) if raw_config else {}
            except Exception:
                config = {}
            if not isinstance(config, dict):
                config = {}
            gateway_runtime = dict(config.get("gateway_runtime") or {})
            if row.get("model") == model and all(gateway_runtime.get(k) == v for k, v in runtime.items()):
                return
            config["gateway_runtime"] = runtime
            db.update_session_meta(session_id, json.dumps(config), model=model)
        except Exception:
            logger.debug("Failed to sync gateway session model metadata", exc_info=True)

    async def _hmwa_resolve_session(self, event, source):
        """Resolve ``source`` to its session entry (topic recovery, internal-route guards, Telegram
        topic-binding heal). Returns ``(source, session_entry, session_key)`` or ``None`` to drop
        the event."""
        # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's last-active topic so a
        # cross-topic Reply or stripped plain reply doesn't fragment the conversation.
        recovered = await asyncio.to_thread(self._recover_telegram_topic_thread_id, source)
        if recovered is not None:
            logger.info(
                "telegram topic recovery: chat=%s user=%s %r -> %s",
                source.chat_id, source.user_id, source.thread_id, recovered,
            )
            source = dataclasses.replace(source, thread_id=recovered)
            with suppress(Exception):
                event.source = source

        event_metadata = getattr(event, "metadata", None) or {}
        expected_session_key = str(event_metadata.get("gateway_session_key") or "").strip()
        if expected_session_key:
            derived_session_key = self._session_key_for_source(source)
            if derived_session_key != expected_session_key:
                logger.warning(
                    "Dropping internally routed event after route recovery: "
                    "expected session=%s derived=%s",
                    expected_session_key, derived_session_key,
                )
                return

        strict_session = bool(event_metadata.get("gateway_session_strict"))
        pinned_session_id = str(event_metadata.get("gateway_session_id") or "").strip()
        if strict_session:
            session_entry = await self.async_session_store.lookup_by_session_key(expected_session_key)
            if (
                session_entry is None
                or not pinned_session_id
                or session_entry.session_id != pinned_session_id
            ):
                logger.warning(
                    "Dropping internally routed event: expected session id=%s is no "
                    "longer current for key=%s",
                    pinned_session_id or "missing", expected_session_key or "missing",
                )
                return
        else:
            # Internal wakes must observe reset policy without counting as user activity, or
            # periodic Kanban/process notifications keep the routing key alive across every
            # daily/idle boundary.
            session_entry = await self.async_session_store.get_or_create_session(
                source, touch_activity=not bool(getattr(event, "internal", False)),
            )
        session_key = session_entry.session_key
        if not strict_session and pinned_session_id:
            resolved_entry = await self._resolve_async_delegation_session(session_entry, pinned_session_id)
            if resolved_entry is None:
                return
            session_entry = resolved_entry
        self._cache_session_source(session_key, source)
        if await asyncio.to_thread(self._is_telegram_topic_lane, source):
            session_entry = await self._hmwa_heal_telegram_topic_binding(source, session_entry, session_key)
        return source, session_entry, session_key

    async def _hmwa_heal_telegram_topic_binding(self, source, session_entry, session_key):
        """Follow the (chat_id, thread_id) topic binding — healed to its compression tip — or record
        a fresh one. Returns the (possibly switched) session entry."""
        try:
            binding = (await self._session_db.get_telegram_topic_binding(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                profile_name=self._telegram_topic_profile_name(source),
            )) if self._session_db else None
        except Exception:
            logger.debug("Failed to read Telegram topic binding", exc_info=True)
            binding = None
        if not binding:
            try:
                await asyncio.to_thread(self._record_telegram_topic_binding, source, session_entry)
            except Exception:
                logger.debug("Failed to record Telegram topic binding", exc_info=True)
            return session_entry
        stored_session_id = str(binding.get("session_id") or "")
        bound_session_id = stored_session_id
        # A binding pointing at a pre-compression parent is walked forward to the tip so the next
        # message resumes the compressed child instead of reloading the oversized parent.
        if bound_session_id and self._session_db is not None:
            try:
                canonical_session_id = await self._session_db.get_compression_tip(bound_session_id)
            except Exception:
                logger.debug("compression-tip lookup failed for %s", bound_session_id, exc_info=True)
                canonical_session_id = bound_session_id
            if canonical_session_id and canonical_session_id != bound_session_id:
                bound_session_id = canonical_session_id
        if bound_session_id and bound_session_id != session_entry.session_id:
            # Route through SessionStore so the session_key → session_id mapping is persisted and
            # the previous lane session ended cleanly; mutating session_entry in place split-brained
            # the JSON index from downstream code.
            switched = await self.async_session_store.switch_session(session_key, bound_session_id)
            if switched is not None:
                session_entry = switched
        if bound_session_id and bound_session_id != stored_session_id:
            # The stored binding pointed at a parent: rewrite it to the canonical descendant.
            await asyncio.to_thread(
                self._sync_telegram_topic_binding, source, session_entry, reason="compression-tip-walk",
            )
        return session_entry

    async def _hmwa_open_session(self, session_entry, session_key, source):
        """Consume auto-reset / fresh-reset flags and emit ``session:start`` for new sessions.
        Returns ``(_was_auto_reset, _is_new_session)``."""
        # Consume was_auto_reset immediately so it cannot re-fire on later messages and wipe
        # model/reasoning overrides set between turns.
        _was_auto_reset = getattr(session_entry, "was_auto_reset", False)
        if _was_auto_reset:
            # Full conversation boundary: one funnel call clears every conversation-scoped
            # per-session dict (no inherited overrides, queued "/model switched" note or stale
            # resolved-model cache); evict the cached agent (keyed on the stable session_key) so
            # context_compressor._previous_summary cannot leak prior history into new summaries.
            self._clear_conversation_scope(session_key, reason="auto_reset")
            self._evict_cached_agent(session_key)
            session_entry.was_auto_reset = False

        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or _was_auto_reset
            or getattr(session_entry, "is_fresh_reset", False)
        )
        # Consume is_fresh_reset so it doesn't leak onto later messages in the same session.
        if getattr(session_entry, "is_fresh_reset", False):
            session_entry.is_fresh_reset = False
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        return _was_auto_reset, _is_new_session

    async def _hmwa_deliver_auto_reset_notice(self, session_entry, source, turn_sidecar_notes):
        """Stage the auto-reset sidecar note for the agent and notify the user (policy-gated)."""
        from gateway.run import _AUTO_RESET_CONTEXT_NOTES, _auto_reset_reason_text
        reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
        context_note = _AUTO_RESET_CONTEXT_NOTES.get(reset_reason, _AUTO_RESET_CONTEXT_NOTES["idle"])
        # Long-lived Slack/Discord channels: point the agent at the specific prior same-channel
        # session so session_search recalls that context (deterministic, no extra API/DB calls).
        try:
            continuity_note = build_channel_continuity_note(session_entry, source)
        except Exception:
            continuity_note = None
        if continuity_note:
            context_note = context_note + "\n\n" + continuity_note
        turn_sidecar_notes.append(context_note)

        try:
            policy = self.session_store.config.get_reset_policy(
                platform=source.platform, session_type=getattr(source, 'chat_type', 'dm'),
            )
            platform_name = source.platform.value if source.platform else ""
            # Suspended / restart-recovery-expired sessions always notify regardless of
            # policy.notify — an active session was silently replaced and the user must learn they
            # can /resume it. Idle/daily resets respect the policy flag, the excluded platforms
            # (api_server, webhook) and require prior activity.
            should_notify = reset_reason in {"suspended", "resume_pending_expired"} or (
                policy.notify
                and getattr(session_entry, 'reset_had_activity', False)
                and platform_name not in policy.notify_exclude_platforms
            )
            adapter = self._adapter_for_source(source) if should_notify else None
            if adapter:
                notice = (
                    f"◐ Session automatically reset ({_auto_reset_reason_text(reset_reason, policy)}). "
                    f"Conversation history cleared.\n"
                    f"Use /resume to browse and restore a previous session.\n"
                    f"Adjust reset timing in config.yaml under session_reset."
                )
                try:
                    session_info = await asyncio.to_thread(self._reset_notice_session_info, source)
                    if session_info:
                        notice = f"{notice}\n\n{session_info}"
                except Exception:
                    pass
                await adapter.send(
                    source.chat_id, notice, metadata=self._thread_metadata_for_source(source),
                )
        except Exception as e:
            logger.debug("Auto-reset notification failed (non-fatal): %s", e)

        # was_auto_reset was consumed in _hmwa_open_session; only the reason needs clearing.
        session_entry.auto_reset_reason = None

    def _hmwa_auto_load_skills(self, event, _auto, _quick_key, session_key):
        """Prepend topic/channel-bound skill payload(s) to ``event.text`` on a new session."""
        _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
        try:
            from agent.skill_commands import _load_skill_payload, _build_skill_message
            _combined_parts: list[str] = []
            _loaded_names: list[str] = []
            for _sname in _skill_names:
                _loaded = _load_skill_payload(_sname, task_id=_quick_key)
                if not _loaded:
                    logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
                    continue
                _loaded_skill, _skill_dir, _display_name = _loaded
                _part = _build_skill_message(
                    _loaded_skill, _skill_dir,
                    f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                    f"Follow its instructions for this session.]",
                )
                if _part:
                    _combined_parts.append(_part)
                    _loaded_names.append(_sname)
            if _combined_parts:
                _combined_parts.append(event.text)  # user's original text after the payloads
                event.text = "\n\n".join(_combined_parts)
                logger.info("[Gateway] Auto-loaded skill(s) %s for session %s", _loaded_names, session_key)
        except Exception as e:
            logger.warning("[Gateway] Failed to auto-load skill(s) %s: %s", _skill_names, e)

    async def _hmwa_acquire_turn_lease(self, _quick_key, run_generation, session_entry, _session_env_tokens):
        """Serialize [load history → run → flush] per resolved SESSION_ID (session resolution is
        FINAL here): another routing key mapped to the same session_id waits for the prior flush
        instead of loading a stale base. Fail-closed on timeout — never enter the transcript
        region without a lease; outer dispatch returns a bounded resend notice. Released in
        _handle_message's finally, granted per (routing key, run generation) so a stale unwind
        can't release a newer turn's."""
        from gateway.run import _float_env
        _lease_registry = getattr(self, "_turn_leases", None)
        if _lease_registry is None:
            return
        try:
            _lease_token = await _lease_registry.acquire(
                session_entry.session_id,
                owner_key=_quick_key,
                generation=run_generation,
                timeout=_float_env("HERMES_TURN_LEASE_TIMEOUT", DEFAULT_LEASE_WAIT),
            )
        except TurnLeaseTimeoutError:
            # The broad session-context cleanup finally starts later; restore the tokens here or
            # this early exit leaks task-local identity.
            self._clear_session_env(_session_env_tokens)
            raise
        if _lease_token is not None:
            _lease_state = self._session_state(_quick_key).turn
            _lease_state.lease_token = _lease_token
            _lease_state.lease_generation = run_generation

    @dataclasses.dataclass
    class _HygienePlan:
        """Hygiene pre-check outcome for one turn."""

        needs_compress: bool
        approx_tokens: int
        msg_count: int
        warn_token_threshold: int

    async def _hmwa_hygiene_settings(self, source, session_key):
        """Resolve model/provider/context-length + hygiene knobs for the pre-agent compression
        safety net (fail-soft: any config/runtime error keeps the defaults).

        The hygiene threshold (0.85) is deliberately HIGHER than the agent's own compressor
        (0.50): it is a safety net for sessions that grew between turns. ``max_turn_hold_seconds``
        bounds how long the user's TURN waits on hygiene before proceeding uncompressed (the
        compressor keeps running detached, commit fenced); kept below transport idle-timeouts."""
        from gateway.run import _load_gateway_config
        hs = self._HygieneSettings(
            model="anthropic/claude-sonnet-4.6",
            threshold_pct=0.85,
            compression_enabled=True,
            hard_msg_limit=5000,
            timeout_seconds=30.0,
            total_ceiling_seconds=600.0,
            max_turn_hold_seconds=10.0,
            failure_cooldown_seconds=300.0,
            config_context_length=None,
            provider=None,
            base_url=None,
            api_key=None,
            data={},
        )
        try:
            hs.data = _load_gateway_config()
            if hs.data:
                # Resolve model name (same logic as run_sync)
                _model_cfg = hs.data.get("model", {})
                if isinstance(_model_cfg, str):
                    hs.model = _model_cfg
                elif isinstance(_model_cfg, dict):
                    hs.model = _model_cfg.get("default") or _model_cfg.get("model") or hs.model
                    _raw_ctx = _model_cfg.get("context_length")
                    if _raw_ctx is not None:
                        with suppress(TypeError, ValueError):
                            hs.config_context_length = int(_raw_ctx)
                    hs.provider = _model_cfg.get("provider") or None
                    hs.base_url = _model_cfg.get("base_url") or None

                # Only the enabled flag is shared with the agent's compression config; hygiene's
                # threshold is deliberately separate (runs higher).
                _comp_cfg = hs.data.get("compression", {})
                if isinstance(_comp_cfg, dict):
                    hs.compression_enabled = str(_comp_cfg.get("enabled", True)).lower() in {"true", "1", "yes"}

                    def _knob(key, current, cast, allow_zero=False):
                        raw = _comp_cfg.get(key)
                        if raw is None:
                            return current
                        try:
                            parsed = cast(raw)
                        except (TypeError, ValueError):
                            return current
                        return parsed if (parsed >= 0 if allow_zero else parsed > 0) else current

                    hs.hard_msg_limit = _knob("hygiene_hard_message_limit", hs.hard_msg_limit, int)
                    hs.timeout_seconds = _knob("hygiene_timeout_seconds", hs.timeout_seconds, float)
                    hs.total_ceiling_seconds = _knob("hygiene_total_ceiling_seconds", hs.total_ceiling_seconds, float)
                    # The ceiling can never be tighter than one idle window, or the extension
                    # loop would be dead code.
                    hs.total_ceiling_seconds = max(hs.total_ceiling_seconds, hs.timeout_seconds)
                    hs.max_turn_hold_seconds = _knob("hygiene_max_turn_hold_seconds", hs.max_turn_hold_seconds, float)
                    hs.failure_cooldown_seconds = _knob(
                        "hygiene_failure_cooldown_seconds", hs.failure_cooldown_seconds, float, allow_zero=True,
                    )

            configured_model, configured_provider, configured_base_url = hs.model, hs.provider, hs.base_url

            try:
                hs.model, _hyg_runtime = self._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                    user_config=hs.data if isinstance(hs.data, dict) else None,
                )
                hs.provider = _hyg_runtime.get("provider") or hs.provider
                hs.base_url = _hyg_runtime.get("base_url") or hs.base_url
                hs.api_key = _hyg_runtime.get("api_key") or hs.api_key
            except Exception:
                pass

            if hs.config_context_length is not None:
                try:
                    from hermes_cli.route_identity import should_clear_context_pin_async

                    if await should_clear_context_pin_async(
                        configured_model, hs.model, configured_base_url, hs.base_url,
                        configured_provider, hs.provider,
                    ):
                        hs.config_context_length = None
                except Exception:
                    hs.config_context_length = None

            # custom_providers per-model context_length fallback (as in run_agent.py); must run
            # after runtime resolution so base_url is set.
            if hs.config_context_length is None and hs.base_url:
                try:
                    try:
                        from hermes_cli.config import (
                            get_compatible_custom_providers as _gw_gcp,
                            get_custom_provider_context_length as _gw_gccl,
                        )
                        _hyg_custom_providers = _gw_gcp(hs.data)
                    except Exception:
                        _hyg_custom_providers = hs.data.get("custom_providers")
                        if not isinstance(_hyg_custom_providers, list):
                            _hyg_custom_providers = []
                    _hyg_custom_ctx = _gw_gccl(
                        model=hs.model, base_url=hs.base_url, custom_providers=_hyg_custom_providers,
                    )
                    if _hyg_custom_ctx:
                        hs.config_context_length = int(_hyg_custom_ctx)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass
        return hs

    async def _hmwa_hygiene_plan(self, hs, history, session_entry, session_key):
        """Decide whether hygiene compression fires this turn (token/message thresholds, DB-backed
        failure cooldown, in-flight compression)."""
        from agent.model_metadata import estimate_messages_tokens_rough, get_model_context_length_async

        _hyg_context_length = await get_model_context_length_async(
            hs.model,
            base_url=hs.base_url or "",
            api_key=hs.api_key or "",
            config_context_length=hs.config_context_length,
            provider=hs.provider or "",
        )
        _compress_token_threshold = int(_hyg_context_length * hs.threshold_pct)
        _warn_token_threshold = int(_hyg_context_length * 0.95)
        _msg_count = len(history)

        # Prefer the API-reported prompt tokens from the last turn over the rough char estimate.
        # Rough estimates run 30-50% high on code/JSON-heavy sessions, which only makes hygiene
        # fire early (safe). Do NOT compensate with a threshold multiplier: 85% * 1.4 = 119% of
        # context kept hygiene from ever firing for ~200K models.
        if session_entry.last_prompt_tokens > 0:
            _approx_tokens, _token_source = session_entry.last_prompt_tokens, "actual"
        else:
            _approx_tokens, _token_source = estimate_messages_tokens_rough(history), "estimated"

        # Hard safety valve: force compression at an extreme message count regardless of token
        # estimates, breaking the spiral where API disconnects prevent token data → no compression
        # → more disconnects. Default 5000 sits clear of legitimate 1M+ context sessions.
        _needs_compress = _approx_tokens >= _compress_token_threshold or _msg_count >= hs.hard_msg_limit

        if _needs_compress:
            # Persistent DB-backed cooldown (shared with context_compressor.py) so it survives
            # gateway restarts; an in-memory dict re-triggered the same failing compression on
            # every restart and wedged session storage.
            _session_db = getattr(self, "_session_db", None)
            if _session_db is not None:
                _session_db = getattr(_session_db, "_db", _session_db)
                _getter = getattr(_session_db, "get_compression_failure_cooldown", None)
                if _getter is not None:
                    try:
                        _cooldown_state = _getter(session_entry.session_id)
                    except Exception:
                        _cooldown_state = None
                    if _cooldown_state and _cooldown_state.get("remaining_seconds", 0) > 0:
                        logger.info(
                            "Session hygiene: skipping compression for %s; "
                            "previous failure cooldown active for %.1fs",
                            session_entry.session_id, _cooldown_state["remaining_seconds"],
                        )
                        _needs_compress = False

        if _needs_compress and await self._session_has_compression_in_flight(session_key):
            # A prior hygiene/agent compression still holds the durable lock (typically a shielded
            # worker left behind by /stop or /restart). Another attempt would wait up to the 600s
            # ceiling behind a commit the fence will refuse, while inbound messages demote to queue.
            logger.info(
                "Session hygiene: skipping compression for %s; "
                "another compression is already in flight",
                session_entry.session_id,
            )
            _needs_compress = False

        if _needs_compress:
            logger.info(
                "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                "(threshold: %s%% of %s = %s tokens)",
                _msg_count, f"{_approx_tokens:,}", _token_source,
                int(hs.threshold_pct * 100), f"{_hyg_context_length:,}", f"{_compress_token_threshold:,}",
            )
        return self._HygienePlan(_needs_compress, _approx_tokens, _msg_count, _warn_token_threshold)

    async def _hmwa_hygiene_wait_for_summary(self, attempt, hs, session_entry):
        """Progress-aware inline wait for the detached hygiene compressor. Returns the compressed
        transcript; raises ``HygieneTurnHoldExceeded`` (turn-hold budget) or
        ``asyncio.TimeoutError`` (idle/ceiling/fence cancel) for the caller's handlers.

        The timeout is an INACTIVITY budget — the worker ticks the fence per streamed token, so a
        slow but still-generating model extends the deadline; a hard ceiling bounds the total so
        a trickle stream can't hold the turn; the turn-hold budget caps how long the user's TURN
        waits regardless."""
        from gateway.run import HygieneTurnHoldExceeded, hygiene_wait_should_extend
        fence = attempt.commit_fence
        while True:
            if fence.is_cancelled:
                raise asyncio.TimeoutError
            # Charge the idle budget from the LAST PROGRESS event, not from the start of this wait
            # slice — otherwise silence can approach 2x the configured timeout.
            _hyg_waited = time.monotonic() - attempt.wait_started
            _idle_left = max(hs.timeout_seconds - fence.seconds_since_progress(), 0.005)
            _slice = min(_idle_left, max(hs.total_ceiling_seconds - _hyg_waited, 0.005))
            # Cap the slice at the remaining turn-hold budget so it is re-checked at least that
            # often — a continuously-streaming worker would otherwise keep the slice large and hold
            # the turn until the ceiling. Budget exhausted → immediate timeout → abandonment path.
            _turn_hold_remaining = hs.max_turn_hold_seconds - (time.monotonic() - attempt.wait_started)
            _slice = 0.005 if _turn_hold_remaining <= 0 else min(_slice, max(_turn_hold_remaining, 0.005))
            # Short poll so a /stop or /restart cancel is not stuck behind a full idle window.
            _slice = min(_slice, 0.25)
            try:
                _compressed, _ = await asyncio.wait_for(asyncio.shield(attempt.future), timeout=_slice)
                return _compressed
            except asyncio.TimeoutError:
                if fence.is_cancelled:
                    raise
                _hyg_waited = time.monotonic() - attempt.wait_started
                _idle = fence.seconds_since_progress()
                # Never hold the user's TURN past the budget even if the summary model is still
                # streaming: the turn-hold path proceeds on the uncompressed transcript so the wire
                # never trips a transport idle-timeout.
                if _hyg_waited >= hs.max_turn_hold_seconds:
                    logger.info(
                        "Session hygiene compression for session %s exceeded the turn-hold "
                        "budget (%.1fs >= %.1fs) — abandoning inline wait, proceeding "
                        "without compression this turn",
                        session_entry.session_id, _hyg_waited, hs.max_turn_hold_seconds,
                    )
                    raise HygieneTurnHoldExceeded(
                        f"turn-hold budget {hs.max_turn_hold_seconds:.1f}s "
                        f"elapsed after {_hyg_waited:.1f}s"
                    )
                if hygiene_wait_should_extend(
                    idle=_idle,
                    timeout=hs.timeout_seconds,
                    waited=_hyg_waited,
                    ceiling=hs.total_ceiling_seconds,
                    fence_cancelled=fence.is_cancelled,
                ):
                    if _slice >= _idle_left - 1e-9:
                        logger.info(
                            "Session hygiene compression for session %s still streaming after "
                            "%.0fs (last progress %.1fs ago) — extending wait (ceiling %.0fs)",
                            session_entry.session_id, _hyg_waited, _idle, hs.total_ceiling_seconds,
                        )
                    continue
                raise

    async def _hmwa_hygiene_cancel_before_commit(self, fence):
        """Cancel the detached worker at the commit fence. Returns ``False`` when the worker had
        already crossed into its commit (caller must consume the result — aborting mid-commit
        would corrupt the message-store transaction), ``True`` when cancelled in time.

        A hung commit retains the fence lock; the lock-free ``commit_in_flight`` marker keeps
        this loop from spinning forever. 25ms polls ride transient lock-setup windows."""
        while True:
            if fence.commit_in_flight:
                return False
            cancelled = fence.try_cancel_before_commit()
            if cancelled is not None:
                return cancelled
            await asyncio.sleep(0.025)

    def _hmwa_hygiene_defer_cleanup(self, attempt, context):
        """Hand the agent's cleanup to the still-running worker future and mark it deferred."""
        self._defer_agent_cleanup_until_future_done(attempt.future, attempt.agent, context=context)
        attempt.cleanup_deferred = True

    @staticmethod
    def _hmwa_hygiene_stamp(agent, desc, provenance_name, debug_label):
        from agent.session_activity import ActivityProvenance
        from gateway.run import _stamp_hygiene_compression_provenance
        _stamp_hygiene_compression_provenance(
            agent, desc, getattr(ActivityProvenance, provenance_name), debug_label,
        )

    async def _hmwa_hygiene_notify(self, source, meta, message, what):
        """Best-effort user notice on the hygiene thread; failure is logged, never raised."""
        try:
            _adapter = self._adapter_for_source(source)
            if _adapter and source.chat_id:
                await _adapter.send(source.chat_id, message, metadata=meta)
        except Exception as _werr:
            logger.warning("Failed to deliver %s to user: %s", what, _werr)

    async def _hmwa_hygiene_record_failure_cooldown(self, hs, session_key, session_id, reason):
        """Escalate the failure streak (off-loop) and persist the cooldown, when enabled."""
        from gateway.run import _hygiene_cooldown_for_failure, _record_hygiene_cooldown
        if hs.failure_cooldown_seconds < 0:
            return
        _hyg_cooldown = await asyncio.to_thread(
            _hygiene_cooldown_for_failure, self, session_key, hs.failure_cooldown_seconds,
        )
        _record_hygiene_cooldown(self, session_id, _hyg_cooldown, reason)

    async def _hmwa_hygiene_on_turn_hold(self, attempt, hs, session_entry, session_key, source):
        """``except HygieneTurnHoldExceeded`` body: keep or cancel the worker's commit admission,
        notify the user, and re-raise; returns the compressed transcript only when the worker
        was already committing.

        Turn-hold expiry is an availability boundary, not a failure: the compressor is healthy,
        so the failure streak must NOT advance — only flat, non-escalating retry spacing is
        recorded. When the worker's commit is watermark-fenced (rows appended after compression
        start — this turn included — survive its late commit as cloned concurrent tail) the
        attempt KEEPS commit admission: the turn proceeds uncompressed NOW and the summary is
        adopted at the worker's own fenced commit; always cancelling burned every attempt for
        thinking summary models whose reasoning prefix alone exceeds the hold. Without the
        watermark fence (no session_db, capture failed, legacy lock API) a late commit could
        clobber newer turns, so cancel."""
        from gateway.run import (
            _HYGIENE_TURNHOLD_RETRY_SECONDS,
            _record_hygiene_cooldown,
            _reset_hygiene_failure_streak,
        )
        fence = attempt.commit_fence
        _hyg_keep_admission = (
            bool(getattr(fence, "commit_watermark_fenced", False)) and not fence.is_cancelled
        )
        if _hyg_keep_admission:
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene turn-hold")
            # NO retry-after here: the attempt is still running toward a real commit, and the flat
            # 60s retry-after would also block the agent-side preflight compressor (same-session
            # cooldown). Re-attempt spacing comes from the durable compression lock: the next
            # turn's hygiene pre-check skips while this worker's lease is held. The done-callback
            # records the flat retry-after ONLY if the worker ends without committing anything.
            _sid, _skey, _agent = session_entry.session_id, session_key, attempt.agent

            def _hyg_adopt_or_space_retry(_fut, _gw=self, _sid=_sid, _skey=_skey, _agent=_agent):
                try:
                    _exc = _fut.exception()
                except (asyncio.CancelledError, Exception):
                    _committed = False
                else:
                    _committed = _exc is None and (
                        bool(getattr(_agent, "_last_compaction_in_place", False))
                        or getattr(_agent, "session_id", _sid) != _sid
                    )
                if _committed:
                    logger.info(
                        "Session hygiene compression for session %s finished after the "
                        "turn-hold was released — summary adopted at the watermark-fenced "
                        "commit boundary (#97963)",
                        _sid,
                    )
                    try:
                        _reset_hygiene_failure_streak(_gw, _skey)
                    except Exception as _rs_err:
                        logger.debug("hygiene streak reset after deferred adoption failed: %s", _rs_err)
                else:
                    # Nothing to adopt (summary failed, fence refused the commit, or the attempt
                    # was superseded). Flat spacing so sustained traffic does not spawn and
                    # abandon a fresh compressor every turn.
                    _record_hygiene_cooldown(
                        _gw, _sid, _HYGIENE_TURNHOLD_RETRY_SECONDS,
                        "hygiene compression deferred: turn-hold budget expired and the "
                        "detached attempt did not commit",
                    )

            attempt.future.add_done_callback(_hyg_adopt_or_space_retry)
            _log_suffix = (
                " — the watermark-fenced worker keeps its commit admission and the summary "
                "will be adopted when it finishes"
            )
        else:
            if not await self._hmwa_hygiene_cancel_before_commit(fence):
                # Bounded overshoot by design: the turn can be held past the budget by up to the
                # commit duration. Do NOT "fix" this into a mid-commit cancellation.
                _compressed, _ = await attempt.future
                return _compressed
            fence.release_cancelled_compression_lock()
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene turn-hold")
            # Short flat retry-after: without it every turn re-spawns a compressor, holds it for
            # the budget and cancels it — token burn that never commits.
            _record_hygiene_cooldown(
                self, session_entry.session_id, _HYGIENE_TURNHOLD_RETRY_SECONDS,
                "hygiene compression deferred: turn-hold budget expired while the "
                "summary was still streaming",
            )
            _log_suffix = ""
        self._hmwa_hygiene_stamp(
            attempt.agent, "session hygiene compression turn-hold",
            "AGENT_COMPRESSION_TURNHOLD", "hygiene compression turn-hold activity stamp failed",
        )
        logger.info(
            "Session hygiene compression for session %s exceeded turn-hold budget (%.1fs); "
            "proceeding without compression this turn%s",
            session_entry.session_id, time.monotonic() - attempt.wait_started, _log_suffix,
        )
        await self._hmwa_hygiene_notify(
            source, attempt.meta, t("gateway.compress.turnhold_deferred"), "compression-turnhold notice",
        )
        raise

    async def _hmwa_hygiene_on_timeout(self, attempt, hs, session_entry, session_key, source):
        """``except asyncio.TimeoutError`` body: cancel at the commit fence, record the failure
        cooldown, warn the user, and re-raise; returns the compressed transcript only when the
        worker crossed the commit boundary first."""
        from gateway.run import _hygiene_compression_timeout_message
        fence = attempt.commit_fence
        _hyg_waited = time.monotonic() - attempt.wait_started
        _hyg_total_exhausted = _hyg_waited >= hs.total_ceiling_seconds or fence.deadline_exceeded
        if _hyg_total_exhausted:
            # The worker cooperatively checks this deadline between digest calls. Keep its lease
            # until it exits so an unchanged session cannot overlap a retry.
            fence.retain_compression_lock_until_worker_done()
        # Capture fence state BEFORE try_cancel — that call itself sets is_cancelled, which would
        # mis-label a genuine idle timeout as a fence cancel.
        _hyg_fence_cancelled = fence.is_cancelled
        if not await self._hmwa_hygiene_cancel_before_commit(fence):
            # The worker crossed the commit boundary just before the timeout: consume the result
            # instead of treating a successful compaction as a timeout.
            _compressed, _ = await attempt.future
            return _compressed
        # Release an inactivity-timed-out worker's holder-qualified lease promptly (no-op for
        # total-ceiling attempts, which retained it above).
        fence.release_cancelled_compression_lock()
        self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene timeout")
        await self._hmwa_hygiene_record_failure_cooldown(
            hs, session_key, session_entry.session_id,
            "session hygiene compression " + (
                "cancelled at commit fence" if _hyg_fence_cancelled
                else "total ceiling exhausted" if _hyg_total_exhausted
                else "timed out with no output from the summary model"
            ),
        )
        self._hmwa_hygiene_stamp(
            attempt.agent,
            "session hygiene compression cancelled at commit fence" if _hyg_fence_cancelled
            else "session hygiene compression timed out",
            "AGENT_COMPRESSION_TIMEOUT", "hygiene compression timeout activity stamp failed",
        )
        if _hyg_fence_cancelled:
            logger.warning(
                "Session hygiene compression for session %s was cancelled at the "
                "commit fence; continuing without compression",
                session_entry.session_id,
            )
            raise
        _hyg_elapsed = time.monotonic() - attempt.wait_started
        if _hyg_total_exhausted:
            logger.warning(
                "Session hygiene compression for session %s reached its total ceiling after "
                "%.1fs (progress observed=%s); continuing without compression",
                session_entry.session_id, _hyg_elapsed, fence.progress_observed,
            )
        else:
            logger.warning(
                "Session hygiene compression for session %s made no progress for %.1fs "
                "(total wait %.1fs, ceiling %.1fs); continuing without compression",
                session_entry.session_id, fence.seconds_since_progress(), _hyg_elapsed,
                hs.total_ceiling_seconds,
            )
        await self._hmwa_hygiene_notify(
            source, attempt.meta,
            _hygiene_compression_timeout_message(
                total_exhausted=_hyg_total_exhausted,
                elapsed=_hyg_elapsed,
                idle_timeout=hs.timeout_seconds,
                progress_observed=fence.progress_observed,
            ),
            "compression-timeout warning",
        )
        raise

    def _hmwa_hygiene_on_unwind(self, attempt, hs, session_entry, session_key):
        """``except BaseException`` body (caller re-raises): revoke commit admission and record a
        cooldown so the next turn does not immediately re-arm hygiene.

        Non-timeout unwind (KeyboardInterrupt, task cancel, unexpected error) while the detached
        worker may still run: revoke admission (and release its durable lease) BEFORE the host
        unwinds so it can never commit later. Restart drain / task cancel must record a cooldown,
        or the next turn re-arms hygiene and waits up to 600s behind a fence that refuses again."""
        from gateway.run import _hygiene_cooldown_for_failure, _record_hygiene_cooldown
        attempt.commit_fence.revoke_commit_admission()
        if not attempt.cleanup_deferred:
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene unwind")
        if hs.failure_cooldown_seconds >= 0:
            try:
                _hyg_cooldown = _hygiene_cooldown_for_failure(self, session_key, hs.failure_cooldown_seconds)
                _record_hygiene_cooldown(
                    self, session_entry.session_id, _hyg_cooldown,
                    "session hygiene compression cancelled at commit fence",
                )
            except Exception as _cd_err:
                logger.debug("hygiene unwind cooldown record failed: %s", _cd_err)

    async def _hmwa_hygiene_apply_result(
        self, attempt, hs, _compressed, history, plan, *,
        session_entry, session_key, source, _quick_key, run_generation,
    ):
        """Adopt a finished hygiene compression (rotation / in-place / refused), rebind the session
        + turn lease, record streak/cooldown, and warn the user on abort. Publishes the
        (possibly replaced) transcript on ``attempt.history``."""
        from gateway.run import _reset_hygiene_failure_streak, hygiene_compaction_recovered
        from agent.model_metadata import estimate_messages_tokens_rough

        _hyg_agent = attempt.agent
        # _compress_context ends the old session and creates a new session_id; compressed messages
        # go into the NEW session so the old transcript stays intact and searchable.
        _hyg_new_sid = _hyg_agent.session_id
        _hyg_rotated = _hyg_new_sid != session_entry.session_id
        _hyg_in_place = bool(getattr(_hyg_agent, "_last_compaction_in_place", False))
        # Anti-growth guard: refuse a compression that did not shrink the transcript (observed:
        # 427K -> 598K). Compare like-for-like rough estimates.
        _hyg_in_toks = estimate_messages_tokens_rough(history)
        _hyg_out_toks = estimate_messages_tokens_rough(_compressed)
        if _hyg_rotated and _hyg_out_toks > _hyg_in_toks:
            logger.warning(
                "Gateway hygiene compression for session %s "
                "would grow transcript (~%s -> ~%s tokens); "
                "keeping the original transcript unchanged",
                session_entry.session_id, f"{_hyg_in_toks:,}", f"{_hyg_out_toks:,}",
            )
            _hyg_rotated = False
            _compressed = history
        # Rewrite the transcript only when rotation produced a NEW session id. In-place compaction
        # needs none: archive_and_compact() already soft-archived the previous rows, and
        # rewrite_transcript() would replace_messages(active_only=False) and DELETE the archived
        # turns. A summary with neither rotation nor a completed archive_and_compact() signals
        # FAILURE; an unconditional rewrite would replace the originals with only the summary.
        # Write-before-repoint (mirrors manual /compress): if session_entry were repointed first
        # and rewrite_transcript then failed (lock/ENOSPC), the live entry would reference an
        # empty session and the conversation silently vanishes.
        if _hyg_rotated:
            if not await self.async_session_store.rewrite_transcript(_hyg_new_sid, _compressed):
                logger.error(
                    "Session hygiene: failed to persist "
                    "compressed transcript for rotated "
                    "session %s → %s; keeping the live "
                    "entry on the original session so the "
                    "conversation is not dropped",
                    session_entry.session_id, _hyg_new_sid,
                )
                # Fail closed: treat like no rotation.
                _hyg_rotated = False
                _hyg_in_place = False
            else:
                session_entry.session_id = _hyg_new_sid
                # The held turn lease follows the rotation so an alias key resolving the fresh
                # child still serializes against this turn.
                self._rebind_turn_lease(_quick_key, run_generation, _hyg_new_sid)
                await self.async_session_store._save()
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="hygiene-compression",
                )

        if _hyg_rotated or _hyg_in_place:
            # Transcript rewritten (rotation) or already persisted by archive_and_compact()
            # (in-place): reset the stored token count to match the new active set.
            session_entry.last_prompt_tokens = 0
            attempt.history = _compressed
            _new_count = len(_compressed)
            _new_tokens = estimate_messages_tokens_rough(_compressed)
        else:
            # No rewrite happened — post-compression counts equal the pre-compression ones.
            _new_count = plan.msg_count
            _new_tokens = plan.approx_tokens
            logger.warning(
                "Gateway hygiene compression for session %s "
                "did not rotate or compact in place "
                "(no session_db on the hygiene agent) — "
                "preserving the original transcript instead "
                "of overwriting it with the summary (#21301).",
                session_entry.session_id,
            )

        logger.info(
            "Session hygiene: compressed %s → %s msgs, "
            "~%s → ~%s tokens",
            plan.msg_count, _new_count, f"{plan.approx_tokens:,}", f"{_new_tokens:,}",
        )
        if _new_tokens >= plan.warn_token_threshold:
            logger.warning("Session hygiene: still ~%s tokens after compression", f"{_new_tokens:,}")

        # Summary failure aborts the compressor entirely (messages unchanged, nothing dropped).
        # Warn the gateway user visibly — agent.log is invisible on TG/Discord/etc. — so they
        # know the chat is "frozen" at this size and can /compress to retry or /reset.
        _comp = getattr(_hyg_agent, "context_compressor", None)
        _hyg_aborted = _comp is not None and getattr(_comp, "_last_compress_aborted", False)
        # A fence-cancelled _compress_context returns the original transcript with
        # _last_compress_aborted still False. Treat that no-op as an abort so hygiene records a
        # cooldown instead of retrying into the 600s wait. A successful rotate/in-place commit is
        # not an abort even if a later invalidation flipped the fence.
        _hyg_fence_cancelled = bool(attempt.commit_fence.is_cancelled and not _hyg_rotated and not _hyg_in_place)
        if _hyg_fence_cancelled:
            _hyg_aborted = True
        if not _hyg_aborted:
            # Recovery decision lives in the unit-tested predicate: the degenerate "neither
            # rotated nor compacted in place" path reuses the pre-compression counts, so a
            # numbers-only check would read a no-op as success and clear the streak.
            if hygiene_compaction_recovered(
                aborted=_hyg_aborted,
                rotated=_hyg_rotated,
                in_place=_hyg_in_place,
                msg_count=plan.msg_count,
                new_count=_new_count,
                approx_tokens=plan.approx_tokens,
                new_tokens=_new_tokens,
            ):
                await asyncio.to_thread(_reset_hygiene_failure_streak, self, session_key)
        if _hyg_aborted:
            await self._hmwa_hygiene_record_failure_cooldown(
                hs, session_key, session_entry.session_id,
                "session hygiene compression cancelled at commit fence" if _hyg_fence_cancelled
                else getattr(_comp, "_last_summary_error", None),
            )
            self._hmwa_hygiene_stamp(
                _hyg_agent, "session hygiene compression aborted",
                "AGENT_COMPRESSION_COOLDOWN", "hygiene compression abort activity stamp failed",
            )
            if not _hyg_fence_cancelled:
                _err = getattr(_comp, "_last_summary_error", None) or "unknown error"
                # Force-redact: provider exception text may contain credentials and this message
                # reaches gateway users.
                from agent.redact import redact_sensitive_text
                _err = redact_sensitive_text(_err, force=True)
                await self._hmwa_hygiene_notify(
                    source, attempt.meta,
                    "⚠️ Context compression aborted "
                    f"({_err}). No messages were dropped — "
                    "conversation is unchanged. Run /compress "
                    "to retry, /reset for a clean session, or "
                    "check your auxiliary.compression model "
                    "configuration.",
                    "compression-failure warning",
                )
        # If the CONFIGURED aux model failed and we recovered on the main model, tell the user —
        # a misconfigured auxiliary.compression.model is something only they can fix.
        elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
            _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
            _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
            await self._hmwa_hygiene_notify(
                source, attempt.meta,
                f"ℹ️ Configured compression model `{_aux_model}` "
                f"failed ({_aux_err}). Recovered using your main "
                "model — context is intact — but you may want to "
                "check `auxiliary.compression.model` in config.yaml.",
                "aux-model-fallback notice",
            )

    async def _hmwa_hygiene_codex_compaction(self, hs, plan, history, session_entry, session_key, _hyg_runtime):
        """codex app-server runtime: the real context is the server-side thread, not the transcript
        mirror. The detached-agent path would only rewrite the mirror and its finally-eviction
        would destroy the live thread (next turn starts blank), so use the cached agent's
        thread/compact/start and KEEP it cached."""
        from gateway.run import run_codex_hygiene_compaction
        _hyg_codex_auto = "native"
        _hyg_comp_cfg = hs.data.get("compression") if isinstance(hs.data, dict) else None
        if isinstance(_hyg_comp_cfg, dict):
            _hyg_codex_auto = str(_hyg_comp_cfg.get("codex_app_server_auto", "native") or "native")
        _hyg_codex_outcome = await run_codex_hygiene_compaction(
            self,
            session_key,
            session_entry.session_id,
            auto_mode=_hyg_codex_auto,
            history=history,
            approx_tokens=plan.approx_tokens,
            timeout_seconds=hs.total_ceiling_seconds,
            failure_cooldown_seconds=hs.failure_cooldown_seconds,
        )
        logger.info(
            "Session hygiene (codex app-server): %s "
            "(session=%s, mode=%s, ~%s tokens)",
            _hyg_codex_outcome, session_entry.session_id, _hyg_codex_auto, f"{plan.approx_tokens:,}",
        )

    async def _hmwa_hygiene_build_agent(self, _hyg_model, _hyg_runtime, session_entry):
        """Build the detached hygiene ``AIAgent`` with the live session's system prompt. Returns
        ``(agent, sync_session_db)``."""
        from gateway.run import _GATEWAY_HYGIENE_PLATFORM, _seed_hygiene_system_prompt
        from run_agent import AIAgent
        try:
            _hyg_session_row = await self._session_db.get_session(session_entry.session_id)
        except Exception as exc:
            _hyg_session_row = None
            logger.warning(
                "Session hygiene could not restore the system "
                "prompt for session %s: %s. Preserving an empty "
                "prompt so the live turn rebuilds it with its "
                "configured providers.",
                session_entry.session_id, exc, exc_info=True,
            )
        _hyg_session_db = getattr(self._session_db, "_db", self._session_db)
        # Hygiene is the same lossy rewrite as normal compression: with
        # compression.checkpoint_required on, load the memory provider so the checkpoint exists
        # before any mutation; otherwise keep the fast path (no provider init).
        from hermes_cli.config import load_config as _load_cfg
        from utils import is_truthy_value as _is_truthy

        _hyg_checkpoint_required = _is_truthy(
            ((_load_cfg() or {}).get("compression") or {}).get("checkpoint_required"), default=False,
        )
        _hyg_agent = AIAgent(
            **_hyg_runtime,
            model=_hyg_model,
            max_iterations=4,
            quiet_mode=True,
            skip_memory=not _hyg_checkpoint_required,
            enabled_toolsets=["memory"],
            session_id=session_entry.session_id,
            session_db=_hyg_session_db,
        )
        _seed_hygiene_system_prompt(_hyg_agent, _hyg_session_row)
        # If compression must rebuild instead of retaining the cached prompt, make the persisted
        # result deliberately stale for every real gateway surface.
        _hyg_agent.platform = _GATEWAY_HYGIENE_PLATFORM
        return _hyg_agent, _hyg_session_db

    async def _hmwa_hygiene_detached_attempt(
        self, attempt, hs, plan, history, _hyg_msgs, _hyg_model, _hyg_runtime,
        source, session_entry, session_key, _quick_key, run_generation,
    ):
        """Run one detached hygiene compression attempt end to end; publishes the transcript to
        continue with (compressed or original) on ``attempt.history``."""
        from gateway.run import HygieneTurnHoldExceeded
        from agent.conversation_compression import CompressionCommitFence
        _hyg_agent, _hyg_session_db = await self._hmwa_hygiene_build_agent(_hyg_model, _hyg_runtime, session_entry)
        attempt.agent = _hyg_agent
        try:
            # Hygiene runs before the turn and owns the session binding, so prefer in-place
            # compaction: archive old rows under the same session id rather than minting a
            # continuation child that must be published back to SessionStore/topic bindings.
            # Without a SessionDB this stays False and the apply guard preserves it.
            _hyg_agent.compression_in_place = True
            _bind_hyg_state = getattr(getattr(_hyg_agent, "context_compressor", None), "bind_session_state", None)
            if callable(_bind_hyg_state):
                _bind_hyg_state(_hyg_session_db, session_entry.session_id)
            # Never finalize on close() — that would end the live gateway session row.
            _hyg_agent._end_session_on_close = False
            _hyg_agent._print_fn = lambda *a, **kw: None

            loop = asyncio.get_running_loop()
            _hyg_commit_fence = CompressionCommitFence(total_ceiling_seconds=hs.total_ceiling_seconds)
            # Default executor (NOT self._get_executor): a fence-cancelled hung summary must never
            # occupy an agent-work slot. But it MUST run in the caller's contextvars: under
            # multiplex_profiles the secret scope / HERMES_HOME live in ContextVars, and an empty
            # Context makes get_secret() fail closed → lossy truncation.
            attempt.commit_fence = _hyg_commit_fence
            attempt.future = loop.run_in_executor(
                None,
                copy_context().run,
                lambda: _hyg_agent._compress_context(
                    _hyg_msgs, "", approx_tokens=plan.approx_tokens, commit_fence=_hyg_commit_fence,
                ),
            )
            attempt.wait_started = time.monotonic()
            try:
                _compressed = await self._hmwa_hygiene_wait_for_summary(attempt, hs, session_entry)
            except HygieneTurnHoldExceeded:
                _compressed = await self._hmwa_hygiene_on_turn_hold(
                    attempt, hs, session_entry, session_key, source,
                )
            except asyncio.TimeoutError:
                _compressed = await self._hmwa_hygiene_on_timeout(
                    attempt, hs, session_entry, session_key, source,
                )
            except BaseException:
                self._hmwa_hygiene_on_unwind(attempt, hs, session_entry, session_key)
                raise

            await self._hmwa_hygiene_apply_result(
                attempt, hs, _compressed, history, plan,
                session_entry=session_entry,
                session_key=session_key,
                source=source,
                _quick_key=_quick_key,
                run_generation=run_generation,
            )
        finally:
            # Evict the cached agent so the next turn rebuilds its system prompt from current
            # SOUL.md, memory, and skills.
            self._evict_cached_agent(session_key)
            if not attempt.cleanup_deferred:
                await self._cleanup_agent_resources_off_loop(_hyg_agent, context="session hygiene")

    async def _hmwa_run_session_hygiene(
        self, event, source, session_entry, session_key, history, _quick_key, run_generation,
    ):
        """Auto-compress pathologically large transcripts before the agent starts so oversized
        histories don't cause repeated truncation/context failures. Token source: the API's
        prompt_tokens from the last turn, else a char/4 estimate."""
        from gateway.run import HygieneTurnHoldExceeded
        if not history or len(history) < 4:
            return history

        hs = await self._hmwa_hygiene_settings(source, session_key)
        if not hs.compression_enabled:
            return history
        plan = await self._hmwa_hygiene_plan(hs, history, session_entry, session_key)
        if not plan.needs_compress:
            return history

        attempt = self._HygieneAttempt(
            agent=None,
            meta=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
            history=history,
        )
        try:
            _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
                user_config=hs.data if isinstance(hs.data, dict) else None,
            )
            if str(_hyg_runtime.get("api_mode") or "").lower() == "codex_app_server":
                await self._hmwa_hygiene_codex_compaction(
                    hs, plan, history, session_entry, session_key, _hyg_runtime,
                )
            elif _hyg_runtime.get("api_key"):
                # Pass the FULL transcript (tool results included), matching the agent loop:
                # filtering to user/assistant starved the compressor — tool results are the bulk
                # of context and short histories tripped the protect-first/last early-return.
                _hyg_msgs = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
                if len(_hyg_msgs) >= 4:
                    await self._hmwa_hygiene_detached_attempt(
                        attempt, hs, plan, history, _hyg_msgs, _hyg_model, _hyg_runtime,
                        source, session_entry, session_key, _quick_key, run_generation,
                    )
        except HygieneTurnHoldExceeded:
            # Availability boundary, not a failure — already logged at INFO by the turn-hold
            # handler; the generic warning below made thinking-model deployments read as broken.
            pass
        except Exception as e:
            logger.warning("Session hygiene auto-compress failed: %s", e)
        return attempt.history

    async def _hmwa_first_contact_notes(self, source, history, turn_sidecar_notes):
        """First-ever-message onboarding note + one-time 'no home channel' prompt (both only when
        the session has no history). Delivered on the user message (sidecar), NOT the ephemeral
        system prompt: present-on-turn-1/absent-on-turn-2 was a guaranteed prompt diff + rebuild."""
        from gateway.run import _hermes_home, _home_target_env_var, _load_gateway_config
        if history:
            return
        if not await self.async_session_store.has_any_sessions():
            _intro_note = (
                "[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
            # Opt-in profile-build path: when onboarding.profile_build is "ask" (default) and not
            # yet offered on this install, swap the plain intro for a consent-gated directive that
            # offers to build a user profile via memory(target="user"). Fires at most once.
            try:
                from agent.onboarding import (
                    PROFILE_BUILD_FLAG,
                    is_seen,
                    mark_seen,
                    profile_build_directive,
                    profile_build_mode,
                )
                _onb_cfg = _load_gateway_config()
                if profile_build_mode(_onb_cfg) == "ask" and not is_seen(_onb_cfg, PROFILE_BUILD_FLAG):
                    turn_sidecar_notes.append(profile_build_directive().strip())
                    mark_seen(_hermes_home / "config.yaml", PROFILE_BUILD_FLAG)
                else:
                    turn_sidecar_notes.append(_intro_note)
            except Exception as _pb_err:
                logger.debug("Profile-build onboarding directive failed, using plain intro: %s", _pb_err)
                turn_sidecar_notes.append(_intro_note)

        # One-time prompt if no home channel is set for this platform. Skipped for webhooks —
        # they deliver directly to configured targets (github_comment, etc.).
        if not source.platform or source.platform in (Platform.LOCAL, Platform.WEBHOOK):
            return
        platform_name = source.platform.value
        env_key = _home_target_env_var(platform_name)
        # Multiplex: the home channel may live only in the profile secret scope / PlatformConfig,
        # not process os.environ.
        home_env = ""
        try:
            from agent.secret_scope import get_secret

            home_env = (get_secret(env_key) or "").strip() if env_key else ""
        except Exception:
            home_env = ""
        if not home_env:
            home_env = (os.getenv(env_key) or "").strip() if env_key else ""
        # Also honor in-memory / yaml home_channel on this platform.
        try:
            if not home_env and self.config.get_home_channel(source.platform):
                home_env = "set"
        except Exception:
            pass
        # Secondary-profile platforms (e.g. Slack on yolo) may only exist under that profile's
        # loaded config — re-read live config inside the already-installed scope.
        if not home_env:
            try:
                from gateway.config import load_gateway_config as _lgc
                prof = (getattr(source, "profile", None) or "").strip()
                if prof and prof != "default" and _lgc().get_home_channel(source.platform):
                    home_env = "set"
            except Exception:
                pass
        if not home_env:
            # Slack routes every Hermes command through the single parent slash command
            # `/hermes`; bare `/sethome` is not registered and would fail.
            sethome_cmd = "/hermes sethome" if source.platform == Platform.SLACK else "/sethome"
            await self._deliver_platform_notice(
                source,
                f"📬 No home channel is set for {platform_name.title()}. "
                f"A home channel is where Hermes delivers cron job results "
                f"and cross-platform messages.\n\n"
                f"Type {sethome_cmd} to make this chat your home channel, "
                f"or ignore to skip.",
            )

    def _hmwa_apply_message_timestamp(self, event, message_text):
        """Capture the platform event time as message metadata and keep the persisted transcript
        clean (strip any leading timestamp prefix) regardless of the toggle; only the in-context
        RENDER is gated behind gateway.message_timestamps.enabled (default OFF)."""
        from gateway.run import _load_gateway_config, _message_timestamps_enabled
        persist_user_message = None
        persist_user_timestamp = None
        try:
            from hermes_time import get_timezone as _get_evt_tz
            from gateway.message_timestamps import (
                coerce_message_timestamp as _coerce_msg_ts,
                render_user_content_with_timestamp as _render_msg_ts,
                strip_leading_message_timestamps as _strip_msg_ts,
            )
            _evt_tz = _get_evt_tz()
            if message_text and isinstance(message_text, str):
                _clean_message_text, _embedded_ts = _strip_msg_ts(message_text, tz=_evt_tz)
                persist_user_message = _clean_message_text
                _event_epoch = _coerce_msg_ts(getattr(event, "timestamp", None), tz=_evt_tz)
                persist_user_timestamp = _event_epoch if _event_epoch is not None else _embedded_ts
                if _message_timestamps_enabled(_load_gateway_config()):
                    message_text = _render_msg_ts(_clean_message_text, persist_user_timestamp, tz=_evt_tz)
                else:
                    # Toggle off: the model sees the clean message; the timestamp is still stored
                    # as metadata for later opt-in.
                    message_text = _clean_message_text
        except Exception as _ts_err:
            logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err)
        return message_text, persist_user_message, persist_user_timestamp

    async def _hmwa_stop_typing_for_turn(self, event, source):
        """Stop the typing indicator (never raises). Slack AI status is scoped to a thread/
        workspace, so preserve the routing metadata used by the response delivery path."""
        try:
            _typing_adapter = self._adapter_for_source(source)
            _stop_with_metadata = getattr(type(_typing_adapter), "_stop_typing_with_metadata", None)
            _stop_typing = getattr(type(_typing_adapter), "stop_typing", None)
            if _typing_adapter and callable(_stop_with_metadata):
                await _typing_adapter._stop_typing_with_metadata(
                    source.chat_id,
                    self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
                )
            elif _typing_adapter and callable(_stop_typing):
                await _typing_adapter.stop_typing(source.chat_id)
        except Exception:
            pass

    async def _hmwa_shape_agent_response(
        self, agent_result, source, history, session_entry, session_key,
        _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
    ):
        """Turn the raw agent result into the outbound text: sentinel/silence handling, response
        logging, resume-pending clear, empty-response normalization, and identity-guarded
        post-compression session_id propagation. Returns
        ``(response, _intentional_silence, agent_messages)``."""
        from gateway.run import (
            _is_gateway_hidden_reasoning_incomplete_turn,
            _normalize_empty_agent_response,
            _sanitize_gateway_final_response,
            _should_clear_resume_pending_after_turn,
        )
        response = agent_result.get("final_response") or ""
        # Hidden-reasoning-only retry exhaustion: the loop's sentinel text doubles as
        # final_response and would be delivered verbatim — where peer agents can ingest it as a
        # completed assistant turn.
        if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
            response = ""
        try:
            from gateway.response_filters import is_intentional_silence_agent_result
            _intentional_silence = is_intentional_silence_agent_result(agent_result, response)
        except Exception:
            _intentional_silence = False

        # "(empty)" = the model produced no visible content after exhausting all retries.
        if response == "(empty)" and not _intentional_silence:
            response = (
                "⚠️ The model returned no response after processing tool "
                "results. This can happen with some models — try again or "
                "rephrase your question."
            )
        agent_messages = agent_result.get("messages", [])
        logger.info(
            "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
            _platform_name, source.chat_id or "unknown",
            time.time() - _msg_start_time, agent_result.get("api_calls", 0), len(response),
        )

        # Successful turn: clear the stuck-loop counter (accumulates only across CONSECUTIVE
        # restarts where the session never completed) and resume_pending (set by drain-timeout
        # shutdown) so later messages don't get the restart-interruption system note.
        if session_key and _should_clear_resume_pending_after_turn(agent_result):
            await self._clear_restart_failure_count(session_key)
            try:
                await self.async_session_store.clear_resume_pending(session_key)
            except Exception as _e:
                logger.debug("clear_resume_pending failed for %s: %s", session_key, _e)

        # Normalize empty responses: surface errors, partial failures, and work-without-text.
        if not _intentional_silence:
            response = _normalize_empty_agent_response(agent_result, response, history_len=len(history))
            response = _sanitize_gateway_final_response(source.platform, response)

        # Ordering contract: the agent thread already updated the contextvar in
        # conversation_compression.py; propagate to SessionEntry + _save() — but only if the
        # binding still points at the session this run was launched against.
        if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
            if session_entry.session_id == _run_start_session_id:
                session_entry.session_id = agent_result["session_id"]
                # The held turn lease follows the rotation: transcript persistence writes to the
                # NEW id, so the serialization boundary must move with it.
                self._rebind_turn_lease(_quick_key, run_generation, session_entry.session_id)
                await self.async_session_store._save()
                await self.async_session_store._record_gateway_session_peer(
                    session_entry.session_id, session_key, source,
                )
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="agent-result-compression",
                )
            else:
                logger.info(
                    "Skipping agent-result session split sync for %s because "
                    "the session binding moved from %s to %s before "
                    "compression finished",
                    session_key or "?", _run_start_session_id, session_entry.session_id,
                )
        return response, _intentional_silence, agent_messages

    # reasoning_style → (header line, per-line quote prefix for blank / non-blank lines)
    _REASONING_QUOTE_STYLES = {
        "subtext": ("-# 💭 Reasoning", "-# ", "-#"),
        "blockquote": ("> 💭 **Reasoning:**", "> ", ">"),
    }

    def _hmwa_prepend_reasoning(self, agent_result, response, source, _intentional_silence):
        """Prepend the last reasoning block when show_reasoning is on for this platform. Mattermost
        requires an explicit per-platform opt-in (scratch text, not final-answer content)."""
        from gateway.run import _load_gateway_config, _platform_config_key, _resolve_gateway_display_bool
        try:
            _show_reasoning_effective = _resolve_gateway_display_bool(
                _load_gateway_config(),
                _platform_config_key(source.platform),
                "show_reasoning",
                default=bool(getattr(self, "_show_reasoning", False)),
                platform=source.platform,
                require_platform_override_for={Platform.MATTERMOST},
            )
        except Exception:
            _show_reasoning_effective = (
                False if source.platform == Platform.MATTERMOST else getattr(self, "_show_reasoning", False)
            )
        if not (_show_reasoning_effective and response and not _intentional_silence):
            return response
        last_reasoning = agent_result.get("last_reasoning")
        if not last_reasoning:
            return response
        from gateway.stream_consumer import escape_code_fences_for_display
        # Collapse long reasoning to keep messages readable
        lines = last_reasoning.strip().splitlines()
        if len(lines) > 15:
            display_reasoning = "\n".join(lines[:15]) + f"\n_... ({len(lines) - 15} more lines)_"
        else:
            display_reasoning = last_reasoning.strip()
        # Render style is per-platform: Discord defaults to "-# " subtext (native small grey
        # metadata text); other platforms keep the fenced code block.
        try:
            from gateway.display_config import resolve_display_setting
            _reasoning_style = resolve_display_setting(
                _load_gateway_config(), _platform_config_key(source.platform), "reasoning_style", "code",
            )
        except Exception:
            _reasoning_style = "code"
        _quote = self._REASONING_QUOTE_STYLES.get(_reasoning_style)
        if _quote:
            header, prefix, empty = _quote
            _quoted = "\n".join(f"{prefix}{ln}" if ln else empty for ln in display_reasoning.splitlines())
            return f"{header}\n{_quoted}\n\n{response}"
        # Escape ``` inside reasoning so inner fences don't break the outer code block.
        display_reasoning = escape_code_fences_for_display(display_reasoning)
        return f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

    def _hmwa_runtime_footer_line(self, agent_result, source, _turn_seconds):
        """Runtime-metadata footer for the FINAL message of the turn; off by default
        (display.runtime_footer.enabled=false)."""
        from gateway.run import _load_gateway_config, _platform_config_key, _terminal_scope_cwd
        try:
            from gateway.runtime_footer import build_footer_line as _bfl
            return _bfl(
                user_config=_load_gateway_config(),
                platform_key=_platform_config_key(source.platform),
                model=agent_result.get("model"),
                context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                context_length=agent_result.get("context_length") or None,
                cwd=_terminal_scope_cwd(""),
                turn_seconds=_turn_seconds,
            )
        except Exception as _footer_err:
            logger.debug("runtime_footer build failed: %s", _footer_err)
            return ""

    async def _hmwa_post_turn_hooks(self, hook_ctx, agent_result, response):
        """agent:end hook, process-watcher scheduling, and watch-notification drain."""
        await self.hooks.emit("agent:end", {
            **hook_ctx,
            "response": (response or "")[:500],
            "model": agent_result.get("model", ""),
            "provider": agent_result.get("provider", ""),
        })

        # Pending process watchers (check_interval on background processes)
        try:
            from tools.process_registry import process_registry
            # Detach the current batch atomically: reassign to a fresh list so a watcher appended
            # by a concurrent session during the yield isn't dropped by clear().
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            for i, watcher in enumerate(watchers):
                asyncio.create_task(self._run_process_watcher(watcher))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Process watcher setup error: %s", e)

        # Drain watch notifications that arrived during the run. The queue also carries process
        # completions (per-process watcher task above) and async-delegation completions (owned by
        # _async_delegation_watcher) — inject only watch-type events, leave the rest queued.
        try:
            from tools.process_registry import process_registry as _pr
            await self._drain_watch_notifications(_pr.completion_queue)
        except Exception as e:
            logger.debug("Watch queue drain error: %s", e)

    _CONTEXT_OVERFLOW_ERROR_PHRASES = (
        "context length", "context size", "context window",
        "maximum context", "token limit", "too many tokens",
        "reduce the length", "exceeds the limit",
        "request entity too large", "prompt is too long",
        "payload too large", "input is too long",
    )

    def _hmwa_classify_turn_failure(self, agent_result, history, session_entry):
        """Classify a finished turn for transcript persistence. Returns
        ``(agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure)``.

        Context-overflow failures (compression exhausted, generic 400 on large sessions) must
        NOT persist the user message — it would grow the session and reproduce the failure
        forever. Transient failures (429, timeout, connection error, 5xx) DO persist it: the
        session is not oversized and dropping the turn causes severe context loss on retry."""
        from gateway.run import _is_gateway_hidden_reasoning_incomplete_turn
        agent_failed_early = bool(agent_result.get("failed"))
        hidden_reasoning_incomplete = _is_gateway_hidden_reasoning_incomplete_turn(agent_result)
        _err = str(agent_result.get("error", "")).lower()
        # Specific multi-word phrases (not bare "exceed"/"token") avoid false positives on
        # transient errors such as "rate limit exceeded"; matches run_agent.py's classifier.
        is_context_overflow_failure = agent_failed_early and (
            bool(agent_result.get("compression_exhausted"))
            or any(p in _err for p in self._CONTEXT_OVERFLOW_ERROR_PHRASES)
            or ("400" in _err and len(history) > 50)
        )
        if is_context_overflow_failure:
            logger.info(
                "Skipping transcript persistence for context-overflow "
                "failure in session %s to prevent session growth loop.",
                session_entry.session_id,
            )
        elif agent_failed_early:
            logger.info(
                "Transient agent failure in session %s — persisting user "
                "message so conversation context is preserved on retry.",
                session_entry.session_id,
            )
        elif hidden_reasoning_incomplete:
            logger.warning(
                "Suppressing hidden-reasoning-only incomplete gateway turn "
                "for session %s: %s",
                session_entry.session_id, agent_result.get("error", "processing incomplete"),
            )
        return agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure

    async def _hmwa_compression_exhaustion_reset(
        self, agent_result, response, session_entry, session_key, source,
    ):
        """Auto-reset a permanently oversized session so the next message starts fresh instead of
        replaying the oversized context forever. Never on a lock-contended defer — that is the
        OPPOSITE case (a concurrent path holds the lock and is shrinking it). Returns
        ``(response, session_entry)``."""
        if agent_result.get("compression_deferred"):
            logger.info(
                "Compression deferred for session %s — the compression "
                "lock is held by a concurrent compressor. Keeping the "
                "session intact; the next message retries normally.",
                session_entry.session_id if session_entry else "?",
            )
        elif agent_result.get("compression_exhausted") and session_entry and session_key:
            logger.info("Auto-resetting session %s after compression exhaustion.", session_entry.session_id)
            new_entry = await self.async_session_store.reset_session(session_key)
            self._evict_cached_agent(session_key)
            # Conversation boundary: one funnel call clears every conversation-scoped per-session
            # dict (see _CONVERSATION_SCOPED_STATE).
            self._clear_conversation_scope(session_key, reason="compression_exhausted_reset")
            if new_entry is not None:
                # Re-point the Telegram topic binding at the fresh session: compression rotated
                # session_entry.session_id to the bloated child earlier this turn and that _sync
                # also rewrote the (chat_id, thread_id) binding. Without a re-sync the binding-heal
                # walk switches the next inbound message back onto the child and re-triggers
                # exhaustion forever. No-op on non-topic lanes.
                session_entry = new_entry
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="compression-exhausted-reset",
                )
            response = (response or "") + (
                "\n\n🔄 Session auto-reset — the conversation exceeded the "
                "maximum context size and could not be compressed further. "
                "Your next message will start a fresh session."
            )
        return response, session_entry

    @staticmethod
    def _hmwa_user_transcript_entry(
        event, message_text, persist_user_message, persist_user_timestamp, persist_user_display_kind, ts,
    ):
        """Transcript row for the inbound user turn (clean text + event time when captured)."""
        _user_entry = {
            "role": "user",
            "content": persist_user_message if persist_user_message is not None else message_text,
            "timestamp": persist_user_timestamp if persist_user_timestamp is not None else ts,
        }
        if persist_user_display_kind:
            _user_entry["display_kind"] = persist_user_display_kind
        if getattr(event, "message_id", None):
            _user_entry["message_id"] = str(event.message_id)
        return _user_entry

    async def _hmwa_persist_turn_transcript(
        self, *, event, source, session_entry, session_key, agent_result, agent_messages,
        history, response, message_text, persist_user_message, persist_user_timestamp,
        persist_user_display_kind, agent_failed_early, hidden_reasoning_incomplete,
        is_context_overflow_failure,
    ):
        """Persist this turn to the transcript (session_meta on first turn, user-only on transient
        failure, nothing on context overflow), update last_prompt_tokens, and re-baseline the
        cached agent's message count."""
        from gateway.run import _resolve_gateway_model
        ts = time.time()  # Unix epoch float — consistent with DB storage
        store = self.async_session_store
        sid = session_entry.session_id
        # The agent already persisted this turn's rows via _flush_messages_to_session_db() (the
        # codex app-server runtime too: it flushes its own projection and reports
        # agent_persisted=True); skip the DB write to avoid duplicates. Default = a session DB
        # exists; a non-persisting runtime opts in via False.
        agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)

        def _user_entry():
            return self._hmwa_user_transcript_entry(
                event, message_text, persist_user_message, persist_user_timestamp,
                persist_user_display_kind, ts,
            )

        if is_context_overflow_failure:
            pass  # Skip all transcript writes — don't grow a broken session
        elif not history:
            # Fresh session: write the full tool definitions as the first entry so the transcript
            # is self-describing — the same dicts sent as tools=[...] in the API request.
            await store.append_to_transcript(sid, {
                "role": "session_meta",
                "tools": agent_result.get("tools", []) or [],
                "model": _resolve_gateway_model(),
                "platform": source.platform.value if source.platform else "",
                "timestamp": ts,
            })

        if is_context_overflow_failure:
            pass
        elif agent_failed_early or hidden_reasoning_incomplete:
            # Transient failure (429/timeout/5xx): persist only the user message so the next
            # message loads a transcript that reflects what was said; the assistant error text is
            # a gateway-generated hint, not model output. Hidden-reasoning incomplete turns follow
            # the same rule so peer-agent channels don't ingest them.
            # Dedupe on platform message_id (Telegram retries after transient failures).
            if event.message_id and await store.has_platform_message_id(sid, str(event.message_id)):
                logger.info(
                    "Skipping duplicate user turn "
                    "(message_id=%s) in session %s",
                    event.message_id, sid,
                )
            else:
                await store.append_to_transcript(sid, _user_entry(), skip_db=agent_persisted)
        else:
            # Only the NEW messages from this turn: use history_offset (what the agent saw), not
            # len(history), which counts session_meta entries stripped before the agent saw them.
            history_len = agent_result.get("history_offset", len(history))
            new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
            if not new_messages:
                # Edge case: fall back to simple user/assistant rows.
                await store.append_to_transcript(sid, _user_entry(), skip_db=agent_persisted)
                if response:
                    await store.append_to_transcript(
                        sid, {"role": "assistant", "content": response, "timestamp": ts},
                        skip_db=agent_persisted,
                    )
            else:
                # Attach the inbound platform message_id to the first user entry written this turn
                # so platform-level quote-resolution (e.g. Yuanbao QuoteContextMiddleware's
                # transcript fallback) can find earlier @bot messages by their original id.
                _user_msg_id_attached = False
                for msg in new_messages:
                    if msg.get("role") == "system":
                        continue  # rebuilt each run
                    entry = {**msg, "timestamp": ts}
                    if (
                        not _user_msg_id_attached
                        and msg.get("role") == "user"
                        and event.message_id
                        and "message_id" not in entry
                    ):
                        entry["message_id"] = str(event.message_id)
                        _user_msg_id_attached = True
                    await store.append_to_transcript(sid, entry, skip_db=agent_persisted)

        # The agent persists token counts and model itself; keep only last_prompt_tokens here for
        # context-window tracking and compression decisions.
        await store.update_session(
            session_entry.session_key,
            last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
            touch_activity=not bool(getattr(event, "internal", False)),
        )

        # Re-baseline the cached agent's message_count snapshot now that ALL of this turn's writes
        # are done (flushed rows AND the first-turn `session_meta` marker, which bumps the count
        # too). The cross-process coherence guard snapshots at agent-BUILD time and never
        # refreshes on reuse, so our own writes would trigger a rebuild next turn (destroying
        # prompt caching).
        await self._refresh_agent_cache_message_count(session_key, sid)

    async def _hmwa_deliver_turn_response(
        self, event, source, session_entry, session_key, run_generation,
        agent_result, agent_messages, response, _footer_line, _intentional_silence,
    ):
        """Final delivery decisions: intentional silence, voice reply, streamed-turn media/footer.
        Returns the text for the adapter to send, or ``None`` when already delivered."""
        # Intentional silence is a delivery decision, not a transcript mutation: the [SILENT]
        # assistant turn stays persisted so later turns keep user/assistant alternation.
        if _intentional_silence:
            logger.info("Suppressing intentional silence marker for session %s", session_entry.session_id)
            response = ""

        adapter = self._adapter_for_source(source)
        # Auto voice reply (TTS audio before the text) unless streaming TTS already delivered
        # audio for this turn.
        _streaming_tts_done = adapter is not None and bool(
            getattr(adapter, "_streaming_tts_turn_completed", lambda *_a, **_k: False)(session_key, run_generation)
        )
        if not _streaming_tts_done and self._should_send_voice_reply(
            event, response, agent_messages, already_sent=bool(agent_result.get("already_sent")),
        ):
            await self._send_voice_reply(event, response)

        # Streamed responses still need MEDIA: files delivered before returning None (chunks carry
        # the tags verbatim and post-processing is skipped when already_sent). Never skip when the
        # agent failed: the error text is new content streaming didn't show.
        if agent_result.get("already_sent") and not agent_result.get("failed"):
            if response and adapter:
                await self._deliver_media_from_response(response, event, adapter)
            # Streaming delivered the body, but the footer was held back (`not already_sent` gate).
            if _footer_line and adapter:
                try:
                    await adapter.send(
                        source.chat_id, _footer_line,
                        metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
                    )
                except Exception as _e:
                    logger.debug("trailing footer send failed: %s", _e)
            # Return None so the adapter does not send the body twice; /loop and /goal hooks in
            # _handle_message read the return value, so stash the delivered text on the event or
            # those hooks never run and a /loop tick stays awaiting.
            with suppress(Exception):
                event._streamed_final_response = str(response or "")
            return None

        return response

    _STATUS_HINTS = {
        401: " Check your API key or run `claude /login` to refresh OAuth credentials.",
        402: " Your API balance or quota is exhausted. Check your provider dashboard.",
        529: " The API is temporarily overloaded. Please try again shortly.",
    }

    async def _hmwa_agent_error_reply(
        self, e, event, source, session_entry, session_key, history, message_text,
        persist_user_message, persist_user_timestamp, persist_user_display_kind,
    ):
        """``except Exception`` body of the agent turn: stop typing, log, persist the inbound user
        turn once, and build the sanitized user-facing error reply."""
        # Retain Slack thread/workspace routing so a failed turn cannot leave its status visible.
        await self._hmwa_stop_typing_for_turn(event, source)
        logger.exception("Agent error in session %s", session_key)
        # Crash-resilience for failures before AIAgent enters run_conversation() (e.g. provider/
        # httpx client init): the agent can't persist the inbound turn there, so append the user
        # message here once; skip if the latest user row already matches it.
        try:
            if message_text is not None and session_entry is not None:
                try:
                    _recent_transcript = await self.async_session_store.load_transcript(session_entry.session_id)
                except Exception:
                    _recent_transcript = []
                _expected_user_content = (
                    persist_user_message if persist_user_message is not None else message_text
                )
                _already_persisted = False
                for _msg in reversed(_recent_transcript[-10:]):
                    if _msg.get("role") == "user":
                        _already_persisted = _msg.get("content") == _expected_user_content
                        break
                if not _already_persisted:
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id,
                        self._hmwa_user_transcript_entry(
                            event, message_text, persist_user_message, persist_user_timestamp,
                            persist_user_display_kind, time.time(),
                        ),
                    )
        except Exception:
            logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
        # Log full details server-side only; never expose raw exception types or messages to end
        # users (info-leakage risk).
        status_code = getattr(e, "status_code", None)
        status_hint = self._STATUS_HINTS.get(status_code, "")
        if status_code == 429:
            # Plan usage limit (resets on a schedule) vs a transient rate limit
            _err_body = getattr(e, "response", None)
            _err_json = {}
            try:
                if _err_body is not None:
                    _err_json = _err_body.json().get("error", {})
                    if not isinstance(_err_json, dict):
                        _err_json = {}
            except Exception:
                pass
            if _err_json.get("type") == "usage_limit_reached":
                _resets_in = _err_json.get("resets_in_seconds")
                if _resets_in and _resets_in > 0:
                    import math
                    _hours = math.ceil(_resets_in / 3600)
                    status_hint = f" Your plan's usage limit has been reached. It resets in ~{_hours}h."
                else:
                    status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
            else:
                status_hint = " You are being rate-limited. Please wait a moment and try again."
        elif status_code in {400, 500}:
            # 400 on a large session is context overflow; 500 on a large session often means the
            # payload is too large for the API — treat it the same way.
            if len(history) > 50:
                return (
                    "⚠️ Session too large for the model's context window.\n"
                    "Use /compact to compress the conversation, or "
                    "/reset to start fresh."
                )
            elif status_code == 400:
                status_hint = " The request was rejected by the API."
        return (
            f"Sorry, I encountered an unexpected error.{status_hint}\n"
            "Try again or use /reset to start a fresh session."
        )

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        from gateway.run import _load_gateway_config
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", (event.text or "")[:80].replace("\n", " "),
            getattr(event, "reply_to_message_id", None),
            (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " "),
        )

        resolved = await self._hmwa_resolve_session(event, source)
        if resolved is None:
            return
        source, session_entry, session_key = resolved
        _was_auto_reset, _is_new_session = await self._hmwa_open_session(session_entry, session_key, source)

        context = build_session_context(source, self.config, session_entry)
        # Session context variables for tools (task-local, concurrency-safe)
        _session_env_tokens = self._set_session_env(context)

        # Synthetic self-injected turns (batch completions, watch notifications, resume wake-ups)
        # arrive as MessageEvent(internal=True). Persist with display_kind="internal_notification"
        # so UIs render timeline notices, not user bubbles. display_kind is a DB-only sidecar
        # stripped from every provider-bound payload; role/content untouched.
        persist_user_display_kind = "internal_notification" if getattr(event, "internal", False) else None
        _redact_pii = False  # privacy.redact_pii, re-read per message
        try:
            _redact_pii = bool((_load_gateway_config().get("privacy") or {}).get("redact_pii", False))
        except Exception:
            pass

        # The context prompt render is pinned per session, keyed by a hash of the exact renderer
        # inputs (_ephemeral_change_key): a hit reuses the pinned bytes so the system prompt cannot
        # drift turn-over-turn; a miss (thread rename, /sethome, redact_pii flip) re-renders.
        context_prompt = self._pinned_session_context_prompt(context, _redact_pii, session_key)

        # Per-turn must-deliver notes ride the user message via the api_content sidecar (staged
        # below, consumed in run_sync → build_turn_context), NOT context_prompt: appending them to
        # the ephemeral system prompt guaranteed a turn1→turn2 diff and a full agent rebuild.
        turn_sidecar_notes: List[str] = []
        if _was_auto_reset:
            await self._hmwa_deliver_auto_reset_notice(session_entry, source, turn_sidecar_notes)

        # Auto-load skill(s) for topic/channel bindings (single name or ordered list) — only on NEW
        # sessions; ongoing conversations already carry the skill content in their history.
        _auto = getattr(event, "auto_skill", None)
        if _is_new_session and _auto:
            self._hmwa_auto_load_skills(event, _auto, _quick_key, session_key)

        await self._hmwa_acquire_turn_lease(_quick_key, run_generation, session_entry, _session_env_tokens)

        # A turn only becomes durable recovery work after it owns (or has explicitly degraded past)
        # the per-session lease. Marking before the await above would falsely recover an alias-
        # routed message that never began processing if the gateway died while it was still waiting.
        await self._mark_durable_active_turn(event, session_entry.session_key)

        # An unreadable canonical store is not an empty conversation: stop before the agent can
        # invent continuity from a plausible-looking []. This return happens before the broad
        # cleanup finally below, so restore task-local context here; the outer dispatch still
        # clears the durable marker and turn lease.
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            self._clear_session_env(_session_env_tokens)
            return (
                "⚠️ This session's history is temporarily unavailable, so "
                "this message was not processed. Ask the operator to inspect "
                "state.db, then resend after it is healthy. Use /reset only "
                "if you intentionally want to start a new conversation."
            )

        history = await self._hmwa_run_session_hygiene(
            event, source, session_entry, session_key, history, _quick_key, run_generation,
        )

        await self._hmwa_first_contact_notes(source, history, turn_sidecar_notes)

        # Voice channel state (who is present / speaking) rides the user message ONLY when changed
        # since the previous turn; in the ephemeral system prompt it forced a full agent rebuild +
        # prompt-cache re-key per message (the prompt carries a static pointer line instead).
        _vc_note = self._voice_channel_sidecar_note(event, source, session_key)
        if _vc_note:
            turn_sidecar_notes.append(_vc_note)

        # Auto-analyze user images (vision tool eagerly, image media_type only) so the model always
        # gets a text description plus the local path for re-examination via vision_analyze.
        message_text = await self._prepare_profile_scoped_inbound_message_text(
            event=event, source=source, history=history, session_key=session_key,
        )
        if message_text is None:
            return

        message_text, persist_user_message, persist_user_timestamp = (
            self._hmwa_apply_message_timestamp(event, message_text)
        )

        # Stage this turn's must-deliver notes (one-shot; consumed in run_sync) AFTER the
        # message_text early-out so an aborted turn cannot leak its notes into the next turn.
        if turn_sidecar_notes and session_key:
            self._set_pending_turn_sidecar_notes(session_key, turn_sidecar_notes)

        # Bind this run generation to the adapter's active-session event so deferred post-delivery
        # callbacks can be released by the same run that registered them.
        self._bind_adapter_run_generation(self._adapter_for_source(source), session_key, run_generation)

        try:
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(getattr(source, "thread_id", None)) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Capture the session id this run launches against so post-run compression publication
            # can be identity-guarded; a /new or another lifecycle transition may move
            # session_entry.session_id while the old run is still unwinding.
            _run_start_session_id = session_entry.session_id
            _turn_started_monotonic = time.monotonic()
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=_run_start_session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=self._reply_anchor_for_event(event),
                inbound_message_id=str(event.message_id) if event.message_id else None,
                channel_prompt=event.channel_prompt,
                moa_config=getattr(event, "_moa_config", None),
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=event.message_type,
            )
            _turn_seconds = time.monotonic() - _turn_started_monotonic

            await self._hmwa_stop_typing_for_turn(event, source)

            if not self._is_session_run_current(_quick_key, run_generation):
                logger.info(
                    "Discarding stale agent result for %s — generation %d is no longer current",
                    _quick_key or "?", run_generation,
                )
                _stale_adapter = self._adapter_for_source(source)
                if getattr(type(_stale_adapter), "pop_post_delivery_callback", None) is not None:
                    _stale_adapter.pop_post_delivery_callback(_quick_key, generation=run_generation)
                elif _stale_adapter and hasattr(_stale_adapter, "_post_delivery_callbacks"):
                    _stale_adapter._post_delivery_callbacks.pop(_quick_key, None)
                return None

            response, _intentional_silence, agent_messages = await self._hmwa_shape_agent_response(
                agent_result, source, history, session_entry, session_key,
                _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
            )
            response = self._hmwa_prepend_reasoning(agent_result, response, source, _intentional_silence)
            _footer_line = self._hmwa_runtime_footer_line(agent_result, source, _turn_seconds)
            # Streaming already delivered the body: the footer goes out as a trailing send instead.
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"
            await self._hmwa_post_turn_hooks(hook_ctx, agent_result, response)

            agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure = (
                self._hmwa_classify_turn_failure(agent_result, history, session_entry)
            )
            response, session_entry = await self._hmwa_compression_exhaustion_reset(
                agent_result, response, session_entry, session_key, source,
            )
            await self._hmwa_persist_turn_transcript(
                event=event,
                source=source,
                session_entry=session_entry,
                session_key=session_key,
                agent_result=agent_result,
                agent_messages=agent_messages,
                history=history,
                response=response,
                message_text=message_text,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                agent_failed_early=agent_failed_early,
                hidden_reasoning_incomplete=hidden_reasoning_incomplete,
                is_context_overflow_failure=is_context_overflow_failure,
            )
            return await self._hmwa_deliver_turn_response(
                event, source, session_entry, session_key, run_generation,
                agent_result, agent_messages, response, _footer_line, _intentional_silence,
            )

        except Exception as e:
            return await self._hmwa_agent_error_reply(
                e, event, source, session_entry, session_key, history, message_text,
                persist_user_message, persist_user_timestamp, persist_user_display_kind,
            )
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)

    def _reset_notice_session_info(self, source: SessionSource) -> str:
        """Session-info block for the auto-reset notice, profile-scoped.

        Under multiplexing, resolve model/provider/context inside the profile serving ``source``
        (mirrors ``_run_agent``'s gating) or the banner advertises the base config's model. Call
        via ``asyncio.to_thread``: resolution can block (credential refresh, context-length
        probes), and the scope is entered here so contextvars behave in the worker thread.
        """
        from gateway.run import _profile_runtime_scope
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            with _profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                return self._format_session_info()
        return self._format_session_info()

    def _format_session_info(self) -> str:
        """Resolve current model config and return a formatted info block.

        Surfaces model, provider, context length, and endpoint so gateway users can immediately
        see if context detection went wrong (e.g. local models falling to the 128K default).
        """
        from gateway.run import _resolve_gateway_model_context
        resolved = _resolve_gateway_model_context()
        model = resolved.model
        provider = resolved.provider
        base_url = resolved.base_url
        context_length = resolved.context_length

        # Format context source hint
        if resolved.context_source == "config":
            ctx_source = "config"
        elif resolved.context_source == "default":
            ctx_source = "default — set model.context_length in config to override"
        else:
            ctx_source = "detected"

        # Format context length for display
        if context_length >= 1_000_000:
            ctx_display = f"{context_length / 1_000_000:.1f}M"
        elif context_length >= 1_000:
            ctx_display = f"{context_length // 1_000}K"
        else:
            ctx_display = str(context_length)

        lines = [
            f"◆ Model: `{model}`",
            f"◆ Provider: {provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]

        # Show endpoint for local/custom setups
        if base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1", "0.0.0.0"):
            lines.append(f"◆ Endpoint: {base_url}")

        return "\n".join(lines)

    async def _run_background_task(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Profile-scoping wrapper around the background agent task.

        When multiplexing is active, resolve the inbound source's profile and run the whole task
        inside ``_profile_runtime_scope`` so credentials resolve from that profile's secret
        scope. Mirrors the pattern in ``_run_agent``.
        """
        from gateway.run import _profile_runtime_scope
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

        profile_home = self._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

    def _resolve_enabled_toolsets_for_source(
        self,
        user_config: dict,
        source: "SessionSource",
        platform_key: str,
    ) -> list:
        """Resolve enabled toolsets for an agent run, honoring per-source overrides.

        An adapter ``toolsets_for_source()`` override (e.g. per-route webhook toolsets) is
        validated through the SAME ``_get_platform_tools`` path as normal platform config, so
        unknown and platform-restricted toolsets are dropped rather than trusted. Absent an
        override, falls back to ``platform_toolsets.<platform>``.
        """
        from hermes_cli.tools_config import _get_platform_tools

        override = None
        try:
            adapter = self._adapter_for_source(source)
            if adapter is not None:
                override = adapter.toolsets_for_source(source)
        except Exception:
            override = None

        if override and isinstance(override, list):
            cfg = dict(user_config)
            pts = dict(cfg.get("platform_toolsets") or {})
            pts[platform_key] = [str(t) for t in override]
            cfg["platform_toolsets"] = pts
            return sorted(_get_platform_tools(cfg, platform_key))

        return sorted(_get_platform_tools(user_config, platform_key))

    async def _run_background_task_inner(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from gateway.run import (
            _checkpoint_agent_kwargs,
            _current_max_iterations,
            _load_gateway_config,
            _platform_config_key,
        )
        from run_agent import AIAgent

        media_urls = media_urls or []
        media_types = media_types or []

        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            enabled_toolsets = self._resolve_enabled_toolsets_for_source(
                user_config, source, platform_key
            )
            agent_cfg = user_config.get("agent") or {}
            from agent.skill_utils import parse_config_string_list

            disabled_toolsets = parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None

            pr = self._provider_routing
            max_iterations = _current_max_iterations()
            reasoning_config = self._resolve_session_reasoning_config(
                source=source, model=model
            )
            self._reasoning_config = reasoning_config
            self._service_tier = self._resolve_session_service_tier(source=source)
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions so the background
            # agent can see user-attached images (same as the main flow).
            enriched_prompt = prompt
            if media_urls:
                image_paths = []
                for i, path in enumerate(media_urls):
                    mtype = media_types[i] if i < len(media_types) else ""
                    if mtype.startswith("image/"):
                        image_paths.append(path)
                if image_paths:
                    try:
                        enriched_prompt = await self._enrich_message_with_vision(
                            prompt, image_paths,
                        )
                    except Exception as e:
                        logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    **_checkpoint_agent_kwargs(user_config),
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    # Reload from disk — do not reuse the startup snapshot (#60955).
                    fallback_model=self._refresh_fallback_model(),
                )
                try:
                    return agent.run_conversation(
                        user_message=enriched_prompt,
                        task_id=task_id,
                    )
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Background tasks start a fresh conversation, so history_offset=0: every message in the
            # run belongs to this turn. Mirrors the repair on the main turn path.
            if response:
                response = repair_explicit_computer_use_media_paths(
                    response,
                    result.get("messages", []),
                )

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                from gateway.platforms.base import BasePlatformAdapter
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    with suppress(Exception):
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                            metadata=_thread_metadata,
                        )

                # Route each media file by type so a TTS clip arrives as a voice bubble and a clip
                # as a video rather than a generic document. Mirrors the streaming + kanban paths.
                from gateway.platforms.base import (
                    should_send_media_as_audio as _should_send_media_as_audio,
                )
                _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
                for media_path, _is_voice in (media_files or []):
                    _ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                            await adapter.send_voice(
                                chat_id=source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                                is_voice=_is_voice,
                            )
                        elif _ext in _VIDEO_EXTS:
                            await adapter.send_video(
                                chat_id=source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _IMAGE_EXTS:
                            await adapter.send_image_file(
                                chat_id=source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await adapter.send_document(
                                chat_id=source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            with suppress(Exception):
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )

    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Actually disconnect, reconnect, and notify MCP tool changes.

        Split out so the confirmation wrapper can invoke the same path for button, text reply,
        or disabled confirm gate. Under multiplex the reload runs inside the requesting profile's
        runtime scope (entered here when the caller did not) and only that profile's servers are
        torn down and rediscovered.
        """
        from gateway.run import _profile_runtime_scope
        multiplex = bool(getattr(self.config, "multiplex_profiles", False))
        if multiplex and not get_hermes_home_override():
            profile_home = self._resolve_profile_home_for_source(event.source)
            with _profile_runtime_scope(Path(profile_home)):
                return await self._execute_mcp_reload(event)
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock
            from tools.mcp_tool import _server_scope_keys, reprobe_tool_availability
            from tools.registry import registry

            reload_scope = registry.current_scope_key() if multiplex else None

            def _scoped_server_names() -> set:
                with _lock:
                    return {
                        name for name in _servers
                        if reload_scope is None or _server_scope_keys.get(name) == reload_scope
                    }

            # Capture old server names before shutdown
            old_servers = _scoped_server_names()

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await self._run_in_executor_with_context(
                lambda: shutdown_mcp_servers(scope=reload_scope)
            )
            # Explicit reload also re-probes tool availability (check_fn).
            reprobe_tool_availability()

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await self._run_in_executor_with_context(discover_mcp_tools)

            # Compute what changed
            connected_servers = _scoped_server_names()
            if reload_scope is not None:
                from tools.mcp_tool import _mcp_tool_server_names

                with _lock:
                    new_tools = [
                        n for n in new_tools
                        if _mcp_tool_server_names.get(n) in connected_servers
                    ]

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = [t("gateway.reload_mcp.header")]
            if reconnected:
                lines.append(t("gateway.reload_mcp.reconnected", names=", ".join(sorted(reconnected))))
            if added:
                lines.append(t("gateway.reload_mcp.added", names=", ".join(sorted(added))))
            if removed:
                lines.append(t("gateway.reload_mcp.removed", names=", ".join(sorted(removed))))
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            # Refresh cached agents so existing sessions see new MCP tools on their next turn —
            # without this, the user has to `/new` (which discards conversation history) to pick up
            # tools from a server that was just added or reconnected.
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools
                _cache = getattr(self, "_agent_cache", None)
                _cache_lock = getattr(self, "_agent_cache_lock", None)
                if _cache_lock is not None and _cache:
                    # Multiplex: only this profile's sessions; rebuilding another profile's agent in
                    # this scope would hand it this profile's tool registry.
                    _ns_prefix = (
                        _session_key_namespace(event.source.profile) + ":"
                        if multiplex else None
                    )
                    with _cache_lock:
                        for _sess_key, _entry in list(_cache.items()):
                            if _ns_prefix and not str(_sess_key).startswith(_ns_prefix):
                                continue
                            try:
                                _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                            except Exception:
                                continue
                            if _agent is None:
                                continue
                            # Preserve each cached agent's build-time toolset selection EXACTLY: a
                            # gateway session built with a restricted enabled_toolsets (e.g.
                            # ["safe"]) must NOT silently gain tools after a reload. Unlike the
                            # CLI/TUI /reload-mcp (one user re-applying their own config), gateway
                            # agents are per-session and may be deliberately locked down.
                            refresh_agent_mcp_tools(_agent, quiet_mode=True)
            except Exception as _exc:
                logger.debug(
                    "Failed to update cached agent tools after MCP reload: %s",
                    _exc,
                )

            # Inject a message at the END of the session history so the model knows tools changed
            # next turn; appending after all existing messages preserves the prompt-cache prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = await self.async_session_store.get_or_create_session(event.source)
                await self.async_session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)

    def _get_proxy_url(self) -> Optional[str]:
        """Return the proxy URL if proxy mode is configured, else None.

        GATEWAY_PROXY_URL env var (Docker-friendly) wins over ``gateway.proxy_url`` in config.yaml.
        """
        from gateway.run import _load_gateway_config
        url = os.getenv("GATEWAY_PROXY_URL", "").strip()
        if url:
            return url.rstrip("/")
        cfg = _load_gateway_config()
        url = (cfg.get("gateway") or {}).get("proxy_url")
        url = (url or "").strip()
        if url:
            return url.rstrip("/")
        return None

    def _build_stream_consumer_config(
        self,
        source: "SessionSource",
        scfg: Any,
        adapter: Any,
        *,
        on_missing_cursor: str,
    ) -> "tuple[Any, Optional[Callable[[], None]]]":
        """Build the shared ``StreamConsumerConfig`` and optional Telegram pause-typing closure.

        ``on_missing_cursor`` handles adapters with ``SUPPORTS_MESSAGE_EDITING = False``:
        ``"fallback"`` (proxy path) streams with an empty cursor; ``"raise"`` (in-process path)
        raises ``RuntimeError`` so the caller's ``except`` skips streaming entirely. Returns
        ``(consumer_cfg, pause_typing_before_finalize)``.
        """
        from gateway.stream_consumer import StreamConsumerConfig

        _pause_typing_before_finalize = None
        if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
            def _pause_typing_before_finalize(
                _adapter=adapter,
                _chat_id=source.chat_id,
            ) -> None:
                _adapter.pause_typing_for_chat(_chat_id)
        # Platforms that can't edit sent messages (e.g. QQ, WeChat) skip streaming entirely: the
        # partial first message could never be updated, yielding duplicates (partial + final).
        _adapter_supports_edit = getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
        # Adapters that can't edit but have a native-streaming transport (e.g. WeCom msgtype "stream"
        # via send_stream_frame) pass the gate — the consumer's native branch delivers the full turn.
        _adapter_supports_native_stream = bool(getattr(
            adapter, "SUPPORTS_NATIVE_STREAMING", False,
        ))
        if (
            not _adapter_supports_edit
            and not _adapter_supports_native_stream
            and on_missing_cursor == "raise"
        ):
            raise RuntimeError("skip streaming for non-editable platform")
        _effective_cursor = scfg.cursor if _adapter_supports_edit else ""
        # Some Matrix clients render the streaming cursor as a visible tofu/white-box artifact: keep
        # streaming text on Matrix, but suppress the cursor.
        _buffer_only = False
        if source.platform == Platform.MATRIX:
            _effective_cursor = ""
            _buffer_only = True
        # Fresh-final applies to Telegram only — other platforms edit in place cheaply (Discord, Slack)
        # or lack the edit-timestamp-stays-stale problem.
        _fresh_final_secs = (
            float(getattr(scfg, "fresh_final_after_seconds", 0.0) or 0.0)
            if source.platform == Platform.TELEGRAM
            else 0.0
        )
        _consumer_cfg = StreamConsumerConfig(
            edit_interval=scfg.edit_interval,
            buffer_threshold=scfg.buffer_threshold,
            cursor=_effective_cursor,
            buffer_only=_buffer_only,
            fresh_final_after_seconds=_fresh_final_secs,
            transport=scfg.transport or "edit",
            chat_type=getattr(source, "chat_type", "") or "",
        )
        return _consumer_cfg, _pause_typing_before_finalize

    async def _run_agent_via_proxy(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: "SessionSource",
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        event_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward the message to a remote Hermes API server instead of running a local AIAgent.

        This lets a Docker container handle Matrix E2EE while the actual agent runs on the host
        with full access to local files, memory, skills, and a unified session store.
        """
        from gateway.run import (
            _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS,
            _load_gateway_config,
            _platform_config_key,
        )
        try:
            from aiohttp import ClientSession as _AioClientSession, ClientTimeout
        except ImportError:
            return {
                "final_response": "⚠️ Proxy mode requires aiohttp. Install with: pip install aiohttp",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            return {
                "final_response": "⚠️ Proxy URL not configured (GATEWAY_PROXY_URL or gateway.proxy_url)",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        # Scope-aware read: the proxy key is a per-profile credential; under multiplex honor the
        # installed scope's verdict (Slack pattern for the unscoped default-profile loop).
        try:
            from agent.secret_scope import UnscopedSecretError, get_secret

            try:
                proxy_key = (get_secret("GATEWAY_PROXY_KEY") or "").strip()
            except UnscopedSecretError:
                proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()
        except Exception:
            proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)

        # Build messages in OpenAI chat format. The remote api_server keeps continuity via
        # X-Hermes-Session-Id and loads its own history, so send only the current message; if the
        # remote has no history yet, include a compact text-only local history (remote replays tools).
        api_messages: List[Dict[str, str]] = []

        if context_prompt:
            api_messages.append({"role": "system", "content": context_prompt})

        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and content:
                api_messages.append({"role": role, "content": content})

        api_messages.append({"role": "user", "content": message})

        # HTTP headers ---------------------------------------------------
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if proxy_key:
            headers["Authorization"] = f"Bearer {proxy_key}"
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id

        body = {
            "model": "hermes-agent",
            "messages": api_messages,
            "stream": True,
        }

        # Set up platform streaming if available -------------------------
        _stream_consumer = None
        _scfg = getattr(getattr(self, "config", None), "streaming", None)
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()

        platform_key = _platform_config_key(source.platform)
        user_config = _load_gateway_config()
        from gateway.display_config import resolve_display_setting
        _plat_streaming = resolve_display_setting(
            user_config, platform_key, "streaming"
        )
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )

        _thread_metadata: Optional[Dict[str, Any]] = self._thread_metadata_for_source(source, event_message_id)

        if _streaming_enabled:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                _adapter = self._adapter_for_source(source)
                if _adapter:
                    _consumer_cfg, _pause_typing_before_finalize = (
                        self._build_stream_consumer_config(
                            source, _scfg, _adapter,
                            on_missing_cursor="fallback",
                        )
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=source.chat_id,
                        config=_consumer_cfg,
                        metadata=_thread_metadata,
                        on_before_finalize=_pause_typing_before_finalize,
                        initial_reply_to_id=event_message_id,
                        run_still_current=_run_still_current,
                    )
            except Exception as _sc_err:
                logger.debug("Proxy: could not set up stream consumer: %s", _sc_err)

        # Run the stream consumer task in the background
        stream_task = None
        if _stream_consumer:
            stream_task = asyncio.create_task(_stream_consumer.run())

        # Send typing indicator
        _adapter = self._adapter_for_source(source)
        if _adapter:
            with suppress(Exception):
                await _adapter.send_typing(source.chat_id, metadata=_thread_metadata)

        # Make the HTTP request with SSE streaming -----------------------
        full_response = ""
        _start = time.time()

        try:
            _timeout = ClientTimeout(total=0, sock_read=1800)
            async with _AioClientSession(timeout=_timeout) as session:
                async with session.post(
                    f"{proxy_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(
                            "Proxy error (%d) from %s: %s",
                            resp.status, proxy_url, error_text[:500],
                        )
                        return {
                            "final_response": f"⚠️ Proxy error ({resp.status}): {error_text[:300]}",
                            "messages": [],
                            "api_calls": 0,
                            "tools": [],
                        }

                    # Parse SSE stream
                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if not _run_still_current():
                            logger.info(
                                "Discarding stale proxy stream for %s — generation %d is no longer current",
                                session_key or "?",
                                run_generation or 0,
                            )
                            return {
                                "final_response": "",
                                "messages": [],
                                "api_calls": 0,
                                "tools": [],
                                "history_offset": len(history),
                                "session_id": session_id,
                                "response_previewed": False,
                            }
                        text = chunk.decode("utf-8", errors="replace")
                        buffer += text

                        # Process complete SSE lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    obj = json.loads(data)
                                    choices = obj.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            full_response += content
                                            if _stream_consumer:
                                                _stream_consumer.on_delta(content)
                                except json.JSONDecodeError:
                                    pass
                        if len(buffer) > _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS:
                            raise ValueError(
                                "Proxy SSE stream exceeded max buffer size without a line boundary"
                            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Proxy connection error to %s: %s", proxy_url, e)
            if not full_response:
                return {
                    "final_response": f"⚠️ Proxy connection error: {e}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }
            # Partial response — return what we got
        finally:
            # Finalize stream consumer
            if _stream_consumer:
                _stream_consumer.finish()
            if stream_task:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()

        _elapsed = time.time() - _start
        if not _run_still_current():
            logger.info(
                "Discarding stale proxy result for %s — generation %d is no longer current",
                session_key or "?",
                run_generation or 0,
            )
            return {
                "final_response": "",
                "messages": [],
                "api_calls": 0,
                "tools": [],
                "history_offset": len(history),
                "session_id": session_id,
                "response_previewed": False,
            }
        logger.info(
            "proxy response: url=%s session=%s time=%.1fs response=%d chars",
            proxy_url, (session_id or "")[:20], _elapsed, len(full_response),
        )

        return {
            "final_response": full_response or "(No response from remote agent)",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            "api_calls": 1,
            "tools": [],
            "history_offset": len(history),
            "session_id": session_id,
            "response_previewed": _stream_consumer is not None and bool(full_response),
        }

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        inbound_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Profile-scoping wrapper around the agent run.

        Under multiplexing, run the turn inside ``_profile_runtime_scope`` so config/skills/memory
        resolve to the source profile's home AND credentials come from its secret scope (never
        process-global ``os.environ``). Transparent pass-through when multiplexing is off.
        """
        from gateway.run import _profile_runtime_scope
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._run_agent_inner(
                message, context_prompt, history, source, session_id,
                session_key=session_key, run_generation=run_generation,
                _interrupt_depth=_interrupt_depth, event_message_id=event_message_id,
                inbound_message_id=inbound_message_id,
                channel_prompt=channel_prompt, moa_config=moa_config,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=message_type,
            )

        profile_home = self._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            return await self._run_agent_inner(
                message, context_prompt, history, source, session_id,
                session_key=session_key, run_generation=run_generation,
                _interrupt_depth=_interrupt_depth, event_message_id=event_message_id,
                inbound_message_id=inbound_message_id,
                channel_prompt=channel_prompt, moa_config=moa_config,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=message_type,
            )

    def _run_agent_display_settings(self, source: SessionSource) -> "GatewayRunner._RunAgentDisplay":
        """Resolve per-platform display, progress, status and streaming-surface settings for a turn."""
        from gateway.run import (
            _gateway_platform_value,
            _has_platform_display_override,
            _load_gateway_config,
            _platform_config_key,
        )
        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)

        enabled_toolsets = self._resolve_enabled_toolsets_for_source(
            user_config, source, platform_key
        )
        agent_cfg_local = user_config.get("agent") or {}
        from agent.skill_utils import parse_config_string_list

        disabled_toolsets = parse_config_string_list(agent_cfg_local.get("disabled_toolsets")) or None

        display_config = user_config.get("display", {})
        if not isinstance(display_config, dict):
            display_config = {}

        # Per-platform display settings via display_config: display.platforms.<platform>.<key>, then
        # display.<key> global, then built-in platform defaults.
        from gateway.display_config import resolve_display_setting

        # Apply tool preview length config (0 = no limit)
        try:
            from agent.display import set_tool_preview_max_len
            _tpl = resolve_display_setting(user_config, platform_key, "tool_preview_length", 0)
            set_tool_preview_max_len(int(_tpl) if _tpl else 0)
        except Exception:
            pass

        # Apply friendly tool labels config (default on) — per-platform aware
        try:
            from agent.display import set_friendly_tool_labels
            _ftl = resolve_display_setting(user_config, platform_key, "friendly_tool_labels", True)
            set_friendly_tool_labels(bool(_ftl))
        except Exception:
            pass

        # Tool progress mode — resolved per-platform with env var fallback
        _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
        _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
        _display_cfg = display_config if isinstance(display_config, dict) else {}
        _platforms_cfg = _display_cfg.get("platforms") or {}
        _platform_cfg = _platforms_cfg.get(platform_key) or {}
        _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
        _tool_progress_configured = (
            "tool_progress" in _display_cfg
            or (
                isinstance(_platform_cfg, dict)
                and "tool_progress" in _platform_cfg
            )
            or (
                isinstance(_legacy_tp_overrides, dict)
                and platform_key in _legacy_tp_overrides
            )
        )
        progress_mode = (
            _env_tp
            if _env_tp and not _tool_progress_configured
            else (_resolved_tp or _env_tp or "all")
        )
        # Tool progress grouping: "accumulate" (edit one bubble) or "separate" (one msg per tool)
        progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
        from gateway.status_phrases import choose_status_phrase, resolve_status_phrase_catalog
        _generic_status_recent: List[str] = []
        _generic_status_catalog = resolve_status_phrase_catalog(user_config, platform_key)

        def _display_surface_mode(
            setting: str,
            *,
            default: bool = False,
            require_platform_override_for: set[Any] | None = None,
            allow_generic: bool = False,
        ) -> str:
            """Return off|raw|generic for a gateway visibility surface."""
            if require_platform_override_for:
                current_platform = _gateway_platform_value(source.platform)
                platform_only = {
                    _gateway_platform_value(item)
                    for item in require_platform_override_for
                }
                if (
                    current_platform in platform_only
                    and not _has_platform_display_override(user_config, platform_key, setting)
                ):
                    return "off"
            value = resolve_display_setting(user_config, platform_key, setting, default)
            if isinstance(value, str) and value.strip().lower() == "generic":
                return "generic" if allow_generic else "off"
            return "raw" if bool(value) else "off"

        def _generic_status_phrase(kind: str, *, tool_name: str | None = None, preview: str | None = None, args: Any = None) -> str:
            try:
                return choose_status_phrase(
                    kind,
                    tool_name=tool_name,
                    preview=preview,
                    args=args,
                    recent=_generic_status_recent,
                    catalog=_generic_status_catalog,
                )
            except Exception as _phrase_err:
                logger.debug("generic status phrase selection failed: %s", _phrase_err)
                return "still on it" if kind in {"heartbeat", "waiting", "long_running", "status"} else "one sec"
        # Disable tool progress for webhooks - they don't support message editing,
        # so each progress line would be sent as a separate message.
        from gateway.config import Platform
        tool_progress_enabled = progress_mode not in {"off", "log"} and source.platform != Platform.WEBHOOK
        # Live working-state status for text-rendering typing indicators (Slack's assistant status
        # line). Independent of tool_progress (Slack defaults it off; the status line is ephemeral).
        # Rides the existing _keep_typing refresh — the callback only stores a phrase, no extra calls.
        _live_status_mode = resolve_display_setting(
            user_config, platform_key, "live_status", "full"
        )
        _live_status_adapter = self._adapter_for_source(source)
        if not getattr(_live_status_adapter, "supports_status_text", False):
            _live_status_adapter = None
        if _live_status_mode == "off":
            _live_status_adapter = None
        # "log" mode: tool calls are written to ~/.hermes/logs/tool_calls.log
        # instead of the chat (#3459 / #3458). Gateway-only by design.
        log_mode_enabled = progress_mode == "log" and source.platform != Platform.WEBHOOK
        log_queue: "queue.Queue | None" = queue.Queue() if log_mode_enabled else None
        # Natural assistant status messages are independent from tool progress and token streaming:
        # tool_progress can stay quiet while users opt into concise mid-turn updates.
        interim_assistant_messages_mode = _display_surface_mode(
            "interim_assistant_messages",
            default=True,
            require_platform_override_for={Platform.MATTERMOST},
        )
        interim_assistant_messages_enabled = (
            source.platform != Platform.WEBHOOK
            and interim_assistant_messages_mode != "off"
        )
        # thinking_progress is independent — if enabled, we need the progress queue even when
        # tool_progress is off (thinking relay uses same infra). Mattermost requires a per-platform
        # opt-in: global scratch-text display is too easy to leak into busy public threads.
        _thinking_mode = _display_surface_mode(
            "thinking_progress",
            default=False,
            require_platform_override_for={Platform.MATTERMOST},
        )
        _thinking_enabled = _thinking_mode != "off"
        # Slack-native task cards: with the Slack adapter's opt-in, tool progress renders as native
        # plan/task cards via chat.startStream, so the progress queue is needed even though Slack keeps
        # text tool_progress off by default (requiring both flags would silently disable the feature).
        _progress_adapter_for_native = self._adapter_for_source(source)
        _native_slack_task_cards = False
        if (
            source.platform == Platform.SLACK
            and _progress_adapter_for_native is not None
            and hasattr(_progress_adapter_for_native, "native_task_cards_enabled")
        ):
            try:
                _native_slack_task_cards = bool(
                    _progress_adapter_for_native.native_task_cards_enabled()
                )
            except Exception:
                logger.debug("Slack native task-card config check failed", exc_info=True)
        needs_progress_queue = (
            tool_progress_enabled or _thinking_enabled or _native_slack_task_cards
        )
        return self._RunAgentDisplay(
            user_config=user_config,
            platform_key=platform_key,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            resolve_display_setting=resolve_display_setting,
            progress_mode=progress_mode,
            progress_grouping=progress_grouping,
            _display_surface_mode=_display_surface_mode,
            tool_progress_enabled=tool_progress_enabled,
            _live_status_mode=_live_status_mode,
            _live_status_adapter=_live_status_adapter,
            log_mode_enabled=log_mode_enabled,
            log_queue=log_queue,
            interim_assistant_messages_enabled=interim_assistant_messages_enabled,
            _thinking_enabled=_thinking_enabled,
            _native_slack_task_cards=_native_slack_task_cards,
            needs_progress_queue=needs_progress_queue,
            _generic_status_phrase=_generic_status_phrase,
        )

    def _run_agent_build_turn_context(
        self,
        disp: "GatewayRunner._RunAgentDisplay",
        AIAgent: Any,
        *,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: Optional[str],
        run_generation: Optional[int],
        _interrupt_depth: int,
        event_message_id: Optional[str],
        inbound_message_id: Optional[str],
        channel_prompt: Optional[str],
        moa_config: Optional[dict],
        persist_user_message: Optional[Any],
        persist_user_timestamp: Optional[float],
        persist_user_display_kind: Optional[str],
    ) -> Tuple[TurnContext, TurnRunner, Any]:
        """Build the progress queues / holders, the ``TurnContext`` and its ``TurnRunner``.

        Returns ``(turn_ctx, turn_runner, cleanup_adapter)``; the progress-bubble cleanup flags travel
        on ``turn_ctx._cleanup_progress`` / ``turn_ctx._cleanup_msg_ids``.
        """
        from gateway.run import TurnRunner
        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)

        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if disp.needs_progress_queue else None
        last_tool = [None]  # Mutable container for tracking in closure
        last_progress_msg = [None]  # Track last message for dedup
        repeat_count = [0]  # How many times the same message repeated
        # True when the previous progress line was a terminal fenced code block — consecutive terminal
        # calls then drop the repeated "💻 terminal" header and render back-to-back blocks.
        last_was_terminal_block = [False]

        # Discord voice "verbal ack before tool calls": with the continuous mixer installed
        # (discord.voice_fx.enabled), speak a short phrase over the idle bed on the FIRST tool call of
        # the turn (from tool_start_callback, independent of the tool-progress text gate); once per turn.
        _voice_ack_fired = [False]
        _voice_ack_guild: List[Optional[int]] = [None]
        if source.platform == Platform.DISCORD:
            _va = self.adapters.get(Platform.DISCORD)
            # source.chat_id is the linked text channel; resolve the guild whose
            # voice connection is bound to it (mirrors DiscordAdapter.play_tts).
            _vtc = getattr(_va, "_voice_text_channels", None)
            if isinstance(_vtc, dict) and hasattr(_va, "voice_mixer_active"):
                for _gid, _tc in _vtc.items():
                    if str(_tc) == str(source.chat_id) and _va.voice_mixer_active(_gid):
                        _voice_ack_guild[0] = _gid
                        break
        _voice_ack_loop = asyncio.get_running_loop()

        # voice_ack_callback extracted to TurnRunner.voice_ack_callback
        # (published onto turn_ctx after the runner is constructed below).

        # Auto-cleanup of temporary progress bubbles (Telegram + any adapter that implements
        # ``delete_message``). Failed runs skip cleanup so the bubbles remain as breadcrumbs.
        _cleanup_progress = bool(
            disp.resolve_display_setting(disp.user_config, disp.platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self._adapter_for_source(source) if _cleanup_progress else None
        # getattr, not attribute access — same duck-typed-adapter guard as the edit_message check in
        # send_progress_messages: a fake adapter without delete_message means "can't delete", not a crash.
        _cleanup_delete = getattr(type(_cleanup_adapter), "delete_message", None) if _cleanup_adapter is not None else None
        if _cleanup_adapter is not None and (
            _cleanup_delete is None
            or _cleanup_delete is BasePlatformAdapter.delete_message
        ):
            # Adapter doesn't support deletion — silently disable.
            _cleanup_progress = False
            _cleanup_adapter = None
        _cleanup_msg_ids: List[str] = []
        # First-touch onboarding latch: fires at most once per run, even if
        # several tools exceed the threshold.
        long_tool_hint_fired = [False]
        _LONG_TOOL_THRESHOLD_S = 30.0

        turn_ctx = TurnContext(
            source=source,
            _run_still_current=_run_still_current,
            _live_status_adapter=disp._live_status_adapter,
            _live_status_mode=disp._live_status_mode,
            _thinking_enabled=disp._thinking_enabled,
            progress_mode=disp.progress_mode,
            progress_grouping=disp.progress_grouping,
            tool_progress_enabled=disp.tool_progress_enabled,
            progress_queue=progress_queue,
            log_queue=disp.log_queue,
            last_progress_msg=last_progress_msg,
            last_tool=last_tool,
            last_was_terminal_block=last_was_terminal_block,
            repeat_count=repeat_count,
            long_tool_hint_fired=long_tool_hint_fired,
            _LONG_TOOL_THRESHOLD_S=_LONG_TOOL_THRESHOLD_S,
            _cleanup_progress=_cleanup_progress,
            _cleanup_msg_ids=_cleanup_msg_ids,
            message=message,
            AIAgent=AIAgent,
            resolve_display_setting=disp.resolve_display_setting,
            user_config=disp.user_config,
            enabled_toolsets=disp.enabled_toolsets,
            disabled_toolsets=disp.disabled_toolsets,
            log_mode_enabled=disp.log_mode_enabled,
            interim_assistant_messages_enabled=disp.interim_assistant_messages_enabled,
            needs_progress_queue=disp.needs_progress_queue,
            _native_slack_task_cards=disp._native_slack_task_cards,
            _voice_ack_fired=_voice_ack_fired,
            _voice_ack_guild=_voice_ack_guild,
            _voice_ack_loop=_voice_ack_loop,
            history=history,
            context_prompt=context_prompt,
            channel_prompt=channel_prompt,
            session_id=session_id,
            session_key=session_key,
            run_generation=run_generation,
            _interrupt_depth=_interrupt_depth,
            event_message_id=event_message_id,
            inbound_message_id=inbound_message_id,
            moa_config=moa_config,
            persist_user_message=persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
        )
        turn_runner = TurnRunner(self, turn_ctx)
        # Callback invoked by agent on tool lifecycle events — extracted to
        # TurnRunner.progress_callback (bound method, same signature).
        turn_ctx.progress_callback = turn_runner.progress_callback
        turn_ctx.voice_ack_callback = turn_runner.voice_ack_callback
        turn_ctx.native_tool_start_callback = turn_runner.combined_tool_start_callback
        turn_ctx.native_tool_complete_callback = (
            turn_runner.native_tool_complete_callback
        )
        return turn_ctx, turn_runner, _cleanup_adapter

    def _run_agent_progress_threading(
        self,
        source: SessionSource,
        event_message_id: Optional[str],
        _native_slack_task_cards: bool,
    ) -> Tuple[Optional[dict], Optional[str], Any, Optional[str]]:
        """Resolve where progress bubbles are threaded.

        Returns ``(progress_metadata, progress_reply_to, progress_thread_id, relay_prospective_thread_id)``.
        """
        from gateway.run import _non_conversational_metadata, _resolve_progress_thread_id
        # Background task accumulating tool lines into one edited progress message. Threading metadata
        # is platform-specific: Slack DM threading needs the event_message_id fallback; Telegram forum
        # topics use message_thread_id and Hermes-created private DM topic lanes need thread metadata
        # plus a reply anchor; Feishu only honors reply_in_thread on a reply, so topic progress replies
        # to the triggering event; others use explicit source.thread_id only. Slack honours
        # reply_in_thread=false: don't synthesise a thread for progress, or every later reply inherits it.
        _progress_reply_in_thread = True
        if source.platform == Platform.SLACK:
            _slack_adapter_for_progress = self._adapter_for_source(source)
            if _slack_adapter_for_progress is not None:
                try:
                    # Relay lane: adapter owns mode resolution (nested platforms.relay.extra.slack subset,
                    # flat-key fallback). Native lane: read the flat extra as before.
                    _mode_fn = getattr(
                        _slack_adapter_for_progress,
                        "_effective_reply_in_thread",
                        None,
                    )
                    if callable(_mode_fn):
                        _progress_reply_in_thread = bool(_mode_fn())
                    else:
                        _progress_reply_in_thread = bool(
                            _slack_adapter_for_progress.config.extra.get(
                                "reply_in_thread", True
                            )
                        )
                except Exception:
                    _progress_reply_in_thread = True
        elif str(getattr(source.platform, "value", source.platform) or "").lower() == "buzz":
            # Buzz honours the same opt-out (reply_to_mode: off / extra.reply_in_thread: false): when the
            # user asked for flat channel replies, progress must not synthesise a thread either.
            _buzz_adapter_for_progress = self._adapter_for_source(source)
            if _buzz_adapter_for_progress is not None:
                try:
                    _progress_reply_in_thread = (
                        getattr(_buzz_adapter_for_progress, "_reply_to_mode", "first")
                        != "off"
                    )
                except Exception:
                    _progress_reply_in_thread = True
        _progress_thread_id = _resolve_progress_thread_id(
            source.platform, source.thread_id, event_message_id,
            reply_in_thread=_progress_reply_in_thread,
        )
        # Relay Discord auto-thread lane: a channel-initiating message has no thread_id at ingest
        # (thread is born on the connector's FIRST send). The connector stamps prospective_thread_id
        # (anchor id == the thread it will create); carry it as reply_to on the progress send so
        # bubbles route into the SAME auto-thread instead of landing flat in the parent channel.
        _relay_prospective_thread_id = (
            str(getattr(source, "prospective_thread_id", None))
            if source.platform == Platform.DISCORD
            and getattr(source, "delivered_via_upstream_relay", False)
            and getattr(source, "prospective_thread_id", None)
            and not source.thread_id
            else None
        )
        _progress_metadata = (
            self._thread_metadata_for_source(source, event_message_id)
            if _progress_thread_id == source.thread_id
            else self._thread_metadata_for_target(
                source.platform,
                source.chat_id,
                _progress_thread_id,
                chat_type=getattr(source, "chat_type", None),
                reply_to_message_id=event_message_id,
            )
        ) if _progress_thread_id else None
        if _progress_metadata is None and _relay_prospective_thread_id:
            # No real thread yet, but the connector will auto-thread on the
            # reply anchor; carry it so progress joins that thread.
            _progress_metadata = {"reply_to_message_id": event_message_id}
        _progress_metadata = _non_conversational_metadata(_progress_metadata, platform=source.platform)
        if _native_slack_task_cards:
            # chat.startStream in channels requires the recipient team/user
            # pair; harmless extras elsewhere, so stamp them whenever known.
            _progress_metadata = dict(_progress_metadata or {})
            if source.scope_id:
                _progress_metadata.setdefault("recipient_team_id", source.scope_id)
                _progress_metadata.setdefault("slack_team_id", source.scope_id)
            if source.user_id:
                _progress_metadata.setdefault("recipient_user_id", source.user_id)
        _progress_reply_to = (
            event_message_id
            if (
                source.platform in (Platform.FEISHU, Platform.MATTERMOST)
                and source.thread_id
                and event_message_id
            )
            or (
                # Buzz has no native thread_id; threading is always via reply-to the triggering event id
                # (channel clutter otherwise); skipped when the user opted out of threaded replies.
                str(getattr(source.platform, "value", source.platform) or "").lower() == "buzz"
                and event_message_id
                and _progress_reply_in_thread
            )
            or _relay_prospective_thread_id
            else None
        )
        return _progress_metadata, _progress_reply_to, _progress_thread_id, _relay_prospective_thread_id

    async def _run_agent_write_tool_log(self, log_queue: Any) -> None:
        """Drain log_queue and append tool-call lines to tool_calls.log (tool_progress=log).

        RotatingFileHandler (5MB × 3) bounds the log; RedactingFormatter keeps secrets off disk.
        """
        from gateway.run import _hermes_home
        if log_queue is None:
            return
        from logging.handlers import RotatingFileHandler

        from agent.redact import RedactingFormatter

        log_dir = _hermes_home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "tool_calls.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(RedactingFormatter("%(message)s"))
        tool_logger = logging.getLogger(f"hermes.tool_calls.{id(log_queue)}")
        tool_logger.setLevel(logging.INFO)
        tool_logger.propagate = False
        tool_logger.addHandler(file_handler)
        try:
            while True:
                try:
                    tool_logger.info("%s", log_queue.get_nowait())
                except queue.Empty:
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error("write_tool_log error: %s", e)
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Drain remaining entries before closing so late tool calls
            # from the final iteration aren't lost.
            while True:
                try:
                    tool_logger.info("%s", log_queue.get_nowait())
                except queue.Empty:
                    break
                except Exception:
                    break
            tool_logger.removeHandler(file_handler)
            try:
                file_handler.flush()
                file_handler.close()
            except Exception:
                pass

    def _run_agent_status_thread_metadata(
        self,
        source: SessionSource,
        event_message_id: Optional[str],
        _progress_thread_id: Any,
        _relay_prospective_thread_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Thread metadata for status / approval / stream sends (Feishu carries the reply anchor)."""
        if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
            # Feishu topics only keep messages inside the topic when they are sent via the reply API
            # with reply_in_thread=true. Status/approval/stream paths usually only get metadata, so
            # carry the triggering message id as a Feishu-specific fallback.
            _status_thread_metadata: Optional[Dict[str, Any]] = {
                "thread_id": _progress_thread_id,
                "reply_to_message_id": event_message_id,
            }
        else:
            _status_thread_metadata = (
                self._thread_metadata_for_source(source, event_message_id)
                if _progress_thread_id == source.thread_id
                else self._thread_metadata_for_target(
                    source.platform,
                    source.chat_id,
                    _progress_thread_id,
                    chat_type=getattr(source, "chat_type", None),
                    reply_to_message_id=event_message_id,
                )
            ) if _progress_thread_id else None
            if _status_thread_metadata is None and _relay_prospective_thread_id:
                # Relay Discord auto-thread lane (see _progress_metadata): carry the reply anchor so
                # status/interim bubbles route into the same connector-created thread as the final reply.
                _status_thread_metadata = {
                    "reply_to_message_id": event_message_id
                }
        return _status_thread_metadata

    def _run_agent_start_streaming_tts(
        self,
        source: SessionSource,
        message_type: Optional[str],
        _status_thread_metadata: Optional[Dict[str, Any]],
        streaming_tts_consumer_holder: list,
    ) -> None:
        # Streaming TTS consumer setup. Created on the gateway event-loop thread (here), NOT inside
        # run_sync's executor worker: the outer interrupt / finalisation paths reference the consumer
        # via ``streaming_tts_consumer_holder[0]`` and would hit a cross-scope NameError.
        _stts_adapter = self._adapter_for_source(source)
        _is_voice_input = (
            message_type is not None
            and str(getattr(message_type, "value", message_type)).lower() == "voice"
        )
        if (
            _stts_adapter is not None
            and _is_voice_input
            and _stts_adapter._should_auto_tts_for_chat(source.chat_id)
        ):
            try:
                from gateway.streaming_tts_consumer import StreamingTTSConsumer
                from tools.tts_tool import _load_tts_config
                _tts_cfg = _load_tts_config()
                _gateway_loop = self._gateway_loop or asyncio.get_event_loop()
                _stts_consumer = StreamingTTSConsumer(
                    adapter=_stts_adapter,
                    chat_id=source.chat_id,
                    tts_config=_tts_cfg,
                    loop=_gateway_loop,
                    metadata=_status_thread_metadata,
                )
                if _stts_consumer.active:
                    streaming_tts_consumer_holder[0] = _stts_consumer
                    _stts_consumer.start()
                # else: consumer inactive (no streaming provider) — leave
                # the holder as None so the whole-file fallback path runs.
            except Exception as _stts_err:
                logger.debug("Could not set up streaming TTS consumer: %s", _stts_err)

    async def _run_agent_stream_consumer_task(self, stream_consumer_holder: list) -> None:
        """Wait for the stream consumer to be created, then run it."""
        for _ in range(200):  # Up to 10s wait
            if stream_consumer_holder[0] is not None:
                await stream_consumer_holder[0].run()
                return
            await asyncio.sleep(0.05)

    async def _run_agent_track_agent(
        self,
        session_key: Optional[str],
        run_generation: Optional[int],
        agent_holder: list,
    ) -> None:
        """Track this agent as running for the session (interrupt support) once it is created."""
        # Wait for agent to be created
        while agent_holder[0] is None:
            await asyncio.sleep(0.05)
        if not session_key:
            return
        # Only promote the sentinel to the real agent if this run is still current. If /stop or
        # /new bumped the generation while we were spinning up, leave the newer run's slot alone
        # — we'll be discarded by the stale-result check in _handle_message_with_agent.
        if run_generation is not None and not self._is_session_run_current(
            session_key, run_generation
        ):
            logger.info(
                "Skipping stale agent promotion for %s — generation %s is no longer current",
                session_key or "",
                run_generation,
            )
            return
        self._session_state(session_key).turn.agent = agent_holder[0]
        if self._draining:
            self._update_runtime_status("draining")

    async def _run_agent_monitor_for_interrupt(
        self,
        source: SessionSource,
        session_key: Optional[str],
        agent_holder: list,
        _interrupt_detected: "asyncio.Event",
        streaming_tts_consumer_holder: list,
    ) -> None:
        # Monitor adapter interrupts (new messages). PRIMARY interrupt path for regular text: Level 1
        # (base.py) catches them before _handle_message(), so the Level 2 running_agent.interrupt() path
        # never fires. The inactivity poll loop has a BACKUP check in case this task dies silently.
        from gateway.run import _build_media_placeholder
        if not session_key:
            return

        while True:
            await asyncio.sleep(0.2)  # Check every 200ms
            try:
                # Re-resolve adapter each iteration so reconnects don't
                # leave us holding a stale reference.
                _adapter = self._adapter_for_source(source)
                if not _adapter:
                    continue
                # Must use session_key (build_session_key output), NOT source.chat_id: the adapter
                # stores interrupt events under the full session key.
                if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                    agent = agent_holder[0]
                    if agent:
                        # Peek WITHOUT consuming: the message must stay in _pending_messages for the
                        # post-run _dequeue_pending_event() (full MessageEvent + media). Popping here
                        # races: the agent may finish before checking _interrupt_requested, losing it.
                        _peek_event = _adapter._pending_messages.get(session_key)
                        pending_text = None
                        if _peek_event is not None:
                            pending_text = _peek_event.text or ""
                            # Transcribe audio BEFORE signaling the agent, so voice messages interrupt
                            # with the real transcript, not an empty string / file-path placeholder.
                            _media_urls = getattr(_peek_event, "media_urls", None) or []
                            if self._pending_event_audio_paths(_peek_event):
                                pending_text, _ = await self._transcribe_and_echo_pending_voice(
                                    _peek_event,
                                    _adapter,
                                    source,
                                    pending_text,
                                    log_context="Voice-interrupt",
                                    metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                )
                            elif not pending_text and _media_urls:
                                pending_text = _build_media_placeholder(_peek_event)
                        logger.debug("Interrupt detected from adapter, signaling agent...")
                        agent.interrupt(pending_text)
                        _interrupt_detected.set()
                        # Abort streaming TTS on barge-in (#60671).
                        _stts = streaming_tts_consumer_holder[0]
                        if _stts is not None:
                            _stts.abort("barge-in")
                        break
            except asyncio.CancelledError:
                raise
            except Exception as _mon_err:
                logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)

    @staticmethod
    def _run_agent_stream_confirmed_final_delivery(
        consumer,
        final_text: str,
        *,
        previewed: bool = False,
    ) -> bool:
        """Return True only when the actual final reply reached the user."""
        if consumer is None:
            return False
        if getattr(consumer, "final_response_sent", False):
            # A successful finalize call is not proof the *content* was final: the edit may carry
            # only the last preview snapshot. Reconcile against the recorded turn-final payload:
            # only a demonstrable mismatch (False, incl. payload-less split delivery) overrides
            # the flag; None keeps legacy trust so timeout dedup isn't regressed.
            matcher = getattr(consumer, "delivered_final_matches", None)
            if callable(matcher):
                try:
                    if matcher(final_text) is False:
                        return False
                except Exception:
                    pass
            return True
        if previewed:
            has_delivered_text = getattr(consumer, "has_delivered_text", None)
            if callable(has_delivered_text):
                try:
                    return bool(has_delivered_text(final_text))
                except Exception:
                    return False
        return False

    def _run_agent_start_turn_worker(
        self,
        turn_ctx: TurnContext,
        run_sync: Callable[[], Any],
        agent_holder: list,
        session_id: str,
        session_key: Optional[str],
        run_generation: Optional[int],
    ) -> "GatewayRunner._RunAgentWorker":
        """Schedule ``run_sync`` on the executor plus the inactivity watchdog thread."""
        from gateway.run import _float_env, _watch_gateway_turn_inactivity
        # Thread pool so we don't block. *Inactivity* timeout, not wall-clock: the agent may run for
        # hours while actively calling tools / streaming, but a hung API call or stuck tool is killed.
        # agent.gateway_timeout / HERMES_AGENT_TIMEOUT (env wins); default 1800s; 0 = unlimited.
        _agent_timeout_raw = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        _agent_timeout = _agent_timeout_raw if _agent_timeout_raw > 0 else None
        _agent_warning_raw = _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900)
        _agent_warning = _agent_warning_raw if _agent_warning_raw > 0 else None

        # A background=true process intentionally survives a successful turn, so capture
        # existing IDs and reap only children created by THIS turn if it times out. The daemon
        # watchdog is independent of asyncio: cgroup memory reclaim can starve the loop that
        # runs the normal timeout poll, and cleanup must not wait for the loop to recover.
        from tools.process_registry import process_registry

        _turn_task_id = session_id or ""
        _turn_process_baseline = process_registry.snapshot_running_ids(_turn_task_id)
        turn_ctx.process_task_id = _turn_task_id
        turn_ctx.process_baseline = _turn_process_baseline
        _turn_worker_done = threading.Event()
        _turn_timeout_fired = threading.Event()
        _turn_cleanup_lock = threading.Lock()
        # task_id is session-scoped, not turn-scoped: gate the eventual reap on this exact claim still
        # being current, so a replacement turn on the same session that starts before the watchdog
        # fires doesn't get its own fresh process killed by this turn's stale baseline.
        _turn_run_generation = run_generation
        _turn_is_current = (
            (lambda: self._is_session_run_current(session_key, _turn_run_generation))
            if _turn_run_generation is not None
            else (lambda: True)
        )

        def _run_sync_with_timeout_lifecycle():
            try:
                return run_sync()
            finally:
                _turn_worker_done.set()
                # `.turn.agent` is only reset to _AGENT_PENDING_SENTINEL when the *next* turn is
                # claimed, so this agent stays reachable from _interrupt_and_clear_session()
                # until then. Clearing ownership markers the instant our worker finishes means a
                # /stop on the finished turn no longer reaps background work it left running.
                _finished_agent = agent_holder[0] if agent_holder else None
                if _finished_agent is not None:
                    _finished_agent._gateway_turn_process_task_id = ""
                    _finished_agent._gateway_turn_process_baseline = frozenset()

        if _agent_timeout is not None:
            threading.Thread(
                target=_watch_gateway_turn_inactivity,
                kwargs={
                    "agent_holder": agent_holder,
                    "task_id": _turn_task_id,
                    "process_baseline": _turn_process_baseline,
                    "timeout": _agent_timeout,
                    "worker_done": _turn_worker_done,
                    "timeout_fired": _turn_timeout_fired,
                    "cleanup_lock": _turn_cleanup_lock,
                    "poll_interval": 5.0,
                    "is_still_current": _turn_is_current,
                },
                name=f"gateway-turn-watchdog-{_turn_task_id[:12]}",
                daemon=True,
            ).start()
        _executor_task = asyncio.ensure_future(
            self._run_in_executor_with_context(_run_sync_with_timeout_lifecycle)
        )
        return self._RunAgentWorker(
            executor_task=_executor_task,
            agent_timeout=_agent_timeout,
            agent_warning=_agent_warning,
            task_id=_turn_task_id,
            process_baseline=_turn_process_baseline,
            worker_done=_turn_worker_done,
            timeout_fired=_turn_timeout_fired,
            cleanup_lock=_turn_cleanup_lock,
            is_current=_turn_is_current,
        )

    async def _run_agent_backup_interrupt_check(
        self,
        source: SessionSource,
        session_key: Optional[str],
        agent_holder: list,
        _interrupt_detected: "asyncio.Event",
        interrupt_monitor: "asyncio.Task",
        streaming_tts_consumer_holder: list,
    ) -> None:
        """Backup interrupt check: if the monitor task died or missed the interrupt, catch it here."""
        from gateway.run import _build_media_placeholder
        if not _interrupt_detected.is_set() and session_key:
            _backup_adapter = self._adapter_for_source(source)
            _backup_agent = agent_holder[0]
            if (_backup_adapter and _backup_agent
                    and hasattr(_backup_adapter, 'has_pending_interrupt')
                    and _backup_adapter.has_pending_interrupt(session_key)):
                _bp_event = _backup_adapter._pending_messages.get(session_key)
                _bp_text = _bp_event.text if _bp_event else None
                if _bp_event is not None:
                    _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                    if self._pending_event_audio_paths(_bp_event):
                        _bp_text, _ = await self._transcribe_and_echo_pending_voice(
                            _bp_event,
                            _backup_adapter,
                            source,
                            _bp_text or "",
                            log_context="Voice-backup-interrupt",
                            metadata={"thread_id": source.thread_id} if source.thread_id else None,
                        )
                    elif not _bp_text and _bp_media_urls:
                        _bp_text = _build_media_placeholder(_bp_event)
                logger.info(
                    "Backup interrupt detected for session %s "
                    "(monitor task state: %s)",
                    session_key,
                    "done" if interrupt_monitor.done() else "running",
                )
                _backup_agent.interrupt(_bp_text)
                _interrupt_detected.set()
                # Abort streaming TTS on barge-in (#60671).
                _stts = streaming_tts_consumer_holder[0]
                if _stts is not None:
                    _stts.abort("barge-in")

    async def _run_agent_await_turn_worker(
        self,
        worker: "GatewayRunner._RunAgentWorker",
        *,
        source: SessionSource,
        session_key: Optional[str],
        agent_holder: list,
        result_holder: list,
        tools_holder: list,
        _interrupt_detected: "asyncio.Event",
        interrupt_monitor: "asyncio.Task",
        streaming_tts_consumer_holder: list,
        _status_thread_metadata: Optional[Dict[str, Any]],
    ) -> Any:
        """Poll the executor future (inactivity timeout + backup interrupt checks); return its result.

        On inactivity timeout the result is a synthetic failed run dict carrying the diagnostic.
        """
        from gateway.run import (
            _INTERRUPT_REASON_TIMEOUT,
            _abandon_timed_out_gateway_turn,
            _interim_metadata,
            request_hard_interrupt,
        )
        _warning_fired = False
        _inactivity_timeout = False
        _POLL_INTERVAL = 5.0

        if worker.agent_timeout is None:
            # Unlimited — still poll periodically for backup interrupt
            # detection in case monitor_for_interrupt() silently died.
            response = None
            while True:
                done, _ = await asyncio.wait(
                    {worker.executor_task}, timeout=_POLL_INTERVAL
                )
                if done:
                    response = worker.executor_task.result()
                    break
                # Backup interrupt check: if the monitor task died or
                # missed the interrupt, catch it here.
                await self._run_agent_backup_interrupt_check(
                    source,
                    session_key,
                    agent_holder,
                    _interrupt_detected,
                    interrupt_monitor,
                    streaming_tts_consumer_holder,
                )

        else:
            # Poll the agent's built-in activity tracker (updated by _touch_activity() on every tool
            # call, API call, and stream delta) every few seconds.
            response = None
            while True:
                done, _ = await asyncio.wait(
                    {worker.executor_task}, timeout=_POLL_INTERVAL
                )
                if done:
                    # Prefer the real result when the worker finished even if the watchdog fired in
                    # the same window: the completed run already persisted its reply, so the "agent
                    # inactive" diagnostic would contradict the stored transcript.
                    response = worker.executor_task.result()
                    break
                if worker.timeout_fired.is_set():
                    _inactivity_timeout = True
                    break
                # Agent still running — check inactivity.
                _agent_ref = agent_holder[0]
                _idle_secs = 0.0
                if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                    try:
                        _act = _agent_ref.get_activity_summary()
                        _idle_secs = _act.get("seconds_since_activity", 0.0)
                    except Exception:
                        pass
                # Staged warning: fire once before escalating to full timeout.
                if (not _warning_fired and worker.agent_warning is not None
                        and _idle_secs >= worker.agent_warning):
                    _warning_fired = True
                    _warn_adapter = self._adapter_for_source(source)
                    if _warn_adapter:
                        _elapsed_warn = int(worker.agent_warning // 60) or 1
                        _remaining_mins = int((worker.agent_timeout - worker.agent_warning) // 60) or 1
                        try:
                            await _warn_adapter.send(
                                source.chat_id,
                                f"⚠️ No activity for {_elapsed_warn} min. "
                                f"If the agent does not respond soon, it will "
                                f"be timed out in {_remaining_mins} min. "
                                f"You can continue waiting or use /reset.",
                                metadata=_interim_metadata(_status_thread_metadata),
                            )
                        except Exception as _warn_err:
                            logger.debug("Inactivity warning send error: %s", _warn_err)
                if _idle_secs >= worker.agent_timeout:
                    _inactivity_timeout = True
                    threading.Thread(
                        target=_abandon_timed_out_gateway_turn,
                        kwargs={
                            "agent_holder": agent_holder,
                            "task_id": worker.task_id,
                            "process_baseline": worker.process_baseline,
                            "worker_done": worker.worker_done,
                            "timeout_fired": worker.timeout_fired,
                            "cleanup_lock": worker.cleanup_lock,
                            "is_still_current": worker.is_current,
                        },
                        name=f"gateway-turn-reaper-{worker.task_id[:12]}",
                        daemon=True,
                    ).start()
                    break
                # Backup interrupt check (same as unlimited path).
                await self._run_agent_backup_interrupt_check(
                    source,
                    session_key,
                    agent_holder,
                    _interrupt_detected,
                    interrupt_monitor,
                    streaming_tts_consumer_holder,
                )

        if _inactivity_timeout:
            # Build a diagnostic summary from the agent's activity tracker.
            _timed_out_agent = agent_holder[0]
            _activity = {}
            if _timed_out_agent and hasattr(_timed_out_agent, "get_activity_summary"):
                with suppress(Exception):
                    _activity = _timed_out_agent.get_activity_summary()

            _last_desc = _activity.get("last_activity_desc", "unknown")
            _secs_ago = _activity.get("seconds_since_activity", 0)
            _cur_tool = _activity.get("current_tool")
            _iter_n = _activity.get("api_call_count", 0)
            _iter_max = _activity.get("max_iterations", 0)

            logger.error(
                "Agent idle for %.0fs (timeout %.0fs) in session %s "
                "| last_activity=%s | iteration=%s/%s | tool=%s",
                _secs_ago, worker.agent_timeout, session_key,
                _last_desc, _iter_n, _iter_max,
                _cur_tool or "none",
            )

            # Interrupt the agent if it's still running so the thread
            # pool worker is freed.
            if _timed_out_agent:
                request_hard_interrupt(_timed_out_agent, _INTERRUPT_REASON_TIMEOUT)

            _timeout_mins = int(worker.agent_timeout // 60) or 1

            # Construct a user-facing message with diagnostic context.
            _diag_lines = [
                f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls "
                f"or API responses."
            ]
            if _cur_tool:
                _diag_lines.append(
                    f"The agent appears stuck on tool `{_cur_tool}` "
                    f"({_secs_ago:.0f}s since last activity, "
                    f"iteration {_iter_n}/{_iter_max})."
                )
            else:
                _diag_lines.append(
                    f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                    f"iteration {_iter_n}/{_iter_max}). "
                    "The agent may have been waiting on an API response."
                )
            _diag_lines.append(
                "To increase the limit, set agent.gateway_timeout in config.yaml "
                "(value in seconds, 0 = no limit) and restart the gateway.\n"
                "Try again, or use /reset to start fresh."
            )

            response = {
                "final_response": "\n".join(_diag_lines),
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": _iter_n,
                "tools": tools_holder[0] or [],
                "history_offset": 0,
                "failed": True,
            }
        return response

    def _run_agent_evict_on_fallback(
        self, session_key: Optional[str], agent_holder: list, result_holder: list,
    ) -> None:
        # Persist fallback-model switches so /model shows the actually-active model. Skip
        # eviction when the run failed — evicting forces MCP reinit on the next message for no
        # benefit (bad model → fallback → evict → recreate → same 400 loop burning CPU).
        from gateway.run import _resolve_gateway_model
        _agent = agent_holder[0]
        _result_for_fb = result_holder[0]
        _run_failed = _result_for_fb.get("failed") if _result_for_fb else False
        if _agent is not None and hasattr(_agent, 'model') and not _run_failed:
            _cfg_model = _resolve_gateway_model()
            # Normalize _cfg_model as AIAgent.__init__ does so a vendor-prefixed config value
            # matches the agent's stripped model on native providers — otherwise the cached agent
            # is evicted every turn, destroying prompt caching. Aggregators keep the vendor slug.
            try:
                from hermes_cli.model_normalize import (
                    _AGGREGATOR_PROVIDERS,
                    normalize_model_for_provider,
                )
                _agent_provider = getattr(_agent, 'provider', '') or ''
                if _agent_provider and _agent_provider not in _AGGREGATOR_PROVIDERS:
                    _cfg_model = normalize_model_for_provider(_cfg_model, _agent_provider)
            except Exception:
                pass
            if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent.model):
                # Fallback activated on a successful run — evict cached
                # agent so the next message retries the primary model.
                self._evict_cached_agent(session_key)

    async def _run_agent_finalize_streaming_tts(
        self,
        streaming_tts_consumer_holder: list,
        adapter: Any,
        session_key: Optional[str],
        run_generation: Optional[int],
    ) -> None:
        # Finalize the streaming-TTS consumer. finish() runs on the outer event-loop thread so
        # early returns from run_sync are also finalised. wait_complete() drains queued audio;
        # on timeout abort unconditionally — if audio was audible keep suppression (no replay
        # from the start); if not, the whole-file fallback is permitted.
        _stts = streaming_tts_consumer_holder[0]
        if _stts is not None:
            _stts.finish()
            try:
                await _stts.wait_complete(timeout=10.0)
            except Exception as _stts_done_err:
                logger.debug("streaming TTS wait_complete error: %s", _stts_done_err)
            if not _stts.done:
                # Timeout before or after audible audio: abort to free the consumer task. Audible
                # streams retain suppression; silent streams stay eligible for whole-file fallback.
                _stts.abort("streaming TTS finalisation timeout")
                await _stts.wait_complete(timeout=2.0)
            if _stts.suppress_whole_file and adapter is not None:
                _mark_turn = getattr(adapter, "_mark_streaming_tts_completed_turn", None)
                if callable(_mark_turn):
                    _mark_turn(session_key, run_generation)

    async def _run_agent_drain_pending(
        self,
        result: Any,
        adapter: Any,
        source: SessionSource,
        session_key: Optional[str],
    ) -> Tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up.

        Returns ``(pending_event, pending)``.
        """
        from gateway.run import (
            _build_media_placeholder,
            _dequeue_pending_event,
            _is_control_interrupt_message,
        )
        # Get pending message from adapter.
        # Use session_key (not source.chat_id) to match adapter's storage keys.
        pending_event = None
        pending = None
        if result and adapter and session_key:
            pending_event = _dequeue_pending_event(adapter, session_key)
            # /queue overflow: after consuming the adapter's "next-up" slot, promote the next
            # queued event into it so the recursive run's drain will see it. Keeping the slot
            # occupied for the whole FIFO chain preserves order and makes a mid-chain /queue
            # route to overflow instead of jumping the queue.
            pending_event = self._promote_queued_event(session_key, adapter, pending_event)
            if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                interrupt_message = result.get("interrupt_message")
                if _is_control_interrupt_message(interrupt_message):
                    logger.info(
                        "Ignoring control interrupt message for session %s: %s",
                        session_key or "?",
                        interrupt_message,
                    )
                else:
                    pending = interrupt_message
            elif pending_event:
                # Transcribe audio on the dequeued event BEFORE it becomes the next user turn, so
                # queued/interrupting voice messages drain with the real transcript, not a file path.
                _pending_text = pending_event.text or ""
                _media_urls = getattr(pending_event, "media_urls", None) or []
                if self._pending_event_audio_paths(pending_event):
                    pending, _ = await self._transcribe_and_echo_pending_voice(
                        pending_event,
                        adapter,
                        source,
                        _pending_text,
                        log_context="Voice-drain",
                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                    )
                    if not pending:
                        pending = _build_media_placeholder(pending_event)
                else:
                    pending = _pending_text or _build_media_placeholder(pending_event)
                if pending:
                    logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

        # Leftover /steer: a steer arriving after the last tool batch (e.g. during the final API
        # call) comes back in result["pending_steer"]; deliver it as the next user turn, not drop it.
        if result and not pending and not pending_event:
            _leftover_steer = result.get("pending_steer")
            if _leftover_steer:
                pending = _leftover_steer
                logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

        # Safety net: if the pending text is a slash command (e.g. "/stop", "/new"), discard it
        # — commands should never be passed to the agent as user input.
        if pending and pending.strip().startswith("/"):
            _pending_parts = pending.strip().split(None, 1)
            _pending_cmd_word = _pending_parts[0][1:].lower() if _pending_parts else ""
            if _pending_cmd_word:
                try:
                    from hermes_cli.commands import resolve_command as _rc_pending
                    if _rc_pending(_pending_cmd_word):
                        logger.info(
                            "Discarding command '/%s' from pending queue — "
                            "commands must not be passed as agent input",
                            _pending_cmd_word,
                        )
                        pending_event = None
                        pending = None
                except Exception:
                    pass

        if self._draining and (pending_event or pending):
            logger.info(
                "Discarding pending follow-up for session %s during gateway %s",
                session_key or "?",
                self._status_action_label(),
            )
            pending_event = None
            pending = None
        return pending_event, pending

    async def _run_agent_deliver_first_response(
        self,
        *,
        source: SessionSource,
        adapter: Any,
        session_key: Optional[str],
        run_generation: Optional[int],
        event_message_id: Optional[str],
        response: Any,
        result: Any,
        stream_consumer_holder: list,
        stream_task: Any,
        _status_thread_metadata: Optional[Dict[str, Any]],
    ) -> None:
        # Queued message after normal completion: deliver the first response before the
        # queued follow-up, unless streaming already delivered it.
        _sc = stream_consumer_holder[0]
        if _sc and stream_task:
            try:
                await asyncio.wait_for(stream_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream_task
            except Exception as e:
                logger.debug("Stream consumer wait before queued message failed: %s", e)
        # The queued branch needs raw ``result`` for interruption, history, and
        # recursion state, but delivery must use the finalized task result — it carries
        # empty/failure normalization and final-response processing from _run_agent_task.
        _delivery_result = response if isinstance(response, dict) else (result or {})
        _previewed = bool(_delivery_result.get("response_previewed"))
        first_response = _delivery_result.get("final_response", "")
        _already_streamed = self._run_agent_stream_confirmed_final_delivery(
            _sc,
            first_response,
            previewed=_previewed,
        )
        # Same predicate as the normal completed-turn path: this direct queued-send branch
        # predates intentional-silence filtering and would leak the literal marker.
        try:
            from gateway.response_filters import is_intentional_silence_agent_result
            _intentional_silence = is_intentional_silence_agent_result(
                _delivery_result, first_response,
            )
        except Exception:
            _intentional_silence = False
        if _intentional_silence:
            logger.info(
                "Queued follow-up for session %s: suppressing intentional silence marker before continuing.",
                session_key or "?",
            )
        elif first_response:
            try:
                if _already_streamed:
                    logger.info(
                        "Queued follow-up for session %s: final text delivery confirmed; delivering explicit media before continuing.",
                        session_key or "?",
                    )
                else:
                    logger.info(
                        "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                        session_key or "?",
                    )
                await self._deliver_queued_first_response(
                    first_response,
                    source=source,
                    adapter=adapter,
                    metadata=_status_thread_metadata,
                    event_message_id=event_message_id,
                    text_already_delivered=_already_streamed,
                    deliver_media=not _delivery_result.get("failed"),
                    stream_consumer=_sc,
                )
            except Exception as e:
                logger.warning("Failed to send first response before queued message: %s", e)
        # Release deferred bg-review notifications now that the first response is delivered:
        # pop from the adapter's callback dict (no double-fire in base.py's finally) and call.
        if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
            _bg_cb = adapter.pop_post_delivery_callback(
                session_key,
                generation=run_generation,
            )
            if callable(_bg_cb):
                try:
                    _bg_result = _bg_cb()
                    if inspect.isawaitable(_bg_result):
                        await _bg_result
                except Exception:
                    pass
        elif adapter and hasattr(adapter, "_post_delivery_callbacks"):
            _bg_cb = adapter._post_delivery_callbacks.pop(session_key, None)
            if callable(_bg_cb):
                try:
                    _bg_result = _bg_cb()
                    if inspect.isawaitable(_bg_result):
                        await _bg_result
                except Exception:
                    pass

    async def _run_agent_queued_followup(
        self,
        *,
        source: SessionSource,
        adapter: Any,
        session_id: str,
        session_key: Optional[str],
        run_generation: Optional[int],
        _interrupt_depth: int,
        event_message_id: Optional[str],
        context_prompt: str,
        history: List[Dict[str, Any]],
        pending: Optional[str],
        pending_event: Any,
        response: Any,
        result: Any,
        result_holder: list,
        stream_consumer_holder: list,
        stream_task: Any,
        _status_thread_metadata: Optional[Dict[str, Any]],
    ) -> Any:
        """Run the queued / interrupting follow-up as the next turn (recursive ``_run_agent``)."""
        from gateway.run import _preserve_queued_followup_history_offset, merge_pending_message_event
        logger.debug("Processing pending message: '%s...'", pending[:40])

        # Clear the adapter's interrupt event so the next _run_agent call doesn't re-trigger the
        # interrupt before the new agent's first API call (infinite loop otherwise).
        if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
            adapter._active_sessions[session_key].clear()

        # Cap recursion depth to prevent resource exhaustion when the
        # user sends multiple messages while the agent keeps failing. (#816)
        if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
            logger.warning(
                "Interrupt recursion depth %d reached for session %s — "
                "queueing message instead of recursing.",
                _interrupt_depth, session_key,
            )
            adapter = self._adapter_for_source(source)
            if adapter and pending_event:
                merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
            elif adapter and hasattr(adapter, 'queue_message'):
                adapter.queue_message(session_key, pending)
            return result_holder[0] or {"final_response": response, "messages": history}

        was_interrupted = result.get("interrupted")
        if not was_interrupted:
            await self._run_agent_deliver_first_response(
                source=source,
                adapter=adapter,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=event_message_id,
                response=response,
                result=result,
                stream_consumer_holder=stream_consumer_holder,
                stream_task=stream_task,
                _status_thread_metadata=_status_thread_metadata,
            )
        # else: interrupted — discard the response ("Operation interrupted." is noise; the user
        # knows they sent a new message).

        updated_history = result.get("messages", history)
        next_source = source
        next_message = pending
        next_message_id = None
        next_channel_prompt = None
        next_session_key = session_key
        # Carry the pending event's message_type into the recursive call so queued voice turns
        # can stream TTS and re-mark the generation for the final delivered turn.
        next_message_type = None
        if pending_event is not None:
            next_source = getattr(pending_event, "source", None) or source
            if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                logger.info(
                    "Discarding stale goal continuation for session %s — goal is no longer active",
                    session_key or "?",
                )
                return result
            # Resolve the follow-up's session key BEFORE preparing the inbound text:
            # _prepare_inbound_message_text buffers native image paths under the key given, and
            # the recursive _run_agent consumes them under next_session_key — mismatch drops them.
            try:
                next_session_key = self._session_key_for_source(next_source)
            except Exception:
                logger.debug(
                    "Queued follow-up session-key resolution failed; reusing %s",
                    session_key or "?",
                    exc_info=True,
                )
            next_message = await self._prepare_profile_scoped_inbound_message_text(
                event=pending_event,
                source=next_source,
                history=updated_history,
                session_key=next_session_key,
            )
            if next_message is None:
                return result
            next_message_id = self._reply_anchor_for_event(pending_event)
            next_channel_prompt = getattr(pending_event, "channel_prompt", None)
            next_message_type = getattr(pending_event, "message_type", None)

        # Clear the prior logical turn's completed streaming marker so the recursive turn's
        # streaming TTS isn't suppressed by that completion.
        _clear_adapter = self._adapter_for_source(source)
        if _clear_adapter is not None and session_key and run_generation is not None:
            _completed_turns = getattr(_clear_adapter, "_streaming_tts_completed_turns", None)
            if _completed_turns is not None:
                _prior_key = getattr(_clear_adapter, "_streaming_tts_turn_key", None)
                if callable(_prior_key):
                    _pk = _prior_key(session_key, run_generation)
                    if _pk:
                        _completed_turns.discard(_pk)

        # Restart the typing indicator for the follow-up turn; the outer
        # _process_message_background typing task is alive but may be stale.
        _followup_adapter = self._adapter_for_source(source)
        if _followup_adapter:
            with suppress(Exception):
                await _followup_adapter.send_typing(
                    source.chat_id,
                    metadata=_status_thread_metadata,
                )

        # Re-baseline the cached agent's message_count before recursing into the /queue follow-up:
        # the coherence guard would otherwise rebuild on OUR OWN flushed rows and destroy the
        # prompt-cache prefix; _handle_message_with_agent re-baselines only after the chain ends.
        await self._refresh_agent_cache_message_count(session_key, session_id)

        followup_result = await self._run_agent(
            message=next_message,
            context_prompt=context_prompt,
            history=updated_history,
            source=next_source,
            session_id=session_id,
            session_key=next_session_key,
            run_generation=run_generation,
            _interrupt_depth=_interrupt_depth + 1,
            event_message_id=next_message_id,
            channel_prompt=next_channel_prompt,
            message_type=next_message_type,
        )
        return _preserve_queued_followup_history_offset(result, followup_result)

    async def _run_agent_cleanup_turn_tasks(
        self,
        *,
        progress_task: Any,
        log_task: Any,
        interrupt_monitor: "asyncio.Task",
        _notify_task: "asyncio.Task",
        tracking_task: "asyncio.Task",
        stream_task: Any,
        stream_consumer_holder: list,
        streaming_tts_consumer_holder: list,
        session_key: Optional[str],
        run_generation: Optional[int],
    ) -> None:
        """``finally`` half of a turn: cancel background tasks, flush stream, release the session slot."""
        # Stop progress sender, interrupt monitor, and notification task
        if progress_task:
            progress_task.cancel()
        if log_task:
            log_task.cancel()
        interrupt_monitor.cancel()
        _notify_task.cancel()

        # Wait for stream consumer to finish its final edit
        if stream_task:
            # If the agent never created a stream consumer (non-streaming path, or a test stub
            # returning synchronously) there is nothing to flush — cancel now instead of waiting
            # out the 5s timeout polling for a consumer that will never arrive.
            _has_stream_consumer = (
                stream_consumer_holder
                and stream_consumer_holder[0] is not None
            )
            if not _has_stream_consumer:
                stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream_task
            else:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await stream_task

        # Unconditional abort + bounded wait for the streaming-TTS consumer: covers cancellation /
        # exception paths where the normal finalisation block was skipped.
        _stts_finally = streaming_tts_consumer_holder[0]
        if _stts_finally is not None and not _stts_finally.done:
            _stts_finally.abort("cleanup")
            with suppress(Exception):
                await _stts_finally.wait_complete(timeout=2.0)

        # Clean up tracking
        tracking_task.cancel()
        if session_key:
            # Release the slot only if this run's generation still owns it: a /stop or /new that
            # bumped the generation while we unwound already installed its own state; keep it.
            self._release_running_agent_state(
                session_key, run_generation=run_generation
            )
        if self._draining:
            self._update_runtime_status("draining")

        # Wait for cancelled tasks
        for task in [progress_task, log_task, interrupt_monitor, tracking_task, _notify_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # A background task that died of a non-cancellation error (transport drop in
                    # a progress/card publish) must not abort the cleanup path — everything
                    # after this loop (final-delivery bookkeeping) still runs (review B7).
                    logger.debug(
                        "background turn task failed during cleanup",
                        exc_info=True,
                    )

    async def _run_agent_mark_streamed_delivery(
        self,
        response: Any,
        stream_consumer_holder: list,
        source: SessionSource,
        session_key: Optional[str],
    ) -> None:
        # If streaming already delivered the response, skip the caller's send() — but never when the
        # agent failed (the error is unseen content) or on "(empty)": interim text ("Let me search…")
        # set already_sent but is NOT the final answer; suppressing would leave the user with silence.
        _sc = stream_consumer_holder[0]
        if isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            # response_previewed means interim_assistant_callback already saw the final text, but only
            # suppress the send if that exact text was delivered — unrelated commentary/progress isn't it.
            _previewed = bool(response.get("response_previewed"))
            _content_delivered = bool(
                _sc and getattr(_sc, "final_content_delivered", False)
            )
            # A *successful* finalize edit can still carry only the last preview snapshot, and both
            # suppression flags reflect call success, not content. Reconcile against the recorded
            # turn-final payload: on mismatch (False, incl. payload-less split delivery) neither flag
            # may suppress the final send; None (no record) keeps legacy trust.
            _stale_finalized = False
            if _content_delivered and not _is_empty_sentinel:
                _matcher = getattr(_sc, "delivered_final_matches", None)
                if callable(_matcher):
                    try:
                        _stale_finalized = _matcher(_final) is False
                    except Exception:
                        _stale_finalized = False
                if _stale_finalized:
                    _content_delivered = False
            # Plugin hooks (e.g. transform_llm_output) may append content after streaming finished — when
            # transformed, always send the final version so the appended content reaches the client.
            _transformed = bool(response.get("response_transformed"))
            # Suppress the normal send only when the actual final reply reached the user (streamed, or
            # interim preview of that *exact* text); commentary shown during a compression/split isn't it.
            _streamed = self._run_agent_stream_confirmed_final_delivery(
                _sc,
                _final,
                previewed=_previewed,
            )
            if not _is_empty_sentinel and not _transformed and (_streamed or _content_delivered):
                logger.info(
                    "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                    session_key or "?",
                    _streamed,
                    _previewed,
                    _content_delivered,
                )
                response["already_sent"] = True
            elif not _is_empty_sentinel and not _transformed and _stale_finalized and _sc is not None:
                # Stale finalize: the streamed message holds only the last preview snapshot. Edit it
                # up to the complete response; on edit failure leave already_sent unset so the normal
                # send delivers. Not for split delivery: message_id is only the LAST chunk, so editing
                # it would repeat every sealed head chunk — fall through to the normal send.
                _sc_msg_id = _sc.message_id
                _sc_adapter = getattr(_sc, "adapter", None)
                if getattr(_sc, "_turn_split_delivery", False):
                    logger.info(
                        "Stale streamed finalize detected for session %s on a multi-message split; skipping the in-place reconciliation edit and delivering the complete response via normal final send (#78541).",
                        session_key or "?",
                    )
                elif _sc_msg_id and _sc_msg_id != "__no_edit__" and _sc_adapter is not None:
                    try:
                        _reconcile_res = await _sc_adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=_final,
                            finalize=True,
                        )
                        if getattr(_reconcile_res, "success", True):
                            response["already_sent"] = True
                            logger.info(
                                "Reconciled stale streamed finalize for session %s: edited message %s with the complete response (#71643).",
                                session_key or "?", _sc_msg_id,
                            )
                        else:
                            logger.warning(
                                "Stale-finalize reconciliation edit failed for session %s (%s); sending complete response via normal final send.",
                                session_key or "?",
                                getattr(_reconcile_res, "error", None),
                            )
                    except Exception as _edit_err:
                        logger.warning(
                            "Stale-finalize reconciliation edit failed for session %s: %s; sending complete response via normal final send.",
                            session_key or "?", _edit_err,
                        )
                else:
                    logger.info(
                        "Stale streamed finalize detected for session %s with no editable message; delivering complete response via normal final send (#71643).",
                        session_key or "?",
                    )
            elif not _is_empty_sentinel and _transformed and _sc is not None:
                # Plugin hooks transformed the response after streaming — edit the
                # existing streamed message instead of sending a duplicate.
                _sc_msg_id = _sc.message_id
                if _sc_msg_id:
                    try:
                        await _sc.adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=response["final_response"],
                            finalize=True,
                        )
                        response["already_sent"] = True
                        logger.info(
                            "Edited streamed message %s for session %s to include plugin-transformed content.",
                            _sc_msg_id, session_key or "?",
                        )
                    except Exception as _edit_err:
                        logger.warning(
                            "Failed to edit streamed message for session %s: %s",
                            session_key or "?", _edit_err,
                        )
            elif _sc is not None and not _is_empty_sentinel:
                # DUPLICATE-RISK DIAGNOSTIC: a stream consumer existed for this turn but suppression
                # did NOT fire, so the gateway's normal final-send is about to run. Log the decision
                # inputs so a recurrence can be pinned to "signal never set" vs "ack-pending race".
                logger.warning(
                    "Normal final-send NOT suppressed despite active stream "
                    "consumer for session %s: streamed=%s previewed=%s "
                    "content_delivered=%s transformed=%s final_len=%d — "
                    "possible duplicate send (see wecom ack-timeout RCA).",
                    session_key or "?",
                    _streamed,
                    _previewed,
                    _content_delivered,
                    _transformed,
                    len(_final),
                )

    def _run_agent_schedule_bubble_cleanup(
        self,
        response: Any,
        _cleanup_progress: bool,
        _cleanup_adapter: Any,
        _cleanup_msg_ids: List[str],
        source: SessionSource,
        session_key: Optional[str],
        run_generation: Optional[int],
    ) -> None:
        # Schedule deletion of tracked temporary progress bubbles after the final response lands; failed
        # runs keep them as breadcrumbs. Only on adapters with ``delete_message``; failures swallowed.
        from gateway.run import safe_schedule_threadsafe
        if (
            _cleanup_progress
            and _cleanup_adapter is not None
            and _cleanup_msg_ids
            and session_key
            and isinstance(response, dict)
            and not response.get("failed")
            and hasattr(_cleanup_adapter, "register_post_delivery_callback")
        ):
            _ids_snapshot = list(_cleanup_msg_ids)
            _chat_id_snapshot = source.chat_id
            _adapter_snapshot = _cleanup_adapter
            _loop_snapshot = asyncio.get_running_loop()

            def _cleanup_temp_bubbles() -> None:
                async def _delete_all() -> None:
                    for _mid in _ids_snapshot:
                        with suppress(Exception):
                            await _adapter_snapshot.delete_message(
                                _chat_id_snapshot, _mid
                            )
                with suppress(Exception):
                    safe_schedule_threadsafe(
                        _delete_all(), _loop_snapshot,
                        logger=logger,
                        log_message="Temp bubble cleanup scheduling error",
                    )

            try:
                _cleanup_adapter.register_post_delivery_callback(
                    session_key,
                    _cleanup_temp_bubbles,
                    generation=run_generation,
                )
            except Exception as _rpe:
                logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

    def _run_agent_bind_turn_wiring(
        self,
        turn_ctx: TurnContext,
        turn_runner: TurnRunner,
        source: SessionSource,
        event_message_id: Optional[str],
        _progress_metadata: Optional[dict],
        _progress_reply_to: Optional[str],
        _progress_thread_id: Any,
        _relay_prospective_thread_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Publish progress metadata, result holders and the sync→async bridges onto ``turn_ctx``.

        Returns ``_status_thread_metadata``; the holders are read back via ``turn_ctx.*_holder``.
        """
        # Extracted to TurnRunner.send_progress_messages; the threading metadata above is published
        # onto the shared TurnContext where the original closure's captured locals were bound.
        turn_ctx._progress_metadata = _progress_metadata
        turn_ctx._progress_reply_to = _progress_reply_to

        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        turn_ctx.agent_holder = agent_holder
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer
        # streaming PCM audio consumer. Created on the gateway event-loop thread (NOT in run_sync's
        # executor worker) so outer finalisation / interrupt paths can reference it without a NameError.
        streaming_tts_consumer_holder: list = [None]
        turn_ctx.result_holder = result_holder
        turn_ctx.tools_holder = tools_holder
        turn_ctx.stream_consumer_holder = stream_consumer_holder
        turn_ctx.streaming_tts_consumer_holder = streaming_tts_consumer_holder

        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_running_loop()
        _hooks_ref = self.hooks

        # Bridge extracted to TurnRunner._step_callback_sync; the loop and
        # hooks refs bound just above are published at their original site.
        turn_ctx._loop_for_step = _loop_for_step
        turn_ctx._hooks_ref = _hooks_ref
        turn_ctx._step_callback_sync = turn_runner._step_callback_sync

        # Bridge sync event_callback → async hooks.emit for lifecycle events (e.g. session:compress
        # after a compression split); extracted to TurnRunner._event_callback_sync.
        turn_ctx._event_callback_sync = turn_runner._event_callback_sync

        # Bridge sync status_callback → async adapter.send for context pressure
        _status_adapter = self._adapter_for_source(source)
        _status_chat_id = source.chat_id
        _status_thread_metadata = self._run_agent_status_thread_metadata(
            source, event_message_id, _progress_thread_id, _relay_prospective_thread_id,
        )

        # Bridge extracted to TurnRunner._status_callback_sync; publish the status wiring computed
        # above onto the shared TurnContext at the exact original binding site.
        turn_ctx._status_adapter = _status_adapter
        turn_ctx._status_chat_id = _status_chat_id
        turn_ctx._status_thread_metadata = _status_thread_metadata
        turn_ctx._status_callback_sync = turn_runner._status_callback_sync
        return _status_thread_metadata

    async def _run_agent_notify_long_running(
        self,
        disp: "GatewayRunner._RunAgentDisplay",
        *,
        source: SessionSource,
        session_key: Optional[str],
        agent_holder: list,
        _executor_task_holder: list,
        _NOTIFY_INTERVAL: Optional[float],
        _long_running_mode: str,
        _notify_start: float,
        _status_thread_metadata: Optional[Dict[str, Any]],
        _cleanup_progress: bool,
        _cleanup_msg_ids: List[str],
    ) -> None:
        """Periodic \"still working\" heartbeat (edited in place where the adapter supports it).

        ``_executor_task_holder[0]`` is populated once the executor future exists; tolerate the
        brief window before then (it reads as None).
        """
        from gateway.run import _interim_metadata, _non_conversational_metadata
        if _NOTIFY_INTERVAL is None:
            return  # Notifications disabled (gateway_notify_interval: 0)
        _notify_adapter = self._adapter_for_source(source)
        if not _notify_adapter:
            return
        # Track the heartbeat message id to edit in place where supported (Telegram, Discord,
        # Slack, ...) instead of a new "Still working" bubble every interval.
        _heartbeat_msg_id: Optional[str] = None
        while True:
            await asyncio.sleep(_NOTIFY_INTERVAL)
            # Stop heartbeating once this run no longer owns the session slot or the executor has
            # finished, else a stale "running: delegate_task" bubble outlives its run. _executor_task
            # is bound just after this task is scheduled; tolerate the brief window before then.
            _exec_ref = _executor_task_holder[0]
            if not self._should_emit_long_running_notification(
                session_key, agent_holder[0], _exec_ref
            ):
                break
            _elapsed_mins = int((time.time() - _notify_start) // 60)
            # Default heartbeat is terse (elapsed + current tool); the verbose iteration counter is
            # gated on busy_ack_detail so users can opt in per platform.
            _agent_ref = agent_holder[0]
            _status_detail = ""
            _want_iteration_detail = bool(
                disp.resolve_display_setting(
                    disp.user_config,
                    disp.platform_key,
                    "busy_ack_detail",
                    True,
                )
            )
            if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                try:
                    _a = _agent_ref.get_activity_summary()
                    _parts = []
                    if _want_iteration_detail:
                        _parts.append(
                            f"iteration {_a['api_call_count']}/{_a['max_iterations']}"
                        )
                    _action = _a.get("current_tool") or _a.get("last_activity_desc")
                    if _action:
                        _parts.append(str(_action))
                    if _parts:
                        _status_detail = " — " + ", ".join(_parts)
                except Exception:
                    pass
            _heartbeat_text = (
                disp._generic_status_phrase("status")
                if _long_running_mode == "generic"
                else f"⏳ Working — {_elapsed_mins} min{_status_detail}"
            )
            try:
                _notify_res = None
                if _heartbeat_msg_id:
                    try:
                        _notify_res = await _notify_adapter.edit_message(
                            source.chat_id,
                            _heartbeat_msg_id,
                            _heartbeat_text,
                        )
                    except Exception as _ee:
                        logger.debug("Heartbeat edit failed: %s", _ee)
                        _notify_res = None
                if not (_notify_res and getattr(_notify_res, "success", False)):
                    _notify_res = await _notify_adapter.send(
                        source.chat_id,
                        _heartbeat_text,
                        metadata=_interim_metadata(_non_conversational_metadata(_status_thread_metadata, platform=source.platform)),
                    )
                    if getattr(_notify_res, "success", False) and getattr(
                        _notify_res, "message_id", None
                    ):
                        _heartbeat_msg_id = str(_notify_res.message_id)
                        if _cleanup_progress:
                            _cleanup_msg_ids.append(_heartbeat_msg_id)
            except Exception as _ne:
                logger.debug("Long-running notification error: %s", _ne)

    async def _run_agent_inner(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        inbound_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the agent; returns the full run_conversation result dict.

        Keys: "final_response", "messages", "api_calls", "completed".
        """
        from gateway.run import _float_env
        # ---- Proxy mode: delegate to remote API server ----
        if self._get_proxy_url():
            return await self._run_agent_via_proxy(
                message=message,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=event_message_id,
            )

        from run_agent import AIAgent

        disp = self._run_agent_display_settings(source)
        _display_surface_mode = disp._display_surface_mode
        needs_progress_queue = disp.needs_progress_queue
        log_mode_enabled = disp.log_mode_enabled
        log_queue = disp.log_queue

        turn_ctx, turn_runner, _cleanup_adapter = self._run_agent_build_turn_context(
            disp,
            AIAgent,
            message=message,
            context_prompt=context_prompt,
            history=history,
            source=source,
            session_id=session_id,
            session_key=session_key,
            run_generation=run_generation,
            _interrupt_depth=_interrupt_depth,
            event_message_id=event_message_id,
            inbound_message_id=inbound_message_id,
            channel_prompt=channel_prompt,
            moa_config=moa_config,
            persist_user_message=persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
        )
        _cleanup_progress = turn_ctx._cleanup_progress
        _cleanup_msg_ids = turn_ctx._cleanup_msg_ids

        (
            _progress_metadata,
            _progress_reply_to,
            _progress_thread_id,
            _relay_prospective_thread_id,
        ) = self._run_agent_progress_threading(source, event_message_id, disp._native_slack_task_cards)

        _status_thread_metadata = self._run_agent_bind_turn_wiring(
            turn_ctx,
            turn_runner,
            source,
            event_message_id,
            _progress_metadata,
            _progress_reply_to,
            _progress_thread_id,
            _relay_prospective_thread_id,
        )
        send_progress_messages = turn_runner.send_progress_messages
        agent_holder = turn_ctx.agent_holder
        result_holder = turn_ctx.result_holder
        tools_holder = turn_ctx.tools_holder
        stream_consumer_holder = turn_ctx.stream_consumer_holder
        streaming_tts_consumer_holder = turn_ctx.streaming_tts_consumer_holder

        self._run_agent_start_streaming_tts(
            source, message_type, _status_thread_metadata, streaming_tts_consumer_holder,
        )

        # run_sync extracted to TurnRunner.run_sync (bound method; executor call unchanged). Its
        # closed-over locals travel on turn_ctx; `nonlocal message` rebinds became ctx.message writes.
        run_sync = turn_runner.run_sync

        # Start the progress sender if enabled. Gate on needs_progress_queue (tool_progress OR
        # thinking_progress), not tool_progress alone: the sender drains BOTH tool-progress lines and
        # _thinking scratch bubbles — a tool_progress-only gate left thinking-only queues never drained.
        progress_task = None
        if needs_progress_queue:
            progress_task = asyncio.create_task(send_progress_messages())

        # Start the tool-call log writer when tool_progress == "log".
        log_task = None
        if log_mode_enabled:
            log_task = asyncio.create_task(self._run_agent_write_tool_log(log_queue))

        # Start stream consumer task — polls for consumer creation since it
        # happens inside run_sync (thread pool) after the agent is constructed.
        stream_task = None
        stream_task = asyncio.create_task(self._run_agent_stream_consumer_task(stream_consumer_holder))

        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        tracking_task = asyncio.create_task(
            self._run_agent_track_agent(session_key, run_generation, agent_holder)
        )

        _interrupt_detected = asyncio.Event()  # shared with backup check
        interrupt_monitor = asyncio.create_task(
            self._run_agent_monitor_for_interrupt(
                source,
                session_key,
                agent_holder,
                _interrupt_detected,
                streaming_tts_consumer_holder,
            )
        )

        # Periodic "still working" notifications so the user knows the agent hasn't died. Config:
        # agent.gateway_notify_interval or HERMES_AGENT_NOTIFY_INTERVAL env; default 180s.
        _NOTIFY_INTERVAL_RAW = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _NOTIFY_INTERVAL = _NOTIFY_INTERVAL_RAW if _NOTIFY_INTERVAL_RAW > 0 else None
        _long_running_mode = _display_surface_mode(
            "long_running_notifications",
            default=True,
            allow_generic=True,
        )
        if _long_running_mode == "off":
            _NOTIFY_INTERVAL = None
        _notify_start = time.time()
        _executor_task_holder: list = [None]  # bound once the executor future exists (see below)
        _notify_task = asyncio.create_task(
            self._run_agent_notify_long_running(
                disp,
                source=source,
                session_key=session_key,
                agent_holder=agent_holder,
                _executor_task_holder=_executor_task_holder,
                _NOTIFY_INTERVAL=_NOTIFY_INTERVAL,
                _long_running_mode=_long_running_mode,
                _notify_start=_notify_start,
                _status_thread_metadata=_status_thread_metadata,
                _cleanup_progress=_cleanup_progress,
                _cleanup_msg_ids=_cleanup_msg_ids,
            )
        )

        try:
            worker = self._run_agent_start_turn_worker(
                turn_ctx, run_sync, agent_holder, session_id, session_key, run_generation,
            )
            _executor_task_holder[0] = worker.executor_task  # read late by _notify_long_running
            response = await self._run_agent_await_turn_worker(
                worker,
                source=source,
                session_key=session_key,
                agent_holder=agent_holder,
                result_holder=result_holder,
                tools_holder=tools_holder,
                _interrupt_detected=_interrupt_detected,
                interrupt_monitor=interrupt_monitor,
                streaming_tts_consumer_holder=streaming_tts_consumer_holder,
                _status_thread_metadata=_status_thread_metadata,
            )

            self._run_agent_evict_on_fallback(session_key, agent_holder, result_holder)

            # Check if we were interrupted OR have a queued message (/queue).
            result = result_holder[0]
            adapter = self._adapter_for_source(source)

            await self._run_agent_finalize_streaming_tts(
                streaming_tts_consumer_holder, adapter, session_key, run_generation,
            )

            pending_event, pending = await self._run_agent_drain_pending(
                result, adapter, source, session_key,
            )

            if pending_event or pending:
                return await self._run_agent_queued_followup(
                    source=source,
                    adapter=adapter,
                    session_id=session_id,
                    session_key=session_key,
                    run_generation=run_generation,
                    _interrupt_depth=_interrupt_depth,
                    event_message_id=event_message_id,
                    context_prompt=context_prompt,
                    history=history,
                    pending=pending,
                    pending_event=pending_event,
                    response=response,
                    result=result,
                    result_holder=result_holder,
                    stream_consumer_holder=stream_consumer_holder,
                    stream_task=stream_task,
                    _status_thread_metadata=_status_thread_metadata,
                )
        finally:
            await self._run_agent_cleanup_turn_tasks(
                progress_task=progress_task,
                log_task=log_task,
                interrupt_monitor=interrupt_monitor,
                _notify_task=_notify_task,
                tracking_task=tracking_task,
                stream_task=stream_task,
                stream_consumer_holder=stream_consumer_holder,
                streaming_tts_consumer_holder=streaming_tts_consumer_holder,
                session_key=session_key,
                run_generation=run_generation,
            )

        await self._run_agent_mark_streamed_delivery(
            response, stream_consumer_holder, source, session_key,
        )
        self._run_agent_schedule_bubble_cleanup(
            response,
            _cleanup_progress,
            _cleanup_adapter,
            _cleanup_msg_ids,
            source,
            session_key,
            run_generation,
        )

        return response
