"""Gateway slash commands that switch or tune the model route: /model, /codex-runtime, /reasoning, /fast, /personality.

Split out of ``gateway/slash_commands.py``; bound onto ``GatewayRunner`` through
``GatewaySlashCommandsMixin``. Origin internals are imported lazily (``from gateway.slash_commands
import ...``) inside the bodies to avoid the import cycle.
"""

from __future__ import annotations

import logging
import asyncio
from agent.i18n import t
from gateway.platforms.base import MessageEvent
from hermes_cli.config import atomic_config_write, clear_model_endpoint_credentials
from typing import Optional
from utils import base_url_host_matches

# Log-record parity with gateway/run.py and the origin module.
logger = logging.getLogger("gateway.run")


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


class GatewayModelCommandsMixin:
    """Gateway slash commands that switch or tune the model route: /model, /codex-runtime, /reasoning, /fast, /personality."""

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
        from gateway.run import _hermes_home
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

    def _save_gateway_config_key(self, key_path: str, value) -> bool:
        """Save a dot-separated key to config.yaml (shared by /reasoning, /fast
        and their interactive pickers)."""
        from gateway.slash_commands import _nested_dict
        from gateway.run import _gateway_config_home
        from hermes_cli.config import read_user_config_raw
        config_path = _gateway_config_home() / "config.yaml"
        try:
            # Write-back round-trip: raw read is correct (merged defaults must
            # not be persisted back to the user's file).
            user_config = read_user_config_raw(config_path)
            *parents, leaf = key_path.split(".")
            _nested_dict(user_config, *parents)[leaf] = value
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

        choices = [{"value": "none", "label": t("gateway.reasoning.choice_none"), "is_current": current_effort == "none"}]
        choices.extend({"value": level, "label": level, "is_current": level == current_effort} for level in VALID_REASONING_EFFORTS)
        choices.extend(
            {"value": v, "label": t(f"gateway.reasoning.choice_{v}"), "is_current": False}
            for v in ("reset", "show", "hide")
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
