"""Model picker, /model switch application, runtime snapshot/restore, and codex runtime handling for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).

Test stubs drive the switch paths with ``object.__new__`` / SimpleNamespace objects that
lack most mixin methods, so shared steps are module-level functions taking ``cli`` and
sibling methods are invoked through ``HermesCLI.<name>(self, ...)``.
"""

from __future__ import annotations

import copy
import sys
import threading

from rich.markup import escape as _escape
from utils import base_url_host_matches

# CLI-level fields that together describe the active model route; snapshotted before a
# switch / one-turn override and restored wholesale on rollback.
_RUNTIME_FIELDS = (
    "model", "provider", "requested_provider", "_explicit_api_key", "_explicit_base_url",
    "api_key", "base_url", "api_mode",
)


def _runtime_fields(cli) -> dict:
    return {key: getattr(cli, key, None) for key in _RUNTIME_FIELDS}


def _heal_bare_custom_provider(provider, *, base_url, model):
    """Bare ``custom`` is the resolved billing class, not a routable identity.

    Persisting/restoring it verbatim makes a later resume hard-fail once the config default
    has moved off the custom endpoint (resolve_runtime_provider only trusts config base_url
    for bare custom while the config provider is still custom-ish). Recover the durable
    ``custom:<name>`` menu key from the endpoint, else drop the provider (None).
    """
    if str(provider or "").strip().lower() != "custom":
        return provider
    try:
        from hermes_cli.runtime_provider import canonical_custom_identity
        return canonical_custom_identity(base_url=base_url or None, model=model or None) or None
    except Exception:
        return None


def _merge_preflight_warning(cli, result, custom_providers) -> None:
    """Fold the context-compression preflight warning into ``result`` (fail-soft)."""
    from cli import logger
    if cli.agent is None:
        return
    try:
        from hermes_cli.context_switch_guard import merge_preflight_compression_warning
        # Prefer the fresh inventory list (same source as switch_model / TUI); fall back
        # to the agent-init snapshot.
        merge_preflight_compression_warning(
            result,
            agent=cli.agent,
            messages=list(cli.conversation_history or []),
            custom_providers=custom_providers if custom_providers is not None
            else getattr(cli.agent, "_custom_providers", None),
            config_context_length=getattr(cli.agent, "_config_context_length", None),
        )
    except Exception as exc:
        logger.debug("preflight-compression switch warning failed: %s", exc)


def _print_switch_summary(cli, result, old_model, *, one_turn: bool, strict_context: bool) -> None:
    """Record the next-turn switch note and print the "Model switched" block.

    The note is prepended to the next user message (not injected as a system message
    mid-history, which breaks providers and prompt caching). ``strict_context``: the
    typed /model path lets context-resolution errors propagate; the picker path swallows them.
    """
    from cli import _cprint
    from hermes_cli.model_switch import format_model_for_display, resolve_display_context_length
    _display_old = format_model_for_display(old_model)
    _display_new = format_model_for_display(result.new_model)
    cli._pending_model_switch_note = (
        f"[Note: model was just switched from {_display_old} to {_display_new} "
        f"via {result.provider_label or result.target_provider}. "
        f"{'This override applies to the next turn only. ' if one_turn else ''}"
        f"Adjust your self-identification accordingly.]"
    )
    _cprint(f"  ✓ Model switched: {_display_new}")
    _cprint(f"    Provider: {result.provider_label or result.target_provider}")

    # Context: always resolve via the provider-aware chain so Codex OAuth, Copilot, and
    # Nous-enforced caps win over the raw models.dev entry (gpt-5.5 is 1.05M on openai
    # but 272K on Codex OAuth).
    mi = result.model_info
    agent = cli.agent
    try:
        ctx = resolve_display_context_length(
            result.new_model, result.target_provider,
            base_url=result.base_url or cli.base_url or "",
            api_key=result.api_key or cli.api_key or "", model_info=mi,
            config_context_length=getattr(agent, "_config_context_length", None) if agent else None,
            custom_providers=getattr(agent, "_custom_providers", None) if agent else None,
        )
    except Exception:
        if strict_context:
            raise
        ctx = None
    if ctx:
        _cprint(f"    Context: {ctx:,} tokens")
    if mi:
        if mi.max_output:
            _cprint(f"    Max output: {mi.max_output:,} tokens")
        _cprint(f"    Capabilities: {mi.format_capabilities()}")
    cache_enabled = (
        (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
        or result.api_mode == "anthropic_messages"
    )
    if cache_enabled:
        _cprint("    Prompt caching: enabled")
    if result.warning_message:
        _cprint(f"    ⚠ {result.warning_message}")


def _switch_model_from(
    cli, raw_input, *, is_global, explicit_provider, user_providers, custom_providers
):
    """``switch_model`` seeded with this CLI's live route."""
    from hermes_cli.model_switch import switch_model
    return switch_model(
        raw_input=raw_input, current_provider=cli.provider or "", current_model=cli.model or "",
        current_base_url=cli.base_url or "", current_api_key=cli.api_key or "", is_global=is_global,
        explicit_provider=explicit_provider, user_providers=user_providers,
        custom_providers=custom_providers,
    )


def _run_confirm_and_apply(cli, target, *args) -> None:
    """Run a confirm+apply sequence off the UI thread when the TUI is live.

    The expensive-model modal blocks its thread on a response queue (_prompt_text_input_modal);
    on the prompt_toolkit main thread that freezes rendering, so the modal never appears and the
    switch silently cancels after the 120s timeout.
    """
    if getattr(cli, "_app", None):
        threading.Thread(target=target, args=args, daemon=True).start()
    else:
        target(*args)


def _commit_model_switch(
    cli, result, *, persist_global: bool, one_turn: bool = False, picker: bool = False
) -> None:
    """Stage + swap, print the summary, persist (session row unless --once; config on --global).

    ``picker``: the picker path tolerates context-resolution errors and labels the config write
    "(--global)"; the typed /model path additionally records the one-turn restore snapshot.
    """
    from cli import HermesCLI, _cprint
    old_model = cli.model
    snapshot = cli._snapshot_model_runtime() if one_turn else None
    if not cli._stage_and_swap_model(result, old_model):
        return
    if not picker:
        cli._pending_one_turn_model_restore = snapshot
    _print_switch_summary(cli, result, old_model, one_turn=one_turn, strict_context=not picker)
    if persist_global:
        _persist_global_switch(cli, result)
        _cprint("    Saved to config.yaml (--global)" if picker else "    Saved to config.yaml")
    elif one_turn:
        _cprint("    (next turn only — restores after one response)")
    else:
        _cprint("    (session only — add --global to persist)")
    # --global also updates config.yaml (future sessions), but the row still records what THIS
    # session runs — otherwise a later resume would restore the stale creation-time model over
    # the new global choice. --once is ephemeral and restored after one turn: never touch the row.
    if not one_turn:
        HermesCLI._persist_model_switch_to_session(cli, result)


def _persist_global_switch(cli, result) -> None:
    """Write the switched route to config.yaml (--global).

    base_url/api_mode are always freshly resolved for the target provider (model_switch.py),
    so sync them every time; None clears a value the new provider doesn't need — otherwise a
    global switch leaves the OLD provider's endpoint/wire-protocol in config.yaml (#25106).
    """
    from cli import HermesCLI, save_config_value
    HermesCLI._clear_persisted_context_for_model_switch(cli, result)
    save_config_value("model.default", result.new_model)
    save_config_value("model.provider", result.target_provider)
    save_config_value("model.base_url", result.base_url or None)
    save_config_value("model.api_mode", result.api_mode or None)


def _show_model_picker(cli, ctx, force_refresh: bool) -> None:
    """``/model`` with no args: open the picker, or print usage when nothing is authed."""
    from cli import _cprint
    from hermes_cli.inventory import build_models_payload
    from hermes_cli.providers import get_label
    try:
        if ctx is None:
            raise RuntimeError("inventory context unavailable")
        providers = build_models_payload(
            ctx, probe_custom_providers=force_refresh,
            probe_current_custom_provider=not force_refresh,
        )["providers"]
    except Exception:
        providers = []
    if not providers:
        _cprint("  No authenticated providers found.")
        _cprint("")
        _cprint("  /model <name>                        switch model (this session)")
        _cprint("  /model <name> --global               switch model and persist as default")
        _cprint("  /model <name> --once                 switch for the next turn only")
        _cprint("  /model <name> --session              switch for this session only")
        _cprint("  /model --provider <slug>             switch provider")
        _cprint("  /model --refresh                     re-fetch live model lists")
        return
    cli._open_model_picker(
        providers, cli.model or "unknown", get_label(cli.provider) if cli.provider else "unknown",
        user_provs=ctx.user_providers if ctx is not None else None,
        custom_provs=ctx.custom_providers if ctx is not None else None,
    )


class CLIModelSwitchMixin:
    """Model picker, /model switch application, runtime snapshot/restore, and codex runtime handling for the interactive CLI"""

    def _normalize_model_for_provider(self, resolved_provider: str) -> bool:
        """Normalize provider-specific model IDs and routing."""
        from cli import _split_model_config_default
        current_model = str(self.model or "").strip()
        if isinstance(self.model, dict):
            current_model, _ = _split_model_config_default(self.model)
        changed = False

        def _adopt(canonical, notice) -> None:
            """Adopt ``canonical`` when it differs; ``notice(new)`` builds the warning text."""
            nonlocal current_model, changed
            if canonical and canonical != current_model:
                if not self._model_is_default:
                    self._console_print(f"[yellow]⚠️  {notice(canonical)}[/]")
                self.model = canonical
                current_model = canonical
                changed = True

        def _set_mode(resolved_mode) -> None:
            nonlocal changed
            if resolved_mode != self.api_mode:
                self.api_mode = resolved_mode
                changed = True

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS, normalize_model_for_provider
            )
            if resolved_provider not in _AGGREGATOR_PROVIDERS:
                _adopt(
                    normalize_model_for_provider(current_model, resolved_provider),
                    lambda new: (
                        f"Normalized model '{current_model}' to '{new}' for {resolved_provider}."
                    ),
                )
        except Exception:
            pass

        if resolved_provider == "copilot":
            try:
                from hermes_cli.models import copilot_model_api_mode, normalize_copilot_model_id
                _adopt(
                    normalize_copilot_model_id(current_model, api_key=self.api_key),
                    lambda new: f"Normalized Copilot model '{current_model}' to '{new}'.",
                )
                _set_mode(copilot_model_api_mode(current_model, api_key=self.api_key))
            except Exception:
                pass
            return changed

        from hermes_cli.models import opencode_provider_family
        if opencode_provider_family(resolved_provider) is not None:
            try:
                from hermes_cli.models import normalize_opencode_model_id, opencode_model_api_mode
                _adopt(
                    normalize_opencode_model_id(resolved_provider, current_model),
                    lambda new: (
                        f"Stripped provider prefix from '{current_model}'; "
                        f"using '{new}' for {resolved_provider}."
                    ),
                )
                _set_mode(opencode_model_api_mode(resolved_provider, current_model))
            except Exception:
                pass
            return changed

        if resolved_provider != "openai-codex":
            return changed

        # 1. Strip provider prefix ("openai/gpt-5.4" → "gpt-5.4")
        if "/" in current_model:
            slug = current_model.split("/", 1)[1]
            if not self._model_is_default:
                self._console_print(
                    f"[yellow]⚠️  Stripped provider prefix from '{current_model}'; "
                    f"using '{slug}' for OpenAI Codex.[/]"
                )
            self.model = slug
            current_model = slug
            changed = True

        # 2. Replace untouched default with a Codex model
        if self._model_is_default:
            fallback_model = "gpt-5.3-codex"
            try:
                from hermes_cli.codex_models import get_codex_model_ids
                available = get_codex_model_ids(access_token=self.api_key if self.api_key else None)
                if available:
                    fallback_model = available[0]
            except Exception:
                pass
            if current_model != fallback_model:
                self.model = fallback_model
                changed = True
        return changed

    def _persist_model_switch_to_session(self, result) -> None:
        """Persist a session-scoped /model switch to the session DB row.

        Writes the model column plus the runtime route so ``--resume`` (CLI, reads
        ``gateway_runtime``) and ``session.resume`` (TUI/desktop, reads top-level
        ``model_config`` keys) both restore the switched provider instead of recombining
        the model with the ambient default. Mirrors the gateway's ``update_session_model()``.
        Both shapes derive from one ``route`` dict with or-None values so stale keys from a
        previous switch are DELETED (``_merge_model_config_json`` only deletes on explicit
        None) and the two shapes can never diverge.
        """
        from cli import logger
        db = getattr(self, "_session_db", None)
        sid = getattr(self, "session_id", None)
        if not db or not sid:
            return
        route = {
            "provider": _heal_bare_custom_provider(
                result.target_provider, base_url=result.base_url, model=result.new_model,
            ) or None,
            "base_url": result.base_url or None,
            "api_mode": result.api_mode or None,
        }
        try:
            db.update_session_model(sid, result.new_model)
            db.patch_session_model_config(sid, {"gateway_runtime": route, **route})
        except Exception:
            logger.debug("Failed to persist model switch to session DB", exc_info=True)

    def _restore_session_model(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Restore model/provider from the session DB row on resume.

        Called from every resume path (startup ``--resume``/``-c`` and mid-chat ``/resume``);
        without it a resumed session silently falls back to the config default model. Skips
        when no model is recorded or the CLI was launched with an explicit ``-m`` (user intent
        wins). When the stored provider differs from the ambient one, credentials are
        re-resolved for it — the ambient ``api_key`` belongs to the config-default provider
        and must not be sent to the session's endpoint; on failure the ambient credentials are
        kept so the session still opens (the first turn surfaces the auth error).
        """
        from cli import logger
        stored_model = (session_meta or {}).get("model")
        if not stored_model or getattr(self, "_explicit_model_override", False):
            return
        # Canonical row-level reader: prefers model_config.gateway_runtime, falls back to
        # the TUI gateway's top-level keys.
        from hermes_state import SessionDB as _SessionDB
        _stored_runtime = _SessionDB.session_gateway_runtime(session_meta)
        stored_base_url = _stored_runtime.get("base_url") or None
        stored_api_mode = _stored_runtime.get("api_mode") or None
        # Stricter than the TUI gateway's recovery (which keeps bare "custom" when a
        # base_url exists) — the CLI's resolve path would hard-fail on it.
        stored_provider = _heal_bare_custom_provider(
            _stored_runtime.get("provider") or None, base_url=stored_base_url, model=stored_model,
        )
        model_changed = stored_model != self.model
        provider_changed = bool(stored_provider) and stored_provider != self.provider
        if not model_changed and not provider_changed:
            return
        self.model = stored_model
        if stored_provider:
            self.provider = stored_provider
            self.requested_provider = stored_provider
            if stored_base_url:
                self.base_url = stored_base_url
            if stored_api_mode:
                self.api_mode = stored_api_mode
        if provider_changed:
            # Stale launch-time explicit overrides belong to the AMBIENT provider; carrying
            # them into the restored provider's resolution poisons
            # _ensure_runtime_credentials on startup resume.
            self._explicit_api_key = None
            self._explicit_base_url = stored_base_url
            # api_key is never persisted to the session DB (by design) — runtime provider
            # resolution owns credentials.
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                resolved = resolve_runtime_provider(requested=stored_provider)
                if resolved.get("api_key"):
                    self.api_key = resolved["api_key"]
                    self._credential_pool = resolved.get("credential_pool")
                if not stored_base_url and resolved.get("base_url"):
                    self.base_url = resolved["base_url"]
                if not stored_api_mode and resolved.get("api_mode"):
                    self.api_mode = resolved["api_mode"]
            except Exception:
                logger.debug(
                    "Credential re-resolution for resumed session provider "
                    "%s failed; keeping ambient credentials",
                    stored_provider, exc_info=True,
                )
        # Mid-chat /resume: swap the live agent in place. On startup --resume the agent
        # isn't built yet — _init_agent picks up self.model / self.provider.
        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=self.model, new_provider=self.provider, api_key=self.api_key or "",
                    base_url=self.base_url or "", api_mode=self.api_mode or "",
                )
            except Exception:
                logger.debug("In-place agent model swap on resume failed", exc_info=True)
        msg = f"Model restored from session: {stored_model}"
        if stored_provider:
            msg += f" ({stored_provider})"
        if quiet:
            print(msg, file=sys.stderr)
        else:
            self._console_print(f"[dim]{_escape(msg)}[/dim]")

    def _open_model_picker(self, providers: list, current_model: str, current_provider: str, user_provs=None, custom_provs=None) -> None:
        """Open prompt_toolkit-native /model picker modal."""
        self._capture_modal_input_snapshot()
        self._model_picker_state = {
            "stage": "provider",
            "providers": providers,
            "selected": next((i for i, p in enumerate(providers) if p.get("is_current")), 0),
            "current_model": current_model,
            "current_provider": current_provider,
            "user_provs": user_provs,
            "custom_provs": custom_provs,
            "filter": "",
        }
        self._invalidate(min_interval=0.0)

    def _confirm_expensive_model_switch(self, result) -> bool:
        """Ask for explicit confirmation before applying costly model switches."""
        if not getattr(result, "success", False):
            return True
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning
            warning = combined_selection_warning(
                result.new_model, provider=result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "", model_info=result.model_info,
            )
        except Exception:
            warning = None
        if warning is None:
            return True
        choices = [
            ("once", "Switch anyway", "Use this model for the current Hermes session."),
            ("cancel", "Cancel", "Keep the current model."),
        ]
        raw = self._prompt_text_input_modal(
            title=f"!!! {warning.title} !!!", detail=warning.message, choices=choices, timeout=120,
        )
        return self._normalize_slash_confirm_choice(raw, choices) == "once"

    def _confirm_and_apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None
    ) -> None:
        from cli import _cprint
        try:
            if result.success and not self._confirm_expensive_model_switch(result):
                _cprint("  Model switch cancelled.")
                return
            self._apply_model_switch_result(
                result, persist_global, custom_providers=custom_providers
            )
        except Exception as exc:
            _cprint(f"  ✗ Model selection failed: {exc}")

    def _close_model_picker(self) -> None:
        self._model_picker_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _snapshot_model_runtime(self) -> dict:
        """Capture current CLI and agent model runtime for one-turn restore."""
        agent = getattr(self, "agent", None)
        return {
            **_runtime_fields(self),
            "agent_primary_runtime": copy.deepcopy(
                getattr(agent, "_primary_runtime", None)
            ) if agent is not None else None,
        }

    def _restore_model_runtime_snapshot(self, snapshot: dict | None) -> None:
        """Restore a model runtime captured before a one-turn override."""
        from cli import logger
        if not snapshot:
            return
        for key in _RUNTIME_FIELDS:
            if key in snapshot:
                setattr(self, key, snapshot.get(key))

        agent = getattr(self, "agent", None)
        if agent is None:
            return
        primary = snapshot.get("agent_primary_runtime")
        if primary and hasattr(agent, "_restore_primary_runtime"):
            try:
                agent._primary_runtime = copy.deepcopy(primary)
                agent._fallback_activated = True
                agent._rate_limited_until = 0
                if agent._restore_primary_runtime():
                    return
            except Exception:
                logger.debug("CLI one-turn model restore via primary runtime failed", exc_info=True)
        if hasattr(agent, "switch_model"):
            try:
                agent.switch_model(
                    new_model=snapshot.get("model", ""), new_provider=snapshot.get("provider", ""),
                    api_key=snapshot.get("api_key", ""), base_url=snapshot.get("base_url", ""),
                    api_mode=snapshot.get("api_mode", ""),
                    capabilities=snapshot.get("capabilities"),
                )
            except Exception as exc:
                logger.warning("CLI one-turn model restore failed: %s", exc)

    @staticmethod
    def _filter_model_picker_entries(entries: list, query: str) -> list:
        """Return (original_index, label) pairs for entries matching ``query``.

        Case-insensitive subsequence match; an empty query matches everything. Pairs carry
        the ORIGINAL index into ``entries`` so a selection in the filtered view resolves to
        exactly one concrete model — filtering never introduces fuzzy *resolution*.
        """
        pairs = list(enumerate(entries))
        q = (query or "").strip().lower()
        if not q:
            return pairs

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        return [(i, e) for (i, e) in pairs if _subseq(q, str(e).lower())]

    @staticmethod
    def _compute_model_picker_viewport(
        selected: int, scroll_offset: int, n: int, term_rows: int, reserved_below: int = 6,
        panel_chrome: int = 6, min_visible: int = 3,
    ) -> tuple[int, int]:
        """Resolve (scroll_offset, visible) for the /model picker viewport.

        ``reserved_below`` matches the approval / clarify panels (input area, status bar,
        separators); ``panel_chrome`` is this panel's borders + blanks + hint row. The
        offset slides to keep ``selected`` on screen.
        """
        max_visible = max(min_visible, term_rows - reserved_below - panel_chrome)
        if n <= max_visible:
            return 0, n
        visible = max_visible
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + visible:
            scroll_offset = selected - visible + 1
        return max(0, min(scroll_offset, n - visible)), visible

    def _clear_persisted_context_for_model_switch(self, result) -> None:
        """Drop a global context pin when its configured owner changes."""
        from cli import save_config_value
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.route_identity import should_clear_context_pin
            config = load_config_readonly()
            model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
            if not isinstance(model_cfg, dict) or "context_length" not in model_cfg:
                return
            if should_clear_context_pin(
                model_cfg.get("default") or model_cfg.get("model"), result.new_model,
                model_cfg.get("base_url"), result.base_url,
                model_cfg.get("provider"), result.target_provider,
            ):
                save_config_value("model.context_length", None)
        except Exception:
            save_config_value("model.context_length", None)

    def _stage_and_swap_model(self, result, old_model) -> bool:
        """Stage ``result`` onto the CLI fields, then swap the live agent in place.

        CLI-level fields are snapshotted first (mirrors the gateway) so a failed in-place
        agent swap rolls the whole CLI back to the old working model — otherwise the broken
        credentials staged here leak into the next turn's resolution even though the agent
        itself rolled back. Returns False after printing the failure so the caller aborts
        the rest of the commit (a failed switch is a no-op, not a dead session).
        """
        from cli import _cprint
        _cli_snapshot = _runtime_fields(self)
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the previous
        # provider (e.g. Ollama api_key/base_url) don't leak into the next resolution.
        self._explicit_api_key = result.api_key
        self._explicit_base_url = result.base_url
        if result.api_key:
            self.api_key = result.api_key
        if result.base_url:
            self.base_url = result.base_url
        if result.api_mode:
            self.api_mode = result.api_mode

        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=result.new_model, new_provider=result.target_provider,
                    api_key=result.api_key, base_url=result.base_url, api_mode=result.api_mode,
                    capabilities=getattr(result, "runtime_capabilities", None),
                )
            except Exception as exc:
                for _k, _v in _cli_snapshot.items():
                    setattr(self, _k, _v)
                _cprint(
                    f"  ⚠ Model switch to {result.new_model} failed ({exc}); "
                    f"staying on {old_model}."
                )
                return False
        return True

    def _apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None
    ) -> None:
        """Picker-path commit (see _commit_model_switch)."""
        from cli import _cprint
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return
        _merge_preflight_warning(self, result, custom_providers)
        _commit_model_switch(self, result, persist_global=persist_global, picker=True)

    def _handle_model_picker_selection(self, persist_global: bool = False) -> None:
        state = self._model_picker_state
        if not state:
            return
        selected = state.get("selected", 0)
        stage = state.get("stage")
        if stage == "provider":
            providers = state.get("providers") or []
            if selected >= len(providers):
                self._close_model_picker()
                return
            provider_data = providers[selected]
            # Curated list from list_authenticated_providers() (same as `hermes model` and
            # gateway pickers); live catalog only when the curated list is empty
            # (user-defined endpoints).
            model_list = provider_data.get("models", [])
            if not model_list:
                try:
                    from hermes_cli.models import provider_model_ids
                    model_list = provider_model_ids(provider_data["slug"]) or model_list
                except Exception:
                    pass
            state.update(
                stage="model", provider_data=provider_data, model_list=model_list,
                selected=0, filter="", _filtered_pairs=None,
            )
            self._invalidate(min_interval=0.0)
            return
        if stage == "model":
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            # Map the selected row through the active fuzzy filter; the filtered pair
            # carries the ORIGINAL index, so the resolved model is one concrete entry.
            filtered_pairs = state.get("_filtered_pairs")
            if filtered_pairs is None:
                filtered_pairs = list(enumerate(model_list))
            visible_labels = [e for (_i, e) in filtered_pairs]
            back_idx = len(visible_labels)
            if selected == back_idx:
                state.update(
                    stage="provider", filter="", _filtered_pairs=None,
                    selected=next((i for i, p in enumerate(state.get("providers") or [])
                                   if p.get("slug") == provider_data.get("slug")), 0),
                )
                self._invalidate(min_interval=0.0)
                return
            if selected > back_idx:  # cancel row (and anything past it)
                self._close_model_picker()
                return
            if 0 <= selected < back_idx:
                result = _switch_model_from(
                    self, visible_labels[selected], is_global=persist_global,
                    explicit_provider=provider_data.get("slug"),
                    user_providers=state.get("user_provs"),
                    custom_providers=state.get("custom_provs"),
                )
                # Capture before close — picker state is cleared on close.
                _picker_custom_provs = state.get("custom_provs")
                self._close_model_picker()
                _run_confirm_and_apply(
                    self, self._confirm_and_apply_model_switch_result,
                    result, persist_global, _picker_custom_provs,
                )
                return
            self._close_model_picker()

    def _handle_model_switch(self, cmd_original: str):
        """Handle /model command — switch model.

        Supports:
          /model                              — show current model + usage hints
          /model <name>                       — switch model (this session only)
          /model <name> --once                — switch for the next turn only
          /model <name> --session             — switch for this session only (explicit)
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model

        Persistence defaults to off (``model.persist_switch_by_default``, default False —
        switches are session-scoped). ``--global`` persists, ``--once`` is next-turn only.
        """
        from cli import _cprint
        from hermes_cli.model_switch import parse_model_switch_args, resolve_persist_behavior

        parts = cmd_original.split(None, 1)  # split off '/model'
        # Single-owner flag parser (hermes_cli.model_switch): --provider/--global/--session/
        # --once/--refresh.
        request = parse_model_switch_args(parts[1].strip() if len(parts) > 1 else "")
        if request.errors:
            # CLI decoration: "  ✗ " prefix over the canonical error copy.
            _cprint(f"  ✗ {request.error_messages()[0]}")
            return
        one_turn = request.is_once
        persist_global = resolve_persist_behavior(
            request.is_global, request.is_session, is_once=one_turn,
            explicit_provider=request.explicit_provider,
        )

        # --refresh: wipe the on-disk picker cache so every authed provider's /v1/models
        # is re-fetched live on this open.
        if request.force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
                _cprint("  Cleared model picker cache. Refreshing...")
            except Exception:
                pass

        # Single inventory context; live session state overlaid via with_overrides
        # (truthy-only) so empty self.* attrs don't clobber disk config.
        from hermes_cli.inventory import load_picker_context
        try:
            ctx = load_picker_context().with_overrides(
                current_provider=self.provider or "", current_model=self.model or "",
                current_base_url=self.base_url or "",
            )
        except Exception:
            ctx = None
        # switch_model() + _open_model_picker still need the raw provider dicts.
        user_provs = ctx.user_providers if ctx is not None else None
        custom_provs = ctx.custom_providers if ctx is not None else None

        if not request.target and not request.explicit_provider:
            return _show_model_picker(self, ctx, request.force_refresh)

        result = _switch_model_from(
            self, request.target, is_global=persist_global,
            explicit_provider=request.explicit_provider,
            user_providers=user_provs, custom_providers=custom_provs,
        )
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return
        _merge_preflight_warning(self, result, custom_provs)
        _run_confirm_and_apply(
            self, self._confirm_and_apply_cli_model_switch,
            result, persist_global, one_turn, custom_provs,
        )

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global: bool, one_turn: bool, custom_provs=None
    ) -> None:
        """Confirm an expensive model switch and apply it to CLI state.

        Runs on a worker thread when the TUI is active (see _run_confirm_and_apply) so the
        confirmation modal can render. Updates requested_provider (via the swap) so
        _ensure_runtime_credentials() doesn't overwrite the switch on the next turn.
        """
        from cli import _cprint
        if not self._confirm_expensive_model_switch(result):
            _cprint("  Model switch cancelled.")
            return
        _commit_model_switch(self, result, persist_global=persist_global, one_turn=one_turn)

    def _handle_codex_runtime(self, cmd_original: str) -> None:
        """Handle /codex-runtime — toggle the codex app-server runtime opt-in.

        Usage:
            /codex-runtime                       — show current state
            /codex-runtime auto                  — Hermes default (chat_completions)
            /codex-runtime codex_app_server      — hand turns to codex subprocess
            /codex-runtime on / off              — synonyms for the above
        """
        from cli import _cprint
        from hermes_cli import codex_runtime_switch as crs

        parts = cmd_original.split(None, 1)
        new_value, errors = crs.parse_args(parts[1].strip() if len(parts) > 1 else "")
        if errors:
            for err in errors:
                _cprint(f"❌ {err}")
            return
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            _cprint(f"❌ could not load config: {exc}")
            return
        result = crs.apply(
            load_config(), new_value,
            persist_callback=(save_config if new_value is not None else None),
        )
        prefix = "✓" if result.success else "✗"
        for line in result.message.splitlines():
            _cprint(f"  {prefix} {line}" if line.startswith("openai_runtime") else f"    {line}")
        if result.success and result.requires_new_session:
            _cprint("    Tip: `/reset` starts a new session immediately.")

    def _should_handle_model_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /model should be handled immediately on the UI thread."""
        from cli import _looks_like_slash_command
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        try:
            from hermes_cli.commands import resolve_command
            cmd = resolve_command(text.split(None, 1)[0].lower().lstrip('/'))
            return bool(cmd and cmd.name == "model")
        except Exception:
            return False

    def _cmd_moa(self, cmd_original: str):
        """/moa is one-shot sugar: run one prompt through the default MoA preset, then restore
        the prior model. Switching to MoA for the session is done via the model picker (MoA
        presets surface as a virtual "Mixture of Agents" provider)."""
        from cli import _cprint, _slash_args
        from hermes_cli.moa_config import moa_usage, normalize_moa_config

        payload = _slash_args(cmd_original)
        if not payload:
            _cprint(f"  {moa_usage()}")
            return True
        moa_cfg = self.config.get("moa") if isinstance(self.config, dict) else {}
        preset = normalize_moa_config(moa_cfg)["default_preset"]
        self._pending_moa_restore_model = {
            key: getattr(self, key, None)
            for key in (
                "requested_provider", "provider", "model", "api_key", "base_url", "api_mode",
            )
        }
        self.requested_provider = "moa"
        self.provider = "moa"
        self.model = preset
        self.api_key = "moa-virtual-provider"
        self.base_url = "moa://local"
        self.api_mode = "chat_completions"
        self.agent = None
        self._pending_moa_disable_after_turn = True
        self._pending_agent_seed = payload
        _cprint(f"  MoA one-shot queued with preset {preset}; previous model will be restored after this turn.")
