"""Model picker, /model switch application, runtime snapshot/restore, and codex runtime handling for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import copy
import sys
import threading

from rich.markup import escape as _escape
from utils import base_url_host_matches


class CLIModelSwitchMixin:
    """Model picker, /model switch application, runtime snapshot/restore, and codex runtime handling for the interactive CLI"""

    def _normalize_model_for_provider(self, resolved_provider: str) -> bool:
        """Normalize provider-specific model IDs and routing."""
        from cli import _split_model_config_default
        current_model = str(self.model or "").strip()
        if isinstance(self.model, dict):
            _m, _ = _split_model_config_default(self.model)
            current_model = _m
        changed = False

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if resolved_provider not in _AGGREGATOR_PROVIDERS:
                normalized_model = normalize_model_for_provider(current_model, resolved_provider)
                if normalized_model and normalized_model != current_model:
                    if not self._model_is_default:
                        self._console_print(
                            f"[yellow]⚠️  Normalized model '{current_model}' to '{normalized_model}' for {resolved_provider}.[/]"
                        )
                    self.model = normalized_model
                    current_model = normalized_model
                    changed = True
        except Exception:
            pass

        if resolved_provider == "copilot":
            try:
                from hermes_cli.models import copilot_model_api_mode, normalize_copilot_model_id

                canonical = normalize_copilot_model_id(current_model, api_key=self.api_key)
                if canonical and canonical != current_model:
                    if not self._model_is_default:
                        self._console_print(
                            f"[yellow]⚠️  Normalized Copilot model '{current_model}' to '{canonical}'.[/]"
                        )
                    self.model = canonical
                    current_model = canonical
                    changed = True

                resolved_mode = copilot_model_api_mode(current_model, api_key=self.api_key)
                if resolved_mode != self.api_mode:
                    self.api_mode = resolved_mode
                    changed = True
            except Exception:
                pass
            return changed

        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(resolved_provider) is not None:
            try:
                from hermes_cli.models import normalize_opencode_model_id, opencode_model_api_mode

                canonical = normalize_opencode_model_id(resolved_provider, current_model)
                if canonical and canonical != current_model:
                    if not self._model_is_default:
                        self._console_print(
                            f"[yellow]⚠️  Stripped provider prefix from '{current_model}'; using '{canonical}' for {resolved_provider}.[/]"
                        )
                    self.model = canonical
                    current_model = canonical
                    changed = True

                resolved_mode = opencode_model_api_mode(resolved_provider, current_model)
                if resolved_mode != self.api_mode:
                    self.api_mode = resolved_mode
                    changed = True
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

                available = get_codex_model_ids(
                    access_token=self.api_key if self.api_key else None,
                )
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

        Writes the model column plus the runtime route so ``--resume``
        (CLI, reads ``gateway_runtime``) and ``session.resume`` (TUI/desktop,
        reads top-level ``model_config`` keys via
        ``_stored_session_runtime_overrides``) both restore the switched
        provider instead of recombining the model with the ambient default
        (#79536). Mirrors the gateway's ``update_session_model()`` call.
        getattr: tests drive the switch paths with ``object.__new__`` stubs.
        """
        from cli import logger
        db = getattr(self, "_session_db", None)
        sid = getattr(self, "session_id", None)
        if not db or not sid:
            return
        provider = result.target_provider
        # Bare "custom" is the resolved billing class, not a routable
        # identity — persisting it verbatim makes a later resume hard-fail
        # when the config default has moved off the custom endpoint
        # (resolve_runtime_provider only trusts config base_url for bare
        # custom while the config provider is still custom-ish). Heal to
        # the durable custom:<name> menu key, else drop the provider —
        # same recovery the TUI gateway applies on its read path.
        if str(provider or "").strip().lower() == "custom":
            try:
                from hermes_cli.runtime_provider import canonical_custom_identity
                provider = canonical_custom_identity(
                    base_url=result.base_url or None,
                    model=result.new_model or None,
                ) or None
            except Exception:
                provider = None
        # Both shapes use the same or-None discipline so stale keys from a
        # previous switch are deleted (not merely omitted) in BOTH the
        # nested gateway_runtime dict (CLI reader) and the top-level keys
        # (TUI gateway reader). _merge_model_config_json only deletes on
        # explicit None, so falsy values must be converted, not filtered.
        # Deriving the top-level from **route guarantees the two shapes
        # can never diverge — the asymmetry that caused the original
        # stale-key bug (#85261 simplify-code review).
        route = {
            "provider": provider or None,
            "base_url": result.base_url or None,
            "api_mode": result.api_mode or None,
        }
        try:
            db.update_session_model(sid, result.new_model)
            db.patch_session_model_config(sid, {
                "gateway_runtime": route,
                **route,
            })
        except Exception:
            logger.debug(
                "Failed to persist model switch to session DB", exc_info=True
            )

    def _restore_session_model(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Restore model/provider from the session DB row on resume.

        Companion to ``_restore_session_cwd`` / ``_restore_session_yolo`` —
        called from every resume path (startup ``--resume``/``-c`` and
        mid-chat ``/resume``). The persisted model lives in the session row's
        ``model`` column (written at creation time and updated on ``/model``
        switches via ``update_session_model``); the provider/endpoint live in
        ``model_config.gateway_runtime`` (written by the gateway's
        ``_sync_session_model_from_agent`` and the CLI ``/model`` persist).
        Without this restore a resumed session silently falls back to the
        config default model, losing the user's last ``/model`` choice.

        When the stored provider differs from the ambient one, credentials
        are re-resolved for the stored provider (mirroring the gateway's
        ``_rehydrate_session_model_override``) — the ambient ``self.api_key``
        belongs to the config-default provider and must not be sent to the
        session's endpoint. On resolution failure the ambient credentials are
        kept so the session still opens (the first turn surfaces the auth
        error instead of the resume dying).

        Skips when the session has no model recorded or when the CLI was
        launched with an explicit ``-m`` override (user intent wins).
        """
        from cli import logger
        stored_model = (session_meta or {}).get("model")
        if not stored_model:
            return
        # An explicit -m / --model on the command line overrides resume.
        if getattr(self, "_explicit_model_override", False):
            return
        # Stored provider/endpoint via the canonical row-level reader
        # (prefers model_config.gateway_runtime, falls back to the TUI
        # gateway's top-level keys).
        from hermes_state import SessionDB as _SessionDB
        _stored_runtime = _SessionDB.session_gateway_runtime(session_meta)
        stored_provider = _stored_runtime.get("provider") or None
        stored_base_url = _stored_runtime.get("base_url") or None
        stored_api_mode = _stored_runtime.get("api_mode") or None
        # Heal bare "custom" persisted by older builds / gateway turns: it's
        # the resolved billing class, not a routable identity. Recover the
        # durable custom:<name> menu key from the endpoint, else drop the
        # provider so resume keeps the ambient default. (Stricter than the
        # TUI gateway's recovery, which keeps bare "custom" when a base_url
        # exists — the CLI's resolve path would hard-fail on it, #14676.)
        if str(stored_provider or "").strip().lower() == "custom":
            try:
                from hermes_cli.runtime_provider import canonical_custom_identity
                stored_provider = canonical_custom_identity(
                    base_url=stored_base_url or None,
                    model=stored_model or None,
                ) or None
            except Exception:
                stored_provider = None
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
            # Stale launch-time explicit overrides belong to the AMBIENT
            # provider; carrying them into the restored provider's
            # resolution poisons _ensure_runtime_credentials on startup
            # resume (same leak _apply_model_switch_result guards against
            # by overwriting _explicit_* on every switch).
            self._explicit_api_key = None
            self._explicit_base_url = stored_base_url
            # Re-resolve credentials for the restored provider. api_key is
            # never persisted to the session DB (by design) — the normal
            # runtime provider resolution owns credentials.
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
        # If the agent is already running (mid-chat /resume), swap it
        # in-place so the next turn uses the restored model. On startup
        # --resume the agent isn't built yet — _init_agent will pick up
        # self.model / self.provider when constructing AIAgent.
        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=self.model,
                    new_provider=self.provider,
                    api_key=self.api_key or "",
                    base_url=self.base_url or "",
                    api_mode=self.api_mode or "",
                )
            except Exception:
                logger.debug(
                    "In-place agent model swap on resume failed", exc_info=True
                )
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
        default_idx = next((i for i, p in enumerate(providers) if p.get("is_current")), 0)
        self._model_picker_state = {
            "stage": "provider",
            "providers": providers,
            "selected": default_idx,
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
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "",
                model_info=result.model_info,
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
            title=f"!!! {warning.title} !!!",
            detail=warning.message,
            choices=choices,
            timeout=120,
        )
        choice = self._normalize_slash_confirm_choice(raw, choices)
        return choice == "once"

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
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "agent_primary_runtime": copy.deepcopy(
                getattr(agent, "_primary_runtime", None)
            ) if agent is not None else None,
        }

    def _restore_model_runtime_snapshot(self, snapshot: dict | None) -> None:
        """Restore a model runtime captured before a one-turn override."""
        from cli import logger
        if not snapshot:
            return
        for key in (
            "model",
            "provider",
            "requested_provider",
            "_explicit_api_key",
            "_explicit_base_url",
            "api_key",
            "base_url",
            "api_mode",
        ):
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
                    new_model=snapshot.get("model", ""),
                    new_provider=snapshot.get("provider", ""),
                    api_key=snapshot.get("api_key", ""),
                    base_url=snapshot.get("base_url", ""),
                    api_mode=snapshot.get("api_mode", ""),
                    capabilities=snapshot.get("capabilities"),
                )
            except Exception as exc:
                logger.warning("CLI one-turn model restore failed: %s", exc)

    @staticmethod
    def _filter_model_picker_entries(entries: list, query: str) -> list:
        """Return (original_index, label) pairs for entries matching ``query``.

        Subsequence ("fuzzy") match, case-insensitive: the query characters
        must appear in order in the label. An empty query matches everything.
        Crucially the returned pairs carry the ORIGINAL index into ``entries``,
        so a selection in the filtered view still resolves to exactly one
        concrete model — filtering only narrows the list, it never introduces
        an ambiguous or fuzzy *resolution* (the anti-"claude→old-model" rule).
        """
        pairs = list(enumerate(entries))
        q = (query or "").strip().lower()
        if not q:
            return pairs

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        out = [(i, e) for (i, e) in pairs if _subseq(q, str(e).lower())]
        return out

    @staticmethod
    def _compute_model_picker_viewport(
        selected: int,
        scroll_offset: int,
        n: int,
        term_rows: int,
        reserved_below: int = 6,
        panel_chrome: int = 6,
        min_visible: int = 3,
    ) -> tuple[int, int]:
        """Resolve (scroll_offset, visible) for the /model picker viewport.

        ``reserved_below`` matches the approval / clarify panels — input area,
        status bar, and separators below the panel. ``panel_chrome`` covers
        this panel's own borders + blanks + hint row. The remaining rows hold
        the scrollable list, with the offset slid to keep ``selected`` on screen.
        """
        max_visible = max(min_visible, term_rows - reserved_below - panel_chrome)
        if n <= max_visible:
            return 0, n
        visible = max_visible
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + visible:
            scroll_offset = selected - visible + 1
        scroll_offset = max(0, min(scroll_offset, n - visible))
        return scroll_offset, visible

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
                model_cfg.get("default") or model_cfg.get("model"),
                result.new_model,
                model_cfg.get("base_url"),
                result.base_url,
                model_cfg.get("provider"),
                result.target_provider,
            ):
                save_config_value("model.context_length", None)
        except Exception:
            save_config_value("model.context_length", None)

    def _stage_and_swap_model(self, result, old_model) -> bool:
        """Stage ``result`` onto the CLI fields, then swap the live agent in place.

        CLI-level fields are snapshotted first (mirrors the gateway) so a failed
        in-place agent swap rolls the whole CLI back to the old working model —
        otherwise the broken credentials staged here leak into the next turn's
        resolution even though the agent itself rolled back (#50163). Returns
        False after printing the failure (caller aborts the rest of the commit:
        note + success print), so a failed switch is a no-op, not a dead session.
        """
        from cli import _cprint
        _cli_snapshot = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
        }
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the
        # previous provider (e.g. Ollama api_key/base_url) don't leak into
        # the new provider's credential resolution on the next turn.
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
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
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
        from cli import HermesCLI, _cprint, logger, save_config_value
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        if self.agent is not None:
            try:
                from hermes_cli.context_switch_guard import merge_preflight_compression_warning

                # Prefer the fresh inventory list (same source as switch_model /
                # TUI); fall back to the agent-init snapshot.
                _cp = (
                    custom_providers
                    if custom_providers is not None
                    else getattr(self.agent, "_custom_providers", None)
                )
                merge_preflight_compression_warning(
                    result,
                    agent=self.agent,
                    messages=list(self.conversation_history or []),
                    custom_providers=_cp,
                    config_context_length=getattr(self.agent, "_config_context_length", None),
                )
            except Exception as exc:
                logger.debug("preflight-compression switch warning failed: %s", exc)

        old_model = self.model
        if not self._stage_and_swap_model(result, old_model):
            return

        from hermes_cli.model_switch import format_model_for_display
        _display_old = format_model_for_display(old_model)
        _display_new = format_model_for_display(result.new_model)

        self._pending_model_switch_note = (
            f"[Note: model was just switched from {_display_old} to {_display_new} "
            f"via {result.provider_label or result.target_provider}. "
            f"Adjust your self-identification accordingly.]"
        )

        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {_display_new}")
        _cprint(f"    Provider: {provider_label}")

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry
        # (e.g. gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
        mi = result.model_info
        try:
            from hermes_cli.model_switch import resolve_display_context_length
            ctx = resolve_display_context_length(
                result.new_model,
                result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "",
                model_info=mi,
                config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None,
                custom_providers=getattr(self.agent, "_custom_providers", None) if self.agent else None,
            )
            if ctx:
                _cprint(f"    Context: {ctx:,} tokens")
        except Exception:
            pass
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
        if persist_global:
            HermesCLI._clear_persisted_context_for_model_switch(self, result)
            save_config_value("model.default", result.new_model)
            save_config_value("model.provider", result.target_provider)
            # base_url/api_mode were previously never persisted here, so a
            # global switch left the OLD provider's endpoint/wire-protocol in
            # config.yaml. result.base_url/api_mode are always freshly
            # resolved for the target provider (see model_switch.py), so sync
            # them every time; None clears a value the new provider doesn't
            # need (#25106).
            save_config_value("model.base_url", result.base_url or None)
            save_config_value("model.api_mode", result.api_mode or None)
            _cprint("    Saved to config.yaml (--global)")
        else:
            _cprint("    (session only — add --global to persist)")

        # Persist the switch to this session's row so --resume /
        # session.resume restore it. --global also updates config.yaml
        # (future sessions), but the row still records what THIS session
        # actually runs — otherwise a later resume would restore the stale
        # creation-time model over the user's new global choice.
        HermesCLI._persist_model_switch_to_session(self, result)

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
            # Use the curated model list from list_authenticated_providers()
            # (same lists as `hermes model` and gateway pickers).
            # Only fall back to the live provider catalog when the curated
            # list is empty (e.g. user-defined endpoints with no curated list).
            model_list = provider_data.get("models", [])
            if not model_list:
                try:
                    from hermes_cli.models import provider_model_ids
                    live = provider_model_ids(provider_data["slug"])
                    if live:
                        model_list = live
                except Exception:
                    pass
            state["stage"] = "model"
            state["provider_data"] = provider_data
            state["model_list"] = model_list
            state["selected"] = 0
            state["filter"] = ""
            state["_filtered_pairs"] = None
            self._invalidate(min_interval=0.0)
            return
        if stage == "model":
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            # Map the selected row through the active fuzzy filter so the
            # index lines up with what the picker is currently showing. The
            # filtered pair carries the ORIGINAL index into model_list, so the
            # resolved model is always one concrete, unambiguous entry.
            filtered_pairs = state.get("_filtered_pairs")
            if filtered_pairs is None:
                filtered_pairs = list(enumerate(model_list))
            visible_labels = [e for (_i, e) in filtered_pairs]
            back_idx = len(visible_labels)
            cancel_idx = len(visible_labels) + 1
            if selected == back_idx:
                state["stage"] = "provider"
                state["filter"] = ""
                state["_filtered_pairs"] = None
                state["selected"] = next((i for i, p in enumerate(state.get("providers") or []) if p.get("slug") == provider_data.get("slug")), 0)
                self._invalidate(min_interval=0.0)
                return
            if selected >= cancel_idx:
                self._close_model_picker()
                return
            if 0 <= selected < len(visible_labels):
                from hermes_cli.model_switch import switch_model
                chosen_model = visible_labels[selected]
                result = switch_model(
                    raw_input=chosen_model,
                    current_provider=self.provider or "",
                    current_model=self.model or "",
                    current_base_url=self.base_url or "",
                    current_api_key=self.api_key or "",
                    is_global=persist_global,
                    explicit_provider=provider_data.get("slug"),
                    user_providers=state.get("user_provs"),
                    custom_providers=state.get("custom_provs"),
                )
                # Capture before close — picker state is cleared on close.
                _picker_custom_provs = state.get("custom_provs")
                self._close_model_picker()
                if getattr(self, "_app", None):
                    threading.Thread(
                        target=self._confirm_and_apply_model_switch_result,
                        args=(result, persist_global, _picker_custom_provs),
                        daemon=True,
                    ).start()
                else:
                    self._confirm_and_apply_model_switch_result(
                        result, persist_global, custom_providers=_picker_custom_provs
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

        Persistence defaults to off (``model.persist_switch_by_default`` in
        config.yaml, default False — switches are session-scoped). Use
        ``--global`` to persist, or ``--once`` for the next turn only.
        """
        from cli import _cprint, logger
        from hermes_cli.model_switch import (
            switch_model,
            parse_model_switch_args,
            resolve_persist_behavior,
        )
        from hermes_cli.providers import get_label

        # Parse args from the original command
        parts = cmd_original.split(None, 1)  # split off '/model'
        raw_args = parts[1].strip() if len(parts) > 1 else ""

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
            # CLI decoration: "  ✗ " prefix over the canonical error copy.
            _cprint(f"  ✗ {request.error_messages()[0]}")
            return
        # Resolve the effective persistence once: --global forces persist,
        # --session/--once force session-scope, otherwise defer to
        # model.persist_switch_by_default (defaults to False so /model is
        # session-scoped unless the user opts in).
        persist_global = resolve_persist_behavior(
            is_global_flag, is_session, is_once=one_turn,
            explicit_provider=explicit_provider,
        )

        # --refresh: wipe the on-disk picker cache before building the
        # provider list. Forces a live re-fetch of every authed provider's
        # /v1/models endpoint on this open.
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
                _cprint("  Cleared model picker cache. Refreshing...")
            except Exception:
                pass

        # Single inventory context — replaces the inline config-slice the
        # dashboard / TUI used to duplicate. Overlay live session state
        # via with_overrides (truthy-only) so empty self.* attrs don't
        # clobber disk config.
        from hermes_cli.inventory import build_models_payload, load_picker_context

        try:
            ctx = load_picker_context().with_overrides(
                current_provider=self.provider or "",
                current_model=self.model or "",
                current_base_url=self.base_url or "",
            )
        except Exception:
            ctx = None

        # switch_model() + _open_model_picker still need the raw provider
        # dicts; ConfigContext is the canonical source for both.
        user_provs = ctx.user_providers if ctx is not None else None
        custom_provs = ctx.custom_providers if ctx is not None else None

        # No args at all: open prompt_toolkit-native picker modal
        if not model_input and not explicit_provider:
            model_display = self.model or "unknown"
            provider_display = get_label(self.provider) if self.provider else "unknown"

            try:
                if ctx is None:
                    raise RuntimeError("inventory context unavailable")
                providers = build_models_payload(
                    ctx,
                    probe_custom_providers=force_refresh,
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

            self._open_model_picker(
                providers,
                model_display,
                provider_display,
                user_provs=user_provs,
                custom_provs=custom_provs,
            )
            return

        # Perform the switch
        result = switch_model(
            raw_input=model_input,
            current_provider=self.provider or "",
            current_model=self.model or "",
            current_base_url=self.base_url or "",
            current_api_key=self.api_key or "",
            is_global=persist_global,
            explicit_provider=explicit_provider,
            user_providers=user_provs,
            custom_providers=custom_provs,
        )

        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        if self.agent is not None:
            try:
                from hermes_cli.context_switch_guard import merge_preflight_compression_warning

                merge_preflight_compression_warning(
                    result,
                    agent=self.agent,
                    messages=list(self.conversation_history or []),
                    # Same fresh inventory list passed to switch_model above.
                    custom_providers=custom_provs
                    if custom_provs is not None
                    else getattr(self.agent, "_custom_providers", None),
                    config_context_length=getattr(self.agent, "_config_context_length", None),
                )
            except Exception as exc:
                logger.debug("preflight-compression switch warning failed: %s", exc)

        # Run the confirm + apply sequence off the main thread. The
        # expensive-model confirmation modal blocks the calling thread on a
        # response queue (see _prompt_text_input_modal); running it on the
        # prompt_toolkit main thread freezes TUI rendering, so the modal never
        # appears and the switch silently cancels after the 120s timeout.
        # Mirror the picker path (_handle_model_picker_selection), which
        # already dispatches confirm+apply on a worker thread.
        if getattr(self, "_app", None):
            threading.Thread(
                target=self._confirm_and_apply_cli_model_switch,
                args=(result, persist_global, one_turn, custom_provs),
                daemon=True,
            ).start()
            return
        self._confirm_and_apply_cli_model_switch(
            result, persist_global, one_turn, custom_provs
        )
        return

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global: bool, one_turn: bool, custom_provs=None
    ) -> None:
        """Confirm an expensive model switch and apply it to CLI state.

        Runs on a worker thread when the TUI is active (see
        _handle_model_switch) so the confirmation modal can render.
        """
        from cli import HermesCLI, _cprint, save_config_value
        if not self._confirm_expensive_model_switch(result):
            _cprint("  Model switch cancelled.")
            return

        # Apply to CLI state.
        # Update requested_provider so _ensure_runtime_credentials() doesn't
        # overwrite the switch on the next turn (it re-resolves from this).
        old_model = self.model
        _one_turn_restore_snapshot = self._snapshot_model_runtime() if one_turn else None
        if not self._stage_and_swap_model(result, old_model):
            return

        # Store a note to prepend to the next user message so the model
        # knows a switch occurred (avoids injecting system messages mid-history
        # which breaks providers and prompt caching).
        from hermes_cli.model_switch import format_model_for_display
        _display_old = format_model_for_display(old_model)
        _display_new = format_model_for_display(result.new_model)

        self._pending_model_switch_note = (
            f"[Note: model was just switched from {_display_old} to {_display_new} "
            f"via {result.provider_label or result.target_provider}. "
            f"{'This override applies to the next turn only. ' if one_turn else ''}"
            f"Adjust your self-identification accordingly.]"
        )
        if one_turn:
            self._pending_one_turn_model_restore = _one_turn_restore_snapshot
        else:
            self._pending_one_turn_model_restore = None

        # Display confirmation with full metadata
        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {_display_new}")
        _cprint(f"    Provider: {provider_label}")

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry
        # (e.g. gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
        mi = result.model_info
        from hermes_cli.model_switch import resolve_display_context_length
        ctx = resolve_display_context_length(
            result.new_model,
            result.target_provider,
            base_url=result.base_url or self.base_url or "",
            api_key=result.api_key or self.api_key or "",
            model_info=mi,
            config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None,
            custom_providers=getattr(self.agent, "_custom_providers", None) if self.agent else None,
        )
        if ctx:
            _cprint(f"    Context: {ctx:,} tokens")
        if mi:
            if mi.max_output:
                _cprint(f"    Max output: {mi.max_output:,} tokens")
            _cprint(f"    Capabilities: {mi.format_capabilities()}")

        # Cache notice
        cache_enabled = (
            (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            _cprint("    Prompt caching: enabled")

        # Warning from validation
        if result.warning_message:
            _cprint(f"    ⚠ {result.warning_message}")

        # Persistence
        if persist_global:
            HermesCLI._clear_persisted_context_for_model_switch(self, result)
            save_config_value("model.default", result.new_model)
            save_config_value("model.provider", result.target_provider)
            # See _apply_model_switch_result above for why base_url/api_mode
            # must be synced on every global switch (#25106).
            save_config_value("model.base_url", result.base_url or None)
            save_config_value("model.api_mode", result.api_mode or None)
            _cprint("    Saved to config.yaml")
        elif one_turn:
            _cprint("    (next turn only — restores after one response)")
        else:
            _cprint("    (session only — add --global to persist)")

        # Persist the switch to this session's row so --resume /
        # session.resume restore it (--global also updates config.yaml but
        # the row still records what THIS session runs; --once is ephemeral
        # and restored after one turn, so it must not touch the row).
        if not one_turn:
            HermesCLI._persist_model_switch_to_session(self, result)

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
        raw_args = parts[1].strip() if len(parts) > 1 else ""
        new_value, errors = crs.parse_args(raw_args)
        if errors:
            for err in errors:
                _cprint(f"❌ {err}")
            return

        # Load + persist via the existing config helpers
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            _cprint(f"❌ could not load config: {exc}")
            return
        cfg = load_config()

        result = crs.apply(
            cfg,
            new_value,
            persist_callback=(save_config if new_value is not None else None),
        )

        prefix = "✓" if result.success else "✗"
        for line in result.message.splitlines():
            _cprint(f"  {prefix} {line}" if line.startswith("openai_runtime")
                    else f"    {line}")
        if result.success and result.requires_new_session:
            _cprint("    Tip: `/reset` starts a new session immediately.")

    def _should_handle_model_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /model should be handled immediately on the UI thread."""
        from cli import _looks_like_slash_command
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        try:
            from hermes_cli.commands import resolve_command
            base = text.split(None, 1)[0].lower().lstrip('/')
            cmd = resolve_command(base)
            return bool(cmd and cmd.name == "model")
        except Exception:
            return False

    def _cmd_moa(self, cmd_original: str):
        # /moa is one-shot sugar only: run a single prompt through the
        # default MoA preset, then restore the prior model. To *switch* to a
        # MoA preset for the session, pick it from the model picker (MoA
        # presets surface as a virtual "Mixture of Agents" provider).
        from cli import _cprint, _slash_args
        from hermes_cli.moa_config import (
            moa_usage,
            normalize_moa_config,
        )

        payload = _slash_args(cmd_original)
        if not payload:
            _cprint(f"  {moa_usage()}")
            return True
        moa_cfg = self.config.get("moa") if isinstance(self.config, dict) else {}
        normalized = normalize_moa_config(moa_cfg)
        preset = normalized["default_preset"]
        self._pending_moa_restore_model = {
            "requested_provider": getattr(self, "requested_provider", None),
            "provider": getattr(self, "provider", None),
            "model": getattr(self, "model", None),
            "api_key": getattr(self, "api_key", None),
            "base_url": getattr(self, "base_url", None),
            "api_mode": getattr(self, "api_mode", None),
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
